#include <dmk/logger.h>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/tree.hpp>
#include <sctl/tree.hpp>

#include <mpi.h>

namespace dmk {

template <typename T, int DIM>
void DMKPtTree<T, DIM>::generate_metadata(int ndiv, int nd) {
    const int n_nodes = n_boxes();
    this->GetData(r_src_sorted, r_src_cnt, "pdmk_src");
    this->GetData(charge_sorted, charge_cnt, "pdmk_charge");
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    const auto &node_lists = this->GetNodeLists();

    leaf_flag.ReInit(n_nodes);
    in_flag.ReInit(n_nodes);
    out_flag.ReInit(n_nodes);
    src_counts_local.ReInit(n_nodes);
    r_src_offsets.resize(n_nodes);
    charge_offsets.resize(n_nodes);
    centers.resize(n_nodes * DIM);
    scale_factors.resize(n_nodes);

    level_indices.resize(SCTL_MAX_DEPTH);

    for (int i_node = 1; i_node < n_nodes; ++i_node) {
        r_src_offsets[i_node] = r_src_offsets[i_node - 1] + DIM * r_src_cnt[i_node - 1];
        charge_offsets[i_node] = charge_offsets[i_node - 1] + nd * charge_cnt[i_node - 1];
    }

    int8_t max_depth = 0;
    for (int i_node = 0; i_node < n_nodes; ++i_node) {
        auto &node = node_mid[i_node];
        level_indices[node.Depth()].push_back(i_node);
        max_depth = std::max(node.Depth(), max_depth);
    }
    max_depth++;
    level_indices.resize(max_depth);
    boxsize.resize(max_depth);
    boxsize[0] = 1.0;
    for (int i = 1; i < max_depth; ++i)
        boxsize[i] = 0.5 * boxsize[i - 1];

    T scale = 1.0;
    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        for (auto i_node : level_indices[i_level]) {
            auto &node = node_mid[i_node];
            auto node_origin = node.template Coord<T>();
            for (int i = 0; i < DIM; ++i)
                centers[i_node * DIM + i] = node_origin[i] + 0.5 * scale;
            scale_factors[i_node] = scale;
        }
        scale *= 0.5;
    }

    src_counts_local.SetZero();
    for (int i_level = max_depth - 1; i_level >= 0; i_level--) {
        for (auto i_node : level_indices[i_level]) {
            auto &node = node_mid[i_node];
            assert(i_level == node.Depth());

            src_counts_local[i_node] += r_src_cnt[i_node];
            if (node_lists[i_node].parent != -1)
                src_counts_local[node_lists[i_node].parent] += src_counts_local[i_node];
        }
    }

    sctl::Vector<long> counts(src_counts_local.Dim());
    for (auto &el : counts)
        el = 1;

    this->template AddData("src_counts", src_counts_local, counts);
    this->template ReduceBroadcast<int>("src_counts");
    this->template GetData<int>(src_counts_global, counts, "src_counts");

    for (int i_node = 0; i_node < n_nodes; ++i_node) {
        leaf_flag[i_node] = 0;
        out_flag[i_node] = 0;
        if (src_counts_global[i_node] > ndiv)
            out_flag[i_node] = true;
        if (src_counts_global[i_node] > 0 && src_counts_global[i_node] <= ndiv && node_lists[i_node].parent >= 0 &&
            src_counts_global[node_lists[i_node].parent] > ndiv)
            leaf_flag[i_node] = true;
    }
    for (int i_node = 0; i_node < n_nodes; ++i_node) {
        in_flag[i_node] = 0;
        for (auto &neighb : node_lists[i_node].nbr) {
            // neighb = -1 -> no neighb at current level in that direction
            if (neighb != -1 && out_flag[neighb] && src_counts_global[neighb] > 0) {
                in_flag[i_node] = true;
                break;
            }
        }
    }
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::build_proxy_charges(int n_mfm, int n_order, const std::vector<T> &c2p) {
    auto &logger = dmk::get_logger();

    const int n_coeffs = n_mfm * sctl::pow<DIM>(n_order);
    proxy_coeffs.ReInit(n_boxes() * n_coeffs);
    proxy_coeffs.SetZero();
    sctl::Vector<sctl::Long> counts(n_boxes());
    counts.SetZero();

    const auto &attrs = this->GetNodeAttr();
    for (int i_box = 0; i_box < n_boxes(); ++i_box) {
        if (leaf_flag[i_box] && !attrs[i_box].Ghost) {
            proxy::charge2proxycharge(DIM, n_mfm, n_order, src_counts_local[i_box], r_src_ptr(i_box), charge_ptr(i_box),
                                      center_ptr(i_box), scale_factors[i_box], &proxy_coeffs[i_box * n_coeffs]);
            counts[i_box] += n_coeffs;
        }
    }
    logger->debug("Finished building leaf proxy charges");

    constexpr int n_children = 1u << DIM;
    const auto &node_lists = this->GetNodeLists();
    for (int i_level = n_levels() - 1; i_level >= 0; --i_level) {
        for (auto parent_box : this->level_indices[i_level]) {
            if (attrs[parent_box].Ghost || this->leaf_flag[parent_box] || !this->out_flag[parent_box])
                continue;

            auto &children = node_lists[parent_box].child;
            for (int i_child = 0; i_child < n_children; ++i_child) {
                const int child_box = children[i_child];

                constexpr bool add_flag = true;
                if (counts[child_box]) {
                    tensorprod::transform(DIM, n_mfm, n_order, n_order, add_flag, &proxy_coeffs[child_box * n_coeffs],
                                          &c2p[i_child * DIM * n_order * n_order], &proxy_coeffs[parent_box * n_coeffs]);
                }
                counts[parent_box] = n_coeffs;
            }
        }
    }
    for (auto &count : counts)
        count = n_coeffs;

    this->AddData("proxy_coeffs", proxy_coeffs, counts);
    this->template ReduceBroadcast<T>("proxy_coeffs");
    this->template GetData<T>(proxy_coeffs, counts, "proxy_coeffs");

    logger->debug("Finished building proxy charges for non-leaf boxes");
}

template struct DMKPtTree<float, 2>;
template struct DMKPtTree<float, 3>;
template struct DMKPtTree<double, 2>;
template struct DMKPtTree<double, 3>;

} // namespace dmk
