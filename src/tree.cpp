#include <dmk/tree.hpp>
#include <sctl/tree.hpp>

#include <mpi.h>

namespace dmk {

template <typename TREE>
TreeData::TreeData(const TREE &tree, int ns) {
    // This probably isn't necessary, but currently needed to compare colleagues
    const int ns_global = 0;
    MPI_Allreduce(&ns, const_cast<int *>(&ns_global), 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const auto &attrs = tree.GetNodeAttr();
    const auto &nodes = tree.GetNodeMID();
    const auto &node_lists = tree.GetNodeLists();
    const int n_nodes = nodes.Dim();
    sctl::Vector<double> data_part;
    sctl::Vector<sctl::Long> cnt_part;
    tree.GetData(data_part, cnt_part, "pdmk_src");

    leaf_flag.resize(n_nodes);
    in_flag.resize(n_nodes);
    out_flag.resize(n_nodes);
    src_counts_local.resize(n_nodes);
    src_counts_global.resize(n_nodes);

    level_indices.resize(SCTL_MAX_DEPTH);

    int8_t max_depth = 0;
    for (int i_node = 0; i_node < n_nodes; ++i_node) {
        auto &node = nodes[i_node];
        level_indices[node.Depth()].push_back(i_node);
        max_depth = std::max(node.Depth(), max_depth);
    }
    max_depth++;
    level_indices.resize(max_depth);
    boxsize.resize(max_depth);
    boxsize[0] = 1.0;
    for (int i = 1; i < max_depth; ++i)
        boxsize[i] = 0.5 * boxsize[i - 1];

    for (int i_level = max_depth - 1; i_level >= 0; i_level--) {
        for (auto i_node : level_indices[i_level]) {
            auto &node = nodes[i_node];
            assert(i_level == node.Depth());

            leaf_flag[i_node] = attrs[i_node].Leaf;
            src_counts_local[i_node] += cnt_part[i_node];
            if (node_lists[i_node].parent != -1)
                src_counts_local[node_lists[i_node].parent] += src_counts_local[i_node];
        }
    }

    // FIXME: this doesn't work, trees aren't identical always...
    // MPI_Allreduce(src_counts_local.data(), src_counts_global.data(), n_nodes, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    src_counts_global = src_counts_local;
    for (int i_node = 0; i_node < n_nodes; ++i_node) {
        if (src_counts_global[i_node] > ns_global)
            out_flag[i_node] = true;
        for (auto &neighb : node_lists[i_node].nbr) {
            // neighb = -1 -> no neighb at current level in that direction
            if (neighb != -1 && neighb != i_node && src_counts_global[neighb] > ns) {
                in_flag[i_node] = true;
                break;
            }
        }
    }
}

template TreeData::TreeData(const sctl::PtTree<float, 2> &, int);
template TreeData::TreeData(const sctl::PtTree<float, 3> &, int);
template TreeData::TreeData(const sctl::PtTree<double, 2> &, int);
template TreeData::TreeData(const sctl::PtTree<double, 3> &, int);

} // namespace dmk
