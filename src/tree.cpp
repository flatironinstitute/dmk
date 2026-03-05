#include <cmath>
#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/direct.hpp>
#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/legeexps.hpp>
#include <dmk/logger.h>
#include <dmk/planewave.hpp>
#include <dmk/prolate0_fun.hpp>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/testing.hpp>
#include <dmk/tree.hpp>
#include <dmk/types.hpp>
#include <dmk/util.hpp>
#include <fstream>
#include <sctl/profile.hpp>
#include <stdexcept>
#include <unistd.h>

#include <dmk/omp_wrapper.hpp>

namespace dmk {

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::dump() const {
    auto logger = dmk::get_rank_logger(comm_);
    logger->info("Dumping DMKPtTree data on rank {} of comm size {}", comm_.Rank(), comm_.Size());

    auto dumper = [&logger, this](const std::string &name, const auto &data) {
        std::string filename = name + "." + std::to_string(comm_.Size()) + "." + std::to_string(comm_.Rank()) + ".dat";
        if constexpr (requires { data.Write(filename.c_str()); })
            data.Write(filename.c_str());
        else {
            auto fout = std::fstream(std::string(filename).c_str(), std::ios::out | std::ios::binary);
            const int64_t dimensions = 1;
            const uint64_t bufsize = data.size();
            fout.write(reinterpret_cast<const char *>(&dimensions), sizeof(uint64_t));
            fout.write(reinterpret_cast<const char *>(&bufsize), sizeof(uint64_t));
            fout.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(decltype(data[0])));
        }

        logger->info("Dumped {}", filename);
    };

    sctl::Vector<bool> is_ghost(n_boxes());
    const auto &node_attr = this->GetNodeAttr();
    for (int i = 0; i < n_boxes(); ++i)
        is_ghost[i] = node_attr[i].Ghost;

    sctl::Vector<bool> is_leaf(n_boxes());
    for (int i = 0; i < n_boxes(); ++i)
        is_leaf[i] = node_attr[i].Leaf;

    sctl::Vector<sctl::Morton<DIM>> morton_ids(n_boxes());
    const auto node_mid = this->GetNodeMID();
    for (int i = 0; i < n_boxes(); ++i)
        morton_ids[i] = node_mid[i];

    if (comm_.Rank() == 0) {
        std::string params_filename = "dmk_params." + std::to_string(comm_.Size()) + ".dat";
        struct ParamsData {
            int n_dim;
            int n_order;
            int n_pw;
            int floatsize;
        } params_data{DIM, n_order, n_pw, sizeof(Real)};
        auto fout = std::fstream(params_filename.c_str(), std::ios::out | std::ios::binary);
        fout.write(reinterpret_cast<const char *>(&params_data), sizeof(ParamsData));
    }

    dumper("dmk_centers", centers);
    dumper("dmk_is_ghost", is_ghost);
    dumper("dmk_is_leaf", is_leaf);
    dumper("dmk_ifpwexp", ifpwexp);
    dumper("dmk_iftensprodeval", iftensprodeval);
    dumper("dmk_pw_out", pw_out);
    dumper("dmk_pw_out_offsets", pw_out_offsets);
    dumper("dmk_proxy_coeffs", proxy_coeffs_upward);
    dumper("dmk_proxy_coeffs_offsets", proxy_coeffs_offsets);
    dumper("dmk_proxy_coeffs_downward", proxy_coeffs_downward);
    dumper("dmk_proxy_coeffs_offsets_downward", proxy_coeffs_offsets_downward);
    dumper("dmk_pw_expansion", ifpwexp);
    dumper("dmk_src_counts_with_halo", src_counts_with_halo);
    dumper("dmk_src_counts_owned", src_counts_owned);
    dumper("dmk_morton_ids", morton_ids);
}

std::pair<int, int> get_pwmax_and_poly_order(int dim, int ndigits, dmk_ikernel kernel) {
    // clang-format off
    if (kernel == DMK_SQRT_LAPLACE && dim == 3) {
        if (ndigits <= 3) return {13, 9};
        if (ndigits <= 6) return {27, 18};
        if (ndigits <= 9) return {39, 28};
        if (ndigits <= 12) return {55, 38};
    }
    if (ndigits <= 3) return {13, 9};
    if (ndigits <= 6) return {25, 18};
    if (ndigits <= 9) return {39, 28};
    if (ndigits <= 12) return {53, 38};
    // clang-format on
    throw std::runtime_error("Requested precision too high");
}

void update_offsets_from_counts(const sctl::Vector<sctl::Long> &counts, sctl::Long N,
                                sctl::Vector<sctl::Long> &offsets) {
    offsets.ReInit(counts.Dim());
    sctl::Long last_offset;
    for (int i = 0; i < counts.Dim(); ++i) {
        if (counts[i]) {
            offsets[i] = last_offset;
            last_offset += N;
        } else
            offsets[i] = -1;
    }
}

template <typename Real, int DIM>
DMKPtTree<Real, DIM>::DMKPtTree(const sctl::Comm &comm, const pdmk_params &params_, const sctl::Vector<Real> &r_src,
                                const sctl::Vector<Real> &r_trg, const sctl::Vector<Real> &charge)
    : sctl::PtTree<Real, DIM>(comm), comm_(comm), params(params_), n_digits(std::round(log10(1.0 / params_.eps) - 0.1)),
      n_pw_max(get_pwmax_and_poly_order(DIM, n_digits, params_.kernel).first),
      n_order(get_pwmax_and_poly_order(DIM, n_digits, params_.kernel).second) {
    auto &logger = dmk::get_logger(comm, params.log_level);
    auto &rank_logger = dmk::get_rank_logger(comm, params.log_level);
    sctl::Profile::Scoped profile("DMKPtTree::DMKPtTree", &comm_);
    debug_omit_pw = getenv("DMK_DEBUG_OMIT_PW") != nullptr;
    debug_omit_direct = getenv("DMK_DEBUG_OMIT_DIRECT") != nullptr;
    debug_dump_tree = getenv("DMK_DEBUG_DUMP_TREE") != nullptr;

    logger->info("tree build started");

    const int n_src = r_src.Dim() / DIM;
    const int n_trg = r_trg.Dim() / DIM;

    // 0: Initialization
    sctl::Vector<Real> pot_vec_src(n_src * params.n_mfm);
    sctl::Vector<Real> pot_vec_trg(n_trg * params.n_mfm);

    logger->debug("Building tree and sorting points");
    // Use "2-1" balancing for the tree, i.e. touching boxes never more than one level away in depth
    constexpr bool balance21 = true;
    // Only grab nearest neighbors as 'ghosts' <-> halo = 0
    constexpr int halo = 0;

    // All data that needs to be tree sorted
    this->AddParticles("pdmk_src", r_src);
    this->AddParticles("pdmk_trg", r_trg);
    this->AddParticleData("pdmk_charge", "pdmk_src", charge);
    this->AddParticleData("pdmk_pot_src", "pdmk_src", pot_vec_src);
    this->AddParticleData("pdmk_pot_trg", "pdmk_trg", pot_vec_trg);
    this->UpdateRefinement(r_src, params.n_per_leaf, balance21, params.use_periodic, halo);

    // Grab sorted particle data without the halo, so it's easier to get anything local to this rank.
    // Direct evaluations need halo data (for source particles), but targets points/particles should be owned by the
    // rank.
    // Planewaves only need owned particles and their charges for calculation
    this->GetData(r_trg_sorted_owned, r_trg_cnt_owned, "pdmk_trg");
    this->GetData(pot_src_sorted, pot_src_cnt, "pdmk_pot_src");
    this->GetData(pot_trg_sorted, pot_trg_cnt, "pdmk_pot_trg");
    // We need temporaries to copy from for things that we broadcast to the halo, since GetData
    // only gets a pointer which gets re-used on Broadcast
    {
        sctl::Vector<Real> data;
        sctl::Vector<long> count;
        this->GetData(data, count, "pdmk_src");
        r_src_sorted_owned = data;
        r_src_cnt_owned = count;

        this->GetData(data, count, "pdmk_charge");
        charge_sorted_owned = data;
        charge_cnt_owned = count;
    }

    // Now grab sorted particle data with the halo, so we have it for direct evaluations
    this->template Broadcast<Real>("pdmk_src");
    this->template Broadcast<Real>("pdmk_charge");
    this->GetData(charge_sorted_with_halo, charge_cnt_with_halo, "pdmk_charge");
    this->GetData(r_src_sorted_with_halo, r_src_cnt_with_halo, "pdmk_src");

    logger->debug("base tree build completed");
    logger->debug("generating tree traversal metadata");
    generate_metadata();
    logger->debug("done generating tree traversal metadata");

    rank_logger->trace("local tree has {} levels and {} boxes", n_levels(), n_boxes());

    // 1: Precomputation
    logger->debug("generating p2c and c2p matrices of order {}", n_order);
    std::tie(c2p, p2c) = dmk::chebyshev::get_c2p_p2c_matrices<Real>(DIM, n_order);
    logger->debug("finished generating matrices");

    fourier_data = FourierData<Real>(params.kernel, DIM, params.eps, n_digits, n_pw_max, params.fparam, boxsize);
    const auto &wk = fourier_data.windowed_kernel();
    logger->debug("planewave params at root box: n: {}, stepsize: {}, weight: {}, radius: {}", fourier_data.n_pw(),
                  wk.hpw, wk.ws, wk.rl);
    fourier_data.update_local_coeffs(params.eps);
    logger->debug("finished updating local potential expansion coefficients");
    logger->info("tree build completed");
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::compute_data_offsets() {
    const auto &node_mid = this->GetNodeMID();
    r_src_offsets_with_halo.ReInit(n_boxes());
    r_src_offsets_owned.ReInit(n_boxes());
    r_trg_offsets_owned.ReInit(n_boxes());
    pot_src_offsets.ReInit(n_boxes());
    pot_trg_offsets.ReInit(n_boxes());
    charge_offsets_owned.ReInit(n_boxes());
    charge_offsets_with_halo.ReInit(n_boxes());

    r_src_offsets_with_halo[0] = r_src_offsets_owned[0] = r_trg_offsets_owned[0] = pot_src_offsets[0] =
        pot_trg_offsets[0] = charge_offsets_owned[0] = charge_offsets_with_halo[0] = 0;

    for (int i = 1; i < n_boxes(); ++i) {
        r_src_offsets_with_halo[i] = r_src_offsets_with_halo[i - 1] + DIM * r_src_cnt_with_halo[i - 1];
        r_trg_offsets_owned[i] = r_trg_offsets_owned[i - 1] + DIM * r_trg_cnt_owned[i - 1];
        pot_src_offsets[i] = pot_src_offsets[i - 1] + params.n_mfm * pot_src_cnt[i - 1];
        pot_trg_offsets[i] = pot_trg_offsets[i - 1] + params.n_mfm * pot_trg_cnt[i - 1];
        charge_offsets_owned[i] = charge_offsets_owned[i - 1] + params.n_mfm * charge_cnt_owned[i - 1];
        charge_offsets_with_halo[i] = charge_offsets_with_halo[i - 1] + params.n_mfm * charge_cnt_with_halo[i - 1];
    }
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::compute_level_indices_and_boxsizes() {
    const auto &node_mid = this->GetNodeMID();
    level_indices.ReInit(SCTL_MAX_DEPTH);
    int8_t max_depth = 0;
    for (int i = 0; i < n_boxes(); ++i) {
        level_indices[node_mid[i].Depth()].PushBack(i);
        max_depth = std::max(node_mid[i].Depth(), max_depth);
    }
    max_depth++;

    level_indices.ReInit(max_depth);
    boxsize.ReInit(max_depth + 1);
    boxsize[0] = 1.0;
    for (int i = 1; i < max_depth + 1; ++i)
        boxsize[i] = 0.5 * boxsize[i - 1];
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::compute_box_centers() {
    const auto &node_mid = this->GetNodeMID();
    centers.ReInit(n_boxes() * DIM);
    T scale = 1.0;
    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        for (auto i_node : level_indices[i_level]) {
            auto node_origin = node_mid[i_node].template Coord<T>();
            for (int i = 0; i < DIM; ++i)
                centers[i_node * DIM + i] = node_origin[i] + 0.5 * scale;
        }
        scale *= 0.5;
    }
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::accumulate_subtree_counts() {
    const auto &node_mid = this->GetNodeMID();
    const auto &node_lists = this->GetNodeLists();
    src_counts_with_halo.ReInit(n_boxes());
    src_counts_with_halo.SetZero();
    src_counts_owned.ReInit(n_boxes());
    src_counts_owned.SetZero();
    trg_counts_owned.ReInit(n_boxes());
    trg_counts_owned.SetZero();

    for (int i_level = n_levels() - 1; i_level >= 0; --i_level) {
        for (auto i_node : level_indices[i_level]) {
            src_counts_with_halo[i_node] += r_src_cnt_with_halo[i_node];
            src_counts_owned[i_node] += r_src_cnt_owned[i_node];
            trg_counts_owned[i_node] += r_trg_cnt_owned[i_node];

            const int parent = node_lists[i_node].parent;
            if (parent != -1) {
                src_counts_with_halo[parent] += src_counts_with_halo[i_node];
                src_counts_owned[parent] += src_counts_owned[i_node];
                trg_counts_owned[parent] += trg_counts_owned[i_node];
            }
        }
    }
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::gather_owned_source_positions() {
    const auto &node_attr = this->GetNodeAttr();
    r_src_sorted_owned.ReInit(DIM * src_counts_owned[0]);
    r_src_offsets_owned.ReInit(n_boxes());
    r_src_offsets_owned[0] = 0;

    for (int i = 1; i < n_boxes(); ++i) {
        r_src_offsets_owned[i] = r_src_offsets_owned[i - 1] + DIM * r_src_cnt_owned[i - 1];

        if (src_counts_owned[i] && node_attr[i].Leaf && !node_attr[i].Ghost) {
            std::copy(r_src_with_halo_ptr(i), r_src_with_halo_ptr(i) + DIM * r_src_cnt_owned[i], r_src_owned_ptr(i));
        }
    }
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::broadcast_global_leaf_status() {
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_lists = this->GetNodeLists();
    is_global_leaf.ReInit(n_boxes());
    is_global_leaf.SetZero();
    for (int box = 0; box < n_boxes(); ++box)
        is_global_leaf[box] = node_attr[box].Leaf;

    sctl::Vector<sctl::Long> counts(n_boxes());
    sctl::Vector<sctl::Long> counts_dum;
    for (int i = 0; i < n_boxes(); ++i)
        counts[i] = 1;

    sctl::Vector<bool> is_global_leaf_halo;
    this->AddData("is_global_leaf", is_global_leaf, counts);
    this->template Broadcast<bool>("is_global_leaf");
    this->GetData(is_global_leaf_halo, counts_dum, "is_global_leaf");

    long offset = 0;
    for (int i = 0; i < n_boxes(); ++i) {
        if (counts_dum[i]) {
            is_global_leaf[i] = is_global_leaf_halo[offset];
            offset++;
        } else {
            is_global_leaf[i] = false;
        }
    }
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::compute_proxy_expansion_flags() {
    const auto &node_lists = this->GetNodeLists();
    ifpwexp.ReInit(n_boxes());
    ifpwexp.SetZero();
    ifpwexp[0] = true;

    for (int box = 0; box < n_boxes(); ++box) {
        if (!is_global_leaf[box]) {
            ifpwexp[box] = true;
            continue;
        }

        for (auto neighbor : node_lists[box].nbr) {
            if (neighbor < 0)
                continue;
            if (!is_global_leaf[neighbor]) {
                ifpwexp[box] = true;
                break;
            }
        }
    }
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::compute_proxy_evaluation_flags() {
    const auto &node_lists = this->GetNodeLists();
    iftensprodeval.ReInit(n_boxes());
    iftensprodeval.SetZero();

    for (const auto &level_boxes : level_indices) {
        for (auto box : level_boxes) {
            if (!(ifpwexp[box] && (src_counts_owned[box] + trg_counts_owned[box])))
                continue;

            const bool iftpeval = [&]() {
                for (auto child : node_lists[box].child) {
                    if (child >= 0 && ifpwexp[child])
                        return false;
                }
                return true;
            }();

            iftensprodeval[box] = iftpeval;

            if (iftpeval)
                continue;

            for (auto child : node_lists[box].child) {
                if (child >= 0 && !ifpwexp[child] && (src_counts_owned[child] + trg_counts_owned[child]))
                    iftensprodeval[child] = true;
            }
        }
    }
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::build_plane_wave_interaction_lists() {
    const auto &node_lists = this->GetNodeLists();
    nlistpw_.resize(n_boxes());
    listpw_.resize(n_boxes());

    for (const auto &level_boxes : level_indices) {
        for (const auto &box : level_boxes) {
            for (const auto &neighb : node_lists[box].nbr) {
                if (neighb < 0 || neighb == box)
                    continue;

                if (is_global_leaf[box] || is_global_leaf[neighb]) {
                    listpw_[box][nlistpw_[box]] = neighb;
                    nlistpw_[box]++;
                }
            }
        }
    }
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::build_direct_interaction_lists() {
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_lists = this->GetNodeLists();
    list1_.resize(n_boxes());
    nlist1_.resize(n_boxes());

    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        // Loop through target boxes at this level (boxes where we loop through neighbors for direct eval)
        for (int box : level_indices[i_level]) {
            if (!is_global_leaf[box] || node_attr[box].Ghost)
                continue;

            // (boxsize + 0.5 boxsize) / 2 is the max distance from center of box to center of child neighbor box
            const double cutoff_child = 1.05 * 0.75 * boxsize[i_level];
            for (auto neighb : node_lists[box].nbr) {
                if (neighb < 0)
                    continue;

                if (is_global_leaf[neighb] && src_counts_with_halo[neighb]) {
                    list1_[box][nlist1_[box]] = neighb;
                    nlist1_[box]++;
                    continue;
                }

                for (auto child : node_lists[neighb].child) {
                    if (child < 0 || !src_counts_with_halo[child])
                        continue;

                    bool inrange = true;
                    for (int k = 0; k < DIM; ++k) {
                        const double distance = std::abs(center_ptr(box)[k] - center_ptr(child)[k]);
                        if (distance > cutoff_child) {
                            inrange = false;
                            break;
                        }
                    }
                    if (inrange) {
                        list1_[box][nlist1_[box]] = child;
                        nlist1_[box]++;
                    }
                }
            }

            // We are checking for the colleagues of our parents for leaves, and level 0 has no parent
            if (i_level == 0)
                continue;

            // Search the colleagues of parent for neighboring leaves
            // (boxsize + 2 * boxsize) / 2 is the max distance from center of box to center of parent neighbor box
            const double cutoff = 1.5 * 1.05 * boxsize[i_level];
            for (auto neighb : node_lists[node_lists[box].parent].nbr) {
                if (neighb < 0 || !is_global_leaf[neighb])
                    continue;

                bool inrange = true;
                for (int k = 0; k < DIM; ++k) {
                    const double distance = std::abs(center_ptr(box)[k] - center_ptr(neighb)[k]);
                    if (distance > cutoff)
                        inrange = false;
                }
                if (inrange) {
                    list1_[box][nlist1_[box]] = neighb;
                    nlist1_[box]++;
                }
            }
        }
    }
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::build_upward_pass_work_lists() {
    const auto &node_lists = this->GetNodeLists();
    has_proxy_from_children.ReInit(n_boxes());
    charge2proxy_work.clear();

    for (int i_level = n_levels() - 1; i_level >= 0; --i_level) {
        for (auto i_box : level_indices[i_level]) {
            has_proxy_from_children[i_box] = false;

            if (!(src_counts_owned[i_box] > 0 && ifpwexp[i_box]))
                continue;

            for (auto child : node_lists[i_box].child) {
                if (child >= 0 && src_counts_owned[child] > 0 && ifpwexp[child]) {
                    has_proxy_from_children[i_box] = true;
                    break;
                }
            }

            if (has_proxy_from_children[i_box]) {
                for (auto cb : node_lists[i_box].child) {
                    if (cb >= 0 && src_counts_owned[cb] > 0 && !ifpwexp[cb])
                        charge2proxy_work.push_back({(int)cb, i_box, i_level});
                }
            } else {
                charge2proxy_work.push_back({i_box, i_box, i_level});
            }
        }
    }
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::allocate_proxy_coefficients() {
    const int n_coeffs = params.n_mfm * sctl::pow<DIM>(n_order);

    long n_proxy_boxes_upward = 0;
    long n_proxy_boxes_downward = 0;
    for (int i = 0; i < n_boxes(); ++i) {
        if (ifpwexp[i])
            n_proxy_boxes_upward++;
        if (ifpwexp[i] || iftensprodeval[i])
            n_proxy_boxes_downward++;
    }

    sctl::Vector<sctl::Long> counts(n_boxes());
    for (int i = 0; i < n_boxes(); ++i)
        counts[i] = ifpwexp[i] ? n_coeffs : 0;

    proxy_coeffs_upward.ReInit(n_coeffs * n_proxy_boxes_upward);
    proxy_coeffs_downward.ReInit(n_coeffs * n_proxy_boxes_downward);

    this->AddData("proxy_coeffs", proxy_coeffs_upward, counts);

    proxy_coeffs_offsets.ReInit(n_boxes());
    proxy_coeffs_offsets_downward.ReInit(n_boxes());

    long last_offset = 0;
    for (int box = 0; box < n_boxes(); ++box) {
        if (counts[box]) {
            proxy_coeffs_offsets[box] = last_offset;
            last_offset += n_coeffs;
        } else {
            proxy_coeffs_offsets[box] = -1;
        }
    }

    last_offset = 0;
    for (int box = 0; box < n_boxes(); ++box) {
        if (ifpwexp[box] || iftensprodeval[box]) {
            proxy_coeffs_offsets_downward[box] = last_offset;
            last_offset += n_coeffs;
        } else {
            proxy_coeffs_offsets_downward[box] = -1;
        }
    }
}

/// @brief Build any bookkeeping data associated with the tree
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename T, int DIM>
void DMKPtTree<T, DIM>::generate_metadata() {
    sctl::Profile::Scoped profile("generate_metadata", &comm_);
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    const auto &node_lists = this->GetNodeLists();

    compute_data_offsets();
    compute_level_indices_and_boxsizes();
    compute_box_centers();
    accumulate_subtree_counts();
    gather_owned_source_positions();
    broadcast_global_leaf_status();
    compute_proxy_expansion_flags();
    compute_proxy_evaluation_flags();
    build_plane_wave_interaction_lists();
    build_direct_interaction_lists();
    build_upward_pass_work_lists();
    allocate_proxy_coefficients();
    r_src_t = nda::transpose(matrixview<T>({DIM, src_counts_owned[0]}, r_src_owned_ptr(0)));
    r_trg_t = nda::transpose(matrixview<T>({DIM, trg_counts_owned[0]}, r_trg_owned_ptr(0)));
}

/// @brief Fill out the proxy coefficients used in the upward pass
///
/// Updates: proxy_coeffs
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename T, int DIM>
void DMKPtTree<T, DIM>::upward_pass() {
    sctl::Profile::Scoped profile("upward_pass", &comm_);
    sctl::Profile::Tic("upward_pass_init", &comm_);
    auto &logger = dmk::get_logger(comm_);
    auto &rank_logger = dmk::get_rank_logger(comm_);
    const std::size_t n_coeffs = params.n_mfm * sctl::pow<DIM>(n_order);
    logger->info("upward pass started");

#pragma omp parallel
#pragma omp single
    workspaces_.ReInit(MY_OMP_GET_NUM_THREADS());

    sctl::Vector<sctl::Long> counts;
    this->GetData(proxy_coeffs_upward, counts, "proxy_coeffs");
    proxy_coeffs_upward.SetZero();

    constexpr int n_children = 1u << DIM;
    const auto &node_lists = this->GetNodeLists();
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();

    sctl::Profile::Toc();
    {
        sctl::Profile::Scoped profile("charge2proxy", &comm_);
        sctl::Profile::Tic("charge2proxy", &comm_);

        // charge2proxycharge
#pragma omp parallel
        {
            sctl::Vector<T> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];

#pragma omp for schedule(static)
            for (int i = 0; i < charge2proxy_work.size(); ++i) {
                const auto &w = charge2proxy_work[i];
                proxy::charge2proxycharge<T, DIM>(r_src_owned_view(w.src_box), charge_owned_view(w.src_box),
                                                  center_view(w.center_box), 2.0 / boxsize[w.level],
                                                  proxy_view_upward(w.center_box), workspace);
            }
        }
        sctl::Profile::Toc();

        sctl::Profile::Tic("tensorprod::transform", &comm_);
#pragma omp parallel
        {
            sctl::Vector<T> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];

            for (int i_level = n_levels() - 1; i_level >= 0; --i_level) {
#pragma omp for schedule(static)
                for (int idx = 0; idx < level_indices[i_level].Dim(); ++idx) {
                    const int i_box = level_indices[i_level][idx];

                    if (!has_proxy_from_children[i_box])
                        continue;

                    auto &children = node_lists[i_box].child;
                    for (int ic = 0; ic < n_children; ++ic) {
                        const int cb = children[ic];
                        if (cb < 0 || !(src_counts_owned[cb] > 0 && ifpwexp[cb]))
                            continue;

                        const ndview<T, 2> c2p_view({n_order, DIM}, &c2p[ic * DIM * n_order * n_order]);
                        tensorprod::transform<T, DIM>(params.n_mfm, true, proxy_view_upward(cb), c2p_view,
                                                      proxy_view_upward(i_box), workspace);
                    }
                }
            }
        }
        sctl::Profile::Toc();

#ifdef SCTL_PROFILE
        comm_.Barrier();
#endif
    }

    sctl::Profile::Tic("broadcast_proxy_coeffs", &comm_);
    logger->debug("Finished building proxy charges");
    this->template ReduceBroadcast<T>("proxy_coeffs");
    this->GetData(proxy_coeffs_upward, counts, "proxy_coeffs");
    long last_offset = 0;
    for (int box = 0; box < n_boxes(); ++box) {
        if (counts[box]) {
            proxy_coeffs_offsets[box] = last_offset;
            last_offset += n_coeffs;
        } else
            proxy_coeffs_offsets[box] = -1;
    }
    sctl::Profile::Toc();
    logger->debug("proxy: finished broadcasting proxy charges");
    logger->info("upward pass finished");
}

template <typename T, int DIM>
void multiply_kernelFT_cd2p(const sctl::Vector<T> &radialft, auto &&pwexp) {
    const int nd = pwexp.extent(DIM);
    const int nexp = radialft.Dim();
    ndview<std::complex<T>, 2> pwexp_flat({nexp, nd}, pwexp.data());

    ndview<const T, 1> radialft_view({nexp}, &radialft[0]);
    for (int ind = 0; ind < nd; ++ind) {
        ndview<std::complex<T>, 1> pwexp_view({nexp}, &pwexp_flat(0, ind));
        pwexp_view *= radialft_view;
    }
    // Real * complex is two multiplies and two adds
    const unsigned long n_flops = 4 * nd * nexp;
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, n_flops);
}

template <typename Complex, int DIM>
void shift_planewave(const ndview<Complex, DIM + 1> &pwexp1_, ndview<Complex, DIM + 1> &pwexp2_,
                     const ndview<const Complex, 1> &wpwshift) {
    // Flatten our views
    const int nd = pwexp1_.extent(DIM);
    const int nexp = wpwshift.extent(0);
    dmk::ndview<const Complex, 2> pwexp1({nexp, nd}, pwexp1_.data());
    dmk::ndview<Complex, 2> pwexp2({nexp, nd}, pwexp2_.data());

    using ArrayMap = ndview<Complex, 1>;
    using ConstArrayMap = ndview<const Complex, 1>;
    ConstArrayMap wpwshift_view({nexp}, &wpwshift(0));
    for (int ind = 0; ind < nd; ++ind) {
        ConstArrayMap pw1_view({nexp}, &pwexp1(0, ind));
        ArrayMap pw2_view({nexp}, &pwexp2(0, ind));
        pw2_view += pw1_view * wpwshift_view;
    }
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::init_planewave_data() {
    sctl::Profile::Scoped profile("init_planewave_data", &comm_);
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * params.n_mfm;

    if (!pw_out_offsets.Dim()) {
        pw_out_offsets.ReInit(n_boxes());
        pw_out_offsets[0] = 0;
        int n_pw_boxes_out = 1;
        int64_t last_offset = n_pw_per_box;
        for (int box = 1; box < n_boxes(); ++box) {
            if (ifpwexp[box]) {
                pw_out_offsets[box] = last_offset;
                last_offset += n_pw_per_box;
                n_pw_boxes_out++;
            } else
                pw_out_offsets[box] = -1;
        }
        pw_out.ReInit(n_pw_per_box * n_pw_boxes_out);
    }
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::form_outgoing_expansions(const sctl::Vector<int> &boxes,
                                                    const ndview<std::complex<Real>, 2> &poly2pw_view,
                                                    const sctl::Vector<Real> &radialft) {
#ifdef DMK_INSTRUMENT
    double dt = -MY_OMP_GET_WTIME();
#endif
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * params.n_mfm;

    // Form the outgoing expansion Φl(box) for the difference kernel Dl from the proxy charge expansion
    // coefficients using Tprox2pw
#pragma omp parallel
    {
        sctl::Vector<Real> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];

#pragma omp for schedule(static)
        for (auto box : boxes) {
            // FIXME: HACK. offsets are set to -1 when not in halo, i assume is the issue
            if (ifpwexp[box] && proxy_coeffs_offsets[box] != -1) {
                dmk::proxy::proxycharge2pw<Real, DIM>(proxy_view_upward(box), poly2pw_view, pw_out_view(box),
                                                      workspace);
                multiply_kernelFT_cd2p<Real, DIM>(radialft, pw_out_view(box));
            } else if (proxy_coeffs_offsets_downward[box] != -1)
                pw_out_view(box) = 0;
        }
    }

#ifdef DMK_INSTRUMENT
    dt += MY_OMP_GET_WTIME();
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::CUSTOM1, (unsigned long)(1e9 * dt));
#endif
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::form_eval_expansions(const sctl::Vector<int> &boxes,
                                                const sctl::Vector<std::complex<Real>> &wpwshift, Real boxsize,
                                                const ndview<std::complex<Real>, 2> &pw2poly_view,
                                                const sctl::Vector<Real> &p2c) {
#ifdef DMK_INSTRUMENT
    double dt = -MY_OMP_GET_WTIME();
#endif
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * params.n_mfm;
    const auto &node_lists = this->GetNodeLists();
    const auto &node_attr = this->GetNodeAttr();
    const Real sc = 2.0 / boxsize;
    const int nd = params.n_mfm;

    unsigned long n_shifts{0};
#pragma omp parallel
    {
        sctl::Vector<Real> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];
        sctl::Vector<std::complex<Real>> pw_in(n_pw_per_box);

        auto pw_in_view = [this, &pw_in]() {
            if constexpr (DIM == 2)
                return ndview<std::complex<Real>, DIM + 1>({n_pw, (n_pw + 1) / 2, params.n_mfm}, &pw_in[0]);
            else if constexpr (DIM == 3)
                return ndview<std::complex<Real>, DIM + 1>({n_pw, n_pw, (n_pw + 1) / 2, params.n_mfm}, &pw_in[0]);
        }();

#pragma omp for schedule(dynamic) reduction(+ : n_shifts)
        for (auto box : boxes) {
            const int nboxpts = src_counts_owned[box] + trg_counts_owned[box];

            if (ifpwexp[box] && nboxpts) {
                memcpy(&pw_in[0], pw_out_ptr(box), n_pw_per_box * sizeof(std::complex<Real>));
                for (auto &neighbor : node_lists[box].nbr) {
                    if (neighbor >= 0 && neighbor != box && (!is_global_leaf[box] || !is_global_leaf[neighbor])) {
                        // Translate the outgoing expansion Φl(colleague) to the center of box and add to the incoming
                        // plane wave expansion Ψl(box) using wpwshift.

                        // note: neighbors in SCTL are sorted in reverse order to wpwshift
                        // FIXME: check if valid for periodic boundary conditions
                        constexpr int n_neighbors = sctl::pow<DIM>(3);
                        const int ind = n_neighbors - 1 - (&neighbor - &node_lists[box].nbr[0]);
                        assert(ind >= 0 && ind < n_neighbors);

                        ndview<const std::complex<Real>, 1> wpwshift_view({n_pw_per_box},
                                                                          &wpwshift[n_pw_per_box * ind]);
                        shift_planewave<std::complex<Real>, DIM>(pw_out_view(neighbor), pw_in_view, wpwshift_view);
                        n_shifts++;
                    }
                }

                // Convert incoming plane wave expansion Ψl(box) to the local expansion Λl(box) using Tpw2poly
                dmk::planewave_to_proxy_potential<Real, DIM>(pw_in_view, pw2poly_view, proxy_view_downward(box),
                                                             workspace);

                if (!iftensprodeval[box]) {
                    // Translate and add the local expansion of Λl(box) to the local expansion of Λl(child).
                    constexpr int n_children = 1u << DIM;
                    for (int i_child = 0; i_child < n_children; ++i_child) {
                        const int child = node_lists[box].child[i_child];
                        if (child < 0 || !(src_counts_owned[child] + trg_counts_owned[child]))
                            continue;
                        const ndview<Real, 2> p2c_view({n_order, DIM},
                                                       const_cast<Real *>(&p2c[i_child * DIM * n_order * n_order]));
                        tensorprod::transform<Real, DIM>(nd, true, proxy_view_downward(box), p2c_view,
                                                         proxy_view_downward(child), workspace);
                    }
                }
            }

            if (iftensprodeval[box]) {
                if (src_counts_owned[box])
                    proxy::eval_targets<Real, DIM>(proxy_view_downward(box), r_src_owned_view(box), center_view(box),
                                                   sc, pot_src_view(box), workspace);
                if (trg_counts_owned[box])
                    proxy::eval_targets<Real, DIM>(proxy_view_downward(box), r_trg_owned_view(box), center_view(box),
                                                   sc, pot_trg_view(box), workspace);
            }
        }
    }

    // 1 complex multiply (4 multiplies and 2 adds) and 1 complex add (2 adds) per plane wave component
    constexpr int flops_per_pw = 8;
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, n_shifts * flops_per_pw * n_pw_per_box);
#ifdef DMK_INSTRUMENT
    dt += MY_OMP_GET_WTIME();
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::CUSTOM2, (unsigned long)(1e9 * dt));
#endif
}

template <typename Real>
Real calc_log_windowed_kernel_value_at_zero(int dim, const FourierData<Real> &fourier_data, Real boxsize) {
    const Real psi0 = fourier_data.prolate0_fun.eval_val(0.0);
    const Real beta = fourier_data.beta();
    constexpr int n_quad = 100;
    std::array<Real, n_quad> xs, whts;
    legerts(1, n_quad, xs.data(), whts.data());
    for (int i = 0; i < n_quad; ++i) {
        xs[i] = 0.5 * (xs[i] + Real{1.0}) * beta / boxsize;
        whts[i] *= 0.5 * beta / boxsize;
    }

    const Real rl = boxsize * sqrt(dim * 1.0) * 2;
    const Real dfac = rl * std::log(rl);

    Real fval = 0.0;
    for (int i = 0; i < n_quad; ++i) {
        const Real xval = xs[i] * boxsize / beta;
        const Real fval0 = fourier_data.prolate0_fun.eval_val(xval);
        const Real z = rl * xs[i];
        const Real dj0 = util::cyl_bessel_j(0, z);
        const Real dj1 = util::cyl_bessel_j(1, z);
        const Real tker = -(1 - dj0) / (xs[i] * xs[i]) + dfac * dj1 / xs[i];
        const Real fhat = tker * fval0 / psi0;
        fval += fhat * whts[i] * xs[i];
    }

    return fval;
}

template <typename Real, int DIM>
Real get_self_interaction_constant(FourierData<Real> &fourier_data, dmk_ikernel kernel, int i_level, Real boxsize) {
    const double bsize = i_level == 0 ? 0.5 * boxsize : boxsize;
    const double w0 = [&]() -> Real {
        if (kernel == DMK_YUKAWA)
            return fourier_data.yukawa_windowed_kernel_value_at_zero(i_level);
        else if (kernel == DMK_LAPLACE) {
            const Real psi0 = fourier_data.prolate0_fun.eval_val(0.0);
            const auto c = fourier_data.prolate0_fun.intvals(fourier_data.beta());
            if constexpr (DIM == 2) {
                const auto log_windowed_kernel_at_zero =
                    calc_log_windowed_kernel_value_at_zero(DIM, fourier_data, Real{1.0});
                return log_windowed_kernel_at_zero - i_level * std::log(2.0);
            } else if constexpr (DIM == 3)
                return psi0 / (c[0] * bsize);
            else
                throw std::runtime_error("Unsupported kernel DMK_LAPLACE, DIM = " + std::to_string(DIM));
        } else if (kernel == DMK_SQRT_LAPLACE) {
            const Real psi0 = fourier_data.prolate0_fun.eval_val(0.0);
            const auto c = fourier_data.prolate0_fun.intvals(fourier_data.beta());
            if constexpr (DIM == 2)
                return psi0 / (c[0] * bsize);
            if constexpr (DIM == 3)
                return psi0 / (2 * c[1] * bsize * bsize);

        } else
            throw std::runtime_error("Unsupported kernel");
    }();

    return w0;
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::evaluate_direct_interactions(const Real *r_src_t, const Real *r_trg_t) {
    sctl::Profile::Scoped profile("evaluate_direct_interactions", &comm_);
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    const auto &node_lists = this->GetNodeLists();
    Real w0[SCTL_MAX_DEPTH];
    for (int i_level = 0; i_level < n_levels(); ++i_level)
        w0[i_level] = get_self_interaction_constant<Real, DIM>(fourier_data, params.kernel, i_level, boxsize[i_level]);

#pragma omp parallel for schedule(dynamic)
    for (int i_box = 0; i_box < n_boxes(); ++i_box) {
        const int n_src_i = src_counts_owned[i_box];
        const int n_trg_i = trg_counts_owned[i_box];
        const int i_level = node_mid[i_box].Depth();
        for (auto j_box : list1(i_box)) {
            int j_level = node_mid[j_box].Depth();
            Real bsize = boxsize[j_level];
            // now find the interaction range of the residual kernel
            if (ifpwexp[j_box] && j_box == i_box) {
                // when ifpwexp(jbox)=1, self interaction at its own
                // level is taken care of by plane - wave expansion
                bsize /= Real{2.0};
                j_level = j_level + 1;
            } else if (j_level < i_level) {
                // when the source box is bigger than the target box, residual interaction
                // starts from the target box level
                bsize = boxsize[i_level];
                j_level = i_level;
            }

            // kernel truncated at bsize, i.e., K(x,y)=0 for |x-y|^2 > d2max
            const Real d2max = bsize * bsize;
            const Real bsizeinv = Real{1} / bsize;

            // used in the kernel approximatin for boxes in list1
            Real rsc = 2 * bsizeinv;
            Real cen = -bsize / Real{2};
            const auto &cheb_coeffs = fourier_data.cheb_coeffs(j_level);

            if ((params.kernel == DMK_SQRT_LAPLACE && DIM == 3) || (params.kernel == DMK_LAPLACE && DIM == 2)) {
                rsc = 2 * bsizeinv * bsizeinv;
                cen = Real{-1.0};
            }
            if (params.kernel == DMK_YUKAWA) {
                rsc = 2 * bsizeinv;
                cen = Real{-1.0};
            }

            std::array<std::span<const Real>, DIM> r_trg;
            assert(is_global_leaf[i_box]);
            assert(is_global_leaf[j_box]);
            if (n_src_i) {
                for (int i = 0; i < DIM; ++i)
                    r_trg[i] = std::span<const Real>(
                        r_src_t + (r_src_offsets_owned[i_box] / DIM) + src_counts_owned[0] * i, n_src_i);

                direct_eval<Real, DIM>(params.kernel, r_src_with_halo_view(j_box), r_trg, charge_with_halo_view(j_box),
                                       cheb_coeffs, &params.fparam, rsc, cen, d2max, pot_src_view(i_box), n_digits);
            }
            if (n_trg_i) {
                for (int i = 0; i < DIM; ++i)
                    r_trg[i] = std::span<const Real>(
                        r_trg_t + (r_trg_offsets_owned[i_box] / DIM) + trg_counts_owned[0] * i, n_trg_i);

                direct_eval<Real, DIM>(params.kernel, r_src_with_halo_view(j_box), r_trg, charge_with_halo_view(j_box),
                                       cheb_coeffs, &params.fparam, rsc, cen, d2max, pot_trg_view(i_box), n_digits);
            }
        }

        if (!n_src_i)
            continue;

        // Correct for self-evaluations
        auto pot = pot_src_view(i_box);
        auto charge = charge_with_halo_view(i_box);
        const auto depth = node_mid[i_box].Depth() + ifpwexp[i_box];
        const auto correction_factor = w0[depth];
        for (int i_src = 0; i_src < r_src_cnt_with_halo[i_box]; ++i_src)
            for (int i = 0; i < params.n_mfm; ++i)
                pot(i, i_src) -= correction_factor * charge(i, i_src);
    }
}

/// @brief Perform the "downward pass"
///
/// Updates: proxy_coeffs_downward, tree 'pdmk_pot' particle data
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename T, int DIM>
void DMKPtTree<T, DIM>::downward_pass() {
    sctl::Profile::Scoped prof("downward_pass", &comm_);
    sctl::Profile::Tic("downward_pass_init", &comm_);
    auto &logger = dmk::get_logger(comm_);
    auto &rank_logger = dmk::get_rank_logger(comm_);
    logger->info("downward pass started");

    pot_src_sorted.SetZero();
    pot_trg_sorted.SetZero();

    // FIXME: This should be assigned automatically at tree construction (fourier_data should be subobject)
    n_pw = fourier_data.n_pw();

    init_planewave_data();
    const long n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    sctl::Vector<std::complex<T>> wpwshift(n_pw_modes * sctl::pow<DIM>(3));
    sctl::Vector<T> radialft(n_pw_modes);
    sctl::Vector<T> kernel_ft;
    get_windowed_kernel_ft<T, DIM>(params.kernel, &params.fparam, fourier_data.beta(), n_digits, boxsize[0],
                                   fourier_data.prolate0_fun, kernel_ft);
    util::mk_tensor_product_fourier_transform(DIM, n_pw, ndview<T, 1>({kernel_ft.Dim()}, &kernel_ft[0]),
                                              ndview<T, 1>({n_pw_modes}, &radialft[0]));

    sctl::Vector<std::complex<T>> poly2pw(n_order * n_pw), pw2poly(n_order * n_pw);
    fourier_data.calc_planewave_coeff_matrices(-1, n_order, poly2pw, pw2poly);
    const ndview<std::complex<T>, 2> poly2pw_view({n_pw, n_order}, &poly2pw[0]);
    const ndview<std::complex<T>, 2> pw2poly_view({n_pw, n_order}, &pw2poly[0]);
    dmk::proxy::proxycharge2pw<T, DIM>(proxy_view_upward(0), poly2pw_view, pw_out_view(0), workspaces_[0]);
    multiply_kernelFT_cd2p<T, DIM>(radialft, pw_out_view(0));
    proxy_coeffs_downward.SetZero();
    dmk::planewave_to_proxy_potential<T, DIM>(pw_out_view(0), pw2poly_view, proxy_view_downward(0), workspaces_[0]);

    sctl::Profile::Toc();
    sctl::Profile::Tic("expansion_propagation_and_eval", &comm_);
    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        // Initialize everything for this level
        // 1. Difference kernel
        // 2. Radial fourier transform of the difference kernel
        // 3. Planewave <-> polynomial coefficient conversion matrices
        // 4. Planewave translation matrix
        {
            // sctl::Profile::Scoped profile("downward_pass_loop_init", &comm_);
            const bool is_root = i_level == 0;
            get_difference_kernel_ft<T, DIM>(is_root, params.kernel, &params.fparam, fourier_data.beta(), n_digits,
                                             boxsize[i_level], fourier_data.prolate0_fun, kernel_ft);
            util::mk_tensor_product_fourier_transform(DIM, n_pw, ndview<T, 1>({kernel_ft.Dim()}, &kernel_ft[0]),
                                                      ndview<T, 1>({n_pw_modes}, &radialft[0]));
            fourier_data.calc_planewave_coeff_matrices(i_level, n_order, poly2pw, pw2poly);
            dmk::calc_planewave_translation_matrix<DIM>(1, boxsize[i_level], n_pw,
                                                        fourier_data.difference_kernel(i_level).hpw, wpwshift);
        }
        form_outgoing_expansions(level_indices[i_level], poly2pw_view, radialft);
        if (!debug_omit_pw)
            form_eval_expansions(level_indices[i_level], wpwshift, boxsize[i_level], pw2poly_view, p2c);
    }
    sctl::Profile::Toc();
    if (!debug_omit_direct)
        evaluate_direct_interactions(r_src_t.data(), r_trg_t.data());

    logger->info("downward pass completed");
    if (debug_dump_tree)
        dump();
}

#ifdef DMK_HAVE_MPI
MPI_TEST_CASE("[DMK] 3D: Proxy charges on upward pass, 2 ranks", 2) {
    constexpr int n_dim = 3;
    constexpr int n_src = 10000;
    constexpr int n_trg = n_src;
    constexpr int n_charge_dim = 1;
    constexpr bool uniform = false;

    sctl::Vector<double> r_src, r_trg, r_src_norms, charges, dipoles, pot_src, grad_src, hess_src, pot_trg, grad_trg,
        hess_trg;
    if (test_rank == 0)
        dmk::util::init_test_data(n_dim, n_charge_dim, n_src, n_trg, uniform, true, r_src, r_trg, r_src_norms, charges,
                                  dipoles, 0);

    pdmk_params params;
    params.eps = 1E-6;
    params.kernel = DMK_YUKAWA;
    params.log_level = SPDLOG_LEVEL_OFF;
    params.fparam = 6.0;
    params.n_dim = n_dim;
    params.n_mfm = n_charge_dim;
    params.n_per_leaf = 80;

    double st = MY_OMP_GET_WTIME();
#ifdef DMK_HAVE_MPI
    auto comm = sctl::Comm(test_comm);
#else
    auto comm = sctl::Comm::Self();
#endif
    DMKPtTree<double, n_dim> tree(comm, params, r_src, r_trg, charges);
    tree.upward_pass();
    tree.downward_pass();
    tree.GetParticleData(pot_src, "pdmk_pot_src");
    tree.GetParticleData(pot_trg, "pdmk_pot_trg");
    if (test_rank == 0)
        std::cout << MY_OMP_GET_WTIME() - st << std::endl;

    if (test_rank == 0) {
        dmk::util::init_test_data(n_dim, n_charge_dim, n_src, n_trg, uniform, true, r_src, r_trg, r_src_norms, charges,
                                  dipoles, 0);

        DMKPtTree<double, n_dim> tree_single(sctl::Comm::Self(), params, r_src, r_trg, charges);
        tree_single.upward_pass();
        tree_single.downward_pass();
        sctl::Vector<double> pot_src_single, pot_trg_single;
        tree_single.GetParticleData(pot_src_single, "pdmk_pot_src");
        tree_single.GetParticleData(pot_trg_single, "pdmk_pot_trg");

        sleep(test_rank);
        auto &node_mid = tree.GetNodeMID();
        for (int ibox = 0; ibox < tree.n_boxes(); ++ibox) {
            std::array<double, 3> x, x_single;
            for (int i = 0; i < 3; ++i)
                x[i] = tree.center_ptr(ibox)[i];

            int single_box = -1;
            for (int jbox = 0; jbox < tree_single.n_boxes(); ++jbox) {
                for (int i = 0; i < 3; ++i)
                    x_single[i] = tree_single.center_ptr(jbox)[i];

                if (x_single == x) {
                    single_box = jbox;
                    break;
                }
            }
            if (single_box < 0) {
                std::cout << fmt::format("No match for box {}\n", ibox);
                continue;
            }

            double maxerr_up = 0.0;
            if (ibox == 0 || tree.ifpwexp[ibox]) {
                // FIXME: ifpwexp is no longer a sufficient check for valid proxy coeffs, since
                // we only care about halos, not the global ifpwexp flag.
                if (tree_single.proxy_coeffs_offsets[single_box] == -1 || tree.proxy_coeffs_offsets[ibox] == -1)
                    break;

                double max_actual = 0.0;
                for (int i = 0; i < sctl::pow<3>(tree.n_order); ++i) {
                    const double actual = tree_single.proxy_ptr_upward(single_box)[i];
                    const double mpi = tree.proxy_ptr_upward(ibox)[i];
                    const double err = std::abs(mpi - actual);
                    maxerr_up = std::max(err, maxerr_up);
                    max_actual = std::max(std::abs(actual), max_actual);
                }
                maxerr_up = max_actual ? maxerr_up / max_actual : 0.0;
            }

            if (maxerr_up > 1E-14)
                std::cout << fmt::format("{:3} {:3} {:3} {:5.4E} {:3} ({} {} {})\n", test_rank, ibox, single_box,
                                         maxerr_up, node_mid[ibox].Depth(), x[0], x[1], x[2]);
        }

        if (test_rank == 0) {
            for (int i = 0; i < n_src; ++i) {
                double err_src = pot_src[i] == 0. ? 0.0 : std::abs(1.0 - pot_src[i] / pot_src_single[i]);
                double err_trg = pot_trg[i] == 0. ? 0.0 : std::abs(1.0 - pot_trg[i] / pot_trg_single[i]);
                if (err_src > 1E-12 || err_trg > 1E-12)
                    std::cout << fmt::format("{} {:5.4E} {:5.4E}\n", i, err_src, err_trg);
            }
        }
    }
}
#endif

template struct DMKPtTree<float, 2>;
template struct DMKPtTree<float, 3>;
template struct DMKPtTree<double, 2>;
template struct DMKPtTree<double, 3>;

} // namespace dmk
