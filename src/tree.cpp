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
#include <omp.h>
#include <sctl/profile.hpp>
#include <stdexcept>
#include <unistd.h>

#include <dmk/omp_wrapper.hpp>

namespace dmk {

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::dump() const {
    rank_logger->info("Dumping DMKPtTree data on rank {} of comm size {}", comm_.Rank(), comm_.Size());

    auto dumper = [this](const std::string &name, const auto &data) {
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

        rank_logger->info("Dumped {}", filename);
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
        } params_data{DIM, n_order, expansion_constants.n_pw_diff, sizeof(Real)}; // FIXME: n_pw_win vs diff
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
    : sctl::PtTree<Real, DIM>(comm), comm_(comm), params(params_),
      kernel_input_dim(get_kernel_input_dim(params.n_dim, params.kernel)),
      kernel_output_dim_src(get_kernel_output_dim(params.n_dim, params.kernel, params.pgh_src)),
      kernel_output_dim_trg(get_kernel_output_dim(params.n_dim, params.kernel, params.pgh_trg)),
      kernel_output_dim_max(std::max(kernel_output_dim_src, kernel_output_dim_trg)),
      n_tables(1), // Placeholder for when we need more proxy coefficient tables
      n_digits(std::round(log10(1.0 / params_.eps) - 0.1)), expansion_constants(params),
      logger(dmk::get_logger(comm, params.log_level)), rank_logger(dmk::get_rank_logger(comm, params.log_level)) {
    sctl::Profile::Scoped profile("DMKPtTree::DMKPtTree", &comm_);
    debug_omit_pw = (params.debug_flags & DMK_DEBUG_OMIT_PW) || util::env_is_set("DMK_DEBUG_OMIT_PW");
    debug_omit_direct = (params.debug_flags & DMK_DEBUG_OMIT_DIRECT) || util::env_is_set("DMK_DEBUG_OMIT_DIRECT");
    debug_dump_tree = (params.debug_flags & DMK_DEBUG_DUMP_TREE) || util::env_is_set("DMK_DEBUG_DUMP_TREE");
    debug_force_aot = (params.debug_flags & DMK_DEBUG_FORCE_AOT) || util::env_is_set("DMK_DEBUG_FORCE_AOT");

    logger->info("tree build started");

    const int n_src = r_src.Dim() / DIM;
    const int n_trg = r_trg.Dim() / DIM;

    // 0: Initialization
    sctl::Vector<Real> pot_vec_src(n_src * kernel_output_dim_src);
    sctl::Vector<Real> pot_vec_trg(n_trg * kernel_output_dim_trg);

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
    generate_metadata();
    logger->info("tree build completed");
}

template <typename Real, int DIM>
int DMKPtTree<Real, DIM>::update_charges(const Real *charge, const Real *normal, const Real *dipole_str) {
    if (normal || dipole_str) {
        std::cerr << "normal updates and dipoles not supported yet\n";
        return 1;
    }
    auto &logger = dmk::get_logger(comm_, params.log_level);
    logger->info("update_charges started");

    const int n_src = r_src_sorted_owned.Dim() / DIM;

    // Wrap the incoming (unsorted) charge data in a Vector without owning it
    sctl::Vector<Real> charge_vec(n_src * kernel_input_dim, const_cast<Real *>(charge), false);

    // Delete the old charge data and re-register with the new values.
    // The PtTree already knows the sort permutation from the "pdmk_src"
    // particle set, so AddParticleData will sort the new charges into
    // tree order automatically.
    this->DeleteParticleData("pdmk_charge");
    this->AddParticleData("pdmk_charge", "pdmk_src", charge_vec);

    // Retrieve the sorted owned charges
    {
        sctl::Vector<Real> data;
        sctl::Vector<long> count;
        this->GetData(data, count, "pdmk_charge");
        charge_sorted_owned = data;
        charge_cnt_owned = count;
    }

    // Recompute charge offsets (owned)
    charge_offsets_owned[0] = 0;
    for (std::size_t i = 1; i < n_boxes(); ++i)
        charge_offsets_owned[i] = charge_offsets_owned[i - 1] + kernel_input_dim * charge_cnt_owned[i - 1];

    // Broadcast to halo/ghost nodes and retrieve
    this->template Broadcast<Real>("pdmk_charge");
    this->GetData(charge_sorted_with_halo, charge_cnt_with_halo, "pdmk_charge");

    // Recompute charge offsets (with halo)
    charge_offsets_with_halo[0] = 0;
    for (std::size_t i = 1; i < n_boxes(); ++i)
        charge_offsets_with_halo[i] = charge_offsets_with_halo[i - 1] + kernel_input_dim * charge_cnt_with_halo[i - 1];

    logger->info("update_charges completed");
    return 0;
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::compute_data_offsets() {
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
        pot_src_offsets[i] = pot_src_offsets[i - 1] + kernel_output_dim_src * pot_src_cnt[i - 1];
        pot_trg_offsets[i] = pot_trg_offsets[i - 1] + kernel_output_dim_trg * pot_trg_cnt[i - 1];
        charge_offsets_owned[i] = charge_offsets_owned[i - 1] + kernel_input_dim * charge_cnt_owned[i - 1];
        charge_offsets_with_halo[i] = charge_offsets_with_halo[i - 1] + kernel_input_dim * charge_cnt_with_halo[i - 1];
    }
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::compute_level_indices_and_boxsizes() {
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

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::compute_box_centers() {
    const auto &node_mid = this->GetNodeMID();
    centers.ReInit(n_boxes() * DIM);
    Real scale = 1.0;
    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        for (auto i_node : level_indices[i_level]) {
            auto node_origin = node_mid[i_node].template Coord<Real>();
            for (int i = 0; i < DIM; ++i)
                centers[i_node * DIM + i] = node_origin[i] + 0.5 * scale;
        }
        scale *= 0.5;
    }
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::accumulate_subtree_counts() {
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

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::gather_owned_source_positions() {
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

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::broadcast_global_leaf_status() {
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

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::compute_proxy_expansion_flags() {
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

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::compute_proxy_evaluation_flags() {
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

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::build_plane_wave_interaction_lists() {
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

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::build_direct_interaction_lists() {
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
                if (neighb < 0 || !is_global_leaf[neighb] || !src_counts_with_halo[neighb])
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

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::build_upward_pass_work_lists() {
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

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::allocate_proxy_coefficients() {
    const int n_coeffs = n_tables * sctl::pow<DIM>(expansion_constants.n_order);

    sctl::Vector<sctl::Long> counts_upward(n_boxes());
    sctl::Vector<sctl::Long> counts_downward(n_boxes());
    long n_proxy_boxes_upward = 0;
    long n_proxy_boxes_downward = 0;
    for (int i = 0; i < n_boxes(); ++i) {
        if (ifpwexp[i] && src_counts_with_halo[i] > 0) {
            counts_upward[i] = n_coeffs;
            n_proxy_boxes_upward++;
        } else {
            counts_upward[i] = 0;
        }

        if ((ifpwexp[i] || iftensprodeval[i]) && (src_counts_owned[i] + trg_counts_owned[i])) {
            counts_downward[i] = n_coeffs;
            n_proxy_boxes_downward++;
        } else {
            counts_downward[i] = 0;
        }
    }

    proxy_coeffs_upward.ReInit(n_coeffs * n_proxy_boxes_upward);
    proxy_coeffs_downward.ReInit(n_coeffs * n_proxy_boxes_downward);

    this->AddData("proxy_coeffs", proxy_coeffs_upward, counts_upward);
    this->GetData(proxy_coeffs_upward, counts_upward, "proxy_coeffs");

    proxy_coeffs_offsets.ReInit(n_boxes());
    proxy_coeffs_offsets_downward.ReInit(n_boxes());

    long last_offset = 0;
    for (int box = 0; box < n_boxes(); ++box) {
        if (counts_upward[box]) {
            proxy_coeffs_offsets[box] = last_offset;
            last_offset += n_coeffs;
        } else {
            proxy_coeffs_offsets[box] = -1;
        }
    }

    last_offset = 0;
    for (int box = 0; box < n_boxes(); ++box) {
        if (counts_downward[box]) {
            proxy_coeffs_offsets_downward[box] = last_offset;
            last_offset += n_coeffs;
        } else {
            proxy_coeffs_offsets_downward[box] = -1;
        }
    }
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::precompute_window_difference_data() {
    const int n_pw = expansion_constants.n_pw_win;
    const long n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    difference_fourier_data.resize(n_levels());

    sctl::Vector<Real> kernel_ft;

    window_fourier_data.poly2pw.ReInit(n_order * n_pw);
    window_fourier_data.pw2poly.ReInit(n_order * n_pw);
    window_fourier_data.radialft.ReInit(n_pw_modes);
    get_windowed_kernel_ft<Real, DIM>(params.kernel, &params.fparam, fourier_data.beta(), n_pw, boxsize[0],
                                      fourier_data.prolate0_fun, kernel_ft);
    util::mk_tensor_product_fourier_transform(DIM, n_pw, ndview<Real, 1>({kernel_ft.Dim()}, &kernel_ft[0]),
                                              ndview<Real, 1>({n_pw_modes}, &window_fourier_data.radialft[0]));
    fourier_data.calc_planewave_coeff_matrices(-1, n_order, window_fourier_data.poly2pw, window_fourier_data.pw2poly);

    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        auto &lfd = difference_fourier_data[i_level];
        lfd.radialft.ReInit(n_pw_modes);
        lfd.wpwshift.ReInit(n_pw_modes * sctl::pow<DIM>(3));
        lfd.poly2pw.ReInit(n_order * n_pw);
        lfd.pw2poly.ReInit(n_order * n_pw);

        const bool is_root = (i_level == 0);
        get_difference_kernel_ft<Real, DIM>(is_root, params.kernel, &params.fparam, fourier_data.beta(), n_pw,
                                            boxsize[i_level], fourier_data.prolate0_fun, kernel_ft);
        util::mk_tensor_product_fourier_transform(DIM, n_pw, ndview<Real, 1>({kernel_ft.Dim()}, &kernel_ft[0]),
                                                  ndview<Real, 1>({n_pw_modes}, &lfd.radialft[0]));
        fourier_data.calc_planewave_coeff_matrices(i_level, n_order, lfd.poly2pw, lfd.pw2poly);
        dmk::calc_planewave_translation_matrix<DIM>(1, boxsize[i_level], n_pw,
                                                    fourier_data.difference_kernel(i_level).hpw, lfd.wpwshift);
    }
}

/// @brief Build list of target boxes. Boxes are sorted in descending
/// order by total number of pairs interacting
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::build_direct_work_lists() {
    const auto &node_attr = this->GetNodeAttr();

    direct_work.clear();
    direct_work.reserve(n_boxes());
    for (int i_box = 0; i_box < n_boxes(); ++i_box) {
        if (is_global_leaf[i_box] && !node_attr[i_box].Ghost && nlist1_[i_box] > 0)
            direct_work.push_back(i_box);
    }

    // Sort by descending interaction cost so heaviest boxes get scheduled first
    std::sort(direct_work.begin(), direct_work.end(), [&](int a, int b) {
        const long pts_a = src_counts_owned[a] + trg_counts_owned[a];
        const long pts_b = src_counts_owned[b] + trg_counts_owned[b];
        long src_a = 0, src_b = 0;
        for (auto j : list1(a))
            src_a += src_counts_with_halo[j];
        for (auto j : list1(b))
            src_b += src_counts_with_halo[j];
        return pts_a * src_a > pts_b * src_b;
    });
}

/// @brief Build list of direct evaluators for each level, source and target.
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::build_evaluators() {
    try {
        auto src_eval = make_evaluator_aot<Real>(params.kernel, params.pgh_src, DIM, n_digits, 3);
        auto trg_eval = make_evaluator_aot<Real>(params.kernel, params.pgh_trg, DIM, n_digits, 3);
#ifdef DMK_USE_JIT
        if (!util::env_is_set("DMK_DEBUG_FORCE_AOT")) {
            src_eval =
                make_evaluator_jit<Real>(params.kernel, params.pgh_src, DIM, n_digits, expansion_constants.beta, 3);
            trg_eval =
                make_evaluator_jit<Real>(params.kernel, params.pgh_trg, DIM, n_digits, expansion_constants.beta, 3);
        }
#endif
        // FIXME: assumes the same src/trg output configuration
        evaluator_by_level_src.assign(n_levels(), src_eval);
        evaluator_by_level_trg.assign(n_levels(), trg_eval);
    } catch (std::exception &e) {
        logger->error("Failed to create direct evaluator: {}", e.what());
    }
    if (params.kernel == DMK_YUKAWA) {
        // FIXME: This should be moved to direct.cpp, but it was annoying to add the coefficient
        // binding cleanly
        for (int level = 0; level < n_levels(); ++level) {
            const auto coeffs = fourier_data.cheb_coeffs(level);
            const Real lambda = params.fparam;
            const int kernel_input_dim = this->kernel_input_dim;
            const int kernel_output_dim = kernel_output_dim_trg;
            // FIXME: assumes the same src/trg output configuration
            evaluator_by_level_src.push_back(
                [coeffs, lambda, kernel_input_dim](Real rsc, Real cen, Real d2max, Real thresh2, int n_src,
                                                   const Real *r_src_ptr, const Real *charge_ptr, int n_trg,
                                                   const Real *r_trg_ptr, Real *pot) {
                    constexpr Real threshq = 1e-30;
                    ndview<Real, 2> u({1, n_trg}, pot);
                    ndview<const Real, 2> charges({kernel_input_dim, n_src}, charge_ptr);
                    ndview<const Real, 2> r_src({DIM, n_src}, r_src_ptr);
                    ndview<const Real, 2> r_trg({DIM, n_trg}, r_trg_ptr);
                    for (int i_trg = 0; i_trg < n_trg; i_trg++) {
                        for (int i_src = 0; i_src < n_src; i_src++) {
                            const Real dx = r_trg(0, i_trg) - r_src(0, i_src);
                            const Real dy = r_trg(1, i_trg) - r_src(1, i_src);
                            Real dd = dx * dx + dy * dy;
                            if constexpr (DIM == 3) {
                                Real dz = r_trg(2, i_trg) - r_src(2, i_src);
                                dd += dz * dz;
                            }

                            if (dd < threshq || dd > d2max)
                                continue;

                            const Real r = sqrt(dd);
                            const Real xval = r * rsc + cen;
                            const Real fval = chebyshev::evaluate(xval, coeffs.size() + 1, coeffs.data());
                            Real dkval;
                            if constexpr (DIM == 2)
                                dkval = util::cyl_bessel_k(0, lambda * r);
                            if constexpr (DIM == 3)
                                dkval = std::exp(-lambda * r) / r;

                            const Real factor = dkval + fval;
                            for (int i = 0; i < kernel_input_dim; ++i)
                                u(i, i_trg) += charges(i, i_src) * factor;
                        }
                    }
                });
        }
        // FIXME: assumes the same src/trg output configuration
        evaluator_by_level_trg = evaluator_by_level_src;
    }
}

/// @brief Build any bookkeeping data associated with the tree
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::generate_metadata() {
    sctl::Profile::Scoped profile("generate_metadata", &comm_);
    logger->debug("generating tree traversal metadata and other constants");

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
    build_direct_work_lists();
    allocate_proxy_coefficients();
    proxy_down_zeroed.resize(n_boxes());
    std::tie(c2p, p2c) = dmk::chebyshev::get_c2p_p2c_matrices<Real>(DIM, expansion_constants.n_order);
    // FIXME. pass in correct expansion constants
    fourier_data = FourierData<Real>(params.kernel, DIM, params.eps, expansion_constants.n_pw_win, params.fparam,
                                     expansion_constants.beta, boxsize);
    precompute_window_difference_data();
    build_evaluators();

    assert(expansion_constants.n_pw_diff == fourier_data.n_pw());
    logger->debug("done generating tree traversal metadata and other constants");
}

/// @brief Fill out the proxy coefficients used in the upward pass
///
/// Updates: proxy_coeffs
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::upward_pass() {
    sctl::Profile::Scoped profile("upward_pass", &comm_);
    sctl::Profile::Tic("upward_pass_init", &comm_);
    const std::size_t n_coeffs = n_tables * sctl::pow<DIM>(expansion_constants.n_order);
    logger->info("upward pass started");

#pragma omp parallel
#pragma omp single
    workspaces_.ReInit(MY_OMP_GET_NUM_THREADS());

    sctl::Vector<sctl::Long> counts;
    this->GetData(proxy_coeffs_upward, counts, "proxy_coeffs");
    proxy_coeffs_upward = 0;

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
            sctl::Vector<Real> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];

#pragma omp for schedule(static)
            for (int i = 0; i < charge2proxy_work.size(); ++i) {
                const auto &w = charge2proxy_work[i];
                proxy::charge2proxycharge<Real, DIM>(r_src_owned_view(w.src_box), charge_owned_view(w.src_box),
                                                     center_view(w.center_box), 2.0 / boxsize[w.level],
                                                     proxy_view_upward(w.center_box), workspace);
            }
        }
        sctl::Profile::Toc();

        sctl::Profile::Tic("tensorprod::transform", &comm_);
#pragma omp parallel
        {
            sctl::Vector<Real> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];

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

                        const auto &n_order = expansion_constants.n_order;
                        const ndview<Real, 2> c2p_view({n_order, DIM}, &c2p[ic * DIM * n_order * n_order]);
                        tensorprod::transform<Real, DIM>(n_tables, true, proxy_view_upward(cb), c2p_view,
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

    this->template ReduceBroadcast<Real>("proxy_coeffs");
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

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::init_planewave_data() {
    // FIXME: Handle windowed/diff separately
    const int n_pw = expansion_constants.n_pw_diff;
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * n_tables;

    if (!pw_out_offsets.Dim()) {
        pw_out_offsets.ReInit(n_boxes());
        pw_out_offsets[0] = 0;
        int n_pw_boxes_out = 1;
        int64_t last_offset = n_pw_per_box;
        for (int box = 1; box < n_boxes(); ++box) {
            if (proxy_coeffs_offsets[box] != -1 || trg_counts_owned[box]) {
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
void DMKPtTree<Real, DIM>::form_outgoing_expansions() {
#ifdef DMK_INSTRUMENT
    double dt = -MY_OMP_GET_WTIME();
#endif
    // FIXME: Split windowed/diff size
    const int n_pw = expansion_constants.n_pw_diff;
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * n_tables;
    const auto &node_mid = this->GetNodeMID();

    // Root box: uses windowed kernel fourier data
    {
        const ndview<std::complex<Real>, 2> p2pw({n_pw, n_order}, &window_fourier_data.poly2pw[0]);
        const ndview<std::complex<Real>, 2> pw2p({n_pw, n_order}, &window_fourier_data.pw2poly[0]);
        dmk::proxy::proxycharge2pw<Real, DIM>(proxy_view_upward(0), p2pw, pw_out_view(0), workspaces_[0]);
        multiply_kernelFT_cd2p<Real, DIM>(window_fourier_data.radialft, pw_out_view(0));
        proxy_view_downward(0) = 0;
        proxy_down_zeroed[0] = true;
        dmk::planewave_to_proxy_potential<Real, DIM>(pw_out_view(0), pw2p, proxy_view_downward(0), workspaces_[0]);
    }

#pragma omp parallel
    {
        sctl::Vector<Real> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];

#pragma omp for schedule(dynamic)
        for (int i_box = 0; i_box < n_boxes(); ++i_box) {
            if (ifpwexp[i_box] && proxy_coeffs_offsets[i_box] != -1) {
                const int level = node_mid[i_box].Depth();
                auto &dfd = difference_fourier_data[level];
                const ndview<std::complex<Real>, 2> poly2pw_view({n_pw, n_order}, &dfd.poly2pw[0]);

                dmk::proxy::proxycharge2pw<Real, DIM>(proxy_view_upward(i_box), poly2pw_view, pw_out_view(i_box),
                                                      workspace);
                multiply_kernelFT_cd2p<Real, DIM>(dfd.radialft, pw_out_view(i_box));
            } else if (pw_out_offsets[i_box] != -1) {
                pw_out_view(i_box) = 0;
            }
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
    const int n_pw = expansion_constants.n_pw_diff;
    const int n_pw_modes = expansion_constants.n_exp_modes_diff;
    const int n_pw_per_box = n_pw_modes * n_tables;
    const auto &node_lists = this->GetNodeLists();
    const auto &node_attr = this->GetNodeAttr();
    const Real sc = 2.0 / boxsize;
    const int nd = n_tables;
    const bool need_grad_src = params.kernel == DMK_LAPLACE && params.pgh_src >= DMK_POTENTIAL_GRAD;
    const bool need_grad_trg = params.kernel == DMK_LAPLACE && params.pgh_trg >= DMK_POTENTIAL_GRAD;

    unsigned long n_shifts{0};
#pragma omp parallel
    {
        sctl::Vector<Real> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];
        sctl::Vector<std::complex<Real>> pw_in(n_pw_per_box);

        auto pw_in_view = [this, &pw_in, n_pw]() {
            if constexpr (DIM == 2)
                return ndview<std::complex<Real>, DIM + 1>({n_pw, (n_pw + 1) / 2, n_tables}, &pw_in[0]);
            else if constexpr (DIM == 3)
                return ndview<std::complex<Real>, DIM + 1>({n_pw, n_pw, (n_pw + 1) / 2, n_tables}, &pw_in[0]);
        }();

#pragma omp for schedule(dynamic) reduction(+ : n_shifts)
        for (auto box : boxes) {
            const int nboxpts = src_counts_owned[box] + trg_counts_owned[box];

            if (ifpwexp[box] && nboxpts) {
                memcpy(&pw_in[0], pw_out_ptr(box), n_pw_per_box * sizeof(std::complex<Real>));
                for (auto &neighbor : node_lists[box].nbr) {
                    if (neighbor >= 0 && neighbor != box && (!is_global_leaf[box] || !is_global_leaf[neighbor]) &&
                        (pw_out_offsets[neighbor] != -1)) {
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
                if (!proxy_down_zeroed[box]) {
                    proxy_view_downward(box) = 0;
                    proxy_down_zeroed[box] = true;
                }
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
                        tensorprod::transform<Real, DIM>(nd, proxy_down_zeroed[child], proxy_view_downward(box),
                                                         p2c_view, proxy_view_downward(child), workspace);
                        proxy_down_zeroed[child] = true;
                    }
                }
            }

            if (iftensprodeval[box]) {
                if (src_counts_owned[box]) {
                    if (params.pgh_src == DMK_POTENTIAL)
                        proxy::eval_targets<Real, DIM, 1>(proxy_view_downward(box), r_src_owned_view(box),
                                                          center_view(box), sc, pot_src_view(box), workspace);
                    else if (params.pgh_src == DMK_POTENTIAL_GRAD)
                        proxy::eval_targets<Real, DIM, 2>(proxy_view_downward(box), r_src_owned_view(box),
                                                          center_view(box), sc, pot_src_view(box), workspace);
                }
                if (trg_counts_owned[box]) {
                    if (params.pgh_trg == DMK_POTENTIAL)
                        proxy::eval_targets<Real, DIM, 1>(proxy_view_downward(box), r_trg_owned_view(box),
                                                          center_view(box), sc, pot_trg_view(box), workspace);
                    else if (params.pgh_trg == DMK_POTENTIAL_GRAD)
                        proxy::eval_targets<Real, DIM, 2>(proxy_view_downward(box), r_trg_owned_view(box),
                                                          center_view(box), sc, pot_trg_view(box), workspace);
                }
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

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::evaluate_direct_interactions() {
    sctl::Profile::Scoped profile("evaluate_direct_interactions", &comm_);
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    const auto &node_lists = this->GetNodeLists();

    Real w0[SCTL_MAX_DEPTH];
    for (int i_level = 0; i_level < n_levels(); ++i_level)
        w0[i_level] = get_self_interaction_constant<Real, DIM>(fourier_data, params.kernel, i_level, boxsize[i_level]);

#pragma omp parallel
    {
        // Thread-local buffers for filtered particles
        constexpr int MAX_CHARGE_DIM = 3;
        constexpr int MAX_OUTPUT_DIM = 9;
        constexpr int MAX_PTS = 1000;
        util::StackOrHeapBuffer<Real, DIM * MAX_PTS> r_buf(DIM * params.n_per_leaf);
        util::StackOrHeapBuffer<Real, MAX_CHARGE_DIM * MAX_PTS> charge_buf(kernel_input_dim * params.n_per_leaf);
        util::StackOrHeapBuffer<Real, DIM * MAX_PTS> r_trg_buf(DIM * params.n_per_leaf);
        util::StackOrHeapBuffer<Real, MAX_OUTPUT_DIM * MAX_PTS> pot_buf(kernel_output_dim_max * params.n_per_leaf);
        util::StackOrHeapBuffer<int, MAX_PTS> index_map(params.n_per_leaf);

#pragma omp for schedule(dynamic)
        for (int idx = 0; idx < direct_work.size(); ++idx) {
            const int trg_box = direct_work[idx];
            const int trg_level = node_mid[trg_box].Depth();

            for (auto src_box : list1(trg_box)) {
                int src_level = node_mid[src_box].Depth();
                Real bsize = boxsize[src_level];

                if (ifpwexp[src_box] && src_box == trg_box) {
                    bsize /= Real{2.0};
                    src_level = src_level + 1;
                } else if (src_level < trg_level) {
                    bsize = boxsize[trg_level];
                    src_level = trg_level;
                }

                const Real d2max = bsize * bsize;
                const Real bsizeinv = Real{1} / bsize;

                Real rsc = 2 * bsizeinv;
                Real cen = -bsize / Real{2};
                const auto &cheb_coeffs = fourier_data.cheb_coeffs(src_level);

                if ((params.kernel == DMK_SQRT_LAPLACE && DIM == 3) || (params.kernel == DMK_LAPLACE && DIM == 2)) {
                    rsc = 2 * bsizeinv * bsizeinv;
                    cen = Real{-1.0};
                } else if (params.kernel == DMK_YUKAWA)
                    cen = Real{-1.0};

                // Determine if we should filter, and on which side
                const bool src_larger = node_mid[src_box].Depth() < node_mid[trg_box].Depth();
                const bool trg_larger = node_mid[src_box].Depth() > node_mid[trg_box].Depth();
                const bool should_filter = src_larger || trg_larger;

                // Precompute contact geometry once per box pair (only if asymmetric)
                auto corner_a = node_mid[src_box].template Coord<Real>();
                auto corner_b = node_mid[trg_box].template Coord<Real>();
                auto size_a = boxsize[node_mid[src_box].Depth()];
                auto size_b = boxsize[node_mid[trg_box].Depth()];

                // Resolve source data: either filtered or original
                int n_src = src_counts_with_halo[src_box];
                const Real *r_src_ptr = r_src_with_halo_ptr(src_box);
                const Real *charge_ptr = charge_with_halo_ptr(src_box);

                // Remove points outside sqrt(d2max) range from shared box boundary
                if (src_larger) {
                    ContactGeometry<Real, DIM> geom(corner_a.data(), corner_b.data(), size_a, size_b, d2max);
                    n_src = filter_sources(geom, n_src, r_src_ptr, charge_ptr, kernel_input_dim, r_buf.data(),
                                           charge_buf.data());
                    r_src_ptr = r_buf.data();
                    charge_ptr = charge_buf.data();
                }
                if (!n_src)
                    continue;

                // Evaluate potential at owned source points in the target box
                if (src_counts_owned[trg_box]) {
                    int n_eval_trg = src_counts_owned[trg_box];
                    Real *eval_r_trg = r_src_owned_ptr(trg_box);
                    Real *eval_pot = pot_src_ptr(trg_box);

                    // Remove target points outside sqrt(d2max) range from shared box boundary
                    if (trg_larger) {
                        ContactGeometry<Real, DIM> geom(corner_a.data(), corner_b.data(), size_a, size_b, d2max);
                        n_eval_trg = filter_targets(geom, n_eval_trg, eval_r_trg, r_trg_buf.data(), index_map.data());
                        std::memset(pot_buf.data(), 0, n_eval_trg * kernel_output_dim_src * sizeof(Real));
                        eval_r_trg = r_trg_buf.data();
                        eval_pot = pot_buf.data();
                    }

                    if (n_eval_trg > 0) {
                        if (evaluator_by_level_src[src_level])
                            evaluator_by_level_src[src_level](rsc, cen, d2max, 1e-30, n_src, r_src_ptr, charge_ptr,
                                                              n_eval_trg, eval_r_trg, eval_pot);
                        if (trg_larger)
                            scatter_add_potential(pot_buf.data(), pot_src_ptr(trg_box), index_map.data(), n_eval_trg,
                                                  kernel_output_dim_src);
                    }
                }

                // Evaluate potential at owned target points in the target box
                if (trg_counts_owned[trg_box]) {
                    int n_eval_trg = trg_counts_owned[trg_box];
                    Real *eval_r_trg = r_trg_owned_ptr(trg_box);
                    Real *eval_pot = pot_trg_ptr(trg_box);

                    // Remove target points outside sqrt(d2max) range from shared box boundary
                    if (trg_larger) {
                        ContactGeometry<Real, DIM> geom(corner_a.data(), corner_b.data(), size_a, size_b, d2max);
                        n_eval_trg = filter_targets(geom, n_eval_trg, eval_r_trg, r_trg_buf.data(), index_map.data());
                        std::memset(pot_buf.data(), 0, n_eval_trg * kernel_output_dim_trg * sizeof(Real));
                        eval_r_trg = r_trg_buf.data();
                        eval_pot = pot_buf.data();
                    }

                    if (n_eval_trg > 0) {
                        evaluator_by_level_trg[src_level](rsc, cen, d2max, 1e-30, n_src, r_src_ptr, charge_ptr,
                                                          n_eval_trg, eval_r_trg, eval_pot);

                        if (trg_larger)
                            scatter_add_potential(pot_buf.data(), pot_trg_ptr(trg_box), index_map.data(), n_eval_trg,
                                                  kernel_output_dim_trg);
                    }
                }
            }

            if (!src_counts_owned[trg_box])
                continue;

            // Correct for self-evaluations
            auto pot = pot_src_view(trg_box);
            auto charge = charge_with_halo_view(trg_box);
            const auto depth = node_mid[trg_box].Depth() + ifpwexp[trg_box];
            const auto correction_factor = w0[depth];
            // FIXME: How do we properly deal with gradients/etc?
            for (int i_src = 0; i_src < r_src_cnt_with_halo[trg_box]; ++i_src)
                for (int i = 0; i < kernel_input_dim; ++i)
                    pot(i, i_src) -= correction_factor * charge(i, i_src);
        }
    }
}

/// @brief Perform the "downward pass"
///
/// Updates: proxy_coeffs_downward, tree 'pdmk_pot' particle data
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::downward_pass() {
    sctl::Profile::Scoped prof("downward_pass", &comm_);
    sctl::Profile::Tic("downward_pass_init", &comm_);
    logger->info("downward pass started");

    pot_src_sorted.SetZero();
    pot_trg_sorted.SetZero();

    init_planewave_data();
    sctl::Profile::Toc();

    sctl::Profile::Tic("expansion_propagation_and_eval", &comm_);
    std::fill(proxy_down_zeroed.begin(), proxy_down_zeroed.end(), 0);
    form_outgoing_expansions();

    const int n_pw = expansion_constants.n_pw_diff;
    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        auto &dfd = difference_fourier_data[i_level];
        const ndview<std::complex<Real>, 2> p2pw({n_pw, n_order}, &dfd.poly2pw[0]);
        const ndview<std::complex<Real>, 2> pw2p({n_pw, n_order}, &dfd.pw2poly[0]);

        if (!debug_omit_pw)
            form_eval_expansions(level_indices[i_level], dfd.wpwshift, boxsize[i_level], pw2p, p2c);
    }
    sctl::Profile::Toc();

    if (!debug_omit_direct)
        evaluate_direct_interactions();

    logger->info("downward pass completed");
    if (debug_dump_tree)
        dump();

#ifdef DMK_INSTRUMENT
    sctl::Profile::Tic("downward_pass_barrier", &comm_);
    comm_.Barrier();
    sctl::Profile::Toc();
#endif
}

/// @brief Evaluate using the standard OpenMP pathway
///
/// Updates: Basically everything
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::eval() {
    sctl::Profile::Scoped prof("pdmk_tree_eval", &comm_);
    logger->info("eval() started");
    upward_pass();
    downward_pass();
    logger->info("eval() completed");
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::desort_potentials(Real *pot_src, Real *pot_trg) {
    logger->info("De-sorting potentials into user arrays");
    sctl::Profile::Tic("pdmk_tree_eval_sync", &comm_);
    sctl::Vector<Real> res;
    this->GetParticleData(res, "pdmk_pot_src");
    sctl::Vector<Real>(res.Dim(), pot_src, false) = res;
    this->GetParticleData(res, "pdmk_pot_trg");
    sctl::Vector<Real>(res.Dim(), pot_trg, false) = res;
    sctl::Profile::Toc();
    logger->info("De-sort complete");
}

#ifdef DMK_HAVE_MPI
MPI_TEST_CASE("[DMK] 3D: Proxy charges on upward pass, 2 ranks", 2) {
    constexpr int n_dim = 3;
    constexpr int n_src = 10000;
    constexpr int n_trg = n_src;
    constexpr int n_charge_dim = 1;
    constexpr bool uniform = false;

    sctl::Vector<double> r_src, r_trg, r_src_norms, charges, dipoles, pot_src, pot_trg;
    if (test_rank == 0)
        dmk::util::init_test_data(n_dim, n_charge_dim, n_src, n_trg, uniform, true, r_src, r_trg, r_src_norms, charges,
                                  dipoles, 0);

    pdmk_params params;
    params.eps = 1E-6;
    params.kernel = DMK_LAPLACE;
    params.log_level = SPDLOG_LEVEL_OFF;
    params.fparam = 6.0;
    params.n_dim = n_dim;
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
                continue;
            }

            double maxerr_up = 0.0;
            if (ibox == 0 || tree.ifpwexp[ibox]) {
                // FIXME: ifpwexp is no longer a sufficient check for valid proxy coeffs, since
                // we only care about halos, not the global ifpwexp flag.
                if (tree_single.proxy_coeffs_offsets[single_box] == -1 || tree.proxy_coeffs_offsets[ibox] == -1)
                    break;

                double max_actual = 0.0;
                for (int i = 0; i < sctl::pow<3>(tree.expansion_constants.n_order); ++i) {
                    const double actual = tree_single.proxy_ptr_upward(single_box)[i];
                    const double mpi = tree.proxy_ptr_upward(ibox)[i];
                    const double err = std::abs(mpi - actual);
                    maxerr_up = std::max(err, maxerr_up);
                    max_actual = std::max(std::abs(actual), max_actual);
                }
                maxerr_up = max_actual ? maxerr_up / max_actual : 0.0;
            }

            if (maxerr_up > 1E-12)
                std::cout << fmt::format("{:3} {:3} {:3} {:5.4E} {:3} ({} {} {})\n", test_rank, ibox, single_box,
                                         maxerr_up, node_mid[ibox].Depth(), x[0], x[1], x[2]);
        }

        if (test_rank == 0) {
            for (int i = 0; i < n_src; ++i) {
                double err_src = pot_src[i] == 0. ? 0.0 : std::abs(1.0 - pot_src[i] / pot_src_single[i]);
                double err_trg = pot_trg[i] == 0. ? 0.0 : std::abs(1.0 - pot_trg[i] / pot_trg_single[i]);
                if (err_src > params.eps || err_trg > params.eps)
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
