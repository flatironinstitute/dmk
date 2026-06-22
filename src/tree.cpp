#include <algorithm>
#include <cassert>
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
#include <filesystem>
#include <fstream>
#include <omp.h>
#include <sctl/profile.hpp>
#include <unistd.h>

#include <dmk/nvtx_wrapper.h>
#include <dmk/omp_wrapper.hpp>

namespace dmk {

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::dump(const std::string &prefix) const {
    rank_logger->info("Dumping DMKPtTree data on rank {} of comm size {} (prefix='{}')", comm_.Rank(), comm_.Size(),
                      prefix);

    if (!prefix.empty())
        std::filesystem::create_directories(prefix);

    auto dumper = [this, &prefix](const std::string &name, const auto &data) {
        std::string filename =
            prefix + name + "." + std::to_string(comm_.Size()) + "." + std::to_string(comm_.Rank()) + ".dat";
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
        std::string params_filename = prefix + "dmk_params." + std::to_string(comm_.Size()) + ".dat";
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

template <int DIM>
inline int get_table_count_up(dmk_ikernel kernel) {
    if (kernel == DMK_STOKESLET)
        return DIM;
    if (kernel == DMK_STRESSLET)
        return DIM * DIM;
    else
        return 1;
}

template <int DIM>
inline int get_table_count_down(dmk_ikernel kernel) {
    if (kernel == DMK_STOKESLET)
        return DIM;
    if (kernel == DMK_STRESSLET)
        return DIM;
    else
        return 1;
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::build_tree(const sctl::Vector<Real> &r_src, const sctl::Vector<Real> &charge,
                                      const sctl::Vector<Real> &normal, const sctl::Vector<Real> &r_trg) {
    sctl::Profile::Scoped profile("build_tree", &comm_);
    logger->info("base tree build started");
    const int n_src = r_src.Dim() / DIM;
    const int n_trg = r_trg.Dim() / DIM;

    // 0: Initialization
    logger->debug("Building tree and sorting points");

    // Use "2-1" balancing for the tree, i.e. touching boxes never more than one level away in depth
    constexpr bool balance21 = true;
    // Only grab nearest neighbors as 'ghosts' <-> halo = 0
    constexpr int halo = 0;

    // All data that needs to be tree sorted
    sctl::Profile::Tic("add_particles", &comm_);
    this->AddParticles("pdmk_src", r_src);
    this->AddParticles("pdmk_trg", r_trg);

    if (params.kernel == DMK_STRESSLET) {
        this->AddParticleData("pdmk_normal", "pdmk_src", normal);
        this->AddParticleData("pdmk_density", "pdmk_src", charge);

        // Stresslet has (force .outer. normal) proxy charges
        sctl::Vector<Real> charge_normal(n_src * DIM * DIM);
        Real *__restrict__ charge_normal_ptr = &charge_normal[0];
        const Real *__restrict__ charge_ptr = &charge[0];
        const Real *__restrict__ normal_ptr = &normal[0];
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_src; ++i)
            for (int k = 0; k < DIM; ++k)
                for (int j = 0; j < DIM; ++j)
                    charge_normal_ptr[i * DIM * DIM + k * DIM + j] = charge_ptr[i * DIM + k] * normal_ptr[i * DIM + j];
        this->AddParticleData("pdmk_charge", "pdmk_src", charge_normal);
    } else {
        this->AddParticleData("pdmk_charge", "pdmk_src", charge);
    }
    this->AddParticleData("pdmk_pot_src", "pdmk_src", kernel_output_dim_src);
    this->AddParticleData("pdmk_pot_trg", "pdmk_trg", kernel_output_dim_trg);
    sctl::Profile::Toc();

    sctl::Profile::Tic("update_refinement", &comm_);
    this->UpdateRefinement(r_src, params.n_per_leaf, balance21, params.use_periodic, halo);
    sctl::Profile::Toc();

    sctl::Profile::Tic("get_non_halo", &comm_);
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
    sctl::Profile::Toc();

    sctl::Profile::Tic("broadcast_get_halo", &comm_);
    // Now grab sorted particle data with the halo, so we have it for direct evaluations
    this->template Broadcast<Real>("pdmk_src");
    this->template Broadcast<Real>("pdmk_charge");
    if (params.kernel == DMK_STRESSLET) {
        this->template Broadcast<Real>("pdmk_normal");
        this->template Broadcast<Real>("pdmk_density");
        this->GetData(normal_sorted_with_halo, normal_cnt_with_halo, "pdmk_normal");
        this->GetData(density_sorted_with_halo, density_cnt_with_halo, "pdmk_density");
    }
    this->GetData(charge_sorted_with_halo, charge_cnt_with_halo, "pdmk_charge");
    this->GetData(r_src_sorted_with_halo, r_src_cnt_with_halo, "pdmk_src");
    sctl::Profile::Toc();

    logger->debug("base tree build completed");
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::build_tree_for_gpu(const sctl::Vector<Real> &r_src, const sctl::Vector<Real> &r_trg) {
    sctl::Profile::Scoped profile("build_tree_for_gpu", &comm_);
    logger->info("gpu tree build started");

    constexpr bool balance21 = true;
    constexpr int halo = 0;

    sctl::Profile::Tic("add_particles", &comm_);
    this->AddParticles("pdmk_src", r_src);
    this->AddParticles("pdmk_trg", r_trg);
    sctl::Profile::Toc();

    sctl::Profile::Tic("update_refinement", &comm_);
    this->UpdateRefinement(r_src, params.n_per_leaf, balance21, params.use_periodic, halo);
    sctl::Profile::Toc();

    sctl::Profile::Tic("get_non_halo", &comm_);
    this->GetData(r_src_sorted_owned, r_src_cnt_owned, "pdmk_src");
    this->GetData(r_trg_sorted_owned, r_trg_cnt_owned, "pdmk_trg");
    sctl::Profile::Toc();

    logger->debug("gpu tree build completed");
}

template <typename Real, int DIM>
DMKPtTree<Real, DIM>::DMKPtTree(const sctl::Comm &comm, const pdmk_params &params_, const sctl::Vector<Real> &r_src,
                                const sctl::Vector<Real> &charge, const sctl::Vector<Real> &normal,
                                const sctl::Vector<Real> &r_trg)
    : sctl::PtTree<Real, DIM>(comm), comm_(comm), params(params_),
      kernel_input_dim(get_kernel_input_dim(params.n_dim, params.kernel)),
      kernel_output_dim_src(get_kernel_output_dim(params.n_dim, params.kernel, params.eval_src)),
      kernel_output_dim_trg(get_kernel_output_dim(params.n_dim, params.kernel, params.eval_trg)),
      kernel_output_dim_max(std::max(kernel_output_dim_src, kernel_output_dim_trg)),
      n_tables_up(get_table_count_up<DIM>(params.kernel)), n_tables_down(get_table_count_down<DIM>(params.kernel)),
      n_digits(std::round(log10(1.0 / params_.eps) - 0.1)), expansion_constants(params),
      logger(dmk::get_logger(comm, params.log_level)), rank_logger(dmk::get_rank_logger(comm, params.log_level)) {
    debug_omit_pw = (params.debug_flags & DMK_DEBUG_OMIT_PW) || util::env_is_set("DMK_DEBUG_OMIT_PW");
    debug_omit_direct = (params.debug_flags & DMK_DEBUG_OMIT_DIRECT) || util::env_is_set("DMK_DEBUG_OMIT_DIRECT");
    debug_dump_tree = (params.debug_flags & DMK_DEBUG_DUMP_TREE) || util::env_is_set("DMK_DEBUG_DUMP_TREE");
    debug_force_aot = (params.debug_flags & DMK_DEBUG_FORCE_AOT) || util::env_is_set("DMK_DEBUG_FORCE_AOT");
    logger->info("tree build started");
    if (debug_omit_pw)
        logger->debug("Ignoring PW interactions");
    if (debug_omit_direct)
        logger->debug("Ignoring direct interactions");
    if (params.eval_path == DMK_EVAL_PATH_GPU) {
#ifdef DMK_GPU_OFFLOAD
        build_tree_for_gpu(r_src, r_trg);
        generate_metadata_for_gpu();
        gpu_init_state();
        if (cuda_shared_state_) {
            const long n_src = r_src.Dim() / DIM;
            const Real *normal_ptr = (params.kernel == DMK_STRESSLET && normal.Dim()) ? &normal[0] : nullptr;
            const Real *charge_ptr = charge.Dim() ? &charge[0] : nullptr;
            cuda_shared_state_->upload_and_sort_charges(params.kernel, charge_ptr, normal_ptr, n_src);
        }
#else
        throw std::runtime_error("DMK was built without DMK_GPU_OFFLOAD; only DMK_EVAL_PATH_CPU is available");
#endif
    } else {
        build_tree(r_src, charge, normal, r_trg);
        generate_metadata();
    }
    logger->info("tree build completed");
}

template <typename Real, int DIM>
int DMKPtTree<Real, DIM>::update_charges(const Real *charge, const Real *normal) {
    auto &logger = dmk::get_logger(comm_, params.log_level);
    logger->info("update_charges started");

    const int n_src = r_src_sorted_owned.Dim() / DIM;

#ifdef DMK_GPU_OFFLOAD
    if (params.eval_path == DMK_EVAL_PATH_GPU && cuda_shared_state_) {
        cuda_shared_state_->upload_and_sort_charges(params.kernel, charge, normal, n_src);
        logger->info("update_charges completed (gpu)");
        return 0;
    }
#endif

    // Delete the old data and re-register with the new values. The PtTree
    // already knows the sort permutation from the "pdmk_src" particle set,
    // so AddParticleData will sort the new data into tree order automatically.
    if (params.kernel == DMK_STRESSLET) {
        if (!normal) {
            std::cerr << "stresslet update_charges requires non-null normal\n";
            return 1;
        }

        sctl::Vector<Real> normal_vec(n_src * DIM, const_cast<Real *>(normal), false);
        sctl::Vector<Real> density_vec(n_src * DIM, const_cast<Real *>(charge), false);

        sctl::Vector<Real> charge_normal(n_src * DIM * DIM);
        Real *__restrict__ charge_normal_ptr = &charge_normal[0];
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_src; ++i)
            for (int k = 0; k < DIM; ++k)
                for (int j = 0; j < DIM; ++j)
                    charge_normal_ptr[i * DIM * DIM + k * DIM + j] = charge[i * DIM + k] * normal[i * DIM + j];

        this->DeleteParticleData("pdmk_normal");
        this->DeleteParticleData("pdmk_density");
        this->DeleteParticleData("pdmk_charge");
        this->AddParticleData("pdmk_normal", "pdmk_src", normal_vec);
        this->AddParticleData("pdmk_density", "pdmk_src", density_vec);
        this->AddParticleData("pdmk_charge", "pdmk_src", charge_normal);
    } else {
        sctl::Vector<Real> charge_vec(n_src * n_tables_up, const_cast<Real *>(charge), false);
        this->DeleteParticleData("pdmk_charge");
        this->AddParticleData("pdmk_charge", "pdmk_src", charge_vec);
    }

    // Retrieve the sorted owned charges
    {
        sctl::Vector<Real> data;
        sctl::Vector<long> count;
        this->GetData(data, count, "pdmk_charge");
        charge_sorted_owned = data;
        charge_cnt_owned = count;
    }

    // Broadcast to halo/ghost nodes and retrieve
    this->template Broadcast<Real>("pdmk_charge");
    this->GetData(charge_sorted_with_halo, charge_cnt_with_halo, "pdmk_charge");
    if (params.kernel == DMK_STRESSLET) {
        this->template Broadcast<Real>("pdmk_normal");
        this->template Broadcast<Real>("pdmk_density");
        this->GetData(normal_sorted_with_halo, normal_cnt_with_halo, "pdmk_normal");
        this->GetData(density_sorted_with_halo, density_cnt_with_halo, "pdmk_density");
    }

    logger->info("update_charges completed");
    return 0;
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::compute_data_offsets() {
    sctl::Profile::Scoped profile("compute_data_offsets", &comm_);
    const auto &node_mid = this->GetNodeMID();
    r_src_offsets_with_halo.ReInit(n_boxes());
    r_src_offsets_owned.ReInit(n_boxes());
    r_trg_offsets_owned.ReInit(n_boxes());
    pot_src_offsets.ReInit(n_boxes());
    pot_trg_offsets.ReInit(n_boxes());
    charge_offsets_owned.ReInit(n_boxes());
    charge_offsets_with_halo.ReInit(n_boxes());
    normal_offsets_with_halo.ReInit(n_boxes());
    density_offsets_with_halo.ReInit(n_boxes());

    r_src_offsets_with_halo[0] = r_src_offsets_owned[0] = r_trg_offsets_owned[0] = pot_src_offsets[0] =
        pot_trg_offsets[0] = charge_offsets_owned[0] = charge_offsets_with_halo[0] = normal_offsets_with_halo[0] =
            density_offsets_with_halo[0] = 0;

    for (int i = 1; i < n_boxes(); ++i) {
        r_src_offsets_owned[i] = r_src_offsets_owned[i - 1] + DIM * r_src_cnt_owned[i - 1];
        r_src_offsets_with_halo[i] = r_src_offsets_with_halo[i - 1] + DIM * r_src_cnt_with_halo[i - 1];
        r_trg_offsets_owned[i] = r_trg_offsets_owned[i - 1] + DIM * r_trg_cnt_owned[i - 1];
        pot_src_offsets[i] = pot_src_offsets[i - 1] + kernel_output_dim_src * pot_src_cnt[i - 1];
        pot_trg_offsets[i] = pot_trg_offsets[i - 1] + kernel_output_dim_trg * pot_trg_cnt[i - 1];
        charge_offsets_owned[i] = charge_offsets_owned[i - 1] + n_tables_up * charge_cnt_owned[i - 1];
        charge_offsets_with_halo[i] = charge_offsets_with_halo[i - 1] + n_tables_up * charge_cnt_with_halo[i - 1];
    }

    if (params.kernel == DMK_STRESSLET) {
        for (int i = 1; i < n_boxes(); ++i) {
            normal_offsets_with_halo[i] = normal_offsets_with_halo[i - 1] + DIM * normal_cnt_with_halo[i - 1];
            density_offsets_with_halo[i] = density_offsets_with_halo[i - 1] + DIM * density_cnt_with_halo[i - 1];
        }
    }
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::compute_level_indices_and_boxsizes() {
    sctl::Profile::Scoped profile("compute_level_indices_and_boxsizes", &comm_);
    const auto &node_mid = this->GetNodeMID();
    level_indices.ReInit(SCTL_MAX_DEPTH);
    uint8_t max_depth = 0;
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
    sctl::Profile::Scoped profile("compute_box_centers", &comm_);

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
    sctl::Profile::Scoped profile("accumulate_subtree_counts", &comm_);
    const auto &node_mid = this->GetNodeMID();
    const auto &node_lists = this->GetNodeLists();
    src_counts_with_halo.ReInit(n_boxes());
    src_counts_with_halo.SetZero();
    src_counts_owned.ReInit(n_boxes());
    src_counts_owned.SetZero();
    trg_counts_owned.ReInit(n_boxes());
    trg_counts_owned.SetZero();

    n_trg_max_ = 0;
    for (int i_level = n_levels() - 1; i_level >= 0; --i_level) {
        for (auto i_node : level_indices[i_level]) {
            src_counts_with_halo[i_node] += r_src_cnt_with_halo[i_node];
            src_counts_owned[i_node] += r_src_cnt_owned[i_node];
            trg_counts_owned[i_node] += r_trg_cnt_owned[i_node];
            n_trg_max_ = std::max(r_trg_cnt_owned[i_node], n_trg_max_);

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
    sctl::Profile::Scoped profile("gather_owned_source_positions", &comm_);
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
    sctl::Profile::Scoped profile("broadcast_global_leaf_status", &comm_);
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
    sctl::Profile::Scoped profile("compute_proxy_expansion_flags", &comm_);
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
    sctl::Profile::Scoped profile("compute_proxy_evaluation_flags", &comm_);
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
    sctl::Profile::Scoped profile("build_plane_wave_interaction_lists", &comm_);
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

// Compute the periodic shift for a neighbor at nbr slot index k.
// The slot k = d0 + 3*d1 + 9*d2 (for DIM=3) where d ∈ {0,1,2} maps to offset {-1,0,+1}.
// The expected neighbor center is center[box] + offset * boxsize.
// The shift is the difference between expected and actual positions, rounded to int.
template <typename Real, int DIM>
static std::array<int, DIM> compute_periodic_shift_from_slot(int k, Real bsize, const Real *center_box,
                                                             const Real *center_nbr) {
    std::array<int, DIM> shift{};
    for (int d = 0; d < DIM; ++d) {
        const int dir = (k % 3) - 1; // -1, 0, or +1
        k /= 3;
        const Real expected = center_box[d] + dir * bsize;
        shift[d] = std::round(expected - center_nbr[d]);
    }
    return shift;
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::build_direct_interaction_lists() {
    sctl::Profile::Scoped profile("build_direct_interaction_lists", &comm_);
    const auto &node_mid = this->GetNodeMID();
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_lists = this->GetNodeLists();
    list1_.resize(n_boxes());
    nlist1_.assign(n_boxes(), 0);
    list1_shift_.resize(n_boxes());

    auto get_shift = [&](int nbr_k, Real bsize, int ref_box, int nbr_box) -> std::array<int, DIM> {
        if (!params.use_periodic)
            return {};
        return compute_periodic_shift_from_slot<Real, DIM>(nbr_k, bsize, center_ptr(ref_box), center_ptr(nbr_box));
    };

    auto within_cutoff = [&](int trg, int src, const std::array<int, DIM> &shift, double cutoff) {
        for (int d = 0; d < DIM; ++d)
            if (std::abs(center_ptr(trg)[d] - (center_ptr(src)[d] + shift[d])) > cutoff)
                return false;
        return true;
    };

#pragma omp parallel for schedule(guided, 4)
    for (int box = 0; box < n_boxes(); ++box) {
        if (!is_global_leaf[box] || node_attr[box].Ghost)
            continue;

        const int i_level = node_mid[box].Depth();
        const Real bsize = boxsize[i_level];
        const double cutoff_child = 1.05 * 0.75 * bsize;
        const double cutoff_parent_nbr = 1.5 * 1.05 * bsize;

        auto add = [&](int neighb_box, const std::array<int, DIM> &shift) {
            int &k = nlist1_[box];
            list1_[box][k] = neighb_box;
            list1_shift_[box][k] = shift;
            ++k;
        };

        // Same-level neighbors: leaf neighbors added directly; for non-leaf neighbors,
        // add their children that are within touching range of box.
        for (int nbr_k = 0; nbr_k < NCOLLEAGUE; ++nbr_k) {
            const int neighb = node_lists[box].nbr[nbr_k];
            if (neighb < 0)
                continue;
            const auto shift = get_shift(nbr_k, bsize, box, neighb);
            if (is_global_leaf[neighb]) {
                if (src_counts_with_halo[neighb])
                    add(neighb, shift);
            } else {
                for (int child : node_lists[neighb].child) {
                    if (child >= 0 && src_counts_with_halo[child] && within_cutoff(box, child, shift, cutoff_child))
                        add(child, shift);
                }
            }
        }

        // Parent's neighbors: coarser leaf boxes that cannot appear as same-level neighbors.
        if (i_level == 0)
            continue;
        const int parent = node_lists[box].parent;
        for (int nbr_k = 0; nbr_k < NCOLLEAGUE; ++nbr_k) {
            const int neighb = node_lists[parent].nbr[nbr_k];
            if (neighb < 0 || !is_global_leaf[neighb] || !src_counts_with_halo[neighb])
                continue;
            const auto shift = get_shift(nbr_k, boxsize[i_level - 1], parent, neighb);
            if (within_cutoff(box, neighb, shift, cutoff_parent_nbr))
                add(neighb, shift);
        }
    }
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::build_upward_pass_work_lists() {
    sctl::Profile::Scoped profile("build_upward_pass_work_lists", &comm_);
    const auto &node_lists = this->GetNodeLists();
    has_proxy_from_children.ReInit(n_boxes());
    charge2proxy_groups.clear();

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

            Charge2ProxyGroup g{.center_box = i_box, .level = i_level, .n_src_boxes = 0};
            if (has_proxy_from_children[i_box]) {
                for (auto cb : node_lists[i_box].child) {
                    if (cb >= 0 && src_counts_owned[cb] > 0 && !ifpwexp[cb])
                        g.src_boxes[g.n_src_boxes++] = cb;
                }
                if (g.n_src_boxes == 0)
                    continue;
            } else {
                g.src_boxes[g.n_src_boxes++] = i_box;
            }

            charge2proxy_groups.push_back(g);
        }
    }
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::allocate_proxy_coefficients() {
    sctl::Profile::Scoped profile("allocate_proxy_coefficients", &comm_);
    const int n_coeffs_up = n_tables_up * sctl::pow<DIM>(expansion_constants.n_order);
    const int n_coeffs_down = n_tables_down * sctl::pow<DIM>(expansion_constants.n_order);

    sctl::Vector<sctl::Long> counts_upward(n_boxes());
    sctl::Vector<sctl::Long> counts_downward(n_boxes());
    long n_proxy_boxes_upward = 0;
    long n_proxy_boxes_downward = 0;
    for (int i = 0; i < n_boxes(); ++i) {
        if (ifpwexp[i] && src_counts_with_halo[i] > 0) {
            counts_upward[i] = n_coeffs_up;
            n_proxy_boxes_upward++;
        } else {
            counts_upward[i] = 0;
        }

        if ((ifpwexp[i] || iftensprodeval[i]) && (src_counts_owned[i] + trg_counts_owned[i])) {
            counts_downward[i] = n_coeffs_down;
            n_proxy_boxes_downward++;
        } else {
            counts_downward[i] = 0;
        }
    }

    proxy_coeffs_downward.ReInit(n_coeffs_down * n_proxy_boxes_downward);

    this->template AddData<Real>("proxy_coeffs", n_coeffs_up * n_proxy_boxes_upward, counts_upward);
    this->GetData(proxy_coeffs_upward, counts_upward, "proxy_coeffs");

    proxy_coeffs_offsets.ReInit(n_boxes());
    proxy_coeffs_offsets_downward.ReInit(n_boxes());

    long last_offset = 0;
    for (int box = 0; box < n_boxes(); ++box) {
        if (counts_upward[box]) {
            proxy_coeffs_offsets[box] = last_offset;
            last_offset += n_coeffs_up;
        } else {
            proxy_coeffs_offsets[box] = -1;
        }
    }

    last_offset = 0;
    for (int box = 0; box < n_boxes(); ++box) {
        if (counts_downward[box]) {
            proxy_coeffs_offsets_downward[box] = last_offset;
            last_offset += n_coeffs_down;
        } else {
            proxy_coeffs_offsets_downward[box] = -1;
        }
    }
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::precompute_window_difference_data() {
    sctl::Profile::Scoped profile("precompute_window_difference_data", &comm_);
    sctl::Vector<Real> kernel_ft;

    if (params.use_periodic) {
        const int n_pw_periodic = expansion_constants.n_pw_periodic;
        // Periodic grid: dk = 2*pi/L, n_pw_periodic modes per dimension
        const Real dk = 2.0 * M_PI / boxsize[0];
        // Multi-level trees use the first child scale for the root smooth kernel.
        // For a single-level tree there is no level-1 box, so fall back to the root scale.
        const int sigma_level = std::min(1, n_levels() - 1);
        const Real sigma1 = boxsize[sigma_level] / fourier_data.beta();
        const Real psi0_at_zero = fourier_data.prolate0_fun.eval_val(0.0);
        const long n_pw_modes_periodic = sctl::pow<DIM - 1>(n_pw_periodic) * ((n_pw_periodic + 1) / 2);
        const int n_fourier = DIM * sctl::pow<2>(n_pw_periodic / 2) + 1;

        // Build periodic radialft: PSWF kernel (4pi/psi0(0)) * psi0(kappa*sigma1) / kappa^2
        kernel_ft.ReInit(n_fourier);
        kernel_ft[0] = 0; // k=0 excluded (charge neutrality)
        for (int i = 1; i < n_fourier; ++i) {
            const Real kappa = std::sqrt(Real(i)) * dk;
            const Real arg = kappa * sigma1;
            const Real psi_val = (std::abs(arg) <= 1.0) ? fourier_data.prolate0_fun.eval_val(arg) : Real(0);
            kernel_ft[i] = (4.0 * M_PI / psi0_at_zero) * psi_val / (Real(i) * dk * dk);
        }

        window_fourier_data.radialft.ReInit(n_pw_modes_periodic);
        util::mk_tensor_product_fourier_transform(
            DIM, n_pw_periodic, ndview<Real, 1>({n_fourier}, &kernel_ft[0]),
            ndview<Real, 1>({n_pw_modes_periodic}, &window_fourier_data.radialft[0]));

        window_fourier_data.poly2pw.ReInit(n_order * n_pw_periodic);
        window_fourier_data.pw2poly.ReInit(n_order * n_pw_periodic);
        dmk::calc_planewave_coeff_matrices(boxsize[0], dk, n_pw_periodic, n_order, window_fourier_data.poly2pw,
                                           window_fourier_data.pw2poly);
    } else {
        const int n_pw = expansion_constants.n_pw_win;
        const long n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);

        window_fourier_data.poly2pw.ReInit(n_order * n_pw);
        window_fourier_data.pw2poly.ReInit(n_order * n_pw);
        window_fourier_data.radialft.ReInit(n_pw_modes);
        get_windowed_kernel_ft<Real, DIM>(params.kernel, &params.fparam, fourier_data.beta(), n_pw, boxsize[0],
                                          fourier_data.prolate0_fun, kernel_ft);
        util::mk_tensor_product_fourier_transform(DIM, n_pw, ndview<Real, 1>({kernel_ft.Dim()}, &kernel_ft[0]),
                                                  ndview<Real, 1>({n_pw_modes}, &window_fourier_data.radialft[0]));
        fourier_data.calc_planewave_coeff_matrices(-1, n_order, n_pw, window_fourier_data.poly2pw,
                                                   window_fourier_data.pw2poly);
    }

    // Difference constants/transformation matrices
    {
        difference_fourier_data.resize(n_levels());
        const int n_pw = expansion_constants.n_pw_diff;
        const long n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
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
            fourier_data.calc_planewave_coeff_matrices(i_level, n_order, n_pw, lfd.poly2pw, lfd.pw2poly);
            dmk::calc_planewave_translation_matrix<DIM>(1, boxsize[i_level], n_pw,
                                                        fourier_data.difference_kernel(i_level).hpw, lfd.wpwshift);
        }
    }
}

/// @brief Build list of target boxes. Boxes are sorted in descending
/// order by total number of pairs interacting
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::build_direct_work_lists() {
    sctl::Profile::Scoped profile("build_direct_work_lists", &comm_);
    const auto &node_attr = this->GetNodeAttr();

    direct_work.clear();
    direct_work.reserve(n_boxes());
    for (int i_box = 0; i_box < n_boxes(); ++i_box) {
        if (is_global_leaf[i_box] && !node_attr[i_box].Ghost && nlist1_[i_box] > 0)
            direct_work.push_back(i_box);
    }

    std::vector<std::pair<long, int>> est_work(direct_work.size());
    for (int i = 0; i < direct_work.size(); ++i) {
        const int box = direct_work[i];
        long src = 0;
        for (auto j : list1(box))
            src += src_counts_with_halo[j];
        est_work[i] = {(src_counts_owned[box] + trg_counts_owned[box]) * src, box};
    }
    std::sort(est_work.begin(), est_work.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    for (int i = 0; i < direct_work.size(); ++i)
        direct_work[i] = est_work[i].second;
}

/// @brief Build list of direct evaluators for each level, source and target.
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::build_evaluators() {
    sctl::Profile::Scoped profile("build_evaluators", &comm_);

    const int n_lvl = n_levels();
    direct_rsc.ReInit(n_lvl);
    direct_cen.ReInit(n_lvl);
    direct_d2max.ReInit(n_lvl);
    for (int lvl = 0; lvl < n_lvl; ++lvl) {
        const Real bsize = boxsize[lvl];
        const Real bsizeinv = Real{1} / bsize;
        Real rsc = Real{2} * bsizeinv;
        Real cen = -bsize / Real{2};
        if ((params.kernel == DMK_SQRT_LAPLACE && DIM == 3) || (params.kernel == DMK_LAPLACE && DIM == 2)) {
            rsc = Real{2} * bsizeinv * bsizeinv;
            cen = Real{-1.0};
        } else if (params.kernel == DMK_YUKAWA) {
            cen = Real{-1.0};
        }
        direct_rsc[lvl] = rsc;
        direct_cen[lvl] = cen;
        direct_d2max[lvl] = bsize * bsize;
    }

    eval_targets_box_list.clear();
    eval_targets_box_list.reserve(n_boxes());
    for (int b = 0; b < n_boxes(); ++b)
        if (iftensprodeval[b])
            eval_targets_box_list.push_back(b);

    // Per-level tensorprod pairs. Mirrors form_eval_expansions's CPU loop
    // gating: parent must do PW work (ifpwexp && nboxpts) and not be an
    // iftensprodeval leaf; child must be a real, non-empty box.
    {
        constexpr int n_children = 1u << DIM;
        const auto &node_mid_local = this->GetNodeMID();
        const auto &node_lists_local = this->GetNodeLists();
        tensorprod_pairs_per_level.assign(n_levels(), {});
        for (int b = 0; b < n_boxes(); ++b) {
            const int nboxpts = src_counts_owned[b] + trg_counts_owned[b];
            if (!ifpwexp[b] || !nboxpts || iftensprodeval[b])
                continue;
            const int level = node_mid_local[b].Depth();
            for (int i_child = 0; i_child < n_children; ++i_child) {
                const int child = node_lists_local[b].child[i_child];
                if (child < 0)
                    continue;
                if (!(src_counts_owned[child] + trg_counts_owned[child]))
                    continue;
                tensorprod_pairs_per_level[level].push_back({b, child, i_child});
            }
        }
    }

    // CPU evaluator lambdas — direct.cpp consumes these. The GPU direct path
    // has its own kernels and ignores evaluator_by_level_*.
    if (params.eval_path == DMK_EVAL_PATH_GPU)
        return;

    try {
        auto src_eval = make_evaluator_aot<Real>(params.kernel, params.eval_src, DIM, n_digits, 3);
        auto trg_eval = make_evaluator_aot<Real>(params.kernel, params.eval_trg, DIM, n_digits, 3);
#ifdef DMK_USE_JIT
        if (!util::env_is_set("DMK_DEBUG_FORCE_AOT")) {
            src_eval =
                make_evaluator_jit<Real>(params.kernel, params.eval_src, DIM, n_digits, expansion_constants.beta, 3);
            trg_eval =
                make_evaluator_jit<Real>(params.kernel, params.eval_trg, DIM, n_digits, expansion_constants.beta, 3);
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
            const int n_charge_dim = n_tables_up;
            const int kernel_output_dim = kernel_output_dim_trg;
            // FIXME: assumes the same src/trg output configuration
            evaluator_by_level_src.push_back([coeffs, lambda, n_charge_dim, kernel_output_dim](
                                                 Real rsc, Real cen, Real d2max, Real thresh2, int n_src,
                                                 const Real *r_src_ptr, const Real *charge_ptr, const Real *normal_ptr,
                                                 int n_trg, const Real *r_trg_ptr, Real *pot) {
                constexpr Real threshq = 1e-30;
                ndview<Real, 2> u({1, n_trg}, pot);
                ndview<const Real, 2> charges({n_charge_dim, n_src}, charge_ptr);
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
                        for (int i = 0; i < kernel_output_dim; ++i)
                            u(i, i_trg) += charges(i, i_src) * factor;
                    }
                }
            });
        }
        // FIXME: assumes the same src/trg output configuration
        evaluator_by_level_trg = evaluator_by_level_src;
    }
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::build_self_correction_work_list() {
    const auto &node_mid = this->GetNodeMID();
    const int n_lvl = n_levels() + 1;
    std::vector<Real> w0(n_lvl);
    for (int i = 0; i < n_lvl; ++i)
        w0[i] = get_self_interaction_constant<Real, DIM>(fourier_data, params.kernel, i, boxsize[i]);

    self_correction_work.resize(direct_work.size());
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < direct_work.size(); ++idx) {
        const int box = direct_work[idx];
        if (params.kernel == DMK_STRESSLET) {
            self_correction_work[idx] = Real{0};
            continue;
        }
        const int depth = node_mid[box].Depth();
        if (params.kernel == DMK_STOKESLET)
            self_correction_work[idx] = ifpwexp[box] ? 2 * w0[depth] : w0[depth];
        else
            self_correction_work[idx] = w0[depth + ifpwexp[box]];
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
    fourier_data = FourierData<Real>(params.kernel, DIM, params.eps, expansion_constants.n_pw_win,
                                     expansion_constants.n_pw_diff, params.fparam, expansion_constants.beta, boxsize);
    precompute_window_difference_data();
    build_evaluators();
    build_self_correction_work_list();

    logger->debug("done generating tree traversal metadata and other constants");
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::generate_metadata_for_gpu() {
    sctl::Profile::Scoped profile("generate_metadata_for_gpu", &comm_);
    logger->debug("generating GPU tree traversal metadata");
    assert(
        charge_sorted_owned.Dim() == 0 && pot_src_sorted.Dim() == 0 &&
        "generate_metadata_for_gpu expects build_tree_for_gpu (positions only) — host charge/pot arrays must be empty");

    // The CPU build registers charge/normal/density/pot particle data with
    // PtTree, which populates these per-box count vectors via GetData. The
    // GPU build skips those registrations (the data lives on the device), so
    // mirror the per-box source/target counts here. Single-rank: no halo
    // exchange, so "with_halo" counts equal "owned".
    r_src_cnt_with_halo = r_src_cnt_owned;
    charge_cnt_owned = r_src_cnt_owned;
    charge_cnt_with_halo = r_src_cnt_owned;
    pot_src_cnt = r_src_cnt_owned;
    pot_trg_cnt = r_trg_cnt_owned;
    if (params.kernel == DMK_STRESSLET) {
        normal_cnt_with_halo = r_src_cnt_owned;
        density_cnt_with_halo = r_src_cnt_owned;
    }

    compute_data_offsets();
    compute_level_indices_and_boxsizes();
    compute_box_centers();
    accumulate_subtree_counts();
    broadcast_global_leaf_status();
    compute_proxy_expansion_flags();
    compute_proxy_evaluation_flags();
    build_direct_interaction_lists();
    build_upward_pass_work_lists();
    build_direct_work_lists();
    allocate_proxy_coefficients();
    std::tie(c2p, p2c) = dmk::chebyshev::get_c2p_p2c_matrices<Real>(DIM, expansion_constants.n_order);
    fourier_data = FourierData<Real>(params.kernel, DIM, params.eps, expansion_constants.n_pw_win,
                                     expansion_constants.n_pw_diff, params.fparam, expansion_constants.beta, boxsize);
    precompute_window_difference_data();
    build_evaluators();
    build_self_correction_work_list();

    logger->debug("done generating GPU tree traversal metadata");
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
    nvtxRangePush("upward_pass");
    logger->info("upward pass started");

#ifdef DMK_GPU_OFFLOAD
    if (cuda_upward_ctx_)
        gpu_upward_pass();
    const bool run_cpu = params.eval_path == DMK_EVAL_PATH_CPU;
#else
    if (params.eval_path != DMK_EVAL_PATH_CPU)
        throw std::runtime_error("DMK was built without DMK_GPU_OFFLOAD; only DMK_EVAL_PATH_CPU is available");
    constexpr bool run_cpu = true;
#endif

    if (run_cpu)
        cpu_upward_pass();

    logger->info("upward pass finished");
    nvtxRangePop();
}

#ifdef DMK_GPU_OFFLOAD
template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::gpu_init_state() {
    nvtxRangePush("device_reset");
    cuda_eval_targets_ctx_.reset();
    cuda_direct_ctx_.reset();
    cuda_downward_ctx_.reset();
    cuda_form_outgoing_ctx_.reset();
    cuda_upward_ctx_.reset();
    cuda_shared_state_.reset();
    nvtxRangePop();

    const bool want_gpu = params.eval_path == DMK_EVAL_PATH_GPU;
    if (!want_gpu)
        return;

    nvtxRangePush("device_init");
    cuda_shared_state_ = std::make_unique<CudaSharedDeviceState<Real, DIM>>(*this);
    cuda_direct_ctx_ = std::make_unique<CudaDirectContext<Real, DIM>>(*this, *cuda_shared_state_);
    cuda_eval_targets_ctx_ = std::make_unique<CudaEvalTargetsContext<Real, DIM>>(*this, *cuda_shared_state_);
    cuda_downward_ctx_ = std::make_unique<CudaDownwardContext<Real, DIM>>(*this, *cuda_shared_state_);
    cuda_form_outgoing_ctx_ = std::make_unique<CudaFormOutgoingContext<Real, DIM>>(*this, *cuda_shared_state_);
    cuda_upward_ctx_ = std::make_unique<CudaUpwardContext<Real, DIM>>(*this, *cuda_shared_state_);
    nvtxRangePop();
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::gpu_upward_pass() {
    // charge2proxy + per-level tensorprod on shared.downward_stream; downward
    // kernels chain naturally on the same stream, so no extra sync needed.
    sctl::Profile::Scoped p("cuda_upward", &comm_);
    nvtxRangePush("cuda_upward");
    nvtxRangePush("residual_direct");
    cuda_direct_ctx_->launch();
    nvtxRangePop();
    cuda_upward_ctx_->run();
    nvtxRangePop();
}
#endif

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::cpu_upward_pass() {
    sctl::Profile::Tic("upward_pass_init", &comm_);
    const std::size_t n_coeffs = n_tables_up * sctl::pow<DIM>(expansion_constants.n_order);
#pragma omp parallel
#pragma omp single
    workspaces_.ReInit(MY_OMP_GET_NUM_THREADS());

    sctl::Vector<sctl::Long> counts;
    this->GetData(proxy_coeffs_upward, counts, "proxy_coeffs");
    proxy_coeffs_upward = 0;

    constexpr int n_children = 1u << DIM;
    const auto &node_lists = this->GetNodeLists();
    sctl::Profile::Toc();

    {
        sctl::Profile::Scoped profile("charge2proxy", &comm_);
        sctl::Profile::Tic("charge2proxy", &comm_);

#pragma omp parallel
        {
            sctl::Vector<Real> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];

#pragma omp for schedule(dynamic)
            for (int gi = 0; gi < (int)charge2proxy_groups.size(); ++gi) {
                const auto &g = charge2proxy_groups[gi];
                const Real scale = 2.0 / boxsize[g.level];
                for (int k = 0; k < g.n_src_boxes; ++k) {
                    const int sb = g.src_boxes[k];
                    proxy::charge2proxycharge<Real, DIM>(r_src_owned_view(sb), charge_owned_view(sb),
                                                         center_view(g.center_box), scale,
                                                         proxy_view_upward(g.center_box), workspace);
                }
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
                        tensorprod::transform<Real, DIM>(n_tables_up, true, proxy_view_upward(cb), c2p_view,
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

    logger->debug("Finished building proxy charges");

    sctl::Profile::Tic("broadcast_proxy_coeffs", &comm_);
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
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::init_planewave_data() {
    // Only care about diff, windowed is in a temp data structure
    const int n_pw = expansion_constants.n_pw_diff;
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * n_tables_down;

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
        pw_out.ReInit(last_offset);
    }
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::form_outgoing_expansions() {
#ifdef DMK_INSTRUMENT
    double dt = -MY_OMP_GET_WTIME();
#endif
    const int n_pw_win = expansion_constants.n_pw_win;
    const int n_pw_diff = expansion_constants.n_pw_diff;
    const int n_pw_modes_win = sctl::pow<DIM - 1>(n_pw_win) * ((n_pw_win + 1) / 2);
    const int n_pw_modes_diff = sctl::pow<DIM - 1>(n_pw_diff) * ((n_pw_diff + 1) / 2);
    const int n_pw_per_box_win = n_pw_modes_win * n_tables_down;
    const int n_pw_per_box_diff = n_pw_modes_diff * n_tables_down;
    const auto &node_mid = this->GetNodeMID();

    auto pw_view = [](int n_pw, int n_tables, auto &pw_vec) {
        if constexpr (DIM == 2)
            return ndview<std::complex<Real>, DIM + 1>({n_pw, (n_pw + 1) / 2, n_tables}, pw_vec.data());
        else if constexpr (DIM == 3)
            return ndview<std::complex<Real>, DIM + 1>({n_pw, n_pw, (n_pw + 1) / 2, n_tables}, pw_vec.data());
    };

    // Root box: uses windowed kernel fourier data
    { // Windowed kernel for root (or periodic PSWF kernel for PBC)
        proxy_view_downward(0) = 0;
        proxy_down_zeroed[0] = true;

        if (params.use_periodic) {
            // FIXME: When adding periodic for Stresslet, this will crash due to up/down asymmetry
            const int n_pw_root = expansion_constants.n_pw_periodic;
            const long n_pw_modes_root = sctl::pow<DIM - 1>(n_pw_root) * ((n_pw_root + 1) / 2);
            const int n_pw_per_box_root = n_pw_modes_root * n_tables_down;

            std::vector<std::complex<Real>> pw_root(n_pw_per_box_root);
            auto pw_root_view = pw_view(n_pw_root, n_tables_up, pw_root);

            const ndview<std::complex<Real>, 2> p2pw({n_pw_root, n_order}, &window_fourier_data.poly2pw[0]);
            const ndview<std::complex<Real>, 2> pw2p({n_pw_root, n_order}, &window_fourier_data.pw2poly[0]);

            dmk::proxy::proxycharge2pw<Real, DIM>(proxy_view_upward(0), p2pw, pw_root_view, workspaces_[0]);
            multiply_kernelFT_cd2p<Real, DIM>(window_fourier_data.radialft, pw_root_view);

            pw_out_view(0) = 0;
            proxy_view_downward(0) = 0;
            proxy_down_zeroed[0] = true;
            dmk::planewave_to_proxy_potential<Real, DIM>(pw_root_view, pw2p, proxy_view_downward(0), workspaces_[0]);
        } else {
            std::vector<std::complex<Real>> pw_out_win(n_pw_per_box_win);
            const ndview<std::complex<Real>, 2> p2pw({n_pw_win, n_order}, &window_fourier_data.poly2pw[0]);
            const ndview<std::complex<Real>, 2> pw2p({n_pw_win, n_order}, &window_fourier_data.pw2poly[0]);

            auto pw_out_win_view_eval = pw_view(n_pw_win, n_tables_down, pw_out_win);

            if (params.kernel == DMK_STRESSLET) {
                std::vector<std::complex<Real>> pw_in_win(n_pw_modes_win * n_tables_up);
                auto pw_in_win_view = pw_view(n_pw_win, n_tables_up, pw_in_win);
                dmk::proxy::proxycharge2pw<Real, DIM>(proxy_view_upward(0), p2pw, pw_in_win_view, workspaces_[0]);
                stresslet_multiply_kernelFT<Real, DIM>(window_fourier_data.radialft, pw_in_win_view,
                                                       pw_out_win_view_eval, expansion_constants.hpw_win);
            } else {
                auto pw_out_win_view_form = pw_view(n_pw_win, n_tables_up, pw_out_win);
                dmk::proxy::proxycharge2pw<Real, DIM>(proxy_view_upward(0), p2pw, pw_out_win_view_form, workspaces_[0]);
                if (params.kernel == DMK_STOKESLET)
                    stokeslet_multiply_kernelFT<Real, DIM, true>(window_fourier_data.radialft, pw_out_win_view_form,
                                                                 expansion_constants.hpw_win);
                else
                    multiply_kernelFT_cd2p<Real, DIM>(window_fourier_data.radialft, pw_out_win_view_form);
            }
            dmk::planewave_to_proxy_potential<Real, DIM>(pw_out_win_view_eval, pw2p, proxy_view_downward(0),
                                                         workspaces_[0]);
        }
    }

#pragma omp parallel
    {
        sctl::Vector<Real> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];
        const int n_pw_per_box_form = n_pw_modes_diff * n_tables_up;
        const int n_pw_per_box_eval = n_pw_modes_diff * n_tables_down;
        std::vector<std::complex<Real>> pw_form(n_pw_per_box_form);
        auto pw_form_view = pw_view(n_pw_diff, n_tables_up, pw_form);

        // When use_periodic, skip form_outgoing at level 0: the periodic root kernel
        // already includes W_0+D_0, so D_0 must not be computed again. pw_out(0) stays
        // zero from init_planewave_data, so form_eval's self-interaction adds nothing.
        // We still run form_eval at level 0 to propagate proxy_downward(0) to children.
        const int start_box = params.use_periodic ? 1 : 0;
#pragma omp for schedule(dynamic)
        for (int i_box = start_box; i_box < n_boxes(); ++i_box) {
            if (ifpwexp[i_box] && proxy_coeffs_offsets[i_box] != -1) {
                const int level = node_mid[i_box].Depth();
                auto &dfd = difference_fourier_data[level];
                const ndview<std::complex<Real>, 2> poly2pw_view({n_pw_diff, n_order}, &dfd.poly2pw[0]);

                dmk::proxy::proxycharge2pw<Real, DIM>(proxy_view_upward(i_box), poly2pw_view, pw_form_view, workspace);
                if (params.kernel == DMK_STRESSLET) {
                    stresslet_multiply_kernelFT<Real, DIM>(difference_fourier_data[level].radialft, pw_form_view,
                                                           pw_out_view(i_box),
                                                           expansion_constants.hpw_diff / boxsize[level]);
                } else {
                    if (params.kernel == DMK_STOKESLET)
                        stokeslet_multiply_kernelFT<Real, DIM, false>(difference_fourier_data[level].radialft,
                                                                      pw_form_view,
                                                                      expansion_constants.hpw_diff / boxsize[level]);
                    else
                        multiply_kernelFT_cd2p<Real, DIM>(dfd.radialft, pw_form_view);
                    std::copy(pw_form.data(), pw_form.data() + n_pw_per_box_eval, pw_out_ptr(i_box));
                }
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
    const int n_pw_diff = expansion_constants.n_pw_diff;
    const int n_pw_modes = expansion_constants.n_exp_modes_diff;
    const int n_pw_per_box = n_pw_modes * n_tables_down;
    const auto &node_lists = this->GetNodeLists();
    const auto &node_attr = this->GetNodeAttr();
    const Real sc = 2.0 / boxsize;
    const bool need_grad_src = params.kernel == DMK_LAPLACE && params.eval_src >= DMK_POTENTIAL_GRAD;
    const bool need_grad_trg = params.kernel == DMK_LAPLACE && params.eval_trg >= DMK_POTENTIAL_GRAD;

    unsigned long n_shifts{0};
#pragma omp parallel
    {
        sctl::Vector<Real> &workspace = workspaces_[MY_OMP_GET_THREAD_NUM()];
        sctl::Vector<std::complex<Real>> pw_in(n_pw_per_box);

        auto pw_in_view = [this, &pw_in, n_pw_diff]() {
            if constexpr (DIM == 2)
                return ndview<std::complex<Real>, DIM + 1>({n_pw_diff, (n_pw_diff + 1) / 2, n_tables_down}, &pw_in[0]);
            else if constexpr (DIM == 3)
                return ndview<std::complex<Real>, DIM + 1>({n_pw_diff, n_pw_diff, (n_pw_diff + 1) / 2, n_tables_down},
                                                           &pw_in[0]);
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

                        ndview<const std::complex<Real>, 1> wpwshift_view({n_pw_modes}, &wpwshift[n_pw_modes * ind]);
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
                        tensorprod::transform<Real, DIM>(n_tables_down, proxy_down_zeroed[child],
                                                         proxy_view_downward(box), p2c_view, proxy_view_downward(child),
                                                         workspace);
                        proxy_down_zeroed[child] = true;
                    }
                }
            }

            // form_eval_expansions only runs on the CPU path; the GPU path
            // uses CudaEvalTargetsContext after the per-level kernels complete.
            if (iftensprodeval[box]) {
                if (src_counts_owned[box]) {
                    if (need_grad_src)
                        proxy::eval_targets<Real, DIM, 2>(proxy_view_downward(box), r_src_owned_view(box),
                                                          center_view(box), sc, pot_src_view(box), workspace);
                    else
                        proxy::eval_targets<Real, DIM, 1>(proxy_view_downward(box), r_src_owned_view(box),
                                                          center_view(box), sc, pot_src_view(box), workspace);
                }
                if (trg_counts_owned[box]) {
                    if (need_grad_trg)
                        proxy::eval_targets<Real, DIM, 2>(proxy_view_downward(box), r_trg_owned_view(box),
                                                          center_view(box), sc, pot_trg_view(box), workspace);
                    else
                        proxy::eval_targets<Real, DIM, 1>(proxy_view_downward(box), r_trg_owned_view(box),
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
void DMKPtTree<Real, DIM>::correct_for_self_interactions() {
    sctl::Profile::Scoped profile("correct_for_self");

#pragma omp for schedule(dynamic)
    for (int idx = 0; idx < direct_work.size(); ++idx) {
        const Real correction_factor = self_correction_work[idx];
        if (correction_factor == Real{0})
            continue;
        const int trg_box = direct_work[idx];
        if (!src_counts_owned[trg_box])
            continue;

        // FIXME: This needs to deal with correction factors where
        // kernel_input_dim != kernel_output_dim (like grad, which
        // needs only a correction factor on the potential, not
        // the gradient. That's why kernel_input_dim here works)
        auto pot = pot_src_view(trg_box);
        auto charge = charge_with_halo_view(trg_box);
        for (int i_src = 0; i_src < r_src_cnt_with_halo[trg_box]; ++i_src)
            for (int i = 0; i < kernel_input_dim; ++i)
                pot(i, i_src) -= correction_factor * charge(i, i_src);
    }
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::evaluate_direct_interactions() {
    sctl::Profile::Scoped profile("evaluate_direct_interactions", &comm_);
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    const auto &node_lists = this->GetNodeLists();

    // For PBC: precompute the periodic shift for each (trg_box, nbr_index) pair.
    // The nbr array index k encodes a direction (dx,dy,dz) ∈ {-1,0,+1}^DIM.
    // k = d0 + 3*d1 + 9*d2 where d ∈ {0,1,2} maps to offset {-1,0,+1}.
    // When the neighbor wraps around the periodic boundary, the source
    // positions must be shifted by ±1 in the wrapped dimension.
    //
    // We detect the shift by comparing the actual center difference with
    // the expected neighbor direction. For a box at level L with boxsize B:
    //   expected offset in dim i = (d_i - 1) * B
    //   actual offset = center[nbr][i] - center[box][i]
    //   shift[i] = expected - actual = round(expected - actual)
    // This is nonzero only when the neighbor wraps periodically.

#pragma omp parallel
    {
        // Thread-local buffers for filtered particles
        constexpr int MAX_CHARGE_DIM = 3;
        constexpr int MAX_OUTPUT_DIM = 9;
        constexpr int MAX_PTS = 1000;
        const bool is_stresslet = params.kernel == DMK_STRESSLET;
        const int normal_dim = is_stresslet ? DIM : 0;
        const int direct_charge_dim = kernel_input_dim;
        const long trg_buff_cnt = std::max(long(params.n_per_leaf), n_trg_max_);

        util::StackOrHeapBuffer<Real, DIM * MAX_PTS> r_buf(DIM * params.n_per_leaf);
        util::StackOrHeapBuffer<Real, MAX_CHARGE_DIM * MAX_PTS> charge_buf(direct_charge_dim * params.n_per_leaf);
        util::StackOrHeapBuffer<Real, DIM * MAX_PTS> normal_buf(DIM * params.n_per_leaf);
        util::StackOrHeapBuffer<Real, DIM * MAX_PTS> r_trg_buf(DIM * trg_buff_cnt);
        util::StackOrHeapBuffer<Real, MAX_OUTPUT_DIM * MAX_PTS> pot_buf(kernel_output_dim_max * trg_buff_cnt);
        util::StackOrHeapBuffer<int, MAX_PTS> index_map(trg_buff_cnt);

        // Buffer for periodically shifted source positions
        std::vector<Real> r_src_shifted;

#pragma omp for schedule(dynamic)
        for (int idx = 0; idx < direct_work.size(); ++idx) {
            const int trg_box = direct_work[idx];
            const int trg_level = node_mid[trg_box].Depth();

            for (int list1_idx = 0; list1_idx < nlist1_[trg_box]; ++list1_idx) {
                const int src_box = list1_[trg_box][list1_idx];
                int src_level = node_mid[src_box].Depth();

                if (ifpwexp[src_box] && src_box == trg_box) {
                    src_level = src_level + 1;
                } else if (src_level < trg_level) {
                    src_level = trg_level;
                }

                // evaluator_by_level_src is sized n_levels()
                // This fix is essentially for when there is *only* a root box.
                src_level = std::min(src_level, n_levels() - 1);

                const Real rsc = direct_rsc[src_level];
                const Real cen = direct_cen[src_level];
                const Real d2max = direct_d2max[src_level];
                const auto &cheb_coeffs = fourier_data.cheb_coeffs(src_level);

                // Determine if we should filter, and on which side
                const bool src_larger = node_mid[src_box].Depth() < node_mid[trg_box].Depth();
                const bool trg_larger = node_mid[src_box].Depth() > node_mid[trg_box].Depth();

                // Precompute contact geometry once per box pair (only if asymmetric)
                auto corner_a = node_mid[src_box].template Coord<Real>();
                auto corner_b = node_mid[trg_box].template Coord<Real>();
                auto size_a = boxsize[node_mid[src_box].Depth()];
                auto size_b = boxsize[node_mid[trg_box].Depth()];

                // Resolve source data: either filtered or original
                int n_src = src_counts_with_halo[src_box];
                const Real *r_src_ptr = r_src_with_halo_ptr(src_box);
                const Real *charge_ptr = is_stresslet ? density_with_halo_ptr(src_box) : charge_with_halo_ptr(src_box);
                const Real *normal_ptr = is_stresslet ? normal_with_halo_ptr(src_box) : nullptr;

                // For PBC: apply the periodic image shift to source positions. The shift
                // for each list-1 pair was computed in build_direct_interaction_lists()
                // and stored in list1_shift_; this routine only applies it. The source-
                // box corner used by ContactGeometry must be shifted by the same vector
                // so source particles and source-box stay in one frame — otherwise
                // asymmetric-depth filtering (src_larger/trg_larger) rejects valid
                // periodic-image interactions at boundary-crossing list-1 pairs.
                if (params.use_periodic) {
                    const auto &shift = list1_shift_[trg_box][list1_idx];
                    bool needs_shift = false;
                    for (int d = 0; d < DIM; ++d)
                        if (shift[d] != 0)
                            needs_shift = true;

                    if (needs_shift) {
                        r_src_shifted.resize(DIM * n_src);
                        for (int i = 0; i < n_src; ++i)
                            for (int d = 0; d < DIM; ++d)
                                r_src_shifted[i * DIM + d] = r_src_ptr[i * DIM + d] + shift[d];
                        r_src_ptr = r_src_shifted.data();
                        for (int d = 0; d < DIM; ++d)
                            corner_a[d] += shift[d];
                    }
                }

                // Remove points outside sqrt(d2max) range from shared box boundary
                if (src_larger) {
                    ContactGeometry<Real, DIM> geom(corner_a.data(), corner_b.data(), size_a, size_b, d2max);
                    n_src = filter_sources(geom, n_src, r_src_ptr, charge_ptr, direct_charge_dim, normal_ptr,
                                           normal_dim, r_buf.data(), charge_buf.data(), normal_buf.data());
                    r_src_ptr = r_buf.data();
                    charge_ptr = charge_buf.data();
                    normal_ptr = normal_buf.data();
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
                        evaluator_by_level_src[src_level](rsc, cen, d2max, 1e-30, n_src, r_src_ptr, charge_ptr,
                                                          normal_ptr, n_eval_trg, eval_r_trg, eval_pot);
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
                                                          normal_ptr, n_eval_trg, eval_r_trg, eval_pot);

                        if (trg_larger)
                            scatter_add_potential(pot_buf.data(), pot_trg_ptr(trg_box), index_map.data(), n_eval_trg,
                                                  kernel_output_dim_trg);
                    }
                }
            }
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
    nvtxRangePush("downward_pass");
    sctl::Profile::Tic("downward_pass_init", &comm_);
    logger->info("downward pass started");

    init_planewave_data();
    sctl::Profile::Toc();

    const bool run_cpu = params.eval_path == DMK_EVAL_PATH_CPU;
    const bool run_gpu = params.eval_path == DMK_EVAL_PATH_GPU;

#ifdef DMK_GPU_OFFLOAD
    // Launches GPU kernels async on shared.downward_stream; GPU-only mode syncs at the
    // end of gpu_downward_pass so host pot is populated before we return.
    if (run_gpu) {
        // The GPU descatter in finalize_pot only does the local permutation
        // (no MPI Alltoallv). Multi-rank GPU is not currently supported.
        SCTL_ASSERT(comm_.Size() == 1 && "GPU eval_path requires a single rank");
        gpu_downward_pass();
        sctl::Profile::Scoped p("cuda_merge", &comm_);
        cuda_shared_state_->finalize_pot(cuda_eval_targets_ctx_->stream(), cuda_eval_targets_ctx_->device_pot_src(),
                                         cuda_eval_targets_ctx_->device_pot_trg(), cuda_direct_ctx_->device_pot_src(),
                                         cuda_direct_ctx_->device_pot_trg());
    }
#endif

    if (run_cpu) {
        pot_src_sorted.SetZero();
        pot_trg_sorted.SetZero();

        cpu_downward_pass();
    }

    if (run_cpu)
        correct_for_self_interactions();

#ifdef DMK_GPU_OFFLOAD
    // GPU dump (into "gpu/") must run before teardown so the device buffers
    // are still alive; the CPU-side dump (below) writes to cwd.
    if (debug_dump_tree && cuda_shared_state_)
        cuda_shared_state_->dump(*this);
#endif

    logger->info("downward pass completed");
    if (debug_dump_tree)
        dump();

#ifdef DMK_INSTRUMENT
    sctl::Profile::Tic("downward_pass_barrier", &comm_);
    comm_.Barrier();
    sctl::Profile::Toc();
#endif
    nvtxRangePop();
}

#ifdef DMK_GPU_OFFLOAD
template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::gpu_downward_pass() {
    if (!debug_omit_pw) {
        sctl::Profile::Scoped p("cuda_downward", &comm_);
        cuda_form_outgoing_ctx_->run();
        cuda_downward_ctx_->run();
    }
    {
        sctl::Profile::Scoped p("cuda_eval_targets_launch", &comm_);
        cuda_eval_targets_ctx_->launch();
    }
}
#endif

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::cpu_downward_pass() {
    sctl::Profile::Tic("downward_pass_init", &comm_);

    pot_src_sorted.SetZero();
    pot_trg_sorted.SetZero();

    init_planewave_data();
    sctl::Profile::Toc();

    sctl::Profile::Tic("expansion_propagation_and_eval", &comm_);
    std::fill(proxy_down_zeroed.begin(), proxy_down_zeroed.end(), 0);
    form_outgoing_expansions();
    if (!debug_omit_pw) {
        const int n_pw = expansion_constants.n_pw_diff;
        for (int i_level = 0; i_level < n_levels(); ++i_level) {
            auto &dfd = difference_fourier_data[i_level];
            const ndview<std::complex<Real>, 2> p2pw({n_pw, n_order}, &dfd.poly2pw[0]);
            const ndview<std::complex<Real>, 2> pw2p({n_pw, n_order}, &dfd.pw2poly[0]);
            form_eval_expansions(level_indices[i_level], dfd.wpwshift, boxsize[i_level], pw2p, p2c);
        }
    }
    sctl::Profile::Toc();

    if (!debug_omit_direct)
        evaluate_direct_interactions();
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
    nvtxRangePush("pdmk_tree_eval_sync");

#ifdef DMK_GPU_OFFLOAD
    if (params.eval_path == DMK_EVAL_PATH_GPU && cuda_shared_state_) {
        // finalize_pot wrote the descattered (user-order) result into
        // d_pot_*_final and synchronized its stream; one D2H per side and
        // we're done.
        auto &s = *cuda_shared_state_;
        if (s.pot_src_size)
            DMK_CHECK_CUDA(
                cudaMemcpy(pot_src, s.d_pot_src_final.data(), s.pot_src_size * sizeof(Real), cudaMemcpyDeviceToHost));
        if (s.pot_trg_size)
            DMK_CHECK_CUDA(
                cudaMemcpy(pot_trg, s.d_pot_trg_final.data(), s.pot_trg_size * sizeof(Real), cudaMemcpyDeviceToHost));
        sctl::Profile::Toc();
        nvtxRangePop();
        logger->info("De-sort complete");
        return;
    }
#endif

    sctl::Vector<Real> res_src, res_trg;
    this->GetParticleData(res_src, "pdmk_pot_src");
    sctl::Vector<Real>(res_src.Dim(), pot_src, false) = res_src;
    this->GetParticleData(res_trg, "pdmk_pot_trg");
    sctl::Vector<Real>(res_trg.Dim(), pot_trg, false) = res_trg;
    sctl::Profile::Toc();
    nvtxRangePop();
    logger->info("De-sort complete");
}

#ifdef DMK_HAVE_MPI
MPI_TEST_CASE("[DMK] 3D: Proxy charges on upward pass, 2 ranks", 2) {
    constexpr int n_dim = 3;
    constexpr int n_src = 10000;
    constexpr int n_trg = n_src;
    constexpr int n_charge_dim = 1;
    constexpr bool uniform = false;

    sctl::Vector<double> r_src, r_trg, r_src_norms, charges, normals, pot_src, pot_trg;
    if (test_rank == 0)
        dmk::util::init_test_data(n_dim, n_charge_dim, n_src, n_trg, uniform, true, r_src, r_trg, r_src_norms, charges,
                                  0);

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
    DMKPtTree<double, n_dim> tree(comm, params, r_src, charges, normals, r_trg);
    tree.upward_pass();
    tree.downward_pass();
    tree.GetParticleData(pot_src, "pdmk_pot_src");
    tree.GetParticleData(pot_trg, "pdmk_pot_trg");

    if (test_rank == 0) {
        dmk::util::init_test_data(n_dim, n_charge_dim, n_src, n_trg, uniform, true, r_src, r_trg, r_src_norms, charges,
                                  0);

        DMKPtTree<double, n_dim> tree_single(sctl::Comm::Self(), params, r_src, charges, normals, r_trg);
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
