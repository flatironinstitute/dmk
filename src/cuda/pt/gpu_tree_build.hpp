#pragma once
// Host-side interface to the device-resident tree build (gpu_tree_build.cu). Kept free of DMK
// headers and of <sctl.hpp>: nvcc crashes in cudafe++ on the SCTL monolith, and every dmk header
// pulls it in (see src/cuda/esp/sort.cu). The sctl headers named below are the narrow ones the GPU
// tree itself uses, which nvcc does parse -- keep the include list this short.

#include <string>

#include <sctl/common.hpp>
#include <sctl/morton.hpp>
#include <sctl/tree.hpp>
#include <sctl/vector.hpp>

namespace dmk::cuda::pt {

/// Owner of the device tree and of every buffer the pointers handed out below alias. Defined in
/// gpu_tree_build.cu, so this header never sees a thrust or gpu_tree type.
template <typename Real, int DIM>
struct GpuTree;

/// Host mirror of the device tree's topology, in the sctl::Tree layout DMK's metadata routines
/// already expect: array-of-struct node lists, with p2n filled from the Morton code.
template <int DIM>
struct GpuTreeTopology {
    sctl::Vector<sctl::Morton<DIM>> node_mid;
    sctl::Vector<typename sctl::Tree<DIM>::NodeAttr> node_attr;
    sctl::Vector<typename sctl::Tree<DIM>::NodeLists> node_lists;
    sctl::Long owned_begin = 0;
    sctl::Long owned_end = 0;
};

/// Particles in tree order. The coordinates stay on the device; only the per-box counts, which the
/// metadata routines walk, come back to the host.
template <typename Real>
struct GpuTreeParticles {
    const Real *d_r_src = nullptr;
    const Real *d_r_trg = nullptr;
    sctl::Vector<sctl::Long> src_cnt;
    sctl::Vector<sctl::Long> trg_cnt;
    sctl::Long n_src = 0;
    sctl::Long n_trg = 0;
};

/// Build the tree on the device over `r_src` (host), sort `r_trg` into it, and fill `topology` and
/// `particles`. Single rank: the GPU path is single-rank by contract (see src/dmk.cpp).
template <typename Real, int DIM>
GpuTree<Real, DIM> *gpu_tree_create(const Real *r_src, sctl::Long n_src, const Real *r_trg, sctl::Long n_trg,
                                    sctl::Long n_per_leaf, bool periodic, GpuTreeTopology<DIM> &topology,
                                    GpuTreeParticles<Real> &particles);

template <typename Real, int DIM>
void gpu_tree_destroy(GpuTree<Real, DIM> *tree);

/// Attach `dof` host values per particle of `group` under `name`, scattered into tree order on the
/// device, and return the device pointer to them. Replaces DMK's own forward-scatter kernel.
template <typename Real, int DIM>
Real *gpu_tree_add_data(GpuTree<Real, DIM> *tree, const std::string &name, const std::string &group,
                        const Real *host_values, int dof);

/// Reserve `dof` unwritten values per particle of `group` in tree order, for a device pass to fill
/// in place, and return the device pointer to them.
template <typename Real, int DIM>
Real *gpu_tree_reserve_data(GpuTree<Real, DIM> *tree, const std::string &name, const std::string &group, int dof);

/// Device pointer to an existing data set, in tree order.
template <typename Real, int DIM>
Real *gpu_tree_data(GpuTree<Real, DIM> *tree, const std::string &name);

/// Drop a data set if it is attached; a no-op otherwise. gpu_tree_add_data and
/// gpu_tree_reserve_data already replace an existing set, so this is rarely needed.
template <typename Real, int DIM>
void gpu_tree_delete_data(GpuTree<Real, DIM> *tree, const std::string &name);

/// Gather `name` back into the caller's particle ordering on the device, then copy it to `host_out`
/// (`dof` values per particle of the set's group). Replaces DMK's own descatter.
template <typename Real, int DIM>
void gpu_tree_get_data(GpuTree<Real, DIM> *tree, const std::string &name, Real *host_out);

/// Per-box metadata derived on the device from the tree's own node lists. Every array stays
/// resident and is owned by the tree; only `n_direct_work` comes back, because launch sizes
/// depend on it.
template <typename Real>
struct GpuTreeMetadata {
    const Real *d_centers = nullptr;              ///< [n_boxes*DIM] box centers
    const int *d_box_levels = nullptr;            ///< [n_boxes] depth per box
    const unsigned char *d_ifpwexp = nullptr;     ///< [n_boxes] has-PW-expansion flag
    const int *d_list1_flat = nullptr;            ///< [n_boxes*nlist1_stride] near source boxes
    const int *d_list1_count = nullptr;           ///< [n_boxes] valid entries per row
    const signed char *d_list1_shift = nullptr;   ///< [n_boxes*nlist1_stride*DIM], null if aperiodic
    const int *d_direct_work = nullptr;           ///< [n_direct_work] target boxes, heaviest first
    const Real *d_self_correction_work = nullptr; ///< [n_direct_work] per-box self-correction factor
    int n_direct_work = 0;
    /// Shallowest box in d_direct_work, or n_levels if it is empty. Yukawa fits one residual
    /// polynomial per level starting here, which is a host decision.
    int min_direct_level = 0;
};

/// Which self-correction constant a box gets. Mirrors the kernel switch in
/// DMKPtTree::build_self_correction_work_list, which is the only place the choice is made.
enum class SelfCorrectionMode { w0, zero, w0_grad };

/// The inputs the device cannot derive: per-level geometry and the accuracy-dependent constants,
/// all O(n_levels), plus the per-box particle counts the tree already handed back.
template <typename Real>
struct GpuTreeMetadataParams {
    const Real *boxsize = nullptr; ///< [n_levels] edge length per level
    const Real *w0 = nullptr;      ///< [n_w0] self-interaction constant per level
    const Real *w0_grad = nullptr; ///< [n_w0]
    const int *src_cnt = nullptr;  ///< [n_boxes]; owned == with-halo at one rank
    const int *trg_cnt = nullptr;  ///< [n_boxes]
    int n_levels = 0;
    int n_boxsize = 0; ///< entries in `boxsize`; must cover every live depth
    int n_w0 = 0;
    int nlist1_stride = 0;
    bool periodic = false;
    SelfCorrectionMode self_mode = SelfCorrectionMode::w0;
};

/// Fill `out` from the device tree. Reproduces DMKPtTree's compute_box_centers,
/// compute_proxy_expansion_flags, build_direct_interaction_lists, build_direct_work_lists and
/// build_self_correction_work_list, in that dependency order.
template <typename Real, int DIM>
void gpu_tree_metadata(GpuTree<Real, DIM> *tree, const GpuTreeMetadataParams<Real> &params, GpuTreeMetadata<Real> &out);

/// Wait for the device work `gpu_tree_metadata` enqueued and fill in the two fields that have to
/// reach the host: `n_direct_work` and `min_direct_level`. Everything else in `out` is a device
/// pointer and is valid as soon as `gpu_tree_metadata` returns. Call this before reading either
/// field -- the point of the split is that host work runs in between.
template <typename Real, int DIM>
void gpu_tree_metadata_finish(GpuTree<Real, DIM> *tree, GpuTreeMetadata<Real> &out);

} // namespace dmk::cuda::pt
