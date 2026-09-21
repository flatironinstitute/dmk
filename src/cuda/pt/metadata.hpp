#pragma once
// Per-box metadata derived on the device from the tree's own node arrays. Kept free of <sctl.hpp>
// and of every dmk header but device_tree.hpp: metadata.cu is compiled by nvcc, which crashes in
// cudafe++ on the SCTL monolith (see src/cuda/esp/sort.cu). device_tree.hpp is safe -- it pulls
// only the narrow sctl headers the device tree itself uses.

#include <dmk/cuda/device_tree.hpp>

#include <sctl/morton.hpp>

namespace dmk::cuda::pt {

/// Which self-correction constant a box gets. Mirrors the kernel switch in
/// DMKPtTree::build_self_correction_work_list, which is the only place the choice is made.
enum class SelfCorrectionMode { w0, zero, w0_grad };

/// The tree's node arrays, as the device already holds them, plus the per-box host quantities the
/// kernels cannot derive. `is_global_leaf` is deliberately separate from `attr`: leafness as the
/// interaction lists define it is "no rank refined this node", which equals attr.Leaf only at one
/// rank.
template <typename Real, int DIM>
struct DeviceMetadataInputs {
    const sctl::Morton<DIM> *d_mid = nullptr;      ///< [n_boxes] device
    const device_tree::NodeAttr *d_attr = nullptr; ///< [n_boxes] device; only Ghost is read
    const sctl::Long *d_parent = nullptr;          ///< [n_boxes] device
    const sctl::Long *d_child = nullptr;           ///< [n_boxes*2^DIM] device
    const sctl::Long *d_nbr = nullptr;             ///< [n_boxes*3^DIM] device
    const unsigned char *is_global_leaf = nullptr; ///< [n_boxes] host
    const unsigned char *ifpwexp = nullptr;        ///< [n_boxes] host
    const int *src_cnt_with_halo = nullptr;        ///< [n_boxes] host
    const int *src_cnt_owned = nullptr;            ///< [n_boxes] host
    const int *trg_cnt_owned = nullptr;            ///< [n_boxes] host
    const Real *boxsize = nullptr;                 ///< [n_boxsize] host, edge length per level
    const Real *w0 = nullptr;                      ///< [n_w0] host, self-interaction constant per level
    const Real *w0_grad = nullptr;                 ///< [n_w0] host
    long n_boxes = 0;
    int n_boxsize = 0; ///< must cover every live depth
    int n_w0 = 0;
    int n_levels = 0;
    int nlist1_stride = 0;
    bool periodic = false;
    SelfCorrectionMode self_mode = SelfCorrectionMode::w0;
};

/// Device-resident outputs. Every pointer is owned by the caller and must be sized before the call;
/// see device_metadata_sizes. Only the two scalars come back to the host.
template <typename Real>
struct DeviceMetadataOutputs {
    Real *d_centers = nullptr;              ///< [n_boxes*DIM]
    int *d_box_levels = nullptr;            ///< [n_boxes]
    int *d_list1 = nullptr;                 ///< [n_boxes*nlist1_stride]
    int *d_list1_count = nullptr;           ///< [n_boxes]
    signed char *d_list1_shift = nullptr;   ///< [n_boxes*nlist1_stride*DIM], null if aperiodic
    int *d_direct_work = nullptr;           ///< [n_boxes]; first n_direct_work entries are the list
    Real *d_self_correction_work = nullptr; ///< [n_boxes], parallel to d_direct_work
    long *d_cost_scratch = nullptr;         ///< [n_boxes], sort key
    int *d_summary = nullptr;               ///< [2]

    int n_direct_work = 0;
    /// Shallowest box in d_direct_work, or n_levels if it is empty. Yukawa fits one residual
    /// polynomial per level starting here, which is a host decision.
    int min_direct_level = 0;
};

/// Reproduces DMKPtTree::compute_box_centers, build_direct_interaction_lists,
/// build_direct_work_lists and build_self_correction_work_list, in that dependency order, and
/// fills the two scalars in `out`.
template <typename Real, int DIM>
void compute_device_metadata(const DeviceMetadataInputs<Real, DIM> &in, DeviceMetadataOutputs<Real> &out);

} // namespace dmk::cuda::pt
