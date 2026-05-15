#ifndef DMK_CUDA_CHARGE2PROXY_KERNELS_HPP
#define DMK_CUDA_CHARGE2PROXY_KERNELS_HPP

// Host-includable header for the GPU charge2proxy kernel. One block per
// Charge2ProxyGroup; the block accumulates contributions from each src_box in
// the group's list into proxy_coeffs_upward[group.center_box]. Output is
// additive — caller must zero d_proxy_coeffs_upward before launch.

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real>
struct Charge2ProxyArgs {
    int n_groups = 0;
    int n_order = 0;
    int n_charge_dim = 0; // = n_tables_up

    // Per-group, length n_groups.
    const int *center_boxes = nullptr;
    const int *levels = nullptr;
    const int *src_box_flat_offsets = nullptr;
    const int *n_src_boxes_per_group = nullptr;
    const int *src_boxes_flat = nullptr; // total length = sum(n_src_boxes_per_group)

    // Shared device state.
    const Real *centers = nullptr;       // [n_boxes * DIM] F-major (axis, box)
    const Real *inv_box_scale = nullptr; // [n_levels]; = 2 / boxsize[L]
    const Real *r_src_owned = nullptr;   // F-major positions per box
    const long *r_src_owned_offsets = nullptr;
    const int *src_counts_owned = nullptr;
    const Real *charge_owned = nullptr; // F-major [n_charge_dim, n_src] per box
    const long *charge_owned_offsets = nullptr;

    Real *proxy_flat = nullptr; // d_proxy_coeffs_upward (additive write)
    const long *proxy_offsets = nullptr;

    // Sort order over groups (largest work first). Length n_groups; the first
    // n_active_groups entries are non-zero-work and are the ones launched.
    const int *group_perm = nullptr;
    int n_active_groups = 0;
};

template <typename Real>
void launch_charge2proxy_dispatch(int dim, const Charge2ProxyArgs<Real> &args, cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_CHARGE2PROXY_KERNELS_HPP
