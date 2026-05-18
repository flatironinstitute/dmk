#ifndef DMK_CUDA_TENSORPROD_KERNELS_HPP
#define DMK_CUDA_TENSORPROD_KERNELS_HPP

// Per-pair tensorprod (proxy propagation between adjacent levels). Used by
// both directions:
//   downward: src=parent, dst=child, umat=p2c (each parent contributes to a
//             unique child, so no two pairs share dst — non-atomic write).
//   upward:   src=child,  dst=parent, umat=c2p (multiple children of the
//             same parent contribute concurrently — caller must set
//             additive_atomic=true to serialize writes).
//
// Output is *additive* into dst. Caller is responsible for zero-initializing
// the relevant proxy buffer before the first kernel launch in a sweep.

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real>
struct TensorprodArgs {
    int n_pairs = 0;
    int n_order = 0;
    int n_charge_dim = 0; // n_tables_down (downward) or n_tables_up (upward)

    // Per-pair (uploaded by the orchestration code per level).
    const int *src_boxes = nullptr;     // [n_pairs] box id read from
    const int *dst_boxes = nullptr;     // [n_pairs] box id written to (additive)
    const int *child_octants = nullptr; // [n_pairs] 0..(2^DIM - 1)

    // Shared-state device pointers.
    Real *proxy_flat = nullptr;          // d_proxy_coeffs_(up|down)ward (read+write)
    const long *proxy_offsets = nullptr; // [n_boxes]

    // umat matrices, layout: [octant][axis][k_out, k_in] in F-major n_order×n_order.
    // Total length = n_octants * DIM * n_order * n_order. Pass d_p2c for
    // downward, d_c2p for upward.
    const Real *umat_flat = nullptr;

    // Per-block global scratch for the ff/ff2 ping-pong buffers. Block uses
    // scratch + blockIdx.x * scratch_stride; ff occupies the first N3 reals,
    // ff2 the next N3.
    Real *scratch = nullptr;
    long scratch_stride = 0; // reals; = 2 * n_order^3

    // Set true when multiple pairs at the same launch can target the same
    // dst_box (upward direction). Phase 3 then uses atomicAdd.
    bool additive_atomic = false;
};

template <typename Real, int DIM>
void launch_tensorprod(const TensorprodArgs<Real> &args, cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_TENSORPROD_KERNELS_HPP
