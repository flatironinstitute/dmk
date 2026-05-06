#ifndef DMK_CUDA_TENSORPROD_KERNELS_HPP
#define DMK_CUDA_TENSORPROD_KERNELS_HPP

// Host-includable header for the GPU tensorprod (proxy_downward
// parent → child propagation). One block per (parent, child) pair.
//
// Output is *additive* into the child slot of d_proxy_downward. The shared
// state allocates d_proxy_downward zero-initialized, so no add_flag tracking
// is needed — every write accumulates.

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real>
struct TensorprodArgs {
    int n_pairs = 0;
    int n_order = 0;
    int n_charge_dim = 0; // = n_tables_down

    // Per-pair (uploaded by the orchestration code per level).
    const int *parents = nullptr;       // [n_pairs] box id of parent
    const int *children = nullptr;      // [n_pairs] box id of child
    const int *child_octants = nullptr; // [n_pairs] 0..(2^DIM - 1)

    // Shared-state device pointers.
    Real *proxy_flat = nullptr;          // d_proxy_coeffs_downward (read+write)
    const long *proxy_offsets = nullptr; // [n_boxes]

    // p2c matrices, layout: [octant][axis][k_out, k_in] in F-major n_order×n_order.
    // Total length = n_octants * DIM * n_order * n_order.
    const Real *p2c_flat = nullptr;

    // Per-block global scratch for the ff/ff2 ping-pong buffers. Block uses
    // scratch + blockIdx.x * scratch_stride; ff occupies the first N3 reals,
    // ff2 the next N3.
    Real *scratch = nullptr;
    long scratch_stride = 0; // reals; = 2 * n_order^3
};

template <typename Real>
void launch_tensorprod_dispatch(int dim, const TensorprodArgs<Real> &args, cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_TENSORPROD_KERNELS_HPP
