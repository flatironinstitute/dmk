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
#include <dmk/cuda/tensorprod_kernelargs.hpp>
namespace dmk::cuda {

template <typename Real, int DIM>
void launch_tensorprod(const TensorprodArgs<Real> &args, cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_TENSORPROD_KERNELS_HPP
