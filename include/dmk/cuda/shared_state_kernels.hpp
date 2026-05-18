#ifndef DMK_CUDA_SHARED_STATE_KERNELS_HPP
#define DMK_CUDA_SHARED_STATE_KERNELS_HPP

// Kernels that operate on buffers owned by CudaSharedDeviceState. Currently
// just the fused merge of long-range (eval_targets) and short-range (direct)
// potential buffers into a descattered (user-order) output buffer:
//   out[scatter_index[i]*dof + j] = pot_eval[i*dof + j] + pot_extra[i*dof + j]

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real>
void launch_accumulate_and_scatter(Real *out_unsorted, const Real *pot_eval, const Real *pot_extra,
                                   const long *scatter_index, int dof, long n_particles, cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_SHARED_STATE_KERNELS_HPP
