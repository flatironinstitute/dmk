#ifndef DMK_CUDA_EVAL_TARGETS_KERNELS_HPP
#define DMK_CUDA_EVAL_TARGETS_KERNELS_HPP

// Per-box proxy::eval_targets kernel. launch_eval_targets dispatches on
// (EVAL_LEVEL, N_CHARGE_DIM); DIM=3 is the only path reachable today since
// the other GPU contexts gate on it.

#include <dmk/cuda/eval_targets_kernelargs.hpp>

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_eval_targets(int eval_level, int n_charge_dim, const EvalTargetsArgs<Real> &args, cudaStream_t stream);

template <typename Real>
void launch_self_correction(const SelfCorrectionArgs<Real> &args, cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_EVAL_TARGETS_KERNELS_HPP
