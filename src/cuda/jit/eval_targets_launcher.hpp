#pragma once

#include "jit_cache.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/eval_targets_kernelargs.hpp>

#include <cuda_runtime.h>

#include <string>

namespace dmk::cuda::jit {

std::string make_eval_targets_source(const JitKey &key);
std::string make_self_correction_source(const JitKey &key);

template <typename Real, int DIM>
void launch_eval_targets_jit(JitCache &cache, int eval_level, int n_charge_dim,
                             const dmk::cuda::EvalTargetsArgs<Real> &args, cudaStream_t stream, int blocksize,
                             int targets_per_thread);

extern template void launch_eval_targets_jit<float, 2>(JitCache &, int, int, const dmk::cuda::EvalTargetsArgs<float> &,
                                                       cudaStream_t, int, int);

extern template void launch_eval_targets_jit<float, 3>(JitCache &, int, int, const dmk::cuda::EvalTargetsArgs<float> &,
                                                       cudaStream_t, int, int);

extern template void launch_eval_targets_jit<double, 2>(JitCache &, int, int,
                                                        const dmk::cuda::EvalTargetsArgs<double> &, cudaStream_t, int,
                                                        int);

extern template void launch_eval_targets_jit<double, 3>(JitCache &, int, int,
                                                        const dmk::cuda::EvalTargetsArgs<double> &, cudaStream_t, int,
                                                        int);

template <typename Real>
void launch_self_correction_jit(JitCache &cache, const dmk::cuda::SelfCorrectionArgs<Real> &args, cudaStream_t stream,
                                int blocksize);

extern template void launch_self_correction_jit<float>(JitCache &, const dmk::cuda::SelfCorrectionArgs<float> &,
                                                       cudaStream_t, int);

extern template void launch_self_correction_jit<double>(JitCache &, const dmk::cuda::SelfCorrectionArgs<double> &,
                                                        cudaStream_t, int);

} // namespace dmk::cuda::jit
