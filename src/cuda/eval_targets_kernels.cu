// Eval-target and self-correction launchers. Kernel implementations live in
// src/cuda/jit_sources/eval_targets.cu and self_correction.cu and are
// compiled with NVRTC.

#include <dmk/cuda/eval_targets_kernels.hpp>

#include "cuda/jit/eval_targets_launcher.hpp"

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_eval_targets(int eval_level, int n_charge_dim, const EvalTargetsArgs<Real> &args, cudaStream_t stream) {
    if (args.n_eval_boxes == 0)
        return;

    constexpr int block_size = 640;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_eval_targets_jit<Real, DIM>(
        jit_cache,
        eval_level,
        n_charge_dim,
        args,
        stream,
        block_size
    );
}

template void launch_eval_targets<float, 2>(int, int, const EvalTargetsArgs<float> &, cudaStream_t);
template void launch_eval_targets<float, 3>(int, int, const EvalTargetsArgs<float> &, cudaStream_t);
template void launch_eval_targets<double, 2>(int, int, const EvalTargetsArgs<double> &, cudaStream_t);
template void launch_eval_targets<double, 3>(int, int, const EvalTargetsArgs<double> &, cudaStream_t);

template <typename Real>
void launch_self_correction(const SelfCorrectionArgs<Real> &args, cudaStream_t stream) {
    if (args.n_direct_work == 0)
        return;

    constexpr int block = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_self_correction_jit<Real>(jit_cache, args, stream, block);
}

template void launch_self_correction<float>(const SelfCorrectionArgs<float> &, cudaStream_t);
template void launch_self_correction<double>(const SelfCorrectionArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
