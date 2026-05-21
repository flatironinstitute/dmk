// Shift plane-wave launchers. Kernel implementation lives in
// src/cuda/jit_sources/shiftpw.cu and is compiled with NVRTC.

#include <dmk/cuda/shift_pw_kernels.hpp>

#include "cuda/jit/shift_pw_launcher.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_shift_pw(const ShiftPwArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 512;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_shift_pw_jit<Real>(jit_cache, args, stream, block_size);
}

template void launch_shift_pw<float, 2>(const ShiftPwArgs<float> &, cudaStream_t);
template void launch_shift_pw<float, 3>(const ShiftPwArgs<float> &, cudaStream_t);
template void launch_shift_pw<double, 2>(const ShiftPwArgs<double> &, cudaStream_t);
template void launch_shift_pw<double, 3>(const ShiftPwArgs<double> &, cudaStream_t);

template <typename Real, int DIM>
void launch_shift_pw_multilevel(
    const std::vector<ShiftPwArgs<Real>> &args_h,
    ShiftPwArgs<Real> *d_args_scratch,
    cudaStream_t stream
) {
    if (args_h.empty())
        return;

    constexpr int block_size = 256;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_shift_pw_multilevel_jit<Real>(
        jit_cache,
        args_h,
        d_args_scratch,
        stream,
        block_size
    );
}

template void launch_shift_pw_multilevel<float, 2>(const std::vector<ShiftPwArgs<float>> &, ShiftPwArgs<float> *,
                                                   cudaStream_t);
template void launch_shift_pw_multilevel<float, 3>(const std::vector<ShiftPwArgs<float>> &, ShiftPwArgs<float> *,
                                                   cudaStream_t);
template void launch_shift_pw_multilevel<double, 2>(const std::vector<ShiftPwArgs<double>> &, ShiftPwArgs<double> *,
                                                    cudaStream_t);
template void launch_shift_pw_multilevel<double, 3>(const std::vector<ShiftPwArgs<double>> &, ShiftPwArgs<double> *,
                                                    cudaStream_t);

} // namespace dmk::cuda
