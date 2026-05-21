#pragma once

#include "jit_cache.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/shift_pw_kernels.hpp>

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace dmk::cuda::jit {

std::string make_shift_pw_source(const JitKey& key);

template <typename Real>
void launch_shift_pw_jit(
    JitCache& cache,
    const dmk::cuda::ShiftPwArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
);

template <typename Real>
void launch_shift_pw_multilevel_jit(
    JitCache& cache,
    const std::vector<dmk::cuda::ShiftPwArgs<Real>>& args_h,
    dmk::cuda::ShiftPwArgs<Real>* d_args_scratch,
    cudaStream_t stream,
    int blocksize
);

extern template void launch_shift_pw_jit<float>(
    JitCache&,
    const dmk::cuda::ShiftPwArgs<float>&,
    cudaStream_t,
    int
);

extern template void launch_shift_pw_jit<double>(
    JitCache&,
    const dmk::cuda::ShiftPwArgs<double>&,
    cudaStream_t,
    int
);

extern template void launch_shift_pw_multilevel_jit<float>(
    JitCache&,
    const std::vector<dmk::cuda::ShiftPwArgs<float>>&,
    dmk::cuda::ShiftPwArgs<float>*,
    cudaStream_t,
    int
);

extern template void launch_shift_pw_multilevel_jit<double>(
    JitCache&,
    const std::vector<dmk::cuda::ShiftPwArgs<double>>&,
    dmk::cuda::ShiftPwArgs<double>*,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit
