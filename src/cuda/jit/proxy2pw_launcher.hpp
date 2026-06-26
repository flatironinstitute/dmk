#pragma once

#include "jit_cache.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/proxy2pw_kernels.hpp>

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace dmk::cuda::jit {

std::string make_proxy2pw_source(const JitKey& key);

template <typename Real>
void launch_proxy2pw_jit(
    JitCache& cache,
    const dmk::cuda::Proxy2PwArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
);

template <typename Real>
void launch_proxy2pw_multilevel_jit(
    JitCache& cache,
    const std::vector<dmk::cuda::Proxy2PwArgs<Real>>& pa_h,
    dmk::cuda::Proxy2PwArgs<Real>* d_args_scratch,
    cudaStream_t stream,
    int blocksize
);

extern template void launch_proxy2pw_jit<float>(
    JitCache&,
    const dmk::cuda::Proxy2PwArgs<float>&,
    cudaStream_t,
    int
);

extern template void launch_proxy2pw_jit<double>(
    JitCache&,
    const dmk::cuda::Proxy2PwArgs<double>&,
    cudaStream_t,
    int
);

extern template void launch_proxy2pw_multilevel_jit<float>(
    JitCache&,
    const std::vector<dmk::cuda::Proxy2PwArgs<float>>&,
    dmk::cuda::Proxy2PwArgs<float>*,
    cudaStream_t,
    int
);

extern template void launch_proxy2pw_multilevel_jit<double>(
    JitCache&,
    const std::vector<dmk::cuda::Proxy2PwArgs<double>>&,
    dmk::cuda::Proxy2PwArgs<double>*,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit