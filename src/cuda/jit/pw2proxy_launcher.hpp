#pragma once

#include "jit_cache.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/pw2proxy_kernels.hpp>

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace dmk::cuda::jit {

std::string make_pw2proxy_source(const JitKey& key);

template <typename Real>
void launch_pw_to_proxy_jit(
    JitCache& cache,
    const dmk::cuda::PwToProxyArgs<Real>& args,
    cudaStream_t stream,
    int k1_tile,
    int col_reg,
    int k2_tile,
    int k3_tile,
    int kr_tile,
    int blocksize
);

template <typename Real>
void launch_pw_to_proxy_multilevel_jit(
    JitCache& cache,
    const std::vector<dmk::cuda::PwToProxyArgs<Real>>& args_h,
    dmk::cuda::PwToProxyArgs<Real>* d_args_scratch,
    cudaStream_t stream,
    int k1_tile,
    int col_reg,
    int k2_tile,
    int k3_tile,
    int kr_tile,
    int blocksize
);

extern template void launch_pw_to_proxy_jit<float>(
    JitCache&,
    const dmk::cuda::PwToProxyArgs<float>&,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int,
    int
);

extern template void launch_pw_to_proxy_jit<double>(
    JitCache&,
    const dmk::cuda::PwToProxyArgs<double>&,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int,
    int
);

extern template void launch_pw_to_proxy_multilevel_jit<float>(
    JitCache&,
    const std::vector<dmk::cuda::PwToProxyArgs<float>>&,
    dmk::cuda::PwToProxyArgs<float>*,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int,
    int
);

extern template void launch_pw_to_proxy_multilevel_jit<double>(
    JitCache&,
    const std::vector<dmk::cuda::PwToProxyArgs<double>>&,
    dmk::cuda::PwToProxyArgs<double>*,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int,
    int
);

} // namespace dmk::cuda::jit