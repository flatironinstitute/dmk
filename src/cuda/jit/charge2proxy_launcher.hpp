// src/cuda/jit/charge2proxy_launcher.hpp
#pragma once

#include "jit_cache.hpp"
#include "jit_types.hpp"

#include <dmk/cuda_charge2proxy_kernels.hpp>

#include <cuda_runtime.h>

#include <string>

namespace dmk::cuda::jit {

std::string make_charge2proxy_source(const JitKey& key);

template <typename Real>
void launch_charge2proxy_jit(
    JitCache& cache,
    const dmk::cuda::Charge2ProxyArgs<Real>& args,
    const int* group_perm,
    int n_launch_groups,
    cudaStream_t stream,
    int chunk,
    int i_tile,
    int j_tile,
    int k_tile,
    int blocksize
);

extern template void launch_charge2proxy_jit<float>(
    JitCache&,
    const dmk::cuda::Charge2ProxyArgs<float>&,
    const int*,
    int,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int
);

extern template void launch_charge2proxy_jit<double>(
    JitCache&,
    const dmk::cuda::Charge2ProxyArgs<double>&,
    const int*,
    int,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int
);

} // namespace dmk::cuda::jit