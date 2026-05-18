#pragma once

#include "jit_cache.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/tensorprod_kernels.hpp>

#include <cuda_runtime.h>

#include <string>

namespace dmk::cuda::jit {

std::string make_tensorprod_source(const JitKey& key);

template <typename Real>
void launch_tensorprod_jit(
    JitCache& cache,
    const dmk::cuda::TensorprodArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
);

extern template void launch_tensorprod_jit<float>(
    JitCache&,
    const dmk::cuda::TensorprodArgs<float>&,
    cudaStream_t,
    int
);

extern template void launch_tensorprod_jit<double>(
    JitCache&,
    const dmk::cuda::TensorprodArgs<double>&,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit