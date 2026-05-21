#pragma once

#include "jit_cache.hpp"
#include "jit_types.hpp"

#include <cuda_runtime.h>

#include <string>

namespace dmk::cuda::jit {

std::string make_shared_state_source(const JitKey& key);

template <typename Real>
void launch_accumulate_and_scatter_jit(
    JitCache& cache,
    Real* out,
    const Real* pot_eval,
    const Real* pot_extra,
    const long* scatter_index,
    int dof,
    long n_particles,
    cudaStream_t stream,
    int blocksize
);

extern template void launch_accumulate_and_scatter_jit<float>(
    JitCache&,
    float*,
    const float*,
    const float*,
    const long*,
    int,
    long,
    cudaStream_t,
    int
);

extern template void launch_accumulate_and_scatter_jit<double>(
    JitCache&,
    double*,
    const double*,
    const double*,
    const long*,
    int,
    long,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit
