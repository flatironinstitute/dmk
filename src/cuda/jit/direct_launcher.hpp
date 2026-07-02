#pragma once

#include "jit_cache.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/direct_kernelargs.hpp>

#include <cuda_runtime.h>

namespace dmk::cuda::jit {

template <typename Evaluator, typename Real = typename Evaluator::scalar_type>
void launch_direct_by_box_jit(JitCache &cache, const dmk::cuda::DirectByBoxArgs<Real> &args, cudaStream_t stream,
                              int src_tile, int blocksize, int targets_per_thread = 1);

} // namespace dmk::cuda::jit
