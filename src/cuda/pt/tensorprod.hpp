#pragma once

// Shared tensorprod launcher: one autotuned launch for a per-level batch of
// (src,dst,octant) proxy-transfer pairs. Used by pt::upward (c2p, additive)
// and pt::downward (p2c). `proxy_count` is the full proxy buffer length (reals)
// used for the autotune snapshot of the additive output. Levels differ in grid by
// three orders of magnitude, so `tune_here` picks the one worth measuring: set it on
// the level with the most work and the rest fall back to the tuned config or defaults.

#include <cstddef>

#include <cuda_runtime.h>
#include <dmk/cuda/tensorprod_kernelargs.hpp>

namespace dmk::cuda::pt {

template <typename Real>
void launch_tensorprod(dmk::cuda::TensorprodArgs<Real> &args, std::size_t proxy_count, cudaStream_t stream,
                       bool tune_here);

} // namespace dmk::cuda::pt
