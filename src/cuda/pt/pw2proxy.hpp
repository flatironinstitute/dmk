#pragma once

// Batched pw2proxy launcher: one autotuned launch over a device array of
// per-level args (n_args == 1 for the windowed root). pw2proxy accumulates into
// proxy_flat, so autotune snapshots/restores the proxy buffer (`proxy_count`
// reals) around each timed run.

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>
#include <dmk/cuda/pw2proxy_kernelargs.hpp>

namespace dmk::cuda::pt {

template <typename Real>
void launch_pw2proxy(std::vector<dmk::cuda::PwToProxyArgs<Real>> &args_h, Real *proxy_flat, std::size_t proxy_count,
                     cudaStream_t stream);

} // namespace dmk::cuda::pt
