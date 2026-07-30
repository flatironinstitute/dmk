#pragma once

// Batched proxy2pw launcher: one autotuned launch over a device array of
// per-level args (n_args == 1 for the windowed root). proxy2pw writes plane-wave
// modes by assignment, so it needs no autotune snapshot.

#include <vector>

#include <cuda_runtime.h>
#include <dmk/cuda/proxy2pw_kernelargs.hpp>

namespace dmk::cuda::pt {

template <typename Real>
void launch_proxy2pw(std::vector<dmk::cuda::Proxy2PwArgs<Real>> &args_h, cudaStream_t stream);

} // namespace dmk::cuda::pt
