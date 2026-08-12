#pragma once

// Batched proxy2pw launcher: one autotuned launch over a device array of
// per-level args (n_args == 1 for the windowed root). proxy2pw writes plane-wave
// modes by assignment, so it needs no autotune snapshot.

#include <cstddef>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>
#include <dmk/cuda/proxy2pw_kernelargs.hpp>

namespace dmk::cuda::pt {

// `variant` separates the autotune entry of the single-box windowed root from
// the many-box per-level batch (see launch_pw2proxy).
template <typename Real>
void launch_proxy2pw(std::vector<dmk::cuda::Proxy2PwArgs<Real>> &args_h, cudaStream_t stream,
                     std::string_view variant = "");

// Smallest shared footprint the tuning space can reach. Fusing the Stokeslet projector needs one
// phase-2 buffer per charge dim (`ff2_copies`), which can outgrow the device limit; the caller then
// leaves the multiply as its own pass.
std::size_t proxy2pw_min_shared_bytes(int n_order, int n_pw, int ff2_copies, std::size_t sizeof_real);

} // namespace dmk::cuda::pt
