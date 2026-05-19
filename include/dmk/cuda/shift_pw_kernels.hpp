#ifndef DMK_CUDA_SHIFT_PW_KERNELS_HPP
#define DMK_CUDA_SHIFT_PW_KERNELS_HPP

// Per-level GPU shift_planewave: one block per box that does PW work at the
// level. The block initialises its slot in `pw_in_pool` from the box's own
// `pw_out`, then loops over its (up to 3^DIM) neighbours, accumulating
// `pw_out[neighbour] * wpwshift[ind]` into `pw_in_pool[slot]`. The next
// per-level kernel (pw_to_proxy) reads from this pool.
//
// Storage convention:
//   pw_out[box]    : interleaved complex (real, imag, real, imag, ...).
//                    Size n_charge_dim * n_pw_modes complex per box.
//   pw_in_pool     : interleaved complex, [slot * stride .. slot*stride + stride).
//                    stride = n_charge_dim * n_pw_modes * 2 reals.
//   wpwshift[ind]  : SoA — n_pw_modes reals (real parts), then n_pw_modes
//                    reals (imag parts). One per neighbour-direction slot,
//                    n_neighbors slots per level. (= 3^DIM)

#include <cuda_runtime.h>

#include <vector>
#include <dmk/cuda/shift_pw_kernelargs.hpp>
namespace dmk::cuda {

template <typename Real, int DIM>
void launch_shift_pw(const ShiftPwArgs<Real> &args, cudaStream_t stream);

// d_args_scratch must point to device memory sized >= args_h.size() entries.
// The orchestrator owns it; this function uploads args_h into it and launches.
template <typename Real, int DIM>
void launch_shift_pw_multilevel(const std::vector<ShiftPwArgs<Real>> &args_h, ShiftPwArgs<Real> *d_args_scratch,
                                cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_SHIFT_PW_KERNELS_HPP
