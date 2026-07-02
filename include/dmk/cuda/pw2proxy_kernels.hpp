#ifndef DMK_CUDA_PW_TO_PROXY_KERNELS_HPP
#define DMK_CUDA_PW_TO_PROXY_KERNELS_HPP

// Per-level GPU planewave_to_proxy_potential. One block per box. Reads from
// pw_in_pool (filled earlier on the same stream by the shift_planewave
// kernel), does the 3 axis-wise complex GEMMs + halve, and accumulates the
// real part into d_proxy_coeffs_downward.
//
// Math (per box, per charge_dim d):
//   f3[m1, m2, k3]    = sum_{m3} halve(m3) *
//                       pw_in[m1, m2, m3, d]          * pw2poly[m3, k3]
//   f2[m1, k2, k3]    = sum_{m2} f3[m1, m2, k3]       * pw2poly[m2, k2]
//   zcoefs[k1,k2,k3]  = sum_{m1} f2[m1, k2, k3]       * pw2poly[m1, k1]
//   proxy[k1,k2,k3,d] += 2 * Re(zcoefs[k1, k2, k3])
//
//   halve(m3) = 0.5 if m3 >= n_pw/2 else 1.0
//
// The GPU implementation tiles k3 first so the first phase maps threads over
// the contiguous (m1,m2) plane of pw_in. Shared memory holds one k3 tile of the
// two complex intermediates.

#include <cuda_runtime.h>
#include <dmk/cuda/pw2proxy_kernelargs.hpp>
#include <vector>
namespace dmk::cuda {

template <typename Real, int DIM>
void launch_pw_to_proxy(const PwToProxyArgs<Real> &args, cudaStream_t stream);

// d_args_scratch must point to device memory sized >= args_h.size() entries.
template <typename Real, int DIM>
void launch_pw_to_proxy_multilevel(const std::vector<PwToProxyArgs<Real>> &args_h, PwToProxyArgs<Real> *d_args_scratch,
                                   cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_PW_TO_PROXY_KERNELS_HPP
