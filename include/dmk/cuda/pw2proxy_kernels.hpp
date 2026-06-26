#ifndef DMK_CUDA_PW_TO_PROXY_KERNELS_HPP
#define DMK_CUDA_PW_TO_PROXY_KERNELS_HPP

// Per-level GPU planewave_to_proxy_potential. One block per box. Reads from
// pw_in_pool (filled earlier on the same stream by the shift_planewave
// kernel), does the 3 axis-wise complex GEMMs + halve, and accumulates the
// real part into d_proxy_coeffs_downward.
//
// Math (per box, per charge_dim d):
//   ff[k1, m2, m3]    = sum_{m1} pw2poly[m1, k1]      * pw_in[m1, m2, m3, d]
//   ff2[k1, k2, m3]   = halve(m3) *
//                       sum_{m2} pw2poly[m2, k2]      * ff[k1, m2, m3]
//   zcoefs[k1,k2,k3]  = sum_{m3} ff2[k1, k2, m3]      * pw2poly[m3, k3]
//   proxy[k1,k2,k3,d] += 2 * Re(zcoefs[k1, k2, k3])
//
//   halve(m3) = 0.5 if m3 >= n_pw/2 else 1.0
//
// To stay shared-memory friendly we serialize over k1: per k1 iteration the
// block computes only ff[k1, *, *] and ff2[k1, *, *], then writes the
// k1-slab of proxy. Per-k1 shared memory ≈ 2 * n_pw * n_pw2 (ff slab) +
// 2 * n_order * n_pw2 (ff2 slab) reals — a few KB even for large n_pw.

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
