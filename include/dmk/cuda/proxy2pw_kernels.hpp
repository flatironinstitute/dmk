#ifndef DMK_CUDA_PROXY2PW_KERNELS_HPP
#define DMK_CUDA_PROXY2PW_KERNELS_HPP

// Per-block GPU proxycharge2pw: one block per box, projects the box's upward
// proxy expansion onto plane-wave modes via 3 axis-wise complex GEMMs with
// poly2pw. Output is interleaved complex.

#include <cuda_runtime.h>
#include <dmk/cuda/proxy2pw_kernelargs.hpp>
#include <vector>
namespace dmk::cuda {


template <typename Real, int DIM>
void launch_proxy2pw(const Proxy2PwArgs<Real> &args, cudaStream_t stream);

// d_args_scratch must point to device memory sized >= pa_h.size() entries.
template <typename Real, int DIM>
void launch_proxy2pw_multilevel(const std::vector<Proxy2PwArgs<Real>> &pa_h, Proxy2PwArgs<Real> *d_args_scratch,
                                cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_PROXY2PW_KERNELS_HPP
