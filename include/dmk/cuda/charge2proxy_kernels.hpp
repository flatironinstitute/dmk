#ifndef DMK_CUDA_CHARGE2PROXY_KERNELS_HPP
#define DMK_CUDA_CHARGE2PROXY_KERNELS_HPP

// Host-includable header for the GPU charge2proxy kernel. One block per
// Charge2ProxyGroup; the block accumulates contributions from each src_box in
// the group's list into proxy_coeffs_upward[group.center_box]. Output is
// additive — caller must zero d_proxy_coeffs_upward before launch.

#include <cuda_runtime.h>
#include <dmk/cuda/charge2proxy_kernelargs.hpp>
namespace dmk::cuda {

template <typename Real, int DIM>
void launch_charge2proxy(const Charge2ProxyArgs<Real> &args, cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_CHARGE2PROXY_KERNELS_HPP
