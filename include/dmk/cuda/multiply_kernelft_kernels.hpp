#ifndef DMK_CUDA_MULTIPLY_KERNELFT_KERNELS_HPP
#define DMK_CUDA_MULTIPLY_KERNELFT_KERNELS_HPP

// Per-block GPU equivalents of the host multiply_kernelFT_* family. Each
// kernel multiplies a per-box plane-wave expansion by a Fourier-space
// kernel (radialft + a kernel-specific formula). Output is interleaved
// complex.
//
//   - cd2p:        scalar/Laplace-style (pw[m, d] *= radialft[m]).
//   - stokeslet_3d: vector kernel; couples 3 charge dims via k.
//   - stresslet_3d: tensor kernel; reads 9 input tables, writes 3 output.

#include <dmk/cuda/multiply_kernelft_kernelargs.hpp>

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_multiply_cd2p(const MultiplyCd2pArgs<Real> &args, cudaStream_t stream);

template <typename Real>
void launch_multiply_stokeslet_3d(const MultiplyStokeslet3DArgs<Real> &args, cudaStream_t stream);

template <typename Real>
void launch_multiply_stresslet_3d(const MultiplyStresslet3DArgs<Real> &args, cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_MULTIPLY_KERNELFT_KERNELS_HPP
