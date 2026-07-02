#pragma once

#include "jit_cache.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/multiply_kernelft_kernelargs.hpp>

#include <cuda_runtime.h>

#include <string>

namespace dmk::cuda::jit {

std::string make_multiply_kernelft_source(const JitKey &key);

template <typename Real, int DIM>
void launch_multiply_cd2p_jit(JitCache &cache, const dmk::cuda::MultiplyCd2pArgs<Real> &args, cudaStream_t stream,
                              int blocksize);

template <typename Real>
void launch_multiply_stokeslet_3d_jit(JitCache &cache, const dmk::cuda::MultiplyStokeslet3DArgs<Real> &args,
                                      cudaStream_t stream, int blocksize);

template <typename Real>
void launch_multiply_stresslet_3d_jit(JitCache &cache, const dmk::cuda::MultiplyStresslet3DArgs<Real> &args,
                                      cudaStream_t stream, int blocksize);

extern template void launch_multiply_cd2p_jit<float, 2>(JitCache &, const dmk::cuda::MultiplyCd2pArgs<float> &,
                                                        cudaStream_t, int);

extern template void launch_multiply_cd2p_jit<float, 3>(JitCache &, const dmk::cuda::MultiplyCd2pArgs<float> &,
                                                        cudaStream_t, int);

extern template void launch_multiply_cd2p_jit<double, 2>(JitCache &, const dmk::cuda::MultiplyCd2pArgs<double> &,
                                                         cudaStream_t, int);

extern template void launch_multiply_cd2p_jit<double, 3>(JitCache &, const dmk::cuda::MultiplyCd2pArgs<double> &,
                                                         cudaStream_t, int);

extern template void launch_multiply_stokeslet_3d_jit<float>(JitCache &,
                                                             const dmk::cuda::MultiplyStokeslet3DArgs<float> &,
                                                             cudaStream_t, int);

extern template void launch_multiply_stokeslet_3d_jit<double>(JitCache &,
                                                              const dmk::cuda::MultiplyStokeslet3DArgs<double> &,
                                                              cudaStream_t, int);

extern template void launch_multiply_stresslet_3d_jit<float>(JitCache &,
                                                             const dmk::cuda::MultiplyStresslet3DArgs<float> &,
                                                             cudaStream_t, int);

extern template void launch_multiply_stresslet_3d_jit<double>(JitCache &,
                                                              const dmk::cuda::MultiplyStresslet3DArgs<double> &,
                                                              cudaStream_t, int);

} // namespace dmk::cuda::jit
