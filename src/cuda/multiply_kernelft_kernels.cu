// Kernel-FT multiply launchers. Kernel implementations live in
// src/cuda/jit_sources/multiply_kernelft.cu and are compiled with NVRTC.

#include <dmk/cuda/multiply_kernelft_kernels.hpp>

#include "cuda/jit/multiply_kernelft_launcher.hpp"

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_multiply_cd2p(const MultiplyCd2pArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_multiply_cd2p_jit<Real, DIM>(jit_cache, args, stream, block_size);
}

template <typename Real>
void launch_multiply_stokeslet_3d(const MultiplyStokeslet3DArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_multiply_stokeslet_3d_jit<Real>(jit_cache, args, stream, block_size);
}

template <typename Real>
void launch_multiply_stresslet_3d(const MultiplyStresslet3DArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_multiply_stresslet_3d_jit<Real>(jit_cache, args, stream, block_size);
}

template void launch_multiply_cd2p<float, 2>(const MultiplyCd2pArgs<float> &, cudaStream_t);
template void launch_multiply_cd2p<float, 3>(const MultiplyCd2pArgs<float> &, cudaStream_t);
template void launch_multiply_cd2p<double, 2>(const MultiplyCd2pArgs<double> &, cudaStream_t);
template void launch_multiply_cd2p<double, 3>(const MultiplyCd2pArgs<double> &, cudaStream_t);
template void launch_multiply_stokeslet_3d<float>(const MultiplyStokeslet3DArgs<float> &, cudaStream_t);
template void launch_multiply_stokeslet_3d<double>(const MultiplyStokeslet3DArgs<double> &, cudaStream_t);
template void launch_multiply_stresslet_3d<float>(const MultiplyStresslet3DArgs<float> &, cudaStream_t);
template void launch_multiply_stresslet_3d<double>(const MultiplyStresslet3DArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
