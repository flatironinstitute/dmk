// Per-pair tensorprod launcher. Kernel implementation lives in
// src/cuda/jit_sources/tensorproduct.cu and is compiled with NVRTC.

#include <dmk/cuda/tensorprod_kernels.hpp>

#include "cuda/jit/tensorprod_launcher.hpp"

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_tensorprod(const TensorprodArgs<Real> &args, cudaStream_t stream) {
    if (args.n_pairs == 0)
        return;

    constexpr int block_size = 512;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_tensorprod_jit<Real>(
        jit_cache,
        args,
        stream,
        block_size
    );
}

template void launch_tensorprod<float, 2>(const TensorprodArgs<float> &, cudaStream_t);
template void launch_tensorprod<float, 3>(const TensorprodArgs<float> &, cudaStream_t);
template void launch_tensorprod<double, 2>(const TensorprodArgs<double> &, cudaStream_t);
template void launch_tensorprod<double, 3>(const TensorprodArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
