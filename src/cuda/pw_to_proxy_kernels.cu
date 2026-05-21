// Plane-wave-to-proxy launchers. Kernel implementations live in
// src/cuda/jit_sources/pw2proxy*.cu and are compiled with NVRTC.

#include <dmk/cuda/pw2proxy_kernels.hpp>

#include "cuda/jit/pw2proxy_launcher.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_pw_to_proxy(const PwToProxyArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_pw_to_proxy_jit<Real>(
        jit_cache,
        args,
        stream,
        6,  // K1_TILE
        2,  // COL_REG
        2,  // K2_TILE
        3,  // K3_TILE
        6,  // KR_TILE
        block_size
    );
}

template void launch_pw_to_proxy<float, 2>(const PwToProxyArgs<float> &, cudaStream_t);
template void launch_pw_to_proxy<float, 3>(const PwToProxyArgs<float> &, cudaStream_t);
template void launch_pw_to_proxy<double, 2>(const PwToProxyArgs<double> &, cudaStream_t);
template void launch_pw_to_proxy<double, 3>(const PwToProxyArgs<double> &, cudaStream_t);

template <typename Real, int DIM>
void launch_pw_to_proxy_multilevel(
    const std::vector<PwToProxyArgs<Real>> &args_h,
    PwToProxyArgs<Real> *d_args_scratch,
    cudaStream_t stream
) {
    if (args_h.empty())
        return;

    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_pw_to_proxy_multilevel_jit<Real>(
        jit_cache,
        args_h,
        d_args_scratch,
        stream,
        18,  // K1_TILE
        1,   // COL_REG
        2,   // K2_TILE
        3,   // K3_TILE
        9,   // KR_TILE
        256  // block_size
    );
}

template void launch_pw_to_proxy_multilevel<float, 2>(const std::vector<PwToProxyArgs<float>> &,
                                                      PwToProxyArgs<float> *, cudaStream_t);
template void launch_pw_to_proxy_multilevel<float, 3>(const std::vector<PwToProxyArgs<float>> &,
                                                      PwToProxyArgs<float> *, cudaStream_t);
template void launch_pw_to_proxy_multilevel<double, 2>(const std::vector<PwToProxyArgs<double>> &,
                                                       PwToProxyArgs<double> *, cudaStream_t);
template void launch_pw_to_proxy_multilevel<double, 3>(const std::vector<PwToProxyArgs<double>> &,
                                                       PwToProxyArgs<double> *, cudaStream_t);

} // namespace dmk::cuda
