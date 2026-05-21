// Proxy-to-plane-wave launchers. Kernel implementations live in
// src/cuda/jit_sources/proxy2pw*.cu and are compiled with NVRTC.

#include <dmk/cuda/proxy2pw_kernels.hpp>

#include "cuda/jit/proxy2pw_launcher.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_proxy2pw(const Proxy2PwArgs<Real>& args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_proxy2pw_jit<Real>(
        jit_cache,
        args,
        stream,
        block_size
    );
}

template <typename Real, int DIM>
void launch_proxy2pw_multilevel(
    const std::vector<Proxy2PwArgs<Real>>& pa_h,
    Proxy2PwArgs<Real>* d_args_scratch,
    cudaStream_t stream
) {
    if (pa_h.empty())
        return;

    constexpr int block_size = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_proxy2pw_multilevel_jit<Real>(
        jit_cache,
        pa_h,
        d_args_scratch,
        stream,
        block_size
    );
}

template void launch_proxy2pw<float, 2>(const Proxy2PwArgs<float>&, cudaStream_t);
template void launch_proxy2pw<float, 3>(const Proxy2PwArgs<float>&, cudaStream_t);
template void launch_proxy2pw<double, 2>(const Proxy2PwArgs<double>&, cudaStream_t);
template void launch_proxy2pw<double, 3>(const Proxy2PwArgs<double>&, cudaStream_t);

template void launch_proxy2pw_multilevel<float, 2>(const std::vector<Proxy2PwArgs<float>>&, Proxy2PwArgs<float>*,
                                                   cudaStream_t);
template void launch_proxy2pw_multilevel<float, 3>(const std::vector<Proxy2PwArgs<float>>&, Proxy2PwArgs<float>*,
                                                   cudaStream_t);
template void launch_proxy2pw_multilevel<double, 2>(const std::vector<Proxy2PwArgs<double>>&, Proxy2PwArgs<double>*,
                                                    cudaStream_t);
template void launch_proxy2pw_multilevel<double, 3>(const std::vector<Proxy2PwArgs<double>>&, Proxy2PwArgs<double>*,
                                                    cudaStream_t);

} // namespace dmk::cuda
