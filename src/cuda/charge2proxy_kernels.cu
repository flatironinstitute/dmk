// Per-group charge2proxy launcher. Kernel implementation lives in
// src/cuda/jit_sources/charge2proxy.cu and is compiled with NVRTC.

#include <dmk/cuda/charge2proxy_kernels.hpp>

#include "cuda/jit/charge2proxy_launcher.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

namespace {

template <typename Real>
void launch_charge2proxy_3d_impl(const Charge2ProxyArgs<Real> &args, const int *group_perm, int n_launch_groups,
                                 cudaStream_t stream) {
    if (args.n_groups == 0 || n_launch_groups == 0)
        return;

    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_charge2proxy_jit<Real>(jit_cache, args, group_perm, n_launch_groups, stream,
                                                  128, // CHUNK
                                                  3,   // I_TILE
                                                  3,   // J_TILE
                                                  4,   // K_TILE
                                                  256  // block size
    );
}

} // namespace

template <typename Real, int DIM>
void launch_charge2proxy(const Charge2ProxyArgs<Real> &args, cudaStream_t stream) {
    if constexpr (DIM != 3) {
        throw std::runtime_error("CUDA charge2proxy: only DIM=3 supported for now");
    } else {
        launch_charge2proxy_3d_impl<Real>(args, args.group_perm, args.n_active_groups, stream);
    }
}

template void launch_charge2proxy<float, 2>(const Charge2ProxyArgs<float> &, cudaStream_t);
template void launch_charge2proxy<float, 3>(const Charge2ProxyArgs<float> &, cudaStream_t);
template void launch_charge2proxy<double, 2>(const Charge2ProxyArgs<double> &, cudaStream_t);
template void launch_charge2proxy<double, 3>(const Charge2ProxyArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
