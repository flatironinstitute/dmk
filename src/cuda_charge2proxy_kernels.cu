#include <dmk/cuda_charge2proxy_kernels.cuh>

#include <stdexcept>
#include <string>

#include <thrust/device_ptr.h>

#ifdef DMK_CUDA_USE_NVRTC_JIT
#include <cstdlib>
#include "cuda/jit/charge2proxy_launcher.hpp"
#endif

namespace dmk::cuda {

#ifdef DMK_CUDA_USE_NVRTC_JIT
namespace {

inline bool charge2proxy_jit_enabled() {
    const char *disable = std::getenv("DMK_DISABLE_CHARGE2PROXY_JIT");
    return !(disable && std::string(disable) == "1");
}

} // namespace
#endif

template <typename Real>
void launch_charge2proxy_dispatch(
    int dim,
    const Charge2ProxyArgs<Real> &args,
    cudaStream_t stream
) {
    if (dim != 3) {
        throw std::runtime_error(
            "CUDA charge2proxy: dim=" + std::to_string(dim) +
            " not supported (only 3D for now)"
        );
    }

    if (args.n_groups == 0) {
        return;
    }

    Charge2ProxyGroupOrder order;
    build_charge2proxy_group_order(args, order, stream);

    if (order.n_active_groups == 0) {
        return;
    }

    const int *group_perm =
        thrust::raw_pointer_cast(order.group_perm.data());

#ifdef DMK_CUDA_USE_NVRTC_JIT
    if (charge2proxy_jit_enabled()) {
        static dmk::cuda::jit::JitCache jit_cache;

        dmk::cuda::jit::launch_charge2proxy_jit<Real>(
            jit_cache,
            args,
            group_perm,
            order.n_active_groups,
            stream,
            128, // CHUNK
            3,   // I_TILE
            3,   // J_TILE
            4,   // K_TILE
            256  // blocksize
        );

        return;
    }
#endif

    launch_charge2proxy_3d<Real>(
        args,
        group_perm,
        order.n_active_groups,
        stream
    );
}

template void launch_charge2proxy_dispatch<float>(
    int,
    const Charge2ProxyArgs<float> &,
    cudaStream_t
);

template void launch_charge2proxy_dispatch<double>(
    int,
    const Charge2ProxyArgs<double> &,
    cudaStream_t
);

} // namespace dmk::cuda