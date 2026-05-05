// Runtime dispatch for the per-level pw_to_proxy kernel.

#include <dmk/cuda_pw_to_proxy_kernels.cuh>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

template <typename Real>
void launch_pw_to_proxy_dispatch(int dim, const PwToProxyArgs<Real> &args, cudaStream_t stream) {
    if (dim == 3) {
        launch_pw_to_proxy_3d<Real>(args, stream);
        return;
    }
    throw std::runtime_error("CUDA pw_to_proxy: dim=" + std::to_string(dim) + " not supported (only 3D for now)");
}

template void launch_pw_to_proxy_dispatch<float>(int, const PwToProxyArgs<float> &, cudaStream_t);
template void launch_pw_to_proxy_dispatch<double>(int, const PwToProxyArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
