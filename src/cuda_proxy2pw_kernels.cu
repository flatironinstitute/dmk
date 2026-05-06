// Runtime dispatch for the per-block proxy2pw kernel.

#include <dmk/cuda_proxy2pw_kernels.cuh>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

template <typename Real>
void launch_proxy2pw_dispatch(int dim, const Proxy2PwArgs<Real> &args, cudaStream_t stream) {
    if (dim == 3) {
        launch_proxy2pw_3d<Real>(args, stream);
        return;
    }
    throw std::runtime_error("CUDA proxy2pw: dim=" + std::to_string(dim) + " not supported (only 3D for now)");
}

template void launch_proxy2pw_dispatch<float>(int, const Proxy2PwArgs<float> &, cudaStream_t);
template void launch_proxy2pw_dispatch<double>(int, const Proxy2PwArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
