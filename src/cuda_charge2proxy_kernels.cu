// Runtime dispatch for the charge2proxy GPU kernel.

#include <dmk/cuda_charge2proxy_kernels.cuh>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

template <typename Real>
void launch_charge2proxy_dispatch(int dim, const Charge2ProxyArgs<Real> &args, cudaStream_t stream) {
    if (dim == 3) {
        launch_charge2proxy_3d<Real>(args, stream);
        return;
    }
    throw std::runtime_error("CUDA charge2proxy: dim=" + std::to_string(dim) +
                             " not supported (only 3D for now)");
}

template void launch_charge2proxy_dispatch<float>(int, const Charge2ProxyArgs<float> &, cudaStream_t);
template void launch_charge2proxy_dispatch<double>(int, const Charge2ProxyArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
