// Runtime dispatch for tensorprod (proxy_downward parent → child).

#include <dmk/cuda_tensorprod_kernels.cuh>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

template <typename Real>
void launch_tensorprod_dispatch(int dim, const TensorprodArgs<Real> &args, cudaStream_t stream) {
    if (dim == 3) {
        launch_tensorprod_3d<Real>(args, stream);
        return;
    }
    throw std::runtime_error("CUDA tensorprod: dim=" + std::to_string(dim) + " not supported (only 3D for now)");
}

template void launch_tensorprod_dispatch<float>(int, const TensorprodArgs<float> &, cudaStream_t);
template void launch_tensorprod_dispatch<double>(int, const TensorprodArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
