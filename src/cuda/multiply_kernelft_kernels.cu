// Runtime dispatches for the per-block multiply_kernelFT family.

#include <dmk/cuda/multiply_kernelft_kernels.cuh>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

template <typename Real>
void launch_multiply_cd2p_dispatch(int dim, const MultiplyCd2pArgs<Real> &args, cudaStream_t stream) {
    if (dim == 3) {
        launch_multiply_cd2p_3d<Real>(args, stream);
        return;
    }
    throw std::runtime_error("CUDA multiply_cd2p: dim=" + std::to_string(dim) + " not supported (only 3D for now)");
}

template <typename Real>
void launch_multiply_stokeslet_3d_dispatch(int dim, const MultiplyStokeslet3DArgs<Real> &args, cudaStream_t stream) {
    if (dim == 3) {
        launch_multiply_stokeslet_3d_3d<Real>(args, stream);
        return;
    }
    throw std::runtime_error("CUDA multiply_stokeslet_3d: dim=" + std::to_string(dim) + " not supported");
}

template <typename Real>
void launch_multiply_stresslet_3d_dispatch(int dim, const MultiplyStresslet3DArgs<Real> &args, cudaStream_t stream) {
    if (dim == 3) {
        launch_multiply_stresslet_3d_3d<Real>(args, stream);
        return;
    }
    throw std::runtime_error("CUDA multiply_stresslet_3d: dim=" + std::to_string(dim) + " not supported");
}

template void launch_multiply_cd2p_dispatch<float>(int, const MultiplyCd2pArgs<float> &, cudaStream_t);
template void launch_multiply_cd2p_dispatch<double>(int, const MultiplyCd2pArgs<double> &, cudaStream_t);
template void launch_multiply_stokeslet_3d_dispatch<float>(int, const MultiplyStokeslet3DArgs<float> &, cudaStream_t);
template void launch_multiply_stokeslet_3d_dispatch<double>(int, const MultiplyStokeslet3DArgs<double> &, cudaStream_t);
template void launch_multiply_stresslet_3d_dispatch<float>(int, const MultiplyStresslet3DArgs<float> &, cudaStream_t);
template void launch_multiply_stresslet_3d_dispatch<double>(int, const MultiplyStresslet3DArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
