// Runtime dispatch for the per-level shift_planewave kernel.

#include <dmk/cuda_shift_pw_kernels.cuh>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

template <typename Real>
void launch_shift_pw_dispatch(int dim, const ShiftPwArgs<Real> &args, cudaStream_t stream) {
    if (dim == 3) {
        launch_shift_pw_3d<Real>(args, stream);
        return;
    }
    throw std::runtime_error("CUDA shift_planewave: dim=" + std::to_string(dim) + " not supported (only 3D for now)");
}

template void launch_shift_pw_dispatch<float>(int, const ShiftPwArgs<float> &, cudaStream_t);
template void launch_shift_pw_dispatch<double>(int, const ShiftPwArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
