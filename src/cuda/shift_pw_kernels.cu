// Runtime dispatch for shift_pw kernels.

#include <dmk/cuda/shift_pw_kernels.hpp>
#include <dmk/cuda/shift_pw_kernels.cuh>
#include <dmk/cuda/helpers.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda {

template <typename Real>
void launch_shift_pw_dispatch(int dim,
                              const ShiftPwArgs<Real> &args,
                              cudaStream_t stream) {
    if (dim == 3) {
        launch_shift_pw_3d<Real>(args, stream);
        return;
    }

    throw std::runtime_error(
        "CUDA shift_pw: dim=" + std::to_string(dim) +
        " not supported (only 3D for now)");
}

template void launch_shift_pw_dispatch<float>(
    int,
    const ShiftPwArgs<float> &,
    cudaStream_t);

template void launch_shift_pw_dispatch<double>(
    int,
    const ShiftPwArgs<double> &,
    cudaStream_t);

template <typename Real>
void launch_shift_pw_multilevel_dispatch(
    int dim,
    const std::vector<ShiftPwArgs<Real>> &args_h,
    cudaStream_t stream) {
    if (dim != 3) {
        throw std::runtime_error(
            "CUDA shift_pw multilevel: dim=" + std::to_string(dim) +
            " not supported (only 3D for now)");
    }

    if (args_h.empty())
        return;

    int max_boxes = 0;

    for (const auto &a : args_h)
        max_boxes = std::max(max_boxes, a.n_boxes_at_level);

    if (max_boxes == 0)
        return;

    ShiftPwArgs<Real> *d_args = nullptr;
    void *tmp = nullptr;

    DMK_CHECK_CUDA(cudaMallocAsync(
        &tmp,
        args_h.size() * sizeof(ShiftPwArgs<Real>),
        stream));

    d_args = static_cast<ShiftPwArgs<Real> *>(tmp);

    
    DMK_CHECK_CUDA(cudaMemcpyAsync(
        d_args,
        args_h.data(),
        args_h.size() * sizeof(ShiftPwArgs<Real>),
        cudaMemcpyHostToDevice,
        stream));

    launch_shift_pw_multilevel_3d<Real>(
        d_args,
        static_cast<int>(args_h.size()),
        max_boxes,
        stream);


    DMK_CHECK_CUDA(cudaFreeAsync(d_args, stream));
}

template void launch_shift_pw_multilevel_dispatch<float>(
    int,
    const std::vector<ShiftPwArgs<float>> &,
    cudaStream_t);

template void launch_shift_pw_multilevel_dispatch<double>(
    int,
    const std::vector<ShiftPwArgs<double>> &,
    cudaStream_t);

} // namespace dmk::cuda
