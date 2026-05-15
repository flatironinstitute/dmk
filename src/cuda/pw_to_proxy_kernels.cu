// Runtime dispatch for pw_to_proxy kernels.

#include <dmk/cuda/pw_to_proxy_kernels.hpp>
#include <dmk/cuda/pw_to_proxy_kernels.cuh>
#include <dmk/cuda/helpers.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda {

template <typename Real>
void launch_pw_to_proxy_dispatch(int dim,
                                 const PwToProxyArgs<Real> &args,
                                 cudaStream_t stream) {
    if (dim == 3) {
        launch_pw_to_proxy_3d<Real>(args, stream);
        return;
    }

    throw std::runtime_error(
        "CUDA pw_to_proxy: dim=" + std::to_string(dim) +
        " not supported (only 3D for now)");
}

template void launch_pw_to_proxy_dispatch<float>(
    int,
    const PwToProxyArgs<float> &,
    cudaStream_t);

template void launch_pw_to_proxy_dispatch<double>(
    int,
    const PwToProxyArgs<double> &,
    cudaStream_t);

template <typename Real>
void launch_pw_to_proxy_multilevel_dispatch(
    int dim,
    const std::vector<PwToProxyArgs<Real>> &args_h,
    cudaStream_t stream) {
    if (dim != 3) {
        throw std::runtime_error(
            "CUDA pw_to_proxy multilevel: dim=" + std::to_string(dim) +
            " not supported (only 3D for now)");
    }

    if (args_h.empty())
        return;

    int max_boxes = 0;
    int max_n_pw = 0;
    int max_n_pw2 = 0;
    int max_n_order = 0;

    for (const auto &a : args_h) {
        max_boxes = std::max(max_boxes, a.n_boxes_at_level);
        max_n_pw = std::max(max_n_pw, a.n_pw);
        max_n_pw2 = std::max(max_n_pw2, a.n_pw2);
        max_n_order = std::max(max_n_order, a.n_order);
    }

    if (max_boxes == 0)
        return;

    PwToProxyArgs<Real> *d_args = nullptr;
    void *tmp = nullptr;

    DMK_CHECK_CUDA(cudaMallocAsync(
        &tmp,
        args_h.size() * sizeof(PwToProxyArgs<Real>),
        stream));

    d_args = static_cast<PwToProxyArgs<Real> *>(tmp);

    DMK_CHECK_CUDA(cudaMemcpyAsync(
        d_args,
        args_h.data(),
        args_h.size() * sizeof(PwToProxyArgs<Real>),
        cudaMemcpyHostToDevice,
        stream));

    launch_pw_to_proxy_multilevel_3d<Real>(
        d_args,
        static_cast<int>(args_h.size()),
        max_boxes,
        max_n_pw,
        max_n_pw2,
        max_n_order,
        stream);


    DMK_CHECK_CUDA(cudaFreeAsync(d_args, stream));
}

template void launch_pw_to_proxy_multilevel_dispatch<float>(
    int,
    const std::vector<PwToProxyArgs<float>> &,
    cudaStream_t);

template void launch_pw_to_proxy_multilevel_dispatch<double>(
    int,
    const std::vector<PwToProxyArgs<double>> &,
    cudaStream_t);

} // namespace dmk::cuda
