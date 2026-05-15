// Runtime dispatch for the per-block proxy2pw kernel.

#include <dmk/cuda/proxy2pw_kernels.cuh>

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

template <typename Real>
void launch_proxy2pw_multilevel_dispatch(
    int dim,
    const std::vector<Proxy2PwArgs<Real>> &pa_h,
    cudaStream_t stream) {
    if (dim != 3) {
        throw std::runtime_error(
            "CUDA proxy2pw multilevel: dim=" + std::to_string(dim) +
            " not supported (only 3D for now)");
    }

    if (pa_h.empty())
        return;

    int max_boxes = 0;
    int max_n_order = 0;
    int max_n_pw = 0;

    for (const auto &pa : pa_h) {
        max_boxes = std::max(max_boxes, pa.n_boxes_at_level);
        max_n_order = std::max(max_n_order, pa.n_order);
        max_n_pw = std::max(max_n_pw, pa.n_pw);
    }

    if (max_boxes == 0)
        return;

    Proxy2PwArgs<Real> *d_args = nullptr;

    cudaMallocAsync(
        reinterpret_cast<void **>(&d_args),
        pa_h.size() * sizeof(Proxy2PwArgs<Real>),
        stream);

    cudaMemcpyAsync(
        d_args,
        pa_h.data(),
        pa_h.size() * sizeof(Proxy2PwArgs<Real>),
        cudaMemcpyHostToDevice,
        stream);

    launch_proxy2pw_multilevel_3d<Real>(
        d_args,
        static_cast<int>(pa_h.size()),
        max_boxes,
        max_n_order,
        max_n_pw,
        stream);
   

    cudaFreeAsync(d_args, stream);
}

template void launch_proxy2pw_multilevel_dispatch<float>(
    int,
    const std::vector<Proxy2PwArgs<float>> &,
    cudaStream_t);

template void launch_proxy2pw_multilevel_dispatch<double>(
    int,
    const std::vector<Proxy2PwArgs<double>> &,
    cudaStream_t);



} // namespace dmk::cuda
