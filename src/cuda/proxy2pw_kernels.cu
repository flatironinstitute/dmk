// Math (per box, per charge_dim d, with proxy_in real and pw_out complex):
//   ff[i, j]   = sum_k    proxy(i, j, k, d) * poly2pw(m3, k)
//   ff2[i, m2] = sum_j    ff(i, j)          * poly2pw(m2, j)
//   pw[m1, m2, m3, d] = sum_i ff2(i, m2)    * poly2pw(m1, i)
//
// We serialize the outer loop over m3 (the z-axis pw index) so the working
// ff/ff2 slabs stay in shared memory: ff is n_order×n_order complex,
// ff2 is n_order×n_pw complex.

#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/proxy2pw_kernels.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda {

template <typename Real>
__device__ inline void Proxy2PwBody(const Proxy2PwArgs<Real> &a, int box_idx) {
    if (box_idx >= a.n_boxes_at_level)
        return;

    extern __shared__ unsigned char shared_raw[];
    Real *ff_slab = reinterpret_cast<Real *>(shared_raw);
    Real *ff2_slab = ff_slab + 2 * (long)a.n_order * a.n_order;

    const int box = a.box_ids[box_idx];

    const long src_off = a.proxy_offsets[box];
    if (src_off < 0)
        return;
    const Real *proxy = a.proxy_flat + src_off;

    const long dst_off_complex = a.dst_offsets ? a.dst_offsets[box] : (long)box_idx * a.dst_stride_complex;
    if (dst_off_complex < 0)
        return;
    Real *pw_dst = a.dst_flat + 2 * dst_off_complex;

    const int n_order = a.n_order;
    const int n_order2 = n_order * n_order;
    const int n_order3 = n_order2 * n_order;
    const int n_pw = a.n_pw;
    const int n_pw2 = a.n_pw2;
    const int n_pw_modes = n_pw * n_pw * n_pw2;

    for (int d = 0; d < a.n_charge_dim; ++d) {
        const Real *proxy_d = proxy + (long)d * n_order3;
        Real *pw_d = pw_dst + 2 * (long)d * n_pw_modes;

        for (int m3 = 0; m3 < n_pw2; ++m3) {
            // Phase 1: ff(i, j) = sum_k proxy(i, j, k, d) * poly2pw(m3, k).
            for (int t = threadIdx.x; t < n_order2; t += blockDim.x) {
                const int i = t % n_order;
                const int j = t / n_order;
                Real sum_r = Real{0}, sum_i = Real{0};
                for (int k = 0; k < n_order; ++k) {
                    const Real p = proxy_d[i + (long)j * n_order + (long)k * n_order2];
                    const Real qr = a.poly2pw[2 * (m3 + (long)k * n_pw)];
                    const Real qi = a.poly2pw[2 * (m3 + (long)k * n_pw) + 1];
                    sum_r += p * qr;
                    sum_i += p * qi;
                }
                ff_slab[2 * (i + j * n_order)] = sum_r;
                ff_slab[2 * (i + j * n_order) + 1] = sum_i;
            }
            __syncthreads();

            // Phase 2: ff2(i, m2) = sum_j ff(i, j) * poly2pw(m2, j).
            for (int t = threadIdx.x; t < n_order * n_pw; t += blockDim.x) {
                const int i = t % n_order;
                const int m2 = t / n_order;
                Real sum_r = Real{0}, sum_i = Real{0};
                for (int j = 0; j < n_order; ++j) {
                    const Real fr = ff_slab[2 * (i + j * n_order)];
                    const Real fi = ff_slab[2 * (i + j * n_order) + 1];
                    const Real qr = a.poly2pw[2 * (m2 + (long)j * n_pw)];
                    const Real qi = a.poly2pw[2 * (m2 + (long)j * n_pw) + 1];
                    sum_r += fr * qr - fi * qi;
                    sum_i += fr * qi + fi * qr;
                }
                ff2_slab[2 * (i + m2 * n_order)] = sum_r;
                ff2_slab[2 * (i + m2 * n_order) + 1] = sum_i;
            }
            __syncthreads();

            // Phase 3: pw(m1, m2, m3, d) = sum_i ff2(i, m2) * poly2pw(m1, i).
            for (int t = threadIdx.x; t < n_pw * n_pw; t += blockDim.x) {
                const int m1 = t % n_pw;
                const int m2 = t / n_pw;
                Real sum_r = Real{0}, sum_i = Real{0};
                for (int i = 0; i < n_order; ++i) {
                    const Real fr = ff2_slab[2 * (i + m2 * n_order)];
                    const Real fi = ff2_slab[2 * (i + m2 * n_order) + 1];
                    const Real qr = a.poly2pw[2 * (m1 + (long)i * n_pw)];
                    const Real qi = a.poly2pw[2 * (m1 + (long)i * n_pw) + 1];
                    sum_r += fr * qr - fi * qi;
                    sum_i += fr * qi + fi * qr;
                }
                const long flat = (long)m1 + (long)m2 * n_pw + (long)m3 * n_pw * n_pw;
                pw_d[2 * flat] = sum_r;
                pw_d[2 * flat + 1] = sum_i;
            }
            __syncthreads();
        }
    }
}

template <typename Real>
__global__ void Proxy2PwByBoxKernel(Proxy2PwArgs<Real> a) {
    Proxy2PwBody<Real>(a, blockIdx.x);
}

template <typename Real>
__global__ void Proxy2PwMultiLevelKernel(const Proxy2PwArgs<Real> *args, int n_args) {
    const int arg_idx = blockIdx.y;
    if (arg_idx >= n_args)
        return;
    Proxy2PwBody<Real>(args[arg_idx], blockIdx.x);
}

template <typename Real>
static void launch_proxy2pw_3d(const Proxy2PwArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 128;
    const std::size_t shared_bytes =
        sizeof(Real) * 2 * (std::size_t)(args.n_order * args.n_order + args.n_order * args.n_pw);

    Proxy2PwByBoxKernel<Real><<<args.n_boxes_at_level, block_size, shared_bytes, stream>>>(args);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_proxy2pw_3d: ") + cudaGetErrorString(err));
}

template <typename Real>
static void launch_proxy2pw_multilevel_3d(const Proxy2PwArgs<Real> *d_args, int n_args, int max_boxes, int n_order,
                                          int n_pw, cudaStream_t stream) {
    if (n_args == 0 || max_boxes == 0)
        return;

    constexpr int block_size = 128;
    const std::size_t shared_bytes =
        sizeof(Real) * 2 * ((std::size_t)n_order * n_order + (std::size_t)n_order * n_pw);

    dim3 grid(max_boxes, n_args, 1);
    Proxy2PwMultiLevelKernel<Real><<<grid, block_size, shared_bytes, stream>>>(d_args, n_args);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_proxy2pw_multilevel_3d: ") + cudaGetErrorString(err));
}

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
void launch_proxy2pw_multilevel_dispatch(int dim, const std::vector<Proxy2PwArgs<Real>> &pa_h,
                                         Proxy2PwArgs<Real> *d_args_scratch, cudaStream_t stream) {
    if (dim != 3)
        throw std::runtime_error("CUDA proxy2pw multilevel: dim=" + std::to_string(dim) +
                                 " not supported (only 3D for now)");
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

    DMK_CHECK_CUDA(cudaMemcpyAsync(d_args_scratch, pa_h.data(), pa_h.size() * sizeof(Proxy2PwArgs<Real>),
                                   cudaMemcpyHostToDevice, stream));

    launch_proxy2pw_multilevel_3d<Real>(d_args_scratch, static_cast<int>(pa_h.size()), max_boxes, max_n_order, max_n_pw,
                                        stream);
}

template void launch_proxy2pw_multilevel_dispatch<float>(int, const std::vector<Proxy2PwArgs<float>> &,
                                                         Proxy2PwArgs<float> *, cudaStream_t);
template void launch_proxy2pw_multilevel_dispatch<double>(int, const std::vector<Proxy2PwArgs<double>> &,
                                                          Proxy2PwArgs<double> *, cudaStream_t);

} // namespace dmk::cuda
