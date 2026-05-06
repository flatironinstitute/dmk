#ifndef DMK_CUDA_PW_TO_PROXY_KERNELS_CUH
#define DMK_CUDA_PW_TO_PROXY_KERNELS_CUH

// See cuda_pw_to_proxy_kernels.hpp for math + storage convention.

#include <dmk/cuda_pw_to_proxy_kernels.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

template <typename Real>
__global__ void PwToProxyByBoxKernel(PwToProxyArgs<Real> a) {
    extern __shared__ unsigned char shared_raw[];
    Real *ff_slab = reinterpret_cast<Real *>(shared_raw);                      // 2 * n_pw * n_pw2 reals
    Real *ff2_slab = ff_slab + 2 * (long)a.n_pw * a.n_pw2;                     // 2 * n_order * n_pw2 reals

    const int box_idx = blockIdx.x;
    if (box_idx >= a.n_boxes_at_level)
        return;
    const int box = a.box_ids[box_idx];

    const Real *pw_in = a.pw_in_pool + (long)box_idx * a.pw_in_stride;
    Real *proxy_dst = a.proxy_flat + a.proxy_offsets[box];

    const int n_pw = a.n_pw;
    const int n_pw2 = a.n_pw2;
    const int n_pw_half = n_pw / 2;
    const int n_order = a.n_order;
    const int n_pw_modes = n_pw * n_pw * n_pw2;
    const int n_order2 = n_order * n_order;
    const int n_order3 = n_order2 * n_order;

    for (int d = 0; d < a.n_charge_dim; ++d) {
        const Real *pw_in_d = pw_in + 2 * (long)d * n_pw_modes;
        Real *proxy_d = proxy_dst + (long)d * n_order3;

        for (int k1 = 0; k1 < n_order; ++k1) {
            const Real *poly_k1 = a.pw2poly + 2 * (long)k1 * n_pw;

            // Phase 1: ff[m2, m3] = sum_m1 pw2poly[m1, k1] * pw_in[m1, m2, m3, d]
            for (int t = threadIdx.x; t < n_pw * n_pw2; t += blockDim.x) {
                const int m2 = t % n_pw;
                const int m3 = t / n_pw;
                Real sum_r = Real{0}, sum_i = Real{0};
                for (int m1 = 0; m1 < n_pw; ++m1) {
                    const long pw_idx = 2 * ((long)m1 + (long)m2 * n_pw + (long)m3 * n_pw * n_pw);
                    const Real pw_r = pw_in_d[pw_idx];
                    const Real pw_i = pw_in_d[pw_idx + 1];
                    const Real poly_r = poly_k1[2 * m1];
                    const Real poly_i = poly_k1[2 * m1 + 1];
                    sum_r += poly_r * pw_r - poly_i * pw_i;
                    sum_i += poly_r * pw_i + poly_i * pw_r;
                }
                ff_slab[2 * (m2 + m3 * n_pw)] = sum_r;
                ff_slab[2 * (m2 + m3 * n_pw) + 1] = sum_i;
            }
            __syncthreads();

            // Phase 2: ff2[k2, m3] = halve(m3) * sum_m2 pw2poly[m2, k2] * ff[m2, m3]
            for (int t = threadIdx.x; t < n_order * n_pw2; t += blockDim.x) {
                const int k2 = t % n_order;
                const int m3 = t / n_order;
                const Real *poly_k2 = a.pw2poly + 2 * (long)k2 * n_pw;
                Real sum_r = Real{0}, sum_i = Real{0};
                for (int m2 = 0; m2 < n_pw; ++m2) {
                    const Real ff_r = ff_slab[2 * (m2 + m3 * n_pw)];
                    const Real ff_i = ff_slab[2 * (m2 + m3 * n_pw) + 1];
                    const Real poly_r = poly_k2[2 * m2];
                    const Real poly_i = poly_k2[2 * m2 + 1];
                    sum_r += poly_r * ff_r - poly_i * ff_i;
                    sum_i += poly_r * ff_i + poly_i * ff_r;
                }
                const Real scale = (m3 >= n_pw_half) ? Real{0.5} : Real{1};
                ff2_slab[2 * (k2 + m3 * n_order)] = scale * sum_r;
                ff2_slab[2 * (k2 + m3 * n_order) + 1] = scale * sum_i;
            }
            __syncthreads();

            // Phase 3: proxy[k1, k2, k3, d] += 2 * Re( sum_m3 ff2[k2, m3] * pw2poly[m3, k3] )
            for (int t = threadIdx.x; t < n_order2; t += blockDim.x) {
                const int k2 = t % n_order;
                const int k3 = t / n_order;
                const Real *poly_k3 = a.pw2poly + 2 * (long)k3 * n_pw;
                Real sum_r = Real{0};
                for (int m3 = 0; m3 < n_pw2; ++m3) {
                    const Real ff2_r = ff2_slab[2 * (k2 + m3 * n_order)];
                    const Real ff2_i = ff2_slab[2 * (k2 + m3 * n_order) + 1];
                    const Real poly_r = poly_k3[2 * m3];
                    const Real poly_i = poly_k3[2 * m3 + 1];
                    sum_r += ff2_r * poly_r - ff2_i * poly_i;
                }
                proxy_d[k1 + (long)k2 * n_order + (long)k3 * n_order2] += Real{2} * sum_r;
            }
            __syncthreads();
        }
    }
}

template <typename Real>
inline void launch_pw_to_proxy_3d(const PwToProxyArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;
    constexpr int block_size = 128;
    const std::size_t shared_bytes =
        sizeof(Real) * 2 * (std::size_t)(args.n_pw * args.n_pw2 + args.n_order * args.n_pw2);
    PwToProxyByBoxKernel<Real><<<args.n_boxes_at_level, block_size, shared_bytes, stream>>>(args);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_pw_to_proxy_3d: ") + cudaGetErrorString(err));
}

} // namespace dmk::cuda

#endif // DMK_CUDA_PW_TO_PROXY_KERNELS_CUH
