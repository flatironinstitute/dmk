#ifndef DMK_CUDA_PW_TO_PROXY_KERNELS_CUH
#define DMK_CUDA_PW_TO_PROXY_KERNELS_CUH

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

template <typename Real>
struct alignas(2 * sizeof(Real)) complx {
    Real r;
    Real i;
};

template <typename Real>
__device__ __forceinline__ complx<Real> complx_zero() {
    return complx<Real>{Real{0}, Real{0}};
}

template <typename Real>
__device__ __forceinline__ complx<Real> complx_load(const Real *__restrict__ p) {
    return complx<Real>{p[0], p[1]};
}

template <typename Real>
__device__ __forceinline__ void complx_madd(complx<Real> &acc, const complx<Real> a, const complx<Real> b) {
    // acc += a * b
    acc.r = fma(a.r, b.r, acc.r);
    acc.r = fma(-a.i, b.i, acc.r);

    acc.i = fma(a.r, b.i, acc.i);
    acc.i = fma(a.i, b.r, acc.i);
}

template <typename Real>
__device__ __forceinline__ Real complx_real_madd(Real acc, const complx<Real> a, const complx<Real> b) {
    // acc += real(a * b)
    acc = fma(a.r, b.r, acc);
    acc = fma(-a.i, b.i, acc);
    return acc;
}

template <typename Real, int K1_TILE = 18, int COL_REG = 2, int K2_TILE = 2, int K3_TILE = 6, int KR_TILE = 6>
__device__ __forceinline__ void PwToProxyBody(const PwToProxyArgs<Real> &a, const int box_idx) {
    if (box_idx >= a.n_boxes_at_level)
        return;

    const int box = a.box_ids[box_idx];

    const long proxy_off = a.proxy_offsets[box];
    if (proxy_off < 0)
        return;

    const int n_pw = a.n_pw;
    const int n_pw2 = a.n_pw2;
    const int n_pw_half = n_pw / 2;
    const int n_order = a.n_order;
    const int n_order2 = n_order * n_order;
    const int n_order3 = n_order2 * n_order;
    const int n_pw_modes = n_pw * n_pw * n_pw2;

    const int k_pad = ((n_order + 3) / 4) * 4;
    const int phase1_cols = n_pw * n_pw2;

    extern __shared__ __align__(16) unsigned char shared_raw[];

    complx<Real> *__restrict__ smem = reinterpret_cast<complx<Real> *>(shared_raw);

    // s_A_T[m, k] = pw2poly(k, m)
    complx<Real> *__restrict__ s_A_T = smem;

    // s_F[kr, c], c = m2 + m3 * n_pw
    complx<Real> *__restrict__ s_F = s_A_T + static_cast<long>(n_pw) * k_pad;

    // s_G[kr, m3, k2], k2 fastest
    complx<Real> *__restrict__ s_G = s_F + static_cast<long>(K1_TILE) * phase1_cols;

    // Stage A once per CTA.
    for (int idx = threadIdx.x; idx < n_pw * k_pad; idx += blockDim.x) {
        const int m = idx / k_pad;
        const int k = idx - m * k_pad;

        complx<Real> z = complx_zero<Real>();

        if (k < n_order) {
            z = complx_load((a.pw2poly + 2 * (static_cast<long>(k) * n_pw + m)));
        }

        s_A_T[idx] = z;
    }

    __syncthreads();

    const Real *__restrict__ pw_in_box = a.pw_in_pool + static_cast<long>(box_idx) * a.pw_in_stride;

    Real *__restrict__ proxy_box = a.proxy_flat + proxy_off;

    const int k2_tiles = (n_order + K2_TILE - 1) / K2_TILE;
    const int k3_tiles = (n_order + K3_TILE - 1) / K3_TILE;
    const int kr_tiles = (K1_TILE + KR_TILE - 1) / KR_TILE;

    for (int d = 0; d < a.n_charge_dim; ++d) {
        const Real *__restrict__ pw_in_d = pw_in_box + 2 * static_cast<long>(d) * n_pw_modes;

        Real *__restrict__ proxy_d = proxy_box + static_cast<long>(d) * n_order3;

        for (int k1_base = 0; k1_base < n_order; k1_base += K1_TILE) {
            // phase 1

            const int col_tiles = (phase1_cols + COL_REG - 1) / COL_REG;

            for (int tile = threadIdx.x; tile < col_tiles; tile += blockDim.x) {
                const int c_base = tile * COL_REG;

                complx<Real> acc[K1_TILE][COL_REG];

#pragma unroll
                for (int kr = 0; kr < K1_TILE; ++kr) {
#pragma unroll
                    for (int cr = 0; cr < COL_REG; ++cr) {
                        acc[kr][cr] = complx_zero<Real>();
                    }
                }

                for (int m1 = 0; m1 < n_pw; ++m1) {
                    complx<Real> p[COL_REG];

#pragma unroll
                    for (int cr = 0; cr < COL_REG; ++cr) {
                        const int c = c_base + cr;

                        if (c < phase1_cols) {
                            const long pidx = static_cast<long>(m1) + static_cast<long>(c) * n_pw;

                            p[cr] = complx_load(pw_in_d + 2 * pidx);
                            // p[cr] = complx<Real>{(pw_in_d + 2 * pidx)[0], (pw_in_d + 2 * pidx)[1]};
                        } else {
                            p[cr] = complx_zero<Real>();
                        }
                    }

#pragma unroll
                    for (int kr = 0; kr < K1_TILE; ++kr) {
                        const int k1 = k1_base + kr;

                        complx<Real> a1 = complx_zero<Real>();

                        if (k1 < n_order) {
                            a1 = s_A_T[m1 * k_pad + k1];
                        }

#pragma unroll
                        for (int cr = 0; cr < COL_REG; ++cr) {
                            complx_madd(acc[kr][cr], a1, p[cr]);
                        }
                    }
                }

#pragma unroll
                for (int kr = 0; kr < K1_TILE; ++kr) {
                    const int k1 = k1_base + kr;

                    if (k1 < n_order) {
#pragma unroll
                        for (int cr = 0; cr < COL_REG; ++cr) {
                            const int c = c_base + cr;

                            if (c < phase1_cols) {
                                s_F[static_cast<long>(kr) * phase1_cols + c] = acc[kr][cr];
                            }
                        }
                    }
                }
            }

            __syncthreads();

            // phase 2

            const int phase2_tiles = kr_tiles * k2_tiles * n_pw2;

            for (int tile = threadIdx.x; tile < phase2_tiles; tile += blockDim.x) {
                int x = tile;

                const int ktile = x % k2_tiles;
                x /= k2_tiles;

                const int m3 = x % n_pw2;
                x /= n_pw2;

                const int kr_tile = x;

                const int k2_base = ktile * K2_TILE;
                const int kr_base = kr_tile * KR_TILE;

                complx<Real> acc[KR_TILE][K2_TILE];

#pragma unroll
                for (int rr = 0; rr < KR_TILE; ++rr) {
#pragma unroll
                    for (int k2r = 0; k2r < K2_TILE; ++k2r) {
                        acc[rr][k2r] = complx_zero<Real>();
                    }
                }

                for (int m2 = 0; m2 < n_pw; ++m2) {
                    const int c = m2 + m3 * n_pw;

                    complx<Real> f[KR_TILE];

#pragma unroll
                    for (int rr = 0; rr < KR_TILE; ++rr) {
                        const int kr = kr_base + rr;
                        const int k1 = k1_base + kr;

                        if (kr < K1_TILE && k1 < n_order) {
                            f[rr] = s_F[static_cast<long>(kr) * phase1_cols + c];
                        } else {
                            f[rr] = complx_zero<Real>();
                        }
                    }

                    complx<Real> a2[K2_TILE];

#pragma unroll
                    for (int k2r = 0; k2r < K2_TILE; ++k2r) {
                        const int k2 = k2_base + k2r;

                        if (k2 < n_order) {
                            a2[k2r] = s_A_T[m2 * k_pad + k2];
                        } else {
                            a2[k2r] = complx_zero<Real>();
                        }
                    }

#pragma unroll
                    for (int rr = 0; rr < KR_TILE; ++rr) {
#pragma unroll
                        for (int k2r = 0; k2r < K2_TILE; ++k2r) {
                            complx_madd(acc[rr][k2r], a2[k2r], f[rr]);
                        }
                    }
                }

                const Real scale = (m3 >= n_pw_half) ? Real{0.5} : Real{1};

#pragma unroll
                for (int rr = 0; rr < KR_TILE; ++rr) {
                    const int kr = kr_base + rr;
                    const int k1 = k1_base + kr;

                    if (kr < K1_TILE && k1 < n_order) {
#pragma unroll
                        for (int k2r = 0; k2r < K2_TILE; ++k2r) {
                            const int k2 = k2_base + k2r;

                            if (k2 < n_order) {
                                complx<Real> v = acc[rr][k2r];

                                v.r *= scale;
                                v.i *= scale;

                                s_G[static_cast<long>(kr) * n_pw2 * n_order + static_cast<long>(m3) * n_order + k2] = v;
                            }
                        }
                    }
                }
            }

            __syncthreads();

            // phase 3
            const int phase3_tiles = kr_tiles * k3_tiles * n_order;

            for (int tile = threadIdx.x; tile < phase3_tiles; tile += blockDim.x) {
                int x = tile;

                const int k2 = x % n_order;
                x /= n_order;

                const int k3tile = x % k3_tiles;
                x /= k3_tiles;

                const int kr_tile = x;

                const int kr_base = kr_tile * KR_TILE;
                const int k3_base = k3tile * K3_TILE;

                Real acc[KR_TILE][K3_TILE];

#pragma unroll
                for (int rr = 0; rr < KR_TILE; ++rr) {
#pragma unroll
                    for (int k3r = 0; k3r < K3_TILE; ++k3r) {
                        acc[rr][k3r] = Real{0};
                    }
                }

                for (int m3 = 0; m3 < n_pw2; ++m3) {
                    complx<Real> g[KR_TILE];

#pragma unroll
                    for (int rr = 0; rr < KR_TILE; ++rr) {
                        const int kr = kr_base + rr;
                        const int k1 = k1_base + kr;

                        if (kr < K1_TILE && k1 < n_order) {
                            g[rr] = s_G[static_cast<long>(kr) * n_pw2 * n_order + static_cast<long>(m3) * n_order + k2];
                        } else {
                            g[rr] = complx_zero<Real>();
                        }
                    }

                    complx<Real> a3[K3_TILE];

#pragma unroll
                    for (int k3r = 0; k3r < K3_TILE; ++k3r) {
                        const int k3 = k3_base + k3r;

                        if (k3 < n_order && m3 < n_pw) {
                            a3[k3r] = s_A_T[m3 * k_pad + k3];
                        } else {
                            a3[k3r] = complx_zero<Real>();
                        }
                    }

#pragma unroll
                    for (int rr = 0; rr < KR_TILE; ++rr) {
#pragma unroll
                        for (int k3r = 0; k3r < K3_TILE; ++k3r) {
                            acc[rr][k3r] = complx_real_madd(acc[rr][k3r], g[rr], a3[k3r]);
                        }
                    }
                }

#pragma unroll
                for (int rr = 0; rr < KR_TILE; ++rr) {
                    const int kr = kr_base + rr;
                    const int k1 = k1_base + kr;

                    if (kr < K1_TILE && k1 < n_order) {
#pragma unroll
                        for (int k3r = 0; k3r < K3_TILE; ++k3r) {
                            const int k3 = k3_base + k3r;

                            if (k3 < n_order) {
                                Real *__restrict__ out =
                                    proxy_d + k1 + static_cast<long>(k2) * n_order + static_cast<long>(k3) * n_order2;

                                const Real value = Real{2} * acc[rr][k3r];

                                *out += value;
                            }
                        }
                    }
                }
            }

            __syncthreads();
        }
    }
}

template <typename Real, int K1_TILE = 18, int COL_REG = 2, int K2_TILE = 2, int K3_TILE = 6, int KR_TILE = 6>
__global__ void PwToProxyMultiLevelKernel(const PwToProxyArgs<Real> *__restrict__ args, int n_args) {
    const int box_idx = blockIdx.x;
    const int arg_idx = blockIdx.y;

    if (arg_idx >= n_args)
        return;

    const PwToProxyArgs<Real> a = args[arg_idx];

    PwToProxyBody<Real, K1_TILE, COL_REG, K2_TILE, K3_TILE, KR_TILE>(a, box_idx);
}

template <typename Real, int K1_TILE = 18, int COL_REG = 2, int K2_TILE = 2, int K3_TILE = 6, int KR_TILE = 6>
__global__ void PwToProxyByBoxKernel(PwToProxyArgs<Real> a) {
    const int box_idx = blockIdx.x;

    PwToProxyBody<Real, K1_TILE, COL_REG, K2_TILE, K3_TILE, KR_TILE>(a, box_idx);
}

template <typename Real, int K1_TILE>
inline std::size_t pw_to_proxy_shared_bytes(int max_n_pw, int max_n_pw2, int max_n_order) {
    const int max_k_pad = ((max_n_order + 3) / 4) * 4;

    const int max_phase1_cols = max_n_pw * max_n_pw2;

    const std::size_t complex_count = static_cast<std::size_t>(max_n_pw) * max_k_pad +
                                      static_cast<std::size_t>(K1_TILE) * max_phase1_cols +
                                      static_cast<std::size_t>(K1_TILE) * max_n_pw2 * max_n_order;

    return complex_count * sizeof(complx<Real>);
}

template <typename Real, int K1_TILE = 6, int COL_REG = 2, int K2_TILE = 2, int K3_TILE = 3, int KR_TILE = 6>
inline void launch_pw_to_proxy_3d(const PwToProxyArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 128;

    const std::size_t shared_bytes = pw_to_proxy_shared_bytes<Real, K1_TILE>(args.n_pw, args.n_pw2, args.n_order);

    if (shared_bytes > 48 * 1024) {
        cudaError_t attr_err =
            cudaFuncSetAttribute(PwToProxyByBoxKernel<Real, K1_TILE, COL_REG, K2_TILE, K3_TILE, KR_TILE>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));

        if (attr_err != cudaSuccess) {
            throw std::runtime_error(std::string("launch_pw_to_proxy_3d_outer_product_nosp: "
                                                 "cudaFuncSetAttribute failed: ") +
                                     cudaGetErrorString(attr_err) + " shared_bytes=" + std::to_string(shared_bytes));
        }
    }

    PwToProxyByBoxKernel<Real, K1_TILE, COL_REG, K2_TILE, K3_TILE, KR_TILE>
        <<<args.n_boxes_at_level, block_size, shared_bytes, stream>>>(args);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("launch_pw_to_proxy_3d_outer_product_nosp: ") + cudaGetErrorString(err) +
                                 " shared_bytes=" + std::to_string(shared_bytes) +
                                 " n_pw=" + std::to_string(args.n_pw) + " n_pw2=" + std::to_string(args.n_pw2) +
                                 " n_order=" + std::to_string(args.n_order));
    }
}

template <typename Real, int K1_TILE = 18, int COL_REG = 1, int K2_TILE = 2, int K3_TILE = 3, int KR_TILE = 9>
inline void launch_pw_to_proxy_multilevel_3d(const PwToProxyArgs<Real> *d_args, int n_args, int max_boxes, int max_n_pw,
                                             int max_n_pw2, int max_n_order, cudaStream_t stream) {
    if (n_args == 0 || max_boxes == 0)
        return;

    constexpr int block_size = 256;

    const std::size_t shared_bytes = pw_to_proxy_shared_bytes<Real, K1_TILE>(max_n_pw, max_n_pw2, max_n_order);

    if (shared_bytes > 48 * 1024) {
        cudaError_t attr_err =
            cudaFuncSetAttribute(PwToProxyMultiLevelKernel<Real, K1_TILE, COL_REG, K2_TILE, K3_TILE, KR_TILE>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));

        if (attr_err != cudaSuccess) {
            throw std::runtime_error(std::string("launch_pw_to_proxy_multilevel_3d: "
                                                 "cudaFuncSetAttribute failed: ") +
                                     cudaGetErrorString(attr_err) + " shared_bytes=" + std::to_string(shared_bytes));
        }
    }

    dim3 grid(max_boxes, n_args, 1);

    PwToProxyMultiLevelKernel<Real, K1_TILE, COL_REG, K2_TILE, K3_TILE, KR_TILE>
        <<<grid, block_size, shared_bytes, stream>>>(d_args, n_args);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("launch_pw_to_proxy_multilevel_3d: ") + cudaGetErrorString(err) +
                                 " shared_bytes=" + std::to_string(shared_bytes) +
                                 " max_n_pw=" + std::to_string(max_n_pw) + " max_n_pw2=" + std::to_string(max_n_pw2) +
                                 " max_n_order=" + std::to_string(max_n_order));
    }
}

} // namespace dmk::cuda

#endif // DMK_CUDA_PW_TO_PROXY_KERNELS_CUH