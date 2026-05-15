#ifndef DMK_CUDA_CHARGE2PROXY_KERNELS_CUH
#define DMK_CUDA_CHARGE2PROXY_KERNELS_CUH

// Per-group charge2proxy: accumulates Chebyshev-polynomial-weighted source
// charges into the parent box's upward proxy slot.
//
//   proxy[i,j,k,d] += sum_s charge[d,s] * T_i(x_s) * T_j(y_s) * T_k(z_s)
//
// where (x_s, y_s, z_s) are scaled coordinates of source point s relative to
// the group's center box. Source points are processed in chunks of CHUNK to
// bound shared-memory usage; per chunk the block cooperatively populates
// poly_x/y/z and the chunk's charges in shared memory, then strides over the
// n_order^3 * n_charge_dim output cells to accumulate.

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>


#include <dmk/cuda/charge2proxy_kernels.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

template <typename Real>
__device__ inline void chebyshev_fill_strided(Real x, Real *out, int n, int stride) {
    Real t0 = Real{1};
    out[0] = t0;
    if (n <= 1)
        return;
    Real t1 = x;
    out[stride] = t1;
    for (int k = 2; k < n; ++k) {
        const Real t2 = Real{2} * x * t1 - t0;
        out[k * stride] = t2;
        t0 = t1;
        t1 = t2;
    }
}

template <
    typename Real,
    int N_ORDER,
    int I_TILE = 3,
    int J_TILE = 6,
    int K_TILE = 2
>
__global__ void Charge2ProxyByGroup3DKernel_GemmMicroKTile(
    Charge2ProxyArgs<Real> a,
    const int *__restrict__ group_perm
) {
    constexpr int CHUNK = 128;
    constexpr int LD = CHUNK + 1;
    constexpr int DIM = 3;

    constexpr int N  = N_ORDER;
    const int NC = a.n_charge_dim;
    constexpr int N2 = N * N;
    constexpr int N3 = N2 * N;

    constexpr int I_TILES = (N + I_TILE - 1) / I_TILE;
    constexpr int J_TILES = (N + J_TILE - 1) / J_TILE;
    constexpr int K_TILES = (N + K_TILE - 1) / K_TILE;

    const int NTILES = I_TILES * J_TILES * K_TILES * NC;

    extern __shared__ unsigned char shared_raw[];

    Real *poly_x = reinterpret_cast<Real *>(shared_raw);
    Real *poly_y = poly_x + N * LD;
    Real *poly_z = poly_y + N * LD;
    Real *charges_s = poly_z + N * LD;

    const int logical_g = blockIdx.x;
    if (logical_g >= a.n_groups)
        return;

    const int g = group_perm ? group_perm[logical_g] : logical_g;

    const int center_box = a.center_boxes[g];
    const int level = a.levels[g];
    const int sb_off = a.src_box_flat_offsets[g];
    const int n_src_boxes = a.n_src_boxes_per_group[g];

    const Real cx = a.centers[center_box * DIM + 0];
    const Real cy = a.centers[center_box * DIM + 1];
    const Real cz = a.centers[center_box * DIM + 2];

    const Real scale = a.inv_box_scale[level];

    Real *__restrict__ proxy =
        a.proxy_flat + a.proxy_offsets[center_box];

    for (int sbi = 0; sbi < n_src_boxes; ++sbi) {
        const int sb = a.src_boxes_flat[sb_off + sbi];
        const int n_src = a.src_counts_owned[sb];

        if (n_src == 0)
            continue;

        const Real *__restrict__ r_src =
            a.r_src_owned + a.r_src_owned_offsets[sb];

        const Real *__restrict__ charge =
            a.charge_owned + a.charge_owned_offsets[sb];

        for (int s_base = 0; s_base < n_src; s_base += CHUNK) {
            const int n_in_chunk =
                (s_base + CHUNK > n_src) ? (n_src - s_base) : CHUNK;

            // Build Chebyshev tables for this source chunk.
            for (int s = threadIdx.x; s < n_in_chunk; s += blockDim.x) {
                const int sp = s_base + s;

                const Real x = (r_src[sp * DIM + 0] - cx) * scale;
                const Real y = (r_src[sp * DIM + 1] - cy) * scale;
                const Real z = (r_src[sp * DIM + 2] - cz) * scale;

                chebyshev_fill_strided<Real>(x, poly_x + s, N, LD);
                chebyshev_fill_strided<Real>(y, poly_y + s, N, LD);
                chebyshev_fill_strided<Real>(z, poly_z + s, N, LD);
            }

            for (int t = threadIdx.x; t < NC * n_in_chunk; t += blockDim.x) {
                const int s = t / NC;
                const int d = t - s * NC;
                const int sp = s_base + s;

                charges_s[d * LD + s] = charge[d + sp * NC];
            }

            __syncthreads();

            for (int tile = threadIdx.x; tile < NTILES; tile += blockDim.x) {
                int idx = tile;

                const int it = idx % I_TILES;
                idx /= I_TILES;

                const int jt = idx % J_TILES;
                idx /= J_TILES;

                const int kt = idx % K_TILES;
                idx /= K_TILES;

                const int d = idx;

                const int i0 = it * I_TILE;
                const int j0 = jt * J_TILE;
                const int k0 = kt * K_TILE;

                Real acc[K_TILE][I_TILE][J_TILE];

                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    #pragma unroll
                    for (int r = 0; r < I_TILE; ++r) {
                        #pragma unroll
                        for (int c = 0; c < J_TILE; ++c) {
                            acc[kk][r][c] = Real{0};
                        }
                    }
                }

                for (int s = 0; s < n_in_chunk; ++s) {
                    Real xreg[I_TILE];
                    Real yreg[J_TILE];
                    Real zqreg[K_TILE];

                    #pragma unroll
                    for (int r = 0; r < I_TILE; ++r) {
                        const int i = i0 + r;

                        if (i < N)
                            xreg[r] = poly_x[i * LD + s];
                        else
                            xreg[r] = Real{0};
                    }

                    #pragma unroll
                    for (int c = 0; c < J_TILE; ++c) {
                        const int j = j0 + c;

                        if (j < N)
                            yreg[c] = poly_y[j * LD + s];
                        else
                            yreg[c] = Real{0};
                    }

                    const Real q = charges_s[d * LD + s];

                    #pragma unroll
                    for (int kk = 0; kk < K_TILE; ++kk) {
                        const int k = k0 + kk;

                        if (k < N)
                            zqreg[kk] = poly_z[k * LD + s] * q;
                        else
                            zqreg[kk] = Real{0};
                    }

                   
                    #pragma unroll
                    for (int kk = 0; kk < K_TILE; ++kk) {
                        #pragma unroll
                        for (int c = 0; c < J_TILE; ++c) {
                            const Real yzq = yreg[c] * zqreg[kk];

                            #pragma unroll
                            for (int r = 0; r < I_TILE; ++r) {
                                acc[kk][r][c] =
                                    fma(xreg[r], yzq, acc[kk][r][c]);
                            }
                        }
                    }
                }

                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    const int k = k0 + kk;

                    if (k < N) {
                        #pragma unroll
                        for (int r = 0; r < I_TILE; ++r) {
                            const int i = i0 + r;

                            if (i < N) {
                                #pragma unroll
                                for (int c = 0; c < J_TILE; ++c) {
                                    const int j = j0 + c;

                                    if (j < N) {
                                        proxy[i + j * N + k * N2 + d * N3] += acc[kk][r][c];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            __syncthreads();
        }
    }
}

template <typename Real, int TILE_I>
__global__ void Charge2ProxyByGroup3DKernel(Charge2ProxyArgs<Real> a,
    const int *__restrict__ group_perm) {
    constexpr int CHUNK = 32;
    constexpr int LD = CHUNK + 1;
    constexpr int DIM = 3;

    const int N = TILE_I;
    const int NC = a.n_charge_dim;
    const int N2 = N * N;
    const int N3 = N2 * N;

    const int NI_TILES = (N + TILE_I - 1) / TILE_I;
    const int NTILES = NI_TILES * N * N * NC;

    extern __shared__ unsigned char shared_raw[];
    Real *poly_x = reinterpret_cast<Real *>(shared_raw);
    Real *poly_y = poly_x + N * LD;
    Real *poly_z = poly_y + N * LD;
    Real *charges_s = poly_z + N * LD;
    
    const int logical_g = blockIdx.x;
    if (logical_g >= a.n_groups)
        return;

    const int g = group_perm ? group_perm[logical_g] : logical_g;
    const int center_box = a.center_boxes[g];
    const int level = a.levels[g];
    const int sb_off = a.src_box_flat_offsets[g];
    const int n_src_boxes = a.n_src_boxes_per_group[g];

    const Real cx = a.centers[center_box * DIM + 0];
    const Real cy = a.centers[center_box * DIM + 1];
    const Real cz = a.centers[center_box * DIM + 2];
    const Real scale = a.inv_box_scale[level];

    Real *proxy = a.proxy_flat + a.proxy_offsets[center_box];

    for (int sbi = 0; sbi < n_src_boxes; ++sbi) {
        const int sb = a.src_boxes_flat[sb_off + sbi];
        const int n_src = a.src_counts_owned[sb];
        if (n_src == 0)
            continue;

        const Real *r_src = a.r_src_owned + a.r_src_owned_offsets[sb];
        const Real *charge = a.charge_owned + a.charge_owned_offsets[sb];

        for (int s_base = 0; s_base < n_src; s_base += CHUNK) {
            const int n_in_chunk =
                (s_base + CHUNK > n_src) ? (n_src - s_base) : CHUNK;

            for (int s = threadIdx.x; s < n_in_chunk; s += blockDim.x) {
                const int sp = s_base + s;

                const Real x = (r_src[sp * DIM + 0] - cx) * scale;
                const Real y = (r_src[sp * DIM + 1] - cy) * scale;
                const Real z = (r_src[sp * DIM + 2] - cz) * scale;

                chebyshev_fill_strided<Real>(x, poly_x + s, N, LD);
                chebyshev_fill_strided<Real>(y, poly_y + s, N, LD);
                chebyshev_fill_strided<Real>(z, poly_z + s, N, LD);
            }

            for (int t = threadIdx.x; t < NC * n_in_chunk; t += blockDim.x) {
                const int s = t / NC;
                const int d = t % NC;
                const int sp = s_base + s;
                charges_s[d * LD + s] = charge[d + sp * NC];
            }

            __syncthreads();
            #pragma unroll
            for (int tile = threadIdx.x; tile < NTILES; tile += blockDim.x) {
                int idx = tile;

                const int it = idx % NI_TILES;
                idx /= NI_TILES;

                const int j = idx % N;
                idx /= N;

                const int k = idx % N;
                idx /= N;

                const int d = idx;

                const int i0 = it * TILE_I;

                Real acc[TILE_I];

#pragma unroll
                for (int r = 0; r < TILE_I; ++r)
                    acc[r] = Real{0};

                for (int s = 0; s < n_in_chunk; ++s) {
                    const Real yzq =
                        poly_y[j * LD + s] *
                        poly_z[k * LD + s] *
                        charges_s[d * LD + s];

#pragma unroll
                    for (int r = 0; r < TILE_I; ++r) {
                        const int i = i0 + r;
                        if (i < N) {
                            acc[r] = fma(poly_x[i * LD + s], yzq, acc[r]);
                        }
                    }
                }

#pragma unroll
                for (int r = 0; r < TILE_I; ++r) {
                    const int i = i0 + r;
                    if (i < N) {
                        proxy[i + j * N + k * N2 + d * N3] += acc[r];
                    }
                }
            }

            __syncthreads();
        }
    }
}

template <typename Real>
__global__ void ComputeCharge2ProxyGroupWorkKernel(
    Charge2ProxyArgs<Real> a,
    long long *__restrict__ work_keys,
    int *__restrict__ group_perm
) {
    constexpr int CHUNK = 32;

    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= a.n_groups)
        return;

    const int sb_off = a.src_box_flat_offsets[g];
    const int n_src_boxes = a.n_src_boxes_per_group[g];

    long long total_sources = 0;
    long long total_chunks = 0;

    for (int sbi = 0; sbi < n_src_boxes; ++sbi) {
        const int sb = a.src_boxes_flat[sb_off + sbi];
        const int n_src = a.src_counts_owned[sb];

        total_sources += n_src;
        total_chunks += (n_src + CHUNK - 1) / CHUNK;
    }

    work_keys[g] = total_sources * 1024LL + total_chunks;
    group_perm[g] = g;
}


struct PositiveWork {
    __host__ __device__
    bool operator()(long long x) const {
        return x > 0;
    }
};

struct Charge2ProxyGroupOrder {
    thrust::device_vector<long long> work_keys;
    thrust::device_vector<int> group_perm;
    int n_active_groups = 0;
};

template <typename Real>
inline void build_charge2proxy_group_order(
    const Charge2ProxyArgs<Real> &args,
    Charge2ProxyGroupOrder &order,
    cudaStream_t stream
) {
    order.work_keys.resize(args.n_groups);
    order.group_perm.resize(args.n_groups);
    order.n_active_groups = 0;

    if (args.n_groups == 0)
        return;

    constexpr int block_size = 256;
    const int grid_size = (args.n_groups + block_size - 1) / block_size;

    ComputeCharge2ProxyGroupWorkKernel<Real>
        <<<grid_size, block_size, 0, stream>>>(
            args,
            thrust::raw_pointer_cast(order.work_keys.data()),
            thrust::raw_pointer_cast(order.group_perm.data())
        );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("ComputeCharge2ProxyGroupWorkKernel: ") +
            cudaGetErrorString(err)
        );
    }

    auto policy = thrust::cuda::par.on(stream);

    thrust::sort_by_key(
        policy,
        order.work_keys.begin(),
        order.work_keys.end(),
        order.group_perm.begin(),
        thrust::greater<long long>()
    );

    order.n_active_groups = static_cast<int>(
        thrust::count_if(
            policy,
            order.work_keys.begin(),
            order.work_keys.end(),
            PositiveWork{}
        )
    );
}

template <
    typename Real,
    int N_ORDER,
    int I_TILE = 3,
    int J_TILE = 6,
    int K_TILE = 2
>
inline void launch_charge2proxy_3d_gemm_micro_ktile_impl(
    const Charge2ProxyArgs<Real> &args,
    const int *group_perm,
    int n_launch_groups,
    cudaStream_t stream
) {
    if (args.n_groups == 0 || n_launch_groups == 0)
        return;

    constexpr int block_size = 256;
    constexpr int CHUNK = 128;
    constexpr int LD = CHUNK + 1;
    const int NC = args.n_charge_dim;

    const std::size_t shared_bytes = (static_cast<std::size_t>(3) * N_ORDER * LD + static_cast<std::size_t>(NC) * LD) * sizeof(Real);

    Charge2ProxyByGroup3DKernel_GemmMicroKTile<Real, N_ORDER, I_TILE, J_TILE, K_TILE> <<<n_launch_groups, block_size, shared_bytes, stream>>>(args, group_perm);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_charge2proxy_3d_gemm_micro_ktile_impl: ") +
            cudaGetErrorString(err) +
            " (n_order=" + std::to_string(N_ORDER) +
            " I_TILE=" + std::to_string(I_TILE) +
            " J_TILE=" + std::to_string(J_TILE) +
            " K_TILE=" + std::to_string(K_TILE) +
            " shared_bytes=" + std::to_string(shared_bytes) +
            " n_launch_groups=" + std::to_string(n_launch_groups) + ")"
        );
    }
}

template <typename Real>
inline void launch_charge2proxy_3d(
    const Charge2ProxyArgs<Real> &args,
    const int *group_perm,
    int n_launch_groups,
    cudaStream_t stream
) {
    if (args.n_groups == 0 || n_launch_groups == 0)
        return;

#define DISPATCH_N_ORDER(N)                                                   \
    case N:                                                                   \
        launch_charge2proxy_3d_gemm_micro_ktile_impl<                         \
            Real, N, 3, 3, 4                                                  \
        >(args, group_perm, n_launch_groups, stream);                         \
        break

    switch (args.n_order) {        
        DISPATCH_N_ORDER(5);
        DISPATCH_N_ORDER(6);
        DISPATCH_N_ORDER(7);
        DISPATCH_N_ORDER(8);
        DISPATCH_N_ORDER(9);
        DISPATCH_N_ORDER(10);
        DISPATCH_N_ORDER(11);
        DISPATCH_N_ORDER(12);
        DISPATCH_N_ORDER(13);
        DISPATCH_N_ORDER(14);
        DISPATCH_N_ORDER(15);
        DISPATCH_N_ORDER(16);
        DISPATCH_N_ORDER(17);
        DISPATCH_N_ORDER(18);   
        DISPATCH_N_ORDER(19);

        default:
            throw std::runtime_error(
                "launch_charge2proxy_3d_gemm_micro_ktile: unsupported n_order=" +
                std::to_string(args.n_order)
            );
    }

#undef DISPATCH_N_ORDER
}

} // namespace dmk::cuda

#endif // DMK_CUDA_CHARGE2PROXY_KERNELS_CUH
