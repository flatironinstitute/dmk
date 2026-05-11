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


#include <dmk/cuda_charge2proxy_kernels.hpp>

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

template <typename Real, int TILE_I>
__global__ void Charge2ProxyByGroup3DKernel(Charge2ProxyArgs<Real> a,
    const int *__restrict__ group_perm) {
    constexpr int CHUNK = 32;
    constexpr int LD = CHUNK + 1;
    constexpr int DIM = 3;

    const int N = a.n_order;
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


template <typename Real, int TILE_I>
inline void launch_charge2proxy_3d_impl(
    const Charge2ProxyArgs<Real> &args,
    const int *group_perm,
    int n_launch_groups,
    cudaStream_t stream
) {
    if (args.n_groups == 0 || n_launch_groups == 0)
        return;

    constexpr int block_size = 256;
    constexpr int CHUNK = 32;
    constexpr int LD = CHUNK + 1; // Avoid shared-memory bank conflicts.

    // TILE_I is the compile-time equivalent of args.n_order.
    const std::size_t shared_bytes = ((std::size_t)3 * TILE_I * LD + (std::size_t)args.n_charge_dim * LD) * sizeof(Real);

    Charge2ProxyByGroup3DKernel<Real, TILE_I>
        <<<n_launch_groups, block_size, shared_bytes, stream>>>(
            args,
            group_perm
        );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_charge2proxy_3d: ") +
            cudaGetErrorString(err) +
            " (n_order=" + std::to_string(args.n_order) +
            " TILE_I=" + std::to_string(TILE_I) +
            " n_charge_dim=" + std::to_string(args.n_charge_dim) +
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
        launch_charge2proxy_3d_impl<Real, N>(                                 \
            args, group_perm, n_launch_groups, stream                         \
        );                                                                    \
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
        DISPATCH_N_ORDER(20);
        DISPATCH_N_ORDER(21);
        DISPATCH_N_ORDER(22);
        DISPATCH_N_ORDER(23);
        DISPATCH_N_ORDER(24);
        DISPATCH_N_ORDER(25);
        DISPATCH_N_ORDER(26);
        DISPATCH_N_ORDER(27);
        DISPATCH_N_ORDER(28);
        DISPATCH_N_ORDER(29);
        DISPATCH_N_ORDER(30);
        DISPATCH_N_ORDER(31);
        DISPATCH_N_ORDER(32);
        DISPATCH_N_ORDER(33);
        DISPATCH_N_ORDER(34);
        DISPATCH_N_ORDER(35);
        DISPATCH_N_ORDER(36);
        DISPATCH_N_ORDER(37);
        DISPATCH_N_ORDER(38);
        DISPATCH_N_ORDER(39);
        DISPATCH_N_ORDER(40);
        DISPATCH_N_ORDER(41);
        DISPATCH_N_ORDER(42);
        DISPATCH_N_ORDER(43);
        DISPATCH_N_ORDER(44);
        DISPATCH_N_ORDER(45);
        DISPATCH_N_ORDER(46); //HELP

        default:
            throw std::runtime_error(
                "launch_charge2proxy_3d: unsupported n_order=" +
                std::to_string(args.n_order)
            );
    }

#undef DISPATCH_N_ORDER
}

} // namespace dmk::cuda

#endif // DMK_CUDA_CHARGE2PROXY_KERNELS_CUH
