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

template <typename Real>
__global__ void Charge2ProxyByGroup3DKernel(Charge2ProxyArgs<Real> a) {
    constexpr int CHUNK = 32;
    constexpr int DIM = 3;
    const int N = a.n_order;
    const int NC = a.n_charge_dim;
    const int N2 = N * N;
    const int N3 = N2 * N;
    const int NCELLS = N3 * NC;

    extern __shared__ unsigned char shared_raw[];
    Real *poly_x = reinterpret_cast<Real *>(shared_raw);
    Real *poly_y = poly_x + N * CHUNK;
    Real *poly_z = poly_y + N * CHUNK;
    Real *charges_s = poly_z + N * CHUNK;

    const int g = blockIdx.x;
    if (g >= a.n_groups)
        return;

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

        const Real *r_src = a.r_src_owned + a.r_src_owned_offsets[sb];   // F-major (DIM, n_src)
        const Real *charge = a.charge_owned + a.charge_owned_offsets[sb]; // F-major (NC, n_src)

        for (int s_base = 0; s_base < n_src; s_base += CHUNK) {
            const int n_in_chunk = (s_base + CHUNK > n_src) ? (n_src - s_base) : CHUNK;

            for (int s = threadIdx.x; s < n_in_chunk; s += blockDim.x) {
                const int sp = s_base + s;
                const Real x = (r_src[sp * DIM + 0] - cx) * scale;
                const Real y = (r_src[sp * DIM + 1] - cy) * scale;
                const Real z = (r_src[sp * DIM + 2] - cz) * scale;
                chebyshev_fill_strided<Real>(x, poly_x + s, N, CHUNK);
                chebyshev_fill_strided<Real>(y, poly_y + s, N, CHUNK);
                chebyshev_fill_strided<Real>(z, poly_z + s, N, CHUNK);
            }
            for (int t = threadIdx.x; t < NC * n_in_chunk; t += blockDim.x) {
                const int s = t / NC;
                const int d = t % NC;
                const int sp = s_base + s;
                charges_s[d * CHUNK + s] = charge[d + sp * NC];
            }
            __syncthreads();

            for (int t = threadIdx.x; t < NCELLS; t += blockDim.x) {
                int idx = t;
                const int i = idx % N;
                idx /= N;
                const int j = idx % N;
                idx /= N;
                const int k = idx % N;
                idx /= N;
                const int d = idx;

                Real acc = Real{0};
                for (int s = 0; s < n_in_chunk; ++s) {
                    acc += poly_x[i * CHUNK + s] * poly_y[j * CHUNK + s] * poly_z[k * CHUNK + s] *
                           charges_s[d * CHUNK + s];
                }
                proxy[i + j * N + k * N2 + d * N3] += acc;
            }
            __syncthreads();
        }
    }
}

template <typename Real>
inline void launch_charge2proxy_3d(const Charge2ProxyArgs<Real> &args, cudaStream_t stream) {
    if (args.n_groups == 0)
        return;
    constexpr int block_size = 256;
    constexpr int CHUNK = 32;
    const std::size_t shared_bytes =
        ((std::size_t)3 * args.n_order * CHUNK + (std::size_t)args.n_charge_dim * CHUNK) * sizeof(Real);
    Charge2ProxyByGroup3DKernel<Real><<<args.n_groups, block_size, shared_bytes, stream>>>(args);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_charge2proxy_3d: ") + cudaGetErrorString(err) +
                                 " (n_order=" + std::to_string(args.n_order) +
                                 " n_charge_dim=" + std::to_string(args.n_charge_dim) +
                                 " shared_bytes=" + std::to_string(shared_bytes) + ")");
}

} // namespace dmk::cuda

#endif // DMK_CUDA_CHARGE2PROXY_KERNELS_CUH
