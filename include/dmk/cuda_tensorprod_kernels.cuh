#ifndef DMK_CUDA_TENSORPROD_KERNELS_CUH
#define DMK_CUDA_TENSORPROD_KERNELS_CUH

// Per-pair tensorprod: child_proxy[i_x, j_y, k_z] +=
//   sum_{ix, jy, kz} parent_proxy[ix, jy, kz]
//                  * umat_x[i_x, ix] * umat_y[j_y, jy] * umat_z[k_z, kz]
//
// Decomposed into 3 sequential axis transforms with shared-memory ping-pong
// buffers (ff and ff2). For n_order=10, ff/ff2 are 8KB each (16KB total) —
// comfortable in 48KB shared mem. Block size 128, threads stride over the
// n_order^DIM output cells.

#include <dmk/cuda_tensorprod_kernels.hpp>

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real>
__global__ void TensorprodByPair3DKernel(TensorprodArgs<Real> a) {
    extern __shared__ unsigned char shared_raw[];
    Real *ff = reinterpret_cast<Real *>(shared_raw);
    const int N = a.n_order;
    const int N2 = N * N;
    const int N3 = N * N * N;
    Real *ff2 = ff + N3;

    const int pair_idx = blockIdx.x;
    if (pair_idx >= a.n_pairs)
        return;

    const int parent = a.parents[pair_idx];
    const int child = a.children[pair_idx];
    const int oct = a.child_octants[pair_idx];

    const Real *p2c_oct = a.p2c_flat + oct * 3 * N2;
    const Real *umat_x = p2c_oct + 0 * N2;
    const Real *umat_y = p2c_oct + 1 * N2;
    const Real *umat_z = p2c_oct + 2 * N2;

    const Real *parent_base = a.proxy_flat + a.proxy_offsets[parent];
    Real *child_base = a.proxy_flat + a.proxy_offsets[child];

    for (int d = 0; d < a.n_charge_dim; ++d) {
        const Real *fin = parent_base + d * N3;
        Real *fout = child_base + d * N3;

        // Phase 1: ff(i, j, kout) = sum_k fin(i, j, k) * umat_z(kout, k)
        for (int t = threadIdx.x; t < N3; t += blockDim.x) {
            const int i = t % N;
            const int j = (t / N) % N;
            const int kout = t / N2;
            Real acc = Real{0};
            for (int k = 0; k < N; ++k)
                acc += fin[i + j * N + k * N2] * umat_z[kout + k * N];
            ff[i + j * N + kout * N2] = acc;
        }
        __syncthreads();

        // Phase 2: ff2(i, jout, kz) = sum_j ff(i, j, kz) * umat_y(jout, j)
        for (int t = threadIdx.x; t < N3; t += blockDim.x) {
            const int i = t % N;
            const int jout = (t / N) % N;
            const int kz = t / N2;
            Real acc = Real{0};
            for (int j = 0; j < N; ++j)
                acc += ff[i + j * N + kz * N2] * umat_y[jout + j * N];
            ff2[i + jout * N + kz * N2] = acc;
        }
        __syncthreads();

        // Phase 3: fout(iout, jy, kz) += sum_i ff2(i, jy, kz) * umat_x(iout, i)
        for (int t = threadIdx.x; t < N3; t += blockDim.x) {
            const int iout = t % N;
            const int jy = (t / N) % N;
            const int kz = t / N2;
            Real acc = Real{0};
            for (int i = 0; i < N; ++i)
                acc += ff2[i + jy * N + kz * N2] * umat_x[iout + i * N];
            fout[iout + jy * N + kz * N2] += acc;
        }
        __syncthreads();
    }
}

template <typename Real>
inline void launch_tensorprod_3d(const TensorprodArgs<Real> &args, cudaStream_t stream) {
    if (args.n_pairs == 0)
        return;
    constexpr int block_size = 128;
    const std::size_t shared_bytes = 2 * args.n_order * args.n_order * args.n_order * sizeof(Real);
    TensorprodByPair3DKernel<Real><<<args.n_pairs, block_size, shared_bytes, stream>>>(args);
}

} // namespace dmk::cuda

#endif // DMK_CUDA_TENSORPROD_KERNELS_CUH
