// Per-pair tensorprod: dst[i_x, j_y, k_z, d] +=
//   sum_{ix, jy, kz} src[ix, jy, kz, d]
//                  * umat_x[i_x, ix] * umat_y[j_y, jy] * umat_z[k_z, kz]
//
// Decomposed into 3 sequential axis transforms with ping-pong buffers ff/ff2
// in a per-block slab of TensorprodArgs::scratch (global memory; the buffers
// are too large for shared at typical n_order). Block size 128, threads
// stride over the n_order^DIM output cells.

#include <dmk/cuda/tensorprod_kernels.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

template <typename Real>
__global__ void TensorprodByPair3DKernel(TensorprodArgs<Real> a) {
    const int N = a.n_order;
    const int N2 = N * N;
    const int N3 = N * N * N;

    const int pair_idx = blockIdx.x;
    if (pair_idx >= a.n_pairs)
        return;

    Real *ff = a.scratch + (long)pair_idx * a.scratch_stride;
    Real *ff2 = ff + N3;

    const int src_box = a.src_boxes[pair_idx];
    const int dst_box = a.dst_boxes[pair_idx];
    const int oct = a.child_octants[pair_idx];

    const Real *umat_oct = a.umat_flat + oct * 3 * N2;
    const Real *umat_x = umat_oct + 0 * N2;
    const Real *umat_y = umat_oct + 1 * N2;
    const Real *umat_z = umat_oct + 2 * N2;

    const Real *src_base = a.proxy_flat + a.proxy_offsets[src_box];
    Real *dst_base = a.proxy_flat + a.proxy_offsets[dst_box];

    for (int d = 0; d < a.n_charge_dim; ++d) {
        const Real *fin = src_base + d * N3;
        Real *fout = dst_base + d * N3;

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
            if (a.additive_atomic)
                atomicAdd(&fout[iout + jy * N + kz * N2], acc);
            else
                fout[iout + jy * N + kz * N2] += acc;
        }
        __syncthreads();
    }
}

template <typename Real>
static void launch_tensorprod_3d(const TensorprodArgs<Real> &args, cudaStream_t stream) {
    if (args.n_pairs == 0)
        return;
    constexpr int block_size = 512;
    TensorprodByPair3DKernel<Real><<<args.n_pairs, block_size, 0, stream>>>(args);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_tensorprod_3d: ") + cudaGetErrorString(err));
}

template <typename Real>
void launch_tensorprod_dispatch(int dim, const TensorprodArgs<Real> &args, cudaStream_t stream) {
    if (dim == 3) {
        launch_tensorprod_3d<Real>(args, stream);
        return;
    }
    throw std::runtime_error("CUDA tensorprod: dim=" + std::to_string(dim) + " not supported (only 3D for now)");
}

template void launch_tensorprod_dispatch<float>(int, const TensorprodArgs<float> &, cudaStream_t);
template void launch_tensorprod_dispatch<double>(int, const TensorprodArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
