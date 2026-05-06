#ifndef DMK_CUDA_SHIFT_PW_KERNELS_CUH
#define DMK_CUDA_SHIFT_PW_KERNELS_CUH

// One block per box. Each block:
//   1. Copies the box's own pw_out into its slot of pw_in_pool.
//   2. Iterates over the box's neighbours; for each valid one, accumulates
//      pw_out[neighbour] * wpwshift[ind] (complex multiply, additive).
//
// Threads stride over the 2 * n_charge_dim * n_pw_modes reals in the slot.
// No shared memory needed — each thread reads/writes its own indices.

#include <dmk/cuda_shift_pw_kernels.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

template <typename Real>
__global__ void ShiftPwByBoxKernel(ShiftPwArgs<Real> a) {
    const int box_idx = blockIdx.x;
    if (box_idx >= a.n_boxes_at_level)
        return;

    const int box = a.box_ids[box_idx];
    Real *pw_in = a.pw_in_pool + (long)box_idx * a.pw_in_stride;
    const int n_pw_modes = a.n_pw_modes;
    const int n_charge_dim = a.n_charge_dim;
    const int total_reals = 2 * n_charge_dim * n_pw_modes;

    // 1. pw_in = pw_out[box]
    {
        const long off = a.pw_out_offsets[box];
        if (off < 0) {
            for (int i = threadIdx.x; i < total_reals; i += blockDim.x)
                pw_in[i] = Real{0};
        } else {
            const Real *src = a.pw_out_flat + off * 2;
            for (int i = threadIdx.x; i < total_reals; i += blockDim.x)
                pw_in[i] = src[i];
        }
    }
    __syncthreads();

    // 2. For each neighbour, accumulate pw_out[neighbour] * wpwshift[ind].
    const int n_neighbors = a.n_neighbors;
    const bool box_is_leaf = a.is_global_leaf[box];
    for (int npos = 0; npos < n_neighbors; ++npos) {
        const int neighbor = a.neighbors[(long)box * n_neighbors + npos];
        if (neighbor < 0 || neighbor == box)
            continue;
        if (box_is_leaf && a.is_global_leaf[neighbor])
            continue;
        const long nbr_off = a.pw_out_offsets[neighbor];
        if (nbr_off < 0)
            continue;

        const int ind = n_neighbors - 1 - npos;
        const Real *shift_r = a.wpwshift + (long)ind * n_pw_modes * 2;
        const Real *shift_i = shift_r + n_pw_modes;
        const Real *nbr_pw = a.pw_out_flat + nbr_off * 2;

        for (int d = 0; d < n_charge_dim; ++d) {
            for (int m = threadIdx.x; m < n_pw_modes; m += blockDim.x) {
                const long pw_idx = (long)d * n_pw_modes * 2 + 2 * m;
                const Real ar = nbr_pw[pw_idx];
                const Real ai = nbr_pw[pw_idx + 1];
                const Real sr = shift_r[m];
                const Real si = shift_i[m];
                pw_in[pw_idx] += ar * sr - ai * si;
                pw_in[pw_idx + 1] += ar * si + ai * sr;
            }
        }
        __syncthreads();
    }
}

template <typename Real>
inline void launch_shift_pw_3d(const ShiftPwArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;
    constexpr int block_size = 128;
    ShiftPwByBoxKernel<Real><<<args.n_boxes_at_level, block_size, 0, stream>>>(args);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_shift_pw_3d: ") + cudaGetErrorString(err));
}

} // namespace dmk::cuda

#endif // DMK_CUDA_SHIFT_PW_KERNELS_CUH
