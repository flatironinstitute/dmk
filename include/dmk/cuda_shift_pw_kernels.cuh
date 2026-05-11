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
struct Vec2Traits;

template <>
struct Vec2Traits<float> {
    using Vec = float2;

    __device__ static Vec make(float x, float y) {
        return make_float2(x, y);
    }
};

template <>
struct Vec2Traits<double> {
    using Vec = double2;

    __device__ static Vec make(double x, double y) {
        return make_double2(x, y);
    }
};

template <typename Real>
__device__ inline void ShiftPwByBoxBody(const ShiftPwArgs<Real> &a,
                                            int box_idx) {
    using Vec = typename Vec2Traits<Real>::Vec;

    if (box_idx >= a.n_boxes_at_level)
        return;

    const int box = a.box_ids[box_idx];

    const int n_pw_modes = a.n_pw_modes;
    const int n_charge_dim = a.n_charge_dim;
    const int n_neighbors = a.n_neighbors;

    const bool box_is_leaf = (a.is_global_leaf[box] != 0);

    Real *pw_in_real = a.pw_in_pool + (long)box_idx * a.pw_in_stride;

    Vec *pw_in = reinterpret_cast<Vec *>(pw_in_real);

    const long self_off = a.pw_out_offsets[box];

    const Vec *self_pw = (self_off >= 0) ? reinterpret_cast<const Vec *>(a.pw_out_flat + 2 * self_off) : nullptr;

    for (int d = 0; d < n_charge_dim; ++d) {
        const long d_base = (long)d * n_pw_modes;

        for (int m = threadIdx.x; m < n_pw_modes; m += blockDim.x) {
            Vec acc;

            if (self_pw) {
                acc = self_pw[d_base + m];
            } else {
                acc = Vec2Traits<Real>::make(Real{0}, Real{0});
            }

            for (int npos = 0; npos < n_neighbors; ++npos) {
                const int neighbor =
                    a.neighbors[(long)box * n_neighbors + npos];

                if (neighbor < 0 || neighbor == box)
                    continue;

                if (box_is_leaf && a.is_global_leaf[neighbor] != 0)
                    continue;

                const long nbr_off = a.pw_out_offsets[neighbor];

                if (nbr_off < 0)
                    continue;

                const int ind = n_neighbors - 1 - npos;

                const Real *shift_r = a.wpwshift + (long)ind * n_pw_modes * 2;

                const Real *shift_i = shift_r + n_pw_modes;

                const Vec *nbr_pw = reinterpret_cast<const Vec *>(a.pw_out_flat + 2 * nbr_off);

                const Vec z = nbr_pw[d_base + m];

                const Real sr = shift_r[m];
                const Real si = shift_i[m];

                acc.x += z.x * sr - z.y * si;
                acc.y += z.x * si + z.y * sr;
            }

            pw_in[d_base + m] = acc;
        }
    }
}

template <typename Real>
__global__ void ShiftPwByBoxKernel(ShiftPwArgs<Real> a) {
    const int box_idx = blockIdx.x;
    ShiftPwByBoxBody<Real>(a, box_idx);
}

template <typename Real>
__global__ void ShiftPwMultiLevelKernel(const ShiftPwArgs<Real> *args,
                                        int n_args) {
    const int box_idx = blockIdx.x;
    const int arg_idx = blockIdx.y;

    if (arg_idx >= n_args)
        return;

    const ShiftPwArgs<Real> a = args[arg_idx];

    ShiftPwByBoxBody<Real>(a, box_idx);
}

template <typename Real>
inline void launch_shift_pw_3d(const ShiftPwArgs<Real> &args,
                               cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 512;

    ShiftPwByBoxKernel<Real>
        <<<args.n_boxes_at_level, block_size, 0, stream>>>(args);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_shift_pw_3d: ") +
            cudaGetErrorString(err));
    }
}

template <typename Real>
inline void launch_shift_pw_multilevel_3d(const ShiftPwArgs<Real> *d_args,
                                          int n_args,
                                          int max_boxes,
                                          cudaStream_t stream) {
    if (n_args == 0 || max_boxes == 0)
        return;

    constexpr int block_size = 256;

    dim3 grid(max_boxes, n_args, 1);

    ShiftPwMultiLevelKernel<Real>
        <<<grid, block_size, 0, stream>>>(d_args, n_args);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_shift_pw_multilevel_3d: ") +
            cudaGetErrorString(err));
    }
}

} // namespace dmk::cuda

#endif // DMK_CUDA_SHIFT_PW_KERNELS_CUH
