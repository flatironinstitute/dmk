// One block per box. Each block:
//   1. Copies the box's own pw_out into its slot of pw_in_pool.
//   2. Iterates over the box's neighbours; for each valid one, accumulates
//      pw_out[neighbour] * wpwshift[ind] (complex multiply, additive).
//
// Threads stride over the 2 * n_charge_dim * n_pw_modes reals in the slot.
// No shared memory needed — each thread reads/writes its own indices.

#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/shift_pw_kernels.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda {

using cuda_helpers::complx;

template <typename Real>
__device__ inline void ShiftPwByBoxBody(const ShiftPwArgs<Real> &a, int box_idx) {
    if (box_idx >= a.n_boxes_at_level)
        return;

    const int box = a.box_ids[box_idx];

    const int n_pw_modes = a.n_pw_modes;
    const int n_charge_dim = a.n_charge_dim;
    const int n_neighbors = a.n_neighbors;

    const bool box_is_leaf = (a.is_global_leaf[box] != 0);

    Real *pw_in_real = a.pw_in_pool + box_idx * a.pw_in_stride;
    complx<Real> *pw_in = reinterpret_cast<complx<Real> *>(pw_in_real);

    const long self_off = a.pw_out_offsets[box];
    const complx<Real> *self_pw =
        (self_off >= 0) ? reinterpret_cast<const complx<Real> *>(a.pw_out_flat + 2 * self_off) : nullptr;

    for (int d = 0; d < n_charge_dim; ++d) {
        const long d_base = (long)d * n_pw_modes;

        for (int m = threadIdx.x; m < n_pw_modes; m += blockDim.x) {
            complx<Real> acc;
            if (self_pw)
                acc = self_pw[d_base + m];
            else
                acc = complx<Real>{Real{0}, Real{0}};

            for (int npos = 0; npos < n_neighbors; ++npos) {
                const int neighbor = a.neighbors[(long)box * n_neighbors + npos];
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
                const complx<Real> *nbr_pw = reinterpret_cast<const complx<Real> *>(a.pw_out_flat + 2 * nbr_off);

                const complx<Real> z = nbr_pw[d_base + m];
                const Real sr = shift_r[m];
                const Real si = shift_i[m];

                acc.r += z.r * sr - z.i * si;
                acc.i += z.r * si + z.i * sr;
            }

            pw_in[d_base + m] = acc;
        }
    }
}

template <typename Real>
__global__ void ShiftPwByBoxKernel(ShiftPwArgs<Real> a) {
    ShiftPwByBoxBody<Real>(a, blockIdx.x);
}

template <typename Real>
__global__ void ShiftPwMultiLevelKernel(const ShiftPwArgs<Real> *args, int n_args) {
    const int arg_idx = blockIdx.y;
    if (arg_idx >= n_args)
        return;
    const ShiftPwArgs<Real> a = args[arg_idx];
    ShiftPwByBoxBody<Real>(a, blockIdx.x);
}

template <typename Real, int DIM>
void launch_shift_pw(const ShiftPwArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;
    constexpr int block_size = 512;
    ShiftPwByBoxKernel<Real><<<args.n_boxes_at_level, block_size, 0, stream>>>(args);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_shift_pw: ") + cudaGetErrorString(err));
}

template void launch_shift_pw<float, 2>(const ShiftPwArgs<float> &, cudaStream_t);
template void launch_shift_pw<float, 3>(const ShiftPwArgs<float> &, cudaStream_t);
template void launch_shift_pw<double, 2>(const ShiftPwArgs<double> &, cudaStream_t);
template void launch_shift_pw<double, 3>(const ShiftPwArgs<double> &, cudaStream_t);

template <typename Real, int DIM>
void launch_shift_pw_multilevel(const std::vector<ShiftPwArgs<Real>> &args_h, ShiftPwArgs<Real> *d_args_scratch,
                                cudaStream_t stream) {
    if (args_h.empty())
        return;

    int max_boxes = 0;
    for (const auto &a : args_h)
        max_boxes = std::max(max_boxes, a.n_boxes_at_level);
    if (max_boxes == 0)
        return;

    DMK_CHECK_CUDA(cudaMemcpyAsync(d_args_scratch, args_h.data(), args_h.size() * sizeof(ShiftPwArgs<Real>),
                                   cudaMemcpyHostToDevice, stream));

    constexpr int block_size = 256;
    dim3 grid(max_boxes, static_cast<int>(args_h.size()), 1);
    ShiftPwMultiLevelKernel<Real><<<grid, block_size, 0, stream>>>(d_args_scratch, static_cast<int>(args_h.size()));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_shift_pw_multilevel: ") + cudaGetErrorString(err));
}

template void launch_shift_pw_multilevel<float, 2>(const std::vector<ShiftPwArgs<float>> &, ShiftPwArgs<float> *,
                                                   cudaStream_t);
template void launch_shift_pw_multilevel<float, 3>(const std::vector<ShiftPwArgs<float>> &, ShiftPwArgs<float> *,
                                                   cudaStream_t);
template void launch_shift_pw_multilevel<double, 2>(const std::vector<ShiftPwArgs<double>> &, ShiftPwArgs<double> *,
                                                    cudaStream_t);
template void launch_shift_pw_multilevel<double, 3>(const std::vector<ShiftPwArgs<double>> &, ShiftPwArgs<double> *,
                                                    cudaStream_t);

} // namespace dmk::cuda
