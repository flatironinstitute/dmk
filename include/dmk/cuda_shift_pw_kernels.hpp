#ifndef DMK_CUDA_SHIFT_PW_KERNELS_HPP
#define DMK_CUDA_SHIFT_PW_KERNELS_HPP

// Per-level GPU shift_planewave: one block per box that does PW work at the
// level. The block initialises its slot in `pw_in_pool` from the box's own
// `pw_out`, then loops over its (up to 3^DIM) neighbours, accumulating
// `pw_out[neighbour] * wpwshift[ind]` into `pw_in_pool[slot]`. The next
// per-level kernel (pw_to_proxy) reads from this pool.
//
// Storage convention:
//   pw_out[box]    : interleaved complex (real, imag, real, imag, ...).
//                    Size n_charge_dim * n_pw_modes complex per box.
//   pw_in_pool     : interleaved complex, [slot * stride .. slot*stride + stride).
//                    stride = n_charge_dim * n_pw_modes * 2 reals.
//   wpwshift[ind]  : SoA — n_pw_modes reals (real parts), then n_pw_modes
//                    reals (imag parts). One per neighbour-direction slot,
//                    n_neighbors slots per level. (= 3^DIM)
//
// Neighbour index → wpwshift index:
//     ind = n_neighbors - 1 - position_in_nbr_array
// (Mirrors the CPU loop in form_eval_expansions.)

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real>
struct ShiftPwArgs {
    int n_boxes_at_level = 0; // gridDim.x
    int n_neighbors = 0;      // 3^DIM
    int n_charge_dim = 0;     // n_tables_down
    int n_pw_modes = 0;       // n_pw * n_pw * n_pw2 for 3D
    long pw_in_stride = 0;    // = n_charge_dim * n_pw_modes * 2 (reals per slot)

    // Per-level box list: which boxes get processed by which block.
    const int *box_ids = nullptr; // [n_boxes_at_level]

    // Tree topology (uploaded once at shared state ctor).
    const int *neighbors = nullptr;                // [n_boxes * n_neighbors]; -1 = invalid
    const long *pw_out_offsets = nullptr;          // [n_boxes]; -1 = no pw_out for this box
    const unsigned char *is_global_leaf = nullptr; // [n_boxes]

    // PW data.
    const Real *pw_out_flat = nullptr; // interleaved complex
    const Real *wpwshift = nullptr;    // SoA, this level's; n_neighbors * n_pw_modes * 2 reals

    // Output: per-block scratch. pw_in_pool[blockIdx.x * pw_in_stride .. ).
    Real *pw_in_pool = nullptr;
};

template <typename Real>
void launch_shift_pw_dispatch(int dim, const ShiftPwArgs<Real> &args, cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_SHIFT_PW_KERNELS_HPP
