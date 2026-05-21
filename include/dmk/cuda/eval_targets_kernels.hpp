#ifndef DMK_CUDA_EVAL_TARGETS_KERNELS_HPP
#define DMK_CUDA_EVAL_TARGETS_KERNELS_HPP

// Per-box proxy::eval_targets kernel. launch_eval_targets dispatches on
// (EVAL_LEVEL, N_CHARGE_DIM); DIM=3 is the only path reachable today since
// the other GPU contexts gate on it.

#include <dmk/cuda/eval_targets_kernelargs.hpp>

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_eval_targets(int eval_level, int n_charge_dim, const EvalTargetsArgs<Real> &args, cudaStream_t stream);

template <typename Real>
struct SelfCorrectionArgs {
    const int *direct_work;          // [n_direct_work]
    const Real *correction_factors;  // [n_direct_work]
    const int *src_counts_owned;     // [n_boxes]
    const int *src_counts_halo;      // [n_boxes]
    const Real *charge_halo;         // flat, AoS stride = n_input_dim
    const long *charge_halo_offsets; // [n_boxes]
    Real *pot_src;                   // d_pot_src_eval, AoS stride = pot_stride
    const long *pot_src_offsets;     // [n_boxes]
    int n_direct_work;
    int n_input_dim; // kernel_input_dim; also the charge AoS stride
    int pot_stride;  // kernel_output_dim_src; the pot AoS stride
};

template <typename Real>
void launch_self_correction(const SelfCorrectionArgs<Real> &args, cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_EVAL_TARGETS_KERNELS_HPP
