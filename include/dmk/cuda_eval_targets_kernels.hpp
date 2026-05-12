#ifndef DMK_CUDA_EVAL_TARGETS_KERNELS_HPP
#define DMK_CUDA_EVAL_TARGETS_KERNELS_HPP

// Host-includable header for the GPU per-box proxy::eval_targets kernel.
// EvalTargetsArgs<Real> is the bag of device pointers + small scalars; the
// orchestration code (cuda_eval_targets.cpp) fills it in once per launch.
//
// launch_eval_targets_dispatch<Real>(...) selects the right
// (DIM, EVAL_LEVEL, N_CHARGE_DIM) instantiation and queues the kernel on
// `stream`. Defined in src/cuda_eval_targets_kernels.cu.

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real>
struct EvalTargetsArgs {
    int n_eval_boxes = 0; // |eval_targets_box_list|
    int n_order = 0;      // tree's Chebyshev order (≤ MAX_N_ORDER)

    const int *eval_targets_box_list = nullptr; // [n_eval_boxes]
    const int *box_levels = nullptr;            // [n_boxes]
    const Real *sc_per_level = nullptr;         // [n_levels], 2/boxsize

    const Real *proxy_flat = nullptr;    // proxy_coeffs_downward
    const long *proxy_offsets = nullptr; // [n_boxes]
    const Real *centers = nullptr;       // [n_boxes * DIM]

    // Either (r_src_owned, src_counts_owned) for the pot_src side or
    // (r_trg_owned, trg_counts_owned) for the pot_trg side.
    const Real *r_target_flat = nullptr;
    const long *r_target_offsets = nullptr;
    const int *target_counts = nullptr;

    Real *pot_flat = nullptr;          // d_pot_*_eval
    const long *pot_offsets = nullptr; // [n_boxes]
};

template <typename Real>
void launch_eval_targets_dispatch(int dim, int eval_level, int n_charge_dim, const EvalTargetsArgs<Real> &args,
                                  cudaStream_t stream);

template <typename Real>
void launch_inplace_accumulate(Real *dst, const Real *src, std::size_t n, cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_EVAL_TARGETS_KERNELS_HPP
