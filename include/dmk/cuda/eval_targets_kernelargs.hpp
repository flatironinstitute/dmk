#ifndef DMK_CUDA_EVAL_TARGETS_KERNELARGS_HPP
#define DMK_CUDA_EVAL_TARGETS_KERNELARGS_HPP

namespace dmk::cuda {

template <typename Real>
struct EvalTargetsArgs {
    int n_eval_boxes = 0; // |eval_targets_box_list|
    int n_order = 0;      // tree's Chebyshev order

    const int *eval_targets_box_list = nullptr; // [n_eval_boxes]
    const int *box_levels = nullptr;            // [n_boxes]
    const Real *sc_per_level = nullptr;         // [n_levels], 2/boxsize

    const Real *proxy_flat = nullptr;    // proxy_coeffs_downward
    const long *proxy_offsets = nullptr; // [n_boxes]
    const Real *centers = nullptr;       // [n_boxes * DIM]

    // Either (r_src, src_counts) for the pot_src side or
    // (r_trg, trg_counts) for the pot_trg side.
    const Real *r_target_flat = nullptr;
    const long *r_target_offsets = nullptr;
    const int *target_counts = nullptr;

    Real *pot_flat = nullptr;          // d_pot_*_eval
    const long *pot_offsets = nullptr; // [n_boxes]
};

template <typename Real>
struct SelfCorrectionArgs {
    const int *direct_work;         // [n_direct_work]
    const Real *correction_factors; // [n_direct_work]
    const int *src_counts;          // [n_boxes]
    const Real *charge;             // flat, AoS stride = n_input_dim
    const long *charge_offsets;     // [n_boxes]
    Real *pot_src;                  // d_pot_src_eval, AoS stride = pot_stride
    const long *pot_src_offsets;    // [n_boxes]
    int n_direct_work;
    int n_input_dim; // kernel_input_dim; also the charge AoS stride
    int pot_stride;  // kernel_output_dim_src; the pot AoS stride
};

} // namespace dmk::cuda

#endif // DMK_CUDA_EVAL_TARGETS_KERNELARGS_HPP
