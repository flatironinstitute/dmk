#ifndef DMK_CUDA_SELF_CORRECTION_KERNELARGS_HPP
#define DMK_CUDA_SELF_CORRECTION_KERNELARGS_HPP

namespace dmk::cuda {

template <typename Real>
struct SelfCorrectionArgs {
    const int *direct_work;         // [n_direct_work]
    const Real *correction_factors; // [n_direct_work]
    const int *src_counts;          // [n_boxes]
    const Real *charge;             // flat, AoS stride = n_input_dim
    const long *charge_offsets;     // [n_boxes]
    Real *pot_src;                  // d_pot_direct_src, AoS stride = pot_stride
    const long *pot_src_offsets;    // [n_boxes]
    int n_direct_work;
    int n_input_dim;       // kernel_input_dim; also the charge AoS stride
    int pot_stride;        // kernel_output_dim_src; the pot AoS stride
    int pot_output_offset; // 0 for the potential term, 1 for the dipole gradient term
};

} // namespace dmk::cuda

#endif // DMK_CUDA_SELF_CORRECTION_KERNELARGS_HPP
