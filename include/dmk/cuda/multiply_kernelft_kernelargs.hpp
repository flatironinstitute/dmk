#ifndef DMK_CUDA_MULTIPLY_KERNELFT_KERNELARGS_HPP
#define DMK_CUDA_MULTIPLY_KERNELFT_KERNELARGS_HPP

namespace dmk::cuda {

template <typename Real>
struct MultiplyCd2pArgs {
    int n_boxes_at_level = 0;
    int n_charge_dim = 0; // = n_tables_up == n_tables_down here
    int n_pw_modes = 0;

    const int *box_ids = nullptr;     // [n_boxes_at_level]
    const Real *radialft = nullptr;   // [n_pw_modes]
    Real *pw_flat = nullptr;          // interleaved complex (in place)
    const long *pw_offsets = nullptr; // in COMPLEX units; if null, use box_idx*pw_stride_complex
    long pw_stride_complex = 0;
};

template <typename Real>
struct MultiplyStokeslet3DArgs {
    int n_boxes_at_level = 0;
    int n_pw = 0;
    int n_pw2 = 0;
    int n_pw_modes = 0;
    Real hpw = 0;
    bool is_windowed = false;

    const int *box_ids = nullptr;
    const Real *radialft = nullptr;
    Real *pw_flat = nullptr;          // 3 charge dims, in place
    const long *pw_offsets = nullptr; // in COMPLEX units
    long pw_stride_complex = 0;
};

template <typename Real>
struct MultiplyStresslet3DArgs {
    int n_boxes_at_level = 0;
    int n_pw = 0;
    int n_pw2 = 0;
    int n_pw_modes = 0;
    Real hpw = 0;

    const int *box_ids = nullptr;
    const Real *radialft = nullptr;
    const Real *src_flat = nullptr;    // 9 input tables (interleaved complex)
    const long *src_offsets = nullptr; // in COMPLEX units (within src layout)
    long src_stride_complex = 0;       // if src_offsets is null
    Real *dst_flat = nullptr;          // 3 output tables (interleaved complex)
    const long *dst_offsets = nullptr; // in COMPLEX units (within dst layout)
    long dst_stride_complex = 0;
};

} // namespace dmk::cuda

#endif // DMK_CUDA_MULTIPLY_KERNELFT_KERNELARGS_HPP
