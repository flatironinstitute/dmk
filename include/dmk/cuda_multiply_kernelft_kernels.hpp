#ifndef DMK_CUDA_MULTIPLY_KERNELFT_KERNELS_HPP
#define DMK_CUDA_MULTIPLY_KERNELFT_KERNELS_HPP

// Per-block GPU equivalents of the host multiply_kernelFT_* family. Each
// kernel multiplies a per-box plane-wave expansion by a Fourier-space
// kernel (radialft + a kernel-specific formula). Output is interleaved
// complex.
//
//   - cd2p:        scalar/Laplace-style (pw[m, d] *= radialft[m]).
//   - stokeslet_3d: vector kernel; couples 3 charge dims via k.
//   - stresslet_3d: tensor kernel; reads 9 input tables, writes 3 output.

#include <cuda_runtime.h>

namespace dmk::cuda {

// ----- cd2p -----
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
void launch_multiply_cd2p_dispatch(int dim, const MultiplyCd2pArgs<Real> &args, cudaStream_t stream);

// ----- stokeslet_3d -----
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
void launch_multiply_stokeslet_3d_dispatch(int dim, const MultiplyStokeslet3DArgs<Real> &args, cudaStream_t stream);

// ----- stresslet_3d -----
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

template <typename Real>
void launch_multiply_stresslet_3d_dispatch(int dim, const MultiplyStresslet3DArgs<Real> &args, cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_MULTIPLY_KERNELFT_KERNELS_HPP
