// V2 form_outgoing kernel-FT multiply (scalar kernels: Laplace / Sqrt-Laplace).
// Per-box in-place multiply of each plane-wave mode by radialft[m]. The
// launcher prepends `using Real` + BLOCK_SIZE (BLOCK_SIZE is unused by the body
// but kept uniform with the thin-launcher contract).

#include <dmk/cuda/multiply_kernelft_kernelargs.hpp>

using dmk::cuda::MultiplyCd2pArgs;

// KERNEL_START

extern "C" __global__ void PtMultiplyCd2pKernel(MultiplyCd2pArgs<Real> a) {
    const int box_idx = blockIdx.x;
    if (box_idx >= a.n_boxes_at_level)
        return;
    const int box = a.box_ids[box_idx];
    const long off_complex = a.pw_offsets ? a.pw_offsets[box] : box_idx * a.pw_stride_complex;
    if (off_complex < 0)
        return;
    Real *pw = a.pw_flat + 2 * off_complex;
    const int total = a.n_pw_modes * a.n_charge_dim;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int m = idx % a.n_pw_modes;
        const Real f = a.radialft[m];
        pw[2 * idx] *= f;
        pw[2 * idx + 1] *= f;
    }
}
