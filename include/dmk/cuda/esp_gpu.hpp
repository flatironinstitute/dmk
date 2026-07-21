#pragma once
// Internal header shared between esp.cpp (g++) and esp_cuda.cu (nvcc).
// Must not pull in SCTL or any x86-intrinsic header.
#ifdef DMK_GPU_OFFLOAD

#include <dmk.h> // dmk_eval_type

namespace dmk {

struct GpuState; // full definition lives in esp_cuda.cu

// Called from esp_create_gpu_plan in esp.cpp.
// Allocates a GpuState, uploads scaling_coeffs and the short-range polynomial
// coefficients to device, and initialises CUDA objects.
// h_scaling_coeffs[nf³]: precomputed scaling coefficients (computed by esp.cpp).
// h_sr_coeffs[n_sr_coeffs]: monomial fit of the short-range kernel S(r), same
// table get_esp_3d_kernel dispatches into on the CPU path (esp.cpp).
// gpu_upsampfac: the GPU spreader's own upsampfac (esp.cpp's GPU_SPREADER_UPSAMPFAC),
// deliberately NOT the PSWF splitting kernel's sigma -- see esp.cpp for why.
GpuState *gpu_create_state(
    int nf, int n_digits,
    double L, double r_c, double gpu_upsampfac, double tol,
    double self_factor_d, float self_factor_f,
    dmk_eval_type eval_type,
    const double *h_scaling_coeffs,
    const double *h_sr_coeffs, int n_sr_coeffs);

// Called from esp_destroy_gpu_plan in esp.cpp.
void gpu_destroy_state(GpuState *gpu);

} // namespace dmk

#endif // DMK_GPU_OFFLOAD
