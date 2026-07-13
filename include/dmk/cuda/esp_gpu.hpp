#pragma once
// Internal header shared between esp.cpp (g++) and esp_cuda.cu (nvcc).
// Must not pull in SCTL or any x86-intrinsic header.
#ifdef DMK_GPU_OFFLOAD

#include <dmk.h> // dmk_eval_type

namespace dmk {

struct GpuState; // full definition lives in esp_cuda.cu

// Called from esp_create_gpu_plan in esp.cpp.
// Allocates a GpuState, uploads scaling_coeffs to device, and initialises CUDA objects.
// h_scaling_coeffs[nf³]: precomputed scaling coefficients (computed by esp.cpp).
GpuState *gpu_create_state(
    int nf, int n_digits,
    double L, double r_c, double sigma, double tol,
    double self_factor_d, float self_factor_f,
    dmk_eval_type eval_type,
    const double *h_scaling_coeffs);

// Called from esp_destroy_gpu_plan in esp.cpp.
void gpu_destroy_state(GpuState *gpu);

} // namespace dmk

#endif // DMK_GPU_OFFLOAD
