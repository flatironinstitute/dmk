#pragma once
// Internal header shared between esp.cpp (g++) and esp_cuda.cu (nvcc).
// Must not pull in SCTL or any x86-intrinsic header.
#ifdef DMK_GPU_OFFLOAD

#include <dmk.h> // dmk_eval_type
#include <dmk/esp.hpp> // GpuSrStrategy

namespace dmk {

struct GpuState; // full definition lives in esp_cuda.cu

// Called from esp_create_gpu_plan in esp.cpp.
// Allocates a GpuState, uploads scaling_coeffs to device, and initialises CUDA
// objects. The short-range polynomial fit of S(r) is NOT passed in here -- it
// rides into the short-range kernel as a compile-time CoeffTag (see
// esp_sr_coeffs.cuh), selected at launch time in esp_cuda.cu from eval_type/
// n_digits, both of which are already plan-level state.
// h_scaling_coeffs[nf³]: precomputed scaling coefficients (computed by esp.cpp).
// gpu_upsampfac: the GPU spreader's own upsampfac (esp.cpp's GPU_SPREADER_UPSAMPFAC),
// deliberately NOT the PSWF splitting kernel's sigma -- see esp.cpp for why.
// use_float: selects the ONE Real (float or double) this plan is created for --
// every long-range resource (cuFFT plan, cuFINUFFT plans, scaling coeffs,
// spread/FFT buffers) is allocated for that Real only. Every later
// esp_eval_gpu*<Real> call on the returned GpuState must use the matching
// Real (a mismatched call throws) -- see esp.hpp's float/double overloads.
// strategy: selects which short-range kernel this plan's short_range_gpu
// calls will use -- see GpuSrStrategy in esp.hpp. Fixed at plan-creation
// time, like use_float; create multiple GpuStates to compare strategies.
GpuState *gpu_create_state(
    int nf, int n_digits,
    double L, double r_c, double gpu_upsampfac, double tol,
    double self_factor_d, float self_factor_f,
    dmk_eval_type eval_type, bool use_float, GpuSrStrategy strategy,
    const double *h_scaling_coeffs);

// Called from esp_destroy_gpu_plan in esp.cpp.
void gpu_destroy_state(GpuState *gpu);

} // namespace dmk

#endif // DMK_GPU_OFFLOAD
