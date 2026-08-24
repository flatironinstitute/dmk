#pragma once
// Internal header shared between esp.cpp and the ESP GPU launchers in src/cuda/esp/.
// Must not pull in SCTL or any x86-intrinsic header.
#ifdef DMK_GPU_OFFLOAD

#include <dmk.h>       // dmk_eval_type
#include <dmk/esp.hpp> // GpuSrStrategy

namespace dmk {

struct GpuState; // full definition lives in src/cuda/esp/state.hpp

// h_scaling_coeffs is nf^3 long. The short-range polynomial is not passed in: it is generated per
// launch from (n_digits, beta) and baked into the NVRTC module (src/cuda/esp/short_range.cpp).
// gpu_upsampfac is the GPU spreader's own upsampfac, deliberately not the PSWF's sigma.
// use_float fixes the one Real this plan exists for; every esp_eval_gpu*<Real> call must match it.
GpuState *gpu_create_state(int nf, int n_digits, double L, double r_c, double gpu_upsampfac, double tol, double beta,
                           double self_factor, dmk_eval_type eval_type, bool use_float, GpuSrStrategy strategy,
                           GpuSortMode sort_mode, const double *h_scaling_coeffs);

void gpu_destroy_state(GpuState *gpu);

} // namespace dmk

#endif // DMK_GPU_OFFLOAD
