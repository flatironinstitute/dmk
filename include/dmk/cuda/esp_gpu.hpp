#pragma once
// Internal header shared between esp.cpp and the ESP GPU launchers in src/cuda/esp/.
// Must not pull in SCTL or any x86-intrinsic header.
#ifdef DMK_GPU_OFFLOAD

#include <dmk.h>       // dmk_ikernel, dmk_eval_type
#include <dmk/esp.hpp> // GpuSrStrategy

namespace dmk {

struct GpuState; // full definition lives in src/cuda/esp/state.hpp

// Everything a GPU plan needs from EspPlan. The short-range polynomial is not here: it is generated
// per launch from (kernel, fparam, r_c, n_digits, beta) and baked into the NVRTC module.
struct GpuPlanConfig {
    int nf = 0;
    int n_digits = 0;
    double L_grid = 0; // FFT grid extent (EspPlan::L_grid): 1 when periodic, padded otherwise
    double r_c = 0;
    // cuFINUFFT's upsampfac; must equal params.sigma so the ES kernel matches this grid.
    double gpu_upsampfac = 0;
    double tol = 0;
    double beta = 0;
    double self_factor = 0;
    double fparam = 0;           // Yukawa lambda; unused by the other kernels
    double dipole_grad_self = 0; // Laplace-dipole gradient self-constant
    dmk_ikernel kernel = DMK_LAPLACE;
    dmk_eval_type eval_type = DMK_POTENTIAL;
    bool use_periodic = true;
    double trunc_rl = 0; // free-space truncation radius; Stokeslet's zero-mode gauge needs it
    // Fixes the one Real this plan exists for; every esp_eval_gpu*<Real> call must match it.
    bool use_float = false;
    GpuSrStrategy strategy = GpuSrStrategy::Dense;
    GpuSortMode sort_mode = GpuSortMode::Bins;
    const double *h_scaling_coeffs = nullptr; // nf^3 long
};

GpuState *gpu_create_state(const GpuPlanConfig &cfg);

void gpu_destroy_state(GpuState *gpu);

} // namespace dmk

#endif // DMK_GPU_OFFLOAD
