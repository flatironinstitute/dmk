#pragma once

#include <dmk.h> // dmk_eval_type

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace dmk {

template <typename Real>
using Vec3T = std::array<Real, 3>;
using Vec3 = Vec3T<double>;

// Formula matches FINUFFT v2.5 kerformula=8 (PSWF):
// https://github.com/flatironinstitute/finufft/blob/704cbfee0375a4f726e8ff5a2c4ef70d5da6257a/devel/find_sigma_bound.cpp#L103
inline int esp_P_from_eps(double eps, double sigma, int dim = 3) {
    const double tolfac = 0.18 * std::pow(1.4, dim - 1);
    //P: spread width = number of grid points used per dimension in the spreading stencil
    const int P = static_cast<int>(std::ceil(std::log(tolfac / eps) / (M_PI * std::sqrt(1.0 - 1.0 / sigma)) + 1.0));
    return std::max(2, P);
}

// PSWF bandwidth parameter c from the spread width P and upsampling factor sigma.
inline double esp_pswf_c_from_P(double sigma, int P) {
    return M_PI * P * (1.0 - 1.0 / (2 * sigma)) - 0.05;
}

inline int esp_digits_from_eps(double eps) {
    return std::clamp(static_cast<int>(std::lround(-std::log10(eps))), 2, 12);
}

struct EspPlan;

EspPlan *esp_create_plan(double L, double r_c, double eps, double sigma, dmk_eval_type eval_type = DMK_POTENTIAL_GRAD);
void esp_destroy_plan(EspPlan *plan);

// force_x/y/z are empty spans if the plan was created with eval_type == DMK_POTENTIAL.
template <typename Real>
struct PotForce {
    std::span<Real> pot, force_x, force_y, force_z;
};

template <typename Real>
PotForce<Real> esp_eval(EspPlan *plan, const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges);

// Component-wise evaluation — useful for testing GPU vs CPU one sub-step at a time.
// Each returns only its own contribution (no self-interaction correction).
template <typename Real>
PotForce<Real> esp_eval_short_range(EspPlan *plan, const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges);
template <typename Real>
PotForce<Real> esp_eval_long_range(EspPlan *plan, const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges);

#ifdef DMK_GPU_OFFLOAD
// GPU plan — a separate, independent object that owns all CUDA resources.
// Create one alongside (or instead of) an EspPlan; destroy when done.
struct GpuState;
GpuState *esp_create_gpu_plan(EspPlan *plan);
void      esp_destroy_gpu_plan(GpuState *gpu);

// GPU evaluation — same semantics as esp_eval but runs on GPU.
// The returned spans are valid until the next call to esp_eval_gpu on the same gpu,
// or until esp_destroy_gpu_plan.
PotForce<float>  esp_eval_gpu(GpuState *gpu, const std::vector<Vec3T<float>>  &r_src, const std::vector<float>  &charges);
PotForce<double> esp_eval_gpu(GpuState *gpu, const std::vector<Vec3T<double>> &r_src, const std::vector<double> &charges);

// GPU component-wise evaluation — mirrors esp_eval_short_range / esp_eval_long_range.
PotForce<float>  esp_eval_gpu_short_range(GpuState *gpu, const std::vector<Vec3T<float>>  &r_src, const std::vector<float>  &charges);
PotForce<double> esp_eval_gpu_short_range(GpuState *gpu, const std::vector<Vec3T<double>> &r_src, const std::vector<double> &charges);
PotForce<float>  esp_eval_gpu_long_range(GpuState *gpu, const std::vector<Vec3T<float>>  &r_src, const std::vector<float>  &charges);
PotForce<double> esp_eval_gpu_long_range(GpuState *gpu, const std::vector<Vec3T<double>> &r_src, const std::vector<double> &charges);
#endif

} // namespace dmk
