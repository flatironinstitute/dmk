#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace dmk {

template <typename Real>
using Vec3T = std::array<Real, 3>;
using Vec3 = Vec3T<double>;

// PSWF stencil width from tolerance, matching FINUFFT v2.5.0's formula for
// kerformula=8 (PSWF), upsampfac=2, dim=3.
inline int esp_ns_from_eps(double eps, double sigma) {
    const double tolfac = 0.18 * 1.96; // 0.18 * 1.4^(dim-1) for dim=3
    int ns = static_cast<int>(std::ceil(std::log(tolfac / eps) / (M_PI * std::sqrt(1.0 - 1.0 / sigma)) + 1.0));
    return std::max(2, ns);
}

// PSWF shape parameter c, derived from the stencil width (upsampfac=2).
inline double esp_pswf_c_from_P(double sigma, double P) {
    return M_PI * P * (1.0 - 1.0 / (2 * sigma)) - 0.05;
}

// Shape parameter c for a PSWF kernel targeting (eps, sigma), matching FINUFFT's PSWF
// spreader: derive the stencil width P from (eps, sigma), then c from (sigma, P).
inline double esp_pswf_c_from_eps(double eps, double sigma) {
    const int P = esp_ns_from_eps(eps, sigma);
    return esp_pswf_c_from_P(sigma, P);
}

// Tolerance digit bucket used to select the precompiled short-range evaluator.
inline int esp_digits_from_eps(double eps) {
    return std::clamp(static_cast<int>(std::lround(-std::log10(eps))), 2, 12);
}

// Opaque plan: holds PSWFKernel, grid params, and precomputed scaling coefficients.
// Heap-allocated; callers only hold a pointer. Always double precision internally.
struct EspPlan;

EspPlan *esp_create_plan(double L, double r_c, double eps, double sigma);
void esp_destroy_plan(EspPlan *plan);

// Per-phase wall times filled in by esp_eval when timings != nullptr.
struct EspTimings {
    double t_short = 0; // short-range: cell-list build + direct sum
    double t_long = 0;  // long-range: FINUFFT spread + FFT + scale + interpolate
    double t_self = 0;  // self-interaction correction
};

// Potential + force at each particle (length = charges.size()). force_i = -charges[i] * grad_i,
// i.e. already includes the particle's own charge — ready to use directly as a physical force.
template <typename Real>
struct PotForce {
    std::vector<Real> pot, force_x, force_y, force_z;
};

// Returns the total potential and force at each particle.
// Real may be float or double; float inputs are promoted to double internally.
template <typename Real>
PotForce<Real> esp_eval(EspPlan *plan, const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges,
                        EspTimings *timings = nullptr);

// Convenience one-shot wrapper (create + eval + destroy).
template <typename Real>
PotForce<Real> esp_potential(const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges, double L,
                             double r_c, double eps);

} // namespace dmk
