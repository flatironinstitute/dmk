#pragma once

#include <dmk.h> // dmk_eval_type

#include <algorithm>
#include <array>
#include <cmath>
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

// Per-phase wall times filled in by esp_eval when timings != nullptr.
struct EspTimings {
    double t_short = 0; // short-range: cell-list build + direct sum
    double t_long = 0;  // long-range: FINUFFT spread + FFT + scale + interpolate
    double t_self = 0;  // self-interaction correction
};

// Potential + force at each particle (length = charges.size()). force_i = -charges[i] * grad_i,
// i.e. already includes the particle's own charge — ready to use directly as a physical force.
// force_x/y/z are left empty if the plan was created with eval_type == DMK_POTENTIAL.
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
                             double r_c, double eps, dmk_eval_type eval_type = DMK_POTENTIAL_GRAD);

} // namespace dmk
