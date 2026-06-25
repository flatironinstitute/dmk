#pragma once

#include <array>
#include <vector>

namespace dmk {

template <typename Real>
using Vec3T = std::array<Real, 3>;
using Vec3 = Vec3T<double>;

// Opaque plan: holds PSWFKernel, grid params, and precomputed scaling coefficients.
// Heap-allocated; callers only hold a pointer. Always double precision internally.
struct EspPlan;

EspPlan *esp_create_plan(double L, double r_c, double eps);
void esp_destroy_plan(EspPlan *plan);

// Returns the total potential at each particle (length = charges.size()).
// Real may be float or double; float inputs are promoted to double internally.
template <typename Real>
std::vector<Real> esp_eval(EspPlan *plan,
                            const std::vector<Vec3T<Real>> &r_src,
                            const std::vector<Real> &charges);

// Convenience one-shot wrapper (create + eval + destroy).
template <typename Real>
std::vector<Real> esp_potential(const std::vector<Vec3T<Real>> &r_src,
                                 const std::vector<Real> &charges,
                                 double L, double r_c, double eps);

} // namespace dmk
