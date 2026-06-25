#pragma once

#include <array>
#include <vector>

namespace dmk {

using Vec3 = std::array<double, 3>;

// Opaque plan: holds PSWFKernel, grid params, and precomputed scaling coefficients.
// Heap-allocated; callers only hold a pointer.
struct EspPlan;

EspPlan *esp_create_plan(double L, double r_c, double eps);
void esp_destroy_plan(EspPlan *plan);

// Returns the total potential at each particle (length = charges.size()).
std::vector<double> esp_eval(EspPlan *plan, const std::vector<Vec3> &r_src, const std::vector<double> &charges);

// Convenience one-shot wrapper (create + eval + destroy).
std::vector<double> esp_potential(const std::vector<Vec3> &r_src, const std::vector<double> &charges, double L,
                                  double r_c, double eps);

} // namespace dmk
