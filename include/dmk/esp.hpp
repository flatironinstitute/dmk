#pragma once

#include <dmk.h> // dmk_eval_type

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace dmk {

template <typename Real, int DIM = 3>
using Vec3T = std::array<Real, DIM>;
using Vec3 = Vec3T<double>;

// Formula matches FINUFFT v2.5 kerformula=8 (PSWF):
// https://github.com/flatironinstitute/finufft/blob/704cbfee0375a4f726e8ff5a2c4ef70d5da6257a/devel/find_sigma_bound.cpp#L103
inline int esp_P_from_eps(double eps, double sigma, int dim = 3) {
    const double tolfac = 0.18 * std::pow(1.4, dim - 1);
    // P: spread width = number of grid points used per dimension in the spreading stencil
    const int P = static_cast<int>(std::ceil(std::log(tolfac / eps) / (M_PI * std::sqrt(1.0 - 1.0 / sigma)) + 1.0));
    return std::max(2, P);
}

// PSWF bandwidth parameter c from the spread width P and upsampling factor sigma.
inline double esp_pswf_c_from_P(double sigma, int P) { return M_PI * P * (1.0 - 1.0 / (2 * sigma)) - 0.05; }

inline int esp_digits_from_eps(double eps) {
    return std::clamp(static_cast<int>(std::lround(-std::log10(eps))), 2, 12);
}

// Internal short-range method selection, carrying the DMK_ESP_* bitmask from pdmk_esp_params
// (see dmk.h) plus the two numeric tunables. Constructed by the C wrapper; not a public type.
struct ShortRangeConfig {
    unsigned flags = DMK_ESP_PRUNE_SOURCE | DMK_ESP_N3L | DMK_ESP_MORTON;
    int bins = 2;  // octant-bin count per axis when DMK_ESP_MORTON is clear
    int stile = 0; // source-tile width for DMK_ESP_PRUNE_TILE (0 -> SIMD width)

    bool prune_tile() const { return flags & DMK_ESP_PRUNE_TILE; }
    bool prune_source() const { return flags & DMK_ESP_PRUNE_SOURCE; }
    bool n3l() const { return flags & DMK_ESP_N3L; }
    bool morton() const { return flags & DMK_ESP_MORTON; }
    bool spatial_sort() const { return flags & (DMK_ESP_PRUNE_TILE | DMK_ESP_PRUNE_SOURCE | DMK_ESP_N3L); }
};

struct EspPlan;

// n_dim selects the templated implementation used internally (DIM=2 or 3). Only DIM=3 has a
// working short-range PSWF correction kernel today (see get_esp_3d_kernel); DIM=2 compiles but
// esp_eval throws at the short-range kernel-dispatch site until that kernel is derived.
EspPlan *esp_create_plan(double L, double r_c, double eps, double sigma, dmk_eval_type eval_type = DMK_POTENTIAL_GRAD,
                         ShortRangeConfig cfg = {}, int n_dim = 3);
void esp_destroy_plan(EspPlan *plan);

// force_x/y/z are empty spans if the plan was created with eval_type == DMK_POTENTIAL. For a
// DIM=2 plan, force_z stays empty even when forces are requested (only force_x/force_y are
// populated) -- callers can distinguish DIM by checking force_z.empty().
template <typename Real>
struct PotForce {
    std::span<Real> pot, force_x, force_y, force_z;
};

// r_src is a flat array of n*n_dim coordinates (n_dim taken from the plan), interleaved
// [x0,y0,(z0),x1,y1,(z1),...] -- the same layout the public C API receives from callers.
template <typename Real>
PotForce<Real> esp_eval(EspPlan *plan, int n, const Real *r_src, const Real *charges);

} // namespace dmk
