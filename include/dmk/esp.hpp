#pragma once

#include <dmk.h>
#include <dmk/prolate0_fun.hpp>
#include <dmk/types.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <span>
#include <vector>

namespace dmk {

template <typename Real, int DIM = 3>
using Vec3T = std::array<Real, DIM>;
using Vec3 = Vec3T<double>;

// Formula matches FINUFFT v2.5 kerformula=8 (PSWF):
// https://github.com/flatironinstitute/finufft/blob/704cbfee0375a4f726e8ff5a2c4ef70d5da6257a/devel/find_sigma_bound.cpp#L103
inline int esp_P_from_eps(double eps, double sigma, int dim) {
    const double tolfac = 0.18 * std::pow(1.4, dim - 1);
    // P: spread width = number of grid points used per dimension in the spreading stencil
    const int P = static_cast<int>(std::ceil(std::log(tolfac / eps) / (M_PI * std::sqrt(1.0 - 1.0 / sigma)) + 1.0));
    return std::max(2, P);
}

// PSWF bandwidth parameter beta from the spread width P and upsampling factor sigma.
inline double esp_beta_from_P(double sigma, int P) { return M_PI * P * (1.0 - 1.0 / (2 * sigma)) - 0.05; }

inline int esp_digits_from_eps(double eps) {
    return std::clamp(static_cast<int>(std::lround(-std::log10(eps))), 2, 12);
}

// Short-range method predicates over pdmk_esp_params.esp_flags (DMK_ESP_* bits, see dmk.h). The
// three strategies (source-pruning granularity, within-cell spatial sort, Newton's-third-law
// reciprocal) are independent; esp_spatial_sort is true if any of them wants particles sorted
// within their cell.
inline bool esp_prune_tile(const pdmk_esp_params &params) { return params.esp_flags & DMK_ESP_PRUNE_TILE; }
inline bool esp_prune_source(const pdmk_esp_params &params) { return params.esp_flags & DMK_ESP_PRUNE_SOURCE; }
inline bool esp_n3l(const pdmk_esp_params &params) { return params.esp_flags & DMK_ESP_N3L; }
inline bool esp_morton(const pdmk_esp_params &params) { return params.esp_flags & DMK_ESP_MORTON; }
inline bool esp_spatial_sort(const pdmk_esp_params &params) {
    return params.esp_flags & (DMK_ESP_PRUNE_TILE | DMK_ESP_PRUNE_SOURCE | DMK_ESP_N3L);
}

// force_x/y/z are empty spans if the plan was created with eval_type == DMK_POTENTIAL. For a
// DIM=2 plan, force_z stays empty even when forces are requested (only force_x/force_y are
// populated) -- callers can distinguish DIM by checking force_z.empty().
template <typename Real>
struct PotForce {
    std::span<Real> pot, force_x, force_y, force_z;
};

// PSWF (prolate spheroidal wave function) far-field kernel: precomputes lambda0/scale and
// polynomial fits of the kernel and its integral for a given (eps, c).
struct PSWFKernel {
    dmk::Prolate0Fun pswf;
    double eps, beta, lambda0, c0, scale;

    PSWFKernel() = default;
    // Heavy (runs prol0ini + two poly_fit calls); declared here, defined in esp.cpp so this header
    // doesn't need finufft_common/kernel.h.
    explicit PSWFKernel(double eps_, double beta_, int lenw = 8000);

    double operator()(double x) const { return pswf.eval_val(x) * scale; }
    double integral_eval(double t) const { return pswf.int_eval(t) * scale; }
    double integral(double a, double b) const {
        double va = (a == 0.0) ? 0.0 : integral_eval(a);
        double vb = integral_eval(b);
        return vb - va;
    }
    double pswf_hat(double k) const {
        const double x = k / beta;
        return std::fabs(x) > 1 ? 0.0 : lambda0 * (*this)(x);
    }
};

template <typename Real>
struct EspPlan {
    int n_digits;
    int n_dim;
    int P, n_f;      // spread width and oversampled grid size per axis
    double h;        // oversampled grid spacing L_grid/n_f
    double pad;      // FFT-grid padding factor per axis (1 periodic; 2*sqrt(n_dim) free-space)
    double L_grid;   // spectral-grid period pad*L (periodic: == L)
    double trunc_rl; // free-space kernel truncation radius = sqrt(n_dim)*L (source-box diagonal)
    PSWFKernel pswf;
    std::vector<Real> scaling_coeffs; // diagonal far-field scaling, computed in double then narrowed to Real once
    Real self_factor{0};              // long-range kernel value at r=0, subtracted per source (self-energy)
    pdmk_esp_params params;
    residual_evaluator_func<Real> evaluator;
    residual_evaluator_range_func<Real> range_evaluator;
    std::vector<Real> buf;

    // long_range scratch, reused across eval calls: the n-sized buffers grow to the largest n seen,
    // the ntot-sized ones are fixed by nf/n_dim. Sized max dim (3) since EspPlan isn't DIM-templated.
    std::array<std::vector<Real>, 3> lr_coord;               // n-sized NU coordinates per axis
    std::vector<std::complex<Real>> lr_c, lr_out;            // n-sized charges / interp output
    std::vector<std::complex<Real>> lr_b;                    // ntot-sized spread / forward-FFT grid
    std::array<std::vector<std::complex<Real>>, 4> lr_u_hat; // ntot-sized output-component spectra (pot + force axes)

    explicit EspPlan(const pdmk_esp_params &params);

    PotForce<Real> eval(int n, const Real *r_src, const Real *charges);

    template <int DIM>
    std::vector<double> precompute_scaling_coefficients();
    template <int DIM>
    void short_range(int n, const Real *r_src, const Real *charges, std::span<Real> pot,
                     std::array<std::span<Real>, DIM> force);
    template <int DIM>
    void long_range(int n, const Real *r_src, const Real *charges, std::span<Real> pot,
                    std::array<std::span<Real>, DIM> force);
    void self_interaction(int n, const Real *charges, std::span<Real> pot);
};

} // namespace dmk
