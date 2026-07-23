#pragma once

#include <dmk.h>
#include <dmk/prolate0_fun.hpp>
#include <dmk/types.hpp>
#include <dmk/util.hpp>

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

// Effective tolerance to resolve internally when gradients/forces are requested. A gradient costs
// extra PSWF resolution relative to the potential, and the amount is kernel- and dimension-dependent
// (the DMK d_eff analog). Force lower-envelope fits (achieved force digits >= a*requested + b) come
// from scripts/analyze_esp_error.py over DMK_ESP_NO_GRAD_BUMP=1 measure_error_esp sweeps: to guarantee
// the requested target on the force, resolve d = ceil((target - b)/a) digits internally.
inline double esp_grad_eps(dmk_ikernel kernel, int dim, double eps) {
    double a = 1.0, b = 0.0;
    if (kernel == DMK_LAPLACE && dim == 2) {
        a = 1.0;
        b = 0.0;
    } else if (kernel == DMK_LAPLACE && dim == 3) {
        a = 1.01;
        b = -0.19;
    } else if (kernel == DMK_SQRT_LAPLACE && dim == 2) {
        a = 0.99;
        b = 0.21;
    } else if (kernel == DMK_SQRT_LAPLACE && dim == 3) {
        a = 1.03;
        b = 0.02;
    } else if (kernel == DMK_YUKAWA && dim == 2) {
        a = 0.88;
        b = 0.02;
    } else if (kernel == DMK_YUKAWA && dim == 3) {
        a = 1.00;
        b = -0.10;
    } else if (kernel == DMK_LAPLACE_DIPOLE && dim == 3) {
        // Dipole shares the Laplace-3D residual profile (its evaluator differentiates it), so reuse the
        // Laplace-3D force fit until a dedicated analyze_esp_error sweep calibrates it.
        a = 1.01;
        b = -0.19;
    }
    b -= 0.2;

    const double target = -std::log10(eps);
    return std::pow(10.0, -std::ceil((target - b) / a));
}

// n_digits the plan resolves to: the requested eps, tightened by esp_grad_eps when forces are wanted
// (the ESP analog of DMK's d_eff bump) so every stage -- eps_d, P, beta, the short-range residual --
// resolves to match. DMK_ESP_NO_GRAD_BUMP disables the bump, used to calibrate the grad curve raw.
inline int esp_plan_digits(const pdmk_esp_params &p) {
    // Laplace-dipole always needs the bump: even its potential is a derivative of the scalar residual.
    const bool wants_deriv = p.eval_type >= DMK_POTENTIAL_GRAD || p.kernel == DMK_LAPLACE_DIPOLE;
    const bool grad = wants_deriv && !util::env_is_set("DMK_ESP_NO_GRAD_BUMP");
    return esp_digits_from_eps(grad ? esp_grad_eps(p.kernel, p.n_dim, p.eps) : p.eps);
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

// Potential-family kernels (scalar, Laplace-dipole) fill pot + force_x/y/z: force_x/y/z are empty if
// eval_type == DMK_POTENTIAL, and force_z stays empty for a DIM=2 plan (callers can distinguish DIM by
// force_z.empty()). Vector-field kernels (Stokeslet velocity) fill vel_x/y/z instead, leaving the
// pot/force_* spans empty.
template <typename Real>
struct PotForce {
    std::span<Real> pot, force_x, force_y, force_z;
    std::span<Real> vel_x, vel_y, vel_z;
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
    Real dipole_grad_self{0};         // Laplace-dipole gradient self-constant (grad[a] -= this * d[a] per source)
    pdmk_esp_params params;
    // Component counts (scalar kernels: 1/1). Vector kernels carry input_dim charge components per
    // source and output_dim potential components per target; grad_is_force distinguishes the scalar
    // "force" gradient (-q*grad, requires the per-target charge) from a raw field gradient (dipole).
    int input_dim{1}, output_dim{1}, normal_dim{0};
    // Per-source payload width fed to short_range/long_range: input_dim charge comps, then normal_dim
    // normal comps packed after them (Stresslet = force(3) + normal(3) = 6; every other kernel = input_dim).
    int charge_dim{1};
    bool grad_is_force{true};
    residual_evaluator_func<Real> evaluator;
    residual_evaluator_range_func<Real> range_evaluator;
    std::vector<Real> buf;

    // long_range scratch, reused across eval calls: the n-sized buffers grow to the largest n seen,
    // the ntot-sized ones are fixed by nf/n_dim. Sized max dim (3) since EspPlan isn't DIM-templated.
    std::array<std::vector<Real>, 3> lr_coord;               // n-sized NU coordinates per axis
    std::vector<std::complex<Real>> lr_c, lr_out;            // n-sized channel charges / interp output
    std::vector<std::vector<std::complex<Real>>> lr_in;      // ntot-sized per-input-channel spread/FFT grids
    std::array<std::vector<std::complex<Real>>, 4> lr_u_hat; // ntot-sized output-component spectra (pot + force axes)

    explicit EspPlan(const pdmk_esp_params &params);

    // normals is required for the Stresslet (force-dipole orientation, normal_dim comps per source) and
    // ignored otherwise.
    PotForce<Real> eval(int n, const Real *r_src, const Real *charges, const Real *normals = nullptr);

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
