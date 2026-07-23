#include <cmath>
#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/legeexps.hpp>
#include <dmk/planewave.hpp>
#include <dmk/polyfit.hpp>
#include <dmk/prolate0_fun.hpp>
#include <dmk/types.hpp>
#include <dmk/util.hpp>

#include <complex.h>
#include <format>
#include <limits>
#include <sctl.hpp>
#include <stdexcept>
#include <string>

namespace dmk {

template <int DIM>
std::tuple<double, double> get_PSWF_difference_kernel_pwterms(int npw, double beta, double boxsize) {
    const int nf = (npw - 1) / 2;
    const double hpw = 2 * beta / (nf + 1) / boxsize;

    constexpr double factor = 0.5 / sctl::pow<DIM - 1>(M_PI);
    const double ws = sctl::pow<DIM>(hpw) * factor;

    return std::make_tuple(hpw, ws);
}

inline std::tuple<double, double> get_PSWF_difference_kernel_pwterms(int dim, int npw, double beta, double boxsize) {
    if (dim == 2)
        return get_PSWF_difference_kernel_pwterms<2>(npw, beta, boxsize);
    if (dim == 3)
        return get_PSWF_difference_kernel_pwterms<3>(npw, beta, boxsize);
    throw std::runtime_error("Invalid dimension " + std::to_string(dim) + " provided");
}

template <int DIM>
std::tuple<double, double, double> get_PSWF_windowed_kernel_pwterms(double boxsize) {
    const double hpw = 1.0 / boxsize;

    constexpr double factor = 0.5 / sctl::pow<DIM - 1>(M_PI);
    const double ws = sctl::pow<DIM>(hpw) * factor;

    constexpr double two_sqrt_dim = DIM == 2 ? 2 * 1.4142135623730951 : 2 * 1.7320508075688772;
    const double rl = boxsize * two_sqrt_dim;

    return std::make_tuple(hpw, ws, rl);
}

inline std::tuple<double, double, double> get_PSWF_windowed_kernel_pwterms(int dim, double boxsize) {
    if (dim == 2)
        return get_PSWF_windowed_kernel_pwterms<2>(boxsize);
    if (dim == 3)
        return get_PSWF_windowed_kernel_pwterms<3>(boxsize);
    throw std::runtime_error("Invalid dimension " + std::to_string(dim) + " provided");
}

template <typename Real, int DIM>
void yukawa_windowed_kernel_ft(const double *params, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                               sctl::Vector<Real> &windowed_ft) {
    auto [hpw, ws, L] = get_PSWF_windowed_kernel_pwterms<DIM>(boxsize);
    const int n_fourier = DIM * sctl::pow<2>(npw / 2) + 1;
    windowed_ft.ReInit(n_fourier);

    const Real lambda = params[0];
    const Real lambda2 = lambda * lambda;

    // determine whether one needs to smooth out the 1/(k^2+lambda^2) factor at the origin.
    // needed in the calculation of kernel-smoothing when there is low-frequency breakdown
    // FIXME: this gives 12 digits for my tested lambdas, so doesn't use beta dependence.
    // Original was beta dependent but thresholded poorly
    const bool near_correction = DIM == 2 ? lambda < 5 : false;
    Real bessk0_Llambda, bessk1_Llambda, exp_Llambda;
    if (near_correction) {
        Real arg = L * lambda;
        if constexpr (DIM == 2) {
            bessk0_Llambda = util::cyl_bessel_k(0, arg);
            bessk1_Llambda = util::cyl_bessel_k(1, arg);
        } else if constexpr (DIM == 3)
            exp_Llambda = std::exp(-arg);
    }

    const Real psi0 = pf.eval_val(0);
    const Real factor = ws / psi0;
    for (int i = 0; i < n_fourier; ++i) {
        const Real k = sqrt((Real)i) * hpw;
        const Real xi2 = k * k + lambda2;
        const Real xi = sqrt(xi2);
        const Real xval = xi * boxsize / beta;
        const Real fval = (xval <= 1.0) ? pf.eval_val(xval) : 0.0;

        windowed_ft[i] = factor * fval / xi2;

        if (near_correction) {
            const Real xsc = L * k;
            if constexpr (DIM == 2) {
                using util::cyl_bessel_j;
                windowed_ft[i] *= -L * lambda * cyl_bessel_j(0, xsc) * bessk1_Llambda + Real{1.0} +
                                  xsc * cyl_bessel_j(1, xsc) * bessk0_Llambda;
            } else if constexpr (DIM == 3) {
                const Real sin_over_k = (k > Real{0}) ? sin(xsc) / k : L;
                windowed_ft[i] *= Real{1} - exp_Llambda * (cos(xsc) + lambda * sin_over_k);
            }
        }
    }
}

template <typename Real>
void laplace_2d_windowed_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                   sctl::Vector<Real> &windowed_ft) {
    constexpr int DIM = 2;
    const auto [hpw, ws, rl] = get_PSWF_windowed_kernel_pwterms<DIM>(boxsize);
    const int n_fourier = DIM * sctl::pow<2>(npw / 2) + 1;
    windowed_ft.ReInit(n_fourier);

    const Real psi0 = pf.eval_val(0.0);
    const Real dfact = rl * std::log(rl);
    for (int i = 0; i < n_fourier; ++i) {
        const Real rk = sqrt((Real)i) * hpw;
        const Real xval = rk * boxsize / beta;
        const Real fval = (xval <= 1.0) ? pf.eval_val(xval) : 0.0;
        const Real x = rl * rk;

        windowed_ft[i] = ws * fval / psi0;
        if (x > 1E-10) {
            const Real dj0 = util::cyl_bessel_j(0, x);
            const Real dj1 = util::cyl_bessel_j(1, x);
            const Real tker = -(1 - dj0) / (rk * rk) + dfact * dj1 / rk;
            windowed_ft[i] *= tker;
        } else
            windowed_ft[i] *= -0.25 * rl * rl + 0.5 * dfact * rl;
    }
}

template <typename Real>
void laplace_3d_windowed_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                   sctl::Vector<Real> &windowed_ft) {
    constexpr int DIM = 3;
    const auto [hpw, ws, rl] = get_PSWF_windowed_kernel_pwterms<DIM>(boxsize);
    const int n_fourier = DIM * sctl::pow<2>(npw / 2) + 1;
    windowed_ft.ReInit(n_fourier);

    auto [c0, c1, c2, c3] = pf.intvals(beta);
    c1 = c0 * boxsize;

    constexpr int n_quad = 100;
    std::array<Real, n_quad> xs, whts, fvals;
    legerts(1, n_quad, xs.data(), whts.data());

    for (int i = 0; i < n_quad; ++i) {
        xs[i] = 0.5 * (xs[i] + 1) * boxsize;
        whts[i] *= 0.5 * boxsize;
    }

    for (int i = 0; i < n_quad; ++i) {
        const Real val = pf.eval_val(xs[i] / boxsize);
        fvals[i] = val * whts[i] / c1;
    }

    for (int i = 0; i < n_fourier; ++i) {
        double rk = sqrt((Real)i) * hpw;

        windowed_ft[i] = Real{0.0};
        for (int j = 0; j < n_quad; ++j)
            windowed_ft[i] += cos(rk * xs[j]) * fvals[j];

        if (i > 0)
            windowed_ft[i] *= ws * Real{2.0} * sctl::pow<2>(sin(0.5 * rk * rl) / rk);
        else
            windowed_ft[i] *= 0.5 * ws * sctl::pow<2>(rl);
    }
}

template <typename Real, int DIM>
inline void laplace_windowed_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                       sctl::Vector<Real> &windowed_ft) {
    if constexpr (DIM == 2)
        return laplace_2d_windowed_kernel_ft<Real>(rpars, beta, npw, boxsize, pf, windowed_ft);
    if constexpr (DIM == 3)
        return laplace_3d_windowed_kernel_ft<Real>(rpars, beta, npw, boxsize, pf, windowed_ft);
}

template <typename Real>
inline void sqrt_laplace_2d_windowed_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                               sctl::Vector<Real> &windowed_ft) {
    constexpr int DIM = 2;
    const auto [hpw, ws, rl] = get_PSWF_windowed_kernel_pwterms<2>(boxsize);
    const int n_fourier = DIM * sctl::pow<2>(npw / 2) + 1;
    windowed_ft.ReInit(n_fourier);

    const auto [c0, c1, g0d2, c4] = pf.intvals(beta);
    const int iw = pf.workarray[0] - 1;
    const int n_terms = pf.workarray[4];

    std::array<Real, 1000> coeffs{0};
    std::vector<Real> wprolate(n_terms + 3 + iw);
    for (int i = 0; i < n_terms + 3 + iw; ++i)
        wprolate[i] = pf.workarray[i];

    legeinte(&wprolate[iw], n_terms, coeffs.data());
    coeffs[0] = 0.0;

    const int n_quad = 200;
    std::array<Real, n_quad> xs, whts, fvals;
    legerts(1, n_quad, xs.data(), whts.data());

    const double factor = 0.25 * boxsize * rl * 3;
    for (int i = 0; i < n_quad; ++i) {
        xs[i] = factor * (xs[i] + 1);
        whts[i] *= factor;
    }

    for (int i = 0; i < n_quad; ++i) {
        auto legint = [&](Real x) {
            Real fval;
            if (std::abs(x) < 1.0)
                legeexev(x, fval, coeffs.data(), n_terms + 1);
            else if (x >= 1.0)
                fval = c0;
            else
                fval = -c0;
            return fval;
        };

        const Real fval0 = legint(xs[i] / boxsize);
        const Real fval1 = legint((rl + xs[i]) / boxsize);
        const Real fval2 = legint((rl - xs[i]) / boxsize);

        fvals[i] = fval0 - 0.5 * fval1 + 0.5 * fval2;
    }

    for (int i = 0; i < n_fourier; ++i) {
        const Real rk = std::sqrt(Real(i)) * hpw;
        windowed_ft[i] = Real{0.0};
        for (int j = 0; j < n_quad; ++j) {
            const Real z = rk * xs[j];
            const Real dj0 = (i == 0) ? 1.0 : util::cyl_bessel_j(0, z);

            windowed_ft[i] += dj0 * fvals[j] * whts[j] / c0;
        }
        windowed_ft[i] *= ws;
    }
}

template <typename Real>
inline void sqrt_laplace_3d_windowed_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                               sctl::Vector<Real> &windowed_ft) {
    constexpr int DIM = 3;
    const auto [hpw, ws, rl] = get_PSWF_windowed_kernel_pwterms<3>(boxsize);
    const int n_fourier = DIM * sctl::pow<2>(npw / 2) + 1;
    windowed_ft.ReInit(n_fourier);

    // calculate Legendre expansion coefficients of x\psi_0^c(x)
    int iw = pf.workarray[0] - 1;
    int n_terms = pf.workarray[4];
    std::array<Real, 1000> coeffs0, coeffs1, coeffs2, fvals;
    for (int i = 0; i <= n_terms; ++i)
        coeffs0[i] = 0.0;

    for (int i = n_terms; i >= 2; --i) {
        coeffs0[i] += pf.workarray[iw + i - 1] * i / (2 * i - 1.0);
        coeffs0[i - 2] += pf.workarray[iw + i - 1] * (i - 1) / (2 * i - 1.0);
    }
    coeffs0[1] += pf.workarray[iw];

    // FIXME: This wouldn't be necessary if Prolate0Fun had the proper value types
    std::vector<Real> wprolate(n_terms + 3 + iw);
    for (int i = 0; i < n_terms + 3 + iw; ++i)
        wprolate[i] = pf.workarray[i];

    // calculate Legendre expansion coefficients of \int_0^x t\psi_0^c(t)dt
    legeinte(coeffs0.data(), n_terms, coeffs1.data());
    Real fval;
    legeexev(Real(0.0), fval, coeffs1.data(), n_terms + 1);
    coeffs1[0] -= fval;
    legeexev(Real(1.0), fval, coeffs1.data(), n_terms + 1);
    Real c1 = fval;

    // calculate Legendre expansion coefficients of \int_0^x \psi_0^c(t)dt
    legeinte(&wprolate[iw], n_terms, coeffs2.data());
    legeexev(Real(1.0), fval, coeffs2.data(), n_terms);
    double c0 = fval;
    legeexev(Real(-1.0), fval, coeffs2.data(), n_terms);

    Real dl = 0.5 * rl;

    int n_quad = 200;
    std::array<Real, 200> xs, whts;
    legerts(1, n_quad, xs.data(), whts.data());
    for (int i = 0; i < n_quad; ++i) {
        xs[i] = (xs[i] + 1) / 2 * dl * 3;
        whts[i] = whts[i] / 2 * dl * 3;
    }

    for (int i = 0; i < n_quad; ++i) {
        Real xval1 = xs[i] / boxsize;
        Real fval1;
        if (xval1 < 1.0)
            legeexev(xval1, fval1, coeffs1.data(), n_terms);
        else
            fval1 = c1;

        //  window function
        Real xval0 = (xs[i] - dl - boxsize) / boxsize;
        Real fval0;
        if (std::abs(xval0) < 1.0)
            legeexev(xval0, fval0, coeffs2.data(), n_terms);
        else if (xval0 >= 1.0)
            fval0 = c0;
        else if (xval0 <= -1.0)
            fval0 = 0;

        fvals[i] = fval1 / c1 * (1 - fval0 / c0);
    }

    for (int i = 0; i < n_fourier; ++i) {
        Real rk = std::sqrt(Real(i)) * hpw;
        windowed_ft[i] = 0;
        for (int j = 0; j < n_quad; ++j) {
            if (i == 0)
                windowed_ft[i] += fvals[j] * whts[j];
            else
                windowed_ft[i] += fvals[j] * whts[j] * sin(rk * xs[j]) / (rk * xs[j]);
        }
        windowed_ft[i] *= ws;
    }
}

template <typename Real, int DIM>
inline void sqrt_laplace_windowed_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                            sctl::Vector<Real> &windowed_ft) {
    if constexpr (DIM == 2)
        return sqrt_laplace_2d_windowed_kernel_ft<Real>(rpars, beta, npw, boxsize, pf, windowed_ft);
    if constexpr (DIM == 3)
        return sqrt_laplace_3d_windowed_kernel_ft<Real>(rpars, beta, npw, boxsize, pf, windowed_ft);
}

template <typename Real>
inline void stokes_2d_windowed_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                         sctl::Vector<Real> &windowed_ft) {
    const Real psi0 = pf.eval_val(0.0);
    const Real rl = boxsize * (std::sqrt(2.0) + 1.0);
    const Real rl4 = rl * rl * rl * rl;
    const Real hpw = 1.0 / boxsize;
    const int nfourier = 2 * (npw / 2) * (npw / 2);

    windowed_ft.ReInit(nfourier + 1);
    for (int i = 0; i <= nfourier; ++i) {
        const Real xi = i * hpw;
        const Real xval = xi * boxsize / beta;

        Real fval = 0.0;
        if (xval <= 1.0) {
            auto [psi, dpsi] = pf.eval_val_derivative(xval);
            fval = (psi - 0.5 * xval * dpsi) / psi0;
        }

        const Real x = rl * xi;
        Real tker;
        if (i == 0) {
            tker = -rl4 / 64.0;
        } else if (x > 0.2) {
            Real j0 = util::cyl_bessel_j(0, x);
            Real j1 = util::cyl_bessel_j(1, x);
            tker = (-1.0 + j0 + 0.5 * x * j1) / (xi * xi * xi * xi);
        } else {
            Real x2 = x * x, x4 = x2 * x2, x6 = x2 * x4, x8 = x2 * x6, x10 = x2 * x8;
            tker =
                (-1.0 / 64 + x2 / 1152 - x4 / 49152 + x6 / 3686400.0 - x8 / 4.24673280e8 + x10 / 6.9363302400e10) * rl4;
        }

        windowed_ft[i] = fval * tker;
    }
}

template <typename Real>
inline void stokes_3d_windowed_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                         sctl::Vector<Real> &windowed_ft) {
    const Real psi0 = pf.eval_val(0.0);
    const Real rl = boxsize * (std::sqrt(3.0) + 1.0);
    const auto [hpw, ws, _] = get_PSWF_windowed_kernel_pwterms<3>(boxsize);
    const double rl4 = sctl::pow<4>(rl);
    const int nfourier = 3 * (npw / 2) * (npw / 2);

    windowed_ft.ReInit(nfourier + 1);
    for (int i = 0; i <= nfourier; ++i) {
        const Real xi = sqrt(Real(i)) * hpw;
        const Real xval = xi * boxsize / beta;

        const Real fval = [&]() {
            if (xval <= 1.0) {
                auto [psi, dpsi] = pf.eval_val_derivative(xval);
                return (psi - 0.5 * xval * dpsi) / psi0;
            } else
                return 0.0;
        }();

        const Real tker = [&]() {
            const Real x = rl * xi;

            if (x > 0.2) {
                return -(1.0 + 0.5 * cos(x) - 1.5 * sin(x) / x) / (xi * xi * xi * xi);
            } else {
                Real x2 = x * x, x4 = x2 * x2, x6 = x2 * x4, x8 = x2 * x6, x10 = x2 * x8;
                return -(1.0 / 120 - x2 / 2520 + x4 / 120960 - x6 / 9979200.0 + x8 / 1.245404160e9 -
                         x10 / 2.17945728e11) *
                       rl4;
            }
        }();

        windowed_ft[i] = fval * tker * ws;
    }
}

template <typename Real, int DIM>
inline void stokes_windowed_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                      sctl::Vector<Real> &windowed_ft) {
    if constexpr (DIM == 2)
        return stokes_2d_windowed_kernel_ft<Real>(rpars, beta, npw, boxsize, pf, windowed_ft);
    if constexpr (DIM == 3)
        return stokes_3d_windowed_kernel_ft<Real>(rpars, beta, npw, boxsize, pf, windowed_ft);
}

template <typename Real, int DIM>
void get_windowed_kernel_ft(dmk_ikernel kernel, const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                            sctl::Vector<Real> &windowed_ft) {
    switch (kernel) {
    case dmk_ikernel::DMK_YUKAWA:
        return yukawa_windowed_kernel_ft<Real, DIM>(rpars, beta, npw, boxsize, pf, windowed_ft);
    case dmk_ikernel::DMK_LAPLACE:
        return laplace_windowed_kernel_ft<Real, DIM>(rpars, beta, npw, boxsize, pf, windowed_ft);
    case dmk_ikernel::DMK_SQRT_LAPLACE:
        return sqrt_laplace_windowed_kernel_ft<Real, DIM>(rpars, beta, npw, boxsize, pf, windowed_ft);
    case dmk_ikernel::DMK_STOKESLET:
        return stokes_windowed_kernel_ft<Real, DIM>(rpars, beta, npw, boxsize, pf, windowed_ft);
    case dmk_ikernel::DMK_STRESSLET:
        return stokes_windowed_kernel_ft<Real, DIM>(rpars, beta, npw, boxsize, pf, windowed_ft);
    case dmk_ikernel::DMK_LAPLACE_DIPOLE:
        return laplace_windowed_kernel_ft<Real, DIM>(rpars, beta, npw, boxsize, pf, windowed_ft);
    }
}

template <typename Real, int DIM>
void yukawa_difference_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                 sctl::Vector<Real> &diff_kernel_ft) {
    const Real bsizesmall = boxsize * 0.5;
    const Real bsizebig = boxsize;
    const Real rlambda = *rpars;
    const Real rlambda2 = rlambda * rlambda;
    const Real psi0 = pf.eval_val(0.0);
    const auto [hpw, ws] = get_PSWF_difference_kernel_pwterms<DIM>(npw, beta, boxsize);
    const int n_fourier = DIM * sctl::pow<2>(npw / 2) + 1;
    diff_kernel_ft.ReInit(n_fourier);

    const Real inv_beta = 1.0 / beta;
    for (int i = 0; i < n_fourier; ++i) {
        Real rk = sqrt((Real)i) * hpw;
        Real xi2 = rk * rk + rlambda2;
        Real xi = sqrt(xi2);
        Real xval = xi * bsizesmall * inv_beta;
        Real fval1 = (xval <= 1.0) ? pf.eval_val(xval) : 0.0;

        xval = xi * bsizebig * inv_beta;
        Real fval2 = (xval <= 1.0) ? pf.eval_val(xval) : 0.0;
        diff_kernel_ft[i] = ws * (fval1 - fval2) / (psi0 * xi2);
    }

    // re-compute fhat[0] accurately when there is a low-frequency breakdown
    if (rlambda * bsizebig / beta < 1E-4) {
        const std::array<double, 4> c = pf.intvals(beta);
        const Real bsizesmall2 = bsizesmall * bsizesmall;
        const Real bsizebig2 = bsizebig * bsizebig;

        diff_kernel_ft[0] = ws * c[2] * (bsizebig2 - bsizesmall2) / 2 +
                            ws * (bsizesmall2 * bsizesmall2 - bsizebig2 * bsizebig2) * rlambda2 * c[3] / (c[0] * 24);
    }

    // Estimate of typical sqrt cost
    const unsigned long n_flops_sqrt = std::is_same_v<Real, float> ? 3 : 5;
    // Estimate of typical cost of the rest of the kernel
    const unsigned long n_flops = n_fourier * (2 * n_flops_sqrt + 20 + 2 * 16);
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, n_flops);
}

template <typename Real>
void laplace_2d_difference_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                     sctl::Vector<Real> &diff_kernel_ft) {
    constexpr int DIM = 2;
    const Real bsizesmall = boxsize * 0.5;
    const Real bsizebig = boxsize;
    const auto [hpw, ws] = get_PSWF_difference_kernel_pwterms<DIM>(npw, beta, boxsize);
    const int n_fourier = DIM * sctl::pow<2>(npw / 2) + 1;
    diff_kernel_ft.ReInit(n_fourier);

    const auto [c0, c1, g0d2, c3] = pf.intvals(beta);
    const Real psi0 = pf.eval_val(0.0);

    diff_kernel_ft[0] = 0.5 * ws * g0d2 * (bsizesmall * bsizesmall - bsizebig * bsizebig);
    for (int i = 1; i < n_fourier; ++i) {
        const Real rk = std::sqrt(Real(i)) * hpw;
        const Real xval1 = rk * bsizesmall / beta;
        const Real xval2 = rk * bsizebig / beta;
        const Real fval1 = (xval1 <= 1) ? pf.eval_val(xval1) : 0.0;
        const Real fval2 = (xval2 <= 1) ? pf.eval_val(xval2) : 0.0;
        diff_kernel_ft[i] = ws * (fval2 - fval1) / (psi0 * rk * rk);
    }
}

template <typename Real>
void laplace_3d_difference_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                     sctl::Vector<Real> &diff_kernel_ft) {
    constexpr int DIM = 3;
    const Real bsizesmall = boxsize * 0.5;
    const Real bsizebig = boxsize;
    const auto [hpw, ws] = get_PSWF_difference_kernel_pwterms<DIM>(npw, beta, boxsize);
    const int n_fourier = DIM * sctl::pow<2>(npw / 2) + 1;
    diff_kernel_ft.ReInit(n_fourier);

    auto [c0, c1, g0d2, c3] = pf.intvals(beta);

    const int n_quad = 100;
    std::array<Real, n_quad> xs, whts;
    legerts(1, n_quad, xs.data(), whts.data());

    std::array<double, n_quad> x1, x2;
    for (int i = 0; i < n_quad; ++i) {
        x1[i] = 0.5 * (xs[i] + 1) * bsizesmall;
        x2[i] = 0.5 * (xs[i] + 1) * bsizebig;
    }

    std::array<double, n_quad> fvals;
    for (int i = 0; i < n_quad; ++i) {
        const auto fval = pf.eval_val(0.5 * (xs[i] + 1));
        fvals[i] = 0.5 * fval * whts[i] / c0;
    }

    const Real bsizesmall2 = bsizesmall * bsizesmall;
    const Real bsizebig2 = bsizebig * bsizebig;
    for (int i = 0; i < n_fourier; ++i) {
        const double rk = std::sqrt(i) * hpw;

        diff_kernel_ft[i] = 0.0;
        for (int j = 0; j < n_quad; ++j)
            diff_kernel_ft[i] += (cos(rk * x1[j]) - cos(rk * x2[j])) * fvals[j];

        // for symmetric trapezoidal rule
        if (i > 0)
            diff_kernel_ft[i] *= ws / (rk * rk);
        else
            diff_kernel_ft[i] = 0.5 * ws * g0d2 * (bsizebig2 - bsizesmall2);
    }

    const int n_flops_cos = 64;
    const int n_flops_sqrt = std::is_same_v<Real, float> ? 3 : 5;
    const int n_flops = n_fourier * (n_flops_sqrt + 2 * n_flops_cos + 10);
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, n_flops);
}

template <typename Real, int DIM>
void laplace_difference_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                  sctl::Vector<Real> &diff_kernel_ft) {
    if constexpr (DIM == 2)
        return laplace_2d_difference_kernel_ft<Real>(rpars, beta, npw, boxsize, pf, diff_kernel_ft);
    if constexpr (DIM == 3)
        return laplace_3d_difference_kernel_ft<Real>(rpars, beta, npw, boxsize, pf, diff_kernel_ft);
}

template <typename Real>
void sqrt_laplace_2d_difference_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                          sctl::Vector<Real> &diff_kernel_ft) {
    constexpr int DIM = 2;
    const auto [hpw, ws] = get_PSWF_difference_kernel_pwterms<2>(npw, beta, boxsize);
    const int n_fourier = DIM * sctl::pow<2>(npw / 2) + 1;
    diff_kernel_ft.ReInit(n_fourier);

    const auto [c0, c1, g0d2, c4] = pf.intvals(beta);
    const int iw = pf.workarray[0] - 1;
    const int n_terms = pf.workarray[4];

    std::array<Real, 1000> coeffs{0};
    std::vector<Real> wprolate(n_terms + 3 + iw);
    for (int i = 0; i < n_terms + 3 + iw; ++i)
        wprolate[i] = pf.workarray[i];

    legeinte(&wprolate[iw], n_terms, coeffs.data());
    coeffs[0] = 0.0;

    const int n_quad = 100;
    std::array<Real, n_quad> xs, whts, fv1, fv2, x2, w2;
    legerts(1, n_quad, xs.data(), whts.data());

    const double factor = 0.5 * boxsize;
    for (int i = 0; i < n_quad; ++i) {
        x2[i] = factor * (xs[i] + 1);
        w2[i] = factor * whts[i];
    }

    for (int i = 0; i < n_quad; ++i) {
        auto legint = [&](Real x) {
            Real fval;
            if (std::abs(x) < 1.0)
                legeexev(x, fval, coeffs.data(), n_terms + 1);
            else
                fval = c0;
            return fval;
        };
        fv1[i] = legint(0.5 * (xs[i] + 1));
        fv2[i] = legint(xs[i] + 1);
    }

    for (int i = 0; i < n_fourier; ++i) {
        const Real rk = std::sqrt(Real(i)) * hpw;
        const Real rk2 = rk * rk;

        diff_kernel_ft[i] = Real{0.0};
        for (int j = 0; j < n_quad; ++j) {
            const Real z = rk * x2[j];
            const Real dj0 = (i == 0) ? 1.0 : util::cyl_bessel_j(0, z);

            diff_kernel_ft[i] += dj0 * (fv2[j] - fv1[j]) * w2[j] / c0;
        }

        diff_kernel_ft[i] *= ws;
    }
}

template <typename Real>
void sqrt_laplace_3d_difference_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                          sctl::Vector<Real> &diff_kernel_ft) {
    constexpr int DIM = 3;
    const auto [hpw, ws] = get_PSWF_difference_kernel_pwterms<3>(npw, beta, boxsize);
    const int n_fourier = DIM * sctl::pow<2>(npw / 2) + 1;
    diff_kernel_ft.ReInit(n_fourier);

    // calculate Legendre expansion coefficients of x\psi_0^c(x)
    int iw = pf.workarray[0];
    int n_terms = pf.workarray[4];
    std::array<Real, 1000> coeffs, coeffs0, fvals, fv1, fv2;
    for (int i = 0; i < n_terms + 1; ++i)
        coeffs0[i] = 0.0;

    for (int i = n_terms - 1; i > 1; --i) {
        coeffs0[i + 1] += pf.workarray[iw + i - 1] * (i + 1) / (2 * (i + 1) - 1.0);
        coeffs0[i - 1] += pf.workarray[iw + i - 1] * i / (2 * (i + 1) - 1.0);
    }
    coeffs0[1] += pf.workarray[iw - 1];

    // calculate Legendre expansion coefficients of \int_0^x t\psi_0^c(t)dt
    legeinte(coeffs0.data(), n_terms, coeffs.data());
    Real fval;
    legeexev(Real(0.0), fval, coeffs.data(), n_terms + 1);
    coeffs[0] -= fval;
    legeexev(Real(1.0), fval, coeffs.data(), n_terms + 1);
    Real c0 = fval;

    const int n_quad = 100;
    std::array<Real, n_quad> xs, whts, x1, x2;
    legerts(1, n_quad, xs.data(), whts.data());
    const Real bsizesmall = boxsize * 0.5;
    const Real bsizebig = boxsize;
    for (int i = 0; i < n_quad; ++i) {
        x1[i] = (xs[i] + 1) / 2 * bsizesmall;
        x2[i] = (xs[i] + 1) / 2 * bsizebig;
        whts[i] = whts[i] / 2 * bsizebig;
    }

    for (int i = 0; i < n_quad; ++i) {
        const Real xval = (xs[i] + 1) / 2;
        legeexev(xval, fval, coeffs.data(), n_terms + 1);
        Real xval2 = xval * 2;
        Real fval2;
        if (xval2 < 1.0)
            legeexev(xval2, fval2, coeffs.data(), n_terms + 1);
        else
            fval2 = c0;

        fv1[i] = fval;
        fv2[i] = fval2;
    }

    for (int i = 0; i < n_fourier; ++i) {
        Real rk = std::sqrt(Real(i)) * hpw;
        Real rk2 = rk * rk;
        diff_kernel_ft[i] = 0;
        if (i > 0) {
            for (int j = 0; j < n_quad; ++j)
                diff_kernel_ft[i] += (fv2[j] - fv1[j]) * whts[j] * std::sin(rk * x2[j]) / (rk * x2[j]);

        } else {
            for (int j = 0; j < n_quad; ++j)
                diff_kernel_ft[i] += (fv2[j] - fv1[j]) * whts[j];
        }

        diff_kernel_ft[i] *= ws / c0;
    }
}

template <typename Real, int DIM>
void sqrt_laplace_difference_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                       sctl::Vector<Real> &diff_kernel_ft) {
    if constexpr (DIM == 2)
        return sqrt_laplace_2d_difference_kernel_ft<Real>(rpars, beta, npw, boxsize, pf, diff_kernel_ft);
    if constexpr (DIM == 3)
        return sqrt_laplace_3d_difference_kernel_ft<Real>(rpars, beta, npw, boxsize, pf, diff_kernel_ft);
}

// Periodic windowed Laplace symbol on the reciprocal grid kappa=sqrt(i)*dk: ghat/kappa^2 *
// psi0(kappa*sigma1)/psi0(0), ghat = 4*pi (3D, 1/r) or -2*pi (2D, log); k=0 dropped (neutrality).
template <typename Real, int DIM>
void laplace_periodic_windowed_kernel_ft(Real ghat, Real dk, int n_fourier, Real sigma1, Prolate0Fun &pf,
                                         sctl::Vector<Real> &kernel_ft) {
    const Real c = ghat / pf.eval_val(0.0);
    kernel_ft.ReInit(n_fourier);
    kernel_ft[0] = 0;
    for (int i = 1; i < n_fourier; ++i) {
        const Real kappa = std::sqrt(Real(i)) * dk;
        const Real arg = kappa * sigma1;
        const Real psi_val = (std::abs(arg) <= 1.0) ? pf.eval_val(arg) : Real(0);
        kernel_ft[i] = c * psi_val / (kappa * kappa);
    }
}

// Free-space windowed Laplace symbol: the closed-form Vico-Greengard truncated Green's-function FT
// (kernel cut at radius rl) times the prolate spectral window psi0(kappa*sigma1)/psi0(0); k=0 kept
// finite. 3D (1/r): 1/kappa^2 -> (1-cos(kappa*rl))/kappa^2. 2D (log): the truncated-log FT tker.
template <typename Real, int DIM>
void laplace_freespace_windowed_kernel_ft(Real ghat, Real dk, int n_fourier, Real sigma1, Real rl, Prolate0Fun &pf,
                                          sctl::Vector<Real> &kernel_ft) {
    const Real c = ghat / pf.eval_val(0.0);
    kernel_ft.ReInit(n_fourier);
    if constexpr (DIM == 3) {
        kernel_ft[0] = Real(2.0 * M_PI) * rl * rl; // c*psi0*(rl^2/2), c=4*pi/psi0
        for (int i = 1; i < n_fourier; ++i) {
            const Real kappa = std::sqrt(Real(i)) * dk;
            const Real arg = kappa * sigma1;
            const Real psi_val = (std::abs(arg) <= 1.0) ? pf.eval_val(arg) : Real(0);
            kernel_ft[i] = c * psi_val * (Real(1) - std::cos(kappa * rl)) / (kappa * kappa);
        }
    } else {
        // free_2d = (2*pi/psi0)*psi_val*tker = -c*psi_val*tker (c = ghat/psi0 = -2*pi/psi0).
        const Real dfact = rl * std::log(rl);
        kernel_ft[0] = Real(2.0 * M_PI) * (Real(-0.25) * rl * rl + Real(0.5) * dfact * rl); // 2*pi*tker(0)
        for (int i = 1; i < n_fourier; ++i) {
            const Real kappa = std::sqrt(Real(i)) * dk;
            const Real arg = kappa * sigma1;
            const Real psi_val = (std::abs(arg) <= 1.0) ? pf.eval_val(arg) : Real(0);
            const Real x = rl * kappa;
            const Real tker =
                -(Real(1) - util::cyl_bessel_j(0, x)) / (kappa * kappa) + dfact * util::cyl_bessel_j(1, x) / kappa;
            kernel_ft[i] = -c * psi_val * tker;
        }
    }
}

// Periodic windowed Yukawa symbol: c*psi0(xi*sigma1)/xi^2, xi=sqrt(kappa^2+lambda^2), ghat=4*pi (3D,
// exp(-lr)/r) or 2*pi (2D, K0); k=0 finite (no neutrality).
template <typename Real, int DIM>
void yukawa_periodic_windowed_kernel_ft(Real ghat, const double *rpars, Real dk, int n_fourier, Real sigma1,
                                        Prolate0Fun &pf, sctl::Vector<Real> &kernel_ft) {
    const Real c = ghat / pf.eval_val(0.0);
    const Real lambda = *rpars;
    const Real lambda2 = lambda * lambda;
    kernel_ft.ReInit(n_fourier);
    for (int i = 0; i < n_fourier; ++i) {
        const Real xi2 = Real(i) * dk * dk + lambda2;
        const Real arg = std::sqrt(xi2) * sigma1;
        const Real psi_val = (std::abs(arg) <= 1.0) ? pf.eval_val(arg) : Real(0);
        kernel_ft[i] = c * psi_val / xi2;
    }
}

// Free-space windowed Yukawa symbol: the periodic symbol times the closed-form truncated-screened-kernel
// factor T (kernel cut at radius rl). 3D: T = 1 - e^{-l*rl}(cos(k*rl)+(l/k)sin(k*rl)); 2D: T = 1 +
// (k*rl)J1(k*rl)K0(rl*l) - (rl*l)J0(k*rl)K1(rl*l).
template <typename Real, int DIM>
void yukawa_freespace_windowed_kernel_ft(Real ghat, const double *rpars, Real dk, int n_fourier, Real sigma1, Real rl,
                                         Prolate0Fun &pf, sctl::Vector<Real> &kernel_ft) {
    const Real c = ghat / pf.eval_val(0.0);
    const Real lambda = *rpars;
    const Real lambda2 = lambda * lambda;
    kernel_ft.ReInit(n_fourier);
    Real k0_rll = 0, k1_rll = 0, exp_rll = 0;
    if constexpr (DIM == 2) {
        k0_rll = util::cyl_bessel_k(0, rl * lambda);
        k1_rll = util::cyl_bessel_k(1, rl * lambda);
    } else
        exp_rll = std::exp(-rl * lambda);
    for (int i = 0; i < n_fourier; ++i) {
        const Real kappa = std::sqrt(Real(i)) * dk;
        const Real xi2 = Real(i) * dk * dk + lambda2;
        const Real arg = std::sqrt(xi2) * sigma1;
        const Real psi_val = (std::abs(arg) <= 1.0) ? pf.eval_val(arg) : Real(0);
        Real val = c * psi_val / xi2;
        const Real x = rl * kappa;
        if constexpr (DIM == 2)
            val *= Real(1) + x * util::cyl_bessel_j(1, x) * k0_rll - rl * lambda * util::cyl_bessel_j(0, x) * k1_rll;
        else {
            const Real sin_over_k = (kappa > Real(0)) ? std::sin(x) / kappa : rl;
            val *= Real(1) - exp_rll * (std::cos(x) + lambda * sin_over_k);
        }
        kernel_ft[i] = val;
    }
}

// F(u) = \int_0^u t psi0(t) dt on [-1, 1] (even, since t psi0 is odd), F(0) = 0. Returns c0 = F(1);
// coeffs evaluate F via legeexev. This is the 3D (r^2-measure) window integral.
template <typename Real>
static Real prolate_F_coeffs(Prolate0Fun &pf, std::array<Real, 1000> &coeffs) {
    const int iw = pf.workarray[0];
    const int n_terms = pf.workarray[4];
    std::array<Real, 1000> coeffs0{};
    for (int i = n_terms - 1; i > 1; --i) {
        coeffs0[i + 1] += pf.workarray[iw + i - 1] * (i + 1) / (2 * (i + 1) - 1.0);
        coeffs0[i - 1] += pf.workarray[iw + i - 1] * i / (2 * (i + 1) - 1.0);
    }
    coeffs0[1] += pf.workarray[iw - 1];
    coeffs.fill(Real(0));
    legeinte(coeffs0.data(), n_terms, coeffs.data());
    Real fval;
    legeexev(Real(0.0), fval, coeffs.data(), n_terms + 1);
    coeffs[0] -= fval; // enforce F(0) = 0
    legeexev(Real(1.0), fval, coeffs.data(), n_terms + 1);
    return fval; // c0 = F(1)
}

// G(u) = \int_0^u psi0(t) dt on [-1, 1] (odd, since psi0 is even), G(0) = 0. Returns c0 = G(1); coeffs
// evaluate G via legeexev. The 2D (r-measure) window integral, also reused as the far-truncation taper
// ramp in both dimensions (the taper shape is free -- any smooth compact cutoff).
template <typename Real>
static Real prolate_G_coeffs(Prolate0Fun &pf, std::array<Real, 1000> &coeffs) {
    const int iw = pf.workarray[0] - 1;
    const int n_terms = pf.workarray[4];
    std::vector<Real> wprolate(n_terms + 3 + iw);
    for (int i = 0; i < n_terms + 3 + iw; ++i)
        wprolate[i] = pf.workarray[i];
    coeffs.fill(Real(0));
    legeinte(&wprolate[iw], n_terms, coeffs.data());
    Real fval;
    legeexev(Real(0.0), fval, coeffs.data(), n_terms + 1);
    coeffs[0] -= fval; // enforce G(0) = 0
    legeexev(Real(1.0), fval, coeffs.data(), n_terms + 1);
    return fval; // c0 = G(1)
}

template <typename Real>
void sqrt_laplace_3d_periodic_windowed_kernel_ft(Real dk, int n_fourier, Real b, Prolate0Fun &pf,
                                                 sctl::Vector<Real> &kernel_ft) {
    // Periodic 3D Sqrt-Laplace (1/r^2): no closed form (the smooth kernel is a sine transform of psi0),
    // built by quadrature. With F(u) = \int_0^u t psi0(t) dt and c0 = F(1):
    //   S(kappa) = 2*pi^2/kappa + (4*pi/(c0*kappa)) * \int_0^b (F(r/b) - c0) sin(kappa r)/r dr,
    // k=0 dropped (neutrality).
    kernel_ft.ReInit(n_fourier);

    std::array<Real, 1000> coeffs;
    const int n_terms = pf.workarray[4];
    const Real c0 = prolate_F_coeffs<Real>(pf, coeffs);

    // Gauss-Legendre nodes on [0, b].
    constexpr int n_quad = 200;
    std::array<Real, n_quad> r, Fm, whts;
    {
        std::array<Real, n_quad> xs;
        legerts(1, n_quad, xs.data(), whts.data());
        for (int j = 0; j < n_quad; ++j) {
            const Real u = (xs[j] + 1) * Real{0.5}; // r/b in (0,1)
            r[j] = u * b;
            Real f;
            legeexev(u, f, coeffs.data(), n_terms + 1);
            Fm[j] = f - c0;
            whts[j] *= Real{0.5} * b; // dr
        }
    }

    const Real two_pi2 = 2.0 * M_PI * M_PI;
    const Real four_pi = 4.0 * M_PI;
    kernel_ft[0] = 0; // k=0 excluded (charge neutrality)
    for (int i = 1; i < n_fourier; ++i) {
        const Real kappa = std::sqrt(Real(i)) * dk;
        Real acc = 0;
        for (int j = 0; j < n_quad; ++j)
            acc += Fm[j] * std::sin(kappa * r[j]) / r[j] * whts[j];
        kernel_ft[i] = two_pi2 / kappa + (four_pi / (c0 * kappa)) * acc;
    }
}

template <typename Real>
void sqrt_laplace_3d_freespace_windowed_kernel_ft(Real dk, int n_fourier, Real b, Real rl, Prolate0Fun &pf,
                                                  sctl::Vector<Real> &kernel_ft) {
    // Free-space 3D Sqrt-Laplace (1/r^2), same clean construction as the 2D case. W(r) = w_near*w_far:
    //   w_near(r) = F(r/b)/c0F     F(u) = \int_0^u t psi0 (the r^2-measure near window; pairs with the
    //                              short-range residual), ramp 0->1 over [0, b] then 1.
    //   w_far(r)  = 0.5*(1 + Glim((rl-b-r)/b)/c0G)   symmetric prolate ramp 1->0 over [rl-2b, rl] using
    //                              the odd G(u) = \int_0^u psi0 (shared with 2D; the taper shape is free).
    // W is C^inf and supported on [0, rl]; the plateau W = 1 covers every pair (r <= rl-2b = box
    // diagonal). 3D radial FT (Vico-Greengard eq. 7; the 1/r^2 Jacobian cancels r^2):
    //   F_hat(kappa) = (4*pi/kappa) \int_0^rl W(r) sin(kappa r)/r dr,   F_hat(0) = 4*pi \int_0^rl W(r) dr.
    kernel_ft.ReInit(n_fourier);

    std::array<Real, 1000> coeffsF, coeffsG;
    const int n_terms = pf.workarray[4];
    const Real c0F = prolate_F_coeffs<Real>(pf, coeffsF);
    const Real c0G = prolate_G_coeffs<Real>(pf, coeffsG);
    auto Flim = [&](Real u) -> Real {
        if (u >= Real(1))
            return c0F;
        Real f;
        legeexev(u, f, coeffsF.data(), n_terms + 1);
        return f;
    };
    auto Glim = [&](Real u) -> Real {
        if (u >= Real(1))
            return c0G;
        if (u <= Real(-1))
            return -c0G;
        Real g;
        legeexev(u, g, coeffsG.data(), n_terms + 1);
        return g;
    };

    // Gauss-Legendre nodes over [0, rl] with the windowed profile W precomputed at each node.
    constexpr int n_quad = 800;
    std::array<Real, n_quad> r, W, whts;
    {
        std::array<Real, n_quad> xs;
        legerts(1, n_quad, xs.data(), whts.data());
        for (int j = 0; j < n_quad; ++j) {
            r[j] = (xs[j] + 1) * Real{0.5} * rl;
            whts[j] *= Real{0.5} * rl;
            const Real w_near = Flim(r[j] / b) / c0F;
            const Real w_far = Real(0.5) * (Real(1) + Glim((rl - b - r[j]) / b) / c0G);
            W[j] = w_near * w_far;
        }
    }

    const Real four_pi = 4.0 * M_PI;
    Real f0 = 0;
    for (int j = 0; j < n_quad; ++j)
        f0 += W[j] * whts[j];
    kernel_ft[0] = four_pi * f0;
    for (int i = 1; i < n_fourier; ++i) {
        const Real kappa = std::sqrt(Real(i)) * dk;
        Real acc = 0;
        for (int j = 0; j < n_quad; ++j)
            acc += W[j] * std::sin(kappa * r[j]) / r[j] * whts[j];
        kernel_ft[i] = (four_pi / kappa) * acc;
    }
}

template <typename Real>
void sqrt_laplace_2d_periodic_windowed_kernel_ft(Real dk, int n_fourier, Real b, Prolate0Fun &pf,
                                                 sctl::Vector<Real> &kernel_ft) {
    // Periodic 2D Sqrt-Laplace (1/r): Ghat = 2*pi/kappa. The smooth kernel is a J0-Hankel transform of
    // psi0 (no closed form), built by quadrature like sqrt_laplace_2d_difference_kernel_ft. With
    // G(u) = \int_0^u psi0(t) dt and c0 = G(1):
    //   S(kappa) = 2*pi/kappa + (2*pi/c0) * \int_0^b J0(kappa r) (G(r/b) - c0) dr,   k=0 dropped (neutrality).
    kernel_ft.ReInit(n_fourier);

    std::array<Real, 1000> coeffs;
    const int n_terms = pf.workarray[4];
    const Real c0 = prolate_G_coeffs<Real>(pf, coeffs);

    // Gauss-Legendre nodes on [0, b].
    constexpr int n_quad = 200;
    std::array<Real, n_quad> r, Gm, whts;
    {
        std::array<Real, n_quad> xs;
        legerts(1, n_quad, xs.data(), whts.data());
        for (int j = 0; j < n_quad; ++j) {
            const Real u = (xs[j] + 1) * Real{0.5}; // r/b in (0,1)
            r[j] = u * b;
            Real f;
            legeexev(u, f, coeffs.data(), n_terms + 1);
            Gm[j] = f - c0;
            whts[j] *= Real{0.5} * b; // dr
        }
    }

    const Real two_pi = 2.0 * M_PI;
    kernel_ft[0] = 0; // k=0 excluded (charge neutrality)
    for (int i = 1; i < n_fourier; ++i) {
        const Real kappa = std::sqrt(Real(i)) * dk;
        Real acc = 0;
        for (int j = 0; j < n_quad; ++j)
            acc += util::cyl_bessel_j(0, kappa * r[j]) * Gm[j] * whts[j];
        // J0 integrand carries no 1/kappa (unlike the 3D sin(kappa r)/(kappa r) form).
        kernel_ft[i] = two_pi / kappa + (two_pi / c0) * acc;
    }
}

template <typename Real>
void sqrt_laplace_2d_freespace_windowed_kernel_ft(Real dk, int n_fourier, Real b, Real rl, Prolate0Fun &pf,
                                                  sctl::Vector<Real> &kernel_ft) {
    // Free-space 2D Sqrt-Laplace (1/r). The long-range kernel is the DMK windowed kernel at ESP's scale:
    // one smooth, compact profile W(r) applied to 1/r, W(r) = w_near(r) * w_far(r) with (b = r_c, the
    // short-range cutoff; G(u) = \int_0^u psi0, c0 = G(1)):
    //   w_near(r) = G(r/b)/c0                          prolate ramp 0->1 over [0, b], then 1 (pairs with
    //                                                  the short-range residual 1 - G(r/b)/c0)
    //   w_far(r)  = 0.5*(1 + Glim((rl-b-r)/b)/c0)      symmetric prolate ramp 1->0 over [rl-2b, rl]
    // (Glim clamps G to +-c0 outside [-1,1]). W is C^inf and supported on [0, rl]; the plateau W = 1 for
    // b <= r <= rl-2b covers every source-target pair (r <= rl-2b = box diagonal), so the kernel is
    // exactly 1/r there. The 1/r Jacobian cancels the r weight in the 2D FT (eq. 8, Vico-Greengard):
    //   F(kappa) = 2*pi \int_0^rl W(r) J0(kappa r) dr,   F(0) = 2*pi \int_0^rl W(r) dr.
    // W is compact (no wrap) and its ramp width 2b band-limits F to ~c/b = the grid Nyquist, so no hard
    // truncation / Gibbs floor and no aliasing.
    kernel_ft.ReInit(n_fourier);

    std::array<Real, 1000> coeffs;
    const int n_terms = pf.workarray[4];
    const Real c0 = prolate_G_coeffs<Real>(pf, coeffs);
    auto Glim = [&](Real u) -> Real {
        if (u >= Real(1))
            return c0;
        if (u <= Real(-1))
            return -c0;
        Real g;
        legeexev(u, g, coeffs.data(), n_terms + 1);
        return g;
    };

    // Gauss-Legendre nodes over [0, rl] with the windowed profile W precomputed at each node.
    constexpr int n_quad = 800;
    std::array<Real, n_quad> r, W, whts;
    {
        std::array<Real, n_quad> xs;
        legerts(1, n_quad, xs.data(), whts.data());
        for (int j = 0; j < n_quad; ++j) {
            r[j] = (xs[j] + 1) * Real{0.5} * rl;
            whts[j] *= Real{0.5} * rl;
            const Real w_near = Glim(r[j] / b) / c0;
            const Real w_far = Real(0.5) * (Real(1) + Glim((rl - b - r[j]) / b) / c0);
            W[j] = w_near * w_far;
        }
    }

    const Real two_pi = 2.0 * M_PI;
    Real f0 = 0;
    for (int j = 0; j < n_quad; ++j)
        f0 += W[j] * whts[j];
    kernel_ft[0] = two_pi * f0;
    for (int i = 1; i < n_fourier; ++i) {
        const Real kappa = std::sqrt(Real(i)) * dk;
        Real acc = 0;
        for (int j = 0; j < n_quad; ++j)
            acc += W[j] * util::cyl_bessel_j(0, kappa * r[j]) * whts[j];
        kernel_ft[i] = two_pi * acc;
    }
}

template <typename Real>
Real calc_log_windowed_kernel_value_at_zero(int dim, const Prolate0Fun &pf, Real beta, Real boxsize) {
    const Real psi0 = pf.eval_val(0.0);
    constexpr int n_quad = 100;
    std::array<Real, n_quad> xs, whts;
    legerts(1, n_quad, xs.data(), whts.data());
    for (int i = 0; i < n_quad; ++i) {
        xs[i] = 0.5 * (xs[i] + Real{1.0}) * beta / boxsize;
        whts[i] *= 0.5 * beta / boxsize;
    }

    const Real rl = boxsize * sqrt(dim * 1.0) * 2;
    const Real dfac = rl * std::log(rl);

    Real fval = 0.0;
    for (int i = 0; i < n_quad; ++i) {
        const Real xval = xs[i] * boxsize / beta;
        const Real fval0 = pf.eval_val(xval);
        const Real z = rl * xs[i];
        const Real dj0 = util::cyl_bessel_j(0, z);
        const Real dj1 = util::cyl_bessel_j(1, z);
        const Real tker = -(1 - dj0) / (xs[i] * xs[i]) + dfac * dj1 / xs[i];
        const Real fhat = tker * fval0 / psi0;
        fval += fhat * whts[i] * xs[i];
    }

    return fval;
}

// Windowed scalar-kernel FT in ESP's reciprocal-lattice convention (kappa = sqrt(i)*dk, dk =
// 2*pi/boxsize), shared by the tree's periodic root box and ESP. Routes on (kernel, freespace) to a
// dedicated periodic or free-space routine -- no freespace flag is threaded into the leaf routines.
// Periodic is the reciprocal-sum symbol (k=0 dropped for the non-screened kernels). Free-space uses the
// Vico-Greengard truncation at radius rl: a closed-form truncated symbol times the prolate spectral
// window for Laplace/Yukawa, and a windowed-profile quadrature (prolate near-window x prolate
// far-truncation) for Sqrt-Laplace, which has no closed form. Distinct from the tree's
// planewave-convention get_windowed_kernel_ft.
template <typename Real, int DIM>
void get_periodic_windowed_kernel_ft(dmk_ikernel kernel, const double *rpars, Real beta, int n_pw_periodic,
                                     Real boxsize, Real sigma1, Prolate0Fun &pf, sctl::Vector<Real> &kernel_ft,
                                     bool freespace, Real rl) {
    const Real dk = 2.0 * M_PI / boxsize;
    const int n_fourier = DIM * sctl::pow<2>(n_pw_periodic / 2) + 1;
    const Real four_pi = 4.0 * M_PI, two_pi = 2.0 * M_PI;
    const Real b = sigma1 * beta; // real-space near-window scale (= r_c)
    switch (kernel) {
    case DMK_YUKAWA:
        // K0 in 2D (2*pi), exp(-lr)/r in 3D (4*pi).
        if (freespace)
            return yukawa_freespace_windowed_kernel_ft<Real, DIM>(DIM == 2 ? two_pi : four_pi, rpars, dk, n_fourier,
                                                                  sigma1, rl, pf, kernel_ft);
        return yukawa_periodic_windowed_kernel_ft<Real, DIM>(DIM == 2 ? two_pi : four_pi, rpars, dk, n_fourier, sigma1,
                                                             pf, kernel_ft);
    case DMK_SQRT_LAPLACE:
        // 1/r in 2D, 1/r^2 in 3D. No closed-form truncated FT -> windowed-profile quadrature.
        if constexpr (DIM == 2) {
            if (freespace)
                return sqrt_laplace_2d_freespace_windowed_kernel_ft<Real>(dk, n_fourier, b, rl, pf, kernel_ft);
            return sqrt_laplace_2d_periodic_windowed_kernel_ft<Real>(dk, n_fourier, b, pf, kernel_ft);
        } else {
            if (freespace)
                return sqrt_laplace_3d_freespace_windowed_kernel_ft<Real>(dk, n_fourier, b, rl, pf, kernel_ft);
            return sqrt_laplace_3d_periodic_windowed_kernel_ft<Real>(dk, n_fourier, b, pf, kernel_ft);
        }
    default:
        // log in 2D (-2*pi), 1/r in 3D (4*pi).
        if (freespace)
            return laplace_freespace_windowed_kernel_ft<Real, DIM>(DIM == 2 ? -two_pi : four_pi, dk, n_fourier, sigma1,
                                                                   rl, pf, kernel_ft);
        return laplace_periodic_windowed_kernel_ft<Real, DIM>(DIM == 2 ? -two_pi : four_pi, dk, n_fourier, sigma1, pf,
                                                              kernel_ft);
    }
}

template <typename Real, int DIM>
inline void stokes_difference_kernel_ft(const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                                        sctl::Vector<Real> &diff_kernel_ft) {
    const Real psi0 = pf.eval_val(0.0);
    const auto intvals = pf.intvals(beta);
    const Real c0 = intvals[0], c4 = intvals[3];
    const Real bsizesmall = boxsize / 2.0;
    const Real bsizebig = boxsize;
    const auto [hpw, ws] = get_PSWF_difference_kernel_pwterms<3>(npw, beta, boxsize);
    const int nfourier = DIM * (npw / 2) * (npw / 2);

    diff_kernel_ft.ReInit(nfourier + 1);
    for (int i = 0; i <= nfourier; ++i) {
        const Real xi = sqrt(Real(i)) * hpw;

        if (xi < 1e-10) {
            diff_kernel_ft[i] = ws * c4 / c0 * (sctl::pow<4>(bsizesmall) - sctl::pow<4>(bsizebig)) / Real{24};
            continue;
        }

        const Real xval_small = xi * bsizesmall / beta;
        Real f1 = 0.0;
        if (xval_small <= 1.0) {
            auto [psi, dpsi] = pf.eval_val_derivative(xval_small);
            f1 = psi - 0.5 * xval_small * dpsi;
        }

        const Real xval_big = xi * bsizebig / beta;
        Real f2 = 0.0;
        if (xval_big <= 1.0) {
            auto [psi, dpsi] = pf.eval_val_derivative(xval_big);
            f2 = psi - 0.5 * xval_big * dpsi;
        }

        diff_kernel_ft[i] = ws * (f2 - f1) / psi0 / sctl::pow<4>(xi);
    }
}

template <typename Real, int DIM>
void get_difference_kernel_ft(bool init, dmk_ikernel kernel, const double *rpars, Real beta, int npw, Real boxsize,
                              Prolate0Fun &pf, sctl::Vector<Real> &diff_kernel_ft) {
    if (init || kernel == DMK_YUKAWA)
        switch (kernel) {
        case DMK_YUKAWA:
            return yukawa_difference_kernel_ft<Real, DIM>(rpars, beta, npw, boxsize, pf, diff_kernel_ft);
        case DMK_LAPLACE:
            return laplace_difference_kernel_ft<Real, DIM>(rpars, beta, npw, boxsize, pf, diff_kernel_ft);
        case DMK_SQRT_LAPLACE:
            return sqrt_laplace_difference_kernel_ft<Real, DIM>(rpars, beta, npw, boxsize, pf, diff_kernel_ft);
        case DMK_STOKESLET:
            return stokes_difference_kernel_ft<Real, DIM>(rpars, beta, npw, boxsize, pf, diff_kernel_ft);
        case DMK_STRESSLET:
            return stokes_difference_kernel_ft<Real, DIM>(rpars, beta, npw, boxsize, pf, diff_kernel_ft);
        case DMK_LAPLACE_DIPOLE:
            return laplace_difference_kernel_ft<Real, DIM>(rpars, beta, npw, boxsize, pf, diff_kernel_ft);
        default:
            throw std::runtime_error("Unsupported kernel " + std::to_string(kernel));
        }

    const Real scale_factor = [](dmk_ikernel kernel) -> Real {
        switch (kernel) {
        case DMK_LAPLACE:
            return DIM == 2 ? Real(1.0) : Real(2.0);
        case DMK_SQRT_LAPLACE:
            return DIM == 2 ? Real(2.0) : Real(4.0);
        case DMK_STOKESLET:
            return DIM == 2 ? Real(0.25) : Real(0.5);
        case DMK_STRESSLET:
            return DIM == 2 ? Real(0.25) : Real(0.5);
        case DMK_LAPLACE_DIPOLE:
            return DIM == 2 ? Real(1.0) : Real(2.0);
        default:
            throw std::runtime_error("Invalid kernel type: " + std::to_string(kernel));
        }
    }(kernel);

    for (int i = 0; i < diff_kernel_ft.Dim(); ++i)
        diff_kernel_ft[i] *= scale_factor;

    sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, diff_kernel_ft.Dim());
}

template <typename T>
FourierData<T>::FourierData(dmk_ikernel kernel, int n_dim, T eps, int n_pw_win, int n_pw_diff, T fparam, double beta,
                            const sctl::Vector<T> &boxsize_)
    : kernel_(kernel), n_dim_(n_dim), fparam_(fparam), box_sizes_(boxsize_), n_levels_(boxsize_.Dim()) {

    beta_ = beta;
    prolate0_fun = Prolate0Fun(beta_, 10000);
    difference_kernels_.ReInit(n_levels_);

    std::tie(windowed_kernel_.hpw, windowed_kernel_.ws, windowed_kernel_.rl) =
        get_PSWF_windowed_kernel_pwterms(n_dim_, box_sizes_[0]);

    difference_kernels_[0].rl = windowed_kernel_.rl;
    for (int i = 1; i < n_levels_; ++i)
        difference_kernels_[i].rl = 0.5 * difference_kernels_[i - 1].rl;

    for (int i_level = 0; i_level < n_levels_; ++i_level)
        std::tie(difference_kernels_[i_level].hpw, difference_kernels_[i_level].ws) =
            get_PSWF_difference_kernel_pwterms(n_dim, n_pw_diff, beta_, box_sizes_[i_level]);
}

template <typename Real>
Real yukawa_windowed_kernel_value_at_zero(int n_dim, Real rlambda, Real beta, Real boxsize, Real rl,
                                          const Prolate0Fun &pf) {
    constexpr Real two_over_pi = 2.0 / M_PI;
    const Real rlambda2 = rlambda * rlambda;

    const bool near_correction = (rlambda * boxsize / beta) < 1.0E-2;
    Real dk0, dk1, delam;
    if (near_correction) {
        if (n_dim == 2) {
            dk0 = util::cyl_bessel_k(0, rl * rlambda);
            dk1 = util::cyl_bessel_k(1, rl * rlambda);
        } else if (n_dim == 3)
            delam = std::exp(-rl * rlambda);
    }

    const Real psi0 = pf.eval_val(0.0);
    const Real fone = pf.eval_val(1.0);

    const int n_quad = 100;
    std::array<Real, n_quad> xs, whts;
    legerts(1, n_quad, xs.data(), whts.data());

    for (int i = 0; i < n_quad; ++i) {
        xs[i] = Real(0.5) * (xs[i] + 1) * beta / boxsize;
        whts[i] = Real(0.5) * whts[i] * beta / boxsize;
    }

    Real fval = 0.0;
    for (int i = 0; i < n_quad; ++i) {
        const Real xi2 = xs[i] * xs[i] + rlambda2;
        const Real xval = std::sqrt(xi2) * boxsize / beta;

        const Real fval0 = (xval <= 1.0) ? pf.eval_val(xval) : 0.0;

        Real fhat = fval0 / psi0 / xi2;
        if (near_correction) {
            const Real xsc = rl * xs[i];
            if (n_dim == 2)
                fhat *= -rl * rlambda * util::cyl_bessel_j(0, xsc) * dk1 + 1 + xsc * util::cyl_bessel_j(1, xsc) * dk0;
            else if (n_dim == 3)
                fhat *= 1 - delam * (std::cos(xsc) + rlambda / xs[i] * std::sin(xsc));
        }

        if (n_dim == 2)
            fval += fhat * whts[i] * xs[i];
        else if (n_dim == 3)
            fval += fhat * whts[i] * xs[i] * xs[i] * two_over_pi;
    }

    return fval;
}

template <typename T>
T FourierData<T>::yukawa_windowed_kernel_value_at_zero(int i_level) {
    const T bsize = (i_level == 0) ? 0.5 * box_sizes_[i_level] : box_sizes_[i_level];
    const T rl = difference_kernels_[std::max(i_level, 0)].rl;

    return dmk::yukawa_windowed_kernel_value_at_zero(n_dim_, fparam_, beta_, bsize, rl, prolate0_fun);
}

namespace {
// Modified Bessel I0(z) via its ascending series (double, generation-time only).
double besseli0_series(double z) {
    const double z2 = 0.25 * z * z;
    double term = 1.0, sum = 1.0;
    for (int k = 1; k < 64; ++k) {
        term *= z2 / (double(k) * double(k));
        sum += term;
        if (term <= 1e-18 * sum)
            break;
    }
    return sum;
}

// Regular (non-log) part of K0: KR(z) = K0(z) + log(z)*I0(z). Finite and even in
// z, so the residual's log singularity can be isolated without inf-inf at r=0.
// KR(z) = (log2 - gamma)*I0(z) + sum_{k>=1} (z^2/4)^k/(k!)^2 * H_k.
double besselk0_regular_series(double z) {
    constexpr double log2 = 0.6931471805599453;
    constexpr double euler_gamma = 0.5772156649015329;
    const double z2 = 0.25 * z * z;
    double term = 1.0, Hk = 0.0, sum = 0.0;
    for (int k = 1; k < 64; ++k) {
        term *= z2 / (double(k) * double(k));
        Hk += 1.0 / double(k);
        const double add = term * Hk;
        sum += add;
        if (add <= 1e-18 * (sum + 1.0))
            break;
    }
    return (log2 - euler_gamma) * besseli0_series(z) + sum;
}
} // namespace

template <typename Real>
typename FourierData<Real>::LocalCorrectionCoeffs FourierData<Real>::local_correction_coeffs(int i_level,
                                                                                             int n_digits) {
    LocalCorrectionCoeffs out;

    constexpr Real two_over_pi = 2.0 / M_PI;
    const Real &rlambda = fparam_;
    const Real rlambda2 = rlambda * rlambda;
    const Real psi0 = prolate0_fun.eval_val(0.0);

    constexpr int n_quad = 100;
    // FIXME: Ideally these would use the Real type, but we get slightly lower errors downstream with double
    std::vector<double> xs(n_quad), whts(n_quad), xs_base(n_quad), whts_base(n_quad), fhat(n_quad);
    legerts(1, n_quad, xs_base.data(), whts_base.data());

    auto bsize = box_sizes_[i_level];
    if (i_level == 0)
        bsize *= 0.5;

    Real scale_factor = beta_ / (2.0 * bsize);
    for (int i = 0; i < n_quad; ++i) {
        xs[i] = scale_factor * (xs_base[i] + 1.0);
        whts[i] = scale_factor * whts_base[i];
    }

    const bool near_correction = rlambda * bsize / beta_ < 1E-2;
    Real dk0, dk1, delam;
    const Real rl = difference_kernels_[std::max(i_level, 0)].rl;
    if (near_correction) {
        Real arg = rl * rlambda;
        if (n_dim_ == 2) {
            dk0 = util::cyl_bessel_k(0, arg);
            dk1 = util::cyl_bessel_k(1, arg);
        } else
            delam = std::exp(-arg);
    }

    for (int i = 0; i < n_quad; ++i) {
        const Real xi2 = xs[i] * xs[i] + rlambda2;
        const Real xval = sqrt(xi2) * bsize / beta_;
        if (xval <= 1.0) {
            fhat[i] = prolate0_fun.eval_val(xval) / (psi0 * xi2);
        } else {
            fhat[i] = 0.0;
            continue;
        }
        fhat[i] *= (n_dim_ == 2) ? whts[i] * xs[i] : whts[i] * xs[i] * xs[i] * two_over_pi;

        if (near_correction) {
            Real xsc = rl * xs[i];
            if (n_dim_ == 2)
                fhat[i] *=
                    -rl * rlambda * util::cyl_bessel_j(0, xsc) * dk1 + 1 + xsc * util::cyl_bessel_j(1, xsc) * dk0;
            else
                fhat[i] *= 1 - delam * (std::cos(xsc) + rlambda / xs[i] * std::sin(xsc));
        }
    }

    if (n_dim_ == 3) {
        // Unified residual path (mirrors the other kernels): fit the smooth,
        // bounded function Q(r) = r * residual(r) = exp(-lambda*r) - r*W(r),
        // where W(r) = Σ_j sinc(r*xs_j) fhat_j is the windowed far-field. At
        // runtime the residual is horner(r*rsc+cen)*Rinv, with Rinv supplying
        // the 1/r singularity and the exp folded into the polynomial (see
        // YukawaPolyEvaluator3D). x in [-1,1] maps to r in [0,bsize], matching
        // the rsc=2/bsize, cen=-1 used at the call site in tree.cpp.
        auto f = [&](double x) {
            const double r = (x + 1.0) * 0.5 * bsize;
            double W = 0.0;
            for (int j = 0; j < n_quad; ++j) {
                const double dd = r * xs[j];
                W += (dd == 0.0 ? 1.0 : std::sin(dd) / dd) * fhat[j];
            }
            return std::exp(-double(rlambda) * r) - r * W;
        };
        out.reg_poly = make_polyfit_abs_error<double>(n_digits, f, -1.0, 1.0);
        if (out.reg_poly.empty())
            throw std::runtime_error(
                std::format("Yukawa local correction fit failed at level {} (lambda={}, bsize={}, "
                            "lambda*bsize={}): exp(-lambda*r) too stiff for a degree<32 polynomial",
                            i_level, double(rlambda), double(bsize), double(rlambda * bsize)));
        return out;
    }

    // 2D log-split fit. The scaled variable x in [-1,1] maps to r^2 in [0,bsize^2]
    // via r = bsize*sqrt((x+1)/2), matching rsc=2/bsize^2, cen=-1 at the call site.
    // residual(r) = K0(lambda*r) + C(r) = log(r/bsize)*PA(x) + PB(x), with
    //   PA(x) = -I0(lambda*r)                              (the log coefficient)
    //   PB(x) = KR(lambda*r) - (log(lambda)+log(bsize))*I0(lambda*r) + C(r)
    //   C(r)  = -Σ_j J0(r*xs_j) fhat_j                     (windowed far-field)
    // KR(z) = K0(z)+log(z)*I0(z) is the regular part of K0, so PB is finite at r=0.
    const double lam = double(rlambda);
    const double bs = double(bsize);
    auto rmap = [bs](double x) { return bs * std::sqrt(0.5 * (x + 1.0)); };
    auto Cfun = [&](double r) {
        double C = 0.0;
        for (int j = 0; j < n_quad; ++j)
            C -= util::cyl_bessel_j(0, r * xs[j]) * fhat[j];
        return C;
    };
    auto fa = [&](double x) { return -besseli0_series(lam * rmap(x)); };
    auto fb = [&](double x) {
        const double r = rmap(x);
        const double z = lam * r;
        return besselk0_regular_series(z) - (std::log(lam) + std::log(bs)) * besseli0_series(z) + Cfun(r);
    };
    out.log_poly = make_polyfit_abs_error<double>(n_digits, fa, -1.0, 1.0);
    out.reg_poly = make_polyfit_abs_error<double>(n_digits, fb, -1.0, 1.0);
    if (out.log_poly.empty() || out.reg_poly.empty())
        throw std::runtime_error(
            std::format("Yukawa 2D local correction fit failed at level {} (lambda={}, bsize={}, lambda*bsize={}): "
                        "residual too stiff for a degree<32 polynomial",
                        i_level, lam, bs, lam * bs));
    return out;
}

template <typename T>
void FourierData<T>::calc_planewave_coeff_matrices(int i_level, int n_order, int n_pw,
                                                   sctl::Vector<std::complex<T>> &prox2pw_vec,
                                                   sctl::Vector<std::complex<T>> &pw2poly_vec) const {
    auto hpw = (i_level + 1) ? difference_kernels_[i_level].hpw : windowed_kernel_.hpw;
    auto bsize = box_sizes_[std::max(i_level, 0)];
    dmk::calc_planewave_coeff_matrices(bsize, hpw, n_pw, n_order, prox2pw_vec, pw2poly_vec);
}

template struct FourierData<float>;
template struct FourierData<double>;

template void get_windowed_kernel_ft<float, 2>(dmk_ikernel kernel, const double *rpars, float beta, int npw,
                                               float boxsize, Prolate0Fun &pf, sctl::Vector<float> &radialft);
template void get_windowed_kernel_ft<float, 3>(dmk_ikernel kernel, const double *rpars, float beta, int npw,
                                               float boxsize, Prolate0Fun &pf, sctl::Vector<float> &radialft);
template void get_windowed_kernel_ft<double, 2>(dmk_ikernel kernel, const double *rpars, double beta, int npw,
                                                double boxsize, Prolate0Fun &pf, sctl::Vector<double> &radialft);
template void get_windowed_kernel_ft<double, 3>(dmk_ikernel kernel, const double *rpars, double beta, int npw,
                                                double boxsize, Prolate0Fun &pf, sctl::Vector<double> &radialft);
template void get_difference_kernel_ft<float, 2>(bool init, dmk_ikernel kernel, const double *rpars, float beta,
                                                 int npw, float boxsize, Prolate0Fun &pf,
                                                 sctl::Vector<float> &diff_kernel_ft);
template void get_difference_kernel_ft<float, 3>(bool init, dmk_ikernel kernel, const double *rpars, float beta,
                                                 int npw, float boxsize, Prolate0Fun &pf,
                                                 sctl::Vector<float> &diff_kernel_ft);
template void get_difference_kernel_ft<double, 2>(bool init, dmk_ikernel kernel, const double *rpars, double beta,
                                                  int npw, double boxsize, Prolate0Fun &pf,
                                                  sctl::Vector<double> &diff_kernel_ft);
template void get_difference_kernel_ft<double, 3>(bool init, dmk_ikernel kernel, const double *rpars, double beta,
                                                  int npw, double boxsize, Prolate0Fun &pf,
                                                  sctl::Vector<double> &diff_kernel_ft);
template void get_periodic_windowed_kernel_ft<float, 2>(dmk_ikernel kernel, const double *rpars, float beta,
                                                        int n_pw_periodic, float boxsize, float sigma1, Prolate0Fun &pf,
                                                        sctl::Vector<float> &kernel_ft, bool freespace, float rl);
template void get_periodic_windowed_kernel_ft<float, 3>(dmk_ikernel kernel, const double *rpars, float beta,
                                                        int n_pw_periodic, float boxsize, float sigma1, Prolate0Fun &pf,
                                                        sctl::Vector<float> &kernel_ft, bool freespace, float rl);
template void get_periodic_windowed_kernel_ft<double, 2>(dmk_ikernel kernel, const double *rpars, double beta,
                                                         int n_pw_periodic, double boxsize, double sigma1,
                                                         Prolate0Fun &pf, sctl::Vector<double> &kernel_ft,
                                                         bool freespace, double rl);
template void get_periodic_windowed_kernel_ft<double, 3>(dmk_ikernel kernel, const double *rpars, double beta,
                                                         int n_pw_periodic, double boxsize, double sigma1,
                                                         Prolate0Fun &pf, sctl::Vector<double> &kernel_ft,
                                                         bool freespace, double rl);

template float calc_log_windowed_kernel_value_at_zero<float>(int dim, const Prolate0Fun &pf, float beta, float boxsize);
template double calc_log_windowed_kernel_value_at_zero<double>(int dim, const Prolate0Fun &pf, double beta,
                                                               double boxsize);

} // namespace dmk
