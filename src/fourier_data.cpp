#include <cmath>
#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/legeexps.hpp>
#include <dmk/planewave.hpp>
#include <dmk/prolate0_fun.hpp>
#include <dmk/types.hpp>

#include <complex.h>
#include <sctl.hpp>
#include <stdexcept>
#include <string>

namespace dmk {

template <int DIM>
std::tuple<int, double, double> get_PSWF_difference_kernel_pwterms(dmk_ikernel kernel, int ndigits, double boxsize) {
    int npw;
    double hpw;

    if (kernel == dmk_ikernel::DMK_SQRT_LAPLACE && DIM == 3) {
        if (ndigits <= 3) {
            npw = 13;
            hpw = M_PI * 0.662 / boxsize;
        } else if (ndigits <= 6) {
            npw = 27;
            hpw = M_PI * 0.667 / boxsize;
        } else if (ndigits <= 9) {
            npw = 39;
            hpw = M_PI * 0.6625 / boxsize;
        } else if (ndigits <= 12) {
            npw = 55;
            hpw = M_PI * 0.667 / boxsize;
        }
    } else {
        if (ndigits <= 3) {
            npw = 13;
            hpw = M_PI * 0.662 / boxsize;
        } else if (ndigits <= 6) {
            // FIXME: was 25, but currently the code requires it be the same as the windowed kernel
            // Breaks with mk_tensor_product_fourier_transform due to inconsistent number of modes
            npw = 27;
            hpw = M_PI * 0.6686 / boxsize;
        } else if (ndigits <= 9) {
            npw = 39;
            hpw = M_PI * 0.6625 / boxsize;
        } else if (ndigits <= 12) {
            npw = 53;
            hpw = M_PI * 0.6677 / boxsize;
        }
    }

    constexpr double factor = 0.5 / sctl::pow<DIM - 1>(M_PI);
    double ws = sctl::pow<DIM>(hpw) * factor;

    return std::make_tuple(npw, hpw, ws);
}

inline std::tuple<int, double, double> get_PSWF_difference_kernel_pwterms(dmk_ikernel kernel, int dim, int ndigits,
                                                                          double boxsize) {
    if (dim == 2)
        return get_PSWF_difference_kernel_pwterms<2>(kernel, ndigits, boxsize);
    if (dim == 3)
        return get_PSWF_difference_kernel_pwterms<3>(kernel, ndigits, boxsize);
    throw std::runtime_error("Invalid dimension " + std::to_string(dim) + " provided");
}

template <int DIM>
std::tuple<int, double, double, double> get_PSWF_windowed_kernel_pwterms(int ndigits, double boxsize) {
    int npw;
    double hpw, ws, rl;

    if (ndigits <= 3) {
        npw = 13;
        hpw = M_PI * 0.34 / boxsize;
    } else if (ndigits <= 6) {
        npw = 27;
        hpw = M_PI * 0.357 / boxsize;
    } else if (ndigits <= 9) {
        npw = 39;
        hpw = M_PI * 0.357 / boxsize;
    } else if (ndigits <= 12) {
        npw = 53;
        hpw = M_PI * 0.338 / boxsize;
    }

    constexpr double factor = 0.5 / sctl::pow<DIM - 1>(M_PI);
    ws = sctl::pow<DIM>(hpw) * factor;

    constexpr double two_sqrt_dim = DIM == 2 ? 2 * 1.4142135623730951 : 2 * 1.7320508075688772;
    rl = boxsize * two_sqrt_dim;

    return std::make_tuple(npw, hpw, ws, rl);
}

inline std::tuple<int, double, double, double> get_PSWF_windowed_kernel_pwterms(int dim, int ndigits, double boxsize) {
    if (dim == 2)
        return get_PSWF_windowed_kernel_pwterms<2>(ndigits, boxsize);
    if (dim == 3)
        return get_PSWF_windowed_kernel_pwterms<3>(ndigits, boxsize);
    throw std::runtime_error("Invalid dimension " + std::to_string(dim) + " provided");
}

template <typename Real, int DIM>
void yukawa_windowed_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize, Prolate0Fun &pf,
                               sctl::Vector<Real> &windowed_ft) {
    auto [npw, hpw, ws, rl] = get_PSWF_windowed_kernel_pwterms<DIM>(ndigits, boxsize);
    const int n_fourier = DIM * sctl::pow<2>(npw / 2) + 1;
    windowed_ft.ReInit(n_fourier);

    const Real rlambda = rpars[0];
    const Real rlambda2 = rlambda * rlambda;

    // determine whether one needs to smooth out the 1/(k^2+lambda^2) factor at the origin.
    // needed in the calculation of kernel-smoothing when there is low-frequency breakdown
    const bool near_correction = (rlambda * boxsize / beta < 1E-2);
    Real dk0, dk1, delam;
    if (near_correction) {
        Real arg = rl * rlambda;
        if constexpr (DIM == 2) {
            dk0 = std::cyl_bessel_j(0, arg);
            dk1 = std::cyl_bessel_j(1, arg);
        } else if constexpr (DIM == 3)
            delam = std::exp(-arg);
    }

    const Real psi0 = pf.eval_val(0);
    for (int i = 0; i < n_fourier; ++i) {
        const Real rk = sqrt((Real)i) * hpw;
        const Real xi2 = rk * rk + rlambda2;
        const Real xi = sqrt(xi2);
        const Real xval = xi * boxsize / beta;
        const Real fval = (xval <= 1.0) ? pf.eval_val(xval) : 0.0;

        windowed_ft[i] = ws * fval / (psi0 * xi2);

        if (near_correction) {
            const Real xsc = rl * rk;
            if constexpr (DIM == 2) {
                using std::cyl_bessel_j;
                windowed_ft[i] *=
                    -rl * rlambda * cyl_bessel_j(0, xsc) * dk1 + Real{1.0} + xsc * cyl_bessel_j(1, xsc) * dk0;
            } else if constexpr (DIM == 3) {
                windowed_ft[i] *= 1 - delam * (cos(xsc) + rlambda / rk * sin(xsc));
            }
        }
    }
}

template <typename Real>
void laplace_2d_windowed_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize, Prolate0Fun &pf,
                                   sctl::Vector<Real> &windowed_ft) {
    constexpr int DIM = 2;
    const auto [npw, hpw, ws, rl] = get_PSWF_windowed_kernel_pwterms<DIM>(ndigits, boxsize);
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
            const Real dj0 = std::cyl_bessel_j(0, x);
            const Real dj1 = std::cyl_bessel_j(1, x);
            const Real tker = -(1 - dj0) / (rk * rk) + dfact * dj1 / rk;
            windowed_ft[i] *= tker;
        } else
            windowed_ft[i] *= -0.25 * rl * rl + 0.5 * dfact * rl;
    }
}

template <typename Real>
void laplace_3d_windowed_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize, Prolate0Fun &pf,
                                   sctl::Vector<Real> &windowed_ft) {
    constexpr int DIM = 3;
    const auto [npw, hpw, ws, rl] = get_PSWF_windowed_kernel_pwterms<DIM>(ndigits, boxsize);
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
inline void laplace_windowed_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize, Prolate0Fun &pf,
                                       sctl::Vector<Real> &windowed_ft) {
    if constexpr (DIM == 2)
        return laplace_2d_windowed_kernel_ft<Real>(rpars, beta, ndigits, boxsize, pf, windowed_ft);
    if constexpr (DIM == 3)
        return laplace_3d_windowed_kernel_ft<Real>(rpars, beta, ndigits, boxsize, pf, windowed_ft);
}

template <typename Real>
inline void sqrt_laplace_2d_windowed_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize,
                                               Prolate0Fun &pf, sctl::Vector<Real> &windowed_ft) {
    constexpr int DIM = 2;
    const auto [npw, hpw, ws, rl] = get_PSWF_windowed_kernel_pwterms<2>(ndigits, boxsize);
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
            const Real dj0 = (i == 0) ? 1.0 : std::cyl_bessel_j(0, z);

            windowed_ft[i] += dj0 * fvals[j] * whts[j] / c0;
        }
        windowed_ft[i] *= ws;
    }
}

template <typename Real>
inline void sqrt_laplace_3d_windowed_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize,
                                               Prolate0Fun &pf, sctl::Vector<Real> &windowed_ft) {
    constexpr int DIM = 3;
    const auto [npw, hpw, ws, rl] = get_PSWF_windowed_kernel_pwterms<3>(ndigits, boxsize);
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
inline void sqrt_laplace_windowed_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize, Prolate0Fun &pf,
                                            sctl::Vector<Real> &windowed_ft) {
    if constexpr (DIM == 2)
        return sqrt_laplace_2d_windowed_kernel_ft<Real>(rpars, beta, ndigits, boxsize, pf, windowed_ft);
    if constexpr (DIM == 3)
        return sqrt_laplace_3d_windowed_kernel_ft<Real>(rpars, beta, ndigits, boxsize, pf, windowed_ft);
}

template <typename Real, int DIM>
void get_windowed_kernel_ft(dmk_ikernel kernel, const double *rpars, Real beta, int ndigits, Real boxsize,
                            Prolate0Fun &pf, sctl::Vector<Real> &windowed_ft) {
    switch (kernel) {
    case dmk_ikernel::DMK_YUKAWA:
        return yukawa_windowed_kernel_ft<Real, DIM>(rpars, beta, ndigits, boxsize, pf, windowed_ft);
    case dmk_ikernel::DMK_LAPLACE:
        return laplace_windowed_kernel_ft<Real, DIM>(rpars, beta, ndigits, boxsize, pf, windowed_ft);
    case dmk_ikernel::DMK_SQRT_LAPLACE:
        return sqrt_laplace_windowed_kernel_ft<Real, DIM>(rpars, beta, ndigits, boxsize, pf, windowed_ft);
    }
}

template <typename Real, int DIM>
void yukawa_difference_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize, Prolate0Fun &pf,
                                 sctl::Vector<Real> &diff_kernel_ft) {
    const Real bsizesmall = boxsize * 0.5;
    const Real bsizebig = boxsize;
    const Real rlambda = *rpars;
    const Real rlambda2 = rlambda * rlambda;
    const Real psi0 = pf.eval_val(0.0);
    const auto [npw, hpw, ws] = get_PSWF_difference_kernel_pwterms<DIM>(DMK_YUKAWA, ndigits, boxsize);
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
void laplace_2d_difference_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize, Prolate0Fun &pf,
                                     sctl::Vector<Real> &diff_kernel_ft) {
    constexpr int DIM = 2;
    const Real bsizesmall = boxsize * 0.5;
    const Real bsizebig = boxsize;
    const auto [npw, hpw, ws] = get_PSWF_difference_kernel_pwterms<DIM>(DMK_LAPLACE, ndigits, boxsize);
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
void laplace_3d_difference_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize, Prolate0Fun &pf,
                                     sctl::Vector<Real> &diff_kernel_ft) {
    constexpr int DIM = 3;
    const Real bsizesmall = boxsize * 0.5;
    const Real bsizebig = boxsize;
    const auto [npw, hpw, ws] = get_PSWF_difference_kernel_pwterms<DIM>(DMK_LAPLACE, ndigits, boxsize);
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
void laplace_difference_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize, Prolate0Fun &pf,
                                  sctl::Vector<Real> &diff_kernel_ft) {
    if constexpr (DIM == 2)
        return laplace_2d_difference_kernel_ft<Real>(rpars, beta, ndigits, boxsize, pf, diff_kernel_ft);
    if constexpr (DIM == 3)
        return laplace_3d_difference_kernel_ft<Real>(rpars, beta, ndigits, boxsize, pf, diff_kernel_ft);
}

template <typename Real>
void sqrt_laplace_2d_difference_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize, Prolate0Fun &pf,
                                          sctl::Vector<Real> &diff_kernel_ft) {
    constexpr int DIM = 2;
    const auto [npw, hpw, ws] = get_PSWF_difference_kernel_pwterms<2>(DMK_SQRT_LAPLACE, ndigits, boxsize);
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
            const Real dj0 = (i == 0) ? 1.0 : std::cyl_bessel_j(0, z);

            diff_kernel_ft[i] += dj0 * (fv2[j] - fv1[j]) * w2[j] / c0;
        }

        diff_kernel_ft[i] *= ws;
    }
}

template <typename Real>
void sqrt_laplace_3d_difference_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize, Prolate0Fun &pf,
                                          sctl::Vector<Real> &diff_kernel_ft) {
    constexpr int DIM = 3;
    const auto [npw, hpw, ws] = get_PSWF_difference_kernel_pwterms<3>(DMK_SQRT_LAPLACE, ndigits, boxsize);
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
void sqrt_laplace_difference_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize, Prolate0Fun &pf,
                                       sctl::Vector<Real> &diff_kernel_ft) {
    if constexpr (DIM == 2)
        return sqrt_laplace_2d_difference_kernel_ft<Real>(rpars, beta, ndigits, boxsize, pf, diff_kernel_ft);
    if constexpr (DIM == 3)
        return sqrt_laplace_3d_difference_kernel_ft<Real>(rpars, beta, ndigits, boxsize, pf, diff_kernel_ft);
}

template <typename Real, int DIM>
void get_difference_kernel_ft(bool init, dmk_ikernel kernel, const double *rpars, Real beta, int ndigits, Real boxsize,
                              Prolate0Fun &pf, sctl::Vector<Real> &diff_kernel_ft) {
    if (init || kernel == DMK_YUKAWA)
        switch (kernel) {
        case dmk_ikernel::DMK_YUKAWA:
            return yukawa_difference_kernel_ft<Real, DIM>(rpars, beta, ndigits, boxsize, pf, diff_kernel_ft);
        case dmk_ikernel::DMK_LAPLACE:
            return laplace_difference_kernel_ft<Real, DIM>(rpars, beta, ndigits, boxsize, pf, diff_kernel_ft);
        case dmk_ikernel::DMK_SQRT_LAPLACE:
            return sqrt_laplace_difference_kernel_ft<Real, DIM>(rpars, beta, ndigits, boxsize, pf, diff_kernel_ft);
        default:
            throw std::runtime_error("Unsupported kernel " + std::to_string(kernel));
        }

    const Real scale_factor = [](dmk_ikernel kernel) -> Real {
        switch (kernel) {
        case DMK_LAPLACE:
            return DIM == 2 ? Real(1.0) : Real(2.0);
        case DMK_SQRT_LAPLACE:
            return DIM == 2 ? Real(2.0) : Real(4.0);
        default:
            throw std::runtime_error("Invalid kernel type: " + std::to_string(kernel));
        }
    }(kernel);

    for (int i = 0; i < diff_kernel_ft.Dim(); ++i)
        diff_kernel_ft[i] *= scale_factor;

    sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, diff_kernel_ft.Dim());
}

template <typename T>
FourierData<T>::FourierData(dmk_ikernel kernel, int n_dim, T eps, int n_digits, int n_pw_max, T fparam,
                            const sctl::Vector<T> &boxsize_)
    : kernel_(kernel), n_dim_(n_dim), n_digits_(n_digits), fparam_(fparam), box_sizes_(boxsize_),
      n_levels_(boxsize_.Dim()) {

    beta_ = procl180_rescale(eps);
    prolate0_fun = Prolate0Fun(beta_, 10000);
    difference_kernels_.ReInit(n_levels_);

    auto n_pw_windowed = 0, n_pw_difference = 0;
    std::tie(n_pw_windowed, windowed_kernel_.hpw, windowed_kernel_.ws, windowed_kernel_.rl) =
        get_PSWF_windowed_kernel_pwterms(n_dim_, n_digits_, box_sizes_[0]);

    difference_kernels_[0].rl = windowed_kernel_.rl;
    for (int i = 1; i < n_levels_; ++i)
        difference_kernels_[i].rl = 0.5 * difference_kernels_[i - 1].rl;

    for (int i_level = 0; i_level < n_levels_; ++i_level)
        std::tie(n_pw_difference, difference_kernels_[i_level].hpw, difference_kernels_[i_level].ws) =
            get_PSWF_difference_kernel_pwterms(kernel_, n_dim, n_digits_, box_sizes_[i_level]);

    n_pw_ = std::max(n_pw_windowed, n_pw_difference);

    assert(n_pw_windowed);
    assert(n_pw_difference);
    assert(n_pw_);
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
            dk0 = std::cyl_bessel_k(0, rl * rlambda);
            dk1 = std::cyl_bessel_k(1, rl * rlambda);
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
                fhat *= -rl * rlambda * std::cyl_bessel_j(0, xsc) * dk1 + 1 + xsc * std::cyl_bessel_j(1, xsc) * dk0;
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

template <typename T>
void FourierData<T>::update_local_coeffs_laplace(T eps) {
    // FIXME: Should do something _only_ if doing dipoles. Implement when
    // we add dipoles :)
}

template <typename Real>
void FourierData<Real>::update_local_coeffs_yukawa(Real eps) {
    // FIXME: This whole routine is a mess and only works with doubles anyway
    const int nr1 = n_coeffs_max, nr2 = n_coeffs_max;

    coeffs1_.ReInit(nr1 * n_levels_);
    coeffs2_.ReInit(nr2 * n_levels_);
    ncoeffs1_.ReInit(n_levels_);
    ncoeffs2_.ReInit(n_levels_);

    constexpr Real two_over_pi = 2.0 / M_PI;
    const Real &rlambda = fparam_;
    const Real rlambda2 = rlambda * rlambda;
    const Real psi0 = prolate0_fun.eval_val(0.0);

    constexpr int n_quad = 100;
    // FIXME: Ideally these would use the Real type, but we get slightly lower errors downstream with double
    std::vector<double> xs(n_quad), whts(n_quad), xs_base(n_quad), whts_base(n_quad), r1(n_quad), r2(n_quad),
        w1(n_quad), w2(n_quad), fhat(n_quad);
    legerts(1, n_quad, xs_base.data(), whts_base.data());
    auto [v1, vlu1] = dmk::chebyshev::get_vandermonde_and_LU<Real>(nr1);
    auto [v2, vlu2] = dmk::chebyshev::get_vandermonde_and_LU<Real>(nr2);
    nda::vector<Real> fvals(nr1);

    for (int i_level = 0; i_level < n_levels_; ++i_level) {
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
                dk0 = std::cyl_bessel_k(0, arg);
                dk1 = std::cyl_bessel_k(1, arg);
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
                        -rl * rlambda * std::cyl_bessel_j(0, xsc) * dk1 + 1 + xsc * std::cyl_bessel_j(1, xsc) * dk0;
                else
                    fhat[i] *= 1 - delam * (std::cos(xsc) + rlambda / xs[i] * std::sin(xsc));
            }
        }

        auto r1 = dmk::chebyshev::get_cheb_nodes(nr1, Real{0.}, bsize);
        if (n_dim_ == 2) {
            for (int i = 0; i < nr1; ++i) {
                fvals(i) = 0.0;
                for (int j = 0; j < n_quad; ++j)
                    fvals(i) -= std::cyl_bessel_j(0, r1[i] * xs[j]) * fhat[j];
            }
        } else if (n_dim_ == 3) {
            for (int i = 0; i < nr1; ++i) {
                fvals(i) = 0.0;
                for (int j = 0; j < n_quad; ++j) {
                    Real dd = r1[i] * xs[j];
                    fvals(i) -= sin(dd) / dd * fhat[j];
                }
            }
        }

        nda::vector_view<Real> coeffs1_lvl({nr1}, &coeffs1_[0] + nr1 * i_level);
        coeffs1_lvl = vlu1.solve(fvals);
        Real coefsmax = nda::max_element(nda::abs(coeffs1_lvl));
        Real releps = eps * coefsmax;

        ncoeffs1_[i_level] = 1;
        for (int i = 0; i < nr1 - 2; ++i) {
            if (std::fabs(coeffs1_lvl(i)) < releps && std::fabs(coeffs1_lvl(i + 1)) < releps &&
                std::fabs(coeffs1_lvl(i + 2)) < releps) {
                ncoeffs1_[i_level] = i + 1;
                break;
            }
        }

        // coeffs2
        nda::vector<Real> r2 = dmk::chebyshev::get_cheb_nodes(nr2, Real{0.25} * bsize * bsize, bsize * bsize);
        if (n_dim_ == 2) {
            for (int i = 0; i < nr2; ++i) {
                fvals(i) = 0.0;
                const Real r = sqrt(r2(i));
                for (int j = 0; j < n_quad; ++j)
                    fvals(i) -= std::cyl_bessel_j(0, r * xs[j]) * fhat[j];

                fvals(i) += std::cyl_bessel_j(0, rlambda * r);
            }
        } else if (n_dim_ == 3) {
            for (int i = 0; i < nr2; ++i) {
                fvals(i) = 0.0;
                const Real r = sqrt(r2(i));
                for (int j = 0; j < n_quad; ++j) {
                    Real dd = r * xs[j];
                    fvals(i) -= std::sin(dd) / dd * fhat[j];
                }
                fvals(i) += std::exp(-rlambda * r) / r;
            }
        }

        nda::vector_view<Real> coeffs2_lvl({nr2}, &coeffs2_[0] + nr2 * i_level);
        coeffs2_lvl = vlu2.solve(fvals);

        coefsmax = nda::max_element(nda::abs(coeffs2_lvl));
        releps = eps * coefsmax;
        ncoeffs2_[i_level] = 1;
        for (int i = 0; i < nr2 - 2; ++i) {
            if (std::fabs(coeffs2_lvl(i)) < releps && std::fabs(coeffs2_lvl(i + 1)) < releps &&
                std::fabs(coeffs2_lvl(i + 2)) < releps) {
                ncoeffs2_[i_level] = i + 1;
                break;
            }
        }
    }
}

template <typename T>
void FourierData<T>::update_local_coeffs(T eps) {
    switch (kernel_) {
    case dmk_ikernel::DMK_YUKAWA:
        return update_local_coeffs_yukawa(eps);
    case dmk_ikernel::DMK_LAPLACE:
        return update_local_coeffs_laplace(eps);
    case dmk_ikernel::DMK_SQRT_LAPLACE:
        return; // No coefficients for sqrt Laplace kernel
    default:
        throw std::runtime_error("Kernel not supported yet: " + std::to_string(kernel_));
    }
}

template <typename T>
void FourierData<T>::calc_planewave_coeff_matrices(int i_level, int n_order, sctl::Vector<std::complex<T>> &prox2pw_vec,
                                                   sctl::Vector<std::complex<T>> &pw2poly_vec) const {
    auto hpw = (i_level + 1) ? difference_kernels_[i_level].hpw : windowed_kernel_.hpw;
    auto bsize = box_sizes_[std::max(i_level, 0)];
    dmk::calc_planewave_coeff_matrices(bsize, hpw, n_pw_, n_order, prox2pw_vec, pw2poly_vec);
}

template struct FourierData<float>;
template struct FourierData<double>;

template void get_windowed_kernel_ft<float, 2>(dmk_ikernel kernel, const double *rpars, float beta, int ndigits,
                                               float boxsize, Prolate0Fun &pf, sctl::Vector<float> &radialft);
template void get_windowed_kernel_ft<float, 3>(dmk_ikernel kernel, const double *rpars, float beta, int ndigits,
                                               float boxsize, Prolate0Fun &pf, sctl::Vector<float> &radialft);
template void get_windowed_kernel_ft<double, 2>(dmk_ikernel kernel, const double *rpars, double beta, int ndigits,
                                                double boxsize, Prolate0Fun &pf, sctl::Vector<double> &radialft);
template void get_windowed_kernel_ft<double, 3>(dmk_ikernel kernel, const double *rpars, double beta, int ndigits,
                                                double boxsize, Prolate0Fun &pf, sctl::Vector<double> &radialft);
template void get_difference_kernel_ft<float, 2>(bool init, dmk_ikernel kernel, const double *rpars, float beta,
                                                 int ndigits, float boxsize, Prolate0Fun &pf,
                                                 sctl::Vector<float> &diff_kernel_ft);
template void get_difference_kernel_ft<float, 3>(bool init, dmk_ikernel kernel, const double *rpars, float beta,
                                                 int ndigits, float boxsize, Prolate0Fun &pf,
                                                 sctl::Vector<float> &diff_kernel_ft);
template void get_difference_kernel_ft<double, 2>(bool init, dmk_ikernel kernel, const double *rpars, double beta,
                                                  int ndigits, double boxsize, Prolate0Fun &pf,
                                                  sctl::Vector<double> &diff_kernel_ft);
template void get_difference_kernel_ft<double, 3>(bool init, dmk_ikernel kernel, const double *rpars, double beta,
                                                  int ndigits, double boxsize, Prolate0Fun &pf,
                                                  sctl::Vector<double> &diff_kernel_ft);

} // namespace dmk
