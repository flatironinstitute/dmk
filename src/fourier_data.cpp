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
template <typename T>
T procl180_rescale(T eps) {
    constexpr float cs[] = {
        .43368E-16, .10048E+01, .17298E+01, .22271E+01, .26382E+01, .30035E+01, .33409E+01, .36598E+01, .39658E+01,
        .42621E+01, .45513E+01, .48347E+01, .51136E+01, .53887E+01, .56606E+01, .59299E+01, .61968E+01, .64616E+01,
        .67247E+01, .69862E+01, .72462E+01, .75049E+01, .77625E+01, .80189E+01, .82744E+01, .85289E+01, .87826E+01,
        .90355E+01, .92877E+01, .95392E+01, .97900E+01, .10040E+02, .10290E+02, .10539E+02, .10788E+02, .11036E+02,
        .11284E+02, .11531E+02, .11778E+02, .12024E+02, .12270E+02, .12516E+02, .12762E+02, .13007E+02, .13251E+02,
        .13496E+02, .13740E+02, .13984E+02, .14228E+02, .14471E+02, .14714E+02, .14957E+02, .15200E+02, .15443E+02,
        .15685E+02, .15927E+02, .16169E+02, .16411E+02, .16652E+02, .16894E+02, .17135E+02, .17376E+02, .17617E+02,
        .17858E+02, .18098E+02, .18339E+02, .18579E+02, .18819E+02, .19059E+02, .19299E+02, .19539E+02, .19778E+02,
        .20018E+02, .20257E+02, .20496E+02, .20736E+02, .20975E+02, .21214E+02, .21452E+02, .21691E+02, .21930E+02,
        .22168E+02, .22407E+02, .22645E+02, .22884E+02, .23122E+02, .23360E+02, .23598E+02, .23836E+02, .24074E+02,
        .24311E+02, .24549E+02, .24787E+02, .25024E+02, .25262E+02, .25499E+02, .25737E+02, .25974E+02, .26211E+02,
        .26448E+02, .26685E+02, .26922E+02, .27159E+02, .27396E+02, .27633E+02, .27870E+02, .28106E+02, .28343E+02,
        .28580E+02, .28816E+02, .29053E+02, .29289E+02, .29526E+02, .29762E+02, .29998E+02, .30234E+02, .30471E+02,
        .30707E+02, .30943E+02, .31179E+02, .31415E+02, .31651E+02, .31887E+02, .32123E+02, .32358E+02, .32594E+02,
        .32830E+02, .33066E+02, .33301E+02, .33537E+02, .33773E+02, .34008E+02, .34244E+02, .34479E+02, .34714E+02,
        .34950E+02, .35185E+02, .35421E+02, .35656E+02, .35891E+02, .36126E+02, .36362E+02, .36597E+02, .36832E+02,
        .37067E+02, .37302E+02, .37537E+02, .37772E+02, .38007E+02, .38242E+02, .38477E+02, .38712E+02, .38947E+02,
        .39181E+02, .39416E+02, .39651E+02, .39886E+02, .40120E+02, .40355E+02, .40590E+02, .40824E+02, .41059E+02,
        .41294E+02, .41528E+02, .41763E+02, .41997E+02, .42232E+02, .42466E+02, .42700E+02, .42935E+02, .43169E+02,
        .43404E+02, .43638E+02, .43872E+02, .44107E+02, .44341E+02, .44575E+02, .44809E+02, .45044E+02, .45278E+02};

    int scale;
    if (eps >= 1.0E-3)
        scale = 8;
    else if (eps >= 1E-6)
        scale = 20;
    else if (eps >= 1E-9)
        scale = 25;
    else if (eps >= 1E-12)
        scale = 25;

    double d = -std::log10(scale * eps);
    int i = d * 10 + 0.1 - 1;
    assert(i >= 0);
    assert(i < sizeof(cs) / sizeof(float));
    return cs[i];
}

template <int DIM>
std::tuple<int, double, double> get_PSWF_difference_kernel_pwterms(dmk_ikernel kernel, int ndigits, double boxsize) {
    int npw;
    double hpw, ws;

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
            npw = 25;
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
    ws = sctl::pow<DIM>(hpw) * factor;

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
        npw = 25;
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

    throw std::runtime_error("2D Laplace kernel not supported yet.");
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
        auto [val, dval] = pf.eval_val_derivative(xs[i] / boxsize);
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

template <typename Real, int DIM>
void get_windowed_kernel_ft(dmk_ikernel kernel, const double *rpars, Real beta, int ndigits, Real boxsize,
                            Prolate0Fun &pf, sctl::Vector<Real> &windowed_ft) {
    switch (kernel) {
    case dmk_ikernel::DMK_YUKAWA:
        return yukawa_windowed_kernel_ft<Real, DIM>(rpars, beta, ndigits, boxsize, pf, windowed_ft);
    case dmk_ikernel::DMK_LAPLACE:
        return laplace_windowed_kernel_ft<Real, DIM>(rpars, beta, ndigits, boxsize, pf, windowed_ft);
    case dmk_ikernel::DMK_SQRT_LAPLACE:
        throw std::runtime_error("SQRT Laplace kernel not supported yet.");
        break;
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

    for (int i = 0; i < n_fourier; ++i) {
        Real rk = sqrt((Real)i) * hpw;
        Real xi2 = rk * rk + rlambda2;
        Real xi = sqrt(xi2);
        Real xval = xi * bsizesmall / beta;
        Real fval1 = (xval <= 1.0) ? pf.eval_val(xval) : 0.0;

        xval = xi * bsizebig / beta;
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
}

template <typename Real, int DIM>
void laplace_difference_kernel_ft(const double *rpars, Real beta, int ndigits, Real boxsize, Prolate0Fun &pf,
                                  sctl::Vector<Real> &diff_kernel_ft) {
    if constexpr (DIM == 2)
        throw std::runtime_error("2D Laplace kernel not supported yet.");

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
}

template <typename Real, int DIM>
void get_difference_kernel_ft(dmk_ikernel kernel, const double *rpars, Real beta, int ndigits, Real boxsize,
                              Prolate0Fun &pf, sctl::Vector<Real> &diff_kernel_ft) {
    switch (kernel) {
    case dmk_ikernel::DMK_YUKAWA:
        return yukawa_difference_kernel_ft<Real, DIM>(rpars, beta, ndigits, boxsize, pf, diff_kernel_ft);
    case dmk_ikernel::DMK_LAPLACE:
        return laplace_difference_kernel_ft<Real, DIM>(rpars, beta, ndigits, boxsize, pf, diff_kernel_ft);
    default:
        throw std::runtime_error("Unsupported kernel" + std::to_string(kernel));
    }
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
            dk0 = std::cyl_bessel_j(0, rl * rlambda);
            dk1 = std::cyl_bessel_j(1, rl * rlambda);
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
    const T rl = difference_kernels_[std::max(i_level - 1, 0)].rl;

    return dmk::yukawa_windowed_kernel_value_at_zero(n_dim_, fparam_, beta_, bsize, rl, prolate0_fun);
}

template <typename T>
void FourierData<T>::update_local_coeffs_laplace(T eps) {
    if (n_dim_ != 2)
        return; // Only coeffs in 2d

    // FIXME: Laplace2D only should update if calculating dipoles
    throw std::runtime_error("2D Laplace kernel not supported yet.");
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
    Eigen::VectorX<Real> fvals(nr1);

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
        const Real rl = difference_kernels_[std::max(i_level - 1, 0)].rl;
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
                        -rl + rlambda + std::cyl_bessel_j(0, xsc) * dk1 + 1 + xsc * std::cyl_bessel_j(1, xsc) * dk0;
                else
                    fhat[i] *= 1 - delam * (std::cos(xsc) + rlambda / xs[i] * std::sin(xsc));
            }
        }

        Eigen::VectorX<Real> r1 = dmk::chebyshev::get_cheb_nodes(nr1, Real{0.}, bsize);
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

        Eigen::Map<Eigen::VectorX<Real>> coeffs1_lvl(&coeffs1_[0] + nr1 * i_level, nr1);
        coeffs1_lvl = vlu1.solve(fvals);
        Real coefsmax = coeffs1_lvl.array().abs().maxCoeff();
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
        Eigen::VectorX<Real> r2 = dmk::chebyshev::get_cheb_nodes(nr2, Real{0.25} * bsize * bsize, bsize * bsize);
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

        Eigen::Map<Eigen::VectorX<Real>> coeffs2_lvl(&coeffs2_[0] + nr2 * i_level, nr2);
        coeffs2_lvl = vlu2.solve(fvals);

        coefsmax = coeffs2_lvl.array().abs().maxCoeff();
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
void FourierData<T>::calc_planewave_translation_matrix(int dim, int i_level, T xmin,
                                                       sctl::Vector<std::complex<T>> &shift_vec) const {
    constexpr int nmax = 1;
    sctl::Vector<T> ts(n_pw_);
    int ts_shift = n_pw_ / 2;
    const T hpw = i_level ? difference_kernels_[i_level - 1].hpw : windowed_kernel_.hpw;
    for (int i = 0; i < n_pw_; ++i)
        ts[i] = (i - ts_shift) * hpw;

    if (dim == 2)
        dmk::calc_planewave_translation_matrix<2>(nmax, xmin, n_pw_, ts, shift_vec);
    else if (dim == 3)
        dmk::calc_planewave_translation_matrix<3>(nmax, xmin, n_pw_, ts, shift_vec);
    else
        throw std::runtime_error("Dimension " + std::to_string(dim) + "not supported");
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
template void get_difference_kernel_ft<float, 2>(dmk_ikernel kernel, const double *rpars, float beta, int ndigits,
                                                 float boxsize, Prolate0Fun &pf, sctl::Vector<float> &diff_kernel_ft);
template void get_difference_kernel_ft<float, 3>(dmk_ikernel kernel, const double *rpars, float beta, int ndigits,
                                                 float boxsize, Prolate0Fun &pf, sctl::Vector<float> &diff_kernel_ft);
template void get_difference_kernel_ft<double, 2>(dmk_ikernel kernel, const double *rpars, double beta, int ndigits,
                                                  double boxsize, Prolate0Fun &pf,
                                                  sctl::Vector<double> &diff_kernel_ft);
template void get_difference_kernel_ft<double, 3>(dmk_ikernel kernel, const double *rpars, double beta, int ndigits,
                                                  double boxsize, Prolate0Fun &pf,
                                                  sctl::Vector<double> &diff_kernel_ft);

} // namespace dmk
