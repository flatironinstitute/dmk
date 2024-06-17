#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/planewave.hpp>
#include <dmk/prolate_funcs.hpp>

#include <complex.h>
#include <sctl.hpp>
#include <stdexcept>
#include <string>

namespace dmk {

template <int DIM>
std::tuple<int, double, double, double> get_PSWF_truncated_kernel_pwterms(int ndigits, double boxsize) {
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

template <typename T>
FourierData<T>::FourierData(dmk_ikernel kernel_, int n_dim_, int n_digits_, int n_pw_max_, T fparam_, T beta_,
                            const std::vector<double> &boxsize_)
    : kernel(kernel_), n_dim(n_dim_), n_digits(n_digits_), fparam(fparam_), beta(beta_), boxsize(boxsize_),
      n_levels(boxsize_.size()), n_fourier(n_dim_ * sctl::pow(n_pw_max_ / 2, 2)) {

    hpw.resize(n_levels + 1);
    ws.resize(n_levels + 1);
    rl.resize(n_levels + 1);

    if (n_dim == 2)
        std::tie(n_pw, hpw[0], ws[0], rl[0]) = get_PSWF_truncated_kernel_pwterms<2>(n_digits, boxsize[0]);
    else if (n_dim == 3)
        std::tie(n_pw, hpw[0], ws[0], rl[0]) = get_PSWF_truncated_kernel_pwterms<3>(n_digits, boxsize[0]);

    dkernelft.resize(n_fourier * (n_levels + 1));

    rl[1] = rl[0];
    for (int i = 2; i < rl.size(); ++i)
        rl[i] = 0.5 * rl[i - 1];
}

template <typename T>
void FourierData<T>::yukawa_windowed_kernel_Fourier_transform(ProlateFuncs &prolate_funcs) {
    // compute the Fourier transform of the truncated kernel
    // of the Yukawa kernel in two and three dimensions
    const T &rlambda = fparam;
    const T rlambda2 = rlambda * rlambda;
    auto fhat = &dkernelft[0];

    // determine whether one needs to smooth out the 1/(k^2+lambda^2) factor at the origin.
    // needed in the calculation of kernel-smoothing when there is low-frequency breakdown
    const bool near_correction = (rlambda * boxsize[0] / beta < 1E-2);
    T dk0, dk1, delam;
    if (near_correction) {
        double arg = rl[0] * rlambda;
        if (n_dim == 2) {
            dk0 = besk0_(&arg);
            dk1 = besk1_(&arg);
        } else if (n_dim == 3)
            delam = std::exp(-arg);
    }

    double psi0 = prolate_funcs.eval_val(0);

    for (int i = 0; i < n_fourier; ++i) {
        const double rk = sqrt((double)i) * hpw[0];
        const double xi2 = rk * rk + rlambda2;
        const double xi = sqrt(xi2);
        const double xval = xi * boxsize[0] / beta;
        const double fval = (xval <= 1.0) ? prolate_funcs.eval_val(xval) : 0.0;

        fhat[i] = ws[0] * fval / (psi0 * xi2);

        if (near_correction) {
            double sker;
            if (n_dim == 2) {
                double xsc = rl[0] * rk;
                sker = -rl[0] * fparam * besj0_(&xsc) * dk1 + 1.0 + xsc * besj1_(&xsc) * dk0;
            } else if (n_dim == 3) {
                double xsc = rl[0] * rk;
                sker = 1 - delam * (cos(xsc) + rlambda / rk * sin(xsc));
            }
            fhat[i] *= sker;
        }
    }
}

template <typename T>
void FourierData<T>::update_windowed_kernel_fourier_transform(ProlateFuncs &pf) {
    switch (kernel) {
    case dmk_ikernel::DMK_YUKAWA:
        return yukawa_windowed_kernel_Fourier_transform(pf);
    case dmk_ikernel::DMK_LAPLACE:
        throw std::runtime_error("Laplace kernel not supported yet.");
    case dmk_ikernel::DMK_SQRT_LAPLACE:
        throw std::runtime_error("SQRT Laplace kernel not supported yet.");
    }
}
template <typename T>
void FourierData<T>::yukawa_difference_kernel_fourier_transform(int i_level, ProlateFuncs &pf) {
    const T bsizesmall = boxsize[i_level] * 0.5;
    const T bsizebig = boxsize[i_level];
    const double &rlambda = fparam;
    const double rlambda2 = rlambda * rlambda;
    const double psi0 = pf.eval_val(0.0);
    T *fhat = &dkernelft[(i_level + 1) * n_fourier];

    for (int i = 0; i < n_fourier; ++i) {
        T rk = sqrt((T)i) * hpw[i_level + 1];
        T xi2 = rk * rk + rlambda2;
        T xi = sqrt(xi2);
        T xval = xi * bsizesmall / beta;
        T fval1 = (xval <= 1.0) ? pf.eval_val(xval) : 0.0;

        xval = xi * bsizebig / beta;
        T fval2 = (xval <= 1.0) ? pf.eval_val(xval) : 0.0;
        fhat[i] = ws[i_level + 1] * (fval1 - fval2) / (psi0 * xi2);
    }

    // re-compute fhat[0] accurately when there is a low-frequency breakdown
    if (rlambda * bsizebig / beta < 1E-4) {
        const std::array<double, 4> c = pf.intvals(beta);
        const double bsizesmall2 = bsizesmall * bsizesmall;
        const double bsizebig2 = bsizebig * bsizebig;

        fhat[0] = ws[i_level + 1] * c[2] * (bsizebig2 - bsizesmall2) / 2 +
                  ws[i_level + 1] * (bsizesmall2 * bsizesmall2 - bsizebig2 * bsizebig2) * rlambda2 * c[3] / (c[0] * 24);
    }
}

template <typename T>
void FourierData<T>::update_difference_kernel(int i_level, ProlateFuncs &pf) {
    switch (kernel) {
    case dmk_ikernel::DMK_YUKAWA:
        return yukawa_difference_kernel_fourier_transform(i_level, pf);
    case dmk_ikernel::DMK_LAPLACE:
        throw std::runtime_error("Laplace kernel not supported yet.");
    case dmk_ikernel::DMK_SQRT_LAPLACE:
        throw std::runtime_error("SQRT Laplace kernel not supported yet.");
    }
}

template <typename T>
void FourierData<T>::update_difference_kernels(ProlateFuncs &pf) {
    auto PSWF_difference_kernel_pwterms =
        (n_dim == 2) ? get_PSWF_difference_kernel_pwterms<2> : get_PSWF_difference_kernel_pwterms<3>;

    for (int i_level = 0; i_level < n_levels; ++i_level) {
        auto &bsize = boxsize[i_level];
        auto bsize_small = bsize * 0.5;
        auto &hpw_i = hpw[i_level + 1];
        auto &ws_i = ws[i_level + 1];
        auto &rl_i = rl[i_level + 1];
        auto dkernelft_i = &dkernelft[(i_level + 1) * n_fourier];

        // FIXME: GIVES DIFFERENT NPW THAN OTHER CODE
        // FIXME: overwrites n_pw
        std::tie(n_pw, hpw_i, ws_i) = PSWF_difference_kernel_pwterms(kernel, n_digits, bsize);

        if (i_level == 0 || kernel == dmk_ikernel::DMK_YUKAWA) {
            update_difference_kernel(i_level, pf);
            continue;
        }

        double scale_factor;
        if (kernel == dmk_ikernel::DMK_SQRT_LAPLACE && n_dim == 3)
            scale_factor = 4.0;
        else if (kernel == dmk_ikernel::DMK_LAPLACE && n_dim == 2)
            scale_factor = 1.0;
        else
            scale_factor = 2.0;

        const T *dkernelft_im1 = &dkernelft[i_level * n_fourier];
        for (int i = 0; i < n_fourier; ++i)
            dkernelft_i[i] = scale_factor * dkernelft_im1[i];
    }
}

template <typename T>
void FourierData<T>::update_local_coeffs_yukawa(T eps, ProlateFuncs &pf) {
    constexpr T two_over_pi = 2.0 / M_PI;
    const T &rlambda = fparam;
    const T rlambda2 = rlambda * rlambda;
    double psi0 = pf.eval_val(0.0);

    constexpr int i_type = 1;
    constexpr int n_quad = 100;
    std::vector<double> xs(n_quad), xs_base(n_quad), whts(n_quad), whts_base(n_quad), r1(n_quad), r2(n_quad),
        w1(n_quad), w2(n_quad), fhat(n_quad);
    double u, v; // dummy vars
    legeexps_(&i_type, &n_quad, xs_base.data(), &u, &v, whts_base.data());
    const int nr1 = 100, nr2 = 100;
    auto [v1, vlu1] = dmk::chebyshev::get_vandermonde_and_LU<double>(nr1);
    auto [v2, vlu2] = dmk::chebyshev::get_vandermonde_and_LU<double>(nr2);
    Eigen::VectorXd fvals(nr1);

    coeffs1.resize(nr1 * (n_levels + 1));
    coeffs2.resize(nr2 * (n_levels + 1));
    ncoeffs1.resize(n_levels + 1);
    ncoeffs2.resize(n_levels + 1);

    for (int i_level = 0; i_level < n_levels; ++i_level) {
        auto &bsize = boxsize[i_level];

        double scale_factor = beta / (2.0 * bsize);
        for (int i = 0; i < n_quad; ++i) {
            xs[i] = scale_factor * (xs_base[i] + 1.0);
            whts[i] = scale_factor * whts_base[i];
        }

        const bool near_correction = rlambda * bsize / beta < 1E-2;
        double dk0, dk1, delam;
        if (near_correction) {
            double arg = rl[i_level + 1] * rlambda;
            if (n_dim == 2) {
                dk0 = besk0_(&arg);
                dk1 = besk1_(&arg);
            } else
                delam = std::exp(-arg);
        }

        for (int i = 0; i < n_quad; ++i) {
            const double xi2 = xs[i] * xs[i] + rlambda2;
            const double xval = sqrt(xi2) * bsize / beta;
            if (xval <= 1.0) {
                fhat[i] = pf.eval_val(xval) / (psi0 * xi2);
            } else {
                fhat[i] = 0.0;
                continue;
            }
            fhat[i] *= (n_dim == 2) ? whts[i] * xs[i] : whts[i] * xs[i] * xs[i] * two_over_pi;

            if (near_correction) {
                double xsc = rl[i_level + 1] * xs[i];
                if (n_dim == 2)
                    fhat[i] *= -rl[i_level + 1] + rlambda + besj0_(&xsc) * dk1 + 1 + xsc * besj1_(&xsc) * dk0;
                else
                    fhat[i] *= 1 - delam * (std::cos(xsc) + rlambda / xs[i] * std::sin(xsc));
            }
        }

        Eigen::VectorXd r1 = dmk::chebyshev::get_cheb_nodes(nr1, 0., bsize);
        if (n_dim == 2) {
            const int actual_hankel = 1;
            for (int i = 0; i < nr1; ++i) {
                fvals(i) = 0.0;
                for (int j = 0; j < n_quad; ++j) {
                    std::complex<double> z = r1[i] * xs[j];
                    std::complex<double> h0, h1;
                    hank103_(reinterpret_cast<_Complex double *>(&z), reinterpret_cast<_Complex double *>(&h0),
                             reinterpret_cast<_Complex double *>(&h1), &actual_hankel);
                    fvals(i) -= h0.real() * fhat[j];
                }
            }
        } else if (n_dim == 3) {
            for (int i = 0; i < nr1; ++i) {
                fvals(i) = 0.0;
                for (int j = 0; j < n_quad; ++j) {
                    double dd = r1[i] * xs[j];
                    fvals(i) -= sin(dd) / dd * fhat[j];
                }
            }
        }

        Eigen::Map<Eigen::VectorXd> coeffs1_lvl(coeffs1.data() + nr1 * (i_level + 1), nr1);
        coeffs1_lvl = vlu1.solve(fvals);
        double coefsmax = coeffs1_lvl.array().abs().maxCoeff();
        double releps = eps * coefsmax;

        ncoeffs1[i_level + 1] = 1;
        for (int i = 0; i < nr1 - 2; ++i) {
            if (std::fabs(coeffs1_lvl(i)) < releps && std::fabs(coeffs1_lvl(i + 1)) < releps &&
                std::fabs(coeffs1_lvl(i + 2)) < releps) {
                ncoeffs1[i_level + 1] = i + 1;
                break;
            }
        }

        // coeffs2
        Eigen::VectorXd r2 = dmk::chebyshev::get_cheb_nodes(nr2, 0.25 * bsize * bsize, bsize * bsize);
        if (n_dim == 2) {
            const int actual_hankel = 1;
            for (int i = 0; i < nr2; ++i) {
                fvals(i) = 0.0;
                const double r = sqrt(r2(i));
                for (int j = 0; j < n_quad; ++j) {
                    std::complex<double> z = r * xs[j];
                    std::complex<double> h0, h1;
                    hank103_(reinterpret_cast<_Complex double *>(&z), reinterpret_cast<_Complex double *>(&h0),
                             reinterpret_cast<_Complex double *>(&h1), &actual_hankel);
                    fvals(i) -= h0.real() * fhat[j];
                }
                double dd = rlambda * r;
                fvals(i) += besk0_(&dd);
            }
        } else if (n_dim == 3) {
            for (int i = 0; i < nr2; ++i) {
                fvals(i) = 0.0;
                const double r = sqrt(r2(i));
                for (int j = 0; j < n_quad; ++j) {
                    double dd = r * xs[j];
                    fvals(i) -= std::sin(dd) / dd * fhat[j];
                }
                fvals(i) += std::exp(-rlambda * r) / r;
            }
        }

        Eigen::Map<Eigen::VectorXd> coeffs2_lvl(coeffs2.data() + nr2 * (i_level + 1), nr2);
        coeffs2_lvl = vlu2.solve(fvals);

        coefsmax = coeffs2_lvl.array().abs().maxCoeff();
        releps = eps * coefsmax;
        ncoeffs2[i_level + 1] = 1;
        for (int i = 0; i < nr2 - 2; ++i) {
            if (std::fabs(coeffs2_lvl(i)) < releps && std::fabs(coeffs2_lvl(i + 1)) < releps &&
                std::fabs(coeffs2_lvl(i + 2)) < releps) {
                ncoeffs2[i_level + 1] = i + 1;
                break;
            }
        }
    }
}

template <typename T>
void FourierData<T>::update_local_coeffs(T eps, ProlateFuncs &pf) {
    switch (kernel) {
    case dmk_ikernel::DMK_YUKAWA:
        return update_local_coeffs_yukawa(eps, pf);
    default:
        return;
    }
}

template <typename T>
void FourierData<T>::calc_planewave_translation_matrix(int dim, int i_level, T xmin,
                                                       sctl::Vector<std::complex<T>> &shift_vec) const {
    constexpr int nmax = 1;
    sctl::Vector<T> ts(n_pw);
    int ts_shift = n_pw / 2;
    for (int i = 0; i < n_pw; ++i)
        ts[i] = (i - ts_shift) * hpw[i_level];

    if (dim == 2)
        dmk::calc_planewave_translation_matrix<2>(nmax, xmin, n_pw, ts, shift_vec);
    else if (dim == 3)
        dmk::calc_planewave_translation_matrix<3>(nmax, xmin, n_pw, ts, shift_vec);
    else
        throw std::runtime_error("Dimension " + std::to_string(dim) + "not supported");
}

template <typename T>
void FourierData<T>::calc_planewave_coeff_matrices(int i_level, int n_order, sctl::Vector<std::complex<T>> &prox2pw_vec,
                                                   sctl::Vector<std::complex<T>> &pw2poly_vec) const {
    dmk::calc_planewave_coeff_matrices(boxsize[i_level], hpw[i_level], n_pw, n_order, prox2pw_vec, pw2poly_vec);
}

// template struct FourierData<float>;
template struct FourierData<double>;

} // namespace dmk
