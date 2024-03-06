#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/prolate_funcs.hpp>

#include <sctl.hpp>

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
FourierData<T>::FourierData(dmk_ikernel kernel_, int n_dim_, int n_digits_, int n_pw_max, T fparam_, T beta_,
                            const std::vector<double> &boxsize_)
    : kernel(kernel_), n_dim(n_dim_), n_digits(n_digits_), fparam(fparam_), beta(beta_), boxsize(boxsize_),
      n_levels(boxsize_.size()), n_fourier_max(n_dim_ * sctl::pow(n_pw_max / 2, 2)) {

    npw.resize(n_levels + 1);
    nfourier.resize(n_levels + 1);
    hpw.resize(n_levels + 1);
    ws.resize(n_levels + 1);
    rl.resize(n_levels + 1);

    if (n_dim == 2)
        std::tie(npw[0], hpw[0], ws[0], rl[0]) = get_PSWF_truncated_kernel_pwterms<2>(n_digits, boxsize[0]);
    else if (n_dim == 3)
        std::tie(npw[0], hpw[0], ws[0], rl[0]) = get_PSWF_truncated_kernel_pwterms<3>(n_digits, boxsize[0]);

    dkernelft.resize(n_fourier_max * (n_levels + 1));

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

    nfourier[0] = n_dim * sctl::pow(npw[0] / 2, 2);
    for (int i = 0; i < nfourier[0]; ++i) {
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
    T *fhat = &dkernelft[(i_level + 1) * n_fourier_max];

    for (int i = 0; i < nfourier[i_level + 1]; ++i) {
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
        auto &npw_i = npw[i_level + 1];
        auto &hpw_i = hpw[i_level + 1];
        auto &ws_i = ws[i_level + 1];
        auto &rl_i = rl[i_level + 1];
        auto dkernelft_i = &dkernelft[(i_level + 1) * n_fourier_max];

        // FIXME: GIVES DIFFERENT NPW THAN OTHER CODE
        std::tie(npw_i, hpw_i, ws_i) = PSWF_difference_kernel_pwterms(kernel, n_digits, bsize);

        nfourier[i_level + 1] = n_dim * sctl::pow(npw[i_level + 1] / 2, 2);

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

        const T *dkernelft_im1 = &dkernelft[i_level * n_fourier_max];
        for (int i = 0; i < nfourier[1]; ++i)
            dkernelft_i[i] = scale_factor * dkernelft_im1[i];
    }
}

template struct FourierData<float>;
template struct FourierData<double>;

} // namespace dmk
