#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/types.hpp>
#include <dmk/util.hpp>

namespace dmk::util {
using dmk::ndview;

double calc_bandlimiting(const pdmk_params &p) {
    if (p.debug_flags & DMK_DEBUG_OVERRIDE_BETA)
        return p.debug_params[DMK_DEBUG_BETA_SLOT];
    double d = -std::log10(p.eps);
    if (p.kernel == DMK_STOKESLET || p.kernel == DMK_STRESSLET) {
        return M_PI / 3.0 * std::ceil(3.0 / M_PI * (0.69 + d) / 0.39);
    }

    // Each spatial derivative taken on the kernel makes solution
    // converge ~1 digit slower
    const int n_derivs_src = (p.kernel == DMK_LAPLACE_DIPOLE) ? 1 : 0;
    const int n_derivs_trg = (p.eval_src >= DMK_POTENTIAL_GRAD || p.eval_trg >= DMK_POTENTIAL_GRAD) ? 1 : 0;
    constexpr double digits_per_deriv = 1.0;
    const double d_eff = d + digits_per_deriv * (n_derivs_src + n_derivs_trg);
    return 2.664 * d_eff + 0.306;
}

template <typename Real>
void mesh_2d(const ndview<const Real, 1> &x, const ndview<const Real, 1> &y, const ndview<Real, 2> &xy) {
    int nx = x.extent(0);
    int ny = y.extent(0);

    int ind = 0;
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            xy(0, ind) = x(ix);
            xy(1, ind) = y(iy);
            ind++;
        }
    }
}

template <typename Real>
void mesh_3d(const ndview<const Real, 1> &x, const ndview<const Real, 1> &y, const ndview<const Real, 1> &z,
             const ndview<Real, 2> &xyz) {
    int nx = x.extent(0);
    int ny = y.extent(0);
    int nz = z.extent(0);

    int ind = 0;
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                xyz(0, ind) = x(ix);
                xyz(1, ind) = y(iy);
                xyz(2, ind) = z(iz);
                ind++;
            }
        }
    }
}

template <typename Real>
void mesh_nd(int dim, const ndview<const Real, 1> &in, const ndview<Real, 2> &out) {
    if (dim == 2)
        return mesh_2d(in, in, out);
    if (dim == 3)
        return mesh_3d(in, in, in, out);

    throw std::runtime_error("Invalid dimension " + std::to_string(dim) + "provided");
}

template <typename Real>
void mesh_nd(int dim, Real *in, int size, Real *out) {
    if (dim == 2) {
        ndview<const Real, 1> x(in, size);
        ndview<Real, 2> xy(out, 2, size * size);

        return mesh_2d(x, x, xy);
    }
    if (dim == 3) {
        ndview<const Real, 1> x(in, size);
        ndview<Real, 2> xyz(out, 3, size * size * size);

        return mesh_3d(x, x, x, xyz);
    }

    throw std::runtime_error("Invalid dimension " + std::to_string(dim) + "provided");
}

template <typename Real>
void mk_tensor_product_fourier_transform_1d(int npw, const ndview<Real, 1> &fhat, ndview<Real, 1> &pswfft) {
    int npw2 = npw / 2;
    int ind = 0;
    for (int j1 = -npw2; j1 <= 0; ++j1) {
        int k2 = j1 * j1;
        pswfft[ind++] = fhat[k2];
    }
}

template <typename Real>
void mk_tensor_product_fourier_transform_2d(int npw, const ndview<Real, 1> &fhat, ndview<Real, 1> &pswfft) {
    int npw2 = npw / 2;
    int npw3 = (npw - 1) / 2;
    int ind = 0;
    for (int j2 = -npw2; j2 <= 0; ++j2) {
        for (int j1 = -npw2; j1 <= npw3; ++j1) {
            // for symmetric trapezoidal rule - npw odd
            int k2 = j1 * j1 + j2 * j2;
            pswfft[ind++] = fhat[k2];
        }
    }
}

template <typename Real>
void mk_tensor_product_fourier_transform_3d(int npw, const ndview<Real, 1> &fhat, ndview<Real, 1> &pswfft) {
    int npw2 = npw / 2;
    int npw3 = (npw - 1) / 2;
    int ind = 0;
    // for symmetric trapezoidal rule - npw odd
    for (int j3 = -npw2; j3 <= 0; ++j3) {
        for (int j2 = -npw2; j2 <= npw3; ++j2) {
            for (int j1 = -npw2; j1 <= npw3; ++j1) {
                // for symmetric trapezoidal rule - npw odd
                int k2 = j1 * j1 + j2 * j2 + j3 * j3;
                pswfft[ind++] = fhat[k2];
            }
        }
    }
}

template <typename Real>
void mk_tensor_product_fourier_transform(int dim, int npw, const ndview<Real, 1> &fhat, ndview<Real, 1> pswfft) {
    if (dim == 1)
        return mk_tensor_product_fourier_transform_1d(npw, fhat, pswfft);
    if (dim == 2)
        return mk_tensor_product_fourier_transform_2d(npw, fhat, pswfft);
    if (dim == 3)
        return mk_tensor_product_fourier_transform_3d(npw, fhat, pswfft);
    throw std::runtime_error("Invalid dimension: " + std::to_string(dim));
}

template <typename Real>
void mk_tensor_product_fourier_transform(int dim, int npw, int nfourier, Real *fhat, int nexp, Real *pswfft) {
    ndview<const Real, 1> fhat_view(fhat, nfourier + 1);
    ndview<Real, 1> pswfft_view(pswfft, nexp);

    if (dim == 1) {
        return mk_tensor_product_fourier_transform_1d(npw, fhat_view, pswfft_view);
    }
    if (dim == 2) {
        return mk_tensor_product_fourier_transform_2d(npw, fhat_view, pswfft_view);
    }
    if (dim == 3) {
        return mk_tensor_product_fourier_transform_3d(npw, fhat_view, pswfft_view);
    }
    throw std::runtime_error("Invalid dimension: " + std::to_string(dim));
}

template void mk_tensor_product_fourier_transform(int dim, int npw, const ndview<float, 1> &fhat,
                                                  ndview<float, 1> pswfft);
template void mk_tensor_product_fourier_transform(int dim, int npw, const ndview<double, 1> &fhat,
                                                  ndview<double, 1> pswfft);

} // namespace dmk::util
