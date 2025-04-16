#include <Eigen/Core>

#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/types.hpp>
#include <dmk/util.hpp>
#include <limits>
#include <random>

namespace dmk::util {
using dmk::ndview;
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
void mk_tensor_product_fourier_transform_1d(int npw, const ndview<const Real, 1> &fhat, const ndview<Real, 1> &pswfft) {
    int npw2 = npw / 2;
    int ind = 0;
    for (int j1 = -npw2; j1 <= 0; ++j1) {
        int k2 = j1 * j1;
        pswfft[ind++] = fhat[k2];
    }
}

template <typename Real>
void mk_tensor_product_fourier_transform_2d(int npw, const ndview<const Real, 1> &fhat, const ndview<Real, 1> &pswfft) {
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
void mk_tensor_product_fourier_transform_3d(int npw, const ndview<const Real, 1> &fhat, const ndview<Real, 1> &pswfft) {
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
void mk_tensor_product_fourier_transform(int dim, int npw, const ndview<const Real, 1> &fhat,
                                         const ndview<Real, 1> &pswfft) {
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

template <typename Real>
void init_test_data(int n_dim, int nd, int n_src, int n_trg, bool uniform, bool set_fixed_charges,
                    sctl::Vector<Real> &r_src, sctl::Vector<Real> &r_trg, sctl::Vector<Real> &rnormal,
                    sctl::Vector<Real> &charges, sctl::Vector<Real> &dipstr, long seed) {
    r_src.ReInit(n_dim * n_src);
    r_trg.ReInit(n_dim * n_trg);
    charges.ReInit(nd * n_src);
    rnormal.ReInit(n_dim * n_src);
    dipstr.ReInit(nd * n_src);

    double rin = 0.45;
    double wrig = 0.12;
    double rwig = 0;
    int nwig = 6;
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> rng;

    for (int i = 0; i < n_src; ++i) {
        if (!uniform) {
            if (n_dim == 2) {
                double phi = rng(eng) * 2 * M_PI;
                r_src[i * 3 + 0] = cos(phi);
                r_src[i * 3 + 1] = sin(phi);
            }
            if (n_dim == 3) {
                double theta = rng(eng) * M_PI;
                double rr = rin + rwig * cos(nwig * theta);
                double ct = cos(theta);
                double st = sin(theta);
                double phi = rng(eng) * 2 * M_PI;
                double cp = cos(phi);
                double sp = sin(phi);

                r_src[i * 3 + 0] = rr * st * cp + 0.5;
                r_src[i * 3 + 1] = rr * st * sp + 0.5;
                r_src[i * 3 + 2] = rr * ct + 0.5;
            }
        } else {
            for (int j = 0; j < n_dim; ++j)
                r_src[i * n_dim + j] = rng(eng);
        }

        for (int j = 0; j < n_dim; ++j)
            rnormal[i * n_dim + j] = rng(eng);

        for (int j = 0; j < nd; ++j) {
            charges[i * nd + j] = rng(eng) - 0.5;
            dipstr[i * nd + j] = rng(eng);
        }
    }

    for (int i_trg = 0; i_trg < n_trg; ++i_trg) {
        if (!uniform) {
            if (n_dim == 2) {
                double phi = rng(eng) * 2 * M_PI;
                r_trg[i_trg * 3 + 0] = cos(phi);
                r_trg[i_trg * 3 + 1] = sin(phi);
            }
            if (n_dim == 3) {
                double theta = rng(eng) * M_PI;
                double rr = rin + rwig * cos(nwig * theta);
                double ct = cos(theta);
                double st = sin(theta);
                double phi = rng(eng) * 2 * M_PI;
                double cp = cos(phi);
                double sp = sin(phi);

                r_trg[i_trg * 3 + 0] = rr * st * cp + 0.5;
                r_trg[i_trg * 3 + 1] = rr * st * sp + 0.5;
                r_trg[i_trg * 3 + 2] = rr * ct + 0.5;
            }
        } else {
            for (int j = 0; j < n_dim; ++j)
                r_trg[i_trg * n_dim + j] = rng(eng);
        }
    }

    if (set_fixed_charges && n_src > 0)
        for (int i = 0; i < n_dim; ++i)
            r_src[i] = 0.0;
    if (set_fixed_charges && n_src > 1)
        for (int i = n_dim; i < 2 * n_dim; ++i)
            r_src[i] = 1 - std::numeric_limits<Real>::epsilon();
    if (set_fixed_charges && n_src > 2)
        for (int i = 2 * n_dim; i < 3 * n_dim; ++i)
            r_src[i] = 0.05;
}

template void init_test_data<float>(int n_dim, int nd, int n_src, int n_trg, bool uniform, bool set_fixed_charges,
                                    sctl::Vector<float> &r_src, sctl::Vector<float> &r_trg,
                                    sctl::Vector<float> &rnormal, sctl::Vector<float> &charges,
                                    sctl::Vector<float> &dipstr, long seed);

template void init_test_data<double>(int n_dim, int nd, int n_src, int n_trg, bool uniform, bool set_fixed_charges,
                                     sctl::Vector<double> &r_src, sctl::Vector<double> &r_trg,
                                     sctl::Vector<double> &rnormal, sctl::Vector<double> &charges,
                                     sctl::Vector<double> &dipstr, long seed);

template void mk_tensor_product_fourier_transform(int dim, int npw, const ndview<const double, 1> &fhat,
                                                  const ndview<double, 1> &pswfft);
template void mk_tensor_product_fourier_transform(int dim, int npw, const ndview<const float, 1> &fhat,
                                                  const ndview<float, 1> &pswfft);

} // namespace dmk::util
