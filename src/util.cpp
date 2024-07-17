#include <Eigen/Core>

#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/types.hpp>
#include <type_traits>
#include <vector>

namespace dmk::util {
using dmk::ndview;
template <typename Real>
void mesh_2d(ndview<const Real, 1> &x, ndview<const Real, 1> &y, ndview<Real, 2> &xy) {
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
void mesh_3d(ndview<const Real, 1> &x, ndview<const Real, 1> &y, ndview<const Real, 1> &z, ndview<Real, 2> &xyz) {
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
void mesh_nd(int dim, ndview<const Real, 1> &in, ndview<Real, 2> &out) {
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
void mk_tensor_product_fourier_transform_1d(int npw, ndview<const Real, 1> &fhat, ndview<Real, 1> &pswfft) {
    int npw2 = npw / 2;
    int ind = 0;
    for (int j1 = -npw2; j1 <= 0; ++j1) {
        int k2 = j1 * j1;
        pswfft[ind++] = fhat[k2];
    }
}

template <typename Real>
void mk_tensor_product_fourier_transform_2d(int npw, ndview<const Real, 1> &fhat, ndview<Real, 1> &pswfft) {
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
void mk_tensor_product_fourier_transform_3d(int npw, ndview<const Real, 1> &fhat, ndview<Real, 1> &pswfft) {
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

TEST_CASE("[DMK] mesh_nd") {
    for (int dim : {2, 3}) {
        std::vector<double> in = {1.0, 2.0, 3.0};
        const int size = in.size();
        const int nxyz = dim * pow(size, dim);
        std::vector<double> out(nxyz);
        std::vector<double> out_fort(nxyz);
        mesh_nd(dim, in.data(), size, out.data());
        meshnd_(&dim, in.data(), &size, out_fort.data());
        CHECK(out == out_fort);

        for (auto &xval : out)
            xval = 0.0;

        ndview<const double, 1> in_view(in.data(), size);
        ndview<double, 2> out_view(out.data(), dim, pow(size, dim));
        mesh_nd(dim, in_view, out_view);
        CHECK(out == out_fort);
    }
}

TEST_CASE("[DMK] mk_tensor_product_fourier_transform") {
    for (int dim : {1, 2, 3}) {
        const int npw = 5;
        const int nexp = pow(npw, dim - 1) * ((npw + 1) / 2);
        std::vector<double> fhat = {1.0, 2.0, 3.0, 4.0, 5.0};
        const int nfourier = fhat.size() - 1;
        std::vector<double> pswfft(nexp);
        std::vector<double> pswfft_fort(nexp);

        mk_tensor_product_fourier_transform(dim, npw, nfourier, fhat.data(), nexp, pswfft.data());
        mk_tensor_product_fourier_transform_(&dim, &npw, &nfourier, fhat.data(), &nexp, pswfft_fort.data());

        for (int i = 0; i < nexp; ++i) {
            CHECK(std::abs(pswfft[i] - pswfft_fort[i]) < 1e-6);
        }

    }
}

} // namespace dmk::util
