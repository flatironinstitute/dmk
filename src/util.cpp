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
void mesh_nd(int dim, ndview<const Real, 1> &x, ndview<Real, 2> &out) {
    if (dim == 2)
        return mesh_2d(x, x, out);
    if (dim == 3)
        return mesh_3d(x, x, x, out);

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

} // namespace dmk::util
