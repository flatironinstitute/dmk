#include <Eigen/Core>

#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <omp.h>

namespace dmk::proxy {

template <typename T>
void charge2proxycharge_2d(int n_charge_dim, int order, int n_src, const T *r_src_, const T *charge_, const T center[2],
                           T scale_factor, T *coeffs) {
    using MatrixMap = Eigen::Map<Eigen::MatrixX<T>>;
    using CMatrixMap = Eigen::Map<const Eigen::MatrixX<T>>;

    constexpr int n_dim = 2;

    Eigen::MatrixX<T> dy(order, n_src);
    Eigen::MatrixX<T> poly_x(order, n_src);

    CMatrixMap r_src(r_src_, n_dim, n_src);
    CMatrixMap charge(charge_, n_charge_dim, n_src);

    for (int i_src = 0; i_src < n_src; ++i_src)
        dmk::chebyshev::calc_polynomial(order, scale_factor * (r_src(0, i_src) - center[0]), &poly_x(0, i_src));

    for (int i_dim = 0; i_dim < n_charge_dim; ++i_dim) {
        for (int i_src = 0; i_src < n_src; ++i_src) {
            // we recalculate the polynomial rather than caching it because it's so cheap and more cache friendly
            T poly_y[order];
            dmk::chebyshev::calc_polynomial(order, scale_factor * (r_src(1, i_src) - center[1]), poly_y);
            for (int i = 0; i < order; ++i)
                dy(i, i_src) = charge(i_dim, i_src) * poly_y[i];
        }

        MatrixMap(&coeffs[i_dim * order * order], order, order) = poly_x * dy.transpose();
    }
}

template <typename T>
void charge2proxycharge_3d(int n_charge_dim, int order, int n_src, const T *r_src_, const T *charge_, const T center[2],
                           T scale_factor, T *coeffs) {
    using MatrixMap = Eigen::Map<Eigen::MatrixX<T>>;
    using CMatrixMap = Eigen::Map<const Eigen::MatrixX<T>>;
    const int n_dim = 3;

    Eigen::MatrixX<T> dz(n_src, order);
    Eigen::MatrixX<T> dyz(n_src, order * order);
    Eigen::MatrixX<T> poly_x(order, n_src);
    Eigen::MatrixX<T> poly_y(order, n_src);
    Eigen::MatrixX<T> poly_z(order, n_src);

    CMatrixMap r_src(r_src_, n_dim, n_src);
    CMatrixMap charge(charge_, n_charge_dim, n_src);

    for (int i_src = 0; i_src < n_src; ++i_src)
        dmk::chebyshev::calc_polynomial(order, scale_factor * (r_src(0, i_src) - center[0]), &poly_x(0, i_src));
    for (int i_src = 0; i_src < n_src; ++i_src)
        dmk::chebyshev::calc_polynomial(order, scale_factor * (r_src(1, i_src) - center[1]), &poly_y(0, i_src));
    for (int i_src = 0; i_src < n_src; ++i_src)
        dmk::chebyshev::calc_polynomial(order, scale_factor * (r_src(2, i_src) - center[2]), &poly_z(0, i_src));

    for (int i_dim = 0; i_dim < n_charge_dim; ++i_dim) {
        for (int k = 0; k < order; ++k)
            for (int m = 0; m < n_src; ++m)
                dz(m, k) = charge(i_dim, m) * poly_z(k, m);

        for (int k = 0; k < order; ++k)
            for (int j = 0; j < order; ++j)
                for (int m = 0; m < n_src; ++m)
                    dyz(m, j + k * order) = poly_y(j, m) * dz(m, k);

        MatrixMap(&coeffs[i_dim * order * order * order], order, order * order) = poly_x * dyz;
    }
}

template <typename T>
void charge2proxycharge(int n_dim, int n_charge_dim, int order, int n_src, const T *r_src, const T *charge,
                        const T *center, T scale_factor, T *coeffs) {
    if (n_dim == 2)
        return charge2proxycharge_2d(n_charge_dim, order, n_src, r_src, charge, center, scale_factor, coeffs);
    if (n_dim == 3)
        return charge2proxycharge_3d(n_charge_dim, order, n_src, r_src, charge, center, scale_factor, coeffs);

    throw std::runtime_error("Invalid dimension " + std::to_string(n_dim) + "provided");
}

template void charge2proxycharge(int n_dim, int n_charge_dim, int order, int n_src, const float *r_src,
                                 const float *charge, const float *center, float scale_factor, float *coeffs);
template void charge2proxycharge(int n_dim, int n_charge_dim, int order, int n_src, const double *r_src,
                                 const double *charge, const double *center, double scale_factor, double *coeffs);
} // namespace dmk::proxy
