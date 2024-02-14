#include <Eigen/Core>

#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace dmk::proxy {

template <typename T>
void charge2proxycharge_2d(int n_charge_dim, int order, const std::vector<T> &r_src_, const std::vector<T> &charge_,
                           T center[2], T scale_factor, std::vector<T> &coeffs) {
    using MatrixMap = Eigen::Map<Eigen::MatrixX<T>>;
    using CMatrixMap = Eigen::Map<const Eigen::MatrixX<T>>;

    const int n_dim = 2;
    const int n_src = r_src_.size() / 2;
    assert(coeffs.size() == order * order * n_src * n_charge_dim);
    constexpr T alpha = 1.0, beta = 1.0;

    Eigen::MatrixX<T> dy(order, n_src);
    Eigen::MatrixX<T> poly_x(order, n_src);

    CMatrixMap r_src(r_src_.data(), n_dim, n_src);
    CMatrixMap charge(charge_.data(), n_charge_dim, n_src);

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
void charge2proxycharge_3d(int n_charge_dim, int order, const std::vector<T> &r_src_, const std::vector<T> &charge_,
                           T center[3], T scale_factor, std::vector<T> &coeffs) {}

template <typename T>
void charge2proxycharge(int n_dim, int n_charge_dim, int order, const std::vector<T> &r_src,
                        const std::vector<T> &charge, T *center, T scale_factor, std::vector<T> &coeffs) {
    if (n_dim == 2)
        return charge2proxycharge_2d(n_charge_dim, order, r_src, charge, center, scale_factor, coeffs);

    throw std::runtime_error("Invalid dimension " + std::to_string(n_dim) + "provided");
}

template void charge2proxycharge(int n_dim, int n_charge_dim, int order, const std::vector<float> &r_src,
                                 const std::vector<float> &charge, float *center, float scale_factor,
                                 std::vector<float> &coeffs);

template void charge2proxycharge(int n_dim, int n_charge_dim, int order, const std::vector<double> &r_src,
                                 const std::vector<double> &charge, double *center, double scale_factor,
                                 std::vector<double> &coeffs);
} // namespace dmk::proxy
