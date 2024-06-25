#include <Eigen/Core>

#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/gemm.hpp>
#include <dmk/planewave.hpp>
#include <mdspan.hpp>
#include <dmk/types.hpp>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <omp.h>

namespace dmk::proxy {

template <typename T>
void proxycharge2pw_2d(int n_charge_dim, int n_order, int n_pw, const T *proxy_coeffs, const std::complex<T> *poly2pw,
                       std::complex<T> *pw_expansion) {
    using dmk::gemm::gemm;
    const int n_pw2 = (n_pw + 1) / 2;
    const int n_pw_coeffs = n_pw * n_pw2;
    const int n_poly2pw_coeffs = n_order * n_pw;
    const int n_proxy_coeffs = n_order * n_order;

    sctl::Vector<std::complex<T>> proxy_coeffs_complex(n_proxy_coeffs);
    sctl::Vector<std::complex<T>> ff(n_order * n_pw2);

    for (int i_dim = 0; i_dim < n_charge_dim; ++i_dim) {
        for (int i = 0; i < n_proxy_coeffs; ++i)
            proxy_coeffs_complex[i] = {proxy_coeffs[i + i_dim * n_proxy_coeffs], 0.0};

        // transform in y
        gemm('n', 't', n_order, n_pw2, n_order, {1.0, 0.0}, &proxy_coeffs_complex[0], n_order, poly2pw, n_pw,
             {0.0, 0.0}, &ff[0], n_order);

        // transform in x
        gemm('n', 'n', n_pw, n_pw2, n_order, {1.0, 0.0}, &poly2pw[0], n_pw, &ff[0], n_order, {0.0, 0.0},
             &pw_expansion[n_pw_coeffs * i_dim], n_pw);
    }
}

template <typename T>
void proxycharge2pw_3d(int n_charge_dim, int n_order, int n_pw, const T *proxy_coeffs, const std::complex<T> *poly2pw,
                       std::complex<T> *pw_expansion) {
    using dmk::gemm::gemm;
    const int n_pw2 = (n_pw + 1) / 2;
    const int n_proxy_coeffs = sctl::pow<3>(n_order);
    const int n_pw_coeffs = n_pw * n_pw * n_pw2;

    sctl::Vector<std::complex<T>> ff(n_order * n_order * n_pw2);
    sctl::Vector<std::complex<T>> fft(n_order * n_pw2 * n_order);
    sctl::Vector<std::complex<T>> ff2(n_pw * n_pw2 * n_order);
    sctl::Vector<std::complex<T>> proxy_coeffs_complex(n_order * n_order * n_order);

    ndview<std::complex<T>, 3> ff_view(&ff[0], n_order, n_order, n_pw2);
    ndview<std::complex<T>, 3> fft_view(&fft[0], n_order, n_pw2, n_order);

    for (int i_dim = 0; i_dim < n_charge_dim; ++i_dim) {
        for (int i = 0; i < n_proxy_coeffs; ++i)
            proxy_coeffs_complex[i] = proxy_coeffs[i + n_proxy_coeffs * i_dim];

        // transform in z
        gemm('n', 't', n_order * n_order, n_pw2, n_order, {1.0, 0.0}, &proxy_coeffs_complex[0], n_order * n_order,
             poly2pw, n_pw, {0.0, 0.0}, &ff[0], n_order * n_order);

        for (int m1 = 0; m1 < n_order; ++m1)
            for (int k3 = 0; k3 < n_pw2; ++k3)
                for (int m2 = 0; m2 < n_order; ++m2)
                    fft_view(m2, k3, m1) = ff_view(m1, m2, k3);

        // transform in y
        gemm('n', 'n', n_pw, n_pw2 * n_order, n_order, {1.0, 0.0}, poly2pw, n_pw, &fft[0], n_order, {0.0, 0.0}, &ff2[0],
             n_pw);

        // transform in x
        gemm('n', 't', n_pw, n_pw * n_pw2, n_order, {1.0, 0.0}, poly2pw, n_pw, &ff2[0], n_pw * n_pw2, {0.0, 0.0},
             &pw_expansion[i_dim * n_pw_coeffs], n_pw);
    }
}

template <typename T>
void proxycharge2pw(int n_dim, int n_charge_dim, int n_order, int n_pw, const T *proxy_coeffs,
                    const std::complex<T> *poly2pw, std::complex<T> *pw_expansion) {
    if (n_dim == 2)
        return proxycharge2pw_2d(n_charge_dim, n_order, n_pw, proxy_coeffs, poly2pw, pw_expansion);
    if (n_dim == 3)
        return proxycharge2pw_3d(n_charge_dim, n_order, n_pw, proxy_coeffs, poly2pw, pw_expansion);

    throw std::runtime_error("Invalid dimension " + std::to_string(n_dim) + "provided");
}

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

// template void charge2proxycharge(int n_dim, int n_charge_dim, int order, int n_src, const float *r_src,
//                                  const float *charge, const float *center, float scale_factor, float *coeffs);
template void charge2proxycharge(int n_dim, int n_charge_dim, int order, int n_src, const double *r_src,
                                 const double *charge, const double *center, double scale_factor, double *coeffs);
// template void proxycharge2pw(int n_dim, int n_charge_dim, int n_order, int n_pw, const float *proxy_coeffs,
//                              const std::complex<float> *poly2pw, std::complex<float> *pw_expansion);
template void proxycharge2pw(int n_dim, int n_charge_dim, int n_order, int n_pw, const double *proxy_coeffs,
                             const std::complex<double> *poly2pw, std::complex<double> *pw_expansion);

TEST_CASE("[DMK] proxycharge2pw") {
    const int n_charge_dim = 1;
    const int n_pw = 10;
    const int n_pw2 = (n_pw + 1) / 2;
    const int n_pw_coeffs = n_pw * n_pw2;

    for (int n_dim : {2, 3}) {
        CAPTURE(n_dim);
        for (int n_order : {10, 16, 24}) {
            const int n_pw_modes = int(std::pow(n_pw, n_dim - 1)) * ((n_pw + 1) / 2);
            const int n_pw_coeffs = n_pw_modes * n_charge_dim;

            CAPTURE(n_order);
            sctl::Vector<double> proxy_coeffs(int(pow(n_order, n_dim)) * n_charge_dim);
            sctl::Vector<std::complex<double>> poly2pw(n_order * n_pw), pw2poly(n_order * n_pw);
            Eigen::VectorX<std::complex<double>> pw_coeffs(n_pw_coeffs), pw_coeffs_fort(n_pw_coeffs);

            dmk::calc_planewave_coeff_matrices(1.0, 1.0, n_pw, n_order, poly2pw, pw2poly);

            for (auto &c : proxy_coeffs)
                c = drand48();

            pw_coeffs.array() = 0.0;
            proxycharge2pw(n_dim, n_charge_dim, n_order, n_pw, &proxy_coeffs[0], &poly2pw[0], &pw_coeffs[0]);

            pw_coeffs_fort.array() = 0.0;
            dmk_proxycharge2pw_(&n_dim, &n_charge_dim, &n_order, &proxy_coeffs[0], &n_pw, (double *)&poly2pw[0],
                                (double *)&pw_coeffs_fort[0]);

            const double l2 = (pw_coeffs - pw_coeffs_fort).norm() / pw_coeffs.size();
            CHECK(l2 < std::numeric_limits<double>::epsilon());
        }
    }
}

TEST_CASE("[DMK] charge2proxycharge") {
    const int n_src = 500;
    const int n_charge_dim = 2;

    for (int n_dim : {2, 3}) {
        CAPTURE(n_dim);
        for (int n_order : {10, 16, 24}) {
            CAPTURE(n_order);
            Eigen::VectorX<double> r_src(n_src * n_dim);
            Eigen::VectorX<double> charge(n_src * n_charge_dim);
            Eigen::VectorX<double> coeffs(int(pow(n_order, n_dim)) * n_charge_dim);
            Eigen::VectorX<double> coeffs_fort(int(pow(n_order, n_dim)) * n_charge_dim);
            const double center[] = {0.5, 0.5, 0.5};
            const double scale_factor = 1.2;

            for (int i = 0; i < n_src * n_dim; ++i)
                r_src[i] = drand48();

            for (int i = 0; i < n_src * n_charge_dim; ++i)
                charge[i] = drand48() - 0.5;

            coeffs.array() = 0.0;
            dmk::proxy::charge2proxycharge(n_dim, n_charge_dim, n_order, n_src, r_src.data(), charge.data(), center,
                                           scale_factor, coeffs.data());

            coeffs_fort.array() = 0.0;
            pdmk_charge2proxycharge_(&n_dim, &n_charge_dim, &n_order, &n_src, r_src.data(), charge.data(), center,
                                     &scale_factor, coeffs_fort.data());

            const double l2 = (coeffs - coeffs_fort).norm() / coeffs.size();
            CHECK(l2 < std::numeric_limits<double>::epsilon());
        }
    }
}

} // namespace dmk::proxy
