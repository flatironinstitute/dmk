#include <Eigen/Core>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/gemm.hpp>
#include <dmk/planewave.hpp>
#include <dmk/types.hpp>
#include <stdexcept>

namespace dmk {

template <typename Real>
void pw2proxypot_2d(const ndview<const std::complex<Real>, 3> &pw_expansion,
                    const ndview<const std::complex<Real>, 2> &pw_to_coefs_mat, const ndview<Real, 3> &proxy_coeffs) {
    using dmk::gemm::gemm;

    const int n_order = proxy_coeffs.extent(0);
    const int n_charge_dim = proxy_coeffs.extent(2);
    const int n_pw = pw_expansion.extent(0);
    const int n_pw2 = pw_expansion.extent(1);

    sctl::Vector<std::complex<Real>> ff_(n_order * n_pw2);
    sctl::Vector<std::complex<Real>> zcoefs_(n_order * n_order);

    ndview<std::complex<Real>, 2> ff(&ff_[0], n_order, n_pw2);
    ndview<std::complex<Real>, 2> zcoefs(&zcoefs_[0], n_order, n_order);

    const int npw_half = n_pw / 2;

    const std::complex<Real> alpha = {1.0, 0.0};
    const std::complex<Real> beta = {0.0, 0.0};
    for (int i = 0; i < n_charge_dim; ++i) {
        gemm('t', 'n', n_order, n_pw2, n_pw, alpha, &pw_to_coefs_mat(0, 0), n_pw, &pw_expansion(0, 0, i), n_pw, beta,
             &ff(0, 0), n_order);

        for (int m2 = 0; m2 < n_pw2; ++m2)
            for (int k1 = 0; k1 < n_order; ++k1)
                if (m2 >= npw_half)
                    ff(k1, m2) = Real{0.5} * ff(k1, m2);

        gemm('n', 'n', n_order, n_order, n_pw2, alpha, &ff(0, 0), n_order, &pw_to_coefs_mat(0, 0), n_pw, beta,
             &zcoefs(0, 0), n_order);

        for (int k2 = 0; k2 < n_order; ++k2)
            for (int k1 = 0; k1 < n_order; ++k1)
                proxy_coeffs(k1, k2, i) += zcoefs(k1, k2).real() * Real{2.0};
    }
}

template <typename Real>
void pw2proxypot_3d(const ndview<const std::complex<Real>, 4> &pw_expansion,
                    const ndview<const std::complex<Real>, 2> &pw_to_coefs_mat, const ndview<Real, 4> &proxy_coeffs) {
    using dmk::gemm::gemm;

    const int n_order = proxy_coeffs.extent(0);
    const int n_charge_dim = proxy_coeffs.extent(3);
    const int n_pw = pw_expansion.extent(0);
    const int n_pw2 = pw_expansion.extent(2);

    sctl::Vector<std::complex<Real>> ff_(n_order * n_pw * n_pw2);
    sctl::Vector<std::complex<Real>> fft_(n_pw * n_pw2 * n_order);
    sctl::Vector<std::complex<Real>> ff2t_(n_order * n_pw2 * n_order);
    sctl::Vector<std::complex<Real>> ff2_(n_order * n_order * n_pw2);
    sctl::Vector<std::complex<Real>> zcoefs_(n_order * n_order * n_order);

    ndview<std::complex<Real>, 3> ff(&ff_[0], n_order, n_pw, n_pw2);
    ndview<std::complex<Real>, 3> fft(&fft_[0], n_pw, n_pw2, n_order);
    ndview<std::complex<Real>, 3> ff2t(&ff2t_[0], n_order, n_pw2, n_order);
    ndview<std::complex<Real>, 3> ff2(&ff2_[0], n_order, n_order, n_pw2);
    ndview<std::complex<Real>, 3> zcoefs(&zcoefs_[0], n_order, n_order, n_order);

    const int npw_half = n_pw / 2;
    const std::complex<Real> alpha = {1.0, 0.0};
    const std::complex<Real> beta = {0.0, 0.0};
    for (int i = 0; i < n_charge_dim; ++i) {
        gemm('t', 'n', n_order, n_pw * n_pw2, n_pw, alpha, &pw_to_coefs_mat(0, 0), n_pw, &pw_expansion(0, 0, 0, i),
             n_pw, beta, &ff(0, 0, 0), n_order);

        for (int k1 = 0; k1 < n_order; ++k1)
            for (int m3 = 0; m3 < n_pw2; ++m3)
                for (int m2 = 0; m2 < n_pw; ++m2)
                    fft(m2, m3, k1) = ff(k1, m2, m3);

        gemm('t', 'n', n_order, n_pw2 * n_order, n_pw, alpha, &pw_to_coefs_mat(0, 0), n_pw, &fft(0, 0, 0), n_pw, beta,
             &ff2t(0, 0, 0), n_order);

        for (int m3 = 0; m3 < n_pw2; ++m3) {
            for (int k2 = 0; k2 < n_order; ++k2) {
                for (int k1 = 0; k1 < n_order; ++k1) {
                    ff2(k1, k2, m3) = ff2t(k2, m3, k1);
                    if (m3 >= npw_half)
                        ff2(k1, k2, m3) = Real{0.5} * ff2t(k2, m3, k1);
                }
            }
        }

        gemm('n', 'n', n_order * n_order, n_order, n_pw2, alpha, &ff2(0, 0, 0), n_order * n_order,
             &pw_to_coefs_mat(0, 0), n_pw, beta, &zcoefs(0, 0, 0), n_order * n_order);

        for (int k3 = 0; k3 < n_order; ++k3)
            for (int k2 = 0; k2 < n_order; ++k2)
                for (int k1 = 0; k1 < n_order; ++k1)
                    proxy_coeffs(k1, k2, k3, i) += zcoefs(k1, k2, k3).real() * Real{2.0};
    }
}

template <typename Real, int DIM>
void planewave_to_proxy_potential(const ndview<const std::complex<Real>, DIM + 1> &pw_expansion,
                                  const ndview<const std::complex<Real>, 2> &pw_to_coefs_mat,
                                  const ndview<Real, DIM + 1> &proxy_coeffs) {
    if constexpr (DIM == 2) {
        return pw2proxypot_2d(pw_expansion, pw_to_coefs_mat, proxy_coeffs);
    }
    if constexpr (DIM == 3) {
        return pw2proxypot_3d(pw_expansion, pw_to_coefs_mat, proxy_coeffs);
    }
    throw std::runtime_error("Invalid dimension " + std::to_string(DIM) + " provided");
}

template <typename T>
void calc_planewave_coeff_matrices(T boxsize, T hpw, int n_pw, int n_order, sctl::Vector<std::complex<T>> &prox2pw_vec,
                                   sctl::Vector<std::complex<T>> &pw2poly_vec) {
    assert(n_pw * n_order == prox2pw_vec.Dim());
    assert(n_pw * n_order == pw2poly_vec.Dim());

    using matrix_t = Eigen::MatrixX<std::complex<T>>;
    const T dsq = 0.5 * boxsize;
    const auto xs = dmk::chebyshev::get_cheb_nodes(n_order, -1.0, 1.0);

    Eigen::Map<matrix_t> prox2pw(&prox2pw_vec[0], n_pw, n_order);
    Eigen::Map<matrix_t> pw2poly(&pw2poly_vec[0], n_pw, n_order);

    matrix_t tmp(n_pw, n_order);
    const int shift = n_pw / 2;
    for (int i = 0; i < n_order; ++i) {
        const T factor = xs[i] * dsq * hpw;
        for (int j = 0; j < n_pw; ++j)
            tmp(j, i) = exp(std::complex<T>{0, T(j - shift) * factor});
    }

    const auto &[vmat, umat_lu] = chebyshev::get_vandermonde_and_LU<T>(n_order);
    // Can't use umat_lu.solve() because eigen doesn't support LU with mixed complex/real types
    const Eigen::MatrixX<T> umat = umat_lu.inverse();
    pw2poly = tmp * umat.transpose();

    for (int i = 0; i < n_order * n_pw; ++i)
        prox2pw(i) = std::conj(pw2poly(i));
}

template <int DIM, typename T>
void calc_planewave_translation_matrix(int nmax, T xmin, int npw, T hpw, sctl::Vector<std::complex<T>> &shift_vec) {
    static_assert(DIM > 0 && DIM <= 3, "Invalid DIM");
    assert(((npw + 1) / 2) * sctl::pow<DIM - 1>(npw) * sctl::pow<DIM>(2 * nmax + 1) == shift_vec.Dim());
    const int shift = npw / 2;

    // Temporary array precomp
    sctl::Vector<std::complex<T>> ww(npw * (2 * nmax + 1));
    for (int j1 = 0; j1 < npw; ++j1) {
        T ts = (j1 - shift) * hpw;
        std::complex<T> ztmp = std::exp<T>(std::complex<T>{0.0, ts * xmin});
        ww[j1 + npw * nmax] = 1;
        for (int k1 = 1; k1 <= nmax; ++k1) {
            ww[j1 + npw * (nmax + k1)] = ztmp;
            ww[j1 + npw * (nmax - k1)] = conj(ztmp);
            ztmp *= ztmp;
        }
    }

    // Calculating wshift
    if constexpr (DIM == 1)
        for (int k1 = 0, i = 0; k1 <= 2 * nmax; ++k1)
            for (int j1 = 0; j1 < (npw + 1) / 2; ++j1)
                shift_vec[i++] = ww[j1 + npw * k1];
    else if constexpr (DIM == 2)
        for (int k1 = 0, i = 0; k1 <= 2 * nmax; ++k1)
            for (int k2 = 0; k2 <= 2 * nmax; ++k2)
                for (int j1 = 0; j1 < (npw + 1) / 2; ++j1)
                    for (int j2 = 0; j2 < npw; ++j2)
                        shift_vec[i++] = ww[j1 + npw * k1] * ww[j2 + npw * k2];
    else if constexpr (DIM == 3)
        for (int k1 = 0, i = 0; k1 <= 2 * nmax; ++k1)
            for (int k2 = 0; k2 <= 2 * nmax; ++k2)
                for (int k3 = 0; k3 <= 2 * nmax; ++k3)
                    for (int j1 = 0; j1 < (npw + 1) / 2; ++j1)
                        for (int j2 = 0; j2 < npw; ++j2)
                            for (int j3 = 0; j3 < npw; ++j3)
                                shift_vec[i++] = ww[j1 + npw * k1] * ww[j2 + npw * k2] * ww[j3 + npw * k3];
}
} // namespace dmk

template void dmk::planewave_to_proxy_potential<float, 2>(const ndview<const std::complex<float>, 3> &pw_expansion,
                                                          const ndview<const std::complex<float>, 2> &pw_to_coefs_mat,
                                                          const ndview<float, 3> &proxy_coeffs);
template void dmk::planewave_to_proxy_potential<float, 3>(const ndview<const std::complex<float>, 4> &pw_expansion,
                                                          const ndview<const std::complex<float>, 2> &pw_to_coefs_mat,
                                                          const ndview<float, 4> &proxy_coeffs);
template void dmk::planewave_to_proxy_potential<double, 2>(const ndview<const std::complex<double>, 3> &pw_expansion,
                                                           const ndview<const std::complex<double>, 2> &pw_to_coefs_mat,
                                                           const ndview<double, 3> &proxy_coeffs);
template void dmk::planewave_to_proxy_potential<double, 3>(const ndview<const std::complex<double>, 4> &pw_expansion,
                                                           const ndview<const std::complex<double>, 2> &pw_to_coefs_mat,
                                                           const ndview<double, 4> &proxy_coeffs);
template void dmk::calc_planewave_translation_matrix<2>(int, float, int, float, sctl::Vector<std::complex<float>> &);
template void dmk::calc_planewave_translation_matrix<3>(int, float, int, float, sctl::Vector<std::complex<float>> &);
template void dmk::calc_planewave_translation_matrix<2>(int, double, int, double, sctl::Vector<std::complex<double>> &);
template void dmk::calc_planewave_translation_matrix<3>(int, double, int, double, sctl::Vector<std::complex<double>> &);

template void dmk::calc_planewave_coeff_matrices<float>(float boxsize, float hpw, int n_pw, int n_order,
                                                        sctl::Vector<std::complex<float>> &prox2pw_vec,
                                                        sctl::Vector<std::complex<float>> &pw2poly_vec);
template void dmk::calc_planewave_coeff_matrices<double>(double boxsize, double hpw, int n_pw, int n_order,
                                                         sctl::Vector<std::complex<double>> &prox2pw_vec,
                                                         sctl::Vector<std::complex<double>> &pw2poly_vec);

TEST_CASE("[DMK] planewave_to_proxy_potential") {
    const int n_pw = 10;
    const int n_charge_dim = 1;
    const int n_pw2 = (n_pw + 1) / 2;

    for (int n_dim : {2, 3}) {
        CAPTURE(n_dim);
        for (int n_order : {10, 16, 24}) {
            const int n_pw_terms = dmk::util::int_pow(n_pw, n_dim - 1) * n_pw2;
            const int n_proxy_terms = dmk::util::int_pow(n_order, n_dim);
            sctl::Vector<std::complex<double>> pw_expansion(n_pw_terms);
            sctl::Vector<std::complex<double>> pw_to_coefs_mat(n_order * n_pw);
            Eigen::VectorX<double> proxy_coeffs(n_proxy_terms), proxy_coeffs_fort(n_proxy_terms);

            for (auto &elem : pw_expansion)
                elem = std::complex<double>{rand() / double(RAND_MAX), rand() / double(RAND_MAX)};
            for (auto &elem : pw_to_coefs_mat)
                elem = std::complex<double>{rand() / double(RAND_MAX), rand() / double(RAND_MAX)};

            proxy_coeffs.setZero();
            proxy_coeffs_fort.setZero();

            if (n_dim == 2) {
                dmk::ndview<const std::complex<double>, 3> pw_expansion_view(&pw_expansion[0], n_pw, n_pw2,
                                                                             n_charge_dim);
                dmk::ndview<const std::complex<double>, 2> pw_to_coefs_mat_view(&pw_to_coefs_mat[0], n_pw, n_order);
                dmk::ndview<double, 3> proxy_coeffs_view(&proxy_coeffs[0], n_order, n_order, n_charge_dim);

                dmk::planewave_to_proxy_potential<double, 2>(pw_expansion_view, pw_to_coefs_mat_view,
                                                             proxy_coeffs_view);
            }

            if (n_dim == 3) {
                dmk::ndview<const std::complex<double>, 4> pw_expansion_view(&pw_expansion[0], n_pw, n_pw, n_pw2,
                                                                             n_charge_dim);
                dmk::ndview<const std::complex<double>, 2> pw_to_coefs_mat_view(&pw_to_coefs_mat[0], n_pw, n_order);
                dmk::ndview<double, 4> proxy_coeffs_view(&proxy_coeffs[0], n_order, n_order, n_order, n_charge_dim);

                dmk::planewave_to_proxy_potential<double, 3>(pw_expansion_view, pw_to_coefs_mat_view,
                                                             proxy_coeffs_view);
            }

            dmk_pw2proxypot_(&n_dim, &n_charge_dim, &n_order, &n_pw, (double *)&pw_expansion[0],
                             (double *)&pw_to_coefs_mat[0], &proxy_coeffs_fort[0]);

            const double l2 = (proxy_coeffs - proxy_coeffs_fort).norm() / proxy_coeffs.size();
            CHECK(l2 < std::numeric_limits<double>::epsilon());
        }
    }
}
