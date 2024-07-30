#include <Eigen/Core>
#include <dmk/chebychev.hpp>
#include <dmk/gemm.hpp>
#include <dmk/planewave.hpp>
#include <dmk/types.hpp>
#include <stdexcept>

namespace dmk {

template <typename Real>
void pw2proxypot_2d(int n_charge_dim, int n_order, int n_pw, const std::complex<Real> *pw_expansion_,
                    const std::complex<Real> *pw_to_coefs_mat_, Real *proxy_coeffs_) {
    using dmk::gemm::gemm;
    const int half_n_pw_p1 = (n_pw + 1) / 2;

    sctl::Vector<std::complex<Real>> ff_(n_order * half_n_pw_p1);
    sctl::Vector<std::complex<Real>> zcoefs_(n_order * n_order);

    ndview<const std::complex<Real>, 3> pw_expansion(pw_expansion_, n_pw, half_n_pw_p1, n_charge_dim);
    ndview<const std::complex<Real>, 2> pw_to_coefs_mat(pw_to_coefs_mat_, n_pw, n_order);
    ndview<Real, 3> proxy_coeffs(proxy_coeffs_, n_order, n_order, n_charge_dim);

    ndview<std::complex<Real>, 2> ff(&ff_[0], n_order, half_n_pw_p1);
    ndview<std::complex<Real>, 2> zcoefs(&zcoefs_[0], n_order, n_order);

    const int npw2 = n_pw / 2;
    const std::complex<Real> alpha = {1.0, 0.0};
    const std::complex<Real> beta = {0.0, 0.0};
    for (int i = 0; i < n_charge_dim; ++i) {
        gemm('t', 'n', n_order, half_n_pw_p1, n_pw, alpha, &pw_to_coefs_mat(0, 0), n_pw, &pw_expansion(0, 0, i), n_pw,
             beta, &ff(0, 0), n_order);

        for (int m2 = 0; m2 < half_n_pw_p1; ++m2)
            for (int k1 = 0; k1 < n_order; ++k1)
                if (m2 >= npw2)
                    ff(k1, m2) = Real{0.5} * ff(k1, m2);

        gemm('n', 'n', n_order, n_order, half_n_pw_p1, alpha, &ff(0, 0), n_order, &pw_to_coefs_mat(0, 0), n_pw, beta,
             &zcoefs(0, 0), n_order);

        for (int k2 = 0; k2 < n_order; ++k2)
            for (int k1 = 0; k1 < n_order; ++k1)
                proxy_coeffs(k1, k2, i) += zcoefs(k1, k2).real() * Real{2.0};
    }
}

template <typename Real>
void pw2proxypot_3d(int n_charge_dim, int n_order, int n_pw, const std::complex<Real> *pw_expansion_,
                    const std::complex<Real> *pw_to_coefs_mat_, Real *proxy_coeffs_) {
    using dmk::gemm::gemm;
    const int half_n_pw_p1 = (n_pw + 1) / 2;

    sctl::Vector<std::complex<Real>> ff_(n_order * n_pw * half_n_pw_p1);
    sctl::Vector<std::complex<Real>> fft_(n_pw * half_n_pw_p1 * n_order);
    sctl::Vector<std::complex<Real>> ff2t_(n_order * half_n_pw_p1 * n_order);
    sctl::Vector<std::complex<Real>> ff2_(n_order * n_order * half_n_pw_p1);
    sctl::Vector<std::complex<Real>> zcoefs_(n_order * n_order * n_order);

    ndview<const std::complex<Real>, 4> pw_expansion(pw_expansion_, n_pw, n_pw, half_n_pw_p1, n_charge_dim);
    ndview<const std::complex<Real>, 2> pw_to_coefs_mat(pw_to_coefs_mat_, n_pw, n_order);
    ndview<Real, 4> proxy_coeffs(proxy_coeffs_, n_order, n_order, n_order, n_charge_dim);

    ndview<std::complex<Real>, 3> ff(&ff_[0], n_order, n_pw, half_n_pw_p1);
    ndview<std::complex<Real>, 3> fft(&fft_[0], n_pw, half_n_pw_p1, n_order);
    ndview<std::complex<Real>, 3> ff2t(&ff2t_[0], n_order, half_n_pw_p1, n_order);
    ndview<std::complex<Real>, 3> ff2(&ff2_[0], n_order, n_order, half_n_pw_p1);
    ndview<std::complex<Real>, 3> zcoefs(&zcoefs_[0], n_order, n_order, n_order);

    const int npw2 = n_pw / 2;
    const std::complex<Real> alpha = {1.0, 0.0};
    const std::complex<Real> beta = {0.0, 0.0};
    for (int i = 0; i < n_charge_dim; ++i) {
        gemm('t', 'n', n_order, n_pw * half_n_pw_p1, n_pw, alpha, &pw_to_coefs_mat(0, 0), n_pw,
             &pw_expansion(0, 0, 0, i), n_pw, beta, &ff(0, 0, 0), n_order);

        for (int k1 = 0; k1 < n_order; ++k1)
            for (int m3 = 0; m3 < half_n_pw_p1; ++m3)
                for (int m2 = 0; m2 < n_pw; ++m2)
                    fft(m2, m3, k1) = ff(k1, m2, m3);

        gemm('t', 'n', n_order, half_n_pw_p1 * n_order, n_pw, alpha, &pw_to_coefs_mat(0, 0), n_pw, &fft(0, 0, 0), n_pw,
             beta, &ff2t(0, 0, 0), n_order);

        for (int m3 = 0; m3 < half_n_pw_p1; ++m3) {
            for (int k2 = 0; k2 < n_order; ++k2) {
                for (int k1 = 0; k1 < n_order; ++k1) {
                    ff2(k1, k2, m3) = ff2t(k2, m3, k1);
                    if (m3 >= npw2)
                        ff2(k1, k2, m3) = Real{0.5} * ff2t(k2, m3, k1);
                }
            }
        }

        gemm('n', 'n', n_order * n_order, n_order, half_n_pw_p1, alpha, &ff2(0, 0, 0), n_order * n_order,
             &pw_to_coefs_mat(0, 0), n_pw, beta, &zcoefs(0, 0, 0), n_order * n_order);

        for (int k3 = 0; k3 < n_order; ++k3)
            for (int k2 = 0; k2 < n_order; ++k2)
                for (int k1 = 0; k1 < n_order; ++k1)
                    proxy_coeffs(k1, k2, k3, i) += zcoefs(k1, k2, k3).real() * Real{2.0};
    }
}

template <typename Real>
void planewave_to_proxy_potential(int dim, int n_charge_dim, int n_order, int n_pw,
                                  const std::complex<Real> *pw_expansion_, const std::complex<Real> *pw_to_coefs_mat_,
                                  Real *proxy_coeffs_) {
    if (dim == 2)
        return pw2proxypot_2d(n_charge_dim, n_order, n_pw, pw_expansion_, pw_to_coefs_mat_, proxy_coeffs_);
    if (dim == 3)
        return pw2proxypot_3d(n_charge_dim, n_order, n_pw, pw_expansion_, pw_to_coefs_mat_, proxy_coeffs_);
    throw std::runtime_error("Invalid dim");
}

template <typename T>
void calc_planewave_coeff_matrices(double boxsize, T hpw, int n_pw, int n_order,
                                   sctl::Vector<std::complex<T>> &prox2pw_vec,
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
void calc_planewave_translation_matrix(int nmax, T xmin, int npw, const sctl::Vector<T> &ts,
                                       sctl::Vector<std::complex<T>> &shift_vec) {
    static_assert(DIM > 0 && DIM <= 3, "Invalid DIM");
    assert(((npw + 1) / 2) * sctl::pow<DIM - 1>(npw) * sctl::pow<DIM>(2 * nmax + 1) == shift_vec.Dim());

    // Temporary array precomp
    sctl::Vector<std::complex<T>> ww(npw * (2 * nmax + 1));
    for (int j1 = 0; j1 < npw; ++j1) {
        std::complex<T> ztmp = exp(std::complex<T>{0.0, ts[j1] * xmin});
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

// template void dmk::calc_planewave_translation_matrix<2>(int, float, int, const sctl::Vector<float> &,
//                                                         sctl::Vector<std::complex<float>> &);
// template void dmk::calc_planewave_translation_matrix<3>(int, float, int, const sctl::Vector<float> &,
//                                                         sctl::Vector<std::complex<float>> &);
template void dmk::planewave_to_proxy_potential<double>(int, int, int, int, const std::complex<double> *,
                                                        const std::complex<double> *, double *);
template void dmk::calc_planewave_translation_matrix<2>(int, double, int, const sctl::Vector<double> &,
                                                        sctl::Vector<std::complex<double>> &);
template void dmk::calc_planewave_translation_matrix<3>(int, double, int, const sctl::Vector<double> &,
                                                        sctl::Vector<std::complex<double>> &);

template void dmk::calc_planewave_coeff_matrices<double>(double boxsize, double hpw, int n_pw, int n_order,
                                                         sctl::Vector<std::complex<double>> &prox2pw_vec,
                                                         sctl::Vector<std::complex<double>> &pw2poly_vec);
