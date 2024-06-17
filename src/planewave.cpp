#include <Eigen/Core>
#include <dmk/chebychev.hpp>
#include <dmk/planewave.hpp>

namespace dmk {

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
template void dmk::calc_planewave_translation_matrix<2>(int, double, int, const sctl::Vector<double> &,
                                                        sctl::Vector<std::complex<double>> &);
template void dmk::calc_planewave_translation_matrix<3>(int, double, int, const sctl::Vector<double> &,
                                                        sctl::Vector<std::complex<double>> &);

template void dmk::calc_planewave_coeff_matrices<double>(double boxsize, double hpw, int n_pw, int n_order,
                                                         sctl::Vector<std::complex<double>> &prox2pw_vec,
                                                         sctl::Vector<std::complex<double>> &pw2poly_vec);
