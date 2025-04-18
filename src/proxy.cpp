#include <Eigen/Core>

#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/gemm.hpp>
#include <dmk/planewave.hpp>
#include <dmk/types.hpp>
#include <limits>
#include <sctl.hpp>
#include <stdexcept>

#include <omp.h>

namespace dmk::proxy {
template <typename T>
void proxycharge2pw_2d(const ndview<const T, 3> &proxy_coeffs, const ndview<const std::complex<T>, 2> &poly2pw,
                       const ndview<std::complex<T>, 3> &pw_expansion, sctl::Vector<T> &workspace) {
    using dmk::gemm::gemm;
    const int n_order = proxy_coeffs.extent(0);
    const int n_charge_dim = proxy_coeffs.extent(2);
    const int n_pw = pw_expansion.extent(0);
    const int n_pw2 = pw_expansion.extent(1);
    const int n_proxy_coeffs = n_order * n_order;

    workspace.ReInit(2 * (n_proxy_coeffs + n_order * n_pw2));
    std::complex<T> *workspace_ptr = (std::complex<T> *)(&workspace[0]);
    std::complex<T> *__restrict proxy_coeffs_complex = workspace_ptr;
    std::complex<T> *__restrict ff = workspace_ptr + n_proxy_coeffs;

    for (int i_dim = 0; i_dim < n_charge_dim; ++i_dim) {
        for (int i = 0; i < n_proxy_coeffs; ++i)
            proxy_coeffs_complex[i] = {proxy_coeffs.data_handle()[i + i_dim * n_proxy_coeffs], 0.0};

        // transform in y
        gemm('n', 't', n_order, n_pw2, n_order, {1.0, 0.0}, proxy_coeffs_complex, n_order, poly2pw.data_handle(), n_pw,
             {0.0, 0.0}, &ff[0], n_order);

        // transform in x
        gemm('n', 'n', n_pw, n_pw2, n_order, {1.0, 0.0}, poly2pw.data_handle(), n_pw, ff, n_order, {0.0, 0.0},
             &pw_expansion(0, 0, i_dim), n_pw);
    }
}

template <typename T>
void proxycharge2pw_3d(const ndview<const T, 4> &proxy_coeffs, const ndview<const std::complex<T>, 2> &poly2pw,
                       const ndview<std::complex<T>, 4> &pw_expansion, sctl::Vector<T> &workspace) {
    using dmk::gemm::gemm;
    const int n_order = proxy_coeffs.extent(0);
    const int n_charge_dim = proxy_coeffs.extent(3);
    const int n_pw = pw_expansion.extent(0);
    const int n_pw2 = pw_expansion.extent(2);
    const int n_proxy_coeffs = sctl::pow<3>(n_order);
    const int n_pw_coeffs = n_pw * n_pw * n_pw2;

    workspace.ReInit(2 * (n_order * n_order * n_pw2 + n_order * n_pw2 * n_order + n_pw * n_pw2 * n_order +
                          n_order * n_order * n_order));
    std::complex<T> *workspace_ptr = (std::complex<T> *)(&workspace[0]);
    ndview<std::complex<T>, 3> ff(workspace_ptr, n_order, n_order, n_pw2);
    ndview<std::complex<T>, 3> fft(ff.data_handle() + ff.size(), n_order, n_pw2, n_order);
    ndview<std::complex<T>, 1> ff2(fft.data_handle() + fft.size(), n_pw * n_pw2 * n_order);
    ndview<std::complex<T>, 1> proxy_coeffs_complex(ff2.data_handle() + ff2.size(), n_order * n_order * n_order);

    for (int i_dim = 0; i_dim < n_charge_dim; ++i_dim) {
        for (int i = 0; i < n_proxy_coeffs; ++i)
            proxy_coeffs_complex[i] = proxy_coeffs.data_handle()[i + i_dim * n_proxy_coeffs];

        // transform in z
        gemm('n', 't', n_order * n_order, n_pw2, n_order, {1.0, 0.0}, &proxy_coeffs_complex[0], n_order * n_order,
             poly2pw.data_handle(), n_pw, {0.0, 0.0}, ff.data_handle(), n_order * n_order);

        for (int m1 = 0; m1 < n_order; ++m1)
            for (int k3 = 0; k3 < n_pw2; ++k3)
                for (int m2 = 0; m2 < n_order; ++m2)
                    fft(m2, k3, m1) = ff(m1, m2, k3);

        // transform in y
        gemm('n', 'n', n_pw, n_pw2 * n_order, n_order, {1.0, 0.0}, poly2pw.data_handle(), n_pw, fft.data_handle(),
             n_order, {0.0, 0.0}, ff2.data_handle(), n_pw);

        // transform in x
        gemm('n', 't', n_pw, n_pw * n_pw2, n_order, {1.0, 0.0}, poly2pw.data_handle(), n_pw, ff2.data_handle(),
             n_pw * n_pw2, {0.0, 0.0}, &pw_expansion(0, 0, 0, i_dim), n_pw);
    }

    const auto n_flops_per_mm = [](int m, int n, int k) { return 8 * m * n * k; };
    // clang-format off
    const unsigned long n_flops = n_charge_dim * (n_flops_per_mm(n_order * n_order, n_pw2, n_order) +
                                                  n_flops_per_mm(n_pw, n_pw2 * n_order, n_order) +
                                                  n_flops_per_mm(n_pw, n_pw * n_pw2, n_order));
    // clang-format on
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, n_flops);
}

template <typename T, int DIM>
void proxycharge2pw(const ndview<const T, DIM + 1> &proxy_coeffs, const ndview<const std::complex<T>, 2> &poly2pw,
                    const ndview<std::complex<T>, DIM + 1> &pw_expansion, sctl::Vector<T> &workspace) {
    if constexpr (DIM == 2)
        return proxycharge2pw_2d(proxy_coeffs, poly2pw, pw_expansion, workspace);
    if constexpr (DIM == 3)
        return proxycharge2pw_3d(proxy_coeffs, poly2pw, pw_expansion, workspace);
    throw std::runtime_error("Invalid dimension " + std::to_string(DIM) + "provided");
}

template <typename T>
void proxycharge2pw(int n_dim, int n_charge_dim, int n_order, int n_pw, const T *proxy_coeffs,
                    const std::complex<T> *poly2pw, std::complex<T> *pw_expansion) {
    sctl::Vector<T> workspace;
    if (n_dim == 2) {
        ndview<const T, 3> proxy_coeffs_view(proxy_coeffs, n_order, n_order, n_charge_dim);
        ndview<const std::complex<T>, 2> poly2pw_view(poly2pw, n_pw, n_order);
        ndview<std::complex<T>, 3> pw_expansion_view(pw_expansion, n_pw, (n_pw + 1) / 2, n_charge_dim);

        return proxycharge2pw_2d(proxy_coeffs_view, poly2pw_view, pw_expansion_view, workspace);
    }
    if (n_dim == 3) {
        ndview<const T, 4> proxy_coeffs_view(proxy_coeffs, n_order, n_order, n_order, n_charge_dim);
        ndview<const std::complex<T>, 2> poly2pw_view(poly2pw, n_pw, n_order);
        ndview<std::complex<T>, 4> pw_expansion_view(pw_expansion, n_pw, n_pw, (n_pw + 1) / 2, n_charge_dim);

        return proxycharge2pw_3d(proxy_coeffs_view, poly2pw_view, pw_expansion_view, workspace);
    }
    throw std::runtime_error("Invalid dimension " + std::to_string(n_dim) + "provided");
}

template <typename T>
void charge2proxycharge_2d(const ndview<const T, 2> &r_src_, const ndview<const T, 2> &charge_,
                           const ndview<const T, 1> &center, T scale_factor, const ndview<T, 3> &coeffs,
                           sctl::Vector<T> &workspace) {
    using MatrixMap = Eigen::Map<Eigen::MatrixX<T>>;
    using CMatrixMap = Eigen::Map<const Eigen::MatrixX<T>>;

    constexpr int n_dim = 2;
    const int order = coeffs.extent(0);
    const int n_charge_dim = coeffs.extent(2);
    const int n_src = r_src_.extent(1);

    workspace.ReInit(2 * order * n_src + order);
    MatrixMap dy(&workspace[0], order, n_src);
    MatrixMap poly_x(&workspace[order * n_src], order, n_src);
    Eigen::Map<Eigen::VectorX<T>> poly_y(&workspace[2 * order * n_src], order);

    CMatrixMap r_src(r_src_.data_handle(), n_dim, n_src);
    CMatrixMap charge(charge_.data_handle(), n_charge_dim, n_src);

    for (int i_src = 0; i_src < n_src; ++i_src)
        dmk::chebyshev::calc_polynomial(order, scale_factor * (r_src(0, i_src) - center(0)), &poly_x(0, i_src));

    for (int i_dim = 0; i_dim < n_charge_dim; ++i_dim) {
        for (int i_src = 0; i_src < n_src; ++i_src) {
            // we recalculate the polynomial rather than caching it because it's so cheap and more cache friendly
            dmk::chebyshev::calc_polynomial(order, scale_factor * (r_src(1, i_src) - center(1)), poly_y.data());
            for (int i = 0; i < order; ++i)
                dy(i, i_src) = charge(i_dim, i_src) * poly_y[i];
        }

        MatrixMap(&coeffs(i_dim * order * order, 0, 0), order, order) += poly_x * dy.transpose();
    }
}

template <typename T>
void charge2proxycharge_3d(const ndview<const T, 2> &r_src_, const ndview<const T, 2> &charge_,
                           const ndview<const T, 1> &center, T scale_factor, const ndview<T, 4> &coeffs,
                           sctl::Vector<T> &workspace) {
    using MatrixMap = Eigen::Map<Eigen::MatrixX<T>>;
    using CMatrixMap = Eigen::Map<const Eigen::MatrixX<T>>;

    const int n_dim = 3;
    const int order = coeffs.extent(0);
    const int n_charge_dim = coeffs.extent(3);
    const int n_src = r_src_.extent(1);

    workspace.ReInit(4 * n_src * order + n_src * order * order);
    MatrixMap dz(&workspace[0], n_src, order);
    MatrixMap dyz(&workspace[n_src * order], n_src, order * order);
    MatrixMap poly_x(&workspace[n_src * order + n_src * order * order], order, n_src);
    MatrixMap poly_y(&workspace[2 * n_src * order + n_src * order * order], order, n_src);
    MatrixMap poly_z(&workspace[3 * n_src * order + n_src * order * order], order, n_src);

    CMatrixMap r_src(r_src_.data_handle(), n_dim, n_src);
    CMatrixMap charge(charge_.data_handle(), n_charge_dim, n_src);

    for (int i_src = 0; i_src < n_src; ++i_src) {
        dmk::chebyshev::calc_polynomial(order, scale_factor * (r_src(0, i_src) - center(0)), &poly_x(0, i_src));
        dmk::chebyshev::calc_polynomial(order, scale_factor * (r_src(1, i_src) - center(1)), &poly_y(0, i_src));
        dmk::chebyshev::calc_polynomial(order, scale_factor * (r_src(2, i_src) - center(2)), &poly_z(0, i_src));
    }

    for (int i_dim = 0; i_dim < n_charge_dim; ++i_dim) {
        for (int k = 0; k < order; ++k)
            for (int m = 0; m < n_src; ++m)
                dz(m, k) = charge(i_dim, m) * poly_z(k, m);

        for (int k = 0; k < order; ++k)
            for (int j = 0; j < order; ++j)
                for (int m = 0; m < n_src; ++m)
                    dyz(m, j + k * order) = poly_y(j, m) * dz(m, k);

        MatrixMap(&coeffs(i_dim * order * order * order, 0, 0, 0), order, order * order) += poly_x * dyz;
    }

    const int n_flops_per_poly = 3 * (order - 2);
    const int n_flops_per_mm = 2 * order * n_src * order * order;
    const int n_flops_tmp = order * n_src + order * order * n_src;
    const int n_flops_per_nd = 3 * n_flops_per_poly + n_flops_per_mm + n_flops_tmp;
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, n_charge_dim * n_flops_per_nd);
}

template <typename T, int DIM>
void charge2proxycharge(const ndview<const T, 2> &r_src_, const ndview<const T, 2> &charge_,
                        const ndview<const T, 1> &center, T scale_factor, const ndview<T, DIM + 1> &coeffs,
                        sctl::Vector<T> &workspace) {
    if constexpr (DIM == 2)
        return charge2proxycharge_2d(r_src_, charge_, center, scale_factor, coeffs, workspace);
    else if constexpr (DIM == 3)
        return charge2proxycharge_3d(r_src_, charge_, center, scale_factor, coeffs, workspace);
    else
        throw std::runtime_error("Invalid dimension " + std::to_string(DIM) + "provided");
}

template <typename T>
void eval_targets_2d(const ndview<const T, 3> &coeffs, const ndview<const T, 2> &r_trg, const ndview<const T, 1> &cen,
                     T sc, const ndview<T, 2> &pot, sctl::Vector<T> &workspace) {
    const int n_order = coeffs.extent(0);
    const int n_charge_dim = coeffs.extent(2);
    const int n_trg = r_trg.extent(1);

    workspace.ReInit(3 * n_order * n_trg);
    ndview<T, 2> poly_x(&workspace[0], n_order, n_trg);
    ndview<T, 2> poly_y(&workspace[n_order * n_trg], n_order, n_trg);
    ndview<T, 2> tmp(&workspace[2 * n_order * n_trg], n_order, n_trg);

    for (int i = 0; i < n_trg; ++i) {
        T x = (r_trg(0, i) - cen(0)) * sc;
        dmk::chebyshev::calc_polynomial(n_order, x, &poly_x(0, i));
    }
    for (int i = 0; i < n_trg; ++i) {
        T y = (r_trg(1, i) - cen(1)) * sc;
        dmk::chebyshev::calc_polynomial(n_order, y, &poly_y(0, i));
    }

    for (int i_dim = 0; i_dim < n_charge_dim; ++i_dim) {
        // Transform in y
        gemm::gemm('n', 'n', n_order, n_trg, n_order, T{1.0}, &coeffs(0, 0, i_dim), n_order, poly_y.data_handle(),
                   n_order, T{0.0}, tmp.data_handle(), n_order);

        for (int k = 0; k < n_trg; ++k) {
            T pp{0.0};
            for (int i = 0; i < n_order; ++i)
                pp += tmp(i, k) * poly_x(i, k);
            pot(i_dim, k) += pp;
        }
    }
}

template <typename T>
void eval_targets_3d(const ndview<const T, 4> &coeffs, const ndview<const T, 2> &r_trg, const ndview<const T, 1> &cen,
                     T sc, const ndview<T, 2> &pot, sctl::Vector<T> &workspace) {
    const int n_dim = 3;
    const int n_order = coeffs.extent(0);
    const int n_charge_dim = coeffs.extent(3);
    const int n_trg = r_trg.extent(1);

    workspace.ReInit(4 * n_order * n_trg + n_order * n_order * n_trg);
    ndview<T, 2> poly_views[] = {ndview<T, 2>(&workspace[0], n_order, n_trg),
                                 ndview<T, 2>(&workspace[n_order * n_trg], n_order, n_trg),
                                 ndview<T, 2>(&workspace[2 * n_order * n_trg], n_order, n_trg)};

    ndview<T, 3> tmp(&workspace[3 * n_order * n_trg], n_order, n_order, n_trg);
    ndview<T, 2> tmp2(&workspace[3 * n_order * n_trg + n_order * n_order * n_trg], n_order, n_trg);

    for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
        for (int i = 0; i < n_trg; ++i) {
            T x = (r_trg(i_dim, i) - cen(i_dim)) * sc;
            dmk::chebyshev::calc_polynomial(n_order, x, &poly_views[i_dim](0, i));
        }
    }

    auto dot_product = [](int dim, T * a, T * b) {
        using vector_map = Eigen::Map<Eigen::VectorX<T>>;
        return vector_map(a, dim).dot(vector_map(b, dim));
    };

    for (int i_dim = 0; i_dim < n_charge_dim; ++i_dim) {
        // Transform in z
        gemm::gemm('n', 'n', n_order * n_order, n_trg, n_order, T{1.0}, &coeffs(0, 0, 0, i_dim), n_order * n_order,
                   poly_views[2].data_handle(), n_order, T{0.0}, tmp.data_handle(), n_order * n_order);

        for (int k = 0; k < n_trg; ++k) {
            for (int i = 0; i < n_order; ++i) {
                T pp{0.0};
                for (int j = 0; j < n_order; ++j)
                    pp += tmp(j, i, k) * poly_views[0](j, k);
                tmp2(i, k) = pp;
            }
        }
        for (int k = 0; k < n_trg; ++k) {
            T pp{0.0};
            for (int i = 0; i < n_order; ++i)
                pp += tmp2(i, k) * poly_views[1](i, k);
            pot(i_dim, k) += pp;
        }
    }

    const int n_flops_poly = (3 * n_order + 2) * n_trg;
    const int n_flops_mm = 2 * n_order * n_order * n_trg * n_order;
    const int n_flops_tmp = 2 * n_trg * n_order * n_order;
    const int n_flops_pot = 2 * n_trg * n_order;
    const unsigned long n_flops = n_charge_dim * (n_flops_poly + n_flops_mm + n_flops_tmp + n_flops_pot);
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, n_flops);
}

template <typename T, int DIM>
void eval_targets(const ndview<const T, DIM + 1> &coeffs, const ndview<const T, 2> &r_trg,
                  const ndview<const T, 1> &cen, T sc, const ndview<T, 2> &pot, sctl::Vector<T> &workspace) {
    if constexpr (DIM == 2)
        return eval_targets_2d(coeffs, r_trg, cen, sc, pot, workspace);
    else if constexpr (DIM == 3)
        return eval_targets_3d(coeffs, r_trg, cen, sc, pot, workspace);
    else
        static_assert(dmk::util::always_false<T>, "Invalid DIM supplied");
}

template void charge2proxycharge<float, 2>(const ndview<const float, 2> &r_src_, const ndview<const float, 2> &charge_,
                                           const ndview<const float, 1> &center, float scale_factor,
                                           const ndview<float, 3> &coeffs, sctl::Vector<float> &workspace);
template void charge2proxycharge<float, 3>(const ndview<const float, 2> &r_src_, const ndview<const float, 2> &charge_,
                                           const ndview<const float, 1> &center, float scale_factor,
                                           const ndview<float, 4> &coeffs, sctl::Vector<float> &workspace);
template void charge2proxycharge<double, 2>(const ndview<const double, 2> &r_src_,
                                            const ndview<const double, 2> &charge_,
                                            const ndview<const double, 1> &center, double scale_factor,
                                            const ndview<double, 3> &coeffs, sctl::Vector<double> &workspace);

template void charge2proxycharge<double, 3>(const ndview<const double, 2> &r_src_,
                                            const ndview<const double, 2> &charge_,
                                            const ndview<const double, 1> &center, double scale_factor,
                                            const ndview<double, 4> &coeffs, sctl::Vector<double> &workspace);

template void proxycharge2pw(int n_dim, int n_charge_dim, int n_order, int n_pw, const float *proxy_coeffs,
                             const std::complex<float> *poly2pw, std::complex<float> *pw_expansion);
template void proxycharge2pw(int n_dim, int n_charge_dim, int n_order, int n_pw, const double *proxy_coeffs,
                             const std::complex<double> *poly2pw, std::complex<double> *pw_expansion);

template void proxycharge2pw<float, 2>(const ndview<const float, 3> &proxy_coeffs,
                                       const ndview<const std::complex<float>, 2> &poly2pw,
                                       const ndview<std::complex<float>, 3> &pw_expansion,
                                       sctl::Vector<float> &workspace);
template void proxycharge2pw<float, 3>(const ndview<const float, 4> &proxy_coeffs,
                                       const ndview<const std::complex<float>, 2> &poly2pw,
                                       const ndview<std::complex<float>, 4> &pw_expansion,
                                       sctl::Vector<float> &workspace);
template void proxycharge2pw<double, 2>(const ndview<const double, 3> &proxy_coeffs,
                                        const ndview<const std::complex<double>, 2> &poly2pw,
                                        const ndview<std::complex<double>, 3> &pw_expansion,
                                        sctl::Vector<double> &workspace);
template void proxycharge2pw<double, 3>(const ndview<const double, 4> &proxy_coeffs,
                                        const ndview<const std::complex<double>, 2> &poly2pw,
                                        const ndview<std::complex<double>, 4> &pw_expansion,
                                        sctl::Vector<double> &workspace);

template void eval_targets<float, 2>(const ndview<const float, 3> &coeffs, const ndview<const float, 2> &targ,
                                     const ndview<const float, 1> &cen, float sc, const ndview<float, 2> &pot,
                                     sctl::Vector<float> &workspace);

template void eval_targets<float, 3>(const ndview<const float, 4> &coeffs, const ndview<const float, 2> &targ,
                                     const ndview<const float, 1> &cen, float sc, const ndview<float, 2> &pot,
                                     sctl::Vector<float> &workspace);

template void eval_targets<double, 2>(const ndview<const double, 3> &coeffs, const ndview<const double, 2> &targ,
                                      const ndview<const double, 1> &cen, double sc, const ndview<double, 2> &pot,
                                      sctl::Vector<double> &workspace);

template void eval_targets<double, 3>(const ndview<const double, 4> &coeffs, const ndview<const double, 2> &targ,
                                      const ndview<const double, 1> &cen, double sc, const ndview<double, 2> &pot,
                                      sctl::Vector<double> &workspace);

TEST_CASE("[DMK] proxycharge2pw") {
    const int n_charge_dim = 1;
    const int n_pw = 10;
    const int n_pw2 = (n_pw + 1) / 2;
    const int n_pw_coeffs = n_pw * n_pw2;

    for (int n_dim : {2, 3}) {
        CAPTURE(n_dim);
        for (int n_order : {10, 16, 24}) {
            const int n_pw_modes = dmk::util::int_pow(n_pw, n_dim - 1) * n_pw2;
            const int n_pw_coeffs = n_pw_modes * n_charge_dim;
            const int n_proxy_coeffs = dmk::util::int_pow(n_order, n_dim) * n_charge_dim;

            CAPTURE(n_order);
            sctl::Vector<double> proxy_coeffs(n_proxy_coeffs);
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

            sctl::Vector<double> workspace;
            if (n_dim == 2) {
                ndview<const double, 3> proxy_coeffs_view(&proxy_coeffs[0], n_order, n_order, n_charge_dim);
                ndview<const std::complex<double>, 2> poly2pw_view(&poly2pw[0], n_pw, n_order);
                ndview<std::complex<double>, 3> pw_expansion_view(&pw_coeffs[0], n_pw, n_pw2, n_charge_dim);

                proxycharge2pw<double, 2>(proxy_coeffs_view, poly2pw_view, pw_expansion_view, workspace);
            }
            if (n_dim == 3) {
                ndview<const double, 4> proxy_coeffs_view(&proxy_coeffs[0], n_order, n_order, n_order, n_charge_dim);
                ndview<const std::complex<double>, 2> poly2pw_view(&poly2pw[0], n_pw, n_order);
                ndview<std::complex<double>, 4> pw_expansion_view(&pw_coeffs[0], n_pw, n_pw, n_pw2, n_charge_dim);
                proxycharge2pw<double, 3>(proxy_coeffs_view, poly2pw_view, pw_expansion_view, workspace);
            }

            const double rel_err = (pw_coeffs - pw_coeffs_fort).norm() / pw_coeffs.size();
            CHECK(rel_err < std::numeric_limits<double>::epsilon());
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
            using dmk::util::int_pow;
            Eigen::VectorX<double> r_src(n_src * n_dim);
            Eigen::VectorX<double> charge(n_src * n_charge_dim);
            Eigen::VectorX<double> coeffs(int_pow(n_order, n_dim) * n_charge_dim);
            Eigen::VectorX<double> coeffs_fort(int_pow(n_order, n_dim) * n_charge_dim);
            const double center[] = {0.5, 0.5, 0.5};
            const double scale_factor = 1.2;

            for (int i = 0; i < n_src * n_dim; ++i)
                r_src[i] = drand48();

            for (int i = 0; i < n_src * n_charge_dim; ++i)
                charge[i] = drand48() - 0.5;

            coeffs.array() = 0.0;
            sctl::Vector<double> workspace;

            if (n_dim == 2) {
                ndview<double, 3> coeffs_view(coeffs.data(), n_order, n_order, n_charge_dim);
                ndview<const double, 2> src_view(r_src.data(), 2, n_src);
                ndview<const double, 1> center_view(center, n_dim);
                ndview<const double, 2> charge_view(charge.data(), n_charge_dim, n_src);
                dmk::proxy::charge2proxycharge<double, 2>(src_view, charge_view, center_view, scale_factor, coeffs_view,
                                                          workspace);
            }
            if (n_dim == 3) {
                ndview<double, 4> coeffs_view(coeffs.data(), n_order, n_order, n_order, n_charge_dim);
                ndview<const double, 2> src_view(r_src.data(), 3, n_src);
                ndview<const double, 1> center_view(center, n_dim);
                ndview<const double, 2> charge_view(charge.data(), n_charge_dim, n_src);
                dmk::proxy::charge2proxycharge<double, 3>(src_view, charge_view, center_view, scale_factor, coeffs_view,
                                                          workspace);
            }
            coeffs_fort.array() = 0.0;
            pdmk_charge2proxycharge_(&n_dim, &n_charge_dim, &n_order, &n_src, r_src.data(), charge.data(), center,
                                     &scale_factor, coeffs_fort.data());

            const double l2 = (coeffs - coeffs_fort).norm() / coeffs.size();
            CHECK(l2 < std::numeric_limits<double>::epsilon());
        }
    }
}

TEST_CASE("[DMK] eval_targets_3d") {
    const int n_trg = 53;
    const int n_charge_dim = 1;
    const int n_dim = 3;

    for (int n_order : {10, 16, 24}) {
        CAPTURE(n_order);
        using dmk::util::int_pow;
        Eigen::VectorX<double> r_trg(n_trg * n_dim);
        Eigen::VectorX<double> coeffs(int_pow(n_order, n_dim) * n_charge_dim);
        Eigen::VectorX<double> pot(n_charge_dim * n_trg);
        Eigen::VectorX<double> pot_fort(n_charge_dim * n_trg);
        const double center[] = {0.5, 0.5, 0.5};
        const double scale_factor = 1.2;

        for (int i = 0; i < n_trg * n_dim; ++i)
            r_trg[i] = drand48();

        for (auto &coeff : coeffs)
            coeff = (drand48() - 0.5);

        pot.setZero();
        pot_fort.setZero();

        ndview<const double, 4> coeffs_view(coeffs.data(), n_order, n_order, n_order, n_charge_dim);
        ndview<const double, 2> trg_view(r_trg.data(), 3, n_trg);
        ndview<const double, 1> center_view(center, n_dim);
        ndview<double, 2> pot_view(pot.data(), n_charge_dim, n_trg);
        sctl::Vector<double> workspace;
        eval_targets<double, 3>(coeffs_view, trg_view, center_view, scale_factor, pot_view, workspace);

        pdmk_ortho_evalt_nd_(&n_dim, &n_charge_dim, &n_order, coeffs.data(), &n_trg, r_trg.data(), center,
                             &scale_factor, pot_fort.data());

        const double l2 = (pot - pot_fort).norm() / coeffs.size();
        CHECK(l2 < std::numeric_limits<double>::epsilon());
    }
}

} // namespace dmk::proxy
