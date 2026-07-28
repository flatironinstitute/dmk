// Comparison tests against the Fortran reference implementation (src/pdmk).
// Kept in its own translation unit (compiled into test_all only when
// DMK_BUILD_REFERENCE=ON) so the core library never references Fortran symbols
// and ordinary consumers of libdmk do not need to link dmk_ref.

#ifdef DMK_HAVE_REFERENCE

#include <doctest/doctest.h>

#include <dmk/fortran.h>
#include <dmk/planewave.hpp>
#include <dmk/proxy.hpp>
#include <dmk/types.hpp>
#include <dmk/util.hpp>

#include <nda/nda.hpp>
#include <sctl.hpp>

#include <complex>
#include <cstdlib>
#include <limits>

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
            nda::vector<double> proxy_coeffs(n_proxy_terms), proxy_coeffs_fort(n_proxy_terms);

            for (auto &elem : pw_expansion)
                elem = std::complex<double>{rand() / double(RAND_MAX), rand() / double(RAND_MAX)};
            for (auto &elem : pw_to_coefs_mat)
                elem = std::complex<double>{rand() / double(RAND_MAX), rand() / double(RAND_MAX)};

            proxy_coeffs = 0;
            proxy_coeffs_fort = 0;
            sctl::Vector<double> workspace;

            if (n_dim == 2) {
                dmk::ndview<std::complex<double>, 3> pw_expansion_view({n_pw, n_pw2, n_charge_dim}, &pw_expansion[0]);
                dmk::ndview<std::complex<double>, 2> pw_to_coefs_mat_view({n_pw, n_order}, &pw_to_coefs_mat[0]);
                dmk::ndview<double, 3> proxy_coeffs_view({n_order, n_order, n_charge_dim}, &proxy_coeffs[0]);

                dmk::planewave_to_proxy_potential<double, 2>(pw_expansion_view, pw_to_coefs_mat_view, proxy_coeffs_view,
                                                             workspace);
            }

            if (n_dim == 3) {
                dmk::ndview<std::complex<double>, 4> pw_expansion_view({n_pw, n_pw, n_pw2, n_charge_dim},
                                                                       &pw_expansion[0]);
                dmk::ndview<std::complex<double>, 2> pw_to_coefs_mat_view({n_pw, n_order}, &pw_to_coefs_mat[0]);
                dmk::ndview<double, 4> proxy_coeffs_view({n_order, n_order, n_order, n_charge_dim}, &proxy_coeffs[0]);

                dmk::planewave_to_proxy_potential<double, 3>(pw_expansion_view, pw_to_coefs_mat_view, proxy_coeffs_view,
                                                             workspace);
            }

            dmk_pw2proxypot_(&n_dim, &n_charge_dim, &n_order, &n_pw, (double *)&pw_expansion[0],
                             (double *)&pw_to_coefs_mat[0], &proxy_coeffs_fort[0]);

            const double l2 = nda::linalg::norm(proxy_coeffs - proxy_coeffs_fort) / proxy_coeffs.size();
            CHECK(l2 < std::numeric_limits<double>::epsilon());
        }
    }
}

namespace dmk::proxy {
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
            nda::vector<std::complex<double>> pw_coeffs(n_pw_coeffs), pw_coeffs_fort(n_pw_coeffs);

            dmk::calc_planewave_coeff_matrices(1.0, 1.0, n_pw, n_order, poly2pw, pw2poly);

            for (auto &c : proxy_coeffs)
                c = drand48();

            pw_coeffs = 0.0;
            proxycharge2pw(n_dim, n_charge_dim, n_order, n_pw, &proxy_coeffs[0], &poly2pw[0], &pw_coeffs[0]);

            pw_coeffs_fort = 0.0;
            dmk_proxycharge2pw_(&n_dim, &n_charge_dim, &n_order, &proxy_coeffs[0], &n_pw, (double *)&poly2pw[0],
                                (double *)&pw_coeffs_fort[0]);

            const double l2 = nda::linalg::norm(pw_coeffs - pw_coeffs_fort) / pw_coeffs.size();
            CHECK(l2 < std::numeric_limits<double>::epsilon());

            sctl::Vector<double> workspace;
            if (n_dim == 2) {
                const ndview<double, 3> proxy_coeffs_view({n_order, n_order, n_charge_dim}, &proxy_coeffs[0]);
                const ndview<std::complex<double>, 2> poly2pw_view({n_pw, n_order}, &poly2pw[0]);
                ndview<std::complex<double>, 3> pw_expansion_view({n_pw, n_pw2, n_charge_dim}, &pw_coeffs[0]);

                proxycharge2pw<double, 2>(proxy_coeffs_view, poly2pw_view, pw_expansion_view, workspace);
            }
            if (n_dim == 3) {
                const ndview<double, 4> proxy_coeffs_view({n_order, n_order, n_order, n_charge_dim}, &proxy_coeffs[0]);
                const ndview<std::complex<double>, 2> poly2pw_view({n_pw, n_order}, &poly2pw[0]);
                ndview<std::complex<double>, 4> pw_expansion_view({n_pw, n_pw, n_pw2, n_charge_dim}, &pw_coeffs[0]);
                proxycharge2pw<double, 3>(proxy_coeffs_view, poly2pw_view, pw_expansion_view, workspace);
            }

            const double rel_err = nda::linalg::norm(pw_coeffs - pw_coeffs_fort) / pw_coeffs.size();
            CHECK(rel_err < std::numeric_limits<double>::epsilon());
        }
    }
}

TEST_CASE("[DMK] charge2proxycharge") {
    const int n_src = 500;
    const int n_charge_dim = 2;

    for (int n_dim : {2, 3}) {
        CAPTURE(n_dim);
        for (int n_order : {9, 18, 28, 38}) {
            CAPTURE(n_order);
            using dmk::util::int_pow;
            nda::vector<double> r_src(n_src * n_dim);
            nda::vector<double> charge(n_src * n_charge_dim);
            nda::vector<double> coeffs(int_pow(n_order, n_dim) * n_charge_dim);
            nda::vector<double> coeffs_fort(int_pow(n_order, n_dim) * n_charge_dim);
            const double center[] = {0.5, 0.5, 0.5};
            const double scale_factor = 1.2;

            for (int i = 0; i < n_src * n_dim; ++i)
                r_src[i] = drand48();

            for (int i = 0; i < n_src * n_charge_dim; ++i)
                charge[i] = drand48() - 0.5;

            coeffs = 0.0;
            sctl::Vector<double> workspace;

            if (n_dim == 2) {
                ndview<double, 3> coeffs_view({n_order, n_order, n_charge_dim}, coeffs.data());
                ndview<const double, 2> src_view({2, n_src}, r_src.data());
                ndview<const double, 1> center_view({n_dim}, center);
                ndview<const double, 2> charge_view({n_charge_dim, n_src}, charge.data());
                dmk::proxy::charge2proxycharge<double, 2>(src_view, charge_view, center_view, scale_factor, coeffs_view,
                                                          workspace);
            }
            if (n_dim == 3) {
                ndview<double, 4> coeffs_view({n_order, n_order, n_order, n_charge_dim}, coeffs.data());
                ndview<const double, 2> src_view({3, n_src}, r_src.data());
                ndview<const double, 1> center_view({n_dim}, center);
                ndview<const double, 2> charge_view({n_charge_dim, n_src}, charge.data());
                dmk::proxy::charge2proxycharge<double, 3>(src_view, charge_view, center_view, scale_factor, coeffs_view,
                                                          workspace);
            }
            coeffs_fort = 0.0;
            pdmk_charge2proxycharge_(&n_dim, &n_charge_dim, &n_order, &n_src, r_src.data(), charge.data(), center,
                                     &scale_factor, coeffs_fort.data());

            const double l2 = nda::linalg::norm(coeffs - coeffs_fort) / coeffs.size();
            CHECK(l2 < std::numeric_limits<double>::epsilon());
        }
    }
}

TEST_CASE("[DMK] eval_targets_3d") {
    const int n_trg = 53;
    const int n_charge_dim = 1;
    const int n_dim = 3;

    for (int n_order : {9, 18, 28, 38}) {
        CAPTURE(n_order);
        using dmk::util::int_pow;
        nda::vector<double> r_trg(n_trg * n_dim);
        nda::vector<double> coeffs(int_pow(n_order, n_dim) * n_charge_dim);
        nda::vector<double> pot(n_charge_dim * n_trg);
        nda::vector<double> pot_fort(n_charge_dim * n_trg);
        const double center[] = {0.5, 0.5, 0.5};
        const double scale_factor = 1.2;

        for (int i = 0; i < n_trg * n_dim; ++i)
            r_trg[i] = drand48();

        for (auto &coeff : coeffs)
            coeff = (drand48() - 0.5);

        pot = 0.0;
        pot_fort = 0.0;

        ndview<double, 4> coeffs_view({n_order, n_order, n_order, n_charge_dim}, coeffs.data());
        ndview<double, 2> trg_view({3, n_trg}, r_trg.data());
        ndview<double, 1> center_view({n_dim}, const_cast<double *>(center));
        ndview<double, 2> pot_view({n_charge_dim, n_trg}, pot.data());
        sctl::Vector<double> workspace;
        eval_targets<double, 3, 1>(coeffs_view, trg_view, center_view, scale_factor, pot_view, workspace);

        pdmk_ortho_evalt_nd_(&n_dim, &n_charge_dim, &n_order, coeffs.data(), &n_trg, r_trg.data(), center,
                             &scale_factor, pot_fort.data());

        const double l2 = nda::linalg::norm(pot - pot_fort) / coeffs.size();
        CHECK(l2 < std::numeric_limits<double>::epsilon());
    }
}
} // namespace dmk::proxy

#endif // DMK_HAVE_REFERENCE
