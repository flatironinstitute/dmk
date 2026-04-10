#include "dmk/util.hpp"
#include <algorithm>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/testing.hpp>
#include <dmk/tree.hpp>
#include <sctl.hpp>

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace PBC direct verification", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 500;
    constexpr double thresh2 = 1e-30;

#ifdef DMK_HAVE_MPI
    auto sctl_comm = sctl::Comm(test_comm);
#else
    auto sctl_comm = sctl::Comm::Self();
#endif

    std::default_random_engine eng(42);
    std::uniform_real_distribution<double> rng(0.01, 0.99);

    sctl::Vector<double> r_src(n_dim * n_src), r_trg(n_dim * n_trg);
    sctl::Vector<double> charges(n_src);

    for (int i = 0; i < n_src * n_dim; ++i)
        r_src[i] = rng(eng);
    for (int i = 0; i < n_trg * n_dim; ++i)
        r_trg[i] = rng(eng);
    for (int i = 0; i < n_src; ++i)
        charges[i] = rng(eng) - 0.5;
    {
        double sum = 0.0;
        for (int i = 0; i < n_src; ++i)
            sum += charges[i];
        for (int i = 0; i < n_src; ++i)
            charges[i] -= sum / n_src;
    }

    using dmk::util::calc_bandlimiting;
    const auto coeffs_3 = dmk::get_local_correction_coeffs<double>(
        DMK_LAPLACE, 3, 3, calc_bandlimiting({.n_dim = 3, .eps = 1e-3, .kernel = DMK_LAPLACE, .debug_flags = 0}));
    const auto coeffs_6 = dmk::get_local_correction_coeffs<double>(
        DMK_LAPLACE, 3, 6, calc_bandlimiting({.n_dim = 3, .eps = 1e-6, .kernel = DMK_LAPLACE, .debug_flags = 0}));
    const auto coeffs_9 = dmk::get_local_correction_coeffs<double>(
        DMK_LAPLACE, 3, 9, calc_bandlimiting({.n_dim = 3, .eps = 1e-9, .kernel = DMK_LAPLACE, .debug_flags = 0}));
    const auto coeffs_12 = dmk::get_local_correction_coeffs<double>(
        DMK_LAPLACE, 3, 12, calc_bandlimiting({.n_dim = 3, .eps = 1e-12, .kernel = DMK_LAPLACE, .debug_flags = 0}));

    struct PrecisionCase {
        int n_digits;
        double eps;
        const double *coeffs;
        size_t n_coeffs;
    };
    const PrecisionCase cases[] = {
        {3, 1e-3, coeffs_3.data(), coeffs_3.size()},
        {6, 1e-6, coeffs_6.data(), coeffs_6.size()},
        {9, 1e-9, coeffs_9.data(), coeffs_9.size()},
        {12, 1e-12, coeffs_12.data(), coeffs_12.size()},
    };

    auto horner_eval = [](double x, const double *c, int n) {
        double val = c[n - 1];
        for (int i = n - 2; i >= 0; --i)
            val = val * x + c[i];
        return val;
    };

    for (const auto &pc : cases) {
        SUBCASE(("n_digits=" + std::to_string(pc.n_digits)).c_str()) {
            pdmk_params params;
            params.eps = pc.eps;
            params.n_dim = n_dim;
            params.n_per_leaf = 280;
            params.pgh_src = DMK_POTENTIAL;
            params.pgh_trg = DMK_POTENTIAL;
            params.kernel = DMK_LAPLACE;
            params.use_periodic = true;
            params.log_level = 6;

            dmk::DMKPtTree<double, n_dim> tree(sctl_comm, params, r_src, r_trg, charges);

            tree.pot_src_sorted.SetZero();
            tree.pot_trg_sorted.SetZero();
            tree.evaluate_direct_interactions();

            const auto &node_mid = tree.GetNodeMID();
            const auto &node_attr = tree.GetNodeAttr();

            struct SourceInfo {
                double r[3];
                double charge;
                int level;
            };
            std::vector<SourceInfo> sources;
            for (int box = 0; box < tree.n_boxes(); ++box) {
                if (!node_attr[box].Leaf || node_attr[box].Ghost)
                    continue;
                const int n = tree.r_src_cnt_owned[box];
                if (!n)
                    continue;
                const int level = node_mid[box].Depth();
                const double *rp = tree.r_src_owned_ptr(box);
                const double *cp = tree.charge_owned_ptr(box);
                for (int i = 0; i < n; ++i)
                    sources.push_back({{rp[i * 3], rp[i * 3 + 1], rp[i * 3 + 2]}, cp[i], level});
            }

            struct TargetInfo {
                double r[3];
                int pot_offset;
            };
            std::vector<TargetInfo> targets;
            for (int box = 0; box < tree.n_boxes(); ++box) {
                if (node_attr[box].Ghost)
                    continue;
                const int n = tree.r_trg_cnt_owned[box];
                if (!n)
                    continue;
                const double *rp = tree.r_trg_owned_ptr(box);
                for (int i = 0; i < n; ++i)
                    targets.push_back({{rp[i * 3], rp[i * 3 + 1], rp[i * 3 + 2]}, (int)tree.pot_trg_offsets[box] + i});
            }

            std::vector<double> ref_pot(targets.size(), 0.0);
            for (int i_trg = 0; i_trg < (int)targets.size(); ++i_trg) {
                const auto &trg = targets[i_trg];
                for (const auto &src : sources) {
                    const double bsize = tree.boxsize[src.level];
                    const double d2max = bsize * bsize;
                    const double rsc = 2.0 / bsize;
                    const double cen = -bsize / 2.0;
                    for (int mx = -1; mx <= 1; ++mx)
                        for (int my = -1; my <= 1; ++my)
                            for (int mz = -1; mz <= 1; ++mz) {
                                double dx = trg.r[0] - (src.r[0] + mx);
                                double dy = trg.r[1] - (src.r[1] + my);
                                double dz = trg.r[2] - (src.r[2] + mz);
                                double r2 = dx * dx + dy * dy + dz * dz;
                                if (r2 < thresh2 || r2 >= d2max)
                                    continue;
                                const double r = std::sqrt(r2);
                                const double x = (r + cen) * rsc;
                                ref_pot[i_trg] += src.charge * horner_eval(x, pc.coeffs, pc.n_coeffs) / r;
                            }
                }
            }

            const int n_test = std::min((int)targets.size(), 200);
            double err2 = 0.0, ref2 = 0.0;
            for (int i = 0; i < n_test; ++i) {
                const double tree_val = tree.pot_trg_sorted[targets[i].pot_offset];
                err2 += sctl::pow<2>(tree_val - ref_pot[i]);
                ref2 += sctl::pow<2>(ref_pot[i]);
            }
            const double l2_err = (ref2 > 0) ? std::sqrt(err2 / ref2) : std::sqrt(err2);
            MESSAGE("n_digits=", pc.n_digits, " eps=", pc.eps, " l2_err=", l2_err, " n_levels=", tree.n_levels(),
                    " n_boxes=", tree.n_boxes());
            const double tol = (pc.n_digits <= 3) ? 1e-3 : 1e-6;
            CHECK(l2_err < tol);
        }
    }
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace PBC single-level public API", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 8;
    constexpr int n_trg = 4;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    sctl::Vector<double> r_src({0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4,
                                0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9});
    sctl::Vector<double> r_trg({0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.65, 0.65, 0.65, 0.85, 0.85, 0.85});
    sctl::Vector<double> charges(n_src);
    sctl::Vector<double> normal(n_src * n_dim);
    sctl::Vector<double> dipstr(n_src);
    sctl::Vector<double> pot_src(n_src);
    sctl::Vector<double> pot_trg(n_trg);

    normal.SetZero();
    dipstr.SetZero();
    for (int i = 0; i < n_src; ++i)
        charges[i] = (i % 2 == 0 ? 1.0 : -1.0) / n_src;

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 1000000;
    params.pgh_src = DMK_POTENTIAL;
    params.pgh_trg = DMK_POTENTIAL;
    params.kernel = DMK_LAPLACE;
    params.use_periodic = true;
    params.log_level = 6;

    pdmk_tree tree =
        pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &normal[0], &dipstr[0], n_trg, &r_trg[0]);
    pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
    pdmk_tree_destroy(tree);

    CHECK(pot_src.Dim() == n_src);
    CHECK(pot_trg.Dim() == n_trg);
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace PBC full pipeline vs Ewald", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 500;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    std::default_random_engine eng(99);
    std::uniform_real_distribution<double> rng(0.01, 0.99);

    sctl::Vector<double> r_src(n_dim * n_src), r_trg(n_dim * n_trg);
    sctl::Vector<double> charges(n_src);
    sctl::Vector<double> rnormal(n_dim * n_src);
    sctl::Vector<double> dipstr(n_src);
    rnormal.SetZero();
    dipstr.SetZero();

    for (int i = 0; i < n_src * n_dim; ++i)
        r_src[i] = rng(eng);
    for (int i = 0; i < n_trg * n_dim; ++i)
        r_trg[i] = rng(eng);
    for (int i = 0; i < n_src; ++i)
        charges[i] = rng(eng) - 0.5;
    {
        double sum = 0.0;
        for (int i = 0; i < n_src; ++i)
            sum += charges[i];
        for (int i = 0; i < n_src; ++i)
            charges[i] -= sum / n_src;
    }

    const double L = 1.0;
    const double V_box = L * L * L;
    const double dk = 2.0 * M_PI / L;
    const double alpha = 10.0;
    const double r_c = 0.5 * L;
    const int n_ewald = 15;
    const int d = 2 * n_ewald + 1;

    std::vector<std::complex<double>> rho(d * d * d, {0.0, 0.0});
    for (int is = 0; is < n_src; ++is) {
        const std::complex<double> ex0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 3 + 0]));
        const std::complex<double> ey0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 3 + 1]));
        const std::complex<double> ez0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 3 + 2]));
        std::vector<std::complex<double>> ex(d), ey(d), ez(d);
        for (int a = -n_ewald; a <= n_ewald; ++a) {
            ex[a + n_ewald] = std::pow(ex0, a);
            ey[a + n_ewald] = std::pow(ey0, a);
            ez[a + n_ewald] = std::pow(ez0, a);
        }
        for (int ix = 0; ix < d; ++ix)
            for (int iy = 0; iy < d; ++iy) {
                const auto t2 = charges[is] * ex[ix] * ey[iy];
                for (int iz = 0; iz < d; ++iz)
                    rho[ix * d * d + iy * d + iz] += t2 * ez[iz];
            }
    }

    auto ewald_pot_grad = [&](const double *r_eval, double &pot_out, double *grad_out) {
        pot_out = 0.0;
        if (grad_out)
            grad_out[0] = grad_out[1] = grad_out[2] = 0.0;

        for (int is = 0; is < n_src; ++is) {
            for (int mx = -1; mx <= 1; ++mx)
                for (int my = -1; my <= 1; ++my)
                    for (int mz = -1; mz <= 1; ++mz) {
                        const double dx = r_eval[0] - r_src[is * 3 + 0] - mx * L;
                        const double dy = r_eval[1] - r_src[is * 3 + 1] - my * L;
                        const double dz = r_eval[2] - r_src[is * 3 + 2] - mz * L;
                        const double r2 = dx * dx + dy * dy + dz * dz;
                        const double r = std::sqrt(r2);
                        if (r > 1e-15 && r <= r_c) {
                            pot_out += charges[is] * std::erfc(alpha * r) / r;
                            if (grad_out) {
                                const double scale = -charges[is] * (std::erfc(alpha * r) / (r * r2) +
                                                                     2.0 * alpha * std::exp(-alpha * alpha * r2) /
                                                                         (std::sqrt(M_PI) * r2));
                                grad_out[0] += scale * dx;
                                grad_out[1] += scale * dy;
                                grad_out[2] += scale * dz;
                            }
                        }
                    }
        }

        const std::complex<double> etx0 = std::exp(std::complex<double>(0.0, dk * r_eval[0]));
        const std::complex<double> ety0 = std::exp(std::complex<double>(0.0, dk * r_eval[1]));
        const std::complex<double> etz0 = std::exp(std::complex<double>(0.0, dk * r_eval[2]));
        std::vector<std::complex<double>> etx(d), ety(d), etz(d);
        for (int a = -n_ewald; a <= n_ewald; ++a) {
            etx[a + n_ewald] = std::pow(etx0, a);
            ety[a + n_ewald] = std::pow(ety0, a);
            etz[a + n_ewald] = std::pow(etz0, a);
        }

        double pot_long = 0.0;
        double grad_long[3] = {0, 0, 0};
        for (int nx = -n_ewald; nx <= n_ewald; ++nx)
            for (int ny = -n_ewald; ny <= n_ewald; ++ny)
                for (int nz = -n_ewald; nz <= n_ewald; ++nz) {
                    if (nx == 0 && ny == 0 && nz == 0)
                        continue;
                    const double kx = dk * nx;
                    const double ky = dk * ny;
                    const double kz = dk * nz;
                    const double k2 = kx * kx + ky * ky + kz * kz;
                    const double G = std::exp(-k2 / (4.0 * alpha * alpha)) / k2;
                    const int ix = nx + n_ewald;
                    const int iy = ny + n_ewald;
                    const int iz = nz + n_ewald;
                    const auto &rho_k = rho[ix * d * d + iy * d + iz];
                    const auto eikr = etx[nx + n_ewald] * ety[ny + n_ewald] * etz[nz + n_ewald];
                    const auto rho_eikr = rho_k * eikr;
                    pot_long += G * std::real(rho_eikr);
                    if (grad_out) {
                        const double im = -std::imag(rho_eikr);
                        grad_long[0] += G * kx * im;
                        grad_long[1] += G * ky * im;
                        grad_long[2] += G * kz * im;
                    }
                }
        pot_out += (4.0 * M_PI / V_box) * pot_long;
        if (grad_out) {
            grad_out[0] += (4.0 * M_PI / V_box) * grad_long[0];
            grad_out[1] += (4.0 * M_PI / V_box) * grad_long[1];
            grad_out[2] += (4.0 * M_PI / V_box) * grad_long[2];
        }
    };

    const double ewald_self_factor = 2.0 * alpha / std::sqrt(M_PI);

    struct PrecisionCase {
        int n_digits;
        double eps;
        double tol_pot;
        double tol_grad;
    };
    const PrecisionCase cases[] = {
        {3, 1e-3, 1e-2, 1e-1},
        {6, 1e-6, 1e-4, 1e-3},
        {9, 1e-9, 1e-7, 1e-6},
        {12, 1e-12, 1e-10, 1e-9},
    };

    for (const auto &pc : cases) {
        for (int with_grad = 0; with_grad <= 1; ++with_grad) {
            const auto pgh = with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
            const int odim = with_grad ? 1 + n_dim : 1;
            const std::string label = "n_digits=" + std::to_string(pc.n_digits) + (with_grad ? " pot+grad" : " pot");

            SUBCASE(label.c_str()) {
                pdmk_params params;
                params.eps = pc.eps;
                params.n_dim = n_dim;
                params.n_per_leaf = 50;
                params.pgh_src = pgh;
                params.pgh_trg = pgh;
                params.kernel = DMK_LAPLACE;
                params.use_periodic = true;
                params.log_level = 6;

                sctl::Vector<double> pot_src(n_src * odim), pot_trg(n_trg * odim);

                pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], &dipstr[0],
                                                  n_trg, &r_trg[0]);
                pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
                pdmk_tree_destroy(tree);

                const int n_test = std::min(n_src, 100);
                double err2_pot_src = 0, ref2_pot_src = 0;
                double err2_grad_src = 0, ref2_grad_src = 0;
                for (int i = 0; i < n_test; ++i) {
                    double ewald_pot;
                    double ewald_grad[3];
                    ewald_pot_grad(&r_src[i * n_dim], ewald_pot, with_grad ? ewald_grad : nullptr);
                    ewald_pot -= charges[i] * ewald_self_factor;
                    err2_pot_src += sctl::pow<2>(pot_src[i * odim] - ewald_pot);
                    ref2_pot_src += sctl::pow<2>(ewald_pot);
                    if (with_grad) {
                        for (int dd = 0; dd < n_dim; ++dd) {
                            err2_grad_src += sctl::pow<2>(pot_src[i * odim + 1 + dd] - ewald_grad[dd]);
                            ref2_grad_src += sctl::pow<2>(ewald_grad[dd]);
                        }
                    }
                }

                const int n_test_trg = std::min(n_trg, 100);
                double err2_pot_trg = 0, ref2_pot_trg = 0;
                double err2_grad_trg = 0, ref2_grad_trg = 0;
                for (int i = 0; i < n_test_trg; ++i) {
                    double ewald_pot;
                    double ewald_grad[3];
                    ewald_pot_grad(&r_trg[i * n_dim], ewald_pot, with_grad ? ewald_grad : nullptr);
                    err2_pot_trg += sctl::pow<2>(pot_trg[i * odim] - ewald_pot);
                    ref2_pot_trg += sctl::pow<2>(ewald_pot);
                    if (with_grad) {
                        for (int dd = 0; dd < n_dim; ++dd) {
                            err2_grad_trg += sctl::pow<2>(pot_trg[i * odim + 1 + dd] - ewald_grad[dd]);
                            ref2_grad_trg += sctl::pow<2>(ewald_grad[dd]);
                        }
                    }
                }

                auto safe_l2 = [](double e2, double r2) { return r2 > 0 ? std::sqrt(e2 / r2) : std::sqrt(e2); };
                const double l2_pot_src = safe_l2(err2_pot_src, ref2_pot_src);
                const double l2_pot_trg = safe_l2(err2_pot_trg, ref2_pot_trg);

                MESSAGE("PBC pipeline: ", label, " pot_src=", l2_pot_src, " pot_trg=", l2_pot_trg);
                CHECK(l2_pot_src < pc.tol_pot);
                CHECK(l2_pot_trg < pc.tol_pot);

                if (with_grad) {
                    const double l2_grad_src = safe_l2(err2_grad_src, ref2_grad_src);
                    const double l2_grad_trg = safe_l2(err2_grad_trg, ref2_grad_trg);
                    MESSAGE("  grad_src=", l2_grad_src, " grad_trg=", l2_grad_trg);
                    CHECK(l2_grad_src < pc.tol_grad);
                    CHECK(l2_grad_trg < pc.tol_grad);
                }
            }
        }
    }
}
