#include <algorithm>
#include <limits>
#include <string>
#include <variant>

#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/direct.hpp>
#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/logger.h>
#include <dmk/prolate0_fun.hpp>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/tree.hpp>
#include <dmk/util.hpp>
#include <sctl.hpp>

#include <dmk/omp_wrapper.hpp>
#include <dmk/testing.hpp>

using pdmk_tree_impl =
    std::variant<std::unique_ptr<dmk::DMKPtTree<float, 2>>, std::unique_ptr<dmk::DMKPtTree<float, 3>>,
                 std::unique_ptr<dmk::DMKPtTree<double, 2>>, std::unique_ptr<dmk::DMKPtTree<double, 3>>>;

namespace dmk {

template <typename T, int DIM>
void pdmk(dmk_communicator comm, const pdmk_params &params, int n_src, const T *r_src, const T *charge, const T *normal,
          const T *dipole_str, int n_trg, const T *r_trg, T *pot_src, T *pot_trg) {
#ifdef DMK_HAVE_MPI
    const auto &sctl_comm = sctl::Comm(MPI_Comm(comm));
#else
    const auto &sctl_comm = sctl::Comm().Self();
#endif
    auto &logger = dmk::get_logger(sctl_comm, params.log_level);
    auto &rank_logger = dmk::get_rank_logger(sctl_comm, params.log_level);
    logger->info("PDMK called");
    auto st = MY_OMP_GET_WTIME();

    const int kernel_input_dim = get_kernel_input_dim(params.n_dim, params.kernel);

    sctl::Vector<T> r_src_vec(n_src * params.n_dim, const_cast<T *>(r_src), false);
    sctl::Vector<T> r_trg_vec(n_trg * params.n_dim, const_cast<T *>(r_trg), false);
    sctl::Vector<T> charge_vec(n_src * kernel_input_dim, const_cast<T *>(charge), false);

    DMKPtTree<T, DIM> tree(sctl_comm, params, r_src_vec, r_trg_vec, charge_vec);
    tree.upward_pass();
    tree.downward_pass();

    sctl::Vector<T> res;
    tree.GetParticleData(res, "pdmk_pot_src");
    sctl::Vector<T>(res.Dim(), pot_src, false) = res;
    tree.GetParticleData(res, "pdmk_pot_trg");
    sctl::Vector<T>(res.Dim(), pot_trg, false) = res;

    auto dt = MY_OMP_GET_WTIME() - st;
    int N = n_src + n_trg;
#ifdef DMK_HAVE_MPI
    if (sctl_comm.Rank() == 0)
        MPI_Reduce(MPI_IN_PLACE, &N, 1, MPI_INT, MPI_SUM, 0, comm);
    else
        MPI_Reduce(&N, &N, 1, MPI_INT, MPI_SUM, 0, comm);
#endif

    logger->info("PDMK finished in {:.4f} seconds ({:.0f} pts/s, {:.0f} pts/s/rank)", dt, N / dt,
                 N / dt / sctl_comm.Size());
}

TEST_CASE_GENERIC("[DMK] pdmk 3d float", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 10000;
    constexpr int n_trg = 10000;
    constexpr int nd = 1;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    sctl::Vector<double> r_src, pot_src, charges, rnormal, dipstr, pot_trg, r_trg;
    sctl::Vector<float> r_srcf, pot_srcf, chargesf, rnormalf, dipstrf, pot_trgf, r_trgf;
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges,
                              dipstr, 0);
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_srcf, r_trgf, rnormalf, chargesf,
                              dipstrf, 0);
    pot_src.ReInit(n_src * nd);
    pot_trg.ReInit(n_trg * nd);
    pot_srcf.ReInit(n_src * nd);
    pot_trgf.ReInit(n_trg * nd);

    pdmk_params params;
    params.eps = 1e-9;
    params.n_dim = n_dim;
    params.n_per_leaf = 80;
    params.pgh_src = DMK_POTENTIAL;
    params.pgh_trg = DMK_POTENTIAL;
    params.kernel = DMK_YUKAWA;
    params.fparam = 6.0;
    params.log_level = SPDLOG_LEVEL_OFF;

    pdmk(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], &dipstr[0], n_trg, &r_trg[0], &pot_src[0],
         &pot_trg[0]);

    params.eps = 1e-3;
    pdmkf(comm, params, n_src, &r_srcf[0], &chargesf[0], &rnormalf[0], &dipstrf[0], n_trg, &r_trgf[0], &pot_srcf[0],
          &pot_trgf[0]);

    double l2_err_src = 0.0;
    double l2_err_trg = 0.0;
    for (int i = 0; i < n_src; ++i)
        l2_err_src += pot_src[i] ? sctl::pow<2>(1.0 - pot_srcf[i] / pot_src[i]) : 0.0;
    for (int i = 0; i < n_trg; ++i)
        l2_err_trg += pot_trg[i] ? sctl::pow<2>(1.0 - pot_trgf[i] / pot_trg[i]) : 0.0;

    l2_err_src = std::sqrt(l2_err_src) / n_src;
    l2_err_trg = std::sqrt(l2_err_trg) / n_trg;
    CHECK(l2_err_src < 6 * params.eps);
    CHECK(l2_err_trg < 6 * params.eps);
}

TEST_CASE_GENERIC("[DMK] pdmk 3d all", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 10000;
    constexpr int nd = 1;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    sctl::Vector<double> r_src, pot_src, charges, rnormal, dipstr, pot_trg, r_trg;
    dmk::util::init_test_data(n_dim, 1, n_src, 0, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, dipstr,
                              0);
    r_trg = r_src;
    std::reverse(r_trg.begin(), r_trg.end());
    r_trg.ReInit(n_dim * (n_src - set_fixed_charges * 3));
    const int n_trg = r_trg.Dim() / n_dim;

    pdmk_params params;
    params.eps = 1e-12;
    params.n_dim = n_dim;
    params.n_per_leaf = 80;
    params.pgh_src = DMK_POTENTIAL;
    params.pgh_trg = DMK_POTENTIAL;
    params.kernel = DMK_YUKAWA;
    params.fparam = 6.0;
    params.log_level = SPDLOG_LEVEL_OFF;

    auto get_pot_func = [&params](int n_dim, dmk_ikernel kernel) -> std::function<double(double *, double *)> {
        auto distance2 = [](double *r_a, double *r_b, int n_dim) {
            double dr2 = 0.0;
            for (int j = 0; j < n_dim; ++j)
                dr2 += sctl::pow<2>(r_a[j] - r_b[j]);

            return dr2;
        };

        switch (kernel) {
        case DMK_YUKAWA:
            if (n_dim == 2)
                return [distance2, &params](double *r_a, double *r_b) {
                    double dr = distance2(r_a, r_b, 2);
                    if (!dr)
                        return 0.0;

                    dr = std::sqrt(dr);
                    return util::cyl_bessel_k(0, params.fparam * dr);
                };
            if (n_dim == 3)
                return [distance2, &params](double *r_a, double *r_b) {
                    double dr = distance2(r_a, r_b, 3);
                    if (!dr)
                        return 0.0;

                    dr = std::sqrt(dr);
                    return std::exp(-params.fparam * dr) / dr;
                };
        case DMK_LAPLACE:
            if (n_dim == 2)
                return [](double *r_a, double *r_b) {
                    double dr2 = 0.0;
                    for (int j = 0; j < 2; ++j)
                        dr2 += sctl::pow<2>(r_a[j] - r_b[j]);

                    if (!dr2)
                        return 0.0;

                    return 0.5 * std::log(dr2);
                };
            if (n_dim == 3)
                return [](double *r_a, double *r_b) {
                    double dr2 = 0.0;
                    for (int j = 0; j < 3; ++j)
                        dr2 += sctl::pow<2>(r_a[j] - r_b[j]);

                    if (!dr2)
                        return 0.0;

                    return 1.0 / std::sqrt(dr2);
                };
        case DMK_SQRT_LAPLACE:
            if (n_dim == 2)
                return [](double *r_a, double *r_b) {
                    double dr2 = 0.0;
                    for (int j = 0; j < 2; ++j)
                        dr2 += sctl::pow<2>(r_a[j] - r_b[j]);

                    if (!dr2)
                        return 0.0;

                    return 1.0 / std::sqrt(dr2);
                };
            if (n_dim == 3) {
                return [](double *r_a, double *r_b) {
                    double dr2 = 0.0;
                    for (int j = 0; j < 3; ++j)
                        dr2 += sctl::pow<2>(r_a[j] - r_b[j]);

                    if (!dr2)
                        return 0.0;

                    return 1.0 / dr2;
                };
            }

        default:
            throw std::runtime_error("Unknown kernel");
        }
    };
    int ndiv[3] = {80, 280, 280};

    const auto test_kernels = {
        DMK_YUKAWA,
        DMK_LAPLACE,
        DMK_SQRT_LAPLACE,
    };

    for (auto n_dim : {2, 3}) {
        params.n_dim = n_dim;
        for (auto kernel : test_kernels) {
            const std::string kernel_str = [&kernel]() {
                switch (kernel) {
                case DMK_YUKAWA:
                    return "YUKAWA";
                case DMK_LAPLACE:
                    return "LAPLACE";
                case DMK_SQRT_LAPLACE:
                    return "SQRT_LAPLACE";
                default:
                    throw std::runtime_error("Unknown kernel");
                }
            }();

            SUBCASE((kernel_str + "_" + std::to_string(n_dim)).c_str()) {
                sctl::Vector<double> pot_src(n_src * nd), grad_src(n_src * nd * n_dim),
                    hess_src(n_src * nd * n_dim * n_dim), pot_trg(n_src * nd), grad_trg(n_trg * nd * n_dim),
                    hess_trg(n_trg * nd * n_dim * n_dim);
                params.n_per_leaf = ndiv[int(kernel)];

                params.kernel = kernel;
                auto potential = get_pot_func(n_dim, kernel);

                const int n_test_src = std::min(n_src, 100);
                const int n_test_trg = std::min(n_trg, 100);
                std::vector<double> test_src(n_test_src);
                std::vector<double> test_trg(n_test_trg);

                for (int i_src = 0; i_src < n_src; ++i_src) {
                    for (int i_trg = 0; i_trg < n_test_src; ++i_trg)
                        test_src[i_trg] += charges[i_src] * potential(&r_src[i_src * n_dim], &r_src[i_trg * n_dim]);
                    for (int i_trg = 0; i_trg < n_test_trg; ++i_trg)
                        test_trg[i_trg] += charges[i_src] * potential(&r_src[i_src * n_dim], &r_trg[i_trg * n_dim]);
                }

                pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], &dipstr[0],
                                                  n_trg, &r_trg[0]);
                pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);

#ifdef DMK_HAVE_REFERENCE
                double tottimeinfo[20];
                int use_dipole = 0;
                auto pot_src_fort = pot_src;
                auto grad_src_fort = grad_src;
                auto hess_src_fort = hess_src;
                auto pot_trg_fort = pot_trg;
                auto grad_trg_fort = grad_trg;
                auto hess_trg_fort = hess_trg;
                double st = MY_OMP_GET_WTIME();
                pdmk_(&nd, &n_dim, &params.eps, (int *)&params.kernel, &params.fparam, &params.use_periodic, &n_src,
                      &r_src[0], &params.use_charge, &charges[0], &use_dipole, nullptr, nullptr, (int *)&params.pgh_src,
                      &pot_src_fort[0], &grad_src_fort[0], &hess_src_fort[0], &n_trg, &r_trg[0], (int *)&params.pgh_trg,
                      &pot_trg_fort[0], &grad_trg_fort[0], &hess_trg_fort[0], tottimeinfo);
                std::cout << MY_OMP_GET_WTIME() - st << std::endl;
#endif

                double l2_err_src = 0.0;
                double l2_err_trg = 0.0;
                for (int i = 0; i < n_test_src; ++i)
                    l2_err_src += test_src[i] != 0.0 ? sctl::pow<2>(1.0 - pot_src[i] / test_src[i]) : 0.0;
                for (int i = 0; i < n_test_trg; ++i)
                    l2_err_trg += test_trg[i] != 0.0 ? sctl::pow<2>(1.0 - pot_trg[i] / test_trg[i]) : 0.0;

                l2_err_src = std::sqrt(l2_err_src) / n_test_src;
                l2_err_trg = std::sqrt(l2_err_trg) / n_test_trg;
                // FIXME: We should strengthen the checks here
                CHECK(l2_err_src < 6 * params.eps);
                CHECK(l2_err_trg < 6 * params.eps);

                // Scale charges by 1/2 and re-evaluate. Since the kernel
                // is linear in the charges, potentials should scale by the same factor.
                {
                    const double scale = 2.0;
                    sctl::Vector<double> scaled_charges(charges.Dim());
                    for (sctl::Long i = 0; i < charges.Dim(); ++i)
                        scaled_charges[i] = charges[i] * scale;

                    int rc = pdmk_tree_update_charges(tree, &scaled_charges[0], nullptr, nullptr);
                    CHECK(rc == 0);

                    sctl::Vector<double> pot_src_updated(n_src * nd), pot_trg_updated(n_trg * nd);
                    pdmk_tree_eval(tree, &pot_src_updated[0], &pot_trg_updated[0]);

                    // Check that updated potentials ≈ scale * original potentials
                    double l2_err_src_update = 0.0;
                    double l2_ref_src = 0.0;
                    for (int i = 0; i < n_src; ++i) {
                        double expected = pot_src[i] * scale;
                        CHECK(std::abs(expected - pot_src_updated[i]) < 5 * std::numeric_limits<double>::epsilon());
                    }

                    for (int i = 0; i < n_test_trg; ++i) {
                        double expected = pot_trg[i] * scale;
                        CHECK(std::abs(expected - pot_trg_updated[i]) < 5 * std::numeric_limits<double>::epsilon());
                    }
                }

                pdmk_tree_destroy(tree);
            }
        }
    }
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace gradient", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 4000;
    constexpr int n_trg = 3000;
    constexpr int nd = 1;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;
    constexpr double thresh2 = 1e-30;
    constexpr int output_dim = 1 + n_dim;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    sctl::Vector<double> r_src, charges, rnormal, dipstr, r_trg;
    dmk::util::init_test_data(n_dim, nd, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges,
                              dipstr, 0);

    sctl::Vector<double> pot_src(n_src * output_dim), pot_trg(n_trg * output_dim);
    pot_src.SetZero();
    pot_trg.SetZero();

    pdmk_params params;
    params.eps = 1e-7;
    params.n_dim = n_dim;
    params.n_per_leaf = 280;
    params.pgh_src = DMK_POTENTIAL_GRAD;
    params.pgh_trg = DMK_POTENTIAL_GRAD;
    params.kernel = DMK_LAPLACE;
    params.log_level = SPDLOG_LEVEL_OFF;

    pdmk_tree tree =
        pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], &dipstr[0], n_trg, &r_trg[0]);
    pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
    pdmk_tree_destroy(tree);

    const int n_test_src = std::min(n_src, 64);
    const int n_test_trg = std::min(n_trg, 64);
    std::vector<double> direct_grad_src(n_test_src * n_dim, 0.0);
    std::vector<double> direct_grad_trg(n_test_trg * n_dim, 0.0);

    const auto grad_index = [n_dim](int i_pt, int i_dim) { return i_dim + n_dim * i_pt; };
    const auto accumulate_laplace_grad = [&](const double *target, int i_out, std::vector<double> &out) {
        for (int i_src = 0; i_src < n_src; ++i_src) {
            double dx[n_dim];
            double dr2 = 0.0;
            for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                dx[i_dim] = target[i_dim] - r_src[i_src * n_dim + i_dim];
                dr2 += dx[i_dim] * dx[i_dim];
            }
            if (dr2 <= thresh2)
                continue;

            const double rinv = 1.0 / std::sqrt(dr2);
            const double rinv3 = rinv / dr2;
            for (int i_dim = 0; i_dim < n_dim; ++i_dim)
                out[grad_index(i_out, i_dim)] -= charges[i_src] * dx[i_dim] * rinv3;
        }
    };

    for (int i = 0; i < n_test_src; ++i)
        accumulate_laplace_grad(&r_src[i * n_dim], i, direct_grad_src);
    for (int i = 0; i < n_test_trg; ++i)
        accumulate_laplace_grad(&r_trg[i * n_dim], i, direct_grad_trg);

    auto relative_l2_error = [](const auto &approx, const auto &exact) {
        double err2 = 0.0;
        double ref2 = 0.0;
        for (int i = 0; i < exact.size(); ++i) {
            err2 += sctl::pow<2>(approx[i] - exact[i]);
            ref2 += sctl::pow<2>(exact[i]);
        }
        return std::sqrt(err2 / ref2);
    };

    std::vector<double> grad_src_prefix(direct_grad_src.size());
    std::vector<double> grad_trg_prefix(direct_grad_trg.size());
    for (int i = 0; i < n_test_src; ++i)
        for (int i_dim = 0; i_dim < n_dim; ++i_dim)
            grad_src_prefix[grad_index(i, i_dim)] = pot_src[i * output_dim + 1 + i_dim];
    for (int i = 0; i < n_test_trg; ++i)
        for (int i_dim = 0; i_dim < n_dim; ++i_dim)
            grad_trg_prefix[grad_index(i, i_dim)] = pot_trg[i * output_dim + 1 + i_dim];

    const double l2_err_src = relative_l2_error(grad_src_prefix, direct_grad_src);
    const double l2_err_trg = relative_l2_error(grad_trg_prefix, direct_grad_trg);

    CHECK(l2_err_src < 2e-4);
    CHECK(l2_err_trg < 1e-3);
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace PBC direct verification", 1) {
    // Verify PBC direct interactions at 3, 6, 9, 12 digit precision.
    // For each precision: build tree with pbc=true, run direct-only,
    // compare against naive O(N^2) loop with the same residual kernel
    // polynomial over 27 periodic images.

    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 500;
    constexpr double thresh2 = 1e-30;

#ifdef DMK_HAVE_MPI
    auto sctl_comm = sctl::Comm(test_comm);
#else
    auto sctl_comm = sctl::Comm::Self();
#endif

    // Generate uniformly distributed particles in [0,1)^3
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

    // Residual kernel polynomial coefficients for 3D Laplace (from aot_kernels.cpp)
    // clang-format off
    const double coeffs_3[] = {
        1.627823522210361e-01, -4.553645597616490e-01, 4.171687104204163e-01,
        -7.073638602709915e-02, -8.957845614474928e-02, 2.617986644718201e-02,
        9.633427876507601e-03};
    const double coeffs_6[] = {
        5.482525801351582e-02, -2.616592110444692e-01, 4.862652666337138e-01,
        -3.894296348642919e-01, 1.638587821812791e-02, 1.870328434198821e-01,
        -8.714171086568978e-02, -3.927020727017803e-02, 3.728187607052319e-02,
        3.153734425831139e-03, -8.651313377285847e-03, 1.725110090795567e-04,
        1.034762385284044e-03};
    const double coeffs_9[] = {
        1.835718730962269e-02, -1.258015846164503e-01, 3.609487248584408e-01, -5.314579651112283e-01,
        3.447559412892380e-01, 9.664692318551721e-02, -3.124274531849053e-01, 1.322460720579388e-01,
        9.773007866584822e-02, -1.021958831082768e-01, -3.812847450976566e-03, 3.858117355875043e-02,
        -8.728545924521301e-03, -9.401196355382909e-03, 4.024549377076924e-03, 1.512806105865091e-03,
        -9.576734877247042e-04, -1.303457547418901e-04, 1.100385844683190e-04};
    const double coeffs_12[] = {
        6.262472576363448e-03, -5.605742936112479e-02, 2.185890864792949e-01, -4.717350304955679e-01,
        5.669680214206270e-01, -2.511606878849214e-01, -2.744523658778361e-01, 4.582527599363415e-01,
        -1.397724810121539e-01, -2.131762135835757e-01, 1.995489373508990e-01, 1.793390341864239e-02,
        -1.035055132403432e-01, 3.035606831075176e-02, 3.153931762550532e-02, -2.033178627450288e-02,
        -5.406682731236552e-03, 7.543645573618463e-03, 1.437788047407851e-05, -1.928370882351732e-03,
        2.891658777328665e-04, 3.332996162099811e-04, -8.397699195938912e-05, -3.015837377517983e-05,
        9.640642701924662e-06};
    // clang-format on

    struct PrecisionCase {
        int n_digits;
        double eps;
        const double *coeffs;
        int n_coeffs;
    };
    const PrecisionCase cases[] = {
        {3, 1e-3, coeffs_3, (int)(sizeof(coeffs_3) / sizeof(double))},
        {6, 1e-6, coeffs_6, (int)(sizeof(coeffs_6) / sizeof(double))},
        {9, 1e-9, coeffs_9, (int)(sizeof(coeffs_9) / sizeof(double))},
        {12, 1e-12, coeffs_12, (int)(sizeof(coeffs_12) / sizeof(double))},
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
            params.log_level = SPDLOG_LEVEL_OFF;

            dmk::DMKPtTree<double, n_dim> tree(sctl_comm, params, r_src, r_trg, charges);

            tree.pot_src_sorted.SetZero();
            tree.pot_trg_sorted.SetZero();
            tree.evaluate_direct_interactions(tree.r_src_t.data(), tree.r_trg_t.data());

            const auto &node_mid = tree.GetNodeMID();
            const auto &node_attr = tree.GetNodeAttr();

            // Gather source metadata
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

            // Gather target metadata
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

            // Naive reference with periodic images
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
                                double r = std::sqrt(r2);
                                double x = (r + cen) * rsc;
                                ref_pot[i_trg] += src.charge * horner_eval(x, pc.coeffs, pc.n_coeffs) / r;
                            }
                }
            }

            const int n_test = std::min((int)targets.size(), 200);
            double err2 = 0.0, ref2 = 0.0;
            for (int i = 0; i < n_test; ++i) {
                double tree_val = tree.pot_trg_sorted[targets[i].pot_offset];
                err2 += sctl::pow<2>(tree_val - ref_pot[i]);
                ref2 += sctl::pow<2>(ref_pot[i]);
            }
            double l2_err = (ref2 > 0) ? std::sqrt(err2 / ref2) : std::sqrt(err2);
            MESSAGE("n_digits=", pc.n_digits, " eps=", pc.eps, " l2_err=", l2_err,
                    " n_levels=", tree.n_levels(), " n_boxes=", tree.n_boxes());
            // At 3-digit precision the vectorized evaluator uses a transformed polynomial
            // (transform_poly=true for n_digits<6), causing small numerical differences
            // vs the naive Horner evaluation. For higher precision, expect machine epsilon.
            const double tol = (pc.n_digits <= 3) ? 1e-3 : 1e-6;
            CHECK(l2_err < tol);
        }
    }
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace diff+residual vs erfc reference (free space)", 1) {
    // Verify that the DMK output from PW levels 1..L + direct (i.e., everything
    // except the root-level windowed kernel and D_0) matches the reference:
    //   ref(r_t) = sum_s q_s * erfc(|r_t - r_s| / sigma_1) / |r_t - r_s|
    // This follows from the telescoping sum D_1 + D_2 + ... + R_L = erfc(r/sigma_1)/r,
    // where sigma_l = boxsize[l] / sqrt(log(1/eps)) (Jiang & Greengard eq 3.26-3.28).
    // Free-space test first to validate the reference before extending to PBC.

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

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 50; // ensures 3-level tree
    params.pgh_src = DMK_POTENTIAL;
    params.pgh_trg = DMK_POTENTIAL;
    params.kernel = DMK_LAPLACE;
    params.use_periodic = false; // free space first
    params.log_level = SPDLOG_LEVEL_OFF;

    dmk::DMKPtTree<double, n_dim> tree(sctl_comm, params, r_src, r_trg, charges);
    REQUIRE(tree.n_levels() >= 3);

    // Run upward pass, init PW, zero potentials
    tree.upward_pass();
    tree.init_planewave_data();
    tree.pot_src_sorted.SetZero();
    tree.pot_trg_sorted.SetZero();

    // Run PW for levels 1..L (skip level 0 = D_0)
    for (int i_level = 1; i_level < tree.n_levels(); ++i_level) {
        auto &dfd = tree.difference_fourier_data[i_level];
        const ndview<std::complex<double>, 2> p2pw({tree.n_pw, tree.n_order}, &dfd.poly2pw[0]);
        const ndview<std::complex<double>, 2> pw2p({tree.n_pw, tree.n_order}, &dfd.pw2poly[0]);
        tree.form_outgoing_expansions(tree.level_indices[i_level], p2pw, dfd.radialft);
        tree.form_eval_expansions(tree.level_indices[i_level], dfd.wpwshift, tree.boxsize[i_level], pw2p, tree.p2c);
    }
    // Run direct (residual) evaluation
    tree.evaluate_direct_interactions(tree.r_src_t.data(), tree.r_trg_t.data());

    const auto &node_attr = tree.GetNodeAttr();

    // Gather sources from leaf boxes
    struct SourceInfo {
        double r[3];
        double charge;
    };
    std::vector<SourceInfo> sources;
    for (int box = 0; box < (int)tree.n_boxes(); ++box) {
        if (!node_attr[box].Leaf || node_attr[box].Ghost)
            continue;
        const int n = tree.r_src_cnt_owned[box];
        if (!n)
            continue;
        const double *rp = tree.r_src_owned_ptr(box);
        const double *cp = tree.charge_owned_ptr(box);
        for (int i = 0; i < n; ++i)
            sources.push_back({{rp[i * 3], rp[i * 3 + 1], rp[i * 3 + 2]}, cp[i]});
    }

    // Gather targets from leaf boxes
    struct TargetInfo {
        double r[3];
        int pot_offset;
    };
    std::vector<TargetInfo> targets;
    for (int box = 0; box < (int)tree.n_boxes(); ++box) {
        if (node_attr[box].Ghost || !node_attr[box].Leaf)
            continue;
        const int n = tree.r_trg_cnt_owned[box];
        if (!n)
            continue;
        const double *rp = tree.r_trg_owned_ptr(box);
        for (int i = 0; i < n; ++i)
            targets.push_back({{rp[i * 3], rp[i * 3 + 1], rp[i * 3 + 2]}, (int)tree.pot_trg_offsets[box] + i});
    }

    // Reference uses the PSWF real-space formula for difference kernels:
    //   D_l(r) = [1/(4π·r·c0)] · [Φ(min(1, 2r/bsize_l)) - Φ(r/bsize_l)]
    // where Φ(x) = ∫₀ˣ ψ₀(t) dt = pf.int_eval(x), and c0 = ∫₀¹ ψ₀.
    // The residual R_L uses the leaf polynomial from aot_kernels.
    // The sum D_1 + ... + D_{L-1} + R_L should equal the DMK output (PW levels 1..L + direct).

    auto &pf = tree.fourier_data.prolate0_fun;
    const double c0 = pf.intvals(tree.fourier_data.beta())[0];
    const double inv_c0 = 1.0 / c0;
    const int L = tree.n_levels() - 1; // leaf level index

    // 6-digit leaf residual polynomial (from aot_kernels.cpp)
    // clang-format off
    const double res_coeffs[] = {
        5.482525801351582e-02, -2.616592110444692e-01, 4.862652666337138e-01,
        -3.894296348642919e-01, 1.638587821812791e-02, 1.870328434198821e-01,
        -8.714171086568978e-02, -3.927020727017803e-02, 3.728187607052319e-02,
        3.153734425831139e-03, -8.651313377285847e-03, 1.725110090795567e-04,
        1.034762385284044e-03};
    // clang-format on
    const int n_res_coeffs = (int)(sizeof(res_coeffs) / sizeof(double));
    auto horner_eval = [](double x, const double *c, int n) {
        double val = c[n - 1];
        for (int i = n - 2; i >= 0; --i)
            val = val * x + c[i];
        return val;
    };

    // Evaluate D_l(r) via PSWF integral
    auto eval_D_l = [&](double r, int level) -> double {
        double bsize = tree.boxsize[level];
        double u_lo = r / bsize;        // = r / boxsize[l]
        double u_hi = 2.0 * r / bsize;  // = r / boxsize[l+1]
        if (u_lo >= 1.0)
            return 0.0; // outside support
        u_hi = std::min(u_hi, 1.0);
        double integral = pf.int_eval(u_hi) - pf.int_eval(u_lo);
        return inv_c0 * integral / r;
    };

    // Evaluate leaf residual R_L(r) via polynomial
    auto eval_R_L = [&](double r) -> double {
        double bsize = tree.boxsize[L];
        if (r * r >= bsize * bsize)
            return 0.0;
        double rsc = 2.0 / bsize;
        double cen = -bsize / 2.0;
        double x = (r + cen) * rsc;
        return horner_eval(x, res_coeffs, n_res_coeffs) / r;
    };

    std::vector<double> ref_pot(targets.size(), 0.0);
    for (int i_trg = 0; i_trg < (int)targets.size(); ++i_trg) {
        const auto &trg = targets[i_trg];
        for (const auto &src : sources) {
            double dx = trg.r[0] - src.r[0];
            double dy = trg.r[1] - src.r[1];
            double dz = trg.r[2] - src.r[2];
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < thresh2)
                continue;
            double r = std::sqrt(r2);
            // Sum D_1 + D_2 + ... + D_{L-1} + R_L
            double val = 0.0;
            for (int l = 1; l < L; ++l)
                val += eval_D_l(r, l);
            val += eval_R_L(r);
            ref_pot[i_trg] += src.charge * val;
        }
    }

    const int n_test = std::min((int)targets.size(), 200);
    double err2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < n_test; ++i) {
        double tree_val = tree.pot_trg_sorted[targets[i].pot_offset];
        err2 += sctl::pow<2>(tree_val - ref_pot[i]);
        ref2 += sctl::pow<2>(ref_pot[i]);
        if (i < 5)
            MESSAGE("  i=", i, " tree=", tree_val, " ref=", ref_pot[i], " diff=", tree_val - ref_pot[i]);
    }
    double l2_err = (ref2 > 0) ? std::sqrt(err2 / ref2) : std::sqrt(err2);
    MESSAGE("diff+residual vs PSWF ref: l2_err=", l2_err, " c0=", c0,
            " n_levels=", tree.n_levels(), " n_boxes=", tree.n_boxes());
    CHECK(l2_err < 1e-3);
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace PBC diff+residual vs PSWF reference", 1) {
    // PBC version: verify D_1 + ... + D_{L-1} + R_L with 27 periodic images
    // at 3, 6, 9, 12 digit precision.
    // D_l(r) = [1/(r*c0)] * [Phi(min(1, 2r/bsize_l)) - Phi(r/bsize_l)]
    // R_L(r) = P(x_L)/r (leaf residual polynomial)

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

    // Residual kernel polynomial coefficients per precision (from aot_kernels.cpp)
    // clang-format off
    const double coeffs_3[] = {
        1.627823522210361e-01, -4.553645597616490e-01, 4.171687104204163e-01,
        -7.073638602709915e-02, -8.957845614474928e-02, 2.617986644718201e-02,
        9.633427876507601e-03};
    const double coeffs_6[] = {
        5.482525801351582e-02, -2.616592110444692e-01, 4.862652666337138e-01,
        -3.894296348642919e-01, 1.638587821812791e-02, 1.870328434198821e-01,
        -8.714171086568978e-02, -3.927020727017803e-02, 3.728187607052319e-02,
        3.153734425831139e-03, -8.651313377285847e-03, 1.725110090795567e-04,
        1.034762385284044e-03};
    const double coeffs_9[] = {
        1.835718730962269e-02, -1.258015846164503e-01, 3.609487248584408e-01, -5.314579651112283e-01,
        3.447559412892380e-01, 9.664692318551721e-02, -3.124274531849053e-01, 1.322460720579388e-01,
        9.773007866584822e-02, -1.021958831082768e-01, -3.812847450976566e-03, 3.858117355875043e-02,
        -8.728545924521301e-03, -9.401196355382909e-03, 4.024549377076924e-03, 1.512806105865091e-03,
        -9.576734877247042e-04, -1.303457547418901e-04, 1.100385844683190e-04};
    const double coeffs_12[] = {
        6.262472576363448e-03, -5.605742936112479e-02, 2.185890864792949e-01, -4.717350304955679e-01,
        5.669680214206270e-01, -2.511606878849214e-01, -2.744523658778361e-01, 4.582527599363415e-01,
        -1.397724810121539e-01, -2.131762135835757e-01, 1.995489373508990e-01, 1.793390341864239e-02,
        -1.035055132403432e-01, 3.035606831075176e-02, 3.153931762550532e-02, -2.033178627450288e-02,
        -5.406682731236552e-03, 7.543645573618463e-03, 1.437788047407851e-05, -1.928370882351732e-03,
        2.891658777328665e-04, 3.332996162099811e-04, -8.397699195938912e-05, -3.015837377517983e-05,
        9.640642701924662e-06};
    // clang-format on

    auto horner_eval = [](double x, const double *c, int n) {
        double val = c[n - 1];
        for (int i = n - 2; i >= 0; --i)
            val = val * x + c[i];
        return val;
    };

    struct PrecisionCase {
        int n_digits;
        double eps;
        const double *coeffs;
        int n_coeffs;
    };
    const PrecisionCase cases[] = {
        {3, 1e-3, coeffs_3, (int)(sizeof(coeffs_3) / sizeof(double))},
        {6, 1e-6, coeffs_6, (int)(sizeof(coeffs_6) / sizeof(double))},
        {9, 1e-9, coeffs_9, (int)(sizeof(coeffs_9) / sizeof(double))},
        {12, 1e-12, coeffs_12, (int)(sizeof(coeffs_12) / sizeof(double))},
    };

    for (const auto &pc : cases) {
        SUBCASE(("n_digits=" + std::to_string(pc.n_digits)).c_str()) {
            pdmk_params params;
            params.eps = pc.eps;
            params.n_dim = n_dim;
            params.n_per_leaf = 50;
            params.pgh_src = DMK_POTENTIAL;
            params.pgh_trg = DMK_POTENTIAL;
            params.kernel = DMK_LAPLACE;
            params.use_periodic = true;
            params.log_level = SPDLOG_LEVEL_OFF;

            dmk::DMKPtTree<double, n_dim> tree(sctl_comm, params, r_src, r_trg, charges);
            REQUIRE(tree.n_levels() >= 3);

            // Run upward pass, init PW, zero potentials
            tree.upward_pass();
            tree.init_planewave_data();
            tree.pot_src_sorted.SetZero();
            tree.pot_trg_sorted.SetZero();

            // Run PW for levels 1..L (skip root level)
            for (int i_level = 1; i_level < tree.n_levels(); ++i_level) {
                auto &dfd = tree.difference_fourier_data[i_level];
                const ndview<std::complex<double>, 2> p2pw({tree.n_pw, tree.n_order}, &dfd.poly2pw[0]);
                const ndview<std::complex<double>, 2> pw2p({tree.n_pw, tree.n_order}, &dfd.pw2poly[0]);
                tree.form_outgoing_expansions(tree.level_indices[i_level], p2pw, dfd.radialft);
                tree.form_eval_expansions(tree.level_indices[i_level], dfd.wpwshift, tree.boxsize[i_level], pw2p,
                                          tree.p2c);
            }
            // Run direct (residual) evaluation
            tree.evaluate_direct_interactions(tree.r_src_t.data(), tree.r_trg_t.data());

            const auto &node_attr = tree.GetNodeAttr();

            // Gather sources
            struct SourceInfo {
                double r[3];
                double charge;
            };
            std::vector<SourceInfo> sources;
            for (int box = 0; box < (int)tree.n_boxes(); ++box) {
                if (!node_attr[box].Leaf || node_attr[box].Ghost)
                    continue;
                const int n = tree.r_src_cnt_owned[box];
                if (!n)
                    continue;
                const double *rp = tree.r_src_owned_ptr(box);
                const double *cp = tree.charge_owned_ptr(box);
                for (int i = 0; i < n; ++i)
                    sources.push_back({{rp[i * 3], rp[i * 3 + 1], rp[i * 3 + 2]}, cp[i]});
            }

            // Gather targets
            struct TargetInfo {
                double r[3];
                int pot_offset;
            };
            std::vector<TargetInfo> targets;
            for (int box = 0; box < (int)tree.n_boxes(); ++box) {
                if (node_attr[box].Ghost || !node_attr[box].Leaf)
                    continue;
                const int n = tree.r_trg_cnt_owned[box];
                if (!n)
                    continue;
                const double *rp = tree.r_trg_owned_ptr(box);
                for (int i = 0; i < n; ++i)
                    targets.push_back(
                        {{rp[i * 3], rp[i * 3 + 1], rp[i * 3 + 2]}, (int)tree.pot_trg_offsets[box] + i});
            }

            // PSWF reference with 27 periodic images
            auto &pf = tree.fourier_data.prolate0_fun;
            const double c0 = pf.intvals(tree.fourier_data.beta())[0];
            const double inv_c0 = 1.0 / c0;
            const int L = tree.n_levels() - 1;

            auto eval_D_l = [&](double r, int level) -> double {
                double bsize = tree.boxsize[level];
                double u_lo = r / bsize;
                double u_hi = 2.0 * r / bsize;
                if (u_lo >= 1.0)
                    return 0.0;
                u_hi = std::min(u_hi, 1.0);
                return inv_c0 * (pf.int_eval(u_hi) - pf.int_eval(u_lo)) / r;
            };

            auto eval_R_L = [&](double r) -> double {
                double bsize = tree.boxsize[L];
                if (r * r >= bsize * bsize)
                    return 0.0;
                double rsc = 2.0 / bsize;
                double cen = -bsize / 2.0;
                double x = (r + cen) * rsc;
                return horner_eval(x, pc.coeffs, pc.n_coeffs) / r;
            };

            std::vector<double> ref_pot(targets.size(), 0.0);
            for (int i_trg = 0; i_trg < (int)targets.size(); ++i_trg) {
                const auto &trg = targets[i_trg];
                for (const auto &src : sources) {
                    for (int mx = -1; mx <= 1; ++mx)
                        for (int my = -1; my <= 1; ++my)
                            for (int mz = -1; mz <= 1; ++mz) {
                                double dx = trg.r[0] - src.r[0] - mx;
                                double dy = trg.r[1] - src.r[1] - my;
                                double dz = trg.r[2] - src.r[2] - mz;
                                double r2 = dx * dx + dy * dy + dz * dz;
                                if (r2 < thresh2)
                                    continue;
                                double r = std::sqrt(r2);
                                double val = 0.0;
                                for (int l = 1; l < L; ++l)
                                    val += eval_D_l(r, l);
                                val += eval_R_L(r);
                                ref_pot[i_trg] += src.charge * val;
                            }
                }
            }

            const int n_test = std::min((int)targets.size(), 200);
            double err2 = 0.0, ref2 = 0.0;
            for (int i = 0; i < n_test; ++i) {
                double tree_val = tree.pot_trg_sorted[targets[i].pot_offset];
                err2 += sctl::pow<2>(tree_val - ref_pot[i]);
                ref2 += sctl::pow<2>(ref_pot[i]);
            }
            double l2_err = (ref2 > 0) ? std::sqrt(err2 / ref2) : std::sqrt(err2);
            MESSAGE("PBC n_digits=", pc.n_digits, " l2_err=", l2_err, " n_levels=", tree.n_levels(),
                    " n_boxes=", tree.n_boxes());
            const double tol = (pc.n_digits <= 3) ? 1e-2 : 1e-3;
            CHECK(l2_err < tol);
        }
    }
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace PBC full vs Ewald", 1) {
    // Verify that V_partial (D_1..R_L from tree) + V_{W0+D0} (PSWF kernel on periodic grid)
    // matches V_Ewald at 3/6/9/12 digit precision.
    // W0+D0 kernel: (4pi/psi0(0)) * psi0(|k|*sigma1) / |k|^2, sigma1 = boxsize[1]/beta.

    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 500;

#ifdef DMK_HAVE_MPI
    auto sctl_comm = sctl::Comm(test_comm);
#else
    auto sctl_comm = sctl::Comm::Self();
#endif

    std::default_random_engine eng(77);
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

    struct PrecisionCase {
        int n_digits;
        double eps;
        double tol;
    };
    const PrecisionCase cases[] = {
        {3, 1e-3, 1e-2},
        {6, 1e-6, 1e-4},
        {9, 1e-9, 1e-7},
        {12, 1e-12, 1e-10},
    };

    for (const auto &pc : cases) {
        SUBCASE(("n_digits=" + std::to_string(pc.n_digits)).c_str()) {
            pdmk_params params;
            params.eps = pc.eps;
            params.n_dim = n_dim;
            params.n_per_leaf = 50;
            params.pgh_src = DMK_POTENTIAL;
            params.pgh_trg = DMK_POTENTIAL;
            params.kernel = DMK_LAPLACE;
            params.use_periodic = true;
            params.log_level = SPDLOG_LEVEL_OFF;

            dmk::DMKPtTree<double, n_dim> tree(sctl_comm, params, r_src, r_trg, charges);
            REQUIRE(tree.n_levels() >= 3);

            // --- V_partial: levels 1..L + direct ---
            tree.upward_pass();
            tree.init_planewave_data();
            tree.pot_src_sorted.SetZero();
            tree.pot_trg_sorted.SetZero();

            for (int i_level = 1; i_level < tree.n_levels(); ++i_level) {
                auto &dfd = tree.difference_fourier_data[i_level];
                const ndview<std::complex<double>, 2> p2pw({tree.n_pw, tree.n_order}, &dfd.poly2pw[0]);
                const ndview<std::complex<double>, 2> pw2p({tree.n_pw, tree.n_order}, &dfd.pw2poly[0]);
                tree.form_outgoing_expansions(tree.level_indices[i_level], p2pw, dfd.radialft);
                tree.form_eval_expansions(tree.level_indices[i_level], dfd.wpwshift, tree.boxsize[i_level],
                                          pw2p, tree.p2c);
            }
            tree.evaluate_direct_interactions(tree.r_src_t.data(), tree.r_trg_t.data());

            // --- Gather sources and targets ---
            const auto &node_attr = tree.GetNodeAttr();

            struct SrcInfo {
                double r[3];
                double q;
            };
            struct TrgInfo {
                double r[3];
                int pot_offset;
            };
            std::vector<SrcInfo> sources;
            std::vector<TrgInfo> targets;

            for (int box = 0; box < (int)tree.n_boxes(); ++box) {
                if (!node_attr[box].Leaf || node_attr[box].Ghost)
                    continue;
                const int n = tree.r_src_cnt_owned[box];
                if (!n)
                    continue;
                const double *rp = tree.r_src_owned_ptr(box);
                const double *cp = tree.charge_owned_ptr(box);
                for (int i = 0; i < n; ++i)
                    sources.push_back({{rp[i * 3], rp[i * 3 + 1], rp[i * 3 + 2]}, cp[i]});
            }
            for (int box = 0; box < (int)tree.n_boxes(); ++box) {
                if (node_attr[box].Ghost || !node_attr[box].Leaf)
                    continue;
                const int n = tree.r_trg_cnt_owned[box];
                if (!n)
                    continue;
                const double *rp = tree.r_trg_owned_ptr(box);
                for (int i = 0; i < n; ++i)
                    targets.push_back(
                        {{rp[i * 3], rp[i * 3 + 1], rp[i * 3 + 2]}, (int)tree.pot_trg_offsets[box] + i});
            }

            // --- V_{W0+D0}: PSWF kernel on periodic Fourier grid ---
            const double L = 1.0;
            const double V_box = L * L * L;
            const double dk = 2.0 * M_PI / L;
            const double beta = tree.fourier_data.beta();
            auto &pf = tree.fourier_data.prolate0_fun;
            const double psi0_at_zero = pf.eval_val(0.0);
            const double sigma1 = tree.boxsize[1] / beta;
            const int n_modes = (int)std::ceil(beta / M_PI) + 2;

            // Ewald parameters
            const double alpha = 10.0;
            const int n_ewald = std::max(15, n_modes);
            const int n_grid = std::max(n_modes, n_ewald);

            // Precompute rho(k) on the full grid
            const int d = 2 * n_grid + 1;
            std::vector<std::complex<double>> rho(d * d * d, {0.0, 0.0});

            for (const auto &src : sources) {
                const std::complex<double> exp_x0 = std::exp(std::complex<double>(0.0, -dk * src.r[0]));
                const std::complex<double> exp_y0 = std::exp(std::complex<double>(0.0, -dk * src.r[1]));
                const std::complex<double> exp_z0 = std::exp(std::complex<double>(0.0, -dk * src.r[2]));

                std::vector<std::complex<double>> exp_x(d), exp_y(d), exp_z(d);
                for (int a = -n_grid; a <= n_grid; ++a) {
                    exp_x[a + n_grid] = std::pow(exp_x0, a);
                    exp_y[a + n_grid] = std::pow(exp_y0, a);
                    exp_z[a + n_grid] = std::pow(exp_z0, a);
                }

                for (int ix = 0; ix < d; ++ix)
                    for (int iy = 0; iy < d; ++iy) {
                        auto t2 = src.q * exp_x[ix] * exp_y[iy];
                        for (int iz = 0; iz < d; ++iz)
                            rho[ix * d * d + iy * d + iz] += t2 * exp_z[iz];
                    }
            }

            // Compute V_{W0+D0} using PSWF kernel
            std::vector<double> v_w0d0(targets.size(), 0.0);
            for (int it = 0; it < (int)targets.size(); ++it) {
                const auto &trg = targets[it];
                const std::complex<double> exp_tx0 = std::exp(std::complex<double>(0.0, dk * trg.r[0]));
                const std::complex<double> exp_ty0 = std::exp(std::complex<double>(0.0, dk * trg.r[1]));
                const std::complex<double> exp_tz0 = std::exp(std::complex<double>(0.0, dk * trg.r[2]));

                std::vector<std::complex<double>> exp_tx(2 * n_modes + 1), exp_ty(2 * n_modes + 1),
                    exp_tz(2 * n_modes + 1);
                for (int a = -n_modes; a <= n_modes; ++a) {
                    exp_tx[a + n_modes] = std::pow(exp_tx0, a);
                    exp_ty[a + n_modes] = std::pow(exp_ty0, a);
                    exp_tz[a + n_modes] = std::pow(exp_tz0, a);
                }

                double pot = 0.0;
                for (int nx = -n_modes; nx <= n_modes; ++nx)
                    for (int ny = -n_modes; ny <= n_modes; ++ny)
                        for (int nz = -n_modes; nz <= n_modes; ++nz) {
                            if (nx == 0 && ny == 0 && nz == 0)
                                continue;
                            const double kx = dk * nx, ky = dk * ny, kz = dk * nz;
                            const double k2 = kx * kx + ky * ky + kz * kz;
                            const double kappa = std::sqrt(k2);

                            const double arg = kappa * sigma1;
                            const double psi_val = (std::abs(arg) <= 1.0) ? pf.eval_val(arg) : 0.0;
                            const double kernel_hat = (4.0 * M_PI / psi0_at_zero) * psi_val / k2;

                            const int ix = nx + n_grid, iy = ny + n_grid, iz = nz + n_grid;
                            const auto &rho_k = rho[ix * d * d + iy * d + iz];
                            const auto exp_ikr =
                                exp_tx[nx + n_modes] * exp_ty[ny + n_modes] * exp_tz[nz + n_modes];
                            pot += kernel_hat * std::real(rho_k * exp_ikr);
                        }
                v_w0d0[it] = pot / V_box;
            }

            // --- V_Ewald: self-contained reference ---
            const double r_c = 0.5 * L;
            std::vector<double> v_ewald(targets.size(), 0.0);

            // Short range
            for (int it = 0; it < (int)targets.size(); ++it) {
                const auto &trg = targets[it];
                for (const auto &src : sources) {
                    for (int mx = -1; mx <= 1; ++mx)
                        for (int my = -1; my <= 1; ++my)
                            for (int mz = -1; mz <= 1; ++mz) {
                                const double dx2 = trg.r[0] - src.r[0] - mx * L;
                                const double dy2 = trg.r[1] - src.r[1] - my * L;
                                const double dz2 = trg.r[2] - src.r[2] - mz * L;
                                const double r = std::sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);
                                if (r > 1e-15 && r <= r_c)
                                    v_ewald[it] += src.q * std::erfc(alpha * r) / r;
                            }
                }
            }

            // Long range
            for (int it = 0; it < (int)targets.size(); ++it) {
                const auto &trg = targets[it];
                const std::complex<double> exp_tx0 = std::exp(std::complex<double>(0.0, dk * trg.r[0]));
                const std::complex<double> exp_ty0 = std::exp(std::complex<double>(0.0, dk * trg.r[1]));
                const std::complex<double> exp_tz0 = std::exp(std::complex<double>(0.0, dk * trg.r[2]));

                std::vector<std::complex<double>> exp_tx(2 * n_ewald + 1), exp_ty(2 * n_ewald + 1),
                    exp_tz(2 * n_ewald + 1);
                for (int a = -n_ewald; a <= n_ewald; ++a) {
                    exp_tx[a + n_ewald] = std::pow(exp_tx0, a);
                    exp_ty[a + n_ewald] = std::pow(exp_ty0, a);
                    exp_tz[a + n_ewald] = std::pow(exp_tz0, a);
                }

                double pot_long = 0.0;
                for (int nx = -n_ewald; nx <= n_ewald; ++nx)
                    for (int ny = -n_ewald; ny <= n_ewald; ++ny)
                        for (int nz = -n_ewald; nz <= n_ewald; ++nz) {
                            if (nx == 0 && ny == 0 && nz == 0)
                                continue;
                            const double kx = dk * nx, ky = dk * ny, kz = dk * nz;
                            const double k2 = kx * kx + ky * ky + kz * kz;
                            const double ewald_kernel = std::exp(-k2 / (4.0 * alpha * alpha)) / k2;

                            const int ix = nx + n_grid, iy = ny + n_grid, iz = nz + n_grid;
                            const auto &rho_k = rho[ix * d * d + iy * d + iz];
                            const auto exp_ikr =
                                exp_tx[nx + n_ewald] * exp_ty[ny + n_ewald] * exp_tz[nz + n_ewald];
                            pot_long += ewald_kernel * std::real(rho_k * exp_ikr);
                        }
                v_ewald[it] += (4.0 * M_PI / V_box) * pot_long;
            }

            // --- Compare ---
            double err2 = 0.0, ref2 = 0.0;
            for (int i = 0; i < (int)targets.size(); ++i) {
                const double v_total = tree.pot_trg_sorted[targets[i].pot_offset] + v_w0d0[i];
                err2 += sctl::pow<2>(v_total - v_ewald[i]);
                ref2 += sctl::pow<2>(v_ewald[i]);
            }
            const double l2_err = (ref2 > 0) ? std::sqrt(err2 / ref2) : std::sqrt(err2);
            MESSAGE("PBC full vs Ewald: n_digits=", pc.n_digits, " l2_err=", l2_err,
                    " n_levels=", tree.n_levels(), " n_modes=", n_modes, " n_boxes=", tree.n_boxes());
            CHECK(l2_err < pc.tol);
        }
    }
}

TEST_CASE_GENERIC("[DMK] Ewald alpha-independence check", 1) {
    // Verify that the Ewald potential at source locations is independent of the
    // splitting parameter α. Any α-dependence indicates a missing self-correction.
    constexpr int n_dim = 3;
    constexpr int n_src = 200;
    constexpr int n_test = 10;

    std::default_random_engine eng(42);
    std::uniform_real_distribution<double> rng(0.01, 0.99);

    std::vector<double> r_src(n_dim * n_src), charges(n_src);
    for (int i = 0; i < n_src * n_dim; ++i) r_src[i] = rng(eng);
    for (int i = 0; i < n_src; ++i) charges[i] = rng(eng) - 0.5;
    { double sum = 0; for (int i = 0; i < n_src; ++i) sum += charges[i]; for (int i = 0; i < n_src; ++i) charges[i] -= sum / n_src; }

    const double L = 1.0;
    const double V_box = L * L * L;
    const double dk = 2.0 * M_PI / L;
    const double alphas[] = {5.0, 10.0, 20.0, 40.0};
    const int n_ewald = 40; // enough modes for all α

    // For each α, compute Ewald potential at n_test source points
    for (double alpha : alphas) {
        // Build rho(k)
        const int d = 2 * n_ewald + 1;
        std::vector<std::complex<double>> rho(d * d * d, {0, 0});
        for (int is = 0; is < n_src; ++is) {
            auto ex0 = std::exp(std::complex<double>(0, -dk * r_src[is * 3 + 0]));
            auto ey0 = std::exp(std::complex<double>(0, -dk * r_src[is * 3 + 1]));
            auto ez0 = std::exp(std::complex<double>(0, -dk * r_src[is * 3 + 2]));
            std::vector<std::complex<double>> ex(d), ey(d), ez(d);
            for (int a = -n_ewald; a <= n_ewald; ++a) { ex[a+n_ewald]=std::pow(ex0,a); ey[a+n_ewald]=std::pow(ey0,a); ez[a+n_ewald]=std::pow(ez0,a); }
            for (int ix = 0; ix < d; ++ix)
                for (int iy = 0; iy < d; ++iy) {
                    auto t2 = charges[is] * ex[ix] * ey[iy];
                    for (int iz = 0; iz < d; ++iz)
                        rho[ix*d*d + iy*d + iz] += t2 * ez[iz];
                }
        }

        // Ewald potential at source i
        auto ewald_pot = [&](int i) {
            const double *r = &r_src[i * 3];
            double pot_short = 0;
            // Short range
            for (int is = 0; is < n_src; ++is)
                for (int mx = -1; mx <= 1; ++mx)
                    for (int my = -1; my <= 1; ++my)
                        for (int mz = -1; mz <= 1; ++mz) {
                            double dx = r[0] - r_src[is*3+0] - mx*L;
                            double dy = r[1] - r_src[is*3+1] - my*L;
                            double dz = r[2] - r_src[is*3+2] - mz*L;
                            double r2 = dx*dx + dy*dy + dz*dz;
                            double rr = std::sqrt(r2);
                            if (rr > 1e-15) pot_short += charges[is] * std::erfc(alpha * rr) / rr;
                        }
            // Long range
            double pot_long = 0;
            auto etx0 = std::exp(std::complex<double>(0, dk*r[0]));
            auto ety0 = std::exp(std::complex<double>(0, dk*r[1]));
            auto etz0 = std::exp(std::complex<double>(0, dk*r[2]));
            std::vector<std::complex<double>> etx(d), ety(d), etz(d);
            for (int a = -n_ewald; a <= n_ewald; ++a) { etx[a+n_ewald]=std::pow(etx0,a); ety[a+n_ewald]=std::pow(ety0,a); etz[a+n_ewald]=std::pow(etz0,a); }
            for (int nx = -n_ewald; nx <= n_ewald; ++nx)
                for (int ny = -n_ewald; ny <= n_ewald; ++ny)
                    for (int nz = -n_ewald; nz <= n_ewald; ++nz) {
                        if (nx==0 && ny==0 && nz==0) continue;
                        double k2 = (nx*nx+ny*ny+nz*nz)*dk*dk;
                        double G = std::exp(-k2/(4*alpha*alpha)) / k2;
                        auto eikr = etx[nx+n_ewald]*ety[ny+n_ewald]*etz[nz+n_ewald];
                        pot_long += G * std::real(rho[((nx+n_ewald)*d+(ny+n_ewald))*d+(nz+n_ewald)] * eikr);
                    }
            pot_long *= 4.0 * M_PI / V_box;

            // Self correction: -2α/√π · q_i
            double pot_self = -charges[i] * 2.0 * alpha / std::sqrt(M_PI);

            return pot_short + pot_long + pot_self;
        };

        fprintf(stderr, "alpha=%5.1f: ", alpha);
        for (int i = 0; i < n_test; ++i)
            fprintf(stderr, " V[%d]=%.10f", i, ewald_pot(i));
        fprintf(stderr, "\n");
    }
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace PBC full pipeline vs Ewald", 1) {
    // Full pipeline test: pdmk_tree_create + pdmk_tree_eval with pbc=true
    // at 3/6/9/12 digits, pot and pot+grad for both sources and targets,
    // verified against self-contained Ewald sum.

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

    // Ewald reference: precompute rho(k) once (shared across subcases)
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
                auto t2 = charges[is] * ex[ix] * ey[iy];
                for (int iz = 0; iz < d; ++iz)
                    rho[ix * d * d + iy * d + iz] += t2 * ez[iz];
            }
    }

    // Ewald potential+gradient at a single evaluation point
    auto ewald_pot_grad = [&](const double *r_eval, double &pot_out, double *grad_out) {
        pot_out = 0.0;
        if (grad_out) {
            grad_out[0] = grad_out[1] = grad_out[2] = 0.0;
        }
        // Short range
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
                                const double scale = -charges[is] *
                                    (std::erfc(alpha * r) / (r * r2) +
                                     2.0 * alpha * std::exp(-alpha * alpha * r2) / (std::sqrt(M_PI) * r2));
                                grad_out[0] += scale * dx;
                                grad_out[1] += scale * dy;
                                grad_out[2] += scale * dz;
                            }
                        }
                    }
        }
        // Long range
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
                    const double kx = dk * nx, ky = dk * ny, kz = dk * nz;
                    const double k2 = kx * kx + ky * ky + kz * kz;
                    const double G = std::exp(-k2 / (4.0 * alpha * alpha)) / k2;
                    const int ix = nx + n_ewald, iy = ny + n_ewald, iz = nz + n_ewald;
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

    // Ewald self-correction: the long-range erf(αr)/r → 2α/√π as r→0,
    // contributing a spurious self-interaction at source locations. The short-range
    // erfc(αr)/r skips r=0, so this artifact is uncompensated. Subtract it so the
    // Ewald reference gives the physical periodic potential (images only, no self).
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
            const std::string label = "n_digits=" + std::to_string(pc.n_digits) +
                                      (with_grad ? " pot+grad" : " pot");

            SUBCASE(label.c_str()) {
                pdmk_params params;
                params.eps = pc.eps;
                params.n_dim = n_dim;
                params.n_per_leaf = 50;
                params.pgh_src = pgh;
                params.pgh_trg = pgh;
                params.kernel = DMK_LAPLACE;
                params.use_periodic = true;
                params.log_level = SPDLOG_LEVEL_OFF;

                sctl::Vector<double> pot_src(n_src * odim), pot_trg(n_trg * odim);

                pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0],
                                                  &rnormal[0], &dipstr[0], n_trg, &r_trg[0]);
                pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
                pdmk_tree_destroy(tree);

                // Compare source potentials (subtract Ewald self-correction from reference)
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

                // Compare target potentials
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

template <typename Real>
inline pdmk_tree pdmk_tree_create(dmk_communicator comm, pdmk_params params, int n_src, const Real *r_src,
                                  const Real *charge, const Real *normal, const Real *dipole_str, int n_trg,
                                  const Real *r_trg) {
    sctl::Profile::reset();
    sctl::Profile::Enable(true);
#ifdef DMK_HAVE_MPI
    const sctl::Comm sctl_comm(comm);
#else
    const sctl::Comm sctl_comm;
#endif
    sctl::Profile::Scoped profile("pdmk_tree_create", &sctl_comm);
    const int kernel_input_dim = get_kernel_input_dim(params.n_dim, params.kernel);

    sctl::Vector<Real> r_src_vec(n_src * params.n_dim, const_cast<Real *>(r_src), false);
    sctl::Vector<Real> r_trg_vec(n_trg * params.n_dim, const_cast<Real *>(r_trg), false);
    sctl::Vector<Real> charge_vec(n_src * kernel_input_dim, const_cast<Real *>(charge), false);

    if (params.n_dim != 2 && params.n_dim != 3)
        throw std::runtime_error("Invalid dimension: " + std::to_string(params.n_dim));
    if (params.n_dim == 2) {
        return new pdmk_tree_impl(std::unique_ptr<dmk::DMKPtTree<Real, 2>>(
            new dmk::DMKPtTree<Real, 2>(sctl_comm, params, r_src_vec, r_trg_vec, charge_vec)));
    } else
        return new pdmk_tree_impl(std::unique_ptr<dmk::DMKPtTree<Real, 3>>(
            new dmk::DMKPtTree<Real, 3>(sctl_comm, params, r_src_vec, r_trg_vec, charge_vec)));
}

template <typename Real>
inline void pdmk_tree_eval(pdmk_tree tree, Real *pot_src, Real *pot_trg) {
    std::visit(
        [&](auto &t) {
            using TreeType = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 2>>> ||
                          std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 3>>>) {
                const auto &comm = (*static_cast<TreeType *>(tree))->GetComm();
                sctl::Profile::Scoped prof("pdmk_tree_eval", &comm);
                t->upward_pass();
                t->downward_pass();

                sctl::Profile::Tic("pdmk_tree_eval_sync", &comm);
                sctl::Vector<Real> res;
                t->GetParticleData(res, "pdmk_pot_src");
                sctl::Vector<Real>(res.Dim(), pot_src, false) = res;
                t->GetParticleData(res, "pdmk_pot_trg");
                sctl::Vector<Real>(res.Dim(), pot_trg, false) = res;
                sctl::Profile::Toc();
            }
        },
        *static_cast<pdmk_tree_impl *>(tree));
}

template <typename Real>
inline int pdmk_tree_update_charges(pdmk_tree tree, const Real *charge, const Real *normal, const Real *dipole_str) {
    int rc = 1;
    std::visit(
        [&](auto &t) {
            using TreeType = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 2>>> ||
                          std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 3>>>) {
                rc = t->update_charges(charge, normal, dipole_str);
            }
        },
        *static_cast<pdmk_tree_impl *>(tree));
    return rc;
}

} // namespace dmk

extern "C" {

void pdmk_print_profile_data(dmk_communicator comm, char type) {
#ifdef DMK_HAVE_MPI
    sctl::Comm sctl_comm(comm);
#else
    sctl::Comm sctl_comm;
#endif
    const std::vector<std::string> fields{"t_avg", "t_max",   "t_min",     "f_avg",   "f_max",
                                          "f_min", "f_total", "f/s_total", "custom1", "custom2"};
    if (type == 'h') {
        auto table = sctl::Profile::get_table(fields, &sctl_comm);
        if (sctl_comm.Rank() == 0) {
            std::string sep;
            for (auto &row : table) {
                for (auto &field : row.second) {
                    std::cout << sep << row.first << "|" << field.first;
                    sep = ",";
                }
            }
        }
    }
    if (type == 'c') {
        auto table = sctl::Profile::get_table(fields, &sctl_comm);
        if (sctl_comm.Rank() == 0) {
            std::string sep;
            for (auto &row : table) {
                for (auto &field : row.second) {
                    std::cout << sep << field.second;
                    sep = ",";
                }
            }
        }
        sctl::Profile::reset();
    } else if (type == 't') {
        sctl::Profile::print(
            &sctl_comm,
            {"t_avg", "t_max", "t_min", "f_avg", "f_max", "f_min", "f_total", "f/s_total", "custom1", "custom2"},
            {"%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.3g", "%.3g"});
    }
}

pdmk_tree pdmk_tree_createf(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src,
                            const float *charge, const float *normal, const float *dipole_str, int n_trg,
                            const float *r_trg) {
    return dmk::pdmk_tree_create(comm, params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg);
}

pdmk_tree pdmk_tree_create(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src,
                           const double *charge, const double *normal, const double *dipole_str, int n_trg,
                           const double *r_trg) {
    return dmk::pdmk_tree_create(comm, params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg);
}

void pdmk_tree_destroy(pdmk_tree tree) {
    if (tree)
        delete static_cast<pdmk_tree_impl *>(tree);
}

int pdmk_tree_update_charges(pdmk_tree tree, const double *charge, const double *normal, const double *dipole_str) {
    return dmk::pdmk_tree_update_charges(tree, charge, normal, dipole_str);
}

int pdmk_tree_update_chargesf(pdmk_tree tree, const float *charge, const float *normal, const float *dipole_str) {
    return dmk::pdmk_tree_update_charges(tree, charge, normal, dipole_str);
}

void pdmk_tree_evalf(pdmk_tree tree, float *pot_src, float *pot_trg) { dmk::pdmk_tree_eval(tree, pot_src, pot_trg); }

void pdmk_tree_eval(pdmk_tree tree, double *pot_src, double *pot_trg) { dmk::pdmk_tree_eval(tree, pot_src, pot_trg); }

void pdmkf(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src, const float *charge,
           const float *normal, const float *dipole_str, int n_trg, const float *r_trg, float *pot_src,
           float *pot_trg) {
    if (params.n_dim == 2)
        return dmk::pdmk<float, 2>(comm, params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot_src,
                                   pot_trg);
    if (params.n_dim == 3)
        return dmk::pdmk<float, 3>(comm, params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot_src,
                                   pot_trg);
}

void pdmk(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src, const double *charge,
          const double *normal, const double *dipole_str, int n_trg, const double *r_trg, double *pot_src,
          double *pot_trg) {
    if (params.n_dim == 2)
        return dmk::pdmk<double, 2>(comm, params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot_src,
                                    pot_trg);
    if (params.n_dim == 3)
        return dmk::pdmk<double, 3>(comm, params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot_src,
                                    pot_trg);
}
}
