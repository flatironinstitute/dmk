#include <algorithm>
#include <limits>
#include <span>
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
          int n_trg, const T *r_trg, T *pot_src, T *pot_trg) {
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
    sctl::Vector<T> normal_vec(n_src * params.n_dim, const_cast<T *>(normal), false);

    DMKPtTree<T, DIM> tree(sctl_comm, params, r_src_vec, charge_vec, normal_vec, r_trg_vec);
    tree.eval();

    tree.desort_potentials(pot_src, pot_trg);
    if (params.log_level <= DMK_LOG_INFO) {
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

    sctl::Vector<double> r_src, pot_src, charges, rnormal, pot_trg, r_trg;
    sctl::Vector<float> r_srcf, pot_srcf, chargesf, rnormalf, pot_trgf, r_trgf;
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, 0);
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_srcf, r_trgf, rnormalf, chargesf,
                              0);
    pot_src.ReInit(n_src * nd);
    pot_trg.ReInit(n_trg * nd);
    pot_srcf.ReInit(n_src * nd);
    pot_trgf.ReInit(n_trg * nd);

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 80;
    params.eval_src = DMK_POTENTIAL;
    params.eval_trg = DMK_POTENTIAL;
    params.kernel = DMK_YUKAWA;
    params.fparam = 6.0;
    params.log_level = SPDLOG_LEVEL_OFF;

    pdmk(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0], &pot_src[0], &pot_trg[0]);

    params.eps = 1e-3;
    pdmkf(comm, params, n_src, &r_srcf[0], &chargesf[0], &rnormalf[0], n_trg, &r_trgf[0], &pot_srcf[0], &pot_trgf[0]);

    double l2_err_src{0.0}, l2_err_trg{0.0}, src2{0.0}, trg2{0.0};
    for (int i = 0; i < n_src; ++i) {
        l2_err_src += pot_src[i] ? sctl::pow<2>(pot_src[i] - pot_srcf[i]) : 0.0;
        src2 += pot_src[i] * pot_src[i];
    }
    for (int i = 0; i < n_trg; ++i) {
        l2_err_trg += pot_trg[i] ? sctl::pow<2>(pot_trg[i] - pot_trgf[i]) : 0.0;
        trg2 += pot_trg[i] * pot_trg[i];
    }

    l2_err_src = std::sqrt(l2_err_src / src2);
    l2_err_trg = std::sqrt(l2_err_trg / trg2);
    CHECK(l2_err_src < params.eps);
    CHECK(l2_err_trg < params.eps);
}

TEST_CASE_GENERIC("[DMK] pdmk all", 1) {
    constexpr int n_src = 10000;
    constexpr int nd = 1;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    pdmk_params params;
    params.eps = 1e-6;
    params.eval_src = DMK_POTENTIAL;
    params.eval_trg = DMK_POTENTIAL;
    params.fparam = 6.0;
    params.log_level = SPDLOG_LEVEL_OFF;
    int ndiv[3] = {80, 280, 280};

    const auto test_kernels = {
        DMK_YUKAWA,
        DMK_LAPLACE,
        DMK_SQRT_LAPLACE,
    };

    for (auto n_dim : {2, 3}) {
        params.n_dim = n_dim;
        std::vector<double> r_src, pot_src, charges, rnormal, pot_trg, r_trg;
        dmk::util::init_test_data(n_dim, 1, n_src, 0, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, 0);
        r_trg = r_src;
        std::reverse(r_trg.begin(), r_trg.end());
        r_trg.resize(n_dim * (n_src - set_fixed_charges * 3));
        const int n_trg = r_trg.size() / n_dim;

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
                std::vector<double> pot_src(n_src * nd), grad_src(n_src * nd * n_dim),
                    hess_src(n_src * nd * n_dim * n_dim), pot_trg(n_src * nd), grad_trg(n_trg * nd * n_dim),
                    hess_trg(n_trg * nd * n_dim * n_dim);
                params.n_per_leaf = ndiv[int(kernel)];

                params.kernel = kernel;

                const int n_test_src = std::min(n_src, 1000);
                const int n_test_trg = std::min(n_trg, 1000);
                std::vector<double> test_src(n_test_src, 0);
                std::vector<double> test_trg(n_test_trg, 0);
                std::span<const double> r_src_trunc(r_src.data(), n_test_src * n_dim);
                std::span<const double> r_trg_trunc(r_trg.data(), n_test_trg * n_dim);

                compute_direct(n_dim, r_src, charges, std::vector<double>{}, r_src_trunc, test_src, kernel,
                               DMK_POTENTIAL);
                compute_direct(n_dim, r_src, charges, std::vector<double>{}, r_trg_trunc, test_trg, kernel,
                               DMK_POTENTIAL);

                pdmk_tree tree =
                    pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
                pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);

                double err_src{0}, err_trg{0};
                double ref_src{0}, ref_trg{0};
                for (int i = 0; i < n_test_src; ++i) {
                    err_src += sctl::pow<2>(test_src[i] - pot_src[i]);
                    ref_src += sctl::pow<2>(test_src[i]);
                }
                for (int i = 0; i < n_test_trg; ++i) {
                    err_trg += sctl::pow<2>(test_trg[i] - pot_trg[i]);
                    ref_trg += sctl::pow<2>(test_trg[i]);
                }

                err_src = std::sqrt(err_src / ref_src);
                err_trg = std::sqrt(err_trg / ref_trg);

                CHECK(err_src < params.eps);
                CHECK(err_trg < params.eps);

                // Scale charges by 1/2 and re-evaluate. Since the kernel
                // is linear in the charges, potentials should scale by the same factor.
                {
                    const double scale = 2.0;
                    sctl::Vector<double> scaled_charges(charges.size());
                    for (sctl::Long i = 0; i < charges.size(); ++i)
                        scaled_charges[i] = charges[i] * scale;

                    int rc = pdmk_tree_update_charges(tree, &scaled_charges[0], nullptr);
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

TEST_CASE_GENERIC("[DMK] pdmk 3d stokeslet velocity", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 2000;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;
    constexpr double thresh2 = 1e-30;
    constexpr int output_dim = n_dim;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    std::vector<double> r_src, charges, rnormal, r_trg;
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, 0);
    charges.resize(n_src * n_dim);
    for (auto &c : charges)
        c = 2 * drand48() - 1.0;

    std::vector<double> vel_src(n_src * output_dim, 0), vel_trg(n_trg * output_dim, 0);

    pdmk_params params;
    params.eps = 1e-3;
    params.n_dim = n_dim;
    params.n_per_leaf = 280;
    params.eval_src = DMK_VELOCITY;
    params.eval_trg = DMK_VELOCITY;
    params.kernel = DMK_STOKESLET;
    params.log_level = SPDLOG_LEVEL_OFF;
    params.debug_flags = 0;

    pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
    pdmk_tree_eval(tree, &vel_src[0], &vel_trg[0]);
    pdmk_tree_destroy(tree);

    const int n_test_src = std::min(n_src, 64);
    const int n_test_trg = std::min(n_trg, 64);
    std::vector<double> vel_src_direct, vel_trg_direct;

    compute_direct(n_dim, r_src, charges, std::vector<double>{}, std::span<double>(r_src.data(), n_test_src * n_dim),
                   vel_src_direct, params.kernel, params.eval_trg);
    compute_direct(n_dim, r_src, charges, std::vector<double>{}, std::span<double>(r_trg.data(), n_test_trg * n_dim),
                   vel_trg_direct, params.kernel, params.eval_trg);

    auto relative_l2_error = [](const auto &approx, const auto &exact) {
        double err2{0.0}, ref2{0.0};
        for (int i = 0; i < exact.size(); ++i) {
            err2 += sctl::pow<2>(approx[i] - exact[i]);
            ref2 += sctl::pow<2>(exact[i]);
        }
        return std::sqrt(err2 / ref2);
    };

    const double l2_err_src = relative_l2_error(vel_src, vel_src_direct);
    const double l2_err_trg = relative_l2_error(vel_trg, vel_trg_direct);

    CHECK(l2_err_src < params.eps);
    CHECK(l2_err_trg < params.eps);
}

TEST_CASE_GENERIC("[DMK] pdmk 3d stresslet velocity", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 2000;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;
    constexpr double thresh2 = 1e-30;
    constexpr int output_dim = n_dim;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    std::vector<double> r_src, charges, rnormal, r_trg;
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, 0);
    charges.resize(n_src * n_dim);
    for (auto &c : charges)
        c = 2 * drand48() - 1.0;

    std::vector<double> vel_src(n_src * output_dim, 0), vel_trg(n_trg * output_dim, 0);

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 280;
    params.eval_src = DMK_VELOCITY;
    params.eval_trg = DMK_VELOCITY;
    params.kernel = DMK_STRESSLET;
    params.log_level = SPDLOG_LEVEL_OFF;
    params.debug_flags = 0;
    params.use_periodic = false;

    pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
    pdmk_tree_eval(tree, &vel_src[0], &vel_trg[0]);
    pdmk_tree_destroy(tree);

    const int n_test_src = std::min(n_src, 64);
    const int n_test_trg = std::min(n_trg, 64);
    std::vector<double> vel_src_direct, vel_trg_direct;

    compute_direct(n_dim, r_src, charges, rnormal, std::span<double>(r_src.data(), n_test_src * n_dim), vel_src_direct,
                   params.kernel, params.eval_trg);
    compute_direct(n_dim, r_src, charges, rnormal, std::span<double>(r_trg.data(), n_test_trg * n_dim), vel_trg_direct,
                   params.kernel, params.eval_trg);

    auto relative_l2_error = [](const auto &approx, const auto &exact) {
        double err2{0.0}, ref2{0.0};
        for (int i = 0; i < exact.size(); ++i) {
            err2 += sctl::pow<2>(approx[i] - exact[i]);
            ref2 += sctl::pow<2>(exact[i]);
        }
        return std::sqrt(err2 / ref2);
    };

    const double l2_err_src = relative_l2_error(vel_src, vel_src_direct);
    const double l2_err_trg = relative_l2_error(vel_trg, vel_trg_direct);

    CHECK(l2_err_src < params.eps);
    CHECK(l2_err_trg < params.eps);
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

    sctl::Vector<double> r_src, charges, rnormal, r_trg;
    dmk::util::init_test_data(n_dim, nd, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, 0);

    sctl::Vector<double> pot_src(n_src * output_dim), pot_trg(n_trg * output_dim);
    pot_src.SetZero();
    pot_trg.SetZero();

    pdmk_params params;
    params.eps = 1e-7;
    params.n_dim = n_dim;
    params.n_per_leaf = 280;
    params.eval_src = DMK_POTENTIAL_GRAD;
    params.eval_trg = DMK_POTENTIAL_GRAD;
    params.kernel = DMK_LAPLACE;
    params.log_level = SPDLOG_LEVEL_OFF;

    pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
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

template <typename Real>
inline pdmk_tree pdmk_tree_create(dmk_communicator comm, pdmk_params params, int n_src, const Real *r_src,
                                  const Real *charge, const Real *normal, int n_trg, const Real *r_trg) {
    sctl::Profile::reset();
    sctl::Profile::Enable(true);
#ifdef DMK_HAVE_MPI
    const sctl::Comm sctl_comm(comm);
#else
    const sctl::Comm sctl_comm;
#endif
    sctl::Profile::Scoped profile("pdmk_tree_create", &sctl_comm);
    const int charge_dim =
        params.kernel == DMK_STRESSLET ? params.n_dim : get_kernel_input_dim(params.n_dim, params.kernel);

    sctl::Vector<Real> r_src_vec(n_src * params.n_dim, const_cast<Real *>(r_src), false);
    sctl::Vector<Real> r_trg_vec(n_trg * params.n_dim, const_cast<Real *>(r_trg), false);
    sctl::Vector<Real> charge_vec(n_src * charge_dim, const_cast<Real *>(charge), false);
    sctl::Vector<Real> normal_vec(n_src * params.n_dim, const_cast<Real *>(normal), false);

    if (params.n_dim != 2 && params.n_dim != 3)
        throw std::runtime_error("Invalid dimension: " + std::to_string(params.n_dim));
    if (params.n_dim == 2) {
        return new pdmk_tree_impl(std::unique_ptr<dmk::DMKPtTree<Real, 2>>(
            new dmk::DMKPtTree<Real, 2>(sctl_comm, params, r_src_vec, charge_vec, normal_vec, r_trg_vec)));
    } else
        return new pdmk_tree_impl(std::unique_ptr<dmk::DMKPtTree<Real, 3>>(
            new dmk::DMKPtTree<Real, 3>(sctl_comm, params, r_src_vec, charge_vec, normal_vec, r_trg_vec)));
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

                t->eval();
                t->desort_potentials(pot_src, pot_trg);
            }
        },
        *static_cast<pdmk_tree_impl *>(tree));
}

template <typename Real>
inline int pdmk_tree_update_charges(pdmk_tree tree, const Real *charge, const Real *normal) {
    int rc = 1;
    std::visit(
        [&](auto &t) {
            using TreeType = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 2>>> ||
                          std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 3>>>) {
                rc = t->update_charges(charge, normal);
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
                            const float *charge, const float *normal, int n_trg, const float *r_trg) {
    return dmk::pdmk_tree_create(comm, params, n_src, r_src, charge, normal, n_trg, r_trg);
}

pdmk_tree pdmk_tree_create(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src,
                           const double *charge, const double *normal, int n_trg, const double *r_trg) {
    return dmk::pdmk_tree_create(comm, params, n_src, r_src, charge, normal, n_trg, r_trg);
}

void pdmk_tree_destroy(pdmk_tree tree) {
    if (tree)
        delete static_cast<pdmk_tree_impl *>(tree);
}

int pdmk_tree_update_charges(pdmk_tree tree, const double *charge, const double *normal) {
    return dmk::pdmk_tree_update_charges(tree, charge, normal);
}

int pdmk_tree_update_chargesf(pdmk_tree tree, const float *charge, const float *normal) {
    return dmk::pdmk_tree_update_charges(tree, charge, normal);
}

void pdmk_tree_evalf(pdmk_tree tree, float *pot_src, float *pot_trg) { dmk::pdmk_tree_eval(tree, pot_src, pot_trg); }

void pdmk_tree_eval(pdmk_tree tree, double *pot_src, double *pot_trg) { dmk::pdmk_tree_eval(tree, pot_src, pot_trg); }

void pdmkf(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src, const float *charge,
           const float *normal, int n_trg, const float *r_trg, float *pot_src, float *pot_trg) {
    if (params.n_dim == 2)
        return dmk::pdmk<float, 2>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
    if (params.n_dim == 3)
        return dmk::pdmk<float, 3>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
}

void pdmk(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src, const double *charge,
          const double *normal, int n_trg, const double *r_trg, double *pot_src, double *pot_trg) {
    if (params.n_dim == 2)
        return dmk::pdmk<double, 2>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
    if (params.n_dim == 3)
        return dmk::pdmk<double, 3>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
}
}
