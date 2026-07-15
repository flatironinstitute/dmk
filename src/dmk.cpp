#include <algorithm>
#include <limits>
#include <span>
#include <string>
#include <variant>

#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/direct.hpp>
#include <dmk/error.hpp>
#include <dmk/esp.hpp>
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

#ifdef DMK_BUILD_ESP
using pdmk_esp_plan_impl = std::variant<std::unique_ptr<dmk::EspPlan<float>>, std::unique_ptr<dmk::EspPlan<double>>>;
#endif

namespace dmk {

namespace {
std::string &last_error_buffer() {
    static thread_local std::string buf;
    return buf;
}
} // namespace

void set_last_error(const std::string &msg) { last_error_buffer() = msg; }

const char *last_error_message() { return last_error_buffer().c_str(); }

/// Validate C API inputs before any object is constructed. Throws api_error
/// (DMK_ERR_INVALID_ARGUMENT) so the boundary guard converts to a clean code.
template <typename Real>
void validate_create_args(const pdmk_params &params, int n_src, const Real *r_src, const Real *charge,
                          const Real *normal, int n_trg, const Real *r_trg) {
    auto fail = [](std::string msg) { throw api_error(DMK_ERR_INVALID_ARGUMENT, std::move(msg)); };

    if (params.n_dim != 2 && params.n_dim != 3)
        fail("Invalid dimension: " + std::to_string(params.n_dim));
    if (params.eps > 1e-2 || params.eps < 1e-12)
        fail("tolerance 'eps' must lie on [1e-12, 1e-2], got " + std::to_string(params.eps));
    if (params.n_per_leaf <= 0)
        fail("n_per_leaf must be positive, got " + std::to_string(params.n_per_leaf));
    if (n_src < 0 || n_trg < 0)
        fail("n_src and n_trg must be non-negative");
    if (n_src == 0)
        fail("n_src is zero: nothing to do");
    if (params.kernel < DMK_YUKAWA || params.kernel > DMK_LAPLACE_DIPOLE)
        fail("Invalid kernel: " + std::to_string(int(params.kernel)));
    if (params.kernel == DMK_YUKAWA && params.fparam <= 0.0)
        fail("Invalid yukawa lambda. lambda must be positive, got " + std::to_string(params.fparam));
    if (params.eval_src < DMK_POTENTIAL || params.eval_src > DMK_VELOCITY_PRESSURE)
        fail("Invalid eval_src: " + std::to_string(int(params.eval_src)));
    if (params.eval_trg < DMK_POTENTIAL || params.eval_trg > DMK_VELOCITY_PRESSURE)
        fail("Invalid eval_trg: " + std::to_string(int(params.eval_trg)));

    // Stokeslet/Stresslet/Laplace-dipole only have 3D evaluators currently
    const bool needs_3d =
        params.kernel == DMK_STOKESLET || params.kernel == DMK_STRESSLET || params.kernel == DMK_LAPLACE_DIPOLE;
    if (needs_3d && params.n_dim != 3)
        fail("kernel " + std::string(util::to_string(params.kernel)) + " is only supported in 3D");

    // Reject unsupported kernel/eval-type combinations
    try {
        get_kernel_output_dim(params.n_dim, params.kernel, params.eval_src);
        get_kernel_output_dim(params.n_dim, params.kernel, params.eval_trg);
    } catch (const std::exception &e) {
        fail(e.what());
    }

    if (r_src == nullptr || charge == nullptr)
        fail("r_src and charge must be non-null");
    if (n_trg > 0 && r_trg == nullptr)
        fail("r_trg must be non-null when n_trg > 0");
    // Stresslet reads a per-source normal in build_tree; a null pointer there is
    // an uncatchable segfault rather than an exception.
    if (params.kernel == DMK_STRESSLET && normal == nullptr)
        fail("Stresslet requires a non-null normal array");
}

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
            const std::string kernel_str(util::to_string(kernel));

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

                    dmk_error rc = pdmk_tree_update_charges(tree, &scaled_charges[0], nullptr);
                    CHECK(rc == DMK_SUCCESS);

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

TEST_CASE_GENERIC("[DMK] pdmk 3d stresslet update_charges", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 2000;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;
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

    pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
    REQUIRE(tree != nullptr);
    pdmk_tree_eval(tree, &vel_src[0], &vel_trg[0]);

    // Stresslet is linear in the density for a fixed normal: scaling the density
    // (passing the same normal) must scale the velocity by the same factor. This
    // exercises the stresslet branch of update_charges (charge .outer. normal
    // rebuild + halo broadcast).
    const double scale = 2.0;
    std::vector<double> scaled_charges(charges.size());
    for (size_t i = 0; i < charges.size(); ++i)
        scaled_charges[i] = charges[i] * scale;

    REQUIRE(pdmk_tree_update_charges(tree, scaled_charges.data(), rnormal.data()) == DMK_SUCCESS);

    std::vector<double> vel_src_updated(n_src * output_dim, 0), vel_trg_updated(n_trg * output_dim, 0);
    pdmk_tree_eval(tree, &vel_src_updated[0], &vel_trg_updated[0]);

    auto relative_l2_error = [](const auto &approx, const auto &exact) {
        double err2{0.0}, ref2{0.0};
        for (size_t i = 0; i < exact.size(); ++i) {
            err2 += sctl::pow<2>(approx[i] - exact[i]);
            ref2 += sctl::pow<2>(exact[i]);
        }
        return std::sqrt(err2 / ref2);
    };

    std::vector<double> vel_src_expected(vel_src.size()), vel_trg_expected(vel_trg.size());
    for (size_t i = 0; i < vel_src.size(); ++i)
        vel_src_expected[i] = vel_src[i] * scale;
    for (size_t i = 0; i < vel_trg.size(); ++i)
        vel_trg_expected[i] = vel_trg[i] * scale;

    CHECK(relative_l2_error(vel_src_updated, vel_src_expected) < 1e-12);
    CHECK(relative_l2_error(vel_trg_updated, vel_trg_expected) < 1e-12);

    // A stresslet charge update with a null normal must be rejected, not crash.
    CHECK(pdmk_tree_update_charges(tree, scaled_charges.data(), nullptr) == DMK_ERR_INVALID_ARGUMENT);

    pdmk_tree_destroy(tree);
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

TEST_CASE_GENERIC("[DMK] pdmk Laplace dipole", 1) {
    constexpr int n_src = 4000;
    constexpr int n_trg = 3000;
    constexpr bool uniform = true;
    constexpr bool set_fixed_charges = false;
    constexpr double thresh2 = 1e-30;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    for (int n_dim : {3}) {
        const int nd = n_dim; // dipole strength components per source

        sctl::Vector<double> r_src, dipoles, rnormal, r_trg;
        dmk::util::init_test_data(n_dim, nd, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, dipoles,
                                  0);

        sctl::Vector<double> pot_src(n_src), pot_trg(n_trg);
        pot_src.SetZero();
        pot_trg.SetZero();

        pdmk_params params;
        params.eps = 1e-6;
        params.n_dim = n_dim;
        params.n_per_leaf = 280;
        params.eval_src = DMK_POTENTIAL;
        params.eval_trg = DMK_POTENTIAL;
        params.kernel = DMK_LAPLACE_DIPOLE;
        params.log_level = SPDLOG_LEVEL_OFF;

        pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &dipoles[0], &rnormal[0], n_trg, &r_trg[0]);
        pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
        pdmk_tree_destroy(tree);

        const int n_test_src = std::min(n_src, 64);
        const int n_test_trg = std::min(n_trg, 64);
        std::vector<double> direct_pot_src(n_test_src, 0.0);
        std::vector<double> direct_pot_trg(n_test_trg, 0.0);

        const auto accumulate_dipole = [&](const double *target, int i_out, std::vector<double> &out) {
            for (int i_src = 0; i_src < n_src; ++i_src) {
                double dx_arr[3] = {0, 0, 0};
                double dr2 = 0.0;
                for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                    dx_arr[i_dim] = target[i_dim] - r_src[i_src * n_dim + i_dim];
                    dr2 += dx_arr[i_dim] * dx_arr[i_dim];
                }
                if (dr2 <= thresh2)
                    continue;

                double dot = 0.0;
                for (int i_dim = 0; i_dim < n_dim; ++i_dim)
                    dot += dipoles[i_src * nd + i_dim] * dx_arr[i_dim];

                if (n_dim == 3) {
                    const double rinv = 1.0 / std::sqrt(dr2);
                    const double rinv3 = rinv / dr2;
                    out[i_out] += dot * rinv3;
                } else {
                    // 2D: phi = d . grad_s log(R) = -(d.dX)/R^2
                    out[i_out] -= dot / dr2;
                }
            }
        };

        for (int i = 0; i < n_test_src; ++i)
            accumulate_dipole(&r_src[i * n_dim], i, direct_pot_src);
        for (int i = 0; i < n_test_trg; ++i)
            accumulate_dipole(&r_trg[i * n_dim], i, direct_pot_trg);

        auto relative_l2_error = [](const auto &approx, const auto &exact) {
            double err2 = 0.0, ref2 = 0.0;
            for (int i = 0; i < (int)exact.size(); ++i) {
                err2 += sctl::pow<2>(approx[i] - exact[i]);
                ref2 += sctl::pow<2>(exact[i]);
            }
            return std::sqrt(err2 / ref2);
        };

        std::vector<double> pot_src_prefix(direct_pot_src.size()), pot_trg_prefix(direct_pot_trg.size());
        for (int i = 0; i < n_test_src; ++i)
            pot_src_prefix[i] = pot_src[i];
        for (int i = 0; i < n_test_trg; ++i)
            pot_trg_prefix[i] = pot_trg[i];

        const double l2_err_src = relative_l2_error(pot_src_prefix, direct_pot_src);
        const double l2_err_trg = relative_l2_error(pot_trg_prefix, direct_pot_trg);

        CHECK(l2_err_src < 10 * params.eps);
        CHECK(l2_err_trg < 10 * params.eps);
    }
}

TEST_CASE_GENERIC("[DMK] pdmk Laplace dipole gradient", 1) {
    constexpr int n_src = 10000;
    constexpr int n_trg = 10000;
    constexpr bool uniform = true;
    constexpr bool set_fixed_charges = false;
    constexpr double thresh2 = 1e-30;
    constexpr int n_to_compare = 10000;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    for (int n_dim : {3}) {
        const int nd = n_dim; // dipole strength components per source
        const int output_dim = 1 + n_dim;

        sctl::Vector<double> r_src, dipoles, rnormal, r_trg;
        dmk::util::init_test_data(n_dim, nd, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, dipoles,
                                  0);

        sctl::Vector<double> pot_src(n_src * output_dim), pot_trg(n_trg * output_dim);
        pot_src.SetZero();
        pot_trg.SetZero();

        pdmk_params params;
        params.eps = 1e-6;
        params.n_dim = n_dim;
        params.n_per_leaf = 280;
        params.eval_src = DMK_POTENTIAL_GRAD;
        params.eval_trg = DMK_POTENTIAL_GRAD;
        params.kernel = DMK_LAPLACE_DIPOLE;
        params.log_level = SPDLOG_LEVEL_OFF;

        pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &dipoles[0], &rnormal[0], n_trg, &r_trg[0]);
        pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
        pdmk_tree_destroy(tree);

        const int n_test_src = std::min(n_src, n_to_compare);
        const int n_test_trg = std::min(n_trg, n_to_compare);
        std::vector<double> direct_src(n_test_src * output_dim, 0.0);
        std::vector<double> direct_trg(n_test_trg * output_dim, 0.0);

        const auto accumulate_dipole_grad = [&](const double *target, int i_out, std::vector<double> &out) {
            for (int i_src = 0; i_src < n_src; ++i_src) {
                double dx_arr[3] = {0, 0, 0};
                double dr2 = 0.0;
                for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                    dx_arr[i_dim] = target[i_dim] - r_src[i_src * n_dim + i_dim];
                    dr2 += dx_arr[i_dim] * dx_arr[i_dim];
                }
                if (dr2 <= thresh2)
                    continue;

                double dot = 0.0;
                for (int i_dim = 0; i_dim < n_dim; ++i_dim)
                    dot += dipoles[i_src * nd + i_dim] * dx_arr[i_dim];

                if (n_dim == 3) {
                    const double rinv = 1.0 / std::sqrt(dr2);
                    const double rinv3 = rinv / dr2;
                    const double rinv5 = rinv3 / dr2;
                    out[i_out * output_dim + 0] += dot * rinv3;
                    for (int i = 0; i < 3; ++i)
                        out[i_out * output_dim + 1 + i] +=
                            dipoles[i_src * nd + i] * rinv3 - 3.0 * dot * dx_arr[i] * rinv5;
                } else {
                    const double r2inv = 1.0 / dr2;
                    const double r4inv = r2inv * r2inv;
                    out[i_out * output_dim + 0] -= dot * r2inv;
                    for (int i = 0; i < 2; ++i)
                        out[i_out * output_dim + 1 + i] +=
                            -dipoles[i_src * nd + i] * r2inv + 2.0 * dot * dx_arr[i] * r4inv;
                }
            }
        };

        for (int i = 0; i < n_test_src; ++i)
            accumulate_dipole_grad(&r_src[i * n_dim], i, direct_src);
        for (int i = 0; i < n_test_trg; ++i)
            accumulate_dipole_grad(&r_trg[i * n_dim], i, direct_trg);

        // Compute relative L2 error over a component range [comp_begin, comp_end),
        auto relative_l2_error_range = [&](const sctl::Vector<double> &approx, const std::vector<double> &exact,
                                           int n_pts, int comp_begin, int comp_end) {
            double err2 = 0.0, ref2 = 0.0;
            for (int i = 0; i < n_pts; ++i) {
                for (int c = comp_begin; c < comp_end; ++c) {
                    const double a = approx[i * output_dim + c];
                    const double e = exact[i * output_dim + c];
                    err2 += sctl::pow<2>(a - e);
                    ref2 += sctl::pow<2>(e);
                }
            }
            return std::sqrt(err2 / ref2);
        };

        const double pot_err_src = relative_l2_error_range(pot_src, direct_src, n_test_src, 0, 1);
        const double grad_err_src = relative_l2_error_range(pot_src, direct_src, n_test_src, 1, output_dim);
        const double pot_err_trg = relative_l2_error_range(pot_trg, direct_trg, n_test_trg, 0, 1);
        const double grad_err_trg = relative_l2_error_range(pot_trg, direct_trg, n_test_trg, 1, output_dim);

        CHECK(pot_err_src < params.eps);
        CHECK(grad_err_src < params.eps);
        CHECK(pot_err_trg < params.eps);
        CHECK(grad_err_trg < params.eps);
    }
}

TEST_CASE_GENERIC("[DMK] error handling", 1) {
#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif
    constexpr int n_dim = 3;
    constexpr int n_src = 1000;

    sctl::Vector<double> r_src, charges, rnormal, r_trg;
    dmk::util::init_test_data(n_dim, 1, n_src, 0, true, false, r_src, r_trg, rnormal, charges, 0);

    pdmk_params params;
    pdmk_init_default_params(&params);
    params.n_dim = n_dim;
    params.kernel = DMK_LAPLACE;
    params.log_level = SPDLOG_LEVEL_OFF;

    SUBCASE("bad dimension returns NULL with message") {
        pdmk_params bad = params;
        bad.n_dim = 5;
        pdmk_tree tree = pdmk_tree_create(comm, bad, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr);
        CHECK(tree == nullptr);
        CHECK(std::string(pdmk_last_error_message()).size() > 0);
    }

    SUBCASE("negative n_src returns NULL") {
        pdmk_tree tree = pdmk_tree_create(comm, params, -1, &r_src[0], &charges[0], nullptr, 0, nullptr);
        CHECK(tree == nullptr);
    }

    SUBCASE("unsupported 2D Stokeslet returns NULL, not an abort") {
        pdmk_params bad = params;
        bad.n_dim = 2;
        bad.kernel = DMK_STOKESLET;
        pdmk_tree tree = pdmk_tree_create(comm, bad, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr);
        CHECK(tree == nullptr);
    }

    SUBCASE("velocity eval on a scalar kernel returns NULL") {
        pdmk_params bad = params;
        bad.eval_trg = DMK_VELOCITY;
        pdmk_tree tree = pdmk_tree_create(comm, bad, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr);
        CHECK(tree == nullptr);
    }

    SUBCASE("velocity-pressure eval is unsupported and returns NULL") {
        pdmk_params bad = params;
        bad.n_dim = 3;
        bad.kernel = DMK_STOKESLET;
        bad.eval_src = DMK_VELOCITY_PRESSURE;
        bad.eval_trg = DMK_VELOCITY_PRESSURE;
        std::vector<double> stokes_charges(n_src * n_dim, 1.0);
        pdmk_tree tree = pdmk_tree_create(comm, bad, n_src, &r_src[0], stokes_charges.data(), nullptr, 0, nullptr);
        CHECK(tree == nullptr);
    }

    SUBCASE("Stresslet without a normal returns NULL, not a segfault") {
        pdmk_params bad = params;
        bad.n_dim = 3;
        bad.kernel = DMK_STRESSLET;
        bad.eval_src = DMK_VELOCITY;
        bad.eval_trg = DMK_VELOCITY;
        std::vector<double> stokes_charges(n_src * n_dim, 1.0);
        pdmk_tree tree = pdmk_tree_create(comm, bad, n_src, &r_src[0], stokes_charges.data(), nullptr, 0, nullptr);
        CHECK(tree == nullptr);
    }

    SUBCASE("null charge with n_src > 0 returns NULL") {
        pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], nullptr, nullptr, 0, nullptr);
        CHECK(tree == nullptr);
    }

    SUBCASE("null tree handle to eval is rejected") {
        std::vector<double> pot_src(n_src);
        CHECK(pdmk_tree_eval(nullptr, &pot_src[0], nullptr) == DMK_ERR_INVALID_ARGUMENT);
    }

    SUBCASE("normal is ignored when updating charges for a scalar kernel") {
        std::vector<double> pot_src(n_src);
        pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr);
        REQUIRE(tree != nullptr);
        CHECK(pdmk_tree_eval(tree, &pot_src[0], nullptr) == DMK_SUCCESS);
        // Laplace has no normal; passing one is harmless and the update succeeds.
        CHECK(pdmk_tree_update_charges(tree, &charges[0], &charges[0]) == DMK_SUCCESS);
        pdmk_tree_destroy(tree);
    }
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
        throw api_error(DMK_ERR_INVALID_ARGUMENT, "Invalid dimension: " + std::to_string(params.n_dim));
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
            } else {
                throw api_error(DMK_ERR_INVALID_ARGUMENT, "tree precision does not match eval precision");
            }
        },
        *static_cast<pdmk_tree_impl *>(tree));
}

template <typename Real>
inline void pdmk_tree_update_charges(pdmk_tree tree, const Real *charge, const Real *normal) {
    std::visit(
        [&](auto &t) {
            using TreeType = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 2>>> ||
                          std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 3>>>) {
                t->update_charges(charge, normal);
            } else {
                throw api_error(DMK_ERR_INVALID_ARGUMENT, "tree precision does not match update_charges precision");
            }
        },
        *static_cast<pdmk_tree_impl *>(tree));
}

#ifdef DMK_BUILD_ESP
// Writes pot_src interleaved [pot, fx, fy, fz] per particle (matching pdmk_tree_eval's
// [pot, dx, dy, dz] convention) when the plan's eval_type requests forces; else just pot.
template <typename Real>
inline void esp_copy_result(const dmk::PotForce<Real> &result, int n, Real *pot_src) {
    if (result.force_x.empty()) {
        std::copy(result.pot.begin(), result.pot.end(), pot_src);
        return;
    }
    const int dim = result.force_z.empty() ? 2 : 3; // scaffolding for a future DIM=2 plan
    const int out_dim = 1 + dim;
    for (int i = 0; i < n; ++i) {
        pot_src[i * out_dim + 0] = result.pot[i];
        pot_src[i * out_dim + 1] = result.force_x[i];
        pot_src[i * out_dim + 2] = result.force_y[i];
        if (dim == 3)
            pot_src[i * out_dim + 3] = result.force_z[i];
    }
}

// Dispatches to the plan's own precision -- evalf genuinely runs EspPlan<float>::eval (float
// FFTs/SIMD throughout), it never up-converts through a double plan.
// (A template can't have C language linkage, so this lives here rather than in the extern "C"
// block below, which only holds the non-template pdmk_esp_eval/evalf wrappers.)
template <typename Real>
inline void pdmk_esp_eval_impl(pdmk_esp_plan plan, int n, const Real *r_src, const Real *charges, Real *pot_src) {
    std::visit(
        [&](auto &p) {
            using PlanType = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<PlanType, std::unique_ptr<dmk::EspPlan<Real>>>)
                esp_copy_result<Real>(p->eval(n, r_src, charges), n, pot_src);
            else
                throw api_error(DMK_ERR_INVALID_ARGUMENT, "ESP plan precision does not match eval precision");
        },
        *static_cast<pdmk_esp_plan_impl *>(plan));
}
#endif // DMK_BUILD_ESP

} // namespace dmk

extern "C" {

void pdmk_init_default_params(pdmk_params *params) {
    if (params)
        *params = pdmk_params{};
}

const char *pdmk_last_error_message(void) { return dmk::last_error_message(); }

dmk_error pdmk_print_profile_data(dmk_communicator comm, char type) {
    return dmk::dmk_guard([&] {
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
    });
}

pdmk_tree pdmk_tree_createf(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src,
                            const float *charge, const float *normal, int n_trg, const float *r_trg) {
    pdmk_tree result = nullptr;
    dmk::dmk_guard([&] {
        dmk::validate_create_args(params, n_src, r_src, charge, normal, n_trg, r_trg);
        result = dmk::pdmk_tree_create(comm, params, n_src, r_src, charge, normal, n_trg, r_trg);
    });
    return result;
}

pdmk_tree pdmk_tree_create(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src,
                           const double *charge, const double *normal, int n_trg, const double *r_trg) {
    pdmk_tree result = nullptr;
    dmk::dmk_guard([&] {
        dmk::validate_create_args(params, n_src, r_src, charge, normal, n_trg, r_trg);
        result = dmk::pdmk_tree_create(comm, params, n_src, r_src, charge, normal, n_trg, r_trg);
    });
    return result;
}

void pdmk_tree_destroy(pdmk_tree tree) {
    if (tree)
        delete static_cast<pdmk_tree_impl *>(tree);
}

dmk_error pdmk_tree_update_charges(pdmk_tree tree, const double *charge, const double *normal) {
    return dmk::dmk_guard([&] {
        if (!tree)
            throw dmk::api_error(DMK_ERR_INVALID_ARGUMENT, "null tree handle");
        if (!charge)
            throw dmk::api_error(DMK_ERR_INVALID_ARGUMENT, "null charge pointer");
        dmk::pdmk_tree_update_charges(tree, charge, normal);
    });
}

dmk_error pdmk_tree_update_chargesf(pdmk_tree tree, const float *charge, const float *normal) {
    return dmk::dmk_guard([&] {
        if (!tree)
            throw dmk::api_error(DMK_ERR_INVALID_ARGUMENT, "null tree handle");
        if (!charge)
            throw dmk::api_error(DMK_ERR_INVALID_ARGUMENT, "null charge pointer");
        dmk::pdmk_tree_update_charges(tree, charge, normal);
    });
}

dmk_error pdmk_tree_evalf(pdmk_tree tree, float *pot_src, float *pot_trg) {
    return dmk::dmk_guard([&] {
        if (!tree)
            throw dmk::api_error(DMK_ERR_INVALID_ARGUMENT, "null tree handle");
        dmk::pdmk_tree_eval(tree, pot_src, pot_trg);
    });
}

dmk_error pdmk_tree_eval(pdmk_tree tree, double *pot_src, double *pot_trg) {
    return dmk::dmk_guard([&] {
        if (!tree)
            throw dmk::api_error(DMK_ERR_INVALID_ARGUMENT, "null tree handle");
        dmk::pdmk_tree_eval(tree, pot_src, pot_trg);
    });
}

dmk_error pdmkf(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src, const float *charge,
                const float *normal, int n_trg, const float *r_trg, float *pot_src, float *pot_trg) {
    return dmk::dmk_guard([&] {
        dmk::validate_create_args(params, n_src, r_src, charge, normal, n_trg, r_trg);
        if (params.n_dim == 2)
            dmk::pdmk<float, 2>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
        else
            dmk::pdmk<float, 3>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
    });
}

dmk_error pdmk(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src, const double *charge,
               const double *normal, int n_trg, const double *r_trg, double *pot_src, double *pot_trg) {
    return dmk::dmk_guard([&] {
        dmk::validate_create_args(params, n_src, r_src, charge, normal, n_trg, r_trg);
        if (params.n_dim == 2)
            dmk::pdmk<double, 2>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
        else
            dmk::pdmk<double, 3>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
    });
}

#ifdef DMK_BUILD_ESP
pdmk_esp_plan pdmk_esp_plan_create(dmk_communicator /*comm*/, pdmk_esp_params params) {
    return new pdmk_esp_plan_impl(std::unique_ptr<dmk::EspPlan<double>>(new dmk::EspPlan<double>(params)));
}

pdmk_esp_plan pdmk_esp_plan_createf(dmk_communicator /*comm*/, pdmk_esp_params params) {
    return new pdmk_esp_plan_impl(std::unique_ptr<dmk::EspPlan<float>>(new dmk::EspPlan<float>(params)));
}

void pdmk_esp_eval(dmk_communicator /*comm*/, pdmk_esp_plan plan, int n, const double *r_src, const double *charges,
                   double *pot_src) {
    dmk::pdmk_esp_eval_impl<double>(plan, n, r_src, charges, pot_src);
}

void pdmk_esp_evalf(dmk_communicator /*comm*/, pdmk_esp_plan plan, int n, const float *r_src, const float *charges,
                    float *pot_src) {
    dmk::pdmk_esp_eval_impl<float>(plan, n, r_src, charges, pot_src);
}

void pdmk_esp_plan_destroy(pdmk_esp_plan plan) { delete static_cast<pdmk_esp_plan_impl *>(plan); }

void pdmk_esp_plan_destroyf(pdmk_esp_plan plan) { pdmk_esp_plan_destroy(plan); }

void pdmk_esp(dmk_communicator comm, pdmk_esp_params params, int n, const double *r_src, const double *charges,
              double *pot_src) {
    auto plan = pdmk_esp_plan_create(comm, params);
    pdmk_esp_eval(comm, plan, n, r_src, charges, pot_src);
    pdmk_esp_plan_destroy(plan);
}

void pdmk_espf(dmk_communicator comm, pdmk_esp_params params, int n, const float *r_src, const float *charges,
               float *pot_src) {
    auto plan = pdmk_esp_plan_createf(comm, params);
    pdmk_esp_evalf(comm, plan, n, r_src, charges, pot_src);
    pdmk_esp_plan_destroyf(plan);
}
#endif // DMK_BUILD_ESP
}
