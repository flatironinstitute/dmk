// GPU accuracy against an all-pairs direct sum: each case requires rel_l2 < eps.
// The tree path's eval_path=GPU is single-rank only, so every case runs on a self
// communicator. The second half of the file covers the GPU brute-force direct sum itself,
// which has no such restriction.

#ifdef DMK_GPU_OFFLOAD

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <dmk.h>
#include <dmk/bessel.hpp>
#include <dmk/direct.hpp>
#include <dmk/testing.hpp>
#include <dmk/util.hpp>

#include <sctl.hpp>

#define VERBOSE_MESSAGE(...)                                                                                           \
    if (std::getenv("DMK_TEST_VERBOSE")) {                                                                             \
        MESSAGE(__VA_ARGS__);                                                                                          \
    }

namespace {

pdmk_tree create_tree(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src, const double *charge,
                      const double *normal, int n_trg, const double *r_trg) {
    return pdmk_tree_create(comm, params, n_src, r_src, charge, normal, n_trg, r_trg);
}
pdmk_tree create_tree(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src, const float *charge,
                      const float *normal, int n_trg, const float *r_trg) {
    return pdmk_tree_createf(comm, params, n_src, r_src, charge, normal, n_trg, r_trg);
}
dmk_error eval_tree(pdmk_tree tree, double *pot_src, double *pot_trg) { return pdmk_tree_eval(tree, pot_src, pot_trg); }
dmk_error eval_tree(pdmk_tree tree, float *pot_src, float *pot_trg) { return pdmk_tree_evalf(tree, pot_src, pot_trg); }

std::vector<double> to_std(const auto &v) {
    std::vector<double> out(v.Dim());
    for (long i = 0; i < v.Dim(); ++i)
        out[i] = double(v[i]);
    return out;
}

/// Relative L2 over components [c0, c0+nc). Potential and gradient must be normalized
/// separately: a gradient exceeds its potential by ~1/r, so a lumped norm reports only
/// the gradient.
double rel_l2(const auto &dmk, const std::vector<double> &ref, int odim, int c0, int nc) {
    double err2 = 0, ref2 = 0;
    for (std::size_t i = 0; i < ref.size() / odim; ++i)
        for (int c = c0; c < c0 + nc; ++c) {
            const std::size_t k = i * odim + c;
            const double d = double(dmk[k]) - ref[k];
            err2 += d * d;
            ref2 += ref[k] * ref[k];
        }
    return ref2 > 0 ? std::sqrt(err2 / ref2) : std::sqrt(err2);
}

/// Evaluation points carried by the direct reference, which is O(n_src * n_eval).
/// The DMK solves still use the full point set.
constexpr int N_CHECK = 2000;

template <typename Real>
struct RunResult {
    sctl::Vector<Real> pot_src, pot_trg;
};

template <typename Real>
RunResult<Real> run_case(dmk_ikernel kernel, dmk_eval_type eval, double eps, dmk_eval_path path, double fparam,
                         int n_dim, int n_src, int n_trg, int odim, const sctl::Vector<Real> &r_src,
                         const sctl::Vector<Real> &r_trg, const sctl::Vector<Real> &charges,
                         const sctl::Vector<Real> &rnormal) {
    pdmk_params params;
    params.eps = eps;
    params.n_dim = n_dim;
    params.kernel = kernel;
    params.eval_src = eval;
    params.eval_trg = eval;
    params.n_per_leaf = 250;
    params.log_level = 6;
    params.eval_path = path;
    if (kernel == DMK_YUKAWA)
        params.fparam = fparam;

    RunResult<Real> out;
    out.pot_src.ReInit(n_src * odim);
    out.pot_trg.ReInit(std::max(1, n_trg * odim));
    out.pot_src.SetZero();
    out.pot_trg.SetZero();

    const std::string label = path == DMK_EVAL_PATH_CPU ? "CPU" : "GPU";
    pdmk_tree tree =
        create_tree(DMK_TEST_COMM_SELF, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
    REQUIRE_MESSAGE(tree != nullptr, label, " tree_create failed (eps=", eps,
                    "): ", std::string(pdmk_last_error_message()));
    const dmk_error rc = eval_tree(tree, &out.pot_src[0], &out.pot_trg[0]);
    pdmk_tree_destroy(tree);
    REQUIRE_MESSAGE(rc == DMK_SUCCESS, label, " eval failed (eps=", eps, "): ", std::string(pdmk_last_error_message()));
    return out;
}

template <typename Real>
void check_accuracy(dmk_ikernel kernel, dmk_eval_type eval, double eps, double fparam = 0.0) {
    constexpr int n_dim = 3;
    constexpr int n_src = 20000;
    constexpr int n_trg = 20000;

    const int idim = dmk::get_kernel_input_dim(n_dim, kernel);
    const int odim = dmk::get_kernel_output_dim(n_dim, kernel, eval);

    // Generated at the precision under test so the reference sees the same
    // (Real-rounded) coordinates.
    sctl::Vector<Real> r_src, r_trg, rnormal, charges;
    dmk::util::init_test_data(n_dim, idim, n_src, n_trg, /*uniform=*/false, /*set_fixed_charges=*/false, r_src, r_trg,
                              rnormal, charges, /*seed=*/0);

    const std::vector<double> r_src_d = to_std(r_src), r_trg_d = to_std(r_trg), charges_d = to_std(charges);
    // Stresslet is the only kernel that reads normals; dipole carries its strength
    // vector in `charges`.
    const std::vector<double> normals_d = (kernel == DMK_STRESSLET) ? to_std(rnormal) : std::vector<double>{};
    const int n_check = std::min(N_CHECK, std::min(n_src, n_trg));
    const std::vector<double> r_eval_src(r_src_d.begin(), r_src_d.begin() + std::size_t(n_check) * n_dim);
    const std::vector<double> r_eval_trg(r_trg_d.begin(), r_trg_d.begin() + std::size_t(n_check) * n_dim);

    std::vector<double> ref_src, ref_trg;
    dmk::compute_direct(n_dim, r_src_d, charges_d, normals_d, r_eval_src, ref_src, kernel, eval, fparam);
    dmk::compute_direct(n_dim, r_src_d, charges_d, normals_d, r_eval_trg, ref_trg, kernel, eval, fparam);

    const auto gpu = run_case<Real>(kernel, eval, eps, DMK_EVAL_PATH_GPU, fparam, n_dim, n_src, n_trg, odim, r_src,
                                    r_trg, charges, rnormal);
    const auto cpu = run_case<Real>(kernel, eval, eps, DMK_EVAL_PATH_CPU, fparam, n_dim, n_src, n_trg, odim, r_src,
                                    r_trg, charges, rnormal);

    const bool has_grad = eval == DMK_POTENTIAL_GRAD;
    const int n_lead = has_grad ? 1 : odim;

    const double gpu_src = rel_l2(gpu.pot_src, ref_src, odim, 0, n_lead);
    const double gpu_trg = rel_l2(gpu.pot_trg, ref_trg, odim, 0, n_lead);
    const double cpu_src = rel_l2(cpu.pot_src, ref_src, odim, 0, n_lead);
    const double cpu_trg = rel_l2(cpu.pot_trg, ref_trg, odim, 0, n_lead);

    VERBOSE_MESSAGE("pot vs direct -- GPU src=", gpu_src, " trg=", gpu_trg, " | CPU src=", cpu_src, " trg=", cpu_trg,
                    " (tol=", eps, ", n_check=", n_check, ")");

    CHECK(gpu_src < eps);
    CHECK(gpu_trg < eps);
    CHECK(cpu_src < eps);
    CHECK(cpu_trg < eps);

    if (has_grad) {
        const double g_gpu_src = rel_l2(gpu.pot_src, ref_src, odim, 1, n_dim);
        const double g_gpu_trg = rel_l2(gpu.pot_trg, ref_trg, odim, 1, n_dim);
        const double g_cpu_src = rel_l2(cpu.pot_src, ref_src, odim, 1, n_dim);
        const double g_cpu_trg = rel_l2(cpu.pot_trg, ref_trg, odim, 1, n_dim);

        VERBOSE_MESSAGE("grad vs direct -- GPU src=", g_gpu_src, " trg=", g_gpu_trg, " | CPU src=", g_cpu_src,
                        " trg=", g_cpu_trg, " (tol=", eps, ")");

        CHECK(g_gpu_src < eps);
        CHECK(g_gpu_trg < eps);
        CHECK(g_cpu_src < eps);
        CHECK(g_cpu_trg < eps);
    }
}

// eps doubles as the tolerance. Some cases overshoot it on the CPU too (Stresslet at 3
// digits runs ~1.3x); those are accuracy-curve work, not a GPU defect.
void parity_subcases(dmk_ikernel kernel, dmk_eval_type eval, const std::string &label, double fparam) {
    const std::string label_d = label + " double";
    const std::string label_f = label + " float";
    SUBCASE(label_d.c_str()) { check_accuracy<double>(kernel, eval, 1e-6, fparam); }
    SUBCASE(label_f.c_str()) { check_accuracy<float>(kernel, eval, 1e-3, fparam); }
}

dmk_error direct_sum(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src, const double *charge,
                     const double *normal, int n_trg, const double *r_trg, double *pot_src, double *pot_trg) {
    return pdmk_direct(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
}
dmk_error direct_sum(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src, const float *charge,
                     const float *normal, int n_trg, const float *r_trg, float *pot_src, float *pot_trg) {
    return pdmk_directf(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
}

/// GPU brute-force direct vs the CPU brute-force direct on identical inputs. Only the summation
/// order differs, so the two agree to round-off but never bit-for-bit: assert a tolerance.
///
/// pot_src is requested as well as pot_trg because evaluating at the sources puts a coincident
/// pair under every target, which both sides must drop.
template <typename Real>
void check_direct_parity(dmk_ikernel kernel, int n_dim, dmk_eval_type eval, double fparam, double tol) {
    constexpr int n_src = 4000;
    constexpr int n_trg = 1500;

    const int idim = dmk::get_kernel_input_dim(n_dim, kernel);
    const int odim = dmk::get_kernel_output_dim(n_dim, kernel, eval);

    sctl::Vector<Real> r_src, r_trg, rnormal, charges;
    dmk::util::init_test_data(n_dim, idim, n_src, n_trg, /*uniform=*/false, /*set_fixed_charges=*/false, r_src, r_trg,
                              rnormal, charges, /*seed=*/0);

    pdmk_params params;
    params.n_dim = n_dim;
    params.kernel = kernel;
    params.eval_src = eval;
    params.eval_trg = eval;
    params.log_level = 6;
    if (kernel == DMK_YUKAWA)
        params.fparam = fparam;

    std::vector<Real> cpu_src(std::size_t(n_src) * odim), cpu_trg(std::size_t(n_trg) * odim);
    std::vector<Real> gpu_src(std::size_t(n_src) * odim), gpu_trg(std::size_t(n_trg) * odim);

    params.eval_path = DMK_EVAL_PATH_CPU;
    REQUIRE_MESSAGE(direct_sum(DMK_TEST_COMM_SELF, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0],
                               cpu_src.data(), cpu_trg.data()) == DMK_SUCCESS,
                    "CPU direct failed: ", std::string(pdmk_last_error_message()));

    params.eval_path = DMK_EVAL_PATH_GPU;
    REQUIRE_MESSAGE(direct_sum(DMK_TEST_COMM_SELF, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0],
                               gpu_src.data(), gpu_trg.data()) == DMK_SUCCESS,
                    "GPU direct failed: ", std::string(pdmk_last_error_message()));

    const std::vector<double> ref_src(cpu_src.begin(), cpu_src.end());
    const std::vector<double> ref_trg(cpu_trg.begin(), cpu_trg.end());

    const bool has_grad = eval == DMK_POTENTIAL_GRAD;
    const int n_lead = has_grad ? 1 : odim;

    const double e_src = rel_l2(gpu_src, ref_src, odim, 0, n_lead);
    const double e_trg = rel_l2(gpu_trg, ref_trg, odim, 0, n_lead);
    VERBOSE_MESSAGE("direct GPU vs CPU -- src=", e_src, " trg=", e_trg, " (tol=", tol, ")");
    CHECK(e_src < tol);
    CHECK(e_trg < tol);

    if (has_grad) {
        const double g_src = rel_l2(gpu_src, ref_src, odim, 1, n_dim);
        const double g_trg = rel_l2(gpu_trg, ref_trg, odim, 1, n_dim);
        VERBOSE_MESSAGE("direct grad GPU vs CPU -- src=", g_src, " trg=", g_trg, " (tol=", tol, ")");
        CHECK(g_src < tol);
        CHECK(g_trg < tol);
    }

    // A mishandled coincident pair shows up as inf/NaN rather than in a norm.
    bool all_finite = true;
    for (Real v : gpu_src)
        all_finite = all_finite && std::isfinite(double(v));
    for (Real v : gpu_trg)
        all_finite = all_finite && std::isfinite(double(v));
    CHECK(all_finite);
}

void direct_parity_subcases(dmk_ikernel kernel, int n_dim, dmk_eval_type eval, const std::string &label,
                            double fparam) {
    const std::string label_d = label + " double";
    const std::string label_f = label + " float";
    SUBCASE(label_d.c_str()) { check_direct_parity<double>(kernel, n_dim, eval, fparam, 1e-12); }
    SUBCASE(label_f.c_str()) { check_direct_parity<float>(kernel, n_dim, eval, fparam, 1e-5); }
}

/// 2D Yukawa is the only kernel carrying its own device transcendental
/// (include/dmk/cuda/bessel_device.hpp). Comparing it against the CPU SIMD path alone would let a
/// shared coefficient-table error pass, so this uses a scalar host sum over dmk::bessel::k0.
void check_yukawa_2d_against_host_bessel(double lambda) {
    constexpr int n_dim = 2;
    constexpr int n_src = 600;
    constexpr int n_trg = 200;

    sctl::Vector<double> r_src, r_trg, rnormal, charges;
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, /*uniform=*/false, /*set_fixed_charges=*/false, r_src, r_trg,
                              rnormal, charges, /*seed=*/1);

    std::vector<double> ref(n_trg, 0.0);
    for (int t = 0; t < n_trg; ++t) {
        double acc = 0.0;
        for (int s = 0; s < n_src; ++s) {
            const double dx = r_trg[t * n_dim] - r_src[s * n_dim];
            const double dy = r_trg[t * n_dim + 1] - r_src[s * n_dim + 1];
            const double r2 = dx * dx + dy * dy;
            if (r2 == 0.0)
                continue;
            acc += charges[s] * dmk::bessel::k0(std::sqrt(r2) * lambda);
        }
        ref[t] = acc;
    }

    pdmk_params params;
    params.n_dim = n_dim;
    params.kernel = DMK_YUKAWA;
    params.eval_src = DMK_POTENTIAL;
    params.eval_trg = DMK_POTENTIAL;
    params.fparam = lambda;
    params.log_level = 6;
    params.eval_path = DMK_EVAL_PATH_GPU;

    std::vector<double> gpu(n_trg);
    REQUIRE_MESSAGE(direct_sum(DMK_TEST_COMM_SELF, params, n_src, &r_src[0], &charges[0], nullptr, n_trg, &r_trg[0],
                               nullptr, gpu.data()) == DMK_SUCCESS,
                    "GPU direct failed: ", std::string(pdmk_last_error_message()));

    const double err = rel_l2(gpu, ref, 1, 0, 1);
    VERBOSE_MESSAGE("2d yukawa GPU vs host K0 -- lambda=", lambda, " rel_l2=", err);
    CHECK(err < 1e-12);
}

} // namespace

TEST_CASE_GENERIC("[GPU] 3d scalar kernels parity", 1) {
    for (auto kernel : {DMK_LAPLACE, DMK_SQRT_LAPLACE}) {
        for (auto eval : {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}) {
            const std::string label =
                std::string(dmk::util::to_string(kernel)) + (eval == DMK_POTENTIAL ? " pot" : " pot+grad");
            parity_subcases(kernel, eval, label, 0.0);
        }
    }
}

TEST_CASE_GENERIC("[GPU] 3d Laplace-dipole parity", 1) {
    for (auto eval : {DMK_POTENTIAL, DMK_POTENTIAL_GRAD})
        parity_subcases(DMK_LAPLACE_DIPOLE, eval, eval == DMK_POTENTIAL ? "pot" : "pot+grad", 0.0);
}

TEST_CASE_GENERIC("[GPU] 3d velocity kernels parity", 1) {
    for (auto kernel : {DMK_STOKESLET, DMK_STRESSLET})
        parity_subcases(kernel, DMK_VELOCITY, std::string(dmk::util::to_string(kernel)), 0.0);
}

// Sweeping lambda is what exercises the per-level coefficient packs.
TEST_CASE_GENERIC("[GPU] 3d Yukawa parity", 1) {
    for (auto eval : {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}) {
        for (double lambda : {1.0, 6.0, 30.0}) {
            const std::string label =
                std::string(eval == DMK_POTENTIAL ? "pot" : "pot+grad") + " lambda=" + std::to_string(int(lambda));
            parity_subcases(DMK_YUKAWA, eval, label, lambda);
        }
    }
}

TEST_CASE_GENERIC("[GPU] direct scalar kernels vs CPU direct", 1) {
    for (int n_dim : {2, 3}) {
        for (auto kernel : {DMK_LAPLACE, DMK_SQRT_LAPLACE, DMK_LAPLACE_DIPOLE}) {
            for (auto eval : {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}) {
                const std::string label = std::string(dmk::util::to_string(kernel)) + " " + std::to_string(n_dim) +
                                          "d " + (eval == DMK_POTENTIAL ? "pot" : "pot+grad");
                direct_parity_subcases(kernel, n_dim, eval, label, 0.0);
            }
        }
    }
}

TEST_CASE_GENERIC("[GPU] direct velocity kernels vs CPU direct", 1) {
    for (auto kernel : {DMK_STOKESLET, DMK_STRESSLET})
        direct_parity_subcases(kernel, 3, DMK_VELOCITY, std::string(dmk::util::to_string(kernel)) + " 3d", 0.0);
}

// 2D goes through the device K0/K1; small lambda drives its x <= 1 branch, large lambda the
// asymptotic one.
TEST_CASE_GENERIC("[GPU] direct Yukawa vs CPU direct", 1) {
    for (int n_dim : {2, 3}) {
        for (auto eval : {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}) {
            for (double lambda : {0.1, 6.0, 30.0}) {
                const std::string label = std::to_string(n_dim) + "d " +
                                          std::string(eval == DMK_POTENTIAL ? "pot" : "pot+grad") +
                                          " lambda=" + std::to_string(lambda);
                direct_parity_subcases(DMK_YUKAWA, n_dim, eval, label, lambda);
            }
        }
    }
}

TEST_CASE_GENERIC("[GPU] direct 2d Yukawa vs host Bessel K0", 1) {
    for (double lambda : {0.1, 1.0, 6.0, 30.0}) {
        const std::string label = "lambda=" + std::to_string(lambda);
        SUBCASE(label.c_str()) { check_yukawa_2d_against_host_bessel(lambda); }
    }
}

#endif // DMK_GPU_OFFLOAD
