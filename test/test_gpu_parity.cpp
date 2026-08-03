// GPU accuracy against an all-pairs direct sum: each case requires rel_l2 < eps.
// DMK_GPU_OFFLOAD and DMK_HAVE_MPI are mutually exclusive, hence the null communicator.

#ifdef DMK_GPU_OFFLOAD

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <dmk.h>
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
    pdmk_tree tree = create_tree(nullptr, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
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
void parity_subcases(dmk_ikernel kernel, dmk_eval_type eval, const std::string &label) {
    const std::string label_d = label + " double";
    const std::string label_f = label + " float";
    SUBCASE(label_d.c_str()) { check_accuracy<double>(kernel, eval, 1e-6); }
    SUBCASE(label_f.c_str()) { check_accuracy<float>(kernel, eval, 1e-3); }
}

} // namespace

TEST_CASE_GENERIC("[GPU] 3d scalar kernels parity", 1) {
    for (auto kernel : {DMK_LAPLACE, DMK_SQRT_LAPLACE}) {
        for (auto eval : {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}) {
            const std::string label =
                std::string(dmk::util::to_string(kernel)) + (eval == DMK_POTENTIAL ? " pot" : " pot+grad");
            parity_subcases(kernel, eval, label);
        }
    }
}

TEST_CASE_GENERIC("[GPU] 3d Laplace-dipole parity", 1) {
    for (auto eval : {DMK_POTENTIAL, DMK_POTENTIAL_GRAD})
        parity_subcases(DMK_LAPLACE_DIPOLE, eval, eval == DMK_POTENTIAL ? "pot" : "pot+grad");
}

TEST_CASE_GENERIC("[GPU] 3d velocity kernels parity", 1) {
    for (auto kernel : {DMK_STOKESLET, DMK_STRESSLET})
        parity_subcases(kernel, DMK_VELOCITY, std::string(dmk::util::to_string(kernel)));
}

// Yukawa is not yet on the GPU (rejected in validate_create_args); enable with Phase 3.

#endif // DMK_GPU_OFFLOAD
