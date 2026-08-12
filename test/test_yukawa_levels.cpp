// Accuracy of the tree solve against an all-pairs direct sum, across a kernel x
// refinement matrix, at a tolerance of eps.
//
// Guards the near-field level indexing: a leaf with a planewave expansion takes its
// residual one level finer than its own depth, so the index reaches n_levels() and the
// per-level arrays must be sized for it. An unrefined tree hits that deepest index on
// every pair. Yukawa alone has per-level coefficients, so a Yukawa-only failure points
// at the packs while an all-kernel failure points at the shared indexing.

#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/testing.hpp>
#include <dmk/util.hpp>

#include <sctl.hpp>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#define VERBOSE_MESSAGE(...)                                                                                           \
    if (std::getenv("DMK_TEST_VERBOSE")) {                                                                             \
        MESSAGE(__VA_ARGS__);                                                                                          \
    }

namespace {

constexpr int N_DIM = 3;

std::vector<double> to_std(const sctl::Vector<double> &v) {
    std::vector<double> out(v.Dim());
    for (long i = 0; i < v.Dim(); ++i)
        out[i] = v[i];
    return out;
}

double rel_l2(const sctl::Vector<double> &dmk, const std::vector<double> &ref) {
    double err2 = 0, ref2 = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = dmk[i] - ref[i];
        err2 += d * d;
        ref2 += ref[i] * ref[i];
    }
    return ref2 > 0 ? std::sqrt(err2 / ref2) : std::sqrt(err2);
}

/// Relative L2 of the source potential against an all-pairs direct sum.
/// n_per_leaf controls refinement: a value above n_src keeps the tree a single box.
double solve_error(auto comm, dmk_ikernel kernel, int n_src, int n_per_leaf, double eps, double lambda) {
    const int idim = dmk::get_kernel_input_dim(N_DIM, kernel);

    sctl::Vector<double> r_src, r_trg, rnormal, charges;
    // Volume fill rather than the default spherical shell, so pair separations cover
    // the whole residual cutoff rather than one radius.
    dmk::util::init_test_data(N_DIM, idim, n_src, /*n_trg=*/0, /*uniform=*/true, /*set_fixed_charges=*/false, r_src,
                              r_trg, rnormal, charges, /*seed=*/0);

    pdmk_params params;
    params.eps = eps;
    params.n_dim = N_DIM;
    params.kernel = kernel;
    params.eval_src = DMK_POTENTIAL;
    params.eval_trg = DMK_POTENTIAL;
    params.n_per_leaf = n_per_leaf;
    params.fparam = lambda;
    params.log_level = 6;

    sctl::Vector<double> pot_src(n_src);
    sctl::Vector<double> pot_trg(1);
    pot_src.SetZero();
    pot_trg.SetZero();

    pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], /*n_trg=*/0, &r_src[0]);
    REQUIRE_MESSAGE(tree != nullptr, "tree_create failed: ", std::string(pdmk_last_error_message()));
    const dmk_error rc = pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
    pdmk_tree_destroy(tree);
    REQUIRE_MESSAGE(rc == DMK_SUCCESS, "eval failed: ", std::string(pdmk_last_error_message()));

    const std::vector<double> r_src_d = to_std(r_src), charges_d = to_std(charges);
    std::vector<double> ref;
    dmk::compute_direct(N_DIM, r_src_d, charges_d, std::vector<double>{}, r_src_d, ref, kernel, DMK_POTENTIAL, lambda);

    return rel_l2(pot_src, ref);
}

} // namespace

TEST_CASE_GENERIC("[DMK] near-field level indexing across refinement", 1) {
#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    constexpr double EPS = 1e-6;

    struct Kernel {
        const char *name;
        dmk_ikernel kernel;
        double lambda;
    };
    const Kernel kernels[] = {
        {"yukawa lambda=1", DMK_YUKAWA, 1.0}, {"yukawa lambda=6", DMK_YUKAWA, 6.0}, {"laplace", DMK_LAPLACE, 0.0}};

    struct Refinement {
        const char *name;
        int n_src;
        int n_per_leaf; ///< above n_src keeps the root unsplit, so src_level hits n_levels()
    };
    const Refinement refinements[] = {{"single box", 200, 1000000}, {"refined", 5000, 100}};

    for (const auto &k : kernels) {
        for (const auto &r : refinements) {
            const std::string label = std::string(k.name) + " / " + r.name;
            SUBCASE(label.c_str()) {
                const double err = solve_error(comm, k.kernel, r.n_src, r.n_per_leaf, EPS, k.lambda);
                VERBOSE_MESSAGE(label, " rel_l2 = ", err, " (tol=", EPS, ")");
                CHECK(err < EPS);
            }
        }
    }
}
