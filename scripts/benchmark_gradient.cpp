// Benchmark: Laplace gradient evaluation accuracy and speed
// Compares potential-only vs potential+gradient, 2D and 3D, multiple problem sizes
// Usage: compile and run, or use as a doctest test file

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <dmk.h>
#include <dmk/omp_wrapper.hpp>
#include <dmk/tree.hpp>
#include <dmk/util.hpp>
#include <sctl.hpp>

#ifdef DMK_HAVE_MPI
auto MYCOMM = MPI_COMM_WORLD;
#else
auto MYCOMM = nullptr;
#endif

struct BenchmarkResult {
    int n_dim;
    int n_src;
    int n_trg;
    double eps;
    bool with_grad;
    double tree_build_time;
    double eval_time;
    double total_time;
    double pot_src_l2_err;
    double pot_trg_l2_err;
    double grad_src_l2_err;
    double grad_trg_l2_err;
    double grad_src_max_err;
    double grad_trg_max_err;
};

template <typename Real>
BenchmarkResult run_benchmark(int n_dim, int n_src, int n_trg, double eps, bool with_grad, int n_per_leaf) {
    BenchmarkResult result{};
    result.n_dim = n_dim;
    result.n_src = n_src;
    result.n_trg = n_trg;
    result.eps = eps;
    result.with_grad = with_grad;

    const int nd = 1;
    constexpr double thresh2 = 1e-30;

    sctl::Vector<Real> r_src, r_trg, rnormal, charges, dipstr;
    dmk::util::init_test_data(n_dim, nd, n_src, n_trg, false, false, r_src, r_trg, rnormal, charges, dipstr, 0);

    const int output_dim = with_grad ? 1 + n_dim : 1;
    sctl::Vector<Real> pot_src(n_src * output_dim), pot_trg(n_trg * output_dim);
    pot_src.SetZero();
    pot_trg.SetZero();

    pdmk_params params;
    params.eps = eps;
    params.n_dim = n_dim;
    params.n_per_leaf = n_per_leaf;
    params.pgh_src = with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
    params.pgh_trg = with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
    params.kernel = DMK_LAPLACE;
    params.log_level = DMK_LOG_OFF;

    // Tree build
    double t0 = MY_OMP_GET_WTIME();
    pdmk_tree tree =
        pdmk_tree_create(MYCOMM, params, n_src, &r_src[0], &charges[0], &rnormal[0], &dipstr[0], n_trg, &r_trg[0]);
    result.tree_build_time = MY_OMP_GET_WTIME() - t0;

    // Eval
    t0 = MY_OMP_GET_WTIME();
    pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
    result.eval_time = MY_OMP_GET_WTIME() - t0;
    result.total_time = result.tree_build_time + result.eval_time;

    pdmk_tree_destroy(tree);

    // Accuracy: compute direct potential (and gradient) for a prefix
    const int n_test = std::min(64, std::min(n_src, n_trg));

    // Direct results in interleaved layout: [pot, (gx, gy, [gz])] per point
    std::vector<double> direct_src(n_test * output_dim, 0.0), direct_trg(n_test * output_dim, 0.0);

    auto compute_direct = [&](const Real *target, int n_dim, int i_out, std::vector<double> &out) {
        for (int i_src = 0; i_src < n_src; ++i_src) {
            double dx[3];
            double dr2 = 0.0;
            for (int d = 0; d < n_dim; ++d) {
                dx[d] = (double)target[d] - (double)r_src[i_src * n_dim + d];
                dr2 += dx[d] * dx[d];
            }
            if (dr2 <= thresh2)
                continue;

            double q = (double)charges[i_src];
            if (n_dim == 2) {
                out[i_out * output_dim] += q * 0.5 * std::log(dr2);
                if (with_grad)
                    for (int d = 0; d < n_dim; ++d)
                        out[i_out * output_dim + 1 + d] += q * dx[d] / dr2;
            } else {
                double rinv = 1.0 / std::sqrt(dr2);
                out[i_out * output_dim] += q * rinv;
                if (with_grad) {
                    double rinv3 = rinv / dr2;
                    for (int d = 0; d < n_dim; ++d)
                        out[i_out * output_dim + 1 + d] -= q * dx[d] * rinv3;
                }
            }
        }
    };

    for (int i = 0; i < n_test; ++i)
        compute_direct(&r_src[i * n_dim], n_dim, i, direct_src);
    for (int i = 0; i < n_test; ++i)
        compute_direct(&r_trg[i * n_dim], n_dim, i, direct_trg);

    // Relative L2 errors over a strided subset of the interleaved arrays
    auto rel_l2 = [](const auto &approx, const std::vector<double> &exact,
                     int n, int stride, int offset) {
        double err2 = 0.0, ref2 = 0.0;
        for (int i = 0; i < n; ++i) {
            double diff = (double)approx[i * stride + offset] - exact[i * stride + offset];
            err2 += diff * diff;
            ref2 += exact[i * stride + offset] * exact[i * stride + offset];
        }
        return ref2 > 0 ? std::sqrt(err2 / ref2) : 0.0;
    };

    auto max_rel = [](const auto &approx, const std::vector<double> &exact,
                      int n, int stride, int offset) {
        double maxe = 0.0;
        for (int i = 0; i < n; ++i) {
            double ref = std::abs(exact[i * stride + offset]);
            if (ref > 1e-15) {
                double err = std::abs((double)approx[i * stride + offset] - exact[i * stride + offset]) / ref;
                maxe = std::max(maxe, err);
            }
        }
        return maxe;
    };

    // Gradient: L2 over all components together
    auto rel_l2_grad = [](const auto &approx, const std::vector<double> &exact,
                          int n, int n_dim, int stride) {
        double err2 = 0.0, ref2 = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < n_dim; ++d) {
                double diff = (double)approx[i * stride + 1 + d] - exact[i * stride + 1 + d];
                err2 += diff * diff;
                ref2 += exact[i * stride + 1 + d] * exact[i * stride + 1 + d];
            }
        }
        return ref2 > 0 ? std::sqrt(err2 / ref2) : 0.0;
    };

    auto max_rel_grad = [](const auto &approx, const std::vector<double> &exact,
                           int n, int n_dim, int stride) {
        double maxe = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < n_dim; ++d) {
                double ref = std::abs(exact[i * stride + 1 + d]);
                if (ref > 1e-15) {
                    double err = std::abs((double)approx[i * stride + 1 + d] - exact[i * stride + 1 + d]) / ref;
                    maxe = std::max(maxe, err);
                }
            }
        }
        return maxe;
    };

    result.pot_src_l2_err = rel_l2(pot_src, direct_src, n_test, output_dim, 0);
    result.pot_trg_l2_err = rel_l2(pot_trg, direct_trg, n_test, output_dim, 0);

    if (with_grad) {
        result.grad_src_l2_err = rel_l2_grad(pot_src, direct_src, n_test, n_dim, output_dim);
        result.grad_trg_l2_err = rel_l2_grad(pot_trg, direct_trg, n_test, n_dim, output_dim);
        result.grad_src_max_err = max_rel_grad(pot_src, direct_src, n_test, n_dim, output_dim);
        result.grad_trg_max_err = max_rel_grad(pot_trg, direct_trg, n_test, n_dim, output_dim);
    }

    return result;
}

void print_header() {
    printf("%-4s %7s %7s %10s %5s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n", "dim", "n_src", "n_trg", "eps",
           "grad", "build(s)", "eval(s)", "total(s)", "pot_src_e", "pot_trg_e", "grd_src_e", "grd_trg_e", "grd_src_mx",
           "grd_trg_mx");
    printf("%-4s %7s %7s %10s %5s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n", "----", "-------", "-------",
           "----------", "-----", "----------", "----------", "----------", "----------", "----------", "----------",
           "----------", "----------", "----------");
}

void print_result(const BenchmarkResult &r) {
    printf("%-4d %7d %7d %10.1e %5s %10.4f %10.4f %10.4f %10.3e %10.3e", r.n_dim, r.n_src, r.n_trg, r.eps,
           r.with_grad ? "yes" : "no", r.tree_build_time, r.eval_time, r.total_time, r.pot_src_l2_err,
           r.pot_trg_l2_err);
    if (r.with_grad)
        printf(" %10.3e %10.3e %10.3e %10.3e", r.grad_src_l2_err, r.grad_trg_l2_err, r.grad_src_max_err,
               r.grad_trg_max_err);
    else
        printf(" %10s %10s %10s %10s", "n/a", "n/a", "n/a", "n/a");
    printf("\n");
}

int main(int argc, char **argv) {
#ifdef DMK_HAVE_MPI
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size > 1) {
        if (!rank)
            std::cerr << "Benchmark not MPI aware. Exiting\n";
        MPI_Finalize();
        return 0;
    }
#endif

    printf("=== FIDMK Laplace Gradient Benchmark (poly-derivative method) ===\n");
    printf("Method: Chebyshev polynomial differentiation of proxy expansion\n");
    printf("Precision: double\n");
#ifdef _OPENMP
    printf("OpenMP threads: %d\n", omp_get_max_threads());
#else
    printf("OpenMP: disabled\n");
#endif
    printf("\n");

    struct TestConfig {
        int n_dim;
        int n_src;
        int n_trg;
        double eps;
        int n_per_leaf;
    };

    std::vector<TestConfig> configs = {
        // Small problems: accuracy focus
        {3, 4000, 3000, 1e-3, 280},
        {3, 4000, 3000, 1e-6, 280},
        {3, 4000, 3000, 1e-9, 280},
        {3, 4000, 3000, 1e-12, 280},
        // Medium problems: speed focus
        {3, 20000, 20000, 1e-6, 280},
        {3, 20000, 20000, 1e-9, 280},
        // Larger problems
        {3, 50000, 50000, 1e-6, 280},
        {3, 100000, 100000, 1e-6, 280},
        {3, 200000, 200000, 1e-6, 280},
    };

    // --- Accuracy sweep ---
    printf("--- Accuracy vs Direct O(N^2) ---\n");
    print_header();
    for (auto &cfg : configs) {
        if (cfg.n_src > 20000)
            continue; // skip large for accuracy (direct is O(N^2))
        auto r_pot = run_benchmark<double>(cfg.n_dim, cfg.n_src, cfg.n_trg, cfg.eps, false, cfg.n_per_leaf);
        auto r_grad = run_benchmark<double>(cfg.n_dim, cfg.n_src, cfg.n_trg, cfg.eps, true, cfg.n_per_leaf);
        print_result(r_pot);
        print_result(r_grad);
    }

    printf("\n--- Speed: Potential-only vs Potential+Gradient ---\n");
    print_header();
    for (auto &cfg : configs) {
        auto r_pot = run_benchmark<double>(cfg.n_dim, cfg.n_src, cfg.n_trg, cfg.eps, false, cfg.n_per_leaf);
        auto r_grad = run_benchmark<double>(cfg.n_dim, cfg.n_src, cfg.n_trg, cfg.eps, true, cfg.n_per_leaf);
        print_result(r_pot);
        print_result(r_grad);
    }

    printf("\n--- Gradient overhead (eval time ratio: grad/pot-only) ---\n");
    printf("%-4s %7s %10s %12s %12s %10s\n", "dim", "n_src", "eps", "pot_eval(s)", "grad_eval(s)", "overhead");
    printf("%-4s %7s %10s %12s %12s %10s\n", "----", "-------", "----------", "------------", "------------",
           "----------");
    for (auto &cfg : configs) {
        auto r_pot = run_benchmark<double>(cfg.n_dim, cfg.n_src, cfg.n_trg, cfg.eps, false, cfg.n_per_leaf);
        auto r_grad = run_benchmark<double>(cfg.n_dim, cfg.n_src, cfg.n_trg, cfg.eps, true, cfg.n_per_leaf);
        double overhead = r_grad.eval_time / r_pot.eval_time;
        printf("%-4d %7d %10.1e %12.4f %12.4f %9.2fx\n", cfg.n_dim, cfg.n_src, cfg.eps, r_pot.eval_time,
               r_grad.eval_time, overhead);
    }

#ifdef DMK_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
