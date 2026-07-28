// Measures DMK accuracy across kernel/dim/digits combinations.
// Compares against direct evaluation at the same working precision.
//
// Usage: ./measure_error [options]
//   -N n_src          Number of source points (default: 10000)
//   -n n_per_leaf     DMK leaf size (default: 250)
//   -D n_direct       Number of points for direct comparison (default: 10000)
//   -t f|d            Precision (default: f)
//   -u                Uniform distribution (default: sphere)
//   -k kernel         Kernel: laplace, sqrt_laplace, laplace_dipole, yukawa, all (default: all)
//   -d dim            Dimension: 2, 3, or 0 for both (default: 0)
//   -l lambda         Yukawa fparam (default: 6.0)
//   -g                Also measure gradient error (potential + gradient)
//   --beta-sweep      Enable beta sweep mode
//   --beta-min val    Min beta for sweep (default: 3.0)
//   --beta-max val    Max beta for sweep (default: 40.0)
//   --beta-step val   Beta step size (default: 0.5)
//   --digits val      Digits for beta sweep (default: 6)

#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/omp_wrapper.hpp>
#include <dmk/util.hpp>

#include <algorithm>
#include <cmath>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <sctl.hpp>
#include <stdexcept>
#include <vector>

#ifdef DMK_HAVE_MPI
auto MYCOMM = MPI_COMM_WORLD;
#else
auto MYCOMM = nullptr;
#endif

struct Config {
    int n_src = 10'000;
    int n_per_leaf = 250;
    int n_direct = 10'000;
    char prec = 'f';
    bool uniform = false;
    bool grad = false;
    double fparam = 6.0; // Yukawa lambda

    // Kernel/dim filtering (-1 = all)
    dmk_ikernel kernel_filter = static_cast<dmk_ikernel>(-1);
    int dim_filter = 0; // 0 = both

    // Beta sweep mode
    bool beta_sweep = false;
    double beta_min = 3.0;
    double beta_max = 40.0;
    double beta_step = 0.5;
    int sweep_digits = 6;
};

struct ErrorMetrics {
    double l2_rel;
    double max_rel;
    double grad_l2_rel;  // 0 if gradient not measured
    double grad_max_rel; // 0 if gradient not measured
    double time;
};

// Reference summed in double from the same (Real-rounded) points DMK uses, so
// float summation error doesn't impose an accuracy floor.
template <typename Real>
std::vector<double> compute_reference(int n_dim, dmk_ikernel kernel, int n_test, const std::vector<Real> &r_src,
                                      const std::vector<Real> &charges, dmk_eval_type eval_level, double lambda) {
    std::vector<double> r_src_d(r_src.begin(), r_src.end());
    std::vector<double> charges_d(charges.begin(), charges.end());
    std::vector<double> r_trg_d(r_src_d.begin(), r_src_d.begin() + n_test * n_dim);
    std::vector<double> pot_direct;
    dmk::compute_direct(n_dim, r_src_d, charges_d, std::vector<double>{}, r_trg_d, pot_direct, kernel, eval_level,
                        lambda);
    return pot_direct;
}

template <typename Real>
ErrorMetrics run_one(int n_dim, dmk_ikernel kernel, int n_digits, const Config &cfg, const std::vector<Real> &r_src,
                     const std::vector<Real> &charges, const std::vector<double> &pot_direct, dmk_eval_type eval_level,
                     double beta_override = -1.0) {
    const int n_src = cfg.n_src;
    const int n_test = std::min(cfg.n_direct, n_src);
    const int out_dim = dmk::get_kernel_output_dim(n_dim, kernel, eval_level);

    double eps = std::pow(10.0, -n_digits);

    pdmk_params params{};
    params.eps = eps;
    params.n_dim = n_dim;
    params.n_per_leaf = cfg.n_per_leaf;
    params.log_level = DMK_LOG_OFF;
    params.eval_src = eval_level;
    params.eval_trg = eval_level;
    params.kernel = kernel;
    if (kernel == DMK_YUKAWA)
        params.fparam = cfg.fparam;

    if (beta_override > 0) {
        params.debug_flags |= DMK_DEBUG_OVERRIDE_BETA;
        params.debug_params[DMK_DEBUG_BETA_SLOT] = beta_override;
    }

    pdmk_tree tree = [&]() {
        if constexpr (std::is_same_v<Real, float>)
            return pdmk_tree_createf(MYCOMM, params, n_src, r_src.data(), charges.data(), nullptr, 0, nullptr);
        else
            return pdmk_tree_create(MYCOMM, params, n_src, r_src.data(), charges.data(), nullptr, 0, nullptr);
    }();

    if (!tree)
        throw std::runtime_error(pdmk_last_error_message());

    std::vector<Real> pot_dmk(n_src * out_dim);

    double st = MY_OMP_GET_WTIME();
    dmk_error rc = [&]() {
        if constexpr (std::is_same_v<Real, float>)
            return pdmk_tree_evalf(tree, pot_dmk.data(), nullptr);
        else
            return pdmk_tree_eval(tree, pot_dmk.data(), nullptr);
    }();
    double dt = MY_OMP_GET_WTIME() - st;

    pdmk_tree_destroy(tree);
    if (rc != DMK_SUCCESS)
        throw std::runtime_error(pdmk_last_error_message());

    const bool has_grad = (eval_level == DMK_POTENTIAL_GRAD);

    double err2 = 0.0, ref2 = 0.0, maxre = 0.0;
    double gerr2 = 0.0, gref2 = 0.0, gmaxre = 0.0;
    for (int i = 0; i < n_test; ++i) {
        double dmk = pot_dmk[i * out_dim];
        double ref = pot_direct[i * out_dim];
        double diff = dmk - ref;
        err2 += diff * diff;
        ref2 += ref * ref;
        if (std::abs(ref) > 0.0)
            maxre = std::max(maxre, std::abs(diff / ref));

        if (has_grad) {
            double gd2 = 0.0, gr2 = 0.0;
            for (int c = 1; c <= n_dim; ++c) {
                double d = pot_dmk[i * out_dim + c] - pot_direct[i * out_dim + c];
                double r = pot_direct[i * out_dim + c];
                gd2 += d * d;
                gr2 += r * r;
            }
            gerr2 += gd2;
            gref2 += gr2;
            if (gr2 > 0.0)
                gmaxre = std::max(gmaxre, std::sqrt(gd2 / gr2));
        }
    }

    return {std::sqrt(err2 / ref2), maxre, has_grad ? std::sqrt(gerr2 / gref2) : 0.0, has_grad ? gmaxre : 0.0, dt};
}

dmk_ikernel parse_kernel(const char *s) {
    std::string k(s);
    if (k == "all")
        return static_cast<dmk_ikernel>(-1);
    if (auto kernel = dmk::util::ikernel_from_string(k))
        return *kernel;
    throw std::runtime_error("Unknown kernel: " + k);
}

template <typename Real>
void run_beta_sweep(const Config &cfg) {
    std::vector<dmk_ikernel> kernels;
    if (static_cast<int>(cfg.kernel_filter) == -1)
        kernels = {DMK_LAPLACE, DMK_SQRT_LAPLACE, DMK_LAPLACE_DIPOLE, DMK_YUKAWA};
    else
        kernels = {cfg.kernel_filter};
    std::vector<int> dims;
    if (cfg.dim_filter == 0)
        dims = {2, 3};
    else
        dims = {cfg.dim_filter};

    const dmk_eval_type eval_level = cfg.grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;

    std::cout << "kernel,dim,beta,L2_rel,max_rel";
    if (cfg.grad)
        std::cout << ",grad_L2_rel,grad_max_rel";
    std::cout << ",time\n";

    for (auto kernel : kernels) {
        for (auto n_dim : dims) {
            std::vector<Real> r_src, charges, r_trg, rnormal;
            std::vector<double> pot_direct;
            try {
                dmk::util::init_test_data(n_dim, dmk::get_kernel_input_dim(n_dim, kernel), cfg.n_src, 0, /*n_trg*/
                                          cfg.uniform, false, r_src, r_trg, rnormal, charges, 0);
                int n_test = std::min(cfg.n_direct, cfg.n_src);
                pot_direct = compute_reference(n_dim, kernel, n_test, r_src, charges, eval_level, cfg.fparam);
            } catch (std::exception &e) {
                std::cout << "# " << dmk::util::to_string(kernel) << " dim=" << n_dim << " FAILED: " << e.what()
                          << "\n";
                continue;
            }
            for (double beta = cfg.beta_min; beta <= cfg.beta_max + 1e-9; beta += cfg.beta_step) {
                try {
                    auto err = run_one<Real>(n_dim, kernel, 12, cfg, r_src, charges, pot_direct, eval_level, beta);
                    std::cout << dmk::util::to_string(kernel) << "," << n_dim << "," << std::fixed
                              << std::setprecision(1) << beta << "," << std::scientific << std::setprecision(6)
                              << err.l2_rel << "," << err.max_rel;
                    if (cfg.grad)
                        std::cout << "," << err.grad_l2_rel << "," << err.grad_max_rel;
                    std::cout << "," << std::fixed << std::setprecision(4) << err.time << "\n" << std::flush;
                } catch (std::exception &e) {
                    std::cout << dmk::util::to_string(kernel) << "," << n_dim << "," << std::fixed
                              << std::setprecision(1) << beta
                              << (cfg.grad ? ",FAILED,FAILED,FAILED,FAILED,0\n" : ",FAILED,FAILED,0\n");
                }
            }
        }
    }
}

template <typename Real>
void run_all(const Config &cfg) {
    std::vector<dmk_ikernel> kernels;
    if (static_cast<int>(cfg.kernel_filter) == -1)
        kernels = {DMK_LAPLACE, DMK_SQRT_LAPLACE, DMK_LAPLACE_DIPOLE, DMK_YUKAWA};
    else
        kernels = {cfg.kernel_filter};

    std::vector<int> dims;
    if (cfg.dim_filter == 0)
        dims = {2, 3};
    else
        dims = {cfg.dim_filter};

    constexpr int min_digits = 3;
    constexpr int max_digits = std::is_same_v<Real, float> ? 6 : 12;

    const dmk_eval_type eval_level = cfg.grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;

    std::cout << std::setw(14) << "kernel" << std::setw(5) << "dim" << std::setw(8) << "digits" << std::setw(12)
              << "eps" << std::setw(14) << "L2_rel" << std::setw(14) << "max_rel";
    if (cfg.grad)
        std::cout << std::setw(14) << "grad_L2" << std::setw(14) << "grad_max";
    std::cout << std::setw(10) << "time(s)" << std::setw(10) << "L2/eps" << "\n";
    std::cout << std::string(cfg.grad ? 115 : 87, '-') << "\n";

    for (auto kernel : kernels) {
        for (auto n_dim : dims) {
            std::vector<Real> r_src, charges, r_trg, rnormal;
            std::vector<double> pot_direct;
            try {
                // targets are a subset of the sources here (see compute_reference), so n_trg=0
                dmk::util::init_test_data(n_dim, dmk::get_kernel_input_dim(n_dim, kernel), cfg.n_src, 0, cfg.uniform,
                                          false, r_src, r_trg, rnormal, charges, 0);
                int n_test = std::min(cfg.n_direct, cfg.n_src);
                pot_direct = compute_reference(n_dim, kernel, n_test, r_src, charges, eval_level, cfg.fparam);
            } catch (std::exception &e) {
                std::cout << std::setw(14) << dmk::util::to_string(kernel) << std::setw(5) << n_dim
                          << "  FAILED: " << e.what() << "\n";
                continue;
            }

            for (int digits = min_digits; digits <= max_digits; ++digits) {
                try {
                    auto err = run_one<Real>(n_dim, kernel, digits, cfg, r_src, charges, pot_direct, eval_level);
                    double eps = std::pow(10.0, -digits);
                    std::cout << std::setw(14) << dmk::util::to_string(kernel) << std::setw(5) << n_dim << std::setw(8)
                              << digits << std::setw(12) << std::scientific << std::setprecision(0) << eps
                              << std::setw(14) << std::scientific << std::setprecision(3) << err.l2_rel << std::setw(14)
                              << err.max_rel;
                    if (cfg.grad)
                        std::cout << std::setw(14) << err.grad_l2_rel << std::setw(14) << err.grad_max_rel;
                    std::cout << std::setw(10) << std::fixed << std::setprecision(4) << err.time << std::setw(10)
                              << std::fixed << std::setprecision(1) << err.l2_rel / eps << "\n"
                              << std::flush;
                } catch (std::exception &e) {
                    std::cout << std::setw(14) << dmk::util::to_string(kernel) << std::setw(5) << n_dim << std::setw(8)
                              << digits << "  FAILED: " << e.what() << "\n";
                }
            }
        }
    }
}

Config parse_args(int argc, char *argv[]) {
    Config cfg;

    static struct option long_opts[] = {
        {"beta-sweep", no_argument, nullptr, 1001},     {"beta-min", required_argument, nullptr, 1002},
        {"beta-max", required_argument, nullptr, 1003}, {"beta-step", required_argument, nullptr, 1004},
        {"digits", required_argument, nullptr, 1005},   {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "N:n:D:t:k:d:l:ugh", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'N':
            cfg.n_src = static_cast<int>(std::atof(optarg));
            break;
        case 'n':
            cfg.n_per_leaf = std::atoi(optarg);
            break;
        case 'D':
            cfg.n_direct = static_cast<int>(std::atof(optarg));
            break;
        case 'k':
            cfg.kernel_filter = parse_kernel(optarg);
            break;
        case 'd':
            cfg.dim_filter = std::atoi(optarg);
            break;
        case 't':
            if (optarg[0] == 'd')
                cfg.prec = 'd';
            else if (optarg[0] == 'f')
                cfg.prec = 'f';
            else {
                std::cerr << "Unknown precision: " << optarg << "\n";
                exit(1);
            }
            break;
        case 'l':
            cfg.fparam = std::atof(optarg);
            break;
        case 'u':
            cfg.uniform = true;
            break;
        case 'g':
            cfg.grad = true;
            break;
        case 1001:
            cfg.beta_sweep = true;
            break;
        case 1002:
            cfg.beta_min = std::atof(optarg);
            break;
        case 1003:
            cfg.beta_max = std::atof(optarg);
            break;
        case 1004:
            cfg.beta_step = std::atof(optarg);
            break;
        case 1005:
            cfg.sweep_digits = std::atoi(optarg);
            break;
        case 'h':
        default:
            std::cout << "Usage: " << argv[0] << "\n"
                      << "  -N n_src          Number of source points\n"
                      << "  -n n_per_leaf     DMK leaf size\n"
                      << "  -D n_direct       Points for direct comparison\n"
                      << "  -t f|d            Precision\n"
                      << "  -k kernel         laplace, sqrt_laplace, laplace_dipole, yukawa, all\n"
                      << "  -d dim            2, 3, or 0 for both\n"
                      << "  -l lambda         Yukawa fparam (default: 6.0)\n"
                      << "  -u                Uniform distribution\n"
                      << "  -g                Also measure gradient error\n"
                      << "  --beta-sweep      Enable beta sweep mode\n"
                      << "  --beta-min val    Min beta (default: 3.0)\n"
                      << "  --beta-max val    Max beta (default: 40.0)\n"
                      << "  --beta-step val   Step size (default: 0.5)\n"
                      << "  --digits val      Digits for sweep (default: 6)\n";
            exit(0);
        }
    }
    return cfg;
}

int main(int argc, char *argv[]) {
#ifdef DMK_HAVE_MPI
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size > 1) {
        if (!rank)
            std::cerr << "Measure error not MPI aware. Exiting\n";
        MPI_Finalize();
        return 0;
    }
#endif

    try {
        Config cfg = parse_args(argc, argv);

        std::cout << "# n_src=" << cfg.n_src << " n_per_leaf=" << cfg.n_per_leaf << " n_direct=" << cfg.n_direct
                  << " prec=" << cfg.prec << " uniform=" << cfg.uniform << " grad=" << cfg.grad
                  << " fparam=" << cfg.fparam << " threads=" << MY_OMP_GET_MAX_THREADS();
        if (cfg.beta_sweep)
            std::cout << " beta_sweep=[" << cfg.beta_min << "," << cfg.beta_max << "," << cfg.beta_step
                      << "] digits=" << cfg.sweep_digits;
        std::cout << "\n\n";

        if (cfg.beta_sweep) {
            if (cfg.prec == 'd')
                run_beta_sweep<double>(cfg);
            else
                run_beta_sweep<float>(cfg);
        } else {
            if (cfg.prec == 'd')
                run_all<double>(cfg);
            else
                run_all<float>(cfg);
        }
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

#ifdef DMK_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
