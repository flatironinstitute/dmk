// measure_error.cpp
//
// Measures DMK accuracy across kernel/dim/digits combinations.
// Compares against double-precision direct evaluation.
//
// Usage: ./measure_error [options]
//   -N n_src          Number of source points (default: 10000)
//   -n n_per_leaf     DMK leaf size (default: 250)
//   -D n_direct       Number of points for direct comparison (default: 10000)
//   -t f|d            Precision (default: f)
//   -u                Uniform distribution (default: sphere)
//   -k kernel         Kernel: laplace, sqrt_laplace, yukawa, all (default: all)
//   -d dim            Dimension: 2, 3, or 0 for both (default: 0)
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
#include <random>
#include <sctl.hpp>
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

void generate_points(int n_dim, int n_src, bool uniform, std::vector<double> &r_src, std::vector<double> &charges,
                     long seed = 0) {
    r_src.resize(n_dim * n_src);
    charges.resize(n_src);

    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> rng;
    const double rin = 0.45;

    for (int i = 0; i < n_src; ++i) {
        if (!uniform && n_dim == 3) {
            double theta = rng(eng) * M_PI;
            double ct = std::cos(theta), st = std::sin(theta);
            double phi = rng(eng) * 2.0 * M_PI;
            r_src[i * 3 + 0] = rin * st * std::cos(phi) + 0.5;
            r_src[i * 3 + 1] = rin * st * std::sin(phi) + 0.5;
            r_src[i * 3 + 2] = rin * ct + 0.5;
        } else if (!uniform && n_dim == 2) {
            double phi = rng(eng) * 2.0 * M_PI;
            r_src[i * 2 + 0] = rin * std::cos(phi) + 0.5;
            r_src[i * 2 + 1] = rin * std::sin(phi) + 0.5;
        } else {
            for (int j = 0; j < n_dim; ++j)
                r_src[i * n_dim + j] = rng(eng);
        }
        charges[i] = rng(eng) - 0.5;
    }
}

template <typename Real>
void parallel_direct_eval(const dmk::direct_evaluator_func<Real> &func, int n_src, const Real *r_src,
                          const Real *charge, int n_trg, const Real *r_trg, Real *pot, int spatial_dim,
                          int charge_dim) {
#pragma omp parallel
    {
        const int nt = MY_OMP_GET_NUM_THREADS();
        const int tid = MY_OMP_GET_THREAD_NUM();
        const int lo = (tid * n_trg) / nt;
        const int hi = ((tid + 1) * n_trg) / nt;
        if (hi > lo)
            func(n_src, r_src, charge, hi - lo, r_trg + lo * spatial_dim, pot + lo * charge_dim);
    }
}
template <typename PotFunc>
void compute_direct(int n_dim, int n_src, int n_test, const std::vector<double> &r_src,
                    const std::vector<double> &charges, std::vector<double> &pot_direct, PotFunc &&pot_func) {
    pot_direct.resize(n_test, 0.0);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_test; ++i) {
        double val = 0.0;
        for (int j = 0; j < n_src; ++j) {
            val += charges[j] * pot_func(&r_src[i * n_dim], &r_src[j * n_dim]);
        }
        pot_direct[i] = val;
    }
}

void compute_direct_dispatch(int n_dim, int n_src, int n_test, const std::vector<double> &r_src,
                             const std::vector<double> &charges, std::vector<double> &pot_direct, dmk_ikernel kernel) {
    const double lambda = 6.0;
    pot_direct.resize(n_test);
    auto potfunc = dmk::get_direct_evaluator<double>(kernel, DMK_POTENTIAL, n_dim, lambda);
    parallel_direct_eval(potfunc, n_src, r_src.data(), charges.data(), n_test, r_src.data(), pot_direct.data(), n_dim,
                         1);
}

struct ErrorMetrics {
    double l2_rel;
    double max_rel;
    double time;
};

template <typename Real>
ErrorMetrics run_one(int n_dim, dmk_ikernel kernel, int n_digits, const Config &cfg, const std::vector<double> &r_src_d,
                     const std::vector<double> &charges_d, const std::vector<double> &pot_direct,
                     double beta_override = -1.0) {
    const int n_src = cfg.n_src;
    const int n_test = std::min(cfg.n_direct, n_src);

    std::vector<Real> r_src(r_src_d.begin(), r_src_d.end());
    std::vector<Real> charges(charges_d.begin(), charges_d.end());

    double eps = std::pow(10.0, -n_digits);

    pdmk_params params{};
    params.eps = eps;
    params.n_dim = n_dim;
    params.n_per_leaf = cfg.n_per_leaf;
    params.log_level = DMK_LOG_OFF;
    params.pgh_src = DMK_POTENTIAL;
    params.pgh_trg = DMK_POTENTIAL;
    params.kernel = kernel;
    if (kernel == DMK_YUKAWA)
        params.fparam = 6.0;

    if (beta_override > 0) {
        params.debug_flags |= DMK_DEBUG_OVERRIDE_BETA;
        params.debug_params[DMK_DEBUG_BETA_SLOT] = beta_override;
    }

    pdmk_tree tree;
    if constexpr (std::is_same_v<Real, float>)
        tree = pdmk_tree_createf(MYCOMM, params, n_src, r_src.data(), charges.data(), nullptr, nullptr, 0, nullptr);
    else
        tree = pdmk_tree_create(MYCOMM, params, n_src, r_src.data(), charges.data(), nullptr, nullptr, 0, nullptr);

    std::vector<Real> pot_dmk(n_src);

    double st = omp_get_wtime();
    if constexpr (std::is_same_v<Real, float>)
        pdmk_tree_evalf(tree, pot_dmk.data(), nullptr);
    else
        pdmk_tree_eval(tree, pot_dmk.data(), nullptr);
    double dt = omp_get_wtime() - st;

    pdmk_tree_destroy(tree);

    double err2 = 0.0, ref2 = 0.0, maxre = 0.0;
    for (int i = 0; i < n_test; ++i) {
        double dmk = static_cast<double>(pot_dmk[i]);
        double ref = pot_direct[i];
        double diff = dmk - ref;
        err2 += diff * diff;
        ref2 += ref * ref;
        if (std::abs(ref) > 0.0)
            maxre = std::max(maxre, std::abs(diff / ref));
    }

    return {std::sqrt(err2 / ref2), maxre, dt};
}

const char *kernel_name(dmk_ikernel k) {
    switch (k) {
    case DMK_LAPLACE:
        return "Laplace";
    case DMK_SQRT_LAPLACE:
        return "SqrtLaplace";
    case DMK_YUKAWA:
        return "Yukawa";
    default:
        return "Unknown";
    }
}

dmk_ikernel parse_kernel(const char *s) {
    std::string k(s);
    if (k == "laplace")
        return DMK_LAPLACE;
    if (k == "sqrt_laplace")
        return DMK_SQRT_LAPLACE;
    if (k == "yukawa")
        return DMK_YUKAWA;
    if (k == "all")
        return static_cast<dmk_ikernel>(-1);
    throw std::runtime_error("Unknown kernel: " + k);
}

template <typename Real>
void run_beta_sweep(const Config &cfg) {
    std::vector<dmk_ikernel> kernels;
    if (static_cast<int>(cfg.kernel_filter) == -1)
        kernels = {DMK_LAPLACE, DMK_SQRT_LAPLACE, DMK_YUKAWA};
    else
        kernels = {cfg.kernel_filter};
    std::vector<int> dims;
    if (cfg.dim_filter == 0)
        dims = {2, 3};
    else
        dims = {cfg.dim_filter};

    std::cout << "kernel,dim,beta,L2_rel,max_rel,time\n";

    for (auto kernel : kernels) {
        for (auto n_dim : dims) {
            std::vector<double> r_src, charges;
            generate_points(n_dim, cfg.n_src, cfg.uniform, r_src, charges);
            int n_test = std::min(cfg.n_direct, cfg.n_src);
            std::vector<double> pot_direct;
            compute_direct_dispatch(n_dim, cfg.n_src, n_test, r_src, charges, pot_direct, kernel);
            for (double beta = cfg.beta_min; beta <= cfg.beta_max + 1e-9; beta += cfg.beta_step) {
                try {
                    auto err = run_one<Real>(n_dim, kernel, 12, cfg, r_src, charges, pot_direct, beta);
                    std::cout << kernel_name(kernel) << "," << n_dim << "," << std::fixed << std::setprecision(1)
                              << beta << "," << std::scientific << std::setprecision(6) << err.l2_rel << ","
                              << err.max_rel << "," << std::fixed << std::setprecision(4) << err.time << "\n"
                              << std::flush;
                } catch (std::exception &e) {
                    std::cout << kernel_name(kernel) << "," << n_dim << "," << std::fixed << std::setprecision(1)
                              << beta << ",FAILED,FAILED,0\n";
                }
            }
        }
    }
}

template <typename Real>
void run_all(const Config &cfg) {
    std::vector<dmk_ikernel> kernels;
    if (static_cast<int>(cfg.kernel_filter) == -1)
        kernels = {DMK_LAPLACE, DMK_SQRT_LAPLACE, DMK_YUKAWA};
    else
        kernels = {cfg.kernel_filter};

    std::vector<int> dims;
    if (cfg.dim_filter == 0)
        dims = {2, 3};
    else
        dims = {cfg.dim_filter};

    constexpr int min_digits = 3;
    constexpr int max_digits = std::is_same_v<Real, float> ? 6 : 12;

    std::cout << std::setw(14) << "kernel" << std::setw(5) << "dim" << std::setw(8) << "digits" << std::setw(12)
              << "eps" << std::setw(14) << "L2_rel" << std::setw(14) << "max_rel" << std::setw(10) << "time(s)"
              << std::setw(10) << "L2/eps" << "\n";
    std::cout << std::string(87, '-') << "\n";

    for (auto kernel : kernels) {
        for (auto n_dim : dims) {
            std::vector<double> r_src, charges;
            generate_points(n_dim, cfg.n_src, cfg.uniform, r_src, charges);

            int n_test = std::min(cfg.n_direct, cfg.n_src);
            std::vector<double> pot_direct;
            compute_direct_dispatch(n_dim, cfg.n_src, n_test, r_src, charges, pot_direct, kernel);

            for (int digits = min_digits; digits <= max_digits; ++digits) {
                try {
                    auto err = run_one<Real>(n_dim, kernel, digits, cfg, r_src, charges, pot_direct);
                    double eps = std::pow(10.0, -digits);
                    std::cout << std::setw(14) << kernel_name(kernel) << std::setw(5) << n_dim << std::setw(8) << digits
                              << std::setw(12) << std::scientific << std::setprecision(0) << eps << std::setw(14)
                              << std::scientific << std::setprecision(3) << err.l2_rel << std::setw(14) << err.max_rel
                              << std::setw(10) << std::fixed << std::setprecision(4) << err.time << std::setw(10)
                              << std::fixed << std::setprecision(1) << err.l2_rel / eps << "\n"
                              << std::flush;
                } catch (std::exception &e) {
                    std::cout << std::setw(14) << kernel_name(kernel) << std::setw(5) << n_dim << std::setw(8) << digits
                              << "  FAILED: " << e.what() << "\n";
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
    while ((opt = getopt_long(argc, argv, "N:n:D:t:k:d:uh", long_opts, nullptr)) != -1) {
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
        case 'u':
            cfg.uniform = true;
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
                      << "  -k kernel         laplace, sqrt_laplace, yukawa, all\n"
                      << "  -d dim            2, 3, or 0 for both\n"
                      << "  -u                Uniform distribution\n"
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

    Config cfg = parse_args(argc, argv);

    std::cout << "# n_src=" << cfg.n_src << " n_per_leaf=" << cfg.n_per_leaf << " n_direct=" << cfg.n_direct
              << " prec=" << cfg.prec << " uniform=" << cfg.uniform << " threads=" << MY_OMP_GET_MAX_THREADS();
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

#ifdef DMK_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
