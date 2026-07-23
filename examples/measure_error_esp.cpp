// Measures ESP accuracy across kernel/dim/eps/r_c combinations at fixed sigma.
// Compares against an exact periodic (Ewald/image) or free-space direct reference.
//
// The eps handed to ESP (eps_fu) need not equal the accuracy the user wants: sweeping
// it against a fixed exact reference reveals, per kernel/dim, how many digits ESP must
// be asked for so the *force* L2 meets a target (ESP does not compensate derivatives, so
// forces trail the potential by ~1 digit). This is the ESP analog of measure_error.cpp's
// --beta-sweep for DMK.
//
// Usage: ./measure_error_esp [options]
//   -N n_src          Number of source points (default: 100000)
//   -D n_direct       Number of points compared against the reference (default: 10000)
//   -t f|d            Precision (default: d)
//   -k kernel         laplace, sqrt_laplace, yukawa, all (default: all)
//   -d dim            2, 3, or 0 for both (default: 0)
//   -l lambda         Yukawa fparam (default: 6.0)
//   -L box            Box side length (default: 1.0)
//   -o                Free-space (open) boundaries instead of periodic
//   -s sigma          FINUFFT upsampling (default 1.35; != 1.35 requires -DDMK_USE_JIT=ON)
//   --dig-min val     Min solver digits (default: 3)
//   --dig-max val     Max solver digits (default: 9)
//   --rc-min val      Min r_c as a fraction of L (default: 0.05)
//   --rc-max val      Max r_c as a fraction of L (default: 0.25)
//   --rc-step val     r_c fraction step (default: 0.05)

#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/esp.hpp>
#include <dmk/omp_wrapper.hpp>
#include <dmk/util.hpp>

#include "periodic_reference.hpp"

#include <cmath>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <string_view>
#include <type_traits>
#include <vector>

#ifdef DMK_HAVE_MPI
auto MYCOMM = MPI_COMM_WORLD;
#else
auto MYCOMM = nullptr;
#endif

struct Config {
    int n_src = 100'000;
    int n_direct = 10'000;
    char prec = 'd';
    dmk_ikernel kernel_filter = static_cast<dmk_ikernel>(-1);
    int dim_filter = 0;
    double fparam = 6.0;
    double L = 1.0;
    bool use_periodic = true;
    double sigma = 1.35;
    bool sigma_set = false;
    int dig_min = 3, dig_max = 9;
    double rc_min = 0.05, rc_max = 0.25, rc_step = 0.05;
    long seed = 42;
};

struct ErrorMetrics {
    double pot_l2, pot_max, force_l2, force_max, time;
};

template <typename Real>
pdmk_esp_plan esp_plan_create(pdmk_esp_params params) {
    if constexpr (std::is_same_v<Real, float>)
        return pdmk_esp_plan_createf(MYCOMM, params);
    else
        return pdmk_esp_plan_create(MYCOMM, params);
}
template <typename Real>
void esp_eval(pdmk_esp_plan plan, int n, const Real *r_src, const Real *charges, Real *pot) {
    if constexpr (std::is_same_v<Real, float>)
        pdmk_esp_evalf(MYCOMM, plan, n, r_src, charges, pot);
    else
        pdmk_esp_eval(MYCOMM, plan, n, r_src, charges, pot);
}
template <typename Real>
void esp_plan_destroy(pdmk_esp_plan plan) {
    if constexpr (std::is_same_v<Real, float>)
        pdmk_esp_plan_destroyf(plan);
    else
        pdmk_esp_plan_destroy(plan);
}

template <typename Real>
std::vector<Real> generate_positions(int n, int n_dim, double L, long seed = 42) {
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> rng(-0.5 * L, 0.5 * L);
    std::vector<Real> r(size_t(n) * n_dim);
    for (size_t i = 0; i < r.size(); ++i)
        r[i] = Real(rng(eng));
    return r;
}

// Alternating +-1 charges: charge neutrality for well-conditioned Ewald.
template <typename Real>
std::vector<Real> generate_charges(int n) {
    std::vector<Real> q(n);
    for (int i = 0; i < n; ++i)
        q[i] = Real(1 - 2 * (i & 1));
    return q;
}

// Exact interleaved [pot, dpot/dx, ...] reference at the first n_cmp sources (self excluded).
bool compute_reference(const Config &cfg, int n_dim, dmk_ikernel kernel, int n, const std::vector<double> &r_src,
                       const std::vector<double> &charges, int n_cmp, std::vector<double> &ref) {
    const int od = 1 + n_dim;
    ref.assign(size_t(n_cmp) * od, 0.0);

    if (!cfg.use_periodic) {
        std::vector<double> r_trg(r_src.begin(), r_src.begin() + size_t(n_cmp) * n_dim);
        dmk::compute_direct(n_dim, r_src, charges, std::vector<double>{}, r_trg, ref, kernel, DMK_POTENTIAL_GRAD,
                            cfg.fparam);
        return true;
    }

    if (kernel == DMK_LAPLACE && n_dim == 2) {
        // The 2D log kernel has no self-contained exact periodic reference (EwaldRef/image_sum don't
        // cover it), so use DMK's own periodic pipeline at eps=1e-12 as the "exact" reference; it
        // shares ESP's log self/gauge convention (validated in test_esp). DMK works in the unit box,
        // so map the centered sweep coords [-L/2,L/2) -> [0,1) by s = r/L + 1/2. For neutral charges
        // the periodic-log potential is unchanged (up to the k=0 gauge run_one removes) and the
        // physical gradient is the unit-box gradient divided by L.
        std::vector<double> r_dmk(size_t(n) * n_dim), rnormal(size_t(n) * n_dim, 0.0);
        for (size_t i = 0; i < r_dmk.size(); ++i)
            r_dmk[i] = r_src[i] / cfg.L + 0.5;

        pdmk_params params;
        params.eps = 1e-12;
        params.n_dim = n_dim;
        params.n_per_leaf = 280;
        params.eval_src = DMK_POTENTIAL_GRAD;
        params.eval_trg = DMK_POTENTIAL_GRAD;
        params.kernel = DMK_LAPLACE;
        params.use_periodic = true;
        params.log_level = 6;
        std::vector<double> pot_src(size_t(n) * od), pot_trg;
        pdmk_tree tree = pdmk_tree_create(MYCOMM, params, n, r_dmk.data(), charges.data(), rnormal.data(), 0, nullptr);
        pdmk_tree_eval(tree, pot_src.data(), pot_trg.data());
        pdmk_tree_destroy(tree);

        for (int i = 0; i < n_cmp; ++i) {
            ref[size_t(i) * od] = pot_src[size_t(i) * od];
            for (int d = 0; d < n_dim; ++d)
                ref[size_t(i) * od + 1 + d] = pot_src[size_t(i) * od + 1 + d] / cfg.L;
        }
        return true;
    }

    if (kernel == DMK_YUKAWA) {
        const int n_img = std::max(2, int(std::ceil(21.0 / (cfg.fparam * cfg.L))));
        std::vector<double> r_trg(r_src.begin(), r_src.begin() + size_t(n_cmp) * n_dim);
        dmk::pbc_ref::image_sum(n_dim, cfg.fparam, n_img, DMK_POTENTIAL_GRAD, n, r_src.data(), charges.data(), cfg.L,
                                n_cmp, r_trg.data(), ref);
        return true;
    }

    dmk::pbc_ref::EwaldRef ewald(kernel, n_dim, n, r_src.data(), charges.data(), cfg.L, 15.0 / cfg.L);
#pragma omp parallel for
    for (int i = 0; i < n_cmp; ++i) {
        double pot, grad[3] = {0, 0, 0};
        ewald.eval(&r_src[size_t(i) * n_dim], i, pot, grad);
        ref[size_t(i) * od] = pot;
        for (int d = 0; d < n_dim; ++d)
            ref[size_t(i) * od + 1 + d] = grad[d];
    }
    return true;
}

template <typename Real>
ErrorMetrics run_one(const Config &cfg, int n_dim, dmk_ikernel kernel, int digits, double r_c,
                     const std::vector<Real> &r_src, const std::vector<Real> &charges, const std::vector<double> &ref,
                     int n_cmp) {
    const int n = cfg.n_src;
    const int od = 1 + n_dim;

    pdmk_esp_params params{};
    params.L = cfg.L;
    params.r_c = r_c;
    params.eps = std::pow(10.0, -digits);
    params.log_level = DMK_LOG_OFF;
    params.kernel = kernel;
    if (kernel == DMK_YUKAWA)
        params.fparam = cfg.fparam;
    params.n_dim = n_dim;
    params.eval_type = DMK_POTENTIAL_GRAD;
    params.sigma = cfg.sigma;
    params.use_periodic = cfg.use_periodic ? 1 : 0;

    pdmk_esp_plan plan = esp_plan_create<Real>(params);
    if (!plan) // e.g. requested precision exceeds the (single-precision) spread-width cap
        throw std::runtime_error(pdmk_last_error_message());
    std::vector<Real> pot(size_t(n) * od);

    double st = MY_OMP_GET_WTIME();
    esp_eval<Real>(plan, n, r_src.data(), charges.data(), pot.data());
    double dt = MY_OMP_GET_WTIME() - st;
    esp_plan_destroy<Real>(plan);

    // Gauge: periodic potential is fixed only up to a constant (charge-neutral fixtures).
    double mean_dmk = 0.0, mean_ref = 0.0;
    if (cfg.use_periodic)
        for (int i = 0; i < n_cmp; ++i) {
            mean_dmk += double(pot[size_t(i) * od]);
            mean_ref += ref[size_t(i) * od];
        }
    mean_dmk /= n_cmp;
    mean_ref /= n_cmp;

    double pe2 = 0, pr2 = 0, pmax = 0, fe2 = 0, fr2 = 0, fmax = 0;
    for (int i = 0; i < n_cmp; ++i) {
        const double pd = double(pot[size_t(i) * od]) - mean_dmk;
        const double pr = ref[size_t(i) * od] - mean_ref;
        const double diff = pd - pr;
        pe2 += diff * diff;
        pr2 += pr * pr;
        if (std::abs(pr) > 0.0)
            pmax = std::max(pmax, std::abs(diff / pr));

        double fd2 = 0, fr2i = 0;
        for (int d = 0; d < n_dim; ++d) {
            // ESP returns force = -q * dpot/dx; reference stores dpot/dx.
            const double f_dmk = double(pot[size_t(i) * od + 1 + d]);
            const double f_ref = -charges[i] * ref[size_t(i) * od + 1 + d];
            const double fd = f_dmk - f_ref;
            fd2 += fd * fd;
            fr2i += f_ref * f_ref;
        }
        fe2 += fd2;
        fr2 += fr2i;
        if (fr2i > 0.0)
            fmax = std::max(fmax, std::sqrt(fd2 / fr2i));
    }

    return {std::sqrt(pe2 / pr2), pmax, std::sqrt(fe2 / fr2), fmax, dt};
}

template <typename Real>
void run_sweep(const Config &cfg) {
    std::vector<dmk_ikernel> kernels;
    if (static_cast<int>(cfg.kernel_filter) == -1)
        kernels = {DMK_LAPLACE, DMK_SQRT_LAPLACE, DMK_YUKAWA};
    else
        kernels = {cfg.kernel_filter};
    std::vector<int> dims = cfg.dim_filter == 0 ? std::vector<int>{2, 3} : std::vector<int>{cfg.dim_filter};

    std::cout << "kernel,dim,digits,eps_fu,r_c,sigma,P,c,n_f,pot_l2,pot_max,force_l2,force_max,time\n" << std::flush;

    const int n = cfg.n_src;
    const int n_cmp = std::min(cfg.n_direct, n);

    for (auto kernel : kernels) {
        for (auto n_dim : dims) {
            auto r_src_d = generate_positions<double>(n, n_dim, cfg.L, cfg.seed);
            auto charges_d = generate_charges<double>(n);

            std::vector<double> ref;
            bool have_ref = compute_reference(cfg, n_dim, kernel, n, r_src_d, charges_d, n_cmp, ref);
            if (!have_ref) {
                std::cout << "# " << dmk::util::to_string(kernel) << " dim=" << n_dim
                          << " periodic: no self-contained reference, skipping\n"
                          << std::flush;
                continue;
            }

            std::vector<Real> r_src(r_src_d.begin(), r_src_d.end());
            std::vector<Real> charges(charges_d.begin(), charges_d.end());

            for (int digits = cfg.dig_min; digits <= cfg.dig_max; ++digits) {
                const double eps_fu = std::pow(10.0, -digits);
                const int P = dmk::esp_P_from_eps(eps_fu, cfg.sigma, n_dim);
                const double c = dmk::esp_beta_from_P(cfg.sigma, P);
                for (double frac = cfg.rc_min; frac <= cfg.rc_max + 1e-9; frac += cfg.rc_step) {
                    const double r_c = frac * cfg.L;
                    const double pad =
                        cfg.use_periodic ? 1.0 : 2.2 * (std::sqrt(double(n_dim)) * cfg.L + 2.0 * r_c) / cfg.L;
                    const int n_f = int(std::ceil(c * pad * cfg.L / (M_PI * r_c)));
                    try {
                        auto e = run_one<Real>(cfg, n_dim, kernel, digits, r_c, r_src, charges, ref, n_cmp);
                        std::cout << dmk::util::to_string(kernel) << "," << n_dim << "," << digits << ","
                                  << std::scientific << std::setprecision(1) << eps_fu << "," << std::fixed
                                  << std::setprecision(4) << r_c << "," << cfg.sigma << "," << P << "," << std::fixed
                                  << std::setprecision(3) << c << "," << n_f << "," << std::scientific
                                  << std::setprecision(4) << e.pot_l2 << "," << e.pot_max << "," << e.force_l2 << ","
                                  << e.force_max << "," << std::fixed << std::setprecision(4) << e.time << "\n"
                                  << std::flush;
                    } catch (std::exception &ex) {
                        std::cout << dmk::util::to_string(kernel) << "," << n_dim << "," << digits << "," << eps_fu
                                  << "," << r_c << "," << cfg.sigma << "," << P << "," << c << "," << n_f
                                  << ",FAILED,FAILED,FAILED,FAILED,0\n"
                                  << std::flush;
                    }
                }
            }
        }
    }
}

Config parse_args(int argc, char *argv[]) {
    Config cfg;
    static struct option long_opts[] = {
        {"dig-min", required_argument, nullptr, 1001},
        {"dig-max", required_argument, nullptr, 1002},
        {"rc-min", required_argument, nullptr, 1003},
        {"rc-max", required_argument, nullptr, 1004},
        {"rc-step", required_argument, nullptr, 1005},
        {"seed", required_argument, nullptr, 1006},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "N:D:t:k:d:l:L:os:h", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'N':
            cfg.n_src = int(std::atof(optarg));
            break;
        case 'D':
            cfg.n_direct = int(std::atof(optarg));
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
        case 'k':
            if (std::string_view(optarg) == "all")
                cfg.kernel_filter = static_cast<dmk_ikernel>(-1);
            else if (auto k = dmk::util::ikernel_from_string(optarg))
                cfg.kernel_filter = *k;
            else {
                std::cerr << "Unknown kernel: " << optarg << " (use laplace, sqrt_laplace, yukawa, all)\n";
                exit(1);
            }
            break;
        case 'd':
            cfg.dim_filter = std::atoi(optarg);
            break;
        case 'l':
            cfg.fparam = std::atof(optarg);
            break;
        case 'L':
            cfg.L = std::atof(optarg);
            break;
        case 'o':
            cfg.use_periodic = false;
            break;
        case 's':
            cfg.sigma = std::atof(optarg);
            cfg.sigma_set = true;
            break;
        case 1001:
            cfg.dig_min = std::atoi(optarg);
            break;
        case 1002:
            cfg.dig_max = std::atoi(optarg);
            break;
        case 1003:
            cfg.rc_min = std::atof(optarg);
            break;
        case 1004:
            cfg.rc_max = std::atof(optarg);
            break;
        case 1005:
            cfg.rc_step = std::atof(optarg);
            break;
        case 1006:
            cfg.seed = std::atol(optarg);
            break;
        case 'h':
        default:
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  -N n_src        Number of source points (default 100000)\n"
                      << "  -D n_direct     Points compared against reference (default 10000)\n"
                      << "  -t f|d          Precision (default d)\n"
                      << "  -k kernel       laplace, sqrt_laplace, yukawa, all (default all)\n"
                      << "  -d dim          2, 3, or 0 for both (default 0)\n"
                      << "  -l lambda       Yukawa fparam (default 6.0)\n"
                      << "  -L box          Box side length (default 1.0)\n"
                      << "  -o              Free-space (open) boundaries instead of periodic\n"
                      << "  -s sigma        FINUFFT upsampling (default 1.35; != 1.35 requires -DDMK_USE_JIT=ON)\n"
                      << "  --dig-min val   Min solver digits (default 3)\n"
                      << "  --dig-max val   Max solver digits (default 9)\n"
                      << "  --rc-min val    Min r_c as fraction of L (default 0.05)\n"
                      << "  --rc-max val    Max r_c as fraction of L (default 0.25)\n"
                      << "  --rc-step val   r_c fraction step (default 0.05)\n"
                      << "  --seed val      RNG seed for source positions (default 42)\n";
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
            std::cerr << "measure_error_esp not MPI aware. Exiting\n";
        MPI_Finalize();
        return 0;
    }
#endif

    Config cfg = parse_args(argc, argv);

#ifndef DMK_USE_JIT
    if (cfg.sigma_set && cfg.sigma != 1.35) {
        std::cerr << "error: -s sigma != 1.35 requires JIT support (recompile with -DDMK_USE_JIT=ON)\n";
        return 1;
    }
#endif

    std::cout << "# n_src=" << cfg.n_src << " n_direct=" << cfg.n_direct << " prec=" << cfg.prec << " L=" << cfg.L
              << " boundary=" << (cfg.use_periodic ? "periodic" : "free-space") << " sigma=" << cfg.sigma
              << " fparam=" << cfg.fparam << " digits=[" << cfg.dig_min << "," << cfg.dig_max << "] rc_frac=["
              << cfg.rc_min << "," << cfg.rc_max << "," << cfg.rc_step << "] seed=" << cfg.seed
              << " threads=" << MY_OMP_GET_MAX_THREADS() << "\n\n";

    try {
        if (cfg.prec == 'd')
            run_sweep<double>(cfg);
        else
            run_sweep<float>(cfg);
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

#ifdef DMK_HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
