#ifdef DMK_BUILD_ESP

#include <dmk.h>
#include <dmk/omp_wrapper.hpp>
#include <dmk/util.hpp>
#include <sctl.hpp>

#include "periodic_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#ifdef DMK_HAVE_MPI
#error "ESP does not support MPI. DMK_BUILD_ESP and DMK_HAVE_MPI are mutually exclusive."
#endif

struct Config {
    int n_src = 100'000;
    int n_dim = 3; // -d: spatial dimension (2 or 3)
    double L = 1.0;
    double r_c = -1.0;
    double eps = 1e-6;
    dmk_ikernel kernel = DMK_LAPLACE; // -k: laplace, sqrt_laplace, or yukawa
    double fparam = 6.0;              // -f: Yukawa screening parameter (used only for -k yukawa)
    int n_runs = 10;
    int n_direct = 10'000; // -D: compare only the first n_direct points against the reference (0 => skip)
    char prec = 'd';
    double sigma = 1.35;
    bool sigma_set = false;
    bool bench_plan = false;
    bool bench_forces = false; // -g: also benchmark forces (potential+force timed together)
    bool check_forces =
        false; // -F: validate forces against FD reference (samples 20 random particles × 6 evals each); implies -g
    int log_level = 6; // DMK_LOG_OFF
    // Short-range method selection (defaults to the fastest combo); mirrors pdmk_esp_params.
    uint32_t esp_flags = DMK_ESP_PRUNE_SOURCE | DMK_ESP_N3L | DMK_ESP_MORTON;
    int esp_bins = 2;
    int esp_stile = 0;
};

// Build a pdmk_esp_params from the benchmark Config; r_c and eval_type are passed explicitly since
// callers need to sweep r_c (init_sensible_defaults) or vary eval_type (potential-only FD reference
// vs the analytic potential+grad plan) independent of the rest of the config.
pdmk_esp_params make_params(const Config &cfg, double r_c, dmk_eval_type eval_type) {
    pdmk_esp_params params{};
    params.L = cfg.L;
    params.r_c = r_c;
    params.eps = cfg.eps;
    params.n_dim = cfg.n_dim;
    params.kernel = cfg.kernel;
    if (cfg.kernel == DMK_YUKAWA)
        params.fparam = cfg.fparam;
    params.log_level = cfg.log_level;
    params.eval_type = eval_type;
    params.sigma = cfg.sigma;
    params.esp_flags = cfg.esp_flags;
    params.esp_bins = cfg.esp_bins;
    params.esp_stile = cfg.esp_stile;
    return params;
}

// One-line render of the short-range config for the benchmark header.
std::string sr_summary(uint32_t esp_flags, int esp_bins, int esp_stile) {
    const bool n3l = esp_flags & DMK_ESP_N3L;
    const bool prune_source = esp_flags & DMK_ESP_PRUNE_SOURCE;
    const bool prune_tile = esp_flags & DMK_ESP_PRUNE_TILE;
    const bool morton = esp_flags & DMK_ESP_MORTON;
    std::string method = n3l ? "n3l" : prune_source ? "prune_source" : prune_tile ? "prune_tile" : "dense";
    std::string s = method + (morton ? " morton" : " bins=" + std::to_string(esp_bins));
    if (esp_stile > 0)
        s += " stile=" + std::to_string(esp_stile);
    return s;
}

// Precision-dispatched wrappers over the public C API's separate _create/_createf, _eval/_evalf,
// _destroy/_destroyf entry points, so the benchmark's Real-templated functions can call one name.
template <typename Real>
pdmk_esp_plan esp_plan_create(dmk_communicator comm, pdmk_esp_params params) {
    if constexpr (std::is_same_v<Real, float>)
        return pdmk_esp_plan_createf(comm, params);
    else
        return pdmk_esp_plan_create(comm, params);
}
template <typename Real>
void esp_eval(dmk_communicator comm, pdmk_esp_plan plan, int n, const Real *r_src, const Real *charges, Real *pot_src) {
    if constexpr (std::is_same_v<Real, float>)
        pdmk_esp_evalf(comm, plan, n, r_src, charges, pot_src);
    else
        pdmk_esp_eval(comm, plan, n, r_src, charges, pot_src);
}
template <typename Real>
void esp_plan_destroy(pdmk_esp_plan plan) {
    if constexpr (std::is_same_v<Real, float>)
        pdmk_esp_plan_destroyf(plan);
    else
        pdmk_esp_plan_destroy(plan);
}

// Generate N random flat [n*n_dim] positions in [-L/2, L/2)^n_dim.
template <typename Real>
std::vector<Real> generate_positions(int n, int n_dim, double L, long seed = 42) {
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> rng(-0.5 * L, 0.5 * L);
    std::vector<Real> r(size_t(n) * n_dim);
    for (size_t i = 0; i < r.size(); ++i)
        r[i] = Real(rng(eng));
    return r;
}

// Alternating ±1 charges — ensures charge neutrality for well-conditioned Ewald.
template <typename Real>
std::vector<Real> generate_charges(int n) {
    std::vector<Real> q(n);
    for (int i = 0; i < n; ++i)
        q[i] = Real(1 - 2 * (i & 1));
    return q;
}

// Periodic reference potential at the first n_direct sources (self-interaction excluded). Returns
// false if no self-contained reference is available (2D Laplace: the log kernel's ESP self term
// matches the DMK pipeline, not the analytic Ewald sum, and is only defined up to a gauge -- its
// accuracy is validated in test_esp.cpp). Laplace-3D / Sqrt-Laplace use the shared Ewald reference
// (structure factor over all n sources, evaluated at n_direct points -> O(n_direct); the cell list
// prunes at scale). Yukawa is screened, so a parallel near-image direct sum of exp(-lambda r)/r
// suffices. Only n_direct points are evaluated.
bool compute_reference(const Config &cfg, int n, const std::vector<double> &r_src_d,
                       const std::vector<double> &charges_d, std::vector<double> &ref) {
    const int nd = cfg.n_dim;
    const int n_cmp = std::min(cfg.n_direct, n);
    const double L = cfg.L;

    if (cfg.kernel == DMK_LAPLACE && nd == 2) {
        std::cout << "# verify: skipping accuracy check for 2D Laplace (log gauge/self is validated in "
                     "test_esp.cpp, not self-contained here)\n"
                  << std::flush;
        return false;
    }

    ref.assign(n_cmp, 0.0);
    std::cout << "# verify: computing reference for first " << n_cmp << " of " << n << " points...\n" << std::flush;

    if (cfg.kernel == DMK_YUKAWA) {
        const double lambda = cfg.fparam;
        const int n_img = std::max(2, int(std::ceil(21.0 / (lambda * L)))); // exp(-lambda*n_img*L) ~ 1e-9
        const int mz_lo = nd == 3 ? -n_img : 0, mz_hi = nd == 3 ? n_img : 0;
#pragma omp parallel for
        for (int i = 0; i < n_cmp; ++i) {
            double pot = 0.0;
            for (int j = 0; j < n; ++j)
                for (int mx = -n_img; mx <= n_img; ++mx)
                    for (int my = -n_img; my <= n_img; ++my)
                        for (int mz = mz_lo; mz <= mz_hi; ++mz) {
                            const double d0 = r_src_d[i * nd + 0] - r_src_d[j * nd + 0] - mx * L;
                            const double d1 = r_src_d[i * nd + 1] - r_src_d[j * nd + 1] - my * L;
                            double r2 = d0 * d0 + d1 * d1;
                            if (nd == 3) {
                                const double d2 = r_src_d[i * nd + 2] - r_src_d[j * nd + 2] - mz * L;
                                r2 += d2 * d2;
                            }
                            if (r2 > 1e-28) {
                                const double r = std::sqrt(r2);
                                pot += charges_d[j] * std::exp(-lambda * r) / r;
                            }
                        }
            ref[i] = pot;
        }
    } else {
        dmk::pbc_ref::EwaldRef ewald(cfg.kernel, nd, n, r_src_d.data(), charges_d.data(), L, 15.0 / L);
#pragma omp parallel for
        for (int i = 0; i < n_cmp; ++i) {
            double pot;
            ewald.eval(&r_src_d[i * nd], i, pot, nullptr);
            ref[i] = pot;
        }
    }
    return true;
}

void print_config(const Config &cfg, int n_threads, std::ostream &os) {
    os << "# n_src:       " << cfg.n_src << "\n"
       << "# n_dim:       " << cfg.n_dim << "\n"
       << "# L:           " << cfg.L << "\n"
       << "# r_c:         " << cfg.r_c << "\n"
       << "# eps:         " << cfg.eps << "\n"
       << "# kernel:      " << dmk::util::to_string(cfg.kernel) << "\n"
       << "# fparam:      " << cfg.fparam << "\n"
       << "# sigma:       " << cfg.sigma << "\n"
       << "# n_runs:      " << cfg.n_runs << "\n"
       << "# n_direct:    " << cfg.n_direct << "\n"
       << "# prec:        " << (cfg.prec == 'd' ? "double" : "float") << "\n"
       << "# bench_plan:  " << (cfg.bench_plan ? "true" : "false") << "\n"
       << "# bench_forces:" << (cfg.bench_forces ? "true" : "false") << "\n"
       << "# check_forces:" << (cfg.check_forces ? "true" : "false") << "\n"
       << "# log_level:   " << cfg.log_level << "\n"
       << "# short_range: " << sr_summary(cfg.esp_flags, cfg.esp_bins, cfg.esp_stile) << "\n"
       << "# omp_threads: " << n_threads << "\n";
}

// Compares only the first ref.size() entries of pot (the n_direct points the reference covers).
template <typename Real>
double l2_rel_err(std::span<Real> pot, const std::vector<double> &ref) {
    const int n = static_cast<int>(ref.size());
    double err2 = 0, ref2 = 0;
    for (int i = 0; i < n; ++i) {
        const double d = double(pot[i]) - ref[i];
        err2 += d * d;
        ref2 += ref[i] * ref[i];
    }
    return std::sqrt(err2 / ref2);
}

// Auto-select r_c if it wasn't explicitly given (for accuracy).
// Sweeps all candidates and picks the one with the smallest l2_rel_err against the Ewald reference.
void init_sensible_defaults(Config &cfg, const std::vector<double> &r_src_d, const std::vector<double> &charges_d,
                            const std::vector<double> &ref, bool have_ref) {
    if (cfg.r_c != -1.0)
        return;

    const std::vector<double> rc_candidates = {0.03 * cfg.L, 0.05 * cfg.L, 0.07 * cfg.L, 0.10 * cfg.L, 0.12 * cfg.L};

    if (!have_ref) {
        cfg.r_c = rc_candidates[rc_candidates.size() / 2];
        std::cout << "# init_sensible_defaults: no reference, defaulting to r_c=" << cfg.r_c << "\n" << std::flush;
        return;
    }

    const int n = static_cast<int>(charges_d.size());
    double best_l2 = std::numeric_limits<double>::max();
    for (double rc : rc_candidates) {
        pdmk_esp_params params = make_params(cfg, rc, DMK_POTENTIAL);
        pdmk_esp_plan plan = pdmk_esp_plan_create(nullptr, params);
        std::vector<double> pot(n);
        pdmk_esp_eval(nullptr, plan, n, r_src_d.data(), charges_d.data(), pot.data());
        const double l2 = l2_rel_err(std::span<double>(pot), ref);
        pdmk_esp_plan_destroy(plan);

        if (l2 < best_l2) {
            best_l2 = l2;
            cfg.r_c = rc;
        }
    }

    std::cout << "# init_sensible_defaults: selected r_c=" << cfg.r_c << " (l2_rel_err=" << best_l2 << ")\n"
              << std::flush;
}

// Validates analytic forces (pot_src_grad, interleaved [pot,fx,fy,fz] per particle from a
// DMK_POTENTIAL_GRAD plan) against central-difference FD for a random sample of n_sample particles.
// Builds its own DMK_POTENTIAL (cheaper) plan for the FD evaluations.
double check_forces_fd(const std::vector<double> &pot_src_grad, const std::vector<double> &r_src_d,
                       const std::vector<double> &charges_d, pdmk_esp_params params, int n_sample = 20) {
    const int n = static_cast<int>(charges_d.size());
    const int nd = params.n_dim;
    const int out_dim = 1 + nd;
    n_sample = std::min(n_sample, n);
    const double step = 1e-12;

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::default_random_engine(123));
    idx.resize(n_sample);

    params.eval_type = DMK_POTENTIAL;
    pdmk_esp_plan plan = pdmk_esp_plan_create(nullptr, params);

    std::vector<double> r_pert = r_src_d;
    std::vector<double> pot(n);
    double err2 = 0, ref2 = 0;
    for (int i : idx) {
        for (int a = 0; a < nd; ++a) {
            r_pert[nd * i + a] = r_src_d[nd * i + a] + step;
            pdmk_esp_eval(nullptr, plan, n, r_pert.data(), charges_d.data(), pot.data());
            double pot_plus = pot[i];

            r_pert[nd * i + a] = r_src_d[nd * i + a] - step;
            pdmk_esp_eval(nullptr, plan, n, r_pert.data(), charges_d.data(), pot.data());
            double pot_minus = pot[i];

            r_pert[nd * i + a] = r_src_d[nd * i + a];

            const double f_ref = -charges_d[i] * (pot_plus - pot_minus) / (2.0 * step);
            const double diff = pot_src_grad[i * out_dim + 1 + a] - f_ref;
            err2 += diff * diff;
            ref2 += f_ref * f_ref;
        }
    }

    pdmk_esp_plan_destroy(plan);
    return std::sqrt(err2 / ref2);
}

template <typename Real>
void warmup(const Config &cfg) {
    constexpr int nw = 100'000;
    auto r_w = generate_positions<Real>(nw, cfg.n_dim, cfg.L);
    auto q_w = generate_charges<Real>(nw);
    pdmk_esp_params params = make_params(cfg, cfg.r_c, DMK_POTENTIAL);
    pdmk_esp_plan plan = esp_plan_create<Real>(nullptr, params);
    std::vector<Real> pot(nw);
    esp_eval<Real>(nullptr, plan, nw, r_w.data(), q_w.data(), pot.data());
    esp_plan_destroy<Real>(plan);
}

// Potential-only benchmark: uses a DMK_POTENTIAL plan (no force computation at all), and reports
// l2_rel_err against the Ewald reference when available.
template <typename Real>
void run_potential_benchmark(const Config &cfg, int n, const std::vector<Real> &r_src, const std::vector<Real> &charges,
                             const std::vector<double> &ref, bool have_ref) {
    pdmk_esp_params params = make_params(cfg, cfg.r_c, DMK_POTENTIAL);
    pdmk_esp_plan plan = esp_plan_create<Real>(nullptr, params);

    std::cout << "# phase: eval_potential\n";
    std::cout << "run,total_time,pts_per_s,l2_rel_err,";
    pdmk_print_profile_data(nullptr, 'h');
    std::cout << "\n" << std::flush;

    std::vector<Real> pot(n);
    for (int run = 0; run < cfg.n_runs; ++run) {
        sctl::Profile::reset();
        double t0 = MY_OMP_GET_WTIME();
        esp_eval<Real>(nullptr, plan, n, r_src.data(), charges.data(), pot.data());
        double t1 = MY_OMP_GET_WTIME();

        std::cout << run << "," << (t1 - t0) << "," << n / (t1 - t0) << ",";
        std::cout << (have_ref ? l2_rel_err(std::span<Real>(pot), ref) : std::numeric_limits<double>::quiet_NaN());
        std::cout << ",";
        pdmk_print_profile_data(nullptr, 'c');
        std::cout << "\n" << std::flush;
    }

    esp_plan_destroy<Real>(plan);
}

// Forces benchmark: uses a DMK_POTENTIAL_GRAD plan, so total_time reflects
// potential and force computed together. When cfg.check_forces is true (user ran -g -F),
// validates analytic forces against FD after the timing loop.
template <typename Real>
void run_forces_benchmark(const Config &cfg, int n, const std::vector<Real> &r_src, const std::vector<Real> &charges,
                          const std::vector<double> &r_src_d, const std::vector<double> &charges_d) {
    pdmk_esp_params params = make_params(cfg, cfg.r_c, DMK_POTENTIAL_GRAD);
    pdmk_esp_plan plan = esp_plan_create<Real>(nullptr, params);

    std::cout << "# phase: eval_forces\n";
    std::cout << "run,total_time,pts_per_s,";
    pdmk_print_profile_data(nullptr, 'h');
    std::cout << "\n" << std::flush;

    std::vector<Real> pot(size_t(n) * (1 + cfg.n_dim));
    for (int run = 0; run < cfg.n_runs; ++run) {
        sctl::Profile::reset();
        double t0 = MY_OMP_GET_WTIME();
        esp_eval<Real>(nullptr, plan, n, r_src.data(), charges.data(), pot.data());
        double t1 = MY_OMP_GET_WTIME();

        std::cout << run << "," << (t1 - t0) << "," << n / (t1 - t0);
        std::cout << ",";
        pdmk_print_profile_data(nullptr, 'c');
        std::cout << "\n" << std::flush;
    }

    if (cfg.check_forces) {
        constexpr int n_fd_sample = 20;
        std::cout << "# phase: force_check (FD on " << n_fd_sample << " random particles × 6 evals each; not all N)\n"
                  << std::flush;
        std::vector<double> pot_d(size_t(n) * (1 + cfg.n_dim));
        pdmk_esp_eval(nullptr, plan, n, r_src_d.data(), charges_d.data(), pot_d.data());
        double force_l2_err = check_forces_fd(pot_d, r_src_d, charges_d, params, n_fd_sample);
        std::cout << "# force_check: l2_rel_err=" << force_l2_err << "\n" << std::flush;
    }

    esp_plan_destroy<Real>(plan);
}

template <typename Real>
void run_benchmark(Config cfg) {
    sctl::Profile::Enable(true);
    const int n_threads = MY_OMP_GET_MAX_THREADS();
    const int n = cfg.n_src;

    auto r_src_d = generate_positions<double>(n, cfg.n_dim, cfg.L);
    auto charges_d = generate_charges<double>(n);

    std::vector<double> ref;
    bool have_ref = false;
    if (cfg.n_direct > 0)
        have_ref = compute_reference(cfg, n, r_src_d, charges_d, ref);

    init_sensible_defaults(cfg, r_src_d, charges_d, ref, have_ref);
    print_config(cfg, n_threads, std::cout);

    std::vector<Real> r_src(size_t(n) * cfg.n_dim), charges(n);
    for (size_t i = 0; i < r_src.size(); ++i)
        r_src[i] = Real(r_src_d[i]);
    for (int i = 0; i < n; ++i)
        charges[i] = Real(charges_d[i]);

    // ---- Optionally benchmark plan creation --------------------------------
    if (cfg.bench_plan) {
        std::cout << "# phase: plan_create\n" << "run,plan_time,pts_per_s\n" << std::flush;
        pdmk_esp_params params = make_params(cfg, cfg.r_c, DMK_POTENTIAL_GRAD);
        for (int run = 0; run < cfg.n_runs; ++run) {
            double t0 = MY_OMP_GET_WTIME();
            pdmk_esp_plan plan = esp_plan_create<Real>(nullptr, params);
            double t1 = MY_OMP_GET_WTIME();
            esp_plan_destroy<Real>(plan);
            std::cout << run << "," << (t1 - t0) << "," << n / (t1 - t0) << "\n" << std::flush;
        }
    }

    warmup<Real>(cfg);
    if (cfg.check_forces && !cfg.bench_forces) {
        // Force correctness check only (no timing benchmark).
        pdmk_esp_params params = make_params(cfg, cfg.r_c, DMK_POTENTIAL_GRAD);
        pdmk_esp_plan plan = esp_plan_create<Real>(nullptr, params);
        constexpr int n_fd_sample = 20;
        std::cout << "# phase: force_check (FD on " << n_fd_sample << " random particles × 6 evals each; not all N)\n"
                  << std::flush;
        std::vector<double> pot_d(size_t(n) * (1 + cfg.n_dim));
        pdmk_esp_eval(nullptr, plan, n, r_src_d.data(), charges_d.data(), pot_d.data());
        double err = check_forces_fd(pot_d, r_src_d, charges_d, params, n_fd_sample);
        std::cout << "# force_check: l2_rel_err=" << err << "\n" << std::flush;
        esp_plan_destroy<Real>(plan);
    } else if (cfg.bench_forces) {
        run_forces_benchmark<Real>(cfg, n, r_src, charges, r_src_d, charges_d);
    } else {
        run_potential_benchmark<Real>(cfg, n, r_src, charges, ref, have_ref);
    }
}

// Long-only options for the short-range method selection (formerly the DMK_ESP_* env vars).
enum {
    OPT_PRUNE = 256,
    OPT_N3L,
    OPT_MORTON,
    OPT_BINS,
    OPT_STILE,
};

Config parse_args(int argc, char *argv[]) {
    Config cfg;
    static const struct option long_opts[] = {
        {"prune", required_argument, nullptr, OPT_PRUNE},   {"n3l", required_argument, nullptr, OPT_N3L},
        {"morton", required_argument, nullptr, OPT_MORTON}, {"bins", required_argument, nullptr, OPT_BINS},
        {"stile", required_argument, nullptr, OPT_STILE},   {nullptr, 0, nullptr, 0}};
    auto set_flag = [&](unsigned bit, bool on) {
        if (on)
            cfg.esp_flags |= bit;
        else
            cfg.esp_flags &= ~bit;
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "N:d:L:c:e:r:t:l:s:D:k:f:pFgh?", long_opts, nullptr)) != -1) {
        switch (opt) {
        case OPT_PRUNE: {
            const int v = std::atoi(optarg);
            set_flag(DMK_ESP_PRUNE_TILE, v == 1);
            set_flag(DMK_ESP_PRUNE_SOURCE, v >= 2);
            break;
        }
        case OPT_N3L:
            set_flag(DMK_ESP_N3L, std::atoi(optarg) != 0);
            break;
        case OPT_MORTON:
            set_flag(DMK_ESP_MORTON, std::atoi(optarg) != 0);
            break;
        case OPT_BINS:
            cfg.esp_bins = std::atoi(optarg);
            break;
        case OPT_STILE:
            cfg.esp_stile = std::atoi(optarg);
            break;
        case 'N':
            cfg.n_src = int(std::atof(optarg));
            break;
        case 'd':
            cfg.n_dim = std::atoi(optarg);
            if (cfg.n_dim != 2 && cfg.n_dim != 3) {
                std::cerr << "Invalid dimension: " << optarg << " (must be 2 or 3)\n";
                exit(1);
            }
            break;
        case 'L':
            cfg.L = std::atof(optarg);
            break;
        case 'c':
            cfg.r_c = std::atof(optarg);
            break;
        case 'e':
            cfg.eps = std::atof(optarg);
            break;
        case 'r':
            cfg.n_runs = std::atoi(optarg);
            break;
        case 'D':
            cfg.n_direct = int(std::atof(optarg));
            break;
        case 'k':
            if (auto k = dmk::util::ikernel_from_string(optarg))
                cfg.kernel = *k;
            else {
                std::cerr << "Unknown kernel: " << optarg << " (use laplace, sqrt_laplace, or yukawa)\n";
                exit(1);
            }
            break;
        case 'f':
            cfg.fparam = std::atof(optarg);
            break;
        case 'l':
            cfg.log_level = std::atoi(optarg);
            break;
        case 'p':
            cfg.bench_plan = true;
            break;
        case 'g':
            cfg.bench_forces = true;
            break;
        case 'F':
            cfg.check_forces = true;
            break;
        case 's':
            cfg.sigma = std::atof(optarg);
            cfg.sigma_set = true;
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
        case 'h':
        case '?':
        default:
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  -N n       Number of particles (default 100000)\n"
                      << "  -d dim     Spatial dimension: 2 or 3 (default 3)\n"
                      << "  -L L       Box side length (default 1.0)\n"
                      << "  -c r_c     Real-space cutoff (default: auto-picked to balance short-/long-range time)\n"
                      << "  -e eps     Tolerance (default 1e-6)\n"
                      << "  -k kernel  laplace, sqrt_laplace, or yukawa (default laplace)\n"
                      << "  -f fparam  Yukawa screening parameter (default 6.0)\n"
                      << "  -r n_runs  Benchmark iterations (default 10)\n"
                      << "  -D n       Compare only the first n points against the reference (default 10000; 0 skips)\n"
                      << "  -t f|d     Precision: float or double (default d)\n"
                      << "  -l level   Log verbosity 0-6 (default 6=off)\n"
                      << "  -p         Also benchmark plan creation\n"
                      << "  -s sigma   FINUFFT upsampling factor for the long-range PSWF kernel (default 1.35).\n"
                      << "             Requires JIT support (-DDMK_USE_JIT=ON at configure time).\n"
                      << "  -g         Benchmark forces instead of potential (potential+force timed together).\n"
                      << "  -F         Validate forces against a finite-difference reference (no benchmark).\n"
                      << "             Samples 20 random particles, not all N.\n"
                      << "  -h         Help\n"
                      << "\n"
                      << "Short-range method (default: the fastest combo --n3l 1 --morton 1):\n"
                      << "  --n3l N         1=Newton's-third-law reciprocal half-stencil (takes precedence\n"
                      << "                  over --prune), 0=off\n"
                      << "  --prune N       source pruning used when --n3l 0: 0=dense, 1=tile-vs-tile,\n"
                      << "                  2=per-source (default 2)\n"
                      << "  --morton N      1=Morton within-cell sort, 0=octant-bin counting sort\n"
                      << "  --bins N        sub-boxes per axis for the bin sort when --morton 0 (default 2)\n"
                      << "  --stile N       source-tile width for --prune 1 (default: SIMD width)\n"
                      << "\n"
                      << "Output CSV columns (eval phase):\n"
                      << "  run, total_time, pts_per_s[, l2_rel_err]\n"
                      << "  Per-phase breakdown (short_range/long_range/self_interaction) printed after\n"
                      << "  each phase when built with -DDMK_INSTRUMENT=ON.\n";
            exit(0);
        }
    }
    return cfg;
}

int main(int argc, char *argv[]) {
    Config cfg = parse_args(argc, argv);

#ifndef DMK_USE_JIT
    if (cfg.sigma_set) {
        std::cerr << "error: -s sigma requires JIT support (recompile with -DDMK_USE_JIT=ON)\n";
        return 1;
    }
#endif

    if (cfg.prec == 'd')
        run_benchmark<double>(cfg);
    else
        run_benchmark<float>(cfg);
    return 0;
}

#else // DMK_BUILD_ESP
#include <iostream>
int main() {
    std::cerr << "benchmark_esp requires -DDMK_BUILD_ESP=ON at configure time.\n";
    return 1;
}
#endif // DMK_BUILD_ESP
