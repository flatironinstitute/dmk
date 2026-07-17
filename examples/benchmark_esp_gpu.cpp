#if defined(DMK_BUILD_ESP) && defined(DMK_GPU_OFFLOAD)

#include <dmk/esp.hpp>
#include <dmk/omp_wrapper.hpp>
#include <sctl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#ifdef DMK_HAVE_MPI
#error "ESP does not support MPI. DMK_BUILD_ESP and DMK_HAVE_MPI are mutually exclusive."
#endif

struct Config {
    int n_src = 100'000;
    double L = 1.0;
    double r_c = -1.0;
    double eps = 1e-6;
    int n_runs = 10;
    char prec = 'd';
    double sigma = 1.35;
    bool bench_plan = false;
    bool bench_forces = false;    // -g: also benchmark forces (potential+force timed together)
    bool check_forces = false;    // -F: validate GPU forces against FD reference; implies -g
    bool skip_cpu_baseline = false; // -S: skip the CPU reference run (also disables l2_rel_err reporting)
};

// Generate N random positions in [-L/2, L/2)^3.
template <typename Real>
std::vector<dmk::Vec3T<Real>> generate_positions(int n, double L, long seed = 42) {
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> rng(-0.5 * L, 0.5 * L);
    std::vector<dmk::Vec3T<Real>> r(n);
    for (int i = 0; i < n; ++i)
        r[i] = {Real(rng(eng)), Real(rng(eng)), Real(rng(eng))};
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

void print_config(const Config &cfg, std::ostream &os) {
    os << "# n_src:             " << cfg.n_src << "\n"
       << "# L:                 " << cfg.L << "\n"
       << "# r_c:               " << cfg.r_c << "\n"
       << "# eps:               " << cfg.eps << "\n"
       << "# sigma:             " << cfg.sigma << "\n"
       << "# n_runs:            " << cfg.n_runs << "\n"
       << "# prec:              " << (cfg.prec == 'd' ? "double" : "float") << "\n"
       << "# bench_plan:        " << (cfg.bench_plan ? "true" : "false") << "\n"
       << "# bench_forces:      " << (cfg.bench_forces ? "true" : "false") << "\n"
       << "# check_forces:      " << (cfg.check_forces ? "true" : "false") << "\n"
       << "# skip_cpu_baseline: " << (cfg.skip_cpu_baseline ? "true" : "false") << "\n";
}

// GPU-vs-CPU relative L2 error, gauged (mean-subtracted) on both sides first: ESP potentials
// for a charge-neutral system carry an arbitrary additive constant, so an ungauged comparison
// would report a spurious error even when GPU and CPU agree on the physically meaningful part.
// Same idea as test_esp_gpu.cpp's gauge()+l2_rel(), just against a CPU reference instead of a
// pointwise CPU/GPU test.
template <typename Real>
double l2_rel_err(std::span<Real> gpu_pot, std::span<Real> cpu_pot) {
    const int n = static_cast<int>(gpu_pot.size());
    double gpu_mean = 0, cpu_mean = 0;
    for (int i = 0; i < n; ++i) {
        gpu_mean += double(gpu_pot[i]);
        cpu_mean += double(cpu_pot[i]);
    }
    gpu_mean /= n;
    cpu_mean /= n;
    double err2 = 0, ref2 = 0;
    for (int i = 0; i < n; ++i) {
        const double d = (double(gpu_pot[i]) - gpu_mean) - (double(cpu_pot[i]) - cpu_mean);
        const double r = double(cpu_pot[i]) - cpu_mean;
        err2 += d * d;
        ref2 += r * r;
    }
    return std::sqrt(err2 / ref2);
}

// Validates GPU analytic forces against central-difference FD, using a fresh DMK_POTENTIAL-only
// GPU plan for the perturbed-position evals (only potential is needed for FD; matches
// benchmark_esp.cpp's check_forces_fd, swapping esp_eval for esp_eval_gpu).
// Step size is much larger than the CPU version's 1e-12: cuFINUFFT's plan tolerance (~eps) puts
// a noise floor under the GPU potential eval, so the FD step needs step^2 << eps/step, i.e.
// step ~ eps^(1/3), not CPU's near-machine-precision step.
double check_forces_fd_gpu(const dmk::PotForce<double> &esp, const std::vector<dmk::Vec3T<double>> &r_src_d,
                           const std::vector<double> &charges_d, double L, double r_c, double eps, double sigma,
                           int n_sample = 20) {
    const int n = static_cast<int>(charges_d.size());
    n_sample = std::min(n_sample, n);
    const double step = std::cbrt(eps);

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::default_random_engine(123));
    idx.resize(n_sample);

    dmk::EspPlan *plan = dmk::esp_create_plan(L, r_c, eps, sigma, DMK_POTENTIAL);
    dmk::GpuState *gpu = dmk::esp_create_gpu_plan(plan);

    std::vector<dmk::Vec3T<double>> r_pert = r_src_d;
    double err2 = 0, ref2 = 0;
    for (int i : idx) {
        double f_ref[3];
        for (int a = 0; a < 3; ++a) {
            r_pert[i][a] = r_src_d[i][a] + step;
            double pot_plus = dmk::esp_eval_gpu(gpu, r_pert, charges_d).pot[i];

            r_pert[i][a] = r_src_d[i][a] - step;
            double pot_minus = dmk::esp_eval_gpu(gpu, r_pert, charges_d).pot[i];

            r_pert[i][a] = r_src_d[i][a];

            f_ref[a] = -charges_d[i] * (pot_plus - pot_minus) / (2.0 * step);
        }
        const double diff_x = esp.force_x[i] - f_ref[0];
        const double diff_y = esp.force_y[i] - f_ref[1];
        const double diff_z = esp.force_z[i] - f_ref[2];
        err2 += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
        ref2 += f_ref[0] * f_ref[0] + f_ref[1] * f_ref[1] + f_ref[2] * f_ref[2];
    }

    dmk::esp_destroy_gpu_plan(gpu);
    dmk::esp_destroy_plan(plan);
    return std::sqrt(err2 / ref2);
}

// One phase (potential-only or forces): creates a CPU+GPU plan pair for eval_type, times
// n_runs of esp_eval_gpu, and (unless -S) also runs one CPU esp_eval call that serves double
// duty as both the speedup baseline and the l2_rel_err accuracy reference.
template <typename Real>
void run_phase(const Config &cfg, int n, const std::vector<dmk::Vec3T<Real>> &r_src,
              const std::vector<Real> &charges, dmk_eval_type eval_type, const char *phase_name) {
    double t_plan0 = MY_OMP_GET_WTIME();
    dmk::EspPlan *plan = dmk::esp_create_plan(cfg.L, cfg.r_c, cfg.eps, cfg.sigma, eval_type);
    dmk::GpuState *gpu = dmk::esp_create_gpu_plan(plan);
    double t_plan1 = MY_OMP_GET_WTIME();
    if (cfg.bench_plan)
        std::cout << "# plan_create_time (" << phase_name << ", cpu+gpu): " << (t_plan1 - t_plan0) << " s\n"
                  << std::flush;

    // Warmup: absorb first-call overhead (cuFINUFFT internal setup, allocator warm-up, etc.)
    // before any timed iteration.
    dmk::esp_eval_gpu(gpu, r_src, charges);

    double cpu_time = std::numeric_limits<double>::quiet_NaN();
    dmk::PotForce<Real> cpu_result{};
    if (!cfg.skip_cpu_baseline) {
        double t0 = MY_OMP_GET_WTIME();
        cpu_result = dmk::esp_eval<Real>(plan, r_src, charges);
        double t1 = MY_OMP_GET_WTIME();
        cpu_time = t1 - t0;
        std::cout << "# cpu_baseline_time (" << phase_name << "): " << cpu_time << " s (" << n / cpu_time
                  << " pts/s)\n"
                  << std::flush;
    }

    const bool report_err = (eval_type == DMK_POTENTIAL) && !cfg.skip_cpu_baseline;
    std::cout << "# phase: " << phase_name << " (gpu)\n";
    std::cout << "run,total_time,pts_per_s" << (report_err ? ",l2_rel_err_vs_cpu" : "") << "\n" << std::flush;

    double gpu_time_sum = 0;
    for (int run = 0; run < cfg.n_runs; ++run) {
        double t0 = MY_OMP_GET_WTIME();
        auto result = dmk::esp_eval_gpu(gpu, r_src, charges);
        double t1 = MY_OMP_GET_WTIME();
        gpu_time_sum += (t1 - t0);

        std::cout << run << "," << (t1 - t0) << "," << n / (t1 - t0);
        if (report_err)
            std::cout << "," << l2_rel_err(result.pot, cpu_result.pot);
        std::cout << "\n" << std::flush;
    }

    if (!cfg.skip_cpu_baseline) {
        const double gpu_time_avg = gpu_time_sum / cfg.n_runs;
        std::cout << "# speedup (" << phase_name << ", cpu_baseline / gpu_avg): " << (cpu_time / gpu_time_avg)
                  << "\n"
                  << std::flush;
    }

    dmk::esp_destroy_gpu_plan(gpu);
    dmk::esp_destroy_plan(plan);
}

template <typename Real>
void run_benchmark(Config cfg) {
    const int n = cfg.n_src;

    if (cfg.r_c == -1.0) {
        cfg.r_c = 0.07 * cfg.L;
        std::cout << "# r_c not given, defaulting to r_c=" << cfg.r_c << "\n" << std::flush;
    }
    print_config(cfg, std::cout);

    auto r_src_d = generate_positions<double>(n, cfg.L);
    auto charges_d = generate_charges<double>(n);

    std::vector<dmk::Vec3T<Real>> r_src(n);
    std::vector<Real> charges(n);
    for (int i = 0; i < n; ++i) {
        r_src[i] = {Real(r_src_d[i][0]), Real(r_src_d[i][1]), Real(r_src_d[i][2])};
        charges[i] = Real(charges_d[i]);
    }

    run_phase<Real>(cfg, n, r_src, charges, DMK_POTENTIAL, "eval_potential");

    if (cfg.bench_forces || cfg.check_forces) { // -F implies running the forces phase it validates against
        run_phase<Real>(cfg, n, r_src, charges, DMK_POTENTIAL_GRAD, "eval_forces");

        if (cfg.check_forces) {
            constexpr int n_fd_sample = 20;
            std::cout << "# phase: force_check (FD on " << n_fd_sample
                      << " random particles x 6 evals each; not all N)\n"
                      << std::flush;

            dmk::EspPlan *plan = dmk::esp_create_plan(cfg.L, cfg.r_c, cfg.eps, cfg.sigma, DMK_POTENTIAL_GRAD);
            dmk::GpuState *gpu = dmk::esp_create_gpu_plan(plan);
            auto esp = dmk::esp_eval_gpu(gpu, r_src_d, charges_d);
            double force_l2_err =
                check_forces_fd_gpu(esp, r_src_d, charges_d, cfg.L, cfg.r_c, cfg.eps, cfg.sigma, n_fd_sample);
            std::cout << "# force_check: l2_rel_err=" << force_l2_err << "\n" << std::flush;
            dmk::esp_destroy_gpu_plan(gpu);
            dmk::esp_destroy_plan(plan);
        }
    }
}

Config parse_args(int argc, char *argv[]) {
    Config cfg;
    int opt;
    while ((opt = getopt(argc, argv, "N:L:c:e:r:t:s:pFSgh?")) != -1) {
        switch (opt) {
        case 'N':
            cfg.n_src = int(std::atof(optarg));
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
        case 'p':
            cfg.bench_plan = true;
            break;
        case 'g':
            cfg.bench_forces = true;
            break;
        case 'F':
            cfg.check_forces = true;
            break;
        case 'S':
            cfg.skip_cpu_baseline = true;
            break;
        case 's':
            cfg.sigma = std::atof(optarg);
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
                      << "  -L L       Box side length (default 1.0)\n"
                      << "  -c r_c     Real-space cutoff (default 0.07*L)\n"
                      << "  -e eps     Tolerance (default 1e-6)\n"
                      << "  -r n_runs  Benchmark iterations (default 10)\n"
                      << "  -t f|d     Precision: float or double (default d)\n"
                      << "  -p         Also report plan creation time (cpu EspPlan + gpu_create_state)\n"
                      << "  -s sigma   FINUFFT/cuFINUFFT upsampling factor (default 1.35)\n"
                      << "  -g         Also benchmark forces (phase eval_forces, DMK_POTENTIAL_GRAD plan).\n"
                      << "             Potential-only timing (phase eval_potential) always runs regardless.\n"
                      << "  -F         Validate GPU forces against a finite-difference reference. Samples 20\n"
                      << "             random particles, not all N. Implies -g.\n"
                      << "  -S         Skip the CPU baseline run (no speedup or l2_rel_err_vs_cpu reporting,\n"
                      << "             GPU timing only, faster to iterate)\n"
                      << "  -h         Help\n"
                      << "\n"
                      << "Output CSV columns (eval phase):\n"
                      << "  run, total_time, pts_per_s[, l2_rel_err_vs_cpu]\n"
                      << "l2_rel_err_vs_cpu and the speedup line compare against one CPU esp_eval call per\n"
                      << "phase (not perilap3d) -- pass -S to skip that reference run entirely.\n";
            exit(0);
        }
    }
    return cfg;
}

int main(int argc, char *argv[]) {
    Config cfg = parse_args(argc, argv);

    if (cfg.prec == 'd')
        run_benchmark<double>(cfg);
    else
        run_benchmark<float>(cfg);
    return 0;
}

#else // defined(DMK_BUILD_ESP) && defined(DMK_GPU_OFFLOAD)
#include <iostream>
int main() {
    std::cerr << "benchmark_esp_gpu requires -DDMK_BUILD_ESP=ON -DDMK_GPU_OFFLOAD=ON at configure time.\n";
    return 1;
}
#endif
