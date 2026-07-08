#ifdef DMK_BUILD_ESP

#include <dmk/esp.hpp>
#include <dmk/omp_wrapper.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#ifdef DMK_HAVE_MPI
#include <mpi.h>
#define MYCOMM MPI_COMM_WORLD
#else
#define MYCOMM nullptr
#endif

static const char *PERILAP_TOLERANCE = "1e-12";

struct Config {
    int n_src = 100'000;
    double L = 1.0;
    double r_c = -1.0; // sentinel for "not explicitly given via -c"
    double eps = 1e-6;
    int n_runs = 10;
    char prec = 'd';
    double sigma = 1.35;
    bool bench_plan = false;
    bool bench_forces = false; // -g: also benchmark forces (potential+force timed together)
    bool skip_verify = false;  // -V: skip perilap3d comparison entirely (slow for large N)
    bool check_forces = false; // -F: validate forces against a finite-difference reference (very slow: O(N) extra evals); implies -g
    std::string perilap3d_dir = PERILAP3D_DIR;
    int log_level = 6; // DMK_LOG_OFF
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
        q[i] = (i % 2 == 0) ? Real(1) : Real(-1);
    return q;
}

// Load (or compute + cache) the perilap3d reference potentials for a given point set.
// Cache is keyed on N alone (VERIFY_CACHE_DIR/perilap_N{n}.bin) and validated against the
// actual positions/charges, so it's safe to call repeatedly for different N in a sweep.
bool get_perilap3d_reference(int n, const std::vector<dmk::Vec3T<double>> &r_src_d,
                             const std::vector<double> &charges_d, const std::string &perilap3d_dir,
                             std::vector<double> &ref) {
    ref.assign(n, 0.0);

    std::vector<double> pos_flat(n * 3);
    for (int i = 0; i < n; ++i) {
        pos_flat[i * 3 + 0] = r_src_d[i][0];
        pos_flat[i * 3 + 1] = r_src_d[i][1];
        pos_flat[i * 3 + 2] = r_src_d[i][2];
    }

    std::string cache_path =
        std::string(VERIFY_CACHE_DIR) + "/perilap_N" + std::to_string(n) + "_tol" + PERILAP_TOLERANCE + ".bin";

    if (FILE *cf = fopen(cache_path.c_str(), "rb")) {
        int32_t cached_n = 0;
        std::vector<double> cached_pos(n * 3), cached_q(n);
        bool ok = fread(&cached_n, sizeof(cached_n), 1, cf) == 1 && cached_n == static_cast<int32_t>(n) &&
                  (int)fread(cached_pos.data(), sizeof(double), n * 3, cf) == n * 3 && cached_pos == pos_flat &&
                  (int)fread(cached_q.data(), sizeof(double), n, cf) == n && cached_q == charges_d &&
                  (int)fread(ref.data(), sizeof(double), n, cf) == n;
        fclose(cf);
        if (ok) {
            std::cout << "# verify: cache hit — skipping perilap3d (N=" << n << ")\n" << std::flush;
            return true;
        }
        std::cout << "# verify: cache stale or corrupt, recomputing perilap3d reference (N=" << n << ")...\n"
                  << std::flush;
    } else {
        std::cout << "# verify: no cache found, calling perilap3d (N=" << n << ", this may take a while)...\n"
                  << std::flush;
    }

    char tmpfile[] = "/tmp/benchmark_esp_XXXXXX";
    int fd = mkstemp(tmpfile);
    int32_t nn = static_cast<int32_t>(n);
    write(fd, &nn, sizeof(nn));
    write(fd, pos_flat.data(), n * 3 * sizeof(double));
    write(fd, charges_d.data(), n * sizeof(double));
    close(fd);

    char errfile[] = "/tmp/benchmark_esp_err_XXXXXX";
    int efd = mkstemp(errfile);
    close(efd);

    std::string cmd = std::string("python3 ") + VERIFY_SCRIPT_PATH + " " + tmpfile + " " + perilap3d_dir + " " +
                      PERILAP_TOLERANCE + " 2>" + errfile;
    FILE *pipe = popen(cmd.c_str(), "r");
    bool have_ref = false;
    if (!pipe) {
        std::cerr << "# verify: failed to launch Python helper\n";
    } else {
        bool ok = true;
        for (int i = 0; i < n; ++i)
            if (fscanf(pipe, "%lf", &ref[i]) != 1) {
                ok = false;
                break;
            }
        int rc = pclose(pipe);

        if (!ok || rc != 0) {
            std::cerr << "# verify: Python helper failed (exit " << rc << "). stderr:\n";
            if (FILE *ef = fopen(errfile, "r")) {
                char buf[256];
                while (fgets(buf, sizeof(buf), ef))
                    std::cerr << buf;
                fclose(ef);
            }
        } else {
            have_ref = true;
            if (FILE *cf = fopen(cache_path.c_str(), "wb")) {
                fwrite(&nn, sizeof(nn), 1, cf);
                fwrite(pos_flat.data(), sizeof(double), n * 3, cf);
                fwrite(charges_d.data(), sizeof(double), n, cf);
                fwrite(ref.data(), sizeof(double), n, cf);
                fclose(cf);
                std::cout << "# verify: wrote perilap3d cache to " << cache_path << "\n" << std::flush;
            }
        }
        unlink(errfile);
    }
    unlink(tmpfile);
    return have_ref;
}

void print_config(const Config &cfg, int np, int n_threads, std::ostream &os) {
    os << "# n_src:       " << cfg.n_src << "\n"
       << "# L:           " << cfg.L << "\n"
       << "# r_c:         " << cfg.r_c << "\n"
       << "# eps:         " << cfg.eps << "\n"
       << "# sigma:       " << cfg.sigma << "\n"
       << "# n_runs:      " << cfg.n_runs << "\n"
       << "# prec:        " << (cfg.prec == 'd' ? "double" : "float") << "\n"
       << "# bench_plan:  " << (cfg.bench_plan ? "true" : "false") << "\n"
       << "# bench_forces:" << (cfg.bench_forces ? "true" : "false") << "\n"
       << "# check_forces:" << (cfg.check_forces ? "true" : "false") << "\n"
       << "# perilap3d:   " << cfg.perilap3d_dir << "\n"
       << "# log_level:   " << cfg.log_level << "\n"
       << "# mpi_ranks:   " << np << "\n"
       << "# omp_threads: " << n_threads << "\n";
}

// Auto-select r_c if it wasn't explicitly given (i.e. still sits at the kUnsetRc sentinel),
// as a function of N, sigma, and eps
// r_c is picked to balance short- and long-range wall time
void init_sensible_defaults(Config &cfg, const std::vector<dmk::Vec3T<double>> &r_src_d,
                            const std::vector<double> &charges_d) {
    if (cfg.r_c != -1.0)
        return; // explicitly given via -c, leave it alone

    const std::vector<double> rc_candidates = {0.03 * cfg.L, 0.05 * cfg.L, 0.07 * cfg.L, 0.10 * cfg.L, 0.12 * cfg.L};

    double best_rc = cfg.r_c;
    double best_t_short = 0, best_t_long = 0;
    double best_balance = std::numeric_limits<double>::max();
    for (double rc : rc_candidates) {
        dmk::EspPlan *plan = dmk::esp_create_plan(cfg.L, rc, cfg.eps, cfg.sigma);
        dmk::EspTimings timings{};
        dmk::esp_eval<double>(plan, r_src_d, charges_d, &timings);
        dmk::esp_destroy_plan(plan);

        double balance = std::fabs(timings.t_short - timings.t_long);
        if (balance < best_balance) {
            best_balance = balance;
            best_rc = rc;
            best_t_short = timings.t_short;
            best_t_long = timings.t_long;
        }
    }

    cfg.r_c = best_rc;
    std::cout << "# init_sensible_defaults: selected r_c=" << cfg.r_c << " (t_short=" << best_t_short
              << ", t_long=" << best_t_long << ")\n"
              << std::flush;
}

// Validate ESP's analytic forces (esp.force_x/y/z) against a central-difference reference,
// following the same recipe as test/test_esp.cpp's fd_force_reference. Very slow: 6*N extra
// esp_eval calls (2 per particle per component), each O(N) — off by default, enable with -F,
// and prefer a small -N when using it. The perturbed evals only need the potential, so they use
// a DMK_POTENTIAL-only plan (skips force computation) to keep the O(N) multiplier as cheap as
// possible.
double check_forces_fd(const dmk::PotForce<double> &esp, const std::vector<dmk::Vec3T<double>> &r_src_d,
                       const std::vector<double> &charges_d, double L, double r_c, double eps, double sigma) {
    const int n = static_cast<int>(charges_d.size());
    const double step = std::pow(eps, 1.0 / 3.0);

    dmk::EspPlan *plan = dmk::esp_create_plan(L, r_c, eps, sigma, DMK_POTENTIAL);

    std::vector<dmk::Vec3T<double>> r_pert = r_src_d;
    double err2 = 0, ref2 = 0;
    for (int i = 0; i < n; ++i) {
        double f_ref[3];
        for (int a = 0; a < 3; ++a) {
            r_pert[i][a] = r_src_d[i][a] + step;
            double pot_plus = dmk::esp_eval<double>(plan, r_pert, charges_d).pot[i];

            r_pert[i][a] = r_src_d[i][a] - step;
            double pot_minus = dmk::esp_eval<double>(plan, r_pert, charges_d).pot[i];

            r_pert[i][a] = r_src_d[i][a]; // restore before perturbing the next component

            f_ref[a] = -charges_d[i] * (pot_plus - pot_minus) / (2.0 * step);
        }
        const double diff_x = esp.force_x[i] - f_ref[0];
        const double diff_y = esp.force_y[i] - f_ref[1];
        const double diff_z = esp.force_z[i] - f_ref[2];
        err2 += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
        ref2 += f_ref[0] * f_ref[0] + f_ref[1] * f_ref[1] + f_ref[2] * f_ref[2];
    }

    dmk::esp_destroy_plan(plan);
    return std::sqrt(err2 / ref2);
}

// Potential-only benchmark: uses a DMK_POTENTIAL plan (no force computation at all), and reports
// l2_rel_err against the perilap3d reference when available.
template <typename Real>
void run_potential_benchmark(const Config &cfg, int rank, int n, const std::vector<dmk::Vec3T<Real>> &r_src,
                             const std::vector<Real> &charges, const std::vector<double> &ref, bool have_ref) {
    dmk::EspPlan *plan = dmk::esp_create_plan(cfg.L, cfg.r_c, cfg.eps, cfg.sigma, DMK_POTENTIAL);

    if (rank == 0) {
        std::cout << "# phase: eval_potential\n";
        std::cout << "run,total_time,pts_per_s,sr_time,lr_time,self_time,l2_rel_err\n";
        std::cout << std::flush;
    }

    for (int run = 0; run < cfg.n_runs; ++run) {
        dmk::EspTimings timings{};
        double t0 = MY_OMP_GET_WTIME();
        auto pot = dmk::esp_eval<Real>(plan, r_src, charges, &timings).pot;
        double t1 = MY_OMP_GET_WTIME();

        if (rank == 0) {
            std::cout << run << "," << (t1 - t0) << "," << n / (t1 - t0) << "," << timings.t_short << ","
                      << timings.t_long << "," << timings.t_self << ",";
            if (have_ref) {
                double esp_mean = 0;
                for (int i = 0; i < n; ++i)
                    esp_mean += double(pot[i]);
                esp_mean /= n;

                double err2 = 0, ref2 = 0;
                for (int i = 0; i < n; ++i) {
                    double diff = (double(pot[i]) - esp_mean) - ref[i];
                    double r = ref[i];
                    err2 += diff * diff;
                    ref2 += r * r;
                }
                std::cout << std::sqrt(err2 / ref2);
            } else {
                std::cout << "nan";
            }
            std::cout << "\n" << std::flush;
        }
    }

    dmk::esp_destroy_plan(plan);
}

// Forces benchmark: uses a DMK_POTENTIAL_GRAD plan, so total_time/sr_time/lr_time reflect
// potential and force computed together. Optionally (-F) also validates the analytic forces
// against a finite-difference reference — very slow (6*N extra evaluations), so it's kept as a
// separate step after the timing loop rather than run every iteration.
template <typename Real>
void run_forces_benchmark(const Config &cfg, int rank, int n, const std::vector<dmk::Vec3T<Real>> &r_src,
                          const std::vector<Real> &charges, const std::vector<dmk::Vec3T<double>> &r_src_d,
                          const std::vector<double> &charges_d) {
    dmk::EspPlan *plan = dmk::esp_create_plan(cfg.L, cfg.r_c, cfg.eps, cfg.sigma, DMK_POTENTIAL_GRAD);

    if (rank == 0) {
        std::cout << "# phase: eval_forces\n";
        std::cout << "run,total_time,pts_per_s,sr_time,lr_time,self_time\n";
        std::cout << std::flush;
    }

    for (int run = 0; run < cfg.n_runs; ++run) {
        dmk::EspTimings timings{};
        double t0 = MY_OMP_GET_WTIME();
        dmk::esp_eval<Real>(plan, r_src, charges, &timings);
        double t1 = MY_OMP_GET_WTIME();

        if (rank == 0)
            std::cout << run << "," << (t1 - t0) << "," << n / (t1 - t0) << "," << timings.t_short << ","
                      << timings.t_long << "," << timings.t_self << "\n"
                      << std::flush;
    }

    if (cfg.check_forces) {
        if (rank == 0)
            std::cout << "# phase: force_check (6*N extra evaluations, this may take a while)\n" << std::flush;
        auto esp = dmk::esp_eval<double>(plan, r_src_d, charges_d);
        double force_l2_err = check_forces_fd(esp, r_src_d, charges_d, cfg.L, cfg.r_c, cfg.eps, cfg.sigma);
        if (rank == 0)
            std::cout << "# force_check: l2_rel_err=" << force_l2_err << "\n" << std::flush;
    }

    dmk::esp_destroy_plan(plan);
}

template <typename Real>
void run_benchmark(Config cfg) {
    int rank = 0, np = 1;
#ifdef DMK_HAVE_MPI
    MPI_Comm_rank(MYCOMM, &rank);
    MPI_Comm_size(MYCOMM, &np);
#endif
    const int n_threads = MY_OMP_GET_MAX_THREADS();
    const int n = cfg.n_src;

    auto r_src_d = generate_positions<double>(n, cfg.L);
    auto charges_d = generate_charges<double>(n);

    std::vector<double> ref;
    bool have_ref = false;
    if (rank == 0 && !cfg.skip_verify)
        have_ref = get_perilap3d_reference(n, r_src_d, charges_d, cfg.perilap3d_dir, ref);

    if (rank == 0)
        init_sensible_defaults(cfg, r_src_d, charges_d);
#ifdef DMK_HAVE_MPI
    MPI_Bcast(&cfg.r_c, 1, MPI_DOUBLE, 0, MYCOMM);
#endif

    if (rank == 0)
        print_config(cfg, np, n_threads, std::cout);

    std::vector<dmk::Vec3T<Real>> r_src(n);
    std::vector<Real> charges(n);
    for (int i = 0; i < n; ++i) {
        r_src[i] = {Real(r_src_d[i][0]), Real(r_src_d[i][1]), Real(r_src_d[i][2])};
        charges[i] = Real(charges_d[i]);
    }

    // ---- Optionally benchmark plan creation --------------------------------
    if (cfg.bench_plan) {
        if (rank == 0) {
            std::cout << "# phase: plan_create\n"
                      << "run,plan_time,pts_per_s\n"
                      << std::flush;
        }
        for (int run = 0; run < cfg.n_runs; ++run) {
            double t0 = MY_OMP_GET_WTIME();
            dmk::EspPlan *plan = dmk::esp_create_plan(cfg.L, cfg.r_c, cfg.eps, cfg.sigma);
            double t1 = MY_OMP_GET_WTIME();
            dmk::esp_destroy_plan(plan);

            if (rank == 0)
                std::cout << run << "," << (t1 - t0) << "," << n / (t1 - t0) << "\n" << std::flush;
        }
    }

    run_potential_benchmark<Real>(cfg, rank, n, r_src, charges, ref, have_ref);
    if (cfg.bench_forces || cfg.check_forces) // -F implies running the forces benchmark it validates against
        run_forces_benchmark<Real>(cfg, rank, n, r_src, charges, r_src_d, charges_d);
}

Config parse_args(int argc, char *argv[]) {
    Config cfg;
    int opt;
    while ((opt = getopt(argc, argv, "N:L:c:e:r:t:l:P:s:pVFgh?")) != -1) {
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
        case 'l':
            cfg.log_level = std::atoi(optarg);
            break;
        case 'p':
            cfg.bench_plan = true;
            break;
        case 'g':
            cfg.bench_forces = true;
            break;
        case 'V':
            cfg.skip_verify = true;
            break;
        case 'F':
            cfg.check_forces = true;
            break;
        case 's':
            cfg.sigma = std::atof(optarg);
            break;
        case 'P':
            cfg.perilap3d_dir = optarg;
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
                      << "  -c r_c     Real-space cutoff (default: auto-picked to balance short-/long-range time)\n"
                      << "  -e eps     Tolerance (default 1e-6)\n"
                      << "  -r n_runs  Benchmark iterations (default 10)\n"
                      << "  -t f|d     Precision: float or double (default d)\n"
                      << "  -l level   Log verbosity 0-6 (default 6=off)\n"
                      << "  -p         Also benchmark plan creation\n"
                      << "  -s sigma   FINUFFT upsampling factor for the long-range PSWF kernel (default 1.35)\n"
                      << "  -V         Skip perilap3d comparison entirely (slow for large N)\n"
                      << "  -g         Also benchmark forces (potential+force timed together, phase eval_forces).\n"
                      << "             Potential-only timing (phase eval_potential) always runs regardless.\n"
                      << "  -F         Validate forces against a finite-difference reference (very slow: 6*N\n"
                      << "             extra evaluations — prefer a small -N when using this). Implies -g.\n"
                      << "  -P dir     Override perilap3d directory (default: compile-time path)\n"
                      << "  -h         Help\n"
                      << "\n"
                      << "Output CSV columns (eval phase):\n"
                      << "  run, total_time, pts_per_s, sr_time, lr_time, self_time\n";
            exit(0);
        }
    }
    return cfg;
}

int main(int argc, char *argv[]) {
#ifdef DMK_HAVE_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#endif

    Config cfg = parse_args(argc, argv);
    if (cfg.prec == 'd')
        run_benchmark<double>(cfg);
    else
        run_benchmark<float>(cfg);

#ifdef DMK_HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}

#else // DMK_BUILD_ESP
#include <iostream>
int main() {
    std::cerr << "benchmark_esp requires -DDMK_BUILD_ESP=ON at configure time.\n";
    return 1;
}
#endif // DMK_BUILD_ESP
