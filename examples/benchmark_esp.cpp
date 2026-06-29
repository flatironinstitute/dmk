#ifdef DMK_BUILD_ESP

#include <dmk/esp.hpp>
#include <dmk/omp_wrapper.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <getopt.h>
#include <iostream>
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

struct Config {
    int n_src = 100'000;
    double L = 1.0;
    double r_c = 0.05;
    double eps = 1e-6;
    int n_runs = 10;
    char prec = 'd';
    bool bench_plan = false;
    bool verify = false;
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

void print_config(const Config &cfg, int np, int n_threads, std::ostream &os) {
    os << "# n_src:       " << cfg.n_src << "\n"
       << "# L:           " << cfg.L << "\n"
       << "# r_c:         " << cfg.r_c << "\n"
       << "# eps:         " << cfg.eps << "\n"
       << "# n_runs:      " << cfg.n_runs << "\n"
       << "# prec:        " << (cfg.prec == 'd' ? "double" : "float") << "\n"
       << "# mpi_ranks:   " << np << "\n"
       << "# omp_threads: " << n_threads << "\n";
}

template <typename Real>
void run_benchmark(const Config &cfg) {
    int rank = 0, np = 1;
#ifdef DMK_HAVE_MPI
    MPI_Comm_rank(MYCOMM, &rank);
    MPI_Comm_size(MYCOMM, &np);
#endif
    const int n_threads = MY_OMP_GET_MAX_THREADS();
    const int n = cfg.n_src;

    // Always generate double-precision data (used for verification and cast to Real for timing).
    auto r_src_d = generate_positions<double>(n, cfg.L);
    auto charges_d = generate_charges<double>(n);

    std::vector<dmk::Vec3T<Real>> r_src(n);
    std::vector<Real> charges(n);
    for (int i = 0; i < n; ++i) {
        r_src[i] = {Real(r_src_d[i][0]), Real(r_src_d[i][1]), Real(r_src_d[i][2])};
        charges[i] = Real(charges_d[i]);
    }

    // ---- Optionally benchmark plan creation --------------------------------
    if (cfg.bench_plan) {
        if (rank == 0) {
            print_config(cfg, np, n_threads, std::cout);
            std::cout << "# phase: plan_create\n"
                      << "run,plan_time,pts_per_s\n"
                      << std::flush;
        }
        for (int run = 0; run < cfg.n_runs; ++run) {
            double t0 = MY_OMP_GET_WTIME();
            dmk::EspPlan *plan = dmk::esp_create_plan(cfg.L, cfg.r_c, cfg.eps);
            double t1 = MY_OMP_GET_WTIME();
            dmk::esp_destroy_plan(plan);

            if (rank == 0)
                std::cout << run << "," << (t1 - t0) << "," << n / (t1 - t0) << "\n" << std::flush;
        }
    }

    // ---- Eval benchmark ----------------------------------------------------
    dmk::EspPlan *plan = dmk::esp_create_plan(cfg.L, cfg.r_c, cfg.eps);

    // ---- Optional verification against perilap3d ---------------------------
    if (cfg.verify && rank == 0) {
        auto pot_d = dmk::esp_eval<double>(plan, r_src_d, charges_d);

        // write binary temp file: int32 N | float64[N,3] positions | float64[N] charges
        char tmpfile[] = "/tmp/benchmark_esp_XXXXXX";
        int fd = mkstemp(tmpfile);
        int32_t nn = static_cast<int32_t>(n);
        write(fd, &nn, sizeof(nn));
        for (int i = 0; i < n; ++i) write(fd, r_src_d[i].data(), 3 * sizeof(double));
        write(fd, charges_d.data(), n * sizeof(double));
        close(fd);

        // redirect Python stderr to a temp file so we can show it on failure
        char errfile[] = "/tmp/benchmark_esp_err_XXXXXX";
        int efd = mkstemp(errfile);
        close(efd);

        std::string cmd = std::string("python3 ") + VERIFY_SCRIPT_PATH
                        + " " + tmpfile + " " + cfg.perilap3d_dir
                        + " 2>" + errfile;
        FILE *pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            std::cerr << "# verify: failed to launch Python helper\n";
            unlink(tmpfile);
            unlink(errfile);
        } else {
            std::vector<double> ref(n);
            bool ok = true;
            for (int i = 0; i < n; ++i) {
                if (fscanf(pipe, "%lf", &ref[i]) != 1) {
                    ok = false;
                    break;
                }
            }
            int rc = pclose(pipe);
            unlink(tmpfile);

            if (!ok || rc != 0) {
                std::cerr << "# verify: Python helper failed (exit " << rc << "). stderr:\n";
                if (FILE *ef = fopen(errfile, "r")) {
                    char buf[256];
                    while (fgets(buf, sizeof(buf), ef)) std::cerr << buf;
                    fclose(ef);
                }
            } else {
                // gauge-correct ESP result: subtract its mean (ref is already zero-mean)
                double esp_mean = 0;
                for (double v : pot_d) esp_mean += v;
                esp_mean /= n;

                double err2 = 0, ref2 = 0, max_rel = 0;
                for (int i = 0; i < n; ++i) {
                    std::cout << "Point " << i << " - Perilap reference: " << ref[i] << " ESP: " << pot_d[i] << " Difference: " << pot_d[i] - ref[i] << "\n";
                    double diff = (pot_d[i] - esp_mean) - ref[i];
                    double r = ref[i];
                    err2 += diff * diff;
                    ref2 += r * r;
                    if (std::abs(r) > 0.0)
                        max_rel = std::max(max_rel, std::abs(diff / r));
                }
                double l2_rel = std::sqrt(err2 / ref2);

                std::cout << "# verify: l2_rel_err = " << l2_rel
                          << ", max_rel_err = " << max_rel
                          << " (eps = " << cfg.eps << ")\n" << std::flush;
            }
            unlink(errfile);
        }
    }

    // ---- Full eval benchmark --------------------------------------------------
    if (rank == 0) {
        std::cout << "# phase: eval\n"
                  << "run,total_time,pts_per_s,sr_time,lr_time,self_time\n"
                  << std::flush;
    }

    for (int run = 0; run < cfg.n_runs; ++run) {
        dmk::EspTimings timings{};
        double t0 = MY_OMP_GET_WTIME();
        auto pot = dmk::esp_eval<Real>(plan, r_src, charges, &timings);
        double t1 = MY_OMP_GET_WTIME();
        (void)pot;

        if (rank == 0) {
            std::cout << run << "," << (t1 - t0) << "," << n / (t1 - t0) << "," << timings.t_short << ","
                      << timings.t_long << "," << timings.t_self << "\n"
                      << std::flush;
        }
    }

    dmk::esp_destroy_plan(plan);
}

Config parse_args(int argc, char *argv[]) {
    Config cfg;
    int opt;
    while ((opt = getopt(argc, argv, "N:L:c:e:r:t:l:P:Vph?")) != -1) {
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
        case 'V':
            cfg.verify = true;
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
                      << "  -c r_c     Real-space cutoff (default 0.05)\n"
                      << "  -e eps     Tolerance (default 1e-6)\n"
                      << "  -r n_runs  Benchmark iterations (default 10)\n"
                      << "  -t f|d     Precision: float or double (default d)\n"
                      << "  -l level   Log verbosity 0-6 (default 6=off)\n"
                      << "  -p         Also benchmark plan creation\n"
                      << "  -V         Verify against perilap3d reference (requires Python + numpy + numba)\n"
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
