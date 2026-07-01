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
            dmk::EspPlan *plan = dmk::esp_create_plan(cfg.L, cfg.r_c, cfg.eps, 1.35); // before: cfg.eps
            double t1 = MY_OMP_GET_WTIME();
            dmk::esp_destroy_plan(plan);

            if (rank == 0)
                std::cout << run << "," << (t1 - t0) << "," << n / (t1 - t0) << "\n" << std::flush;
        }
    }

    // ---- Eval benchmark ----------------------------------------------------
    dmk::EspPlan *plan = dmk::esp_create_plan(cfg.L, cfg.r_c, cfg.eps); // before: cfg.eps 

    // ---- Optional: load perilap3d reference for per-run error reporting ----
    std::vector<double> ref;
    bool have_ref = false;

    if (rank == 0) {
        ref.resize(n);

        // Flatten positions for cache validation and temp file
        std::vector<double> pos_flat(n * 3);
        for (int i = 0; i < n; ++i) {
            pos_flat[i*3+0] = r_src_d[i][0];
            pos_flat[i*3+1] = r_src_d[i][1];
            pos_flat[i*3+2] = r_src_d[i][2];
        }

        // Cache file: VERIFY_CACHE_DIR/perilap_N{n}.bin
        // Format: int32 N | float64[N*3] positions | float64[N] charges | float64[N] potentials
        std::string cache_path = std::string(VERIFY_CACHE_DIR) + "/perilap_N" + std::to_string(n) + ".bin";

        if (FILE *cf = fopen(cache_path.c_str(), "rb")) {
            int32_t cached_n = 0;
            std::vector<double> cached_pos(n * 3), cached_q(n);
            bool ok = fread(&cached_n, sizeof(cached_n), 1, cf) == 1
                   && cached_n == static_cast<int32_t>(n)
                   && (int)fread(cached_pos.data(), sizeof(double), n * 3, cf) == n * 3
                   && cached_pos == pos_flat
                   && (int)fread(cached_q.data(), sizeof(double), n, cf) == n
                   && cached_q == charges_d
                   && (int)fread(ref.data(), sizeof(double), n, cf) == n;
            fclose(cf);
            if (ok) {
                have_ref = true;
                std::cout << "# verify: cache hit — skipping perilap3d (N=" << n << ")\n" << std::flush;
            } else {
                std::cout << "# verify: cache stale or corrupt, recomputing perilap3d reference...\n" << std::flush;
            }
        } else {
            std::cout << "# verify: no cache found, calling perilap3d (this may take a while for first run)...\n" << std::flush;
        }

        if (!have_ref) {
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

            std::string cmd = std::string("python3 ") + VERIFY_SCRIPT_PATH
                            + " " + tmpfile + " " + cfg.perilap3d_dir
                            + " 2>" + errfile;
            FILE *pipe = popen(cmd.c_str(), "r");
            if (!pipe) {
                std::cerr << "# verify: failed to launch Python helper\n";
            } else {
                bool ok = true;
                for (int i = 0; i < n; ++i)
                    if (fscanf(pipe, "%lf", &ref[i]) != 1) { ok = false; break; }
                int rc = pclose(pipe);

                if (!ok || rc != 0) {
                    std::cerr << "# verify: Python helper failed (exit " << rc << "). stderr:\n";
                    if (FILE *ef = fopen(errfile, "r")) {
                        char buf[256];
                        while (fgets(buf, sizeof(buf), ef)) std::cerr << buf;
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
        }
    }

    // ---- 2D sweep: r_c x sigma — find the largest r_c and sigma within eps ----
    if (cfg.verify && have_ref && rank == 0) {
        const std::vector<double> rc_vals = {
            0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12};
        const double sig_lo = 1.1, sig_hi = 2.5, sig_step = 0.25;

        std::cout << "# phase: rc_sigma_sweep (eps=" << cfg.eps << ", N=" << n << ")\n"
                  << "r_c,sigma,l2_rel_err\n" << std::flush;

        struct SweepPt { double rc, sigma, l2; };
        std::vector<SweepPt> pts;

        for (double rc : rc_vals) {
            for (double sig = sig_lo; sig <= sig_hi + 1e-9; sig += sig_step) {
                dmk::EspPlan *sp = dmk::esp_create_plan(cfg.L, rc, cfg.eps, sig);
                auto pot_s = dmk::esp_eval<double>(sp, r_src_d, charges_d);
                dmk::esp_destroy_plan(sp);

                double esp_mean = 0;
                for (double v : pot_s) esp_mean += v;
                esp_mean /= n;

                double err2 = 0, ref2 = 0;
                for (int i = 0; i < n; ++i) {
                    double diff = (pot_s[i] - esp_mean) - ref[i];
                    err2 += diff * diff;
                    ref2 += ref[i] * ref[i];
                }
                double l2 = std::sqrt(err2 / ref2);
                std::cout << rc << "," << sig << "," << l2 << "\n" << std::flush;
                pts.push_back({rc, sig, l2});
            }
        }

        // For each r_c, recommend the largest sigma that achieves l2_rel_err < eps.
        // Larger sigma => smaller stencil width P => fewer spread points => faster.
        // r_c is a user choice; sigma is the tunable we optimize here.
        std::cout << "# recommendations (largest sigma achieving l2_rel_err < eps=" << cfg.eps << "):\n";
        for (double rc : rc_vals) {
            SweepPt best{rc, -1.0, -1.0};
            for (const auto &p : pts)
                if (p.rc == rc && p.l2 < cfg.eps && p.sigma > best.sigma)
                    best = p;
            std::cout << "#   r_c=" << rc << ": ";
            if (best.sigma > 0)
                std::cout << "sigma=" << best.sigma << " (l2_rel_err=" << best.l2 << ")\n";
            else
                std::cout << "no sigma in [" << sig_lo << ", " << sig_hi << "] achieved eps\n";
        }
        std::cout << std::flush;
    }

    // ---- Full eval benchmark --------------------------------------------------
    if (rank == 0) {
        std::cout << "# phase: eval\n";
        std::cout << "run,total_time,pts_per_s,sr_time,lr_time,self_time,l2_rel_err\n";
        std::cout << std::flush;
    }

    for (int run = 0; run < cfg.n_runs; ++run) {
        dmk::EspTimings timings{};
        double t0 = MY_OMP_GET_WTIME();
        auto pot = dmk::esp_eval<Real>(plan, r_src, charges, &timings);
        double t1 = MY_OMP_GET_WTIME();

        if (rank == 0) {
            std::cout << run << "," << (t1 - t0) << "," << n / (t1 - t0) << ","
                      << timings.t_short << "," << timings.t_long << "," << timings.t_self;
                double esp_mean = 0;
                for (int i = 0; i < n; ++i) esp_mean += double(pot[i]);
                esp_mean /= n;

                double err2 = 0, ref2 = 0;
                for (int i = 0; i < n; ++i) {
                    double diff = (double(pot[i]) - esp_mean) - ref[i];
                    double r = ref[i];
                    err2 += diff * diff;
                    ref2 += r * r;
                }
            std::cout << "," << std::sqrt(err2 / ref2);
            std::cout << "\n" << std::flush;
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
