#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/omp_wrapper.hpp>
#include <dmk/util.hpp>

#include <algorithm>
#include <cmath>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#ifdef DMK_HAVE_MPI
#include <mpi.h>
#define MYCOMM MPI_COMM_WORLD
#else
#define MYCOMM nullptr
#endif

struct Config {
    int n_src = 1'000'000;
    int n_per_leaf = 280;
    double eps = 1e-5;
    char prec = 'f';
    bool uniform = false;
    bool enable_direct = true;
    int n_direct = -1;
    int n_runs = 100;
    int log_level = DMK_LOG_OFF;
    dmk_ikernel kernel = DMK_LAPLACE;
    int n_dim = 3;
    double fparam = 6.0;
};

struct TimingResult {
    double elapsed;
    double pts_per_sec;
    double pts_per_sec_per_rank;
    double pts_per_sec_per_thread;
};

struct ErrorMetrics {
    double l2_rel;
    double max_rel;
};

inline int local_count(int n, int np, int r) { return n / np + (r < (n % np) ? 1 : 0); }

TimingResult make_timing(double elapsed, int n_total, int n_per_rank, int n_threads) {
    return {elapsed, n_total / elapsed, n_per_rank / elapsed, n_per_rank / elapsed / n_threads};
}

template <typename Real>
ErrorMetrics compute_error(const std::vector<Real> &computed, const std::vector<Real> &reference, int rank, int np) {
    double local_err2 = 0.0, local_ref2 = 0.0, local_maxre = 0.0;

    for (size_t i = 0; i < reference.size(); ++i) {
        double diff = double(computed[i]) - double(reference[i]);
        double ref = double(reference[i]);
        local_err2 += diff * diff;
        local_ref2 += ref * ref;
        if (std::abs(ref) > 0.0)
            local_maxre = std::max(local_maxre, std::abs(diff / ref));
    }

#ifdef DMK_HAVE_MPI
    double glob_err2 = 0.0, glob_ref2 = 0.0, glob_maxre = 0.0;
    MPI_Allreduce(&local_err2, &glob_err2, 1, MPI_DOUBLE, MPI_SUM, MYCOMM);
    MPI_Allreduce(&local_ref2, &glob_ref2, 1, MPI_DOUBLE, MPI_SUM, MYCOMM);
    MPI_Allreduce(&local_maxre, &glob_maxre, 1, MPI_DOUBLE, MPI_MAX, MYCOMM);
#else
    double glob_err2 = local_err2, glob_ref2 = local_ref2, glob_maxre = local_maxre;
#endif

    return {std::sqrt(glob_err2 / glob_ref2), glob_maxre};
}

template <typename Real>
void generate_and_scatter(int n_dim, int nd, int n_src, bool uniform, bool set_fixed_charges, std::vector<Real> &r_src,
                          std::vector<Real> &charges, long seed, int rank, int np) {
    const int n_local = local_count(n_src, np, rank);

    std::vector<Real> r_all, c_all;

    if (rank == 0) {
        r_all.resize(size_t(n_dim) * n_src);
        c_all.resize(size_t(nd) * n_src);

        const double rin = 0.45;
        std::default_random_engine eng(seed);
        std::uniform_real_distribution<double> rng;

        for (int i = 0; i < n_src; ++i) {
            if (!uniform && n_dim == 3) {
                double theta = rng(eng) * M_PI;
                double ct = std::cos(theta), st = std::sin(theta);
                double phi = rng(eng) * 2.0 * M_PI;
                r_all[i * n_dim + 0] = Real(rin * st * std::cos(phi) + 0.5);
                r_all[i * n_dim + 1] = Real(rin * st * std::sin(phi) + 0.5);
                r_all[i * n_dim + 2] = Real(rin * ct + 0.5);
            } else if (!uniform && n_dim == 2) {
                double phi = rng(eng) * 2.0 * M_PI;
                r_all[i * n_dim + 0] = Real(rin * std::cos(phi) + 0.5);
                r_all[i * n_dim + 1] = Real(rin * std::sin(phi) + 0.5);
            } else {
                for (int j = 0; j < n_dim; ++j)
                    r_all[i * n_dim + j] = Real(rng(eng));
            }
            for (int j = 0; j < nd; ++j)
                c_all[i * nd + j] = Real(rng(eng) - 0.5);
        }

        if (set_fixed_charges && n_src > 0)
            for (int j = 0; j < n_dim; ++j)
                r_all[j] = Real(0);
        if (set_fixed_charges && n_src > 1)
            for (int j = 0; j < n_dim; ++j)
                r_all[n_dim + j] = Real(1) - std::numeric_limits<Real>::epsilon();
        if (set_fixed_charges && n_src > 2)
            for (int j = 0; j < n_dim; ++j)
                r_all[2 * n_dim + j] = Real(0.05);
    }

#ifdef DMK_HAVE_MPI
    std::vector<int> sendcounts_r(np), displs_r(np);
    std::vector<int> sendcounts_c(np), displs_c(np);
    for (int i = 0; i < np; ++i) {
        int ni = local_count(n_src, np, i);
        sendcounts_r[i] = ni * n_dim;
        sendcounts_c[i] = ni * nd;
    }
    displs_r[0] = displs_c[0] = 0;
    for (int i = 1; i < np; ++i) {
        displs_r[i] = displs_r[i - 1] + sendcounts_r[i - 1];
        displs_c[i] = displs_c[i - 1] + sendcounts_c[i - 1];
    }

    auto mpi_t = std::is_same_v<Real, float> ? MPI_FLOAT : MPI_DOUBLE;
    r_src.resize(size_t(n_dim) * n_local);
    charges.resize(size_t(nd) * n_local);
    MPI_Scatterv(rank == 0 ? r_all.data() : nullptr, sendcounts_r.data(), displs_r.data(), mpi_t, r_src.data(),
                 n_local * n_dim, mpi_t, 0, MYCOMM);
    MPI_Scatterv(rank == 0 ? c_all.data() : nullptr, sendcounts_c.data(), displs_c.data(), mpi_t, charges.data(),
                 n_local * nd, mpi_t, 0, MYCOMM);
#else
    r_src = std::move(r_all);
    charges = std::move(c_all);
#endif
}

template <typename Real>
void run_direct(const Config &cfg, int n_dim, int charge_dim, const std::vector<Real> &r_src,
                const std::vector<Real> &charges, const std::vector<Real> &r_trg, std::vector<Real> &pot, int rank,
                int np) {
    const int n_src_local = r_src.size() / n_dim;
    int n_trg_local = r_trg.size() / n_dim;

#ifdef DMK_HAVE_MPI
    // Gather all sources to all ranks
    int n_src_global = 0;
    MPI_Allreduce(&n_src_local, &n_src_global, 1, MPI_INT, MPI_SUM, MYCOMM);

    auto mpi_t = std::is_same_v<Real, float> ? MPI_FLOAT : MPI_DOUBLE;

    std::vector<int> recv_cnts_r(np), recv_disp_r(np);
    std::vector<int> recv_cnts_c(np), recv_disp_c(np);
    {
        int send_cnt_r = n_src_local * n_dim;
        int send_cnt_c = n_src_local * charge_dim;
        MPI_Allgather(&send_cnt_r, 1, MPI_INT, recv_cnts_r.data(), 1, MPI_INT, MYCOMM);
        MPI_Allgather(&send_cnt_c, 1, MPI_INT, recv_cnts_c.data(), 1, MPI_INT, MYCOMM);
        recv_disp_r[0] = recv_disp_c[0] = 0;
        for (int i = 1; i < np; ++i) {
            recv_disp_r[i] = recv_disp_r[i - 1] + recv_cnts_r[i - 1];
            recv_disp_c[i] = recv_disp_c[i - 1] + recv_cnts_c[i - 1];
        }
    }

    std::vector<Real> glb_r_src(n_src_global * n_dim);
    std::vector<Real> glb_charges(n_src_global * charge_dim);
    MPI_Allgatherv(r_src.data(), n_src_local * n_dim, mpi_t, glb_r_src.data(), recv_cnts_r.data(), recv_disp_r.data(),
                   mpi_t, MYCOMM);
    MPI_Allgatherv(charges.data(), n_src_local * charge_dim, mpi_t, glb_charges.data(), recv_cnts_c.data(),
                   recv_disp_c.data(), mpi_t, MYCOMM);
#else
    int n_src_global = n_src_local;
    const auto &glb_r_src = r_src;
    const auto &glb_charges = charges;
#endif

    // Convert sources to double for reference evaluation
    std::vector<double> r_src_d(glb_r_src.begin(), glb_r_src.end());
    std::vector<double> charges_d(glb_charges.begin(), glb_charges.end());
    std::vector<double> r_trg_d(r_trg.begin(), r_trg.end());

    // Evaluate: each rank handles its own local targets
    const auto eval_level = cfg.kernel == DMK_STOKESLET ? DMK_VELOCITY : DMK_POTENTIAL;
    const int kdim = dmk::get_kernel_output_dim(n_dim, cfg.kernel, eval_level);
    std::vector<double> pot_d(n_trg_local * kdim, 0.0);

    const auto eval = dmk::get_direct_evaluator<double>(cfg.kernel, eval_level, n_dim, cfg.fparam);
    dmk::parallel_direct_eval<double>(eval, n_src_global, r_src_d.data(), charges_d.data(), nullptr, n_trg_local,
                                      r_trg_d.data(), pot_d.data(), n_dim, kdim);

    pot.resize(n_trg_local * kdim);
    for (size_t i = 0; i < pot_d.size(); ++i)
        pot[i] = pot_d[i];
}

template <typename Real>
double run_dmk(pdmk_tree tree, std::vector<Real> &pot, int n_src_per_rank, int kdim, int rank, int np) {
    pot.resize(n_src_per_rank * kdim);

#ifdef DMK_HAVE_MPI
    MPI_Barrier(MYCOMM);
#endif

    double st = omp_get_wtime();
    if constexpr (std::is_same_v<Real, float>)
        pdmk_tree_evalf(tree, pot.data(), nullptr);
    else
        pdmk_tree_eval(tree, pot.data(), nullptr);
    double ft = omp_get_wtime();

    return ft - st;
}

void print_csv_config_comment(const Config &cfg, int np, int n_threads, std::ostream &os) {
    const char *kernel_str = [&] {
        switch (cfg.kernel) {
        case DMK_LAPLACE:
            return "laplace";
        case DMK_SQRT_LAPLACE:
            return "sqrt_laplace";
        case DMK_YUKAWA:
            return "yukawa";
        case DMK_STOKESLET:
            return "stokes";
        default:
            return "unknown";
        }
    }();
    os << "# mpi_ranks:            " << np << "\n"
       << "# omp_threads_per_rank: " << n_threads << "\n"
       << "# n_src:                " << cfg.n_src << "\n"
       << "# n_dim:                " << cfg.n_dim << "\n"
       << "# kernel:               " << kernel_str << "\n"
       << "# precision:            " << (cfg.prec == 'd' ? "double" : "float") << "\n"
       << "# uniform_dist:         " << cfg.uniform << "\n"
       << "# eps:                  " << cfg.eps << "\n"
       << "# n_per_leaf:           " << cfg.n_per_leaf << "\n"
       << "# n_runs:               " << cfg.n_runs << "\n"
       << "# direct_enabled:       " << cfg.enable_direct << "\n"
       << "# n_direct:             " << cfg.n_direct << "\n"
       << "# log_level:            " << cfg.log_level << "\n";
}

void print_csv_header(std::ostream &os) {
    os << "dmk_time,dmk_pts_s,dmk_pts_s_rank,dmk_pts_s_thread,dmk_l2_rel_err,dmk_max_rel_err";
}

void print_csv_row(const TimingResult &t, const ErrorMetrics *err, std::ostream &os) {
    auto nan = std::numeric_limits<double>::quiet_NaN();
    os << t.elapsed << "," << t.pts_per_sec << "," << t.pts_per_sec_per_rank << "," << t.pts_per_sec_per_thread << ",";
    if (err)
        os << err->l2_rel << "," << err->max_rel;
    else
        os << nan << "," << nan;
}

dmk_ikernel parse_kernel(const char *s) {
    std::string k(s);
    if (k == "laplace")
        return DMK_LAPLACE;
    if (k == "sqrt_laplace")
        return DMK_SQRT_LAPLACE;
    if (k == "yukawa")
        return DMK_YUKAWA;
    if (k == "stokes")
        return DMK_STOKESLET;
    throw std::runtime_error("Unknown kernel: " + k);
}

template <typename Real>
void run_benchmark(const Config &cfg) {
    int rank = 0, np = 1;
#ifdef DMK_HAVE_MPI
    MPI_Comm_rank(MYCOMM, &rank);
    MPI_Comm_size(MYCOMM, &np);
#endif

    const int n_dim = cfg.n_dim;
    const int n_threads = MY_OMP_GET_MAX_THREADS();
    const int n_src = cfg.n_src;
    const int n_src_per_rank = local_count(n_src, np, rank);

    pdmk_params params{};
    params.eps = cfg.eps;
    params.n_dim = n_dim;
    params.n_per_leaf = cfg.n_per_leaf;
    params.log_level = cfg.log_level;
    params.eval_src = DMK_POTENTIAL;
    params.eval_trg = DMK_POTENTIAL;
    params.kernel = cfg.kernel;
    if (cfg.kernel == DMK_YUKAWA)
        params.fparam = cfg.fparam;
    if (cfg.kernel == DMK_STOKESLET) {
        params.eval_src = DMK_VELOCITY;
        params.eval_trg = DMK_VELOCITY;
    }

    const int charge_dim = dmk::get_kernel_input_dim(n_dim, cfg.kernel);
    const int pot_dim = dmk::get_kernel_output_dim(n_dim, cfg.kernel, params.eval_src);

    std::vector<Real> r_src, charges;
    generate_and_scatter<Real>(n_dim, charge_dim, n_src, cfg.uniform, true, r_src, charges, 0, rank, np);

    pdmk_tree tree;
    if constexpr (std::is_same_v<Real, float>)
        tree = pdmk_tree_createf(MYCOMM, params, n_src_per_rank, r_src.data(), charges.data(), nullptr, 0, nullptr);
    else
        tree = pdmk_tree_create(MYCOMM, params, n_src_per_rank, r_src.data(), charges.data(), nullptr, 0, nullptr);

    // Direct reference
    std::vector<Real> pot_direct;
    if (cfg.enable_direct) {
        int n_direct_global = (cfg.n_direct > 0) ? cfg.n_direct : n_src;
        int n_direct_per_rank = local_count(n_direct_global, np, rank);

        if (n_direct_global == n_src) {
            run_direct(cfg, n_dim, charge_dim, r_src, charges, r_src, pot_direct, rank, np);
        } else {
            std::vector<Real> r_trg(r_src.begin(), r_src.begin() + n_direct_per_rank * n_dim);
            run_direct(cfg, n_dim, charge_dim, r_src, charges, r_trg, pot_direct, rank, np);
        }
    }

#ifdef DMK_HAVE_MPI
    MPI_Barrier(MYCOMM);
#endif

    if (rank == 0) {
        print_csv_config_comment(cfg, np, n_threads, std::cout);
        print_csv_header(std::cout);
        std::cout << std::flush;
    }

    for (int run = 0; run < cfg.n_runs; ++run) {
        std::vector<Real> pot_dmk;
        sctl::Profile::reset();
        double dt = run_dmk<Real>(tree, pot_dmk, n_src_per_rank, pot_dim, rank, np);
        TimingResult t = make_timing(dt, n_src, n_src_per_rank, n_threads);

        if (run == 0) {
            if (rank == 0)
                std::cout << ",";
            pdmk_print_profile_data(MYCOMM, 'h');
            if (rank == 0)
                std::cout << "\n";
        }

        ErrorMetrics err{};
        bool have_err = false;
        if (cfg.enable_direct) {
            int n_compare = std::min(int(pot_direct.size()), int(pot_dmk.size()));
            std::vector<Real> pot_dmk_sub(pot_dmk.begin(), pot_dmk.begin() + n_compare);
            std::vector<Real> pot_dir_sub(pot_direct.begin(), pot_direct.begin() + n_compare);
            err = compute_error(pot_dmk_sub, pot_dir_sub, rank, np);
            have_err = true;
        }

        if (rank == 0) {
            print_csv_row(t, have_err ? &err : nullptr, std::cout);
        }
        if (rank == 0)
            std::cout << ",";
        pdmk_print_profile_data(MYCOMM, 'c');
        if (rank == 0)
            std::cout << std::endl << std::flush;
    }

    pdmk_tree_destroy(tree);
}

Config parse_args(int argc, char *argv[]) {
    Config cfg;

    static struct option long_opts[] = {
        {"direct", no_argument, nullptr, 1001},
        {"no-direct", no_argument, nullptr, 1002},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "N:n:e:t:r:D:l:k:d:f:uh?", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'N':
            cfg.n_src = int(std::atof(optarg));
            break;
        case 'n':
            cfg.n_per_leaf = std::atoi(optarg);
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
        case 'l':
            cfg.log_level = std::atoi(optarg);
            break;
        case 'k':
            cfg.kernel = parse_kernel(optarg);
            break;
        case 'd':
            cfg.n_dim = std::atoi(optarg);
            break;
        case 'f':
            cfg.fparam = std::atof(optarg);
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
            cfg.enable_direct = true;
            break;
        case 1002:
            cfg.enable_direct = false;
            break;
        case 'h':
        case '?':
        default:
            std::cout << "Usage: " << argv[0] << "\n"
                      << "  -N n_src           Number of source points\n"
                      << "  -n n_per_leaf      DMK leaf size\n"
                      << "  -e eps             Tolerance\n"
                      << "  -t f|d             Precision\n"
                      << "  -k kernel          laplace, sqrt_laplace, yukawa, stokes\n"
                      << "  -d dim             2 or 3\n"
                      << "  -f fparam          Yukawa parameter (default: 6.0)\n"
                      << "  -r n_runs          Benchmark iterations\n"
                      << "  -D n_direct        Points for direct comparison\n"
                      << "  -l log_level       DMK log verbosity\n"
                      << "  -u                 Uniform distribution\n"
                      << "  --direct/--no-direct  Enable/disable reference\n"
                      << "  -h                 Help\n";
            exit(0);
        }
    }
    cfg.n_direct = cfg.n_direct > 0 ? cfg.n_direct : cfg.n_src;
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
