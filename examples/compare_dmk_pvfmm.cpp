#include <dmk.h>
#include <dmk/omp_wrapper.hpp>

#include <pvfmm.hpp>

#include <algorithm>
#include <cmath>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include <mpi.h>

struct Config {
    int n_src = 1'000'000;
    int n_per_leaf_dmk = 280;
    int n_per_leaf_pvfmm = 600;
    int m_pvfmm = 6;
    double eps = 1e-5;
    char prec = 'f';
    bool uniform = false;
    bool enable_pvfmm = true;
    bool enable_direct = true;
    int n_direct = -1; // -1 means "same as n_src"
    int n_runs = 100;
    int log_level = DMK_LOG_OFF; // 6: off, 5: critical, 4: err, 3: warn, 2: info, 1: debug, 0: trace
};

struct TimingResult {
    double elapsed;
    double pts_per_sec;
    double pts_per_sec_per_rank;
    double pts_per_sec_per_thread;
};

struct ErrorMetrics {
    double avg_rel;
    double max_rel;
};

TimingResult make_timing(double elapsed, int n_total, int n_per_rank, int n_threads) {
    return {elapsed, n_total / elapsed, n_per_rank / elapsed, n_per_rank / elapsed / n_threads};
}

template <typename Real>
ErrorMetrics compute_error(const std::vector<Real> &computed, const std::vector<Real> &reference, MPI_Comm comm) {
    double local_err2 = 0.0;
    double local_ref2 = 0.0;
    double local_maxre = 0.0;

    for (size_t i = 0; i < reference.size(); ++i) {
        double diff = static_cast<double>(computed[i]) - static_cast<double>(reference[i]);
        double ref = static_cast<double>(reference[i]);
        local_err2 += diff * diff;
        local_ref2 += ref * ref;
        if (std::abs(ref) > 0.0)
            local_maxre = std::max(local_maxre, std::abs(diff) / std::abs(ref));
    }

    double glob_err2 = 0.0, glob_ref2 = 0.0, glob_maxre = 0.0;
    MPI_Allreduce(&local_err2, &glob_err2, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_ref2, &glob_ref2, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_maxre, &glob_maxre, 1, MPI_DOUBLE, MPI_MAX, comm);

    return {std::sqrt(glob_err2 / glob_ref2), glob_maxre};
}

void print_csv_config_comment(const Config &cfg, int np, int n_threads, std::ostream &os) {
    os << "# mpi_ranks:            " << np << "\n"
       << "# omp_threads_per_rank: " << n_threads << "\n"
       << "# n_src:                " << cfg.n_src << "\n"
       << "# precision:            " << (cfg.prec == 'd' ? "double" : "float") << "\n"
       << "# uniform_dist:         " << cfg.uniform << "\n"
       << "# eps:                  " << cfg.eps << "\n"
       << "# n_per_leaf_dmk:       " << cfg.n_per_leaf_dmk << "\n"
       << "# n_runs:               " << cfg.n_runs << "\n"
       << "# pvfmm_enabled:        " << cfg.enable_pvfmm << "\n"
       << "# n_per_leaf_pvfmm:     " << cfg.n_per_leaf_pvfmm << "\n"
       << "# m_pvfmm:              " << cfg.m_pvfmm << "\n"
       << "# direct_enabled:       " << cfg.enable_direct << "\n"
       << "# n_direct:             " << cfg.n_direct << "\n"
       << "# log_level:            " << cfg.log_level << "\n";
}

void print_csv_header(std::ostream &os) {
    os << "dmk_time,dmk_pts_s,dmk_pts_s_rank,dmk_pts_s_thread,dmk_avg_rel_err,dmk_max_rel_err,"
       << "pvfmm_time,pvfmm_pts_s,pvfmm_pts_s_rank,pvfmm_pts_s_thread,pvfmm_avg_rel_err,pvfmm_max_rel_err";
}

void print_csv_row(const TimingResult &t_dmk, const ErrorMetrics *err_dmk, const TimingResult *t_pv,
                   const ErrorMetrics *err_pv, std::ostream &os) {
    auto nan = std::numeric_limits<double>::quiet_NaN();

    os << t_dmk.elapsed << "," << t_dmk.pts_per_sec << "," << t_dmk.pts_per_sec_per_rank << ","
       << t_dmk.pts_per_sec_per_thread << ",";

    if (err_dmk)
        os << err_dmk->avg_rel << "," << err_dmk->max_rel << ",";
    else
        os << nan << "," << nan << ",";

    if (t_pv)
        os << t_pv->elapsed << "," << t_pv->pts_per_sec << "," << t_pv->pts_per_sec_per_rank << ","
           << t_pv->pts_per_sec_per_thread << ",";
    else
        os << nan << "," << nan << "," << nan << "," << nan << ",";

    if (err_pv)
        os << err_pv->avg_rel << "," << err_pv->max_rel;
    else
        os << nan << "," << nan;
}

// Return how many items rank `r` owns when distributing `n` items over `np` ranks.
inline int local_count(int n, int np, int r) { return n / np + (r < (n % np) ? 1 : 0); }

template <typename Real>
void generate_and_scatter(int n_dim, int nd, int n_src, bool uniform, bool set_fixed_charges, std::vector<Real> &r_src,
                          std::vector<Real> &charges, long int seed, MPI_Comm comm) {
    int rank, np;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);

    const int n_local = local_count(n_src, np, rank);

    // Build scatterv displacements (in units of elements, not points)
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

    std::vector<Real> r_all, c_all;

    if (rank == 0) {
        r_all.resize(static_cast<size_t>(n_dim) * n_src);
        c_all.resize(static_cast<size_t>(nd) * n_src);

        const double rin = 0.45;
        std::default_random_engine eng(seed);
        std::uniform_real_distribution<double> rng;

        for (int i = 0; i < n_src; ++i) {
            if (!uniform && n_dim == 3) {
                double theta = rng(eng) * M_PI;
                double ct = std::cos(theta);
                double st = std::sin(theta);
                double phi = rng(eng) * 2.0 * M_PI;
                r_all[i * n_dim + 0] = static_cast<Real>(rin * st * std::cos(phi) + 0.5);
                r_all[i * n_dim + 1] = static_cast<Real>(rin * st * std::sin(phi) + 0.5);
                r_all[i * n_dim + 2] = static_cast<Real>(rin * ct + 0.5);
            } else if (!uniform && n_dim == 2) {
                double phi = rng(eng) * 2.0 * M_PI;
                r_all[i * n_dim + 0] = static_cast<Real>(std::cos(phi));
                r_all[i * n_dim + 1] = static_cast<Real>(std::sin(phi));
            } else {
                for (int j = 0; j < n_dim; ++j)
                    r_all[i * n_dim + j] = static_cast<Real>(rng(eng));
            }
            for (int j = 0; j < nd; ++j)
                c_all[i * nd + j] = static_cast<Real>(rng(eng) - 0.5);
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

    r_src.resize(static_cast<size_t>(n_dim) * n_local);
    charges.resize(static_cast<size_t>(nd) * n_local);

    MPI_Datatype mpi_t = pvfmm::par::Mpi_datatype<Real>::value();

    MPI_Scatterv(rank == 0 ? r_all.data() : nullptr, sendcounts_r.data(), displs_r.data(), mpi_t, r_src.data(),
                 n_local * n_dim, mpi_t, 0, comm);

    MPI_Scatterv(rank == 0 ? c_all.data() : nullptr, sendcounts_c.data(), displs_c.data(), mpi_t, charges.data(),
                 n_local * nd, mpi_t, 0, comm);
}

template <typename Real>
void run_direct(const std::vector<Real> &r_src, const std::vector<Real> &charges, const std::vector<Real> &r_trg,
                std::vector<Real> &pot, const pvfmm::Kernel<Real> &kernel_fn, MPI_Comm comm) {
    int np, rank;
    MPI_Comm_size(comm, &np);
    MPI_Comm_rank(comm, &rank);

    const int kdim1 = kernel_fn.ker_dim[1];
    const long long n_sl = static_cast<long long>(r_src.size()) / PVFMM_COORD_DIM;
    long long n_trg = static_cast<long long>(r_trg.size()) / PVFMM_COORD_DIM;
    long long n_trg_glb = 0;
    MPI_Allreduce(&n_trg, &n_trg_glb, 1, MPI_LONG_LONG, MPI_SUM, comm);

    // Gather all targets
    std::vector<Real> glb_trg_coord(n_trg_glb * PVFMM_COORD_DIM);
    std::vector<int> recv_cnts(np), recv_disp(np);
    {
        int send_cnt = static_cast<int>(n_trg * PVFMM_COORD_DIM);
        MPI_Allgather(&send_cnt, 1, MPI_INT, recv_cnts.data(), 1, MPI_INT, comm);
        pvfmm::omp_par::scan(recv_cnts.data(), recv_disp.data(), np);
        MPI_Allgatherv(r_trg.data(), send_cnt, pvfmm::par::Mpi_datatype<Real>::value(), glb_trg_coord.data(),
                       recv_cnts.data(), recv_disp.data(), pvfmm::par::Mpi_datatype<Real>::value(), comm);
    }

    // Evaluate
    std::vector<Real> glb_val_local(n_trg_glb * kdim1, Real(0));
    {
        int omp_p = MY_OMP_GET_MAX_THREADS();

#pragma omp parallel for schedule(static)
        for (int tid = 0; tid < omp_p; ++tid) {
            size_t a = (tid * n_trg_glb) / omp_p;
            size_t b = ((tid + 1) * n_trg_glb) / omp_p;
            if (kernel_fn.ker_poten != nullptr)
                kernel_fn.ker_poten(const_cast<Real *>(r_src.data()), n_sl, const_cast<Real *>(charges.data()), 1,
                                    glb_trg_coord.data() + a * PVFMM_COORD_DIM, b - a, glb_val_local.data() + a * kdim1,
                                    nullptr);
        }
    }

    std::vector<Real> glb_val(n_trg_glb * kdim1);
    MPI_Allreduce(glb_val_local.data(), glb_val.data(), static_cast<int>(glb_val.size()),
                  pvfmm::par::Mpi_datatype<Real>::value(), MPI_SUM, comm);

    // Extract local slice
    int offset = recv_disp[rank] / PVFMM_COORD_DIM * kdim1;
    pot.assign(glb_val.begin() + offset, glb_val.begin() + offset + n_trg * kdim1);
}

template <typename Real>
double run_dmk(pdmk_tree tree, std::vector<Real> &pot, int n_src_per_rank, MPI_Comm comm) {
    pot.resize(n_src_per_rank);
    MPI_Barrier(comm);
    double st = omp_get_wtime();

    if constexpr (std::is_same_v<Real, float>)
        pdmk_tree_evalf(tree, pot.data(), nullptr);
    else
        pdmk_tree_eval(tree, pot.data(), nullptr);

    double ft = omp_get_wtime();

    // DMK returns un-normalized Laplace; divide by 4π to match PVFMM/direct
    for (auto &v : pot)
        v /= static_cast<Real>(4.0 * M_PI);

    return ft - st;
}

template <typename Real>
double run_pvfmm_eval(pvfmm::PtFMM_Tree<Real> *pvfmm_tree, std::vector<Real> &pot, int n_src_per_rank, MPI_Comm comm) {
    pvfmm_tree->ClearFMMData();
    pot.resize(n_src_per_rank);
    MPI_Barrier(comm);

    double st = omp_get_wtime();
    pvfmm::PtFMM_Evaluate(pvfmm_tree, pot, n_src_per_rank);
    double ft = omp_get_wtime();

    return ft - st;
}

template <typename Real>
void run_comparison(const Config &cfg) {
    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    const int n_dim = 3;
    const int nd = 1;
    const int n_threads = MY_OMP_GET_MAX_THREADS();

    const int n_src = cfg.n_src;
    const int n_src_per_rank = local_count(n_src, np, rank);

    std::vector<Real> r_src, charges;
    generate_and_scatter<Real>(n_dim, nd, n_src, cfg.uniform, /*set_fixed_charges=*/true, r_src, charges, /*seed=*/0,
                               MPI_COMM_WORLD);

    pdmk_params params{};
    params.eps = cfg.eps;
    params.n_dim = n_dim;
    params.n_per_leaf = cfg.n_per_leaf_dmk;
    params.n_mfm = nd;
    params.log_level = cfg.log_level;
    params.pgh_src = DMK_POTENTIAL;
    params.pgh_trg = DMK_POTENTIAL;
    params.kernel = DMK_LAPLACE;

    std::vector<Real> r_trg; // self-evaluation: empty target list
    pdmk_tree dmk_tree;
    if constexpr (std::is_same_v<Real, float>)
        dmk_tree = pdmk_tree_createf(MPI_COMM_WORLD, params, n_src_per_rank, r_src.data(), charges.data(), nullptr,
                                     nullptr, 0, r_trg.data());
    else
        dmk_tree = pdmk_tree_create(MPI_COMM_WORLD, params, n_src_per_rank, r_src.data(), charges.data(), nullptr,
                                    nullptr, 0, r_trg.data());

    const auto kernel_fn = pvfmm::LaplaceKernel<Real>::potential();
    pvfmm::PtFMM_Tree<Real> *pvfmm_tree = nullptr;
    std::unique_ptr<pvfmm::mem::MemoryManager> mem_mgr;
    std::unique_ptr<pvfmm::PtFMM<Real>> matrices;

    if (cfg.enable_pvfmm) {
        std::vector<Real> r_dl, c_dl;
        pvfmm_tree = pvfmm::PtFMM_CreateTree(r_src, charges, r_dl, c_dl, r_src, MPI_COMM_WORLD, cfg.n_per_leaf_pvfmm,
                                             pvfmm::FreeSpace);
        mem_mgr = std::make_unique<pvfmm::mem::MemoryManager>(10'000'000);
        matrices = std::make_unique<pvfmm::PtFMM<Real>>(mem_mgr.get());
        matrices->Initialize(cfg.m_pvfmm, MPI_COMM_WORLD, &kernel_fn);
        pvfmm_tree->SetupFMM(matrices.get());
    }

    std::vector<Real> pot_direct;
    if (cfg.enable_direct) {
        int n_direct_global = (cfg.n_direct > 0) ? cfg.n_direct : n_src;
        int n_direct_per_rank = local_count(n_direct_global, np, rank);

        if (n_direct_global == n_src) {
            run_direct(r_src, charges, r_src, pot_direct, kernel_fn, MPI_COMM_WORLD);
        } else {
            std::vector<Real> r_trg(n_direct_per_rank * params.n_dim);
            for (int i = 0; i < r_trg.size(); ++i)
                r_trg[i] = r_src[i];
            run_direct(r_src, charges, r_trg, pot_direct, kernel_fn, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        print_csv_config_comment(cfg, np, n_threads, std::cout);
        print_csv_header(std::cout);
        std::cout << std::flush;
    }

    for (int run = 0; run < cfg.n_runs; ++run) {
        // DMK
        std::vector<Real> pot_dmk;
        sctl::Profile::reset();
        double dt_dmk = run_dmk<Real>(dmk_tree, pot_dmk, n_src_per_rank, MPI_COMM_WORLD);
        TimingResult t_dmk = make_timing(dt_dmk, n_src, n_src_per_rank, n_threads);
        if (run == 0) {
            if (rank == 0)
                std::cout << ",";
            pdmk_print_profile_data(MPI_COMM_WORLD, 'h');
            if (rank == 0)
                std::cout << "\n";
        }

        ErrorMetrics err_dmk{};
        bool have_err_dmk = false;
        if (cfg.enable_direct) {
            err_dmk = compute_error(pot_dmk, pot_direct, MPI_COMM_WORLD);
            have_err_dmk = true;
        }

        // PVFMM
        TimingResult t_pv{};
        ErrorMetrics err_pv{};
        bool have_pv = false;
        bool have_err_pv = false;

        if (cfg.enable_pvfmm) {
            std::vector<Real> pot_pv;
            double dt_pv = run_pvfmm_eval(pvfmm_tree, pot_pv, n_src_per_rank, MPI_COMM_WORLD);
            t_pv = make_timing(dt_pv, n_src, n_src_per_rank, n_threads);
            have_pv = true;

            if (cfg.enable_direct) {
                err_pv = compute_error(pot_pv, pot_direct, MPI_COMM_WORLD);
                have_err_pv = true;
            }
        }

        if (rank == 0) {
            print_csv_row(t_dmk, have_err_dmk ? &err_dmk : nullptr, have_pv ? &t_pv : nullptr,
                          have_err_pv ? &err_pv : nullptr, std::cout);
        }
        if (rank == 0)
            std::cout << ",";
        pdmk_print_profile_data(MPI_COMM_WORLD, 'c');
        if (rank == 0)
            std::cout << std::endl << std::flush;
    }

    pdmk_tree_destroy(dmk_tree);
    delete pvfmm_tree;
}

Config parse_args(int argc, char *argv[]) {
    Config cfg;

    // clang-format off
    static struct option long_opts[] = {
        {"pvfmm",    no_argument, nullptr, 1001},
        {"no-pvfmm", no_argument, nullptr, 1002},
        {"direct",   no_argument, nullptr, 1003},
        {"no-direct",no_argument, nullptr, 1004},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "N:n:p:m:e:t:r:D:l:uh?", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'N':  cfg.n_src            = static_cast<int>(std::atof(optarg)); break;
        case 'n':  cfg.n_per_leaf_dmk   = std::atoi(optarg);                   break;
        case 'p':  cfg.n_per_leaf_pvfmm = std::atoi(optarg);                   break;
        case 'm':  cfg.m_pvfmm          = std::atoi(optarg);                   break;
        case 'e':  cfg.eps              = std::atof(optarg);                   break;
        case 'r':  cfg.n_runs           = std::atoi(optarg);                   break;
        case 'D':  cfg.n_direct         = static_cast<int>(std::atof(optarg)); break;
        case 'l':  cfg.log_level        = std::atoi(optarg);                   break;
        case 't':
            if (optarg[0] == 'd')      cfg.prec = 'd';
            else if (optarg[0] == 'f') cfg.prec = 'f';
            else { std::cerr << "Unknown precision: " << optarg << "\n"; std::exit(1); }
            break;
        case 'u':    cfg.uniform       = true;  break;
        case 1001:   cfg.enable_pvfmm  = true;  break;
        case 1002:   cfg.enable_pvfmm  = false; break;
        case 1003:   cfg.enable_direct = true;  break;
        case 1004:   cfg.enable_direct = false; break;
        case 'h':
        case '?':
        default:
            std::cout
                << "Usage: " << argv[0] << "\n"
                << "  -N n_src               Number of source points\n"
                << "  -n n_per_leaf_dmk      DMK leaf size\n"
                << "  -p n_per_leaf_pvfmm    PVFMM leaf size\n"
                << "  -m multipole_order     PVFMM multipole order\n"
                << "  -e eps                 DMK tolerance\n"
                << "  -t f|d                 Float or double precision\n"
                << "  -r n_runs              Number of benchmark iterations\n"
                << "  -D n_direct            Number of points for direct comparison\n"
                << "  -l log_level           Verbosity of DMK logs\n"
                << "  -u                     Uniform random distribution\n"
                << "  --pvfmm / --no-pvfmm   Enable/disable PVFMM comparison\n"
                << "  --direct / --no-direct Enable/disable all-pairs reference\n"
                << "  -h                     This help message\n";
            std::exit(0);
        }
    }
    // clang-format on
    cfg.n_direct = cfg.n_direct > 0 ? cfg.n_direct : cfg.n_src;

    return cfg;
}

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    Config cfg = parse_args(argc, argv);

    if (cfg.prec == 'd')
        run_comparison<double>(cfg);
    else
        run_comparison<float>(cfg);

    MPI_Finalize();
    return 0;
}
