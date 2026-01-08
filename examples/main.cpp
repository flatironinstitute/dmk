#include <dmk.h>

#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <getopt.h>
#include <mpi.h>
#include <omp.h>

template <typename Real>
void init_test_data(int n_dim, int nd, int n_src, int n_trg, bool uniform, bool set_fixed_charges,
                    std::vector<Real> &r_src, std::vector<Real> &r_trg, std::vector<Real> &rnormal,
                    std::vector<Real> &charges, std::vector<Real> &dipstr, long seed) {
    r_src.resize(n_dim * n_src);
    r_trg.resize(n_dim * n_trg);
    charges.resize(nd * n_src);
    rnormal.resize(n_dim * n_src);
    dipstr.resize(nd * n_src);

    double rin = 0.45;
    double wrig = 0.12;
    double rwig = 0;
    int nwig = 6;
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> rng;

    for (int i = 0; i < n_src; ++i) {
        if (!uniform) {
            if (n_dim == 2) {
                double phi = rng(eng) * 2 * M_PI;
                r_src[i * 3 + 0] = cos(phi);
                r_src[i * 3 + 1] = sin(phi);
            }
            if (n_dim == 3) {
                double theta = rng(eng) * M_PI;
                double rr = rin + rwig * cos(nwig * theta);
                double ct = cos(theta);
                double st = sin(theta);
                double phi = rng(eng) * 2 * M_PI;
                double cp = cos(phi);
                double sp = sin(phi);

                r_src[i * 3 + 0] = rr * st * cp + 0.5;
                r_src[i * 3 + 1] = rr * st * sp + 0.5;
                r_src[i * 3 + 2] = rr * ct + 0.5;
            }
        } else {
            for (int j = 0; j < n_dim; ++j)
                r_src[i * n_dim + j] = rng(eng);
        }

        for (int j = 0; j < n_dim; ++j)
            rnormal[i * n_dim + j] = rng(eng);

        for (int j = 0; j < nd; ++j) {
            charges[i * nd + j] = rng(eng) - 0.5;
            dipstr[i * nd + j] = rng(eng);
        }
    }

    for (int i_trg = 0; i_trg < n_trg; ++i_trg) {
        if (!uniform) {
            if (n_dim == 2) {
                double phi = rng(eng) * 2 * M_PI;
                r_trg[i_trg * 3 + 0] = cos(phi);
                r_trg[i_trg * 3 + 1] = sin(phi);
            }
            if (n_dim == 3) {
                double theta = rng(eng) * M_PI;
                double rr = rin + rwig * cos(nwig * theta);
                double ct = cos(theta);
                double st = sin(theta);
                double phi = rng(eng) * 2 * M_PI;
                double cp = cos(phi);
                double sp = sin(phi);

                r_trg[i_trg * 3 + 0] = rr * st * cp + 0.5;
                r_trg[i_trg * 3 + 1] = rr * st * sp + 0.5;
                r_trg[i_trg * 3 + 2] = rr * ct + 0.5;
            }
        } else {
            for (int j = 0; j < n_dim; ++j)
                r_trg[i_trg * n_dim + j] = rng(eng);
        }
    }

    if (set_fixed_charges && n_src > 0)
        for (int i = 0; i < n_dim; ++i)
            r_src[i] = 0.0;
    if (set_fixed_charges && n_src > 1)
        for (int i = n_dim; i < 2 * n_dim; ++i)
            r_src[i] = 1 - std::numeric_limits<Real>::epsilon();
    if (set_fixed_charges && n_src > 2)
        for (int i = 2 * n_dim; i < 3 * n_dim; ++i)
            r_src[i] = 0.05;
}

template <typename Real>
void run_example(const pdmk_params &params, int n_src_per_rank, int n_runs, bool uniform) {
    const int n_trg = 0;
    const int n_dim = 3;
    const bool set_fixed_charges = false;
    const int nd = 1;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_threads = 1;
#pragma omp parallel
    n_threads = omp_get_num_threads();

    // Build random sources + 3 fixed charges
    std::vector<Real> r_src, charges, rnormal, dipstr, r_trg;
    init_test_data(n_dim, 1, n_src_per_rank, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, dipstr,
                   rank);

    std::vector<Real> pot_src(n_src_per_rank * nd), pot_trg(n_src_per_rank * nd);
    // FIXME: No way to update charges so completely worthless API right now :)
    std::vector<Real> pot_src_split(n_src_per_rank * nd), pot_trg_split(n_src_per_rank * nd);

    pdmk_tree tree;
    if constexpr (std::is_same_v<Real, float>)
        tree = pdmk_tree_createf(MPI_COMM_WORLD, params, n_src_per_rank, &r_src[0], &charges[0], &rnormal[0],
                                 &dipstr[0], n_trg, &r_trg[0]);
    else
        tree = pdmk_tree_create(MPI_COMM_WORLD, params, n_src_per_rank, &r_src[0], &charges[0], &rnormal[0], &dipstr[0],
                                n_trg, &r_trg[0]);

    for (int i = 0; i < n_runs; ++i) {
        const auto st = omp_get_wtime();

        if constexpr (std::is_same_v<Real, float>)
            pdmk_tree_evalf(tree, &pot_src_split[0], nullptr, nullptr, &pot_trg_split[0], nullptr, nullptr);
        else
            pdmk_tree_eval(tree, &pot_src_split[0], nullptr, nullptr, &pot_trg_split[0], nullptr, nullptr);

        pdmk_print_profile_data(MPI_COMM_WORLD);

        auto points_per_sec_per_rank = n_src_per_rank / (omp_get_wtime() - st);
        auto point_per_sec_per_thread = points_per_sec_per_rank / n_threads;
        auto points_per_sec = (n_src_per_rank * size) / (omp_get_wtime() - st);

        if (rank == 0)
            std::cout << omp_get_wtime() - st << " " << points_per_sec_per_rank << " " << point_per_sec_per_thread
                      << " " << points_per_sec << std::endl;
    }

    pdmk_tree_destroy(tree);
}

int main(int argc, char *argv[]) {
    int provided, rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n_dim = 3;
    int n_src_per_rank = 1e6;
    const int n_trg = 0;
    bool uniform = false;
    const bool set_fixed_charges = rank == 0;
    const int nd = 1;
    int n_per_leaf = 280;
    double eps = 1e-3;
    char prec = 'f';
    int log_level = 6;
    int n_runs = 10;

    int opt;
    while ((opt = getopt(argc, argv, "N:n:k:l:e:t:uh?")) != -1) {
        switch (opt) {
        case 'N':
            n_src_per_rank = std::atof(optarg);
            break;
        case 'n':
            n_per_leaf = std::atoi(optarg);
            break;
        case 'k':
            n_runs = std::atoi(optarg);
            break;
        case 'e':
            eps = std::atof(optarg);
            break;
        case 'l':
            log_level = std::atoi(optarg);
            break;
        case 't':
            if (optarg[0] == 'd')
                prec = 'd';
            else if (optarg[0] == 'f')
                prec = 'f';
            else {
                std::cerr << "Unknown precision: " << optarg << std::endl;
                return 1;
            }
            break;
        case 'u':
            uniform = true;
            break;
        case 'h':
        case '?':
        default:
            std::cout << "Usage: " << argv[0]
                      << " [-N n_src_per_rank] [-n n_per_leaf] [-k n_runs] [-e tolerance] [-l log_level] "
                         "[-t float_or_double] [-u] [-h]"
                      << std::endl;
            break;
        }
    }

    pdmk_params params;
    params.eps = eps;
    params.n_dim = n_dim;
    params.n_per_leaf = n_per_leaf;
    params.n_mfm = nd;
    params.pgh_src = DMK_POTENTIAL;
    params.pgh_trg = DMK_POTENTIAL;
    params.kernel = DMK_LAPLACE;
    params.log_level = log_level;

    if (prec == 'f')
        run_example<float>(params, n_src_per_rank, n_runs, uniform);
    else
        run_example<double>(params, n_src_per_rank, n_runs, uniform);

    MPI_Finalize();
    return 0;
}
