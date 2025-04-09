#include <dmk.h>
#include <pvfmm.hpp>

#include <getopt.h>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <omp.h>

template <typename Real>
void direct_eval(std::vector<Real> &sl_coord, std::vector<Real> &sl_den, std::vector<Real> &dl_coord,
                 std::vector<Real> &dl_den, std::vector<Real> &trg_coord, std::vector<Real> &trg_value,
                 const pvfmm::Kernel<Real> &kernel_fn, MPI_Comm comm) {
    int np, rank;
    MPI_Comm_size(comm, &np);
    MPI_Comm_rank(comm, &rank);

    long long n_sl = sl_coord.size() / PVFMM_COORD_DIM;
    long long n_dl = dl_coord.size() / PVFMM_COORD_DIM;
    long long n_trg_glb = 0, n_trg = trg_coord.size() / PVFMM_COORD_DIM;
    MPI_Allreduce(&n_trg, &n_trg_glb, 1, MPI_LONG_LONG, MPI_SUM, comm);

    std::vector<Real> glb_trg_coord(n_trg_glb * PVFMM_COORD_DIM);
    std::vector<Real> glb_trg_value(n_trg_glb * kernel_fn.ker_dim[1]);
    std::vector<int> recv_disp(np);
    { // Gather all target coordinates.
        int send_cnt = n_trg * PVFMM_COORD_DIM;
        std::vector<int> recv_cnts(np);
        MPI_Allgather(&send_cnt, 1, MPI_INT, &recv_cnts[0], 1, MPI_INT, comm);
        pvfmm::omp_par::scan(&recv_cnts[0], &recv_disp[0], np);
        MPI_Allgatherv(&trg_coord[0], send_cnt, pvfmm::par::Mpi_datatype<Real>::value(), &glb_trg_coord[0],
                       &recv_cnts[0], &recv_disp[0], pvfmm::par::Mpi_datatype<Real>::value(), comm);
    }

    { // Evaluate target potential.
        std::vector<Real> glb_trg_value_(n_trg_glb * kernel_fn.ker_dim[1]);
        int omp_p = omp_get_max_threads();
#pragma omp parallel for
        for (int i = 0; i < omp_p; i++) {
            size_t a = (i * n_trg_glb) / omp_p;
            size_t b = ((i + 1) * n_trg_glb) / omp_p;

            if (kernel_fn.ker_poten != NULL)
                kernel_fn.ker_poten(&sl_coord[0], n_sl, &sl_den[0], 1, &glb_trg_coord[0] + a * PVFMM_COORD_DIM, b - a,
                                    &glb_trg_value_[0] + a * kernel_fn.ker_dim[1], NULL);

            if (kernel_fn.dbl_layer_poten != NULL && n_dl)
                kernel_fn.dbl_layer_poten(&dl_coord[0], n_dl, &dl_den[0], 1, &glb_trg_coord[0] + a * PVFMM_COORD_DIM,
                                          b - a, &glb_trg_value_[0] + a * kernel_fn.ker_dim[1], NULL);
        }
        MPI_Allreduce(&glb_trg_value_[0], &glb_trg_value[0], glb_trg_value.size(),
                      pvfmm::par::Mpi_datatype<Real>::value(), MPI_SUM, comm);
    }

    // Get local target values.
    trg_value.assign(&glb_trg_value[0] + recv_disp[rank] / PVFMM_COORD_DIM * kernel_fn.ker_dim[1],
                     &glb_trg_value[0] + (recv_disp[rank] / PVFMM_COORD_DIM + n_trg) * kernel_fn.ker_dim[1]);
}

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
void run_comparison(pdmk_params params, int n_src, int m, int n_per_leaf_pvfmm, bool uniform) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int n_trg = 0;
    const bool set_fixed_charges = rank == 0;
    const int nd = 1;
    const int n_dim = 3;

    int n_threads = 1;
#pragma omp parallel
    n_threads = omp_get_num_threads();

    int n_src_per_rank = n_src / size;
    n_src = n_src_per_rank * size;

    // Build random sources + 3 fixed charges
    std::vector<Real> r_src, charges, rnormal, dipstr, r_trg;
    init_test_data(n_dim, 1, n_src_per_rank, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, dipstr,
                   rank);

    std::vector<Real> pot_src_split(n_src_per_rank * nd), pot_trg_split(n_src_per_rank * nd);

    pdmk_tree tree;
    if constexpr (std::is_same_v<Real, float>)
        tree = pdmk_tree_createf(MPI_COMM_WORLD, params, n_src_per_rank, &r_src[0], &charges[0], &rnormal[0],
                                 &dipstr[0], n_trg, &r_trg[0]);
    else
        tree = pdmk_tree_create(MPI_COMM_WORLD, params, n_src_per_rank, &r_src[0], &charges[0], &rnormal[0], &dipstr[0],
                                n_trg, &r_trg[0]);

    const auto kernel_fn = pvfmm::LaplaceKernel<Real>::potential();
    std::vector<Real> r_dl, charges_dl;
    auto *pvfmm_tree = pvfmm::PtFMM_CreateTree(r_src, charges, r_dl, charges_dl, r_src, MPI_COMM_WORLD,
                                               n_per_leaf_pvfmm, pvfmm::FreeSpace);
    pvfmm::PtFMM<Real> matrices;
    matrices.Initialize(m, MPI_COMM_WORLD, &kernel_fn);
    pvfmm_tree->SetupFMM(&matrices);
    std::vector<Real> pot_src_pvfmm(n_src_per_rank * nd);

    std::vector<Real> pot_direct;
    direct_eval(r_src, charges, r_dl, charges_dl, r_src, pot_direct, kernel_fn, MPI_COMM_WORLD);

    const int n_runs = 100;
    for (int i = 0; i < n_runs; ++i) {
        pvfmm_tree->ClearFMMData();
        const auto st = omp_get_wtime();

        if constexpr (std::is_same_v<Real, float>)
            pdmk_tree_evalf(tree, &pot_src_split[0], nullptr, nullptr, &pot_trg_split[0], nullptr, nullptr);
        else
            pdmk_tree_eval(tree, &pot_src_split[0], nullptr, nullptr, &pot_trg_split[0], nullptr, nullptr);
        pdmk_print_profile_data(MPI_COMM_WORLD);

        auto ft = omp_get_wtime();
        auto points_per_sec_per_rank = n_src_per_rank / (ft - st);
        auto point_per_sec_per_thread = points_per_sec_per_rank / n_threads;
        auto points_per_sec = n_src / (ft - st);

        const auto st2 = omp_get_wtime();
        pvfmm::PtFMM_Evaluate(pvfmm_tree, pot_src_pvfmm, n_src_per_rank);
        const auto ft2 = omp_get_wtime();
        auto points_per_sec_per_rank_pvfmm = n_src_per_rank / (ft2 - st2);
        auto point_per_sec_per_thread_pvfmm = points_per_sec_per_rank_pvfmm / n_threads;
        auto points_per_sec_pvfmm = n_src / (ft2 - st2);

        double max_rel_err_dmk = 0;
        double avg_rel_err_dmk = 0.0;
        double direct_sum = 0.0;
        for (int i = 0; i < n_src_per_rank; ++i) {
            avg_rel_err_dmk += sctl::pow<2>(pot_src_split[i] - 4 * M_PI * pot_direct[i]);
            direct_sum += sctl::pow<2>(pot_direct[i]);
            max_rel_err_dmk =
                std::max((Real)max_rel_err_dmk, std::abs(pot_src_split[i] - 4 * Real(M_PI) * pot_direct[i]) /
                                                    std::abs(4 * Real(M_PI) * pot_direct[i]));
        }
        avg_rel_err_dmk = std::sqrt(avg_rel_err_dmk / direct_sum) / (4 * M_PI);
        MPI_Allreduce(&avg_rel_err_dmk, &avg_rel_err_dmk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&max_rel_err_dmk, &max_rel_err_dmk, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        avg_rel_err_dmk /= size;

        double max_rel_err_pvfmm = 0;
        double avg_rel_err_pvfmm = 0.0;
        for (int i = 0; i < n_src_per_rank; ++i) {
            avg_rel_err_pvfmm += sctl::pow<2>(pot_src_pvfmm[i] - pot_direct[i]);
            max_rel_err_pvfmm =
                std::max((Real)max_rel_err_pvfmm, std::abs(pot_src_pvfmm[i] - pot_direct[i]) / std::abs(pot_direct[i]));
        }
        avg_rel_err_pvfmm = std::sqrt(avg_rel_err_pvfmm / direct_sum);
        MPI_Allreduce(&avg_rel_err_pvfmm, &avg_rel_err_pvfmm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&max_rel_err_pvfmm, &max_rel_err_pvfmm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        avg_rel_err_pvfmm /= size;

        if (rank == 0) {
            std::cout << ft - st << " " << points_per_sec_per_rank << " " << point_per_sec_per_thread << " "
                      << points_per_sec << " " << max_rel_err_dmk << " " << avg_rel_err_dmk << std::endl;
            std::cout << ft2 - st2 << " " << points_per_sec_per_rank_pvfmm << " " << point_per_sec_per_thread_pvfmm
                      << " " << points_per_sec_pvfmm << " " << max_rel_err_pvfmm << " " << avg_rel_err_pvfmm
                      << std::endl;
            std::cout << (ft - st) / (ft2 - st2) << std::endl << std::endl;
        }
    }

    pdmk_tree_destroy(tree);
    delete pvfmm_tree;
}

int main(int argc, char *argv[]) {
    int provided, rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    const int n_dim = 3;
    int n_src = 1e6;
    const int n_trg = 0;
    bool uniform = false;
    const bool set_fixed_charges = rank == 0;
    const int nd = 1;
    int n_per_leaf_dmk = 280;
    int n_per_leaf_pvfmm = 600;
    double eps = 1e-5;
    int m = 6;
    char prec = 'f';

    int opt;
    while ((opt = getopt(argc, argv, "N:n:p:m:e:t:uh?")) != -1) {
        switch (opt) {
        case 'N':
            n_src = std::atof(optarg);
            break;
        case 'n':
            n_per_leaf_dmk = std::atoi(optarg);
            break;
        case 'p':
            n_per_leaf_pvfmm = std::atoi(optarg);
            break;
        case 'm':
            m = std::atoi(optarg);
            break;
        case 'e':
            eps = std::atof(optarg);
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
                      << " [-N n_src] [-n n_per_leaf_dmk] [-p n_per_leaf_pvfmm] [-m multipole_order_dmk] [-e eps_dmk] "
                         "[-t float_or_double] [-u] [-h]"
                      << std::endl;
            break;
        }
    }

    pdmk_params params;
    params.eps = eps;
    params.n_dim = n_dim;
    params.n_per_leaf = n_per_leaf_dmk;
    params.n_mfm = nd;
    params.pgh_src = DMK_POTENTIAL;
    params.pgh_trg = DMK_POTENTIAL;
    params.kernel = DMK_LAPLACE;
    params.log_level = DMK_LOG_OFF;

    if (prec == 'd')
        run_comparison<double>(params, n_src, m, n_per_leaf_pvfmm, uniform);
    else if (prec == 'f')
        run_comparison<float>(params, n_src, m, n_per_leaf_pvfmm, uniform);
    else {
        std::cerr << "Unknown precision: " << prec << std::endl;
        return 1;
    }

    MPI_Finalize();
    return 0;
}
