#include <dmk.h>

#include <iostream>
#include <limits>
#include <random>
#include <vector>

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

int main(int argc, char *argv[]) {
    int provided, rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n_dim = 3;
    int n_src = 1e6;
    const int n_trg = 0;
    const bool uniform = false;
    const bool set_fixed_charges = true;
    const int nd = 1;

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 280;
    params.n_mfm = nd;
    params.pgh_src = DMK_POTENTIAL;
    params.pgh_trg = DMK_POTENTIAL;
    params.kernel = DMK_LAPLACE;
    params.log_level = DMK_LOG_INFO;

    if (argc > 1)
        n_src = std::atoi(argv[1]);
    if (argc > 2)
        params.n_per_leaf = std::atoi(argv[2]);

    // Build random sources + 3 fixed charges
    std::vector<double> r_src, charges, rnormal, dipstr, r_trg;
    init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, dipstr, rank);

    std::vector<double> pot_src(n_src * nd), pot_trg(n_src * nd);
    pdmk(MPI_COMM_WORLD, params, n_src, r_src.data(), charges.data(), rnormal.data(), dipstr.data(), n_trg,
         r_trg.data(), pot_src.data(), nullptr, nullptr, pot_trg.data(), nullptr, nullptr);

    std::vector<double> pot_src_split(n_src * nd), pot_trg_split(n_src * nd);
    params.log_level = DMK_LOG_OFF;
    double st = omp_get_wtime();
    pdmk_tree tree = pdmk_tree_create(MPI_COMM_WORLD, params, n_src, &r_src[0], &charges[0], &rnormal[0], &dipstr[0],
                                      n_trg, &r_trg[0]);
    auto tree_build = omp_get_wtime();
    pdmk_tree_eval(tree, &pot_src_split[0], nullptr, nullptr, &pot_trg_split[0], nullptr, nullptr);
    auto tree_eval = omp_get_wtime();
    pdmk_tree_destroy(tree);

    if (rank == 0)
        std::cout << "Time: " << omp_get_wtime() - st << " " << tree_build - st << " " << tree_eval - tree_build
                  << std::endl;

    MPI_Finalize();
    return 0;
}
