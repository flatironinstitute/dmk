#include <dmk.h>
#include <mpi.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

void init_data(int n_dim, int nd, int n_src, bool uniform, std::vector<double> &r_src, std::vector<double> &rnormal,
               std::vector<double> &charges, std::vector<double> &dipstr, long seed) {
    r_src.resize(n_dim * n_src);
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
        }
        else {
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

    for (int i = 0; i < nd; ++i) {
        dipstr[0 * nd + i] = 0.0;
        dipstr[1 * nd + i] = 0.0;
        charges[0 * nd + i] = 0.0;
        charges[1 * nd + i] = 0.0;
        charges[2 * nd + i] = 1.0;
    }

    for (int i = 0; i < 3; ++i)
        r_src[i] = 0.0;
    for (int i = 3; i < 6; ++i)
        r_src[i] = 1 - std::numeric_limits<double>::epsilon();
    for (int i = 6; i < 9; ++i)
        r_src[i] = 0.05;
}

std::vector<double> scatter(const std::vector<double> root, int dim) {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int N = root.size() / dim;
    int N_local = N / world_size + (rank < (N % world_size));

    std::vector<int> counts(world_size);
    std::vector<int> displs(world_size + 1);
    for (int i = 0; i < world_size; ++i) {
        counts[i] = (N / world_size + (i < (N % world_size))) * dim;
        displs[i + 1] = displs[i] + counts[i];
    }

    MPI_Bcast(&N_local, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> res(N_local * dim);
    MPI_Scatterv(root.data(), counts.data(), displs.data(), MPI_DOUBLE, res.data(), res.size(), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
    return res;
}

int main(int argc, char *argv[]) {
    int req = MPI_THREAD_SERIALIZED, prov, rank, world_size;
    MPI_Init_thread(&argc, &argv, req, &prov);

    assert(prov == req);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    constexpr int n_dim = 3;
    constexpr int n_src = 1e6;
    constexpr int n_trg = 0;
    constexpr int nd = 1;
    int n_src_local = n_src / world_size + (rank < (n_src % world_size));
    int n_trg_local = n_trg / world_size + (rank < (n_trg % world_size));

    std::vector<double> X_root, charges_root, rnormal_root, dipstr_root, pot_root, r_trg_root;
    if (rank == 0)
        init_data(n_dim, 1, n_src, true, X_root, rnormal_root, charges_root, dipstr_root, 0);

    auto r_src = scatter(X_root, n_dim);
    auto r_trg = scatter(r_trg_root, n_dim);
    auto rnormal = scatter(rnormal_root, n_dim);
    auto charges = scatter(charges_root, nd);
    auto dipstr = scatter(dipstr_root, nd);
    std::vector<double> pot(nd * n_src_local);

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 10000;
    params.n_mfm = nd;
    params.pgh = DMK_POTENTIAL;
    params.kernel = DMK_YUKAWA;
    params.use_periodic = false;
    params.use_dipole = false;
    params.log_level = 0;

    pdmk(params, n_src_local, r_src.data(), charges.data(), rnormal.data(), dipstr.data(), n_trg_local, r_trg.data(),
         pot.data(), nullptr, nullptr, nullptr, nullptr, nullptr);
    pdmk(params, n_src_local, r_src.data(), charges.data(), rnormal.data(), dipstr.data(), n_trg_local, r_trg.data(),
         pot.data(), nullptr, nullptr, nullptr, nullptr, nullptr);

    MPI_Finalize();

    return 0;
}
