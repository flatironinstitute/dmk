#include <dmk.h>
#include <mpi.h>

#include <cassert>
#include <cstdlib>
#include <vector>

int main(int argc, char *argv[]) {
    int req = MPI_THREAD_SERIALIZED, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);

    constexpr int DIM = 3;
    constexpr int n_src_per_rank = 1e6;
    std::vector<double> X(DIM * n_src_per_rank);
    std::vector<double> rho(n_src_per_rank);
    std::vector<double> pot(n_src_per_rank);

    for (auto &x : X)
        x = drand48();

    pdmk_params params;
    params.eps = 1e-8;
    params.n_dim = DIM;
    params.n_per_leaf = 2000;
    params.n_mfm = 1;
    params.pgh = DMK_POTENTIAL;
    params.kernel = DMK_YUKAWA;
    params.use_periodic = false;
    params.use_dipole = false;
    params.log_level = 0;

    pdmk(params, n_src_per_rank, X.data(), pot.data(), nullptr, nullptr, 0, nullptr, pot.data(), nullptr, nullptr,
         nullptr, nullptr, nullptr);
    pdmk(params, n_src_per_rank, X.data(), pot.data(), nullptr, nullptr, 0, nullptr, pot.data(), nullptr, nullptr,
         nullptr, nullptr, nullptr);

    MPI_Finalize();

    return 0;
}
