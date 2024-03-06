#include <dmk.h>
#include <limits>
#include <mpi.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <dmk/fortran.h>

void init_data(int n_dim, int nd, int n_src, bool uniform, double *src, double *rnormal, double *charges,
               double *dipstr) {
    double rin = 0.45;
    double wrig = 0.12;
    double rwig = 0;
    int nwig = 6;
    int zero = 0;

    for (int i = 0; i < n_src; ++i) {
        if (!uniform) {
            double theta = hkrand_(&zero) * M_PI;
            double rr = rin + rwig * cos(nwig * theta);
            double ct = cos(theta);
            double st = sin(theta);
            double phi = hkrand_(&zero) * 2 * M_PI;
            double cp = cos(phi);
            double sp = sin(phi);

            if (n_dim == 3) {
                src[i * 3 + 0] = rr * st * cp + 0.5;
                src[i * 3 + 1] = rr * st * sp + 0.5;
                src[i * 3 + 2] = rr * ct + 0.5;
            }
        }

        for (int j = 0; j < n_dim; ++j)
            rnormal[i * n_dim + j] = hkrand_(&zero);

        for (int j = 0; j < nd; ++j) {
            charges[i * nd + j] = hkrand_(&zero) - 0.5;
            dipstr[i * nd + j] = hkrand_(&zero);
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
        src[i] = 0.0;
    for (int i = 3; i < 6; ++i)
        src[i] = 1 - std::numeric_limits<double>::epsilon();
    for (int i = 6; i < 9; ++i)
        src[i] = 0.05;
}

int main(int argc, char *argv[]) {
    int req = MPI_THREAD_SERIALIZED, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);

    constexpr int n_dim = 3;
    constexpr int n_src_per_rank = 1e6;
    constexpr int n_trg_per_rank = 0;
    constexpr int nd = 1;
    std::vector<double> X(n_dim * n_src_per_rank);
    std::vector<double> charges(nd * n_src_per_rank);
    std::vector<double> rnormal(n_dim * n_src_per_rank);
    std::vector<double> dipstr(nd * n_src_per_rank);
    std::vector<double> pot(nd * n_src_per_rank);
    std::vector<double> r_trg(n_dim * n_trg_per_rank);

    init_data(n_dim, 1, n_src_per_rank, false, X.data(), rnormal.data(), charges.data(), dipstr.data());

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 80;
    params.n_mfm = nd;
    params.pgh = DMK_POTENTIAL;
    params.kernel = DMK_YUKAWA;
    params.use_periodic = false;
    params.use_dipole = false;
    params.log_level = 0;

    pdmk(params, n_src_per_rank, X.data(), charges.data(), rnormal.data(), dipstr.data(), n_trg_per_rank, r_trg.data(),
         pot.data(), nullptr, nullptr, nullptr, nullptr, nullptr);
    pdmk(params, n_src_per_rank, X.data(), charges.data(), rnormal.data(), dipstr.data(), n_trg_per_rank, r_trg.data(),
         pot.data(), nullptr, nullptr, nullptr, nullptr, nullptr);

    MPI_Finalize();

    return 0;
}
