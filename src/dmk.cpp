#include <algorithm>
#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/logger.h>
#include <dmk/prolate_funcs.hpp>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/tree.hpp>
#include <dmk/util.hpp>
#include <sctl.hpp>

#include <mpi.h>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <doctest/extensions/doctest_mpi.h>

namespace dmk {

template <typename T, int DIM>
void pdmk(const pdmk_params &params, int n_src, const T *r_src, const T *charge, const T *normal, const T *dipole_str,
          int n_trg, const T *r_trg, T *pot_src, T *grad_src, T *hess_src, T *pot_trg, T *grad_trg, T *hess_trg) {
    auto &logger = dmk::get_logger(params.log_level);
    auto &rank_logger = dmk::get_rank_logger(params.log_level);
    logger->info("PDMK called");
    auto st = omp_get_wtime();

    sctl::Vector<T> r_src_vec(n_src * params.n_dim, const_cast<T *>(r_src), false);
    sctl::Vector<T> r_trg_vec(n_trg * params.n_dim, const_cast<T *>(r_trg), false);
    sctl::Vector<T> charge_vec(n_src * params.n_mfm, const_cast<T *>(charge), false);

    DMKPtTree<T, DIM> tree(sctl::Comm::World(), params, r_src_vec, r_trg_vec, charge_vec);
    sctl::Comm::Self();
    tree.upward_pass();
    tree.downward_pass();

    sctl::Vector<T> res;
    tree.GetParticleData(res, "pdmk_pot_src");
    sctl::Vector<T>(res.Dim(), pot_src, false) = res;
    tree.GetParticleData(res, "pdmk_pot_trg");
    sctl::Vector<T>(res.Dim(), pot_trg, false) = res;

    auto dt = omp_get_wtime() - st;
    int N = n_src + n_trg;
    if (tree.GetComm().Rank() == 0)
        MPI_Reduce(MPI_IN_PLACE, &N, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    else
        MPI_Reduce(&N, &N, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    logger->info("PDMK finished in {:.4f} seconds ({:.0f} pts/s)", dt, N / dt);
}

MPI_TEST_CASE("[DMK] pdmk 3d", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 10000;
    constexpr int nd = 1;

    std::vector<double> r_src, pot_src, grad_src, hess_src, charges, rnormal, dipstr, pot_trg, r_trg, grad_trg,
        hess_trg;
    dmk::util::init_test_data(n_dim, 1, n_src, false, r_src, rnormal, charges, dipstr, 0);
    r_trg = r_src;
    std::reverse(r_trg.begin(), r_trg.end());
    r_trg.resize(n_dim * (n_src - 3));
    const int n_trg = r_trg.size() / n_dim;

    pot_src.resize(n_src * nd);
    grad_src.resize(n_src * nd * n_dim);
    hess_src.resize(n_src * nd * n_dim * n_dim);
    pot_trg.resize(n_trg * nd);
    grad_trg.resize(n_trg * nd * n_dim);
    hess_trg.resize(n_trg * nd * n_dim * n_dim);

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 80;
    params.n_mfm = nd;
    params.pgh_src = DMK_POTENTIAL;
    params.kernel = DMK_YUKAWA;
    params.fparam = 6.0;
    params.log_level = SPDLOG_LEVEL_OFF;

    auto yukawa = [&n_dim, &charges, &params](double *r_a, double *r_b) {
        double dr = 0.0;
        for (int j = 0; j < n_dim; ++j)
            dr += sctl::pow<2>(r_a[j] - r_b[j]);
        dr = std::sqrt(dr);
        if (!dr)
            return 0.0;

        return std::exp(-params.fparam * dr) / dr;
    };

    double test_src = 0.0;
    double test_trg = 0.0;
    int test_src_i = n_src / 3;
    int test_trg_i = n_trg / 3;

    for (int i = 0; i < n_src; ++i) {
        test_src += charges[i] * yukawa(&r_src[i * n_dim], &r_src[test_src_i * n_dim]);
        test_trg += charges[i] * yukawa(&r_src[i * n_dim], &r_trg[test_trg_i * n_dim]);
    }

    for (int i = 0; i < 2; ++i) {
        pdmk(params, n_src, r_src.data(), charges.data(), rnormal.data(), dipstr.data(), n_trg, r_trg.data(),
             pot_src.data(), nullptr, nullptr, pot_trg.data(), nullptr, nullptr);
        params.log_level = SPDLOG_LEVEL_INFO;
    }

    REQUIRE(std::abs(1.0 - pot_src[test_src_i] / test_src) < params.eps);
    REQUIRE(std::abs(1.0 - pot_trg[test_trg_i] / test_trg) < params.eps);
}

} // namespace dmk

extern "C" {
// void pdmkf(pdmk_params params, int n_src, const float *r_src, const float *charge, const float *normal,
//            const float *dipole_str, int n_trg, const float *r_trg, float *pot, float *grad, float *hess, float
//            *pottarg, float *gradtarg, float *hesstarg) {
//     if (params.n_dim == 2)
//         return dmk::pdmk<float, 2>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot, grad, hess,
//                                    pottarg, gradtarg, hesstarg);
//     if (params.n_dim == 3)
//         return dmk::pdmk<float, 3>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot, grad, hess,
//                                    pottarg, gradtarg, hesstarg);
// }

void pdmk(pdmk_params params, int n_src, const double *r_src, const double *charge, const double *normal,
          const double *dipole_str, int n_trg, const double *r_trg, double *pot_src, double *grad_src, double *hess_src,
          double *pot_trg, double *grad_trg, double *hess_trg) {
    if (params.n_dim == 2)
        return dmk::pdmk<double, 2>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot_src, grad_src,
                                    hess_src, pot_trg, grad_trg, hess_trg);
    if (params.n_dim == 3)
        return dmk::pdmk<double, 3>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot_src, grad_src,
                                    hess_src, pot_trg, grad_trg, hess_trg);
}
}
