#include <algorithm>
#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/logger.h>
#include <dmk/prolate0_fun.hpp>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/tree.hpp>
#include <dmk/util.hpp>
#include <sctl.hpp>

#include <mpi.h>
#include <omp.h>

#include <doctest/extensions/doctest_mpi.h>

namespace dmk {

template <typename T, int DIM>
void pdmk(MPI_Comm comm, const pdmk_params &params, int n_src, const T *r_src, const T *charge, const T *normal,
          const T *dipole_str, int n_trg, const T *r_trg, T *pot_src, T *grad_src, T *hess_src, T *pot_trg, T *grad_trg,
          T *hess_trg) {
    const auto &sctl_comm = sctl::Comm(comm);
    auto &logger = dmk::get_logger(sctl_comm, params.log_level);
    auto &rank_logger = dmk::get_rank_logger(sctl_comm, params.log_level);
    logger->info("PDMK called");
    auto st = omp_get_wtime();

    sctl::Vector<T> r_src_vec(n_src * params.n_dim, const_cast<T *>(r_src), false);
    sctl::Vector<T> r_trg_vec(n_trg * params.n_dim, const_cast<T *>(r_trg), false);
    sctl::Vector<T> charge_vec(n_src * params.n_mfm, const_cast<T *>(charge), false);

    DMKPtTree<T, DIM> tree(sctl_comm, params, r_src_vec, r_trg_vec, charge_vec);
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
        MPI_Reduce(MPI_IN_PLACE, &N, 1, MPI_INT, MPI_SUM, 0, comm);
    else
        MPI_Reduce(&N, &N, 1, MPI_INT, MPI_SUM, 0, comm);

    logger->info("PDMK finished in {:.4f} seconds ({:.0f} pts/s, {:.0f} pts/s/rank)", dt, N / dt,
                 N / dt / sctl_comm.Size());
}

MPI_TEST_CASE("[DMK] pdmk 3d float", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 10000;
    constexpr int n_trg = 10000;
    constexpr int nd = 1;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;

    sctl::Vector<double> r_src, pot_src, grad_src, hess_src, charges, rnormal, dipstr, pot_trg, r_trg;
    sctl::Vector<float> r_srcf, pot_srcf, grad_srcf, hess_srcf, chargesf, rnormalf, dipstrf, pot_trgf, r_trgf;
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges,
                              dipstr, 0);
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_srcf, r_trgf, rnormalf, chargesf,
                              dipstrf, 0);
    pot_src.ReInit(n_src * nd);
    pot_trg.ReInit(n_trg * nd);
    pot_srcf.ReInit(n_src * nd);
    pot_trgf.ReInit(n_trg * nd);

    pdmk_params params;
    params.eps = 1e-10;
    params.n_dim = n_dim;
    params.n_per_leaf = 80;
    params.n_mfm = nd;
    params.pgh_src = DMK_POTENTIAL;
    params.pgh_trg = DMK_POTENTIAL;
    params.kernel = DMK_YUKAWA;
    params.fparam = 6.0;
    params.log_level = SPDLOG_LEVEL_OFF;

    pdmk(test_comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], &dipstr[0], n_trg, &r_trg[0], &pot_src[0],
         nullptr, nullptr, &pot_trg[0], nullptr, nullptr);

    params.eps = 1e-5;
    pdmkf(test_comm, params, n_src, &r_srcf[0], &chargesf[0], &rnormalf[0], &dipstrf[0], n_trg, &r_trgf[0],
          &pot_srcf[0], nullptr, nullptr, &pot_trgf[0], nullptr, nullptr);

    double l2_err_src = 0.0;
    double l2_err_trg = 0.0;
    for (int i = 0; i < n_src; ++i)
        l2_err_src += pot_src[i] != 0.0 ? sctl::pow<2>(1.0 - pot_srcf[i] / pot_src[i]) : 0.0;
    for (int i = 0; i < n_trg; ++i)
        l2_err_trg += pot_trg[i] != 0.0 ? sctl::pow<2>(1.0 - pot_trgf[i] / pot_trg[i]) : 0.0;

    l2_err_src = std::sqrt(l2_err_src) / n_src;
    l2_err_trg = std::sqrt(l2_err_trg) / n_trg;
    CHECK(l2_err_src < 6 * params.eps);
    CHECK(l2_err_trg < 6 * params.eps);
}

MPI_TEST_CASE("[DMK] pdmk 3d all", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 10000;
    constexpr int nd = 1;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;

    sctl::Vector<double> r_src, pot_src, grad_src, hess_src, charges, rnormal, dipstr, pot_trg, r_trg, grad_trg,
        hess_trg;
    dmk::util::init_test_data(n_dim, 1, n_src, 0, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, dipstr,
                              0);
    r_trg = r_src;
    std::reverse(r_trg.begin(), r_trg.end());
    r_trg.ReInit(n_dim * (n_src - set_fixed_charges * 3));
    const int n_trg = r_trg.Dim() / n_dim;

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 80;
    params.n_mfm = nd;
    params.pgh_src = DMK_POTENTIAL;
    params.pgh_trg = DMK_POTENTIAL;
    params.kernel = DMK_YUKAWA;
    params.fparam = 6.0;
    params.log_level = SPDLOG_LEVEL_OFF;

    auto get_pot_func = [&n_dim, &params](dmk_ikernel kernel) -> std::function<double(double *, double *)> {
        switch (kernel) {
        case DMK_YUKAWA:
            return [&n_dim, &params](double *r_a, double *r_b) {
                double dr = 0.0;
                for (int j = 0; j < n_dim; ++j)
                    dr += sctl::pow<2>(r_a[j] - r_b[j]);

                if (!dr)
                    return 0.0;

                dr = std::sqrt(dr);
                return std::exp(-params.fparam * dr) / dr;
            };
        case DMK_LAPLACE:
            return [&n_dim](double *r_a, double *r_b) {
                double dr2 = 0.0;
                for (int j = 0; j < n_dim; ++j)
                    dr2 += sctl::pow<2>(r_a[j] - r_b[j]);

                if (!dr2)
                    return 0.0;

                return 1.0 / std::sqrt(dr2);
            };
        case DMK_SQRT_LAPLACE:
            return [&n_dim](double *r_a, double *r_b) {
                double dr2 = 0.0;
                for (int j = 0; j < n_dim; ++j)
                    dr2 += sctl::pow<2>(r_a[j] - r_b[j]);

                if (!dr2)
                    return 0.0;

                return 1.0 / dr2;
            };
        default:
            throw std::runtime_error("Unknown kernel");
        }
    };
    int ndiv[3] = {80, 280, 280};

    const auto test_kernels = {
        DMK_YUKAWA,
        DMK_LAPLACE,
        DMK_SQRT_LAPLACE,
    };
    for (auto kernel : test_kernels) {
        const std::string kernel_str = [&kernel]() {
            switch (kernel) {
            case DMK_YUKAWA:
                return "YUKAWA";
            case DMK_LAPLACE:
                return "LAPLACE";
            case DMK_SQRT_LAPLACE:
                return "SQRT_LAPLACE";
            default:
                throw std::runtime_error("Unknown kernel");
            }
        }();

        SUBCASE(kernel_str.c_str()) {
            sctl::Vector<double> pot_src(n_src * nd), grad_src(n_src * nd * n_dim),
                hess_src(n_src * nd * n_dim * n_dim), pot_trg(n_src * nd), grad_trg(n_trg * nd * n_dim),
                hess_trg(n_trg * nd * n_dim * n_dim);
            params.n_per_leaf = ndiv[int(kernel)];

            auto pot_src_fort = pot_src;
            auto grad_src_fort = grad_src;
            auto hess_src_fort = hess_src;
            auto pot_trg_fort = pot_trg;
            auto grad_trg_fort = grad_trg;
            auto hess_trg_fort = hess_trg;

            params.kernel = kernel;
            auto potential = get_pot_func(kernel);

            const int n_test_src = std::min(n_src, 100);
            const int n_test_trg = std::min(n_trg, 100);
            std::vector<double> test_src(n_test_src);
            std::vector<double> test_trg(n_test_trg);

            for (int i_src = 0; i_src < n_src; ++i_src) {
                for (int i_trg = 0; i_trg < n_test_src; ++i_trg)
                    test_src[i_trg] += charges[i_src] * potential(&r_src[i_src * n_dim], &r_src[i_trg * n_dim]);
                for (int i_trg = 0; i_trg < n_test_trg; ++i_trg)
                    test_trg[i_trg] += charges[i_src] * potential(&r_src[i_src * n_dim], &r_trg[i_trg * n_dim]);
            }

            pdmk(test_comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], &dipstr[0], n_trg, &r_trg[0],
                 &pot_src[0], nullptr, nullptr, &pot_trg[0], nullptr, nullptr);

            // double tottimeinfo[20];
            // int use_dipole = 0;
            // double st = omp_get_wtime();
            // pdmk_(&nd, &n_dim, &params.eps, (int *)&params.kernel, &params.fparam, &params.use_periodic, &n_src,
            //       &r_src[0], &params.use_charge, &charges[0], &use_dipole, nullptr, nullptr, (int *)&params.pgh_src,
            //       &pot_src_fort[0], &grad_src_fort[0], &hess_src_fort[0], &n_trg, &r_trg[0], (int *)&params.pgh_trg,
            //       &pot_trg_fort[0], &grad_trg_fort[0], &hess_trg_fort[0], tottimeinfo);
            // std::cout << omp_get_wtime() - st << std::endl;

            double l2_err_src = 0.0;
            double l2_err_trg = 0.0;
            for (int i = 0; i < n_test_src; ++i)
                l2_err_src += test_src[i] != 0.0 ? sctl::pow<2>(1.0 - pot_src[i] / test_src[i]) : 0.0;
            for (int i = 0; i < n_test_trg; ++i)
                l2_err_trg += test_trg[i] != 0.0 ? sctl::pow<2>(1.0 - pot_trg[i] / test_trg[i]) : 0.0;

            l2_err_src = std::sqrt(l2_err_src) / n_test_src;
            l2_err_trg = std::sqrt(l2_err_trg) / n_test_trg;
            CHECK(l2_err_src < 5 * params.eps);
            CHECK(l2_err_trg < 5 * params.eps);
        }
    }
}

} // namespace dmk

extern "C" {

void pdmkf(MPI_Comm comm, pdmk_params params, int n_src, const float *r_src, const float *charge, const float *normal,
           const float *dipole_str, int n_trg, const float *r_trg, float *pot_src, float *grad_src, float *hess_src,
           float *pot_trg, float *grad_trg, float *hess_trg) {
    if (params.n_dim == 2)
        return dmk::pdmk<float, 2>(comm, params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot_src,
                                   grad_src, hess_src, pot_trg, grad_trg, hess_trg);
    if (params.n_dim == 3)
        return dmk::pdmk<float, 3>(comm, params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot_src,
                                   grad_src, hess_src, pot_trg, grad_trg, hess_trg);
}

void pdmk(MPI_Comm comm, pdmk_params params, int n_src, const double *r_src, const double *charge, const double *normal,
          const double *dipole_str, int n_trg, const double *r_trg, double *pot_src, double *grad_src, double *hess_src,
          double *pot_trg, double *grad_trg, double *hess_trg) {
    if (params.n_dim == 2)
        return dmk::pdmk<double, 2>(comm, params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot_src,
                                    grad_src, hess_src, pot_trg, grad_trg, hess_trg);
    if (params.n_dim == 3)
        return dmk::pdmk<double, 3>(comm, params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot_src,
                                    grad_src, hess_src, pot_trg, grad_trg, hess_trg);
}
}
