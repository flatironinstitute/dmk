#include <algorithm>
#include <string>
#include <variant>

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

using pdmk_tree_impl =
    std::variant<std::unique_ptr<dmk::DMKPtTree<float, 2>>, std::unique_ptr<dmk::DMKPtTree<float, 3>>,
                 std::unique_ptr<dmk::DMKPtTree<double, 2>>, std::unique_ptr<dmk::DMKPtTree<double, 3>>>;

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
    if (sctl_comm.Rank() == 0)
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

    auto get_pot_func = [&params](int n_dim, dmk_ikernel kernel) -> std::function<double(double *, double *)> {
        auto distance2 = [](double *r_a, double *r_b, int n_dim) {
            double dr2 = 0.0;
            for (int j = 0; j < n_dim; ++j)
                dr2 += sctl::pow<2>(r_a[j] - r_b[j]);

            return dr2;
        };

        switch (kernel) {
        case DMK_YUKAWA:
            if (n_dim == 2)
                return [distance2, &params](double *r_a, double *r_b) {
                    double dr = distance2(r_a, r_b, 2);
                    if (!dr)
                        return 0.0;

                    dr = std::sqrt(dr);
                    return std::cyl_bessel_k(0, params.fparam * dr);
                };
            if (n_dim == 3)
                return [distance2, &params](double *r_a, double *r_b) {
                    double dr = distance2(r_a, r_b, 3);
                    if (!dr)
                        return 0.0;

                    dr = std::sqrt(dr);
                    return std::exp(-params.fparam * dr) / dr;
                };
        case DMK_LAPLACE:
            if (n_dim == 2)
                return [](double *r_a, double *r_b) {
                    double dr2 = 0.0;
                    for (int j = 0; j < 2; ++j)
                        dr2 += sctl::pow<2>(r_a[j] - r_b[j]);

                    if (!dr2)
                        return 0.0;

                    return 0.5 * std::log(dr2);
                };
            if (n_dim == 3)
                return [](double *r_a, double *r_b) {
                    double dr2 = 0.0;
                    for (int j = 0; j < 3; ++j)
                        dr2 += sctl::pow<2>(r_a[j] - r_b[j]);

                    if (!dr2)
                        return 0.0;

                    return 1.0 / std::sqrt(dr2);
                };
        case DMK_SQRT_LAPLACE:
            if (n_dim == 2)
                return [](double *r_a, double *r_b) {
                    double dr2 = 0.0;
                    for (int j = 0; j < 2; ++j)
                        dr2 += sctl::pow<2>(r_a[j] - r_b[j]);

                    if (!dr2)
                        return 0.0;

                    return 1.0 / std::sqrt(dr2);
                };
            if (n_dim == 3) {
                return [](double *r_a, double *r_b) {
                    double dr2 = 0.0;
                    for (int j = 0; j < 3; ++j)
                        dr2 += sctl::pow<2>(r_a[j] - r_b[j]);

                    if (!dr2)
                        return 0.0;

                    return 1.0 / dr2;
                };
            }

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

    for (auto n_dim : {2, 3}) {
        params.n_dim = n_dim;
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

            SUBCASE((kernel_str + "_" + std::to_string(n_dim)).c_str()) {
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
                auto potential = get_pot_func(n_dim, kernel);

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

                pdmk_tree tree = pdmk_tree_create(test_comm, params, n_src, &r_src[0], &charges[0], &rnormal[0],
                                                  &dipstr[0], n_trg, &r_trg[0]);
                pdmk_tree_eval(tree, &pot_src[0], &grad_src[0], &hess_src[0], &pot_trg[0], &grad_trg[0], &hess_trg[0]);
                pdmk_tree_destroy(tree);

                // double tottimeinfo[20];
                // int use_dipole = 0;
                // double st = omp_get_wtime();
                // pdmk_(&nd, &n_dim, &params.eps, (int *)&params.kernel, &params.fparam, &params.use_periodic, &n_src,
                //       &r_src[0], &params.use_charge, &charges[0], &use_dipole, nullptr, nullptr, (int
                //       *)&params.pgh_src, &pot_src_fort[0], &grad_src_fort[0], &hess_src_fort[0], &n_trg, &r_trg[0],
                //       (int *)&params.pgh_trg, &pot_trg_fort[0], &grad_trg_fort[0], &hess_trg_fort[0], tottimeinfo);
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
}

template <typename Real>
inline pdmk_tree pdmk_tree_create(MPI_Comm comm, pdmk_params params, int n_src, const Real *r_src, const Real *charge,
                                  const Real *normal, const Real *dipole_str, int n_trg, const Real *r_trg) {
    sctl::Profile::reset();
    sctl::Profile::Enable(true);
    const sctl::Comm sctl_comm(comm);
    sctl::Profile::Scoped profile("pdmk_tree_create", &sctl_comm);

    sctl::Vector<Real> r_src_vec(n_src * params.n_dim, const_cast<Real *>(r_src), false);
    sctl::Vector<Real> r_trg_vec(n_trg * params.n_dim, const_cast<Real *>(r_trg), false);
    sctl::Vector<Real> charge_vec(n_src * params.n_mfm, const_cast<Real *>(charge), false);

    if (params.n_dim != 2 && params.n_dim != 3)
        throw std::runtime_error("Invalid dimension: " + std::to_string(params.n_dim));
    if (params.n_dim == 2) {
        return new pdmk_tree_impl(std::unique_ptr<dmk::DMKPtTree<Real, 2>>(
            new dmk::DMKPtTree<Real, 2>(sctl_comm, params, r_src_vec, r_trg_vec, charge_vec)));
    } else
        return new pdmk_tree_impl(std::unique_ptr<dmk::DMKPtTree<Real, 3>>(
            new dmk::DMKPtTree<Real, 3>(sctl_comm, params, r_src_vec, r_trg_vec, charge_vec)));
}

template <typename Real>
inline void pdmk_tree_eval(pdmk_tree tree, Real *pot_src, Real *grad_src, Real *hess_src, Real *pot_trg, Real *grad_trg,
                           Real *hess_trg) {
    std::visit(
        [&](auto &t) {
            using TreeType = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 2>>> ||
                          std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 3>>>) {
                const auto &comm = (*static_cast<TreeType *>(tree))->GetComm();
                sctl::Profile::Tic("pdmk_tree_eval", &comm);
                t->upward_pass();
                t->downward_pass();
                sctl::Profile::Toc();

#ifdef DMK_INSTRUMENT
                sctl::Profile::Tic("pdmk_tree_eval_sync_barrier", &comm);
                comm.Barrier();
                sctl::Profile::Toc();
#endif

                sctl::Profile::Tic("pdmk_tree_eval_sync", &comm);
                sctl::Vector<Real> res;
                t->GetParticleData(res, "pdmk_pot_src");
                sctl::Vector<Real>(res.Dim(), pot_src, false) = res;
                t->GetParticleData(res, "pdmk_pot_trg");
                sctl::Vector<Real>(res.Dim(), pot_trg, false) = res;
                sctl::Profile::Toc();
            }
        },
        *static_cast<pdmk_tree_impl *>(tree));
}

} // namespace dmk

extern "C" {

void pdmk_print_profile_data(MPI_Comm comm) {
    sctl::Comm sctl_comm(comm);
    sctl::Profile::print(
        &sctl_comm,
        {"t_avg", "t_max", "t_min", "f_avg", "f_max", "f_min", "f_total", "f/s_total", "custom1", "custom2", "custom3", "custom4", "custom5"},
        {"%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.3g", "%.3g", "%.3g", "%.3g", "%.3g"});
}

pdmk_tree pdmk_tree_createf(MPI_Comm comm, pdmk_params params, int n_src, const float *r_src, const float *charge,
                            const float *normal, const float *dipole_str, int n_trg, const float *r_trg) {
    return dmk::pdmk_tree_create(comm, params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg);
}

pdmk_tree pdmk_tree_create(MPI_Comm comm, pdmk_params params, int n_src, const double *r_src, const double *charge,
                           const double *normal, const double *dipole_str, int n_trg, const double *r_trg) {
    return dmk::pdmk_tree_create(comm, params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg);
}

void pdmk_tree_destroy(pdmk_tree tree) {
    if (tree)
        delete static_cast<pdmk_tree_impl *>(tree);
}

void pdmk_tree_evalf(pdmk_tree tree, float *pot_src, float *grad_src, float *hess_src, float *pot_trg, float *grad_trg,
                     float *hess_trg) {
    dmk::pdmk_tree_eval(tree, pot_src, grad_src, hess_src, pot_trg, grad_trg, hess_trg);
}

void pdmk_tree_eval(pdmk_tree tree, double *pot_src, double *grad_src, double *hess_src, double *pot_trg,
                    double *grad_trg, double *hess_trg) {
    dmk::pdmk_tree_eval(tree, pot_src, grad_src, hess_src, pot_trg, grad_trg, hess_trg);
}

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
