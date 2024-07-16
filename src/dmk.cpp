#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/logger.h>
#include <dmk/prolate_funcs.hpp>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/tree.hpp>
#include <sctl.hpp>

#include <mpi.h>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <doctest/extensions/doctest_mpi.h>

namespace dmk {

std::pair<int, int> get_pwmax_and_poly_order(int dim, int ndigits, dmk_ikernel kernel) {
    // clang-format off
    if (kernel == DMK_SQRT_LAPLACE && dim == 3) {
        if (ndigits <= 3) return {13, 9};
        if (ndigits <= 6) return {27, 18};
        if (ndigits <= 9) return {39, 28};
        if (ndigits <= 12) return {55, 38};
    }
    if (ndigits <= 3) return {13, 9};
    if (ndigits <= 6) return {25, 18};
    if (ndigits <= 9) return {39, 28};
    if (ndigits <= 12) return {53, 38};
    // clang-format on
    throw std::runtime_error("Requested precision too high");
}

template <typename T>
T procl180_rescale(T eps) {
    constexpr T cs[] = {
        .43368E-16, .10048E+01, .17298E+01, .22271E+01, .26382E+01, .30035E+01, .33409E+01, .36598E+01, .39658E+01,
        .42621E+01, .45513E+01, .48347E+01, .51136E+01, .53887E+01, .56606E+01, .59299E+01, .61968E+01, .64616E+01,
        .67247E+01, .69862E+01, .72462E+01, .75049E+01, .77625E+01, .80189E+01, .82744E+01, .85289E+01, .87826E+01,
        .90355E+01, .92877E+01, .95392E+01, .97900E+01, .10040E+02, .10290E+02, .10539E+02, .10788E+02, .11036E+02,
        .11284E+02, .11531E+02, .11778E+02, .12024E+02, .12270E+02, .12516E+02, .12762E+02, .13007E+02, .13251E+02,
        .13496E+02, .13740E+02, .13984E+02, .14228E+02, .14471E+02, .14714E+02, .14957E+02, .15200E+02, .15443E+02,
        .15685E+02, .15927E+02, .16169E+02, .16411E+02, .16652E+02, .16894E+02, .17135E+02, .17376E+02, .17617E+02,
        .17858E+02, .18098E+02, .18339E+02, .18579E+02, .18819E+02, .19059E+02, .19299E+02, .19539E+02, .19778E+02,
        .20018E+02, .20257E+02, .20496E+02, .20736E+02, .20975E+02, .21214E+02, .21452E+02, .21691E+02, .21930E+02,
        .22168E+02, .22407E+02, .22645E+02, .22884E+02, .23122E+02, .23360E+02, .23598E+02, .23836E+02, .24074E+02,
        .24311E+02, .24549E+02, .24787E+02, .25024E+02, .25262E+02, .25499E+02, .25737E+02, .25974E+02, .26211E+02,
        .26448E+02, .26685E+02, .26922E+02, .27159E+02, .27396E+02, .27633E+02, .27870E+02, .28106E+02, .28343E+02,
        .28580E+02, .28816E+02, .29053E+02, .29289E+02, .29526E+02, .29762E+02, .29998E+02, .30234E+02, .30471E+02,
        .30707E+02, .30943E+02, .31179E+02, .31415E+02, .31651E+02, .31887E+02, .32123E+02, .32358E+02, .32594E+02,
        .32830E+02, .33066E+02, .33301E+02, .33537E+02, .33773E+02, .34008E+02, .34244E+02, .34479E+02, .34714E+02,
        .34950E+02, .35185E+02, .35421E+02, .35656E+02, .35891E+02, .36126E+02, .36362E+02, .36597E+02, .36832E+02,
        .37067E+02, .37302E+02, .37537E+02, .37772E+02, .38007E+02, .38242E+02, .38477E+02, .38712E+02, .38947E+02,
        .39181E+02, .39416E+02, .39651E+02, .39886E+02, .40120E+02, .40355E+02, .40590E+02, .40824E+02, .41059E+02,
        .41294E+02, .41528E+02, .41763E+02, .41997E+02, .42232E+02, .42466E+02, .42700E+02, .42935E+02, .43169E+02,
        .43404E+02, .43638E+02, .43872E+02, .44107E+02, .44341E+02, .44575E+02, .44809E+02, .45044E+02, .45278E+02};

    int scale;
    if (eps >= 1.0E-3)
        scale = 8;
    else if (eps >= 1E-6)
        scale = 20;
    else if (eps >= 1E-9)
        scale = 25;
    else if (eps >= 1E-12)
        scale = 25;

    double d = -std::log10(scale * eps);
    int i = d * 10 + 0.1 - 1;
    assert(i >= 0);
    assert(i < sizeof(cs) / sizeof(T));
    return cs[i];
}

template <typename T, int DIM>
void pdmk(const pdmk_params &params, int n_src, const T *r_src, const T *charge, const T *normal, const T *dipole_str,
          int n_trg, const T *r_trg, T *pot, T *grad, T *hess) {
    auto &logger = dmk::get_logger(params.log_level);
    auto &rank_logger = dmk::get_rank_logger(params.log_level);
    logger->info("PDMK called");
    auto st = omp_get_wtime();

    const int ndigits = std::round(log10(1.0 / params.eps) - 0.1);
    const auto [n_pw_max, n_order] = get_pwmax_and_poly_order(DIM, ndigits, params.kernel);

    // 0: Initialization
    dmk::DMKPtTree<T, DIM> tree(sctl::Comm::World(), params, n_order);
    sctl::Vector<T> r_src_vec(n_src * params.n_dim, const_cast<T *>(r_src), false);
    sctl::Vector<T> r_trg_vec(n_trg * params.n_dim, const_cast<T *>(r_trg), false);
    sctl::Vector<T> charge_vec(n_src * params.n_mfm, const_cast<T *>(charge), false);
    sctl::Vector<T> pot_vec(n_trg * params.n_mfm);
    pot_vec.SetZero();

    logger->debug("Building tree and sorting points");
    tree.AddParticles("pdmk_src", r_src_vec);
    tree.AddParticleData("pdmk_charge", "pdmk_src", charge_vec);
    tree.AddParticles("pdmk_trg", r_trg_vec);
    tree.AddParticleData("pdmk_pot", "pdmk_trg", pot_vec);
    tree.UpdateRefinement(r_src_vec, params.n_per_leaf, true, params.use_periodic); // balance21 = true
    logger->debug("Tree build completed");

    logger->debug("Generating tree traversal metadata");
    tree.generate_metadata();
    logger->debug("Done generating tree traversal metadata");

    rank_logger->trace("Local tree has {} levels {} boxes", tree.n_levels(), tree.n_boxes());

    double beta = procl180_rescale(params.eps);
    logger->debug("prolate parameter value = {}", beta);
    ProlateFuncs prolate_funcs(beta, 10000);
    logger->debug("Initialized prolate function data");

    // 1: Precomputation
    logger->debug("Generating p2c and c2p matrices of order {}", n_order);
    auto [c2p, p2c] = dmk::chebyshev::get_c2p_p2c_matrices<T>(DIM, n_order);
    logger->debug("Finished generating matrices");

    FourierData<T> fourier_data(params.kernel, DIM, ndigits, n_pw_max, params.fparam, beta, tree.boxsize,
                                prolate_funcs);
    logger->debug("Planewave params at root box: n: {}, stepsize: {}, weight: {}, radius: {}", fourier_data.n_pw,
                  fourier_data.hpw[0], fourier_data.ws[0], fourier_data.rl[0]);
    fourier_data.update_windowed_kernel_fourier_transform();
    logger->debug("Truncated fourier transform for kernel {} at root box generated", int(params.kernel));
    fourier_data.update_difference_kernels();
    logger->debug("Finished calculating difference kernels");
    fourier_data.update_local_coeffs(params.eps);
    logger->debug("Finished updating local potential expansion coefficients");

    // upward pass
    tree.upward_pass(c2p);
    tree.downward_pass(fourier_data, p2c);

    sctl::Vector<T> res;
    tree.GetParticleData(res, "pdmk_pot");
    sctl::Vector<T>(res.Dim(), pot, false) = res;

    auto dt = omp_get_wtime() - st;
    int N = n_trg;
    if (tree.GetComm().Rank() == 0)
        MPI_Reduce(MPI_IN_PLACE, &N, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    else
        MPI_Reduce(&N, &N, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    logger->info("PDMK finished in {:.2f} seconds ({:.2f} pts/s)", dt, N / dt);
}

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

    for (int i = 0; i < 3; ++i)
        r_src[i] = 0.0;
    for (int i = 3; i < 6; ++i)
        r_src[i] = 1 - std::numeric_limits<double>::epsilon();
    for (int i = 6; i < 9; ++i)
        r_src[i] = 0.05;
}

MPI_TEST_CASE("[DMK] pdmk 2d", 1) {}

MPI_TEST_CASE("[DMK] pdmk 3d", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 10000;
    constexpr int n_trg = n_src;
    constexpr int nd = 1;

    std::vector<double> r_src, pot_src, grad_src, hess_src, charges, rnormal, dipstr, pot_trg, r_trg, grad_trg,
        hess_trg;
    init_data(n_dim, 1, n_src, false, r_src, rnormal, charges, dipstr, 0);
    r_trg = r_src;
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
    params.pgh = DMK_POTENTIAL;
    params.kernel = DMK_YUKAWA;
    params.log_level = 0;
    params.fparam = 6.0;

    pdmk(params, n_src, r_src.data(), charges.data(), rnormal.data(), dipstr.data(), n_trg, r_trg.data(),
         pot_trg.data(), nullptr, nullptr);

    const int ifdipole = 0;
    const int iperiod = 0;
    const int pgh_src = 1;
    double tottimeinfo[20];

    double test_pot = 0.0;
    int test_targ = n_trg / 3;
    for (int i = 0; i < n_src; ++i) {
        double dr = 0.0;
        for (int j = 0; j < n_dim; ++j)
            dr += sctl::pow<2>(r_src[i * n_dim + j] - r_trg[test_targ * n_dim + j]);
        dr = std::sqrt(dr);
        if (!dr)
            continue;

        test_pot += charges[i] * exp(-params.fparam * dr) / dr;
    }
    std::cout << "pot[test_targ] = " << pot_trg[test_targ] << std::endl;
    std::cout << "test_pot_direct = " << test_pot << std::endl;
    std::cout << "rel_err[test_targ] = " << std::abs(1.0 - test_pot / pot_trg[test_targ]) << std::endl;

    int zero = 0;
    pdmk_(&params.n_mfm, &params.n_dim, &params.eps, (int *)&params.kernel, &params.fparam, &iperiod, &n_src,
          r_src.data(), &params.use_charge, charges.data(), &ifdipole, nullptr, nullptr, &pgh_src, pot_src.data(),
          grad_src.data(), hess_src.data(), &zero, nullptr, &zero, nullptr, nullptr, nullptr, tottimeinfo);
    std::cout << "pot_ref[test_targ] = " << pot_src[test_targ] << std::endl;
    std::cout << "rel_err_ref[test_targ] = " << std::abs(1.0 - test_pot / pot_src[test_targ]) << std::endl;
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
          const double *dipole_str, int n_trg, const double *r_trg, double *pot, double *grad, double *hess) {
    if (params.n_dim == 2)
        return dmk::pdmk<double, 2>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot, grad, hess);
    if (params.n_dim == 3)
        return dmk::pdmk<double, 3>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot, grad, hess);
}
}
