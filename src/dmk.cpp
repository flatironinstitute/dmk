#include <dmk.h>
#include <dmk/logger.h>
#include <sctl.hpp>

namespace DMK {

template <typename T, int DIM>
void zero_potentials(dmk_pgh level, int ns, int nd, T *pot, T *grad, T *hess) {
    if (level >= DMK_POTENTIAL)
        memset(pot, 0, ns * nd * sizeof(T));
    if (level >= DMK_POTENTIAL_GRAD)
        memset(grad, 0, DIM * ns * nd * sizeof(T));
    if (level >= DMK_POTENTIAL_GRAD_HESSIAN)
        memset(hess, 0, (DIM * (DIM + 1)) / 2 * ns * nd * sizeof(T));
}

template <typename T>
T procl180_rescale(T eps) {
    T cs[] = {
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
          int n_trg, const T *r_trg, T *pot, T *grad, T *hess, T *pottarg, T *gradtarg, T *hesstarg) {
    auto &logger = dmk::get_logger(params.log_level);
    logger->debug("PDMK called");

    sctl::PtTree<T, DIM> tree(sctl::Comm::World());
    sctl::Vector<T> r_src_vec(n_src, const_cast<T *>(r_src), false);
    sctl::Vector<T> r_trg_vec(n_trg, const_cast<T *>(r_trg), false);
    sctl::Vector<T> charge_vec(n_src * params.n_mfm, const_cast<T *>(charge), false);

    logger->debug("Building tree");
    tree.AddParticles("pdmk_src", r_src_vec);
    tree.AddParticleData("pdmk_charge", "pdmk_src", charge_vec);
    tree.AddParticles("pdmk_trg", r_trg_vec);
    tree.UpdateRefinement(r_src_vec, params.n_per_leaf, true, params.use_periodic); // balance21 = true
    logger->debug("Tree build completed");
    logger->debug("Zeroing source and target potentials");
    zero_potentials<T, DIM>(params.pgh, n_src, params.n_mfm, pot, grad, hess);
    zero_potentials<T, DIM>(params.pgh_target, n_trg, params.n_mfm, pot, grad, hess);
    logger->debug("Zeroing complete");

    T beta = procl180_rescale(params.eps);
    logger->debug("prolate parameter value = {}", beta);


}

} // namespace DMK
extern "C" {
void pdmkf(pdmk_params params, int n_src, const float *r_src, const float *charge, const float *normal,
           const float *dipole_str, int n_trg, const float *r_trg, float *pot, float *grad, float *hess, float *pottarg,
           float *gradtarg, float *hesstarg) {
    if (params.n_dim == 2)
        return DMK::pdmk<float, 2>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot, grad, hess,
                                   pottarg, gradtarg, hesstarg);
    if (params.n_dim == 3)
        return DMK::pdmk<float, 3>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot, grad, hess,
                                   pottarg, gradtarg, hesstarg);
}

void pdmk(pdmk_params params, int n_src, const double *r_src, const double *charge, const double *normal,
          const double *dipole_str, int n_trg, const double *r_trg, double *pot, double *grad, double *hess,
          double *pottarg, double *gradtarg, double *hesstarg) {
    if (params.n_dim == 2)
        return DMK::pdmk<double, 2>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot, grad, hess,
                                    pottarg, gradtarg, hesstarg);
    if (params.n_dim == 3)
        return DMK::pdmk<double, 3>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot, grad, hess,
                                    pottarg, gradtarg, hesstarg);
}
}
