#ifndef FOURIER_DATA_HPP
#define FOURIER_DATA_HPP

#include <complex>
#include <dmk.h>
#include <dmk/prolate0_fun.hpp>
#include <dmk/types.hpp>
#include <sctl.hpp>

namespace dmk {

template <typename T>
struct FourierData {
    FourierData() = default;
    FourierData(dmk_ikernel kernel_, int n_dim_, T eps, int n_digits_, int n_pw_max, T fparam_,
                const sctl::Vector<T> &boxsize_);

    T yukawa_windowed_kernel_value_at_zero(int i_level);
    void update_local_coeffs(T eps);
    void calc_planewave_coeff_matrices(int i_level, int n_order, sctl::Vector<std::complex<T>> &prox2pw,
                                       sctl::Vector<std::complex<T>> &pw2poly) const;

    ndview<T, 1> cheb_coeffs(int i_level) {
        if (coeffs1_.Dim())
            return ndview<T, 1>({ncoeffs1_[i_level]}, &coeffs1_[i_level * n_coeffs_max]);
        else
            return ndview<T, 1>({0}, nullptr);
    };

    int n_pw() const { return n_pw_; };
    T beta() const { return beta_; }

    struct kernel_params {
        T hpw{0};
        T ws{0};
        T rl{0};
    };

    const struct kernel_params &windowed_kernel() const { return windowed_kernel_; }
    const struct kernel_params &difference_kernel(int i_level) const { return difference_kernels_[i_level]; }

    Prolate0Fun prolate0_fun;

  private:
    dmk_ikernel kernel_;
    int n_dim_{0};
    int n_digits_{0};
    int n_levels_{0};
    int n_pw_{0};
    T fparam_{0};
    T beta_{0};

    struct kernel_params windowed_kernel_;
    sctl::Vector<struct kernel_params> difference_kernels_;
    sctl::Vector<T> box_sizes_;

    // Local chebyshev polynomial coefficients for yukawa potential
    sctl::Vector<T> coeffs1_;
    sctl::Vector<T> coeffs2_;
    sctl::Vector<int> ncoeffs1_;
    sctl::Vector<int> ncoeffs2_;
    static constexpr int n_coeffs_max = 100;

    void update_local_coeffs_yukawa(T eps);
    void update_local_coeffs_laplace(T eps);
};

template <int DIM, typename T>
void calc_planewave_translation_matrix(int nmax, T xmin, int npw, const sctl::Vector<T> &ts,
                                       sctl::Vector<std::complex<T>> &shift_vec);

template <typename Real, int DIM>
void get_windowed_kernel_ft(dmk_ikernel kernel, const double *rpars, Real beta, int ndigits, Real boxsize,
                            Prolate0Fun &pf, sctl::Vector<Real> &windowed_kernel);
template <typename Real, int DIM>
void get_difference_kernel_ft(bool init, dmk_ikernel kernel, const double *rpars, Real beta, int ndigits, Real boxsize,
                              Prolate0Fun &pf, sctl::Vector<Real> &windowed_kernel);

template <typename T>
inline T procl180_rescale(T eps) {
    constexpr float cs[] = {
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
    assert(i < sizeof(cs) / sizeof(float));
    return cs[i];
}

} // namespace dmk

#endif
