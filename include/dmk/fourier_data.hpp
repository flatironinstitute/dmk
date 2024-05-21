#ifndef FOURIER_DATA_HPP
#define FOURIER_DATA_HPP

#include "../dmk.h"
#include <complex>
#include <vector>

namespace dmk {
struct ProlateFuncs;

template <typename T>
struct FourierData {
    FourierData<T>(dmk_ikernel kernel_, int n_dim_, int n_digits_, int n_pw_max, T fparam_, T beta_,
                   const std::vector<double> &boxsize_);
    void yukawa_windowed_kernel_Fourier_transform(ProlateFuncs &pf);
    void update_windowed_kernel_fourier_transform(ProlateFuncs &pf);
    void yukawa_difference_kernel_fourier_transform(int i_level, ProlateFuncs &pf);
    void update_difference_kernel(int i_level, ProlateFuncs &pf);
    void update_difference_kernels(ProlateFuncs &pf);
    void update_local_coeffs_yukawa(T eps, ProlateFuncs &pf);
    void update_local_coeffs(T eps, ProlateFuncs &pf);

    const dmk_ikernel kernel;
    const T beta;
    const int n_dim;
    const int n_digits;
    const int n_levels;
    const int n_fourier_max;
    const int n_pw_max;
    const T fparam;
    std::vector<T> dkernelft;
    std::vector<int> npw;
    std::vector<int> nfourier;
    std::vector<T> hpw;
    std::vector<T> ws;
    std::vector<T> rl;

    // Local chebyshev polynomial coefficients for yukawa potential
    std::vector<double> coeffs1;
    std::vector<double> coeffs2;
    std::vector<int> ncoeffs1;
    std::vector<int> ncoeffs2;

    const std::vector<double> &boxsize;
    void calc_planewave_coeff_matrices(int i_level, int n_order, std::complex<T> *prox2pw,
                                       std::complex<T> *pw2poly) const;
};

} // namespace dmk

#endif
