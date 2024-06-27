#ifndef FOURIER_DATA_HPP
#define FOURIER_DATA_HPP

#include <complex>
#include <dmk.h>
#include <sctl.hpp>
#include <vector>

namespace dmk {
struct ProlateFuncs;

template <typename T>
struct FourierData {
    FourierData(dmk_ikernel kernel_, int n_dim_, int n_digits_, int n_pw_max, T fparam_, T beta_,
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
    int n_pw;
    const int n_fourier;
    const T fparam;
    std::vector<T> dkernelft;
    std::vector<T> hpw;
    std::vector<T> ws;
    std::vector<T> rl;

    // Local chebyshev polynomial coefficients for yukawa potential
    std::vector<double> coeffs1;
    std::vector<double> coeffs2;
    std::vector<int> ncoeffs1;
    std::vector<int> ncoeffs2;

    const std::vector<double> &boxsize;
    void calc_planewave_coeff_matrices(int i_level, int n_order, sctl::Vector<std::complex<T>> &prox2pw,
                                       sctl::Vector<std::complex<T>> &pw2poly) const;
    void calc_planewave_translation_matrix(int dim, int i_level, T xmin,
                                           sctl::Vector<std::complex<T>> &shift_vec) const;
};

template <int DIM, typename T>
void calc_planewave_translation_matrix(int nmax, T xmin, int npw, const sctl::Vector<T> &ts,
                                       sctl::Vector<std::complex<T>> &shift_vec);

} // namespace dmk

#endif
