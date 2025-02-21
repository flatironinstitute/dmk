#ifndef FOURIER_DATA_HPP
#define FOURIER_DATA_HPP

#include <complex>
#include <dmk.h>
#include <dmk/prolate_funcs.hpp>
#include <sctl.hpp>
#include <vector>

namespace dmk {
struct ProlateFuncs;

template <typename T>
struct FourierData {
    FourierData() = default;
    FourierData(dmk_ikernel kernel_, int n_dim_, T eps, int n_digits_, int n_pw_max, T fparam_,
                const std::vector<T> &boxsize_);

    void yukawa_windowed_kernel_Fourier_transform();
    void update_windowed_kernel_fourier_transform();
    void yukawa_difference_kernel_fourier_transform(int i_level);
    T yukawa_windowed_kernel_value_at_zero(int i_level);
    void update_difference_kernel(int i_level);
    void update_difference_kernels();
    void update_local_coeffs_yukawa(T eps);
    void update_local_coeffs(T eps);

    dmk_ikernel kernel;
    int n_dim;
    int n_digits;
    int n_levels;
    int n_pw;
    int n_fourier;
    T fparam;
    std::vector<T> dkernelft;
    std::vector<T> hpw;
    std::vector<T> ws;
    std::vector<T> rl;

    // Local chebyshev polynomial coefficients for yukawa potential
    std::vector<T> coeffs1;
    std::vector<T> coeffs2;
    std::vector<int> ncoeffs1;
    std::vector<int> ncoeffs2;
    int n_coeffs_max = 100;

    T beta;
    ProlateFuncs prolate_funcs;

    std::vector<T> boxsize;
    void calc_planewave_coeff_matrices(int i_level, int n_order, sctl::Vector<std::complex<T>> &prox2pw,
                                       sctl::Vector<std::complex<T>> &pw2poly) const;
    void calc_planewave_translation_matrix(int dim, int i_level, T xmin,
                                           sctl::Vector<std::complex<T>> &shift_vec) const;
};

template <int DIM, typename T>
void calc_planewave_translation_matrix(int nmax, T xmin, int npw, const sctl::Vector<T> &ts,
                                       sctl::Vector<std::complex<T>> &shift_vec);

template <typename Real, int DIM>
void get_windowed_kernel_ft(dmk_ikernel kernel, const double *rpars, Real beta, int ndigits, Real boxsize,
                            ProlateFuncs &pf, sctl::Vector<Real> &windowed_kernel);
template <typename Real, int DIM>
void get_difference_kernel_ft(dmk_ikernel kernel, const double *rpars, Real beta, int ndigits, Real boxsize,
                              ProlateFuncs &pf, sctl::Vector<Real> &windowed_kernel);
} // namespace dmk

#endif
