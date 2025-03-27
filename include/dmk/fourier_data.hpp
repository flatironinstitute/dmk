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

    const ndview<const T, 1> cheb_coeffs(int i_level) const {
        if (coeffs1_.Dim())
            return ndview<const T, 1>(&coeffs1_[i_level * n_coeffs_max], ncoeffs1_[i_level]);
        else
            return ndview<const T, 1>(nullptr, 0);
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
void get_difference_kernel_ft(dmk_ikernel kernel, const double *rpars, Real beta, int ndigits, Real boxsize,
                              Prolate0Fun &pf, sctl::Vector<Real> &windowed_kernel);
} // namespace dmk

#endif
