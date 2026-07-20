#ifndef FOURIER_DATA_HPP
#define FOURIER_DATA_HPP

#include <cmath>
#include <complex>
#include <dmk.h>
#include <dmk/prolate0_fun.hpp>
#include <dmk/types.hpp>
#include <sctl.hpp>
#include <vector>

namespace dmk {

template <typename T>
struct FourierData {
    FourierData() = default;
    FourierData(dmk_ikernel kernel_, int n_dim_, T eps, int n_pw_win, int n_pw_diff, T fparam_, double beta,
                const sctl::Vector<T> &boxsize_);

    T yukawa_windowed_kernel_value_at_zero(int i_level);
    void calc_planewave_coeff_matrices(int i_level, int n_order, int n_pw, sctl::Vector<std::complex<T>> &prox2pw,
                                       sctl::Vector<std::complex<T>> &pw2poly) const;

    // Per-level Yukawa short-range residual coefficients, generated on demand
    // (captured into the residual evaluator at construction; not stored here).
    // reg_poly is the monomial remainder polynomial (3D: the full fit Q; 2D: PB).
    // log_poly is the 2D log-coefficient polynomial PA and is empty in 3D.
    struct LocalCorrectionCoeffs {
        std::vector<double> log_poly;
        std::vector<double> reg_poly;
    };
    LocalCorrectionCoeffs local_correction_coeffs(int i_level, int n_digits);

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
    int n_levels_{0};
    T fparam_{0};
    T beta_{0};

    struct kernel_params windowed_kernel_;
    sctl::Vector<struct kernel_params> difference_kernels_;
    sctl::Vector<T> box_sizes_;
};

template <int DIM, typename T>
void calc_planewave_translation_matrix(int nmax, T xmin, int npw, const sctl::Vector<T> &ts,
                                       sctl::Vector<std::complex<T>> &shift_vec);

template <typename Real, int DIM>
void get_windowed_kernel_ft(dmk_ikernel kernel, const double *rpars, Real beta, int npw, Real boxsize, Prolate0Fun &pf,
                            sctl::Vector<Real> &windowed_kernel);
template <typename Real, int DIM>
void get_difference_kernel_ft(bool init, dmk_ikernel kernel, const double *rpars, Real beta, int npw, Real boxsize,
                              Prolate0Fun &pf, sctl::Vector<Real> &windowed_kernel);

// Periodic windowed kernel FT for the root box, sampled on the reciprocal grid kappa = sqrt(i)*dk.
// Scalar kernels (Laplace, Yukawa, Sqrt-Laplace), 3D.
template <typename Real, int DIM>
void get_periodic_windowed_kernel_ft(dmk_ikernel kernel, const double *rpars, Real beta, int n_pw_periodic,
                                     Real boxsize, Real sigma1, Prolate0Fun &pf, sctl::Vector<Real> &kernel_ft);

// Real-space value at r=0 of the windowed log kernel at the given box scale (the log-kernel
// self-interaction constant). Used by the tree's self correction and by ESP.
template <typename Real>
Real calc_log_windowed_kernel_value_at_zero(int dim, const Prolate0Fun &pf, Real beta, Real boxsize);

} // namespace dmk

#endif
