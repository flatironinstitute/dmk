#ifndef PLANEWAVE_HPP
#define PLANEWAVE_HPP

#include <complex>
#include <sctl.hpp>

namespace dmk {
template <typename Real>
void planewave_to_proxy_potential(int dim, int n_charge_dim, int n_order, int n_pw,
                                  const std::complex<Real> *pw_expansion_, const std::complex<Real> *pw_to_coefs_mat_,
                                  Real *proxy_coeffs_);

template <typename T>
void calc_planewave_coeff_matrices(double boxsize, T hpw, int n_pw, int n_order,
                                   sctl::Vector<std::complex<T>> &prox2pw_vec,
                                   sctl::Vector<std::complex<T>> &pw2poly_vec);

template <int DIM, typename T>
void calc_planewave_translation_matrix(int nmax, T xmin, int npw, const sctl::Vector<T> &ts,
                                       sctl::Vector<std::complex<T>> &shift_vec);
} // namespace dmk

#endif
