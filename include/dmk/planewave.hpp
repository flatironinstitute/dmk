#ifndef PLANEWAVE_HPP
#define PLANEWAVE_HPP

#include <complex>
#include <dmk/types.hpp>
#include <sctl.hpp>

namespace dmk {
template <typename Real, int DIM>
void planewave_to_proxy_potential(const ndview<std::complex<Real>, DIM + 1> &pw_expansion,
                                  const ndview<std::complex<Real>, 2> &pw_to_coefs_mat,
                                  ndview<Real, DIM + 1> proxy_coeffs, sctl::Vector<Real> &workspace);

template <typename T>
void calc_planewave_coeff_matrices(T boxsize, T hpw, int n_pw, int n_order, sctl::Vector<std::complex<T>> &prox2pw_vec,
                                   sctl::Vector<std::complex<T>> &pw2poly_vec);

template <int DIM, typename T>
void calc_planewave_translation_matrix(int nmax, T xmin, int npw, T hpw, sctl::Vector<std::complex<T>> &shift_vec);
} // namespace dmk

#endif
