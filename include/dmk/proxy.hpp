#ifndef DMK_PROXY_HPP
#define DMK_PROXY_HPP

#include <dmk/fortran.h>
#include <dmk/types.hpp>

#include <complex>
#include <dmk/types.hpp>
#include <sctl.hpp>

namespace dmk::proxy {
template <typename T>
void calc_planewave_coeff_matrices(double boxsize, T hpw, int n_pw, int n_order,
                                   sctl::Vector<std::complex<T>> &prox2pw_vec,
                                   sctl::Vector<std::complex<T>> &pw2poly_vec);

template <typename T>
void charge2proxycharge(int n_dim, int n_charge_dim, int order, int n_src, const T *r_src, const T *charge,
                        const T *center, T scale_factor, T *coeffs);

template <typename T>
void proxycharge2pw(int n_dim, int n_charge_dim, int n_order, int n_pw, const T *proxy_coeffs,
                    const std::complex<T> *poly2pw, std::complex<T> *pw_expansion);

template <typename T, int DIM>
void proxycharge2pw(dmk::ndview<const T, DIM + 1> &proxy_coeffs,
                    dmk::ndview<const std::complex<T>, 2> &poly2pw, dmk::ndview<std::complex<T>, DIM + 1> &pw_expansion);

template <typename T, int DIM>
void eval_targets(const ndview<const T, DIM + 1> &coefs, const ndview<const T, 2> &r_trg, const ndview<const T, 1> &cen,
                  T sc, const ndview<T, 2> &pot);

} // namespace dmk::proxy
#endif
