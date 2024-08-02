#ifndef DMK_PROXY_HPP
#define DMK_PROXY_HPP

#include <dmk/fortran.h>

#include <complex>
#include <dmk/types.hpp>
#include <sctl.hpp>

namespace dmk::proxy {
template <typename T>
void calc_planewave_coeff_matrices(double boxsize, T hpw, int n_pw, int n_order,
                                   sctl::Vector<std::complex<T>> &prox2pw_vec,
                                   sctl::Vector<std::complex<T>> &pw2poly_vec);

template <typename T, int DIM>
void charge2proxycharge(const ndview<const T, 2> &r_src, const ndview<const T, 2> &charge,
                        const ndview<const T, 1> &center, T scale_factor, const ndview<T, DIM + 1> &coeffs);

template <typename T>
void proxycharge2pw(int n_dim, int n_charge_dim, int n_order, int n_pw, const T *proxy_coeffs,
                    const std::complex<T> *poly2pw, std::complex<T> *pw_expansion);

template <typename T, int DIM>
void proxycharge2pw(const ndview<const T, DIM + 1> &proxy_coeffs, const ndview<const std::complex<T>, 2> &poly2pw,
                    const ndview<std::complex<T>, DIM + 1> &pw_expansion);

template <typename T, int DIM>
void eval_targets(const ndview<const T, DIM + 1> &coefs, const ndview<const T, 2> &r_trg, const ndview<const T, 1> &cen,
                  T sc, const ndview<T, 2> &pot);

} // namespace dmk::proxy
#endif
