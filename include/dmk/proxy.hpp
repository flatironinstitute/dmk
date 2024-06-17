#ifndef PROXY_HPP
#define PROXY_HPP

#include <complex>
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
}

#endif
