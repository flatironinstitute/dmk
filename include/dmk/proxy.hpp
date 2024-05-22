#ifndef PROXY_HPP
#define PROXY_HPP

#include <complex>
#include <vector>

namespace dmk::proxy {
template <typename T>
void charge2proxycharge(int n_dim, int n_charge_dim, int order, int n_src, const T *r_src, const T *charge,
                        const T *center, T scale_factor, T *coeffs);

template <typename T>
void proxycharge2pw(int n_dim, int n_charge_dim, int n_order, int n_pw, T *proxy_coeffs, const std::complex<T> *poly2pw,
                    std::complex<T> *pw_expansion);
}

#endif
