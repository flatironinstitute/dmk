#ifndef PROXY_HPP
#define PROXY_HPP

#include <vector>

namespace dmk::proxy {
template <typename T>
void charge2proxycharge(int n_dim, int n_charge_dim, int order, int n_src, const T *r_src, const T *charge,
                        const T *center, T scale_factor, T *coeffs);
}

#endif
