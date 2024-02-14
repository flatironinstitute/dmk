#ifndef PROXY_HPP
#define PROXY_HPP

#include <vector>

namespace dmk::proxy {
template <typename T>
void charge2proxycharge(int n_dim, int n_charge_dim, int order, const std::vector<T> &r_src,
                        const std::vector<T> &charge, T *center, T scale_factor, std::vector<T> &coeffs);
}

#endif
