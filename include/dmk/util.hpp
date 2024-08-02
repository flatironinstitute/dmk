#ifndef UTIL_HPP
#define UTIL_HPP

#include <dmk/types.hpp>
#include <type_traits>

namespace dmk::util {
template <class...>
constexpr std::false_type always_false{};

template <typename Real>
void mesh_nd(int dim, Real *in, int size, Real *out);

template <typename Real>
void mesh_nd(int dim, const ndview<const Real, 1> &in, const ndview<Real, 2> &out);

template <typename Real>
void mk_tensor_product_fourier_transform(int dim, int npw, int nfourier, Real *fhat, int nexp, Real *pswfft);

template <typename Real>
void mk_tensor_product_fourier_transform(int dim, int npw, const ndview<const Real, 1> &fhat,
                                         const ndview<Real, 1> &pswfft);

template <typename Real>
void init_test_data(int n_dim, int nd, int n_src, bool uniform, std::vector<Real> &r_src, std::vector<Real> &rnormal,
                    std::vector<Real> &charges, std::vector<Real> &dipstr, long seed);
} // namespace dmk::util

#endif
