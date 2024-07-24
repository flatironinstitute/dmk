#ifndef UTIL_HPP
#define UTIL_HPP

#include <type_traits>
#include <dmk/types.hpp>

namespace dmk::util {
template <class...>
constexpr std::false_type always_false{};

template <typename Real>
void mesh_nd(int dim, Real *in, int size, Real *out);

template <typename Real>
void mesh_nd(int dim, dmk::ndview<const Real, 1> &in, dmk::ndview<Real, 2> &out);

template <typename Real>
void mk_tensor_product_fourier_transform(int dim, int npw, int nfourier, Real *fhat, int nexp, Real *pswfft);

template <typename Real>
void mk_tensor_product_fourier_transform(int dim, int npw, dmk::ndview<const Real, 1> &fhat, dmk::ndview<Real, 1> &pswfft);
} // namespace dmk::util

#endif
