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
void mesh_nd(int dim, dmk::ndview<const Real, 1> &x, dmk::ndview<Real, 2> &out);
} // namespace dmk::util

#endif
