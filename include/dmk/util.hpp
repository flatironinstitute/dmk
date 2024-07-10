#ifndef UTIL_HPP
#define UTIL_HPP

#include <type_traits>

namespace dmk::util {
template <class...>
constexpr std::false_type always_false{};

template <typename Real>
void meshnd(int dim, Real *in, int size, Real *out);
} // namespace dmk::util

#endif
