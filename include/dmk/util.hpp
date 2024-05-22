#ifndef UTIL_HPP
#define UTIL_HPP

#include <type_traits>

namespace dmk::util {
template <class...>
constexpr std::false_type always_false{};
}

#endif
