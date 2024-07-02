#ifndef DMK_TYPES_HPP
#define DMK_TYPES_HPP

#include <mdspan.hpp>

namespace dmk {
template <typename T, int DIM>
using ndview = std::experimental::mdspan<T, std::experimental::dextents<size_t, DIM>, std::experimental::layout_left>;
} // namespace dmk

#endif