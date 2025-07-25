#ifndef DMK_TYPES_HPP
#define DMK_TYPES_HPP

#include <nda/nda.hpp>

namespace dmk {
template <typename T, int DIM>
using ndview = nda::basic_array_view<T, DIM, nda::F_layout>;
template <typename T>
using matrixview = nda::matrix_view<T, nda::F_layout>;
} // namespace dmk

#endif
