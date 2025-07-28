#ifndef DMK_TYPES_HPP
#define DMK_TYPES_HPP

#include <nda/nda.hpp>

namespace dmk {
template <typename T, int DIM>
using ndview = nda::basic_array_view<T, DIM, nda::F_layout>;
template <typename T>
using matrixview = nda::matrix_view<T, nda::F_layout>;

template <typename T>
using ndamatrix = nda::matrix<T, nda::F_layout, nda::heap<>>;

template <typename T>
using ndavector = nda::vector<T, nda::heap<>>;
} // namespace dmk

#endif
