#ifndef DMK_TYPES_HPP
#define DMK_TYPES_HPP

#include <nda/nda.hpp>
#include <functional>

namespace dmk {
template <typename T, int DIM>
using ndview = nda::basic_array_view<T, DIM, nda::F_layout>;
template <typename T>
using matrixview = nda::matrix_view<T, nda::F_layout>;

template <typename T>
using ndamatrix = nda::matrix<T, nda::F_layout, nda::heap<>>;

template <typename T>
using ndavector = nda::vector<T, nda::heap<>>;

template <typename T>
using direct_evaluator_func = std::function<void(int nd, T rsc, T cen, T d2max, T thresh2, int n_src, const T *r_src,
                                                 const T *charge, int n_trg, const T *r_trg, T *pot)>;
} // namespace dmk

#endif
