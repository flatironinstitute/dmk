#ifndef DMK_TYPES_HPP
#define DMK_TYPES_HPP

#include <dmk.h>

#include <nda/nda.hpp>

namespace dmk {
// Compile-time range of proxy/Chebyshev expansion orders we instantiate
// specializations for: get_polynomial_calculator (chebychev.hpp), dispatch_order
// (proxy.cpp), get_opt_dot (util.hpp), and the stack scratch in
// charge2proxycharge. The per-level order is n_order = ceil(1.43*beta - 3.26),
// which can be as small as 2 at low beta (e.g. in beta sweeps). The lower bound
// is 2 because calc_polynomial writes poly[0] and poly[1] unconditionally.
inline constexpr int min_proxy_order = 2;
inline constexpr int max_proxy_order = 64;
inline constexpr int n_proxy_orders = max_proxy_order - min_proxy_order + 1;

template <typename T, int DIM>
using ndview = nda::basic_array_view<T, DIM, nda::F_layout>;
template <typename T>
using matrixview = nda::matrix_view<T, nda::F_layout>;

template <typename T>
using ndamatrix = nda::matrix<T, nda::F_layout, nda::heap<>>;

template <typename T>
using ndavector = nda::vector<T, nda::heap<>>;

template <typename T>
using direct_evaluator_func = std::function<void(int n_src, const T *r_src, const T *charge, const T *normals,
                                                 int n_trg, const T *r_trg, T *pot)>;

template <typename T>
using residual_evaluator_func =
    std::function<void(T rsc, T cen, T d2max, T thresh2, int n_src, const T *r_src, const T *charge, const T *normals,
                       int n_trg, const T *r_trg, T *pot)>;

// Range-list variant: instead of a contiguous [0, n_src) block, the source loop
// iterates over n_ranges disjoint sub-ranges given by (range_starts[i], range_lens[i]).
// This eliminates data copies when geometric pruning leaves gaps in the source array.
template <typename T>
using residual_evaluator_range_func =
    std::function<void(T rsc, T cen, T d2max, T thresh2, int n_src, const T *r_src, const T *charge, const T *normals,
                       int n_ranges, const int *range_starts, const int *range_lens, int n_trg, const T *r_trg,
                       T *pot)>;
} // namespace dmk

#endif
