#ifndef DIRECT_HPP
#define DIRECT_HPP

#include <dmk.h>
#include <dmk/types.hpp>

namespace dmk {
template <typename Real, int DIM>
void direct_eval(dmk_ikernel ikernel, const ndview<Real, 2> &r_src, const std::array<std::span<const Real>, DIM> &r_trg,
                 const ndview<Real, 2> &charges, const ndview<Real, 1> &coeffs, const double *kernel_params, Real scale,
                 Real center, Real d2max, ndview<Real, 2> u, int n_digits);
}

#endif
