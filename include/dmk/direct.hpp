#ifndef DIRECT_HPP
#define DIRECT_HPP

#include <dmk.h>
#include <dmk/types.hpp>

namespace dmk {
template <typename Real, int DIM>
void direct_eval(dmk_ikernel ikernel, const ndview<Real, 2> &r_src, const std::array<std::span<const Real>, DIM> &r_trg,
                 const ndview<Real, 2> &charges, const ndview<Real, 1> &coeffs, const double *kernel_params, Real scale,
                 Real center, Real d2max, ndview<Real, 2> u, Real *grad, int n_digits);

inline int get_kernel_input_dim(int dim, dmk_ikernel kernel) {
    switch (kernel) {
    case DMK_YUKAWA:
        return 1;
    case DMK_LAPLACE:
        return 1;
    case DMK_SQRT_LAPLACE:
        return 1;
    }
    throw std::runtime_error("Invalid kernel");
}

inline int get_kernel_output_dim(int dim, dmk_ikernel kernel, dmk_pgh flags) {
    switch (kernel) {
    case DMK_YUKAWA:
        if (flags == DMK_POTENTIAL)
            return 1;
        if (flags == DMK_POTENTIAL_GRAD)
            return 1 + dim;
        if (flags == DMK_POTENTIAL_GRAD_HESSIAN)
            return 1 + dim + dim * dim;
    case DMK_LAPLACE:
        if (flags == DMK_POTENTIAL)
            return 1;
        if (flags == DMK_POTENTIAL_GRAD)
            return 1 + dim;
        if (flags == DMK_POTENTIAL_GRAD_HESSIAN)
            return 1 + dim + dim * dim;
    case DMK_SQRT_LAPLACE:
        if (flags == DMK_POTENTIAL)
            return 1;
        if (flags == DMK_POTENTIAL_GRAD)
            return 1 + dim;
        if (flags == DMK_POTENTIAL_GRAD_HESSIAN)
            return 1 + dim + dim * dim;
    }
    throw std::runtime_error("Invalid kernel");
}

template <typename Real>
std::vector<Real> get_local_correction_coeffs(dmk_ikernel kernel, int n_dim, int n_digits, double beta);
template <typename Real>
direct_evaluator_func<Real> make_evaluator_aot(dmk_ikernel kernel, dmk_pgh eval_level, int n_dim, int n_digits,
                                               int unroll_factor);
template <typename Real>
direct_evaluator_func<Real> make_evaluator_jit(dmk_ikernel kernel, dmk_pgh eval_level, int n_dim, int n_digits,
                                               int unroll_factor);
} // namespace dmk

#endif
