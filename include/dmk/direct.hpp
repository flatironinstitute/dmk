#ifndef DIRECT_HPP
#define DIRECT_HPP

#include <dmk.h>
#include <dmk/types.hpp>
#include <dmk/util.hpp>

#include <format>
#include <stdexcept>

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
    case DMK_STOKESLET:
        return dim;
    case DMK_STRESSLET:
        return dim * dim;
    }
    throw std::runtime_error("Invalid kernel");
}

inline int get_kernel_output_dim(int dim, dmk_ikernel kernel, dmk_eval_type flags) {
    switch (kernel) {
    case DMK_YUKAWA:
        if (flags == DMK_POTENTIAL)
            return 1;
        if (flags == DMK_POTENTIAL_GRAD)
            return 1 + dim;
        if (flags == DMK_POTENTIAL_GRAD_HESSIAN)
            return 1 + dim + dim * dim;
        break;
    case DMK_LAPLACE:
        if (flags == DMK_POTENTIAL)
            return 1;
        if (flags == DMK_POTENTIAL_GRAD)
            return 1 + dim;
        if (flags == DMK_POTENTIAL_GRAD_HESSIAN)
            return 1 + dim + dim * dim;
        break;
    case DMK_SQRT_LAPLACE:
        if (flags == DMK_POTENTIAL)
            return 1;
        if (flags == DMK_POTENTIAL_GRAD)
            return 1 + dim;
        if (flags == DMK_POTENTIAL_GRAD_HESSIAN)
            return 1 + dim + dim * dim;
        break;
    case DMK_STOKESLET:
        if (flags == DMK_VELOCITY)
            return dim;
        break;
    case DMK_STRESSLET:
        if (flags == DMK_VELOCITY)
            return dim;
        break;
    }
    using dmk::util::to_string;
    throw std::runtime_error(
        std::format("Invalid kernel/output combination {} + {}\n", to_string(kernel), to_string(flags)));
}

template <typename Real>
direct_evaluator_func<Real> get_direct_evaluator(dmk_ikernel kernel, dmk_eval_type eval_level, int n_dim, Real lambda);

template <typename Real>
inline void parallel_direct_eval(const dmk::direct_evaluator_func<Real> &func, int n_src, const Real *r_src,
                                 const Real *charge, const Real *normals, int n_trg, const Real *r_trg, Real *pot,
                                 int spatial_dim, int out_dim) {
#pragma omp parallel
    {
        const int nt = MY_OMP_GET_NUM_THREADS();
        const int tid = MY_OMP_GET_THREAD_NUM();
        const int lo = (tid * n_trg) / nt;
        const int hi = ((tid + 1) * n_trg) / nt;
        if (hi > lo)
            func(n_src, r_src, charge, normals, hi - lo, r_trg + lo * spatial_dim, pot + lo * out_dim);
    }
}

inline void compute_direct(int n_dim, const auto &r_src, const auto &charges, const auto &normals, const auto &r_trg,
                           auto &pot_direct, dmk_ikernel kernel, dmk_eval_type eval_level) {
    using Real = std::decay_t<decltype(r_src)>::value_type;
    const long n_src = r_src.size() / n_dim;
    const long n_trg = r_trg.size() / n_dim;
    const long out_dim = get_kernel_output_dim(n_dim, kernel, eval_level);
    const double lambda = 6.0;
    pot_direct.assign(n_trg * out_dim, 0);
    auto potfunc = dmk::get_direct_evaluator<Real>(kernel, eval_level, n_dim, lambda);
    parallel_direct_eval(potfunc, n_src, r_src.data(), charges.data(), normals.data(), n_trg, r_trg.data(),
                         pot_direct.data(), n_dim, out_dim);
}

template <typename Real>
std::vector<std::vector<Real>> get_local_correction_coeffs(dmk_ikernel kernel, int n_dim, int n_digits, double beta);
template <typename Real>
residual_evaluator_func<Real> make_evaluator_aot(dmk_ikernel kernel, dmk_eval_type eval_level, int n_dim, int n_digits,
                                                 int unroll_factor);
template <typename Real>
residual_evaluator_func<Real> make_evaluator_jit(dmk_ikernel kernel, dmk_eval_type eval_level, int n_dim, int n_digits,
                                                 double beta, int unroll_factor);
} // namespace dmk

#endif
