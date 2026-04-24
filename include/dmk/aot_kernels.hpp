#ifndef AOT_KERNELS_HPP
#define AOT_KERNELS_HPP

#include <dmk/types.hpp>

namespace dmk {
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_laplace_2d_kernel(dmk_eval_type eval_level_rt, int n_digits);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_laplace_3d_kernel(dmk_eval_type eval_level_rt, int n_digits);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_sqrt_laplace_2d_kernel(dmk_eval_type eval_level_rt, int n_digits);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_sqrt_laplace_3d_kernel(dmk_eval_type eval_level_rt, int n_digits);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_stokeslet_3d_kernel(dmk_eval_type eval_level_rt, int n_digits);
} // namespace dmk
#endif
