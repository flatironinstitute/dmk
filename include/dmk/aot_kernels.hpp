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
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_stresslet_3d_kernel(dmk_eval_type eval_level_rt, int n_digits);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_laplace_2d_kernel(dmk_eval_type eval_level_rt, int n_digits);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_laplace_3d_kernel(dmk_eval_type eval_level_rt, int n_digits);
template <class Real, int MaxVecLen>
residual_evaluator_range_func<Real> get_esp_laplace_3d_kernel_ranges(dmk_eval_type eval_level_rt, int n_digits);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_sqrt_laplace_2d_kernel(dmk_eval_type eval_level_rt, int n_digits);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_sqrt_laplace_3d_kernel(dmk_eval_type eval_level_rt, int n_digits);
template <class Real, int MaxVecLen>
residual_evaluator_range_func<Real> get_esp_sqrt_laplace_3d_kernel_ranges(dmk_eval_type eval_level_rt, int n_digits);
// Yukawa's coeff count is lambda-dependent, so its getters take the runtime-computed coeffs and
// dispatch on n_coeffs. In 2D, coeffs is [PA | PB] and n_coeffs is PB's (reg) length, n_coeffs_log
// is PA's; in 3D n_coeffs is the single poly's length and n_coeffs_log is unused.
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_yukawa_2d_kernel(dmk_eval_type eval_level_rt, int n_digits, const Real *coeffs,
                                                       int n_coeffs, int n_coeffs_log);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_yukawa_3d_kernel(dmk_eval_type eval_level_rt, int n_digits, const Real *coeffs,
                                                       int n_coeffs, int n_coeffs_log);
template <class Real, int MaxVecLen>
residual_evaluator_range_func<Real> get_esp_yukawa_3d_kernel_ranges(dmk_eval_type eval_level_rt, int n_digits,
                                                                    const Real *coeffs, int n_coeffs, int n_coeffs_log);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_laplace_dipole_2d_kernel(dmk_eval_type eval_level_rt, int n_digits);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_laplace_dipole_3d_kernel(dmk_eval_type eval_level_rt, int n_digits);
} // namespace dmk
#endif
