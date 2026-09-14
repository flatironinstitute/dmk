#ifndef AOT_KERNELS_HPP
#define AOT_KERNELS_HPP

#include <dmk/types.hpp>

#include <vector>

namespace dmk {
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_laplace_2d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                    const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_laplace_3d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                    const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_sqrt_laplace_2d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                         const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_sqrt_laplace_3d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                         const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_stokeslet_3d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                      const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_stresslet_3d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                      const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_laplace_2d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                        const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_laplace_3d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                        const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_range_func<Real> get_esp_laplace_3d_kernel_ranges(dmk_eval_type eval_level_rt, int n_digits,
                                                                     const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_sqrt_laplace_2d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                             const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_sqrt_laplace_3d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                             const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_range_func<Real> get_esp_sqrt_laplace_3d_kernel_ranges(dmk_eval_type eval_level_rt, int n_digits,
                                                                          const std::vector<std::vector<Real>> &coeffs);
// Yukawa's coeff count depends on lambda*bsize, so in 2D the log half keeps a dynamic length while
// the regular half selects the branch. coeffs is [PA | PB] in 2D and a single polynomial in 3D.
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_yukawa_2d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                   const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_yukawa_3d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                   const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_yukawa_2d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                       const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_yukawa_3d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                       const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_range_func<Real> get_esp_yukawa_3d_kernel_ranges(dmk_eval_type eval_level_rt, int n_digits,
                                                                    const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_laplace_dipole_2d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                           const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_laplace_dipole_3d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                           const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_laplace_dipole_3d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                               const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_range_func<Real>
get_esp_laplace_dipole_3d_kernel_ranges(dmk_eval_type eval_level_rt, int n_digits,
                                        const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_stokeslet_3d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                          const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_range_func<Real> get_esp_stokeslet_3d_kernel_ranges(dmk_eval_type eval_level_rt, int n_digits,
                                                                       const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_esp_stresslet_3d_kernel(dmk_eval_type eval_level_rt, int n_digits,
                                                          const std::vector<std::vector<Real>> &coeffs);
template <class Real, int MaxVecLen>
residual_evaluator_range_func<Real> get_esp_stresslet_3d_kernel_ranges(dmk_eval_type eval_level_rt, int n_digits,
                                                                       const std::vector<std::vector<Real>> &coeffs);
} // namespace dmk
#endif
