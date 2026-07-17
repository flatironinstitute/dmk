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
residual_evaluator_func<Real> get_esp_3d_kernel(dmk_eval_type eval_level_rt, int n_digits);

// Raw (coeffs pointer, count) access to the same monomial fit tables get_esp_3d_kernel
// dispatches into — used by the GPU short-range path, which evaluates the polynomial
// itself in a CUDA kernel rather than going through the AOT/JIT host closure.
struct EspKernelCoeffs {
    const double *coeffs;
    int n_coeffs;
};
EspKernelCoeffs get_esp_3d_kernel_coeffs(dmk_eval_type eval_level, int n_digits);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_laplace_dipole_2d_kernel(dmk_eval_type eval_level_rt, int n_digits);
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> get_laplace_dipole_3d_kernel(dmk_eval_type eval_level_rt, int n_digits);
} // namespace dmk
#endif
