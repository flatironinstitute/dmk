#include <dmk.h>
#include <dmk/aot_kernels.hpp>
#include <dmk/direct.hpp>

#include <stdexcept>

#include <sctl.hpp>

namespace dmk {

template <typename Real>
residual_evaluator_func<Real> make_evaluator_aot(dmk_ikernel kernel, dmk_eval_type eval_level, int n_dim, int n_digits,
                                                 int unroll_factor) {
    constexpr int MaxVecLen = sctl::DefaultVecLen<Real>();
    switch (kernel) {
    case dmk_ikernel::DMK_LAPLACE:
        if (n_dim == 2)
            return get_laplace_2d_kernel<Real, MaxVecLen>(eval_level, n_digits);
        if (n_dim == 3)
            return get_laplace_3d_kernel<Real, MaxVecLen>(eval_level, n_digits);
    case dmk_ikernel::DMK_SQRT_LAPLACE:
        if (n_dim == 2)
            return get_sqrt_laplace_2d_kernel<Real, MaxVecLen>(eval_level, n_digits);
        if (n_dim == 3)
            return get_sqrt_laplace_3d_kernel<Real, MaxVecLen>(eval_level, n_digits);
    case dmk_ikernel::DMK_STOKESLET:
        if (n_dim == 3)
            return get_stokeslet_3d_kernel<Real, MaxVecLen>(eval_level, n_digits);
    case dmk_ikernel::DMK_STRESSLET:
        if (n_dim == 3)
            return get_stresslet_3d_kernel<Real, MaxVecLen>(eval_level, n_digits);
    default:
        throw std::runtime_error("Unsupported kernel for local evaluator");
    }
}

template residual_evaluator_func<float> make_evaluator_aot<float>(dmk_ikernel kernel, dmk_eval_type eval_level,
                                                                  int n_dim, int n_digits, int unroll_factor);
template residual_evaluator_func<double> make_evaluator_aot<double>(dmk_ikernel kernel, dmk_eval_type eval_level,
                                                                    int n_dim, int n_digits, int unroll_factor);

} // namespace dmk
