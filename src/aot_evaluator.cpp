#include <dmk.h>
#include <dmk/aot_kernels.hpp>
#include <dmk/direct.hpp>

#include <stdexcept>
#include <vector>

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
    case dmk_ikernel::DMK_LAPLACE_DIPOLE:
        if (n_dim == 3)
            return get_laplace_dipole_3d_kernel<Real, MaxVecLen>(eval_level, n_digits);
    default:
        throw std::runtime_error("Unsupported kernel for local evaluator");
    }
}

template residual_evaluator_func<float> make_evaluator_aot<float>(dmk_ikernel kernel, dmk_eval_type eval_level,
                                                                  int n_dim, int n_digits, int unroll_factor);
template residual_evaluator_func<double> make_evaluator_aot<double>(dmk_ikernel kernel, dmk_eval_type eval_level,
                                                                    int n_dim, int n_digits, int unroll_factor);

template <typename Real>
residual_evaluator_func<Real> make_esp_evaluator_aot(dmk_ikernel kernel, double fparam, double r_c, int n_dim,
                                                     dmk_eval_type eval_level, int n_digits, double sigma) {
    constexpr int MaxVecLen = sctl::DefaultVecLen<Real>();
    if (kernel == DMK_LAPLACE) {
        if (n_dim == 2)
            return get_esp_laplace_2d_kernel<Real, MaxVecLen>(eval_level, n_digits);
        if (n_dim == 3)
            return get_esp_laplace_3d_kernel<Real, MaxVecLen>(eval_level, n_digits);
    } else if (kernel == DMK_SQRT_LAPLACE) {
        if (n_dim == 2)
            return get_esp_sqrt_laplace_2d_kernel<Real, MaxVecLen>(eval_level, n_digits);
        if (n_dim == 3)
            return get_esp_sqrt_laplace_3d_kernel<Real, MaxVecLen>(eval_level, n_digits);
    } else if (kernel == DMK_YUKAWA) {
        const auto cc = get_esp_correction_coeffs<Real>(kernel, fparam, r_c, n_dim, n_digits, sigma);
        std::vector<Real> coeffs;
        for (const auto &v : cc)
            coeffs.insert(coeffs.end(), v.begin(), v.end());
        const int n_coeffs_log = (n_dim == 2) ? int(cc[0].size()) : 0;
        const int n_coeffs = (n_dim == 2) ? int(cc[1].size()) : int(cc[0].size());
        if (n_dim == 2)
            return get_esp_yukawa_2d_kernel<Real, MaxVecLen>(eval_level, n_digits, coeffs.data(), n_coeffs,
                                                             n_coeffs_log);
        if (n_dim == 3)
            return get_esp_yukawa_3d_kernel<Real, MaxVecLen>(eval_level, n_digits, coeffs.data(), n_coeffs,
                                                             n_coeffs_log);
    }
    throw std::runtime_error("Unsupported kernel/dim for ESP AOT evaluator");
}

// Range twin (3D only; 2D short-range is dense and never uses range_evaluator).
template <typename Real>
residual_evaluator_range_func<Real> make_esp_range_evaluator_aot(dmk_ikernel kernel, double fparam, double r_c,
                                                                 int n_dim, dmk_eval_type eval_level, int n_digits,
                                                                 double sigma) {
    constexpr int MaxVecLen = sctl::DefaultVecLen<Real>();
    if (kernel == DMK_LAPLACE)
        return get_esp_laplace_3d_kernel_ranges<Real, MaxVecLen>(eval_level, n_digits);
    if (kernel == DMK_SQRT_LAPLACE)
        return get_esp_sqrt_laplace_3d_kernel_ranges<Real, MaxVecLen>(eval_level, n_digits);
    if (kernel == DMK_YUKAWA) {
        const auto cc = get_esp_correction_coeffs<Real>(kernel, fparam, r_c, n_dim, n_digits, sigma);
        std::vector<Real> coeffs(cc[0].begin(), cc[0].end());
        return get_esp_yukawa_3d_kernel_ranges<Real, MaxVecLen>(eval_level, n_digits, coeffs.data(), int(coeffs.size()),
                                                                0);
    }
    throw std::runtime_error("Unsupported kernel for ESP AOT range evaluator");
}

template residual_evaluator_func<float> make_esp_evaluator_aot<float>(dmk_ikernel, double, double, int, dmk_eval_type,
                                                                      int, double);
template residual_evaluator_func<double> make_esp_evaluator_aot<double>(dmk_ikernel, double, double, int, dmk_eval_type,
                                                                        int, double);
template residual_evaluator_range_func<float> make_esp_range_evaluator_aot<float>(dmk_ikernel, double, double, int,
                                                                                  dmk_eval_type, int, double);
template residual_evaluator_range_func<double> make_esp_range_evaluator_aot<double>(dmk_ikernel, double, double, int,
                                                                                    dmk_eval_type, int, double);

} // namespace dmk
