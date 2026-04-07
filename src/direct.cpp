#include <dmk.h>
#include <dmk/aot_kernels.hpp>
#include <dmk/direct.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/legeexps.hpp>
#include <dmk/prolate0_fun.hpp>
#include <dmk/util.hpp>

#include <format>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <typeinfo>
#include <unordered_map>

#include <polyfit/fast_eval.hpp>
#include <sctl.hpp>

#ifdef DMK_USE_JIT
#include "jit_kernels_ir.h"
#include <rufus.hpp>
#endif

namespace dmk {
namespace {
template <class Vec>
auto to_vector(const auto &arr) {
    Vec vec(arr.size());
    std::copy_n(arr.data(), arr.size(), &vec[0]);
    return vec;
}

template <class Real, class Func>
std::vector<Real> make_polyfit_abs_error(int digits, Func &&f, Real a, Real b) {
    const Real tol = std::pow(10.0, -digits);

    for (int n_coeffs = 5; n_coeffs < 32; ++n_coeffs) {
        try {
            auto prolate_int_fun = poly_eval::make_func_eval(f, n_coeffs, a, b);

            bool passed = true;
            for (double x = a; x <= b; x += 0.01 * (b - a)) {
                const Real fit = prolate_int_fun(x);
                const Real act = f(x);
                const Real abs_err = std::abs(act - fit);
                if (abs_err > tol) {
                    passed = false;
                    continue;
                }
            }
            if (passed) {
                auto coeffs = to_vector<std::vector<Real>>(prolate_int_fun.coeffs());
                std::reverse(coeffs.begin(), coeffs.end());
                return coeffs;
            }
        } catch (std::exception &e) {
            std::cout << "Failed to fit with n_coeffs = " << n_coeffs << "\n";
            std::cout << e.what() << std::endl;
        }
    }
    return {};
}

struct CoeffsCache {
  public:
    template <typename T, class FitFunction>
    const std::vector<T> &get(int digits, FitFunction &fit_function) {
        auto key = std::make_tuple(typeid(FitFunction).hash_code(), digits);

        // Always fit in double first
        if (!double_map.contains(key)) {
            double_map[key] = fit_function(digits);
        }

        if constexpr (std::is_same_v<T, double>) {
            return double_map[key];
        } else {
            if (!float_map.contains(key)) {
                const auto &d = double_map[key];
                float_map[key] = std::vector<float>(d.begin(), d.end());
            }
            return float_map[key];
        }
    }

  private:
    using CacheKey = std::tuple<std::size_t, int>;
    struct hash_tuple {
        template <typename T1, typename T2>
        size_t operator()(const std::tuple<T1, T2> &x) const {
            auto a = std::hash<T1>()(std::get<0>(x));
            auto b = std::hash<T2>()(std::get<1>(x));
            return a ^ (b << 32);
        }
    };
    std::unordered_map<CacheKey, std::vector<double>, hash_tuple> double_map;
    std::unordered_map<CacheKey, std::vector<float>, hash_tuple> float_map;
};

CoeffsCache coeffs_cache;
} // namespace

template <typename Real>
Real log_windowed_kernel(Real x, Real beta, int dim, dmk::Prolate0Fun &prolate) {
    const Real psi0 = prolate.eval_val(0.0);

    constexpr int n_quad = 200;
    std::array<Real, n_quad> xs, whts;
    legerts(1, n_quad, xs.data(), whts.data());
    for (int i = 0; i < n_quad; ++i) {
        xs[i] = 0.5 * (xs[i] + Real{1.0}) * beta;
        whts[i] *= 0.5 * beta;
    }

    const Real rl = sqrt(Real(dim)) * 2;
    const Real dfac = rl * std::log(rl);

    Real fval = 0.0;
    for (int i = 0; i < n_quad; ++i) {
        const Real xval = xs[i] / beta;
        const Real fval0 = prolate.eval_val(xval);
        const Real z = rl * xs[i];
        Real dj0 = util::cyl_bessel_j(0, z);
        const Real dj1 = util::cyl_bessel_j(1, z);
        const Real tker = -(1 - dj0) / (xs[i] * xs[i]) + dfac * dj1 / xs[i];
        const Real fhat = tker * fval0 / psi0;
        dj0 = x > 0 ? util::cyl_bessel_j(0, x * xs[i]) : Real{1.0};
        fval += fhat * dj0 * whts[i] * xs[i];
    }

    return fval;
}

template <typename Real>
Real sl3d_local_kernel(Real r2, Real bsize, dmk::Prolate0Fun &prolate) {
    const Real r = std::sqrt(r2);
    auto compute_F = [&](Real upper) -> Real {
        constexpr int n_quad = 400; // overkill but it's offline
        static std::array<Real, n_quad> xs, whts;
        static bool init = false;
        if (!init) {
            legerts(1, n_quad, xs.data(), whts.data());
            init = true;
        }
        Real val = 0;
        for (int i = 0; i < n_quad; ++i) {
            const Real t = Real{0.5} * (xs[i] + Real{1}) * upper;
            const Real w = Real{0.5} * whts[i] * upper;
            val += prolate.eval_val(t / bsize) * t * w;
        }
        return val;
    };
    const Real c0 = compute_F(bsize);
    const Real F_r = compute_F(r);

    return Real{1} - F_r / c0;
}

template <typename Real>
std::vector<Real> get_local_correction_coeffs(dmk_ikernel kernel, int n_dim, int n_digits) {
    const double tol = std::pow(10.0, -n_digits);
    const double c0 = procl180_rescale(tol);
    dmk::Prolate0Fun prolate_fun(c0, 10000);

    auto fit = [&](auto func, double lo, double hi) -> std::vector<Real> {
        auto fit_func = [&](int digits) { return make_polyfit_abs_error<double>(digits, func, lo, hi); };
        return coeffs_cache.get<Real>(n_digits, fit_func);
    };

    switch (kernel) {
    case DMK_LAPLACE:
        if (n_dim == 2) {
            return fit([&](double x) { return -log_windowed_kernel<double>(std::sqrt(x), c0, 2, prolate_fun); }, 0.0,
                       1.0);
        }
        if (n_dim == 3) {
            const double prolate_inf_inv = 1.0 / prolate_fun.int_eval(1.0);
            return fit([&](double x) { return 1.0 - prolate_inf_inv * prolate_fun.int_eval(x); }, 0.0, 1.0);
        }
        break;
    case DMK_SQRT_LAPLACE:
        if (n_dim == 2) {
            const double prolate_inf_inv = 1.0 / prolate_fun.int_eval(1.0);
            return fit([&](double x) { return 1.0 - prolate_inf_inv * prolate_fun.int_eval((x + 1) / 2.0); }, -1.0,
                       1.0);
        }
        if (n_dim == 3) {
            return fit([&](double x) { return sl3d_local_kernel<double>(x, 1.0, prolate_fun); }, 0.0, 1.0);
        }
        break;
    default:
        break;
    }
    throw std::runtime_error("Unsupported kernel/dim for local correction coefficients");
}

#ifdef DMK_USE_JIT
namespace {
std::unique_ptr<RuFuS> RS;
__attribute__((constructor)) void init() {
    RS = std::make_unique<RuFuS>();
    RS->load_ir_string(rufus::embedded::jit_kernels_ir);
}
} // namespace

template <typename Real>
direct_evaluator_func<Real> make_evaluator_jit(dmk_ikernel kernel, dmk_pgh eval_level, int n_dim, int n_digits,
                                               int unroll_factor) {
    static std::mutex lock;
    std::lock_guard<std::mutex> lock_guard(lock);
    constexpr int VECWIDTH = sctl::DefaultVecLen<Real>();
    constexpr auto T_str = std::is_same_v<Real, float> ? "float" : "double";

    auto build_func_name = [&](const std::string &base_name) {
        return std::format("void {}<{}, {}, -1, -1, -1>", base_name, T_str, VECWIDTH);
    };
    const auto func_name = [&]() {
        switch (kernel) {
        case dmk_ikernel::DMK_LAPLACE:
            return build_func_name(std::format("laplace_{}d_poly_all_pairs", n_dim));
        case dmk_ikernel::DMK_SQRT_LAPLACE:
            return build_func_name(std::format("sqrt_laplace_{}d_poly_all_pairs", n_dim));
        default:
            throw std::runtime_error("Unsupported kernel for direct evaluator");
        }
    }();

    const auto coeffs = get_local_correction_coeffs<Real>(kernel, n_dim, n_digits);
    auto jit_func = RS->compile<void (*)(Real, Real, Real, Real, const Real *, int, const Real *, const Real *, int,
                                         const Real *, Real *)>(func_name, {{"eval_level_rt", int(eval_level)},
                                                                            {"n_coeffs_rt", coeffs.size()},
                                                                            {"n_digits_rt", n_digits},
                                                                            {"unroll_factor", unroll_factor}});
    if (!jit_func)
        throw std::runtime_error("Error compiling direct kernel");

    return [jit_func, coeffs](Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,
                              const Real *charge, int n_trg, const Real *r_trg, Real *pot) {
        jit_func(rsc, cen, d2max, thresh2, coeffs.data(), n_src, r_src, charge, n_trg, r_trg, pot);
    };
}

template direct_evaluator_func<float> make_evaluator_jit<float>(dmk_ikernel kernel, dmk_pgh eval_level, int n_dim,
                                                                int n_digits, int unroll_factor);
template direct_evaluator_func<double> make_evaluator_jit<double>(dmk_ikernel kernel, dmk_pgh eval_level, int n_dim,
                                                                  int n_digits, int unroll_factor);
#endif
// (DMK_USE_JIT)

template <typename Real>
direct_evaluator_func<Real> make_evaluator_aot(dmk_ikernel kernel, dmk_pgh eval_level, int n_dim, int n_digits,
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
    default:
        throw std::runtime_error("Unsupported kernel for direct evaluator");
    }
}

template std::vector<float> get_local_correction_coeffs<float>(dmk_ikernel kernel, int n_dim, int n_digits);
template std::vector<double> get_local_correction_coeffs<double>(dmk_ikernel kernel, int n_dim, int n_digits);
template direct_evaluator_func<float> make_evaluator_aot<float>(dmk_ikernel kernel, dmk_pgh eval_level, int n_dim,
                                                                int n_digits, int unroll_factor);
template direct_evaluator_func<double> make_evaluator_aot<double>(dmk_ikernel kernel, dmk_pgh eval_level, int n_dim,
                                                                  int n_digits, int unroll_factor);
} // namespace dmk
