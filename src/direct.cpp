#include <dmk.h>
#include <dmk/aot_kernels.hpp>
#include <dmk/direct.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/legeexps.hpp>
#include <dmk/prolate0_fun.hpp>
#include <dmk/util.hpp>
#include <dmk/vector_kernels.hpp>

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
                    break;
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
    const std::vector<T> &get(int digits, double beta, FitFunction &fit_function) {
        auto key = std::make_tuple(typeid(FitFunction).hash_code(), digits, beta);
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
    using CacheKey = std::tuple<std::size_t, int, double>;
    struct hash_tuple {
        size_t operator()(const CacheKey &x) const {
            auto a = std::hash<std::size_t>()(std::get<0>(x));
            auto b = std::hash<int>()(std::get<1>(x));
            auto c = std::hash<double>()(std::get<2>(x));
            return a ^ (b << 16) ^ (c << 32);
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

// Computes the diagonal and off-diagonal residual kernel functions for the
// Stokes DMK, then fits them with polynomials.
//
// The Stokes Green's function decomposes as:
//   G_{ij}(r) = g_diag(r) * delta_{ij} + g_offd(r) * r_i * r_j / r^2
//
// g_diag and g_offd are built from the first and second derivatives of
// a radial function f(r), which is defined via its Fourier transform:
//   fhat(k) = pswf_split(k / beta) / psi(0) * T(rl * k)
//
// where pswf_split(x) = psi(x) - 0.5 * x * psi'(x),
// and T is the truncated modified biharmonic kernel FT.

// Evaluate the truncated modified biharmonic kernel FT
inline double biharmonic_ft(int dim, double rl, double k) {
    double x = rl * k;
    if (dim == 3) {
        if (x > 0.2)
            return -(1.0 + 0.5 * cos(x) - 1.5 * sin(x) / x) / (k * k * k * k);
        double x2 = x * x, x4 = x2 * x2, x6 = x2 * x4, x8 = x2 * x6, x10 = x2 * x8;
        return -(1.0 / 120 - x2 / 2520 + x4 / 120960 - x6 / 9979200.0 + x8 / 1.24540416e9 - x10 / 2.17945728e11) * rl *
               rl * rl * rl;
    } else {
        if (x > 0.2) {
            double j0 = util::cyl_bessel_j(0, x);
            double j1 = util::cyl_bessel_j(1, x);
            return (-1.0 + j0 + 0.5 * x * j1) / (k * k * k * k);
        }
        double x2 = x * x, x4 = x2 * x2, x6 = x2 * x4, x8 = x2 * x6, x10 = x2 * x8;
        return (-1.0 / 64 + x2 / 1152 - x4 / 49152 + x6 / 3686400.0 - x8 / 4.2467328e8 + x10 / 6.93633024e10) * rl *
               rl * rl * rl;
    }
}

inline std::pair<double, double> stokes_residual_at_point(int dim, double rval, double beta, double bsize,
                                                          const Prolate0Fun &prolate) {

    if (rval < 1e-15)
        return {0.0, 0.0};

    const double psi0 = prolate.eval_val(0.0);
    const double rl = bsize * (std::sqrt(double(dim)) + 1.0);
    const double twooverpi = 2.0 / M_PI;
    const double cval = (dim == 2) ? 0.5 * (1.0 - std::log(rl)) : 1.0 / (2.0 * rl);

    // Step 1: Legendre quadrature on [0, beta/bsize]
    constexpr int n_quad = 200;
    std::array<double, n_quad> rks, whts, fhat;
    legerts(1, n_quad, rks.data(), whts.data());
    for (int i = 0; i < n_quad; ++i) {
        rks[i] = 0.5 * (rks[i] + 1.0) * beta / bsize;
        whts[i] *= 0.5 * beta / bsize;
    }

    // Step 2: Evaluate windowed kernel FT at quadrature nodes
    for (int i = 0; i < n_quad; ++i) {
        double xi = rks[i];
        double xval = xi * bsize / beta;

        double fval = 0.0;
        if (xval <= 1.0) {
            auto [psi, dpsi] = prolate.eval_val_derivative(xval);
            fval = (psi - 0.5 * xval * dpsi) / psi0;
        }

        fhat[i] = fval * biharmonic_ft(dim, rl, xi);
    }

    // Step 3: Weight by quadrature weights and k factors
    for (int i = 0; i < n_quad; ++i) {
        if (dim == 2)
            fhat[i] *= whts[i] * rks[i];
        else
            fhat[i] *= whts[i] * rks[i] * rks[i] * twooverpi;
    }

    // Step 4: Compute f'(r) and f''(r) at the given point
    double r = (dim == 2) ? std::sqrt(rval) : rval;
    double df = 0.0, d2f = 0.0;

    for (int i = 0; i < n_quad; ++i) {
        double k = rks[i];
        double dd = r * k;
        double drft, d2rft;

        if (dim == 2) {
            if (dd > 1e-15) {
                double j0 = util::cyl_bessel_j(0, dd);
                double j1 = util::cyl_bessel_j(1, dd);
                drft = -k * j1;
                d2rft = -k * k * (j0 - j1 / dd);
            } else {
                drft = 0.0;
                d2rft = -k * k * 0.5;
            }
        } else {
            if (dd > 1e-15) {
                drft = (dd * cos(dd) - sin(dd)) * k / (dd * dd);
                d2rft = ((2.0 - dd * dd) * sin(dd) - 2.0 * dd * cos(dd)) * k * k / (dd * dd * dd);
            } else {
                drft = -k * k / 3.0;
                d2rft = 0.0;
            }
        }

        df += drft * fhat[i];
        d2f += d2rft * fhat[i];
    }

    // Step 5: Biharmonic correction
    df += cval * r;
    d2f += cval;

    // Step 6: Combine
    double g_diag, g_offd;
    if (dim == 3) {
        g_diag = r * d2f + df;
        g_offd = -r * d2f + df;
    } else {
        g_diag = d2f;
        g_offd = (r > 1e-15) ? -d2f + df / r : 0.0;
    }

    return {g_diag, g_offd};
}

template <typename Real>
std::vector<std::vector<Real>> get_stokes_local_correction_coeffs(int dim, int n_digits, double beta,
                                                                  double bsize = 1.0) {

    Prolate0Fun prolate(beta, 10000);

    auto fit_diag = [&](int digits) {
        return make_polyfit_abs_error<double>(
            digits, [&](double x) { return stokes_residual_at_point(dim, x, beta, bsize, prolate).first; }, 0.0, 1.0);
    };

    auto fit_offd = [&](int digits) {
        return make_polyfit_abs_error<double>(
            digits, [&](double x) { return stokes_residual_at_point(dim, x, beta, bsize, prolate).second; }, 0.0, 1.0);
    };

    return {coeffs_cache.get<Real>(n_digits, beta, fit_diag), coeffs_cache.get<Real>(n_digits, beta, fit_offd)};
}

template <typename Real>
std::vector<std::vector<Real>> get_local_correction_coeffs(dmk_ikernel kernel, int n_dim, int n_digits, double beta) {
    static std::mutex lock;
    std::lock_guard<std::mutex> lock_guard(lock);

    const double tol = std::pow(10.0, -n_digits);
    dmk::Prolate0Fun prolate_fun(beta, 10000);

    auto fit = [&](auto func, double lo, double hi) -> std::vector<Real> {
        auto fit_func = [&](int digits) { return make_polyfit_abs_error<double>(digits, func, lo, hi); };
        return coeffs_cache.get<Real>(n_digits, beta, fit_func);
    };

    switch (kernel) {
    case DMK_LAPLACE:
        if (n_dim == 2) {
            return {fit([&](double x) { return -log_windowed_kernel<double>(std::sqrt(x), beta, 2, prolate_fun); }, 0.0,
                        1.0)};
        }
        if (n_dim == 3) {
            const double prolate_inf_inv = 1.0 / prolate_fun.int_eval(1.0);
            return {fit([&](double x) { return 1.0 - prolate_inf_inv * prolate_fun.int_eval(x); }, 0.0, 1.0)};
        }
        break;
    case DMK_SQRT_LAPLACE:
        if (n_dim == 2) {
            const double prolate_inf_inv = 1.0 / prolate_fun.int_eval(1.0);
            return {
                fit([&](double x) { return 1.0 - prolate_inf_inv * prolate_fun.int_eval((x + 1) / 2.0); }, -1.0, 1.0)};
        }
        if (n_dim == 3) {
            return {fit([&](double x) { return sl3d_local_kernel<double>(x, 1.0, prolate_fun); }, 0.0, 1.0)};
        }
        break;
    case DMK_STOKESLET:
        return get_stokes_local_correction_coeffs<Real>(n_dim, n_digits, beta);
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
residual_evaluator_func<Real> make_evaluator_jit(dmk_ikernel kernel, dmk_eval_type eval_level, int n_dim, int n_digits,
                                                 double beta, int unroll_factor) {
    static std::mutex lock;
    std::lock_guard<std::mutex> lock_guard(lock);
    constexpr int VECWIDTH = sctl::DefaultVecLen<Real>();
    constexpr auto T_str = std::is_same_v<Real, float> ? "float" : "double";
    using ft =
        void (*)(Real, Real, Real, Real, const Real *, int, const Real *, const Real *, int, const Real *, Real *);

    auto build_func_name = [&](const std::string &base_name, int n_polynomials = 1) {
        std::string args = std::format("void {}<{}, {}, -1, -1", base_name, T_str, VECWIDTH);
        for (int i = 0; i < n_polynomials; ++i)
            args += ", -1";
        return args + ">";
    };
    const auto func_name = [&]() {
        switch (kernel) {
        case dmk_ikernel::DMK_LAPLACE:
            return build_func_name(std::format("laplace_{}d_poly_all_pairs", n_dim), 1);
        case dmk_ikernel::DMK_SQRT_LAPLACE:
            return build_func_name(std::format("sqrt_laplace_{}d_poly_all_pairs", n_dim), 1);
        case dmk_ikernel::DMK_STOKESLET:
            return build_func_name(std::format("stokeslet_{}d_poly_all_pairs", n_dim), 2);
        default:
            throw std::runtime_error("Unsupported kernel for direct evaluator");
        }
    }();

    const auto coeffs = get_local_correction_coeffs<Real>(kernel, n_dim, n_digits, beta);
    std::map<std::string, int> args_to_consume = {
        {"eval_level_rt", int(eval_level)}, {"n_digits_rt", n_digits}, {"unroll_factor", unroll_factor}};
    for (int i = 0; i < coeffs.size(); ++i)
        args_to_consume["n_coeffs_rt_" + std::to_string(i)] = coeffs[i].size();

    ft jit_func = RS->compile<void (*)(Real, Real, Real, Real, const Real *, int, const Real *, const Real *, int,
                                       const Real *, Real *)>(func_name, args_to_consume);
    if (!jit_func)
        throw std::runtime_error("Error compiling direct kernel");

    std::vector<Real> coeffs_cat;
    for (const auto &cvec : coeffs)
        coeffs_cat.insert(coeffs_cat.end(), cvec.begin(), cvec.end());

    return [jit_func, coeffs_cat](Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,
                                  const Real *charge, int n_trg, const Real *r_trg, Real *pot) {
        jit_func(rsc, cen, d2max, thresh2, coeffs_cat.data(), n_src, r_src, charge, n_trg, r_trg, pot);
    };
}

template residual_evaluator_func<float> make_evaluator_jit<float>(dmk_ikernel kernel, dmk_eval_type eval_level,
                                                                  int n_dim, int n_digits, double beta,
                                                                  int unroll_factor);
template residual_evaluator_func<double> make_evaluator_jit<double>(dmk_ikernel kernel, dmk_eval_type eval_level,
                                                                    int n_dim, int n_digits, double beta,
                                                                    int unroll_factor);
#endif
// (DMK_USE_JIT)

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
    default:
        throw std::runtime_error("Unsupported kernel for local evaluator");
    }
}

template <typename Real>
direct_evaluator_func<Real> get_direct_evaluator(dmk_ikernel kernel, dmk_eval_type eval_level, int n_dim, Real lambda) {
    constexpr int MaxVecLen = sctl::DefaultVecLen<Real>();
    constexpr int unroll_factor = 3;
    switch (kernel) {
    case dmk_ikernel::DMK_YUKAWA:
        if (n_dim == 2)
            return [lambda](int n_src, const Real *r_src, const Real *charge, const Real *normals, int n_trg,
                            const Real *r_trg, Real *pot) {
                for (int i = 0; i < n_trg; ++i) {
                    for (int j = 0; j < n_src; ++j) {
                        const double dr2 = sctl::pow<2>(r_src[j * 2] - r_trg[i * 2]) +
                                           sctl::pow<2>(r_src[j * 2 + 1] - r_trg[i * 2 + 1]);
                        if (!dr2)
                            continue;
                        pot[i] += charge[j] * util::cyl_bessel_k(0, lambda * std::sqrt(dr2));
                    }
                }
            };
        if (n_dim == 3)
            return [lambda](int n_src, const Real *r_src, const Real *charge, const Real *normals, int n_trg,
                            const Real *r_trg, Real *pot) {
                yukawa_3d_all_pairs_direct<Real, MaxVecLen>(n_src, r_src, charge, n_trg, r_trg, pot, unroll_factor,
                                                            lambda);
            };

    case dmk_ikernel::DMK_LAPLACE:
        if (n_dim == 2)
            return [eval_level](int n_src, const Real *r_src, const Real *charge, const Real *normals, int n_trg,
                                const Real *r_trg, Real *pot) {
                laplace_2d_all_pairs_direct<Real, MaxVecLen>(n_src, r_src, charge, n_trg, r_trg, pot, unroll_factor,
                                                             eval_level);
            };
        if (n_dim == 3)
            return [eval_level](int n_src, const Real *r_src, const Real *charge, const Real *normals, int n_trg,
                                const Real *r_trg, Real *pot) {
                laplace_3d_all_pairs_direct<Real, MaxVecLen>(n_src, r_src, charge, n_trg, r_trg, pot, unroll_factor,
                                                             eval_level);
            };
    case dmk_ikernel::DMK_SQRT_LAPLACE:
        if (n_dim == 2)
            return [](int n_src, const Real *r_src, const Real *charge, const Real *normals, int n_trg,
                      const Real *r_trg, Real *pot) {
                sqrt_laplace_2d_all_pairs_direct<Real, MaxVecLen>(n_src, r_src, charge, n_trg, r_trg, pot,
                                                                  unroll_factor);
            };
        if (n_dim == 3)
            return [](int n_src, const Real *r_src, const Real *charge, const Real *normals, int n_trg,
                      const Real *r_trg, Real *pot) {
                sqrt_laplace_3d_all_pairs_direct<Real, MaxVecLen>(n_src, r_src, charge, n_trg, r_trg, pot,
                                                                  unroll_factor);
            };
    case dmk_ikernel::DMK_STOKESLET:
        if (n_dim == 3)
            return [](int n_src, const Real *r_src, const Real *charge, const Real *normals, int n_trg,
                      const Real *r_trg, Real *pot) {
                stokeslet_3d_all_pairs_direct<Real, MaxVecLen>(n_src, r_src, charge, n_trg, r_trg, pot, unroll_factor);
            };
    case dmk_ikernel::DMK_STRESSLET:
        if (n_dim == 3)
            return [](int n_src, const Real *r_src, const Real *charge, const Real *normals, int n_trg,
                      const Real *r_trg, Real *pot) {
                stresslet_3d_all_pairs_direct<Real, MaxVecLen>(n_src, r_src, charge, normals, n_trg, r_trg, pot,
                                                               unroll_factor);
            };
    default:
        throw std::runtime_error("Unsupported kernel for direct evaluator");
    }
}

template std::vector<std::vector<float>> get_local_correction_coeffs<float>(dmk_ikernel kernel, int n_dim, int n_digits,
                                                                            double beta);
template std::vector<std::vector<double>> get_local_correction_coeffs<double>(dmk_ikernel kernel, int n_dim,
                                                                              int n_digits, double beta);
template residual_evaluator_func<float> make_evaluator_aot<float>(dmk_ikernel kernel, dmk_eval_type eval_level,
                                                                  int n_dim, int n_digits, int unroll_factor);
template residual_evaluator_func<double> make_evaluator_aot<double>(dmk_ikernel kernel, dmk_eval_type eval_level,
                                                                    int n_dim, int n_digits, int unroll_factor);

template direct_evaluator_func<float> get_direct_evaluator(dmk_ikernel kernel, dmk_eval_type eval_level, int n_dim,
                                                           float lambda);
template direct_evaluator_func<double> get_direct_evaluator(dmk_ikernel kernel, dmk_eval_type eval_level, int n_dim,
                                                            double lambda);
} // namespace dmk
