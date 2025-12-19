#include <format>
#include <tuple>
#include <typeinfo>
#include <unordered_map>
#include <variant>

#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/direct.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/legeexps.hpp>
#include <dmk/prolate0_fun.hpp>
#include <dmk/types.hpp>
#include <dmk/vector_kernels.hpp>

#include <polyfit/fast_eval.hpp>
#include <rufus.hpp>

#include "jit_kernels_ir.h"

#define VECDIM 4

#ifdef __AVX512F__
#undef VECDIM
#define VECDIM 8
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
    std::vector<T> &get(int digits, FitFunction &fit_function) {
        auto key = std::make_tuple(typeid(FitFunction).hash_code(), digits);
        if (coeffs_map.contains(key)) {
            return std::get<std::vector<T>>(coeffs_map[key]);
        } else {
            coeffs_map[key] = fit_function(digits);
            return std::get<std::vector<T>>(coeffs_map[key]);
        }
    }

  private:
    using Coefficients = std::variant<std::vector<float>, std::vector<double>>;
    using CacheKey = std::tuple<std::size_t, int>;

    struct hash_tuple {
        template <typename T1, typename T2>
        size_t operator()(const std::tuple<T1, T2> &x) const {
            auto a = std::hash<T1>()(std::get<0>(x));
            auto b = std::hash<T2>()(std::get<1>(x));
            return a ^ (b << 32);
        }
    };

    std::unordered_map<CacheKey, Coefficients, hash_tuple> coeffs_map;
};

std::unique_ptr<RuFuS> RS;
CoeffsCache coeffs_cache;
__attribute__((constructor)) void init() {
    RS = std::make_unique<RuFuS>();
    RS->load_ir_string(rufus::embedded::jit_kernels_ir);
}
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

    const Real rl = sqrt(dim * 1.0) * 2;
    const Real dfac = rl * std::log(rl);

    Real fval = 0.0;
    for (int i = 0; i < n_quad; ++i) {
        const Real xval = xs[i] / beta;
        const Real fval0 = prolate.eval_val(xval);
        const Real z = rl * xs[i];
        Real dj0 = std::cyl_bessel_j(0, z);
        const Real dj1 = std::cyl_bessel_j(1, z);
        const Real tker = -(1 - dj0) / (xs[i] * xs[i]) + dfac * dj1 / xs[i];
        const Real fhat = tker * fval0 / psi0;
        dj0 = x > 0 ? std::cyl_bessel_j(0, x * xs[i]) : Real{1.0};
        fval += fhat * dj0 * whts[i] * xs[i];
    }

    return fval;
}

template <typename Real>
direct_evaluator_func<Real> make_evaluator(dmk_ikernel kernel, int n_dim, int n_digits) {
    constexpr int VECWIDTH = std::is_same_v<Real, float> ? 2 * VECDIM : VECDIM;
    constexpr auto T_str = std::is_same_v<Real, float> ? "float" : "double";
    Real tol = std::pow(10.0, -n_digits);

    switch (kernel) {
    case dmk_ikernel::DMK_LAPLACE:
        if (n_dim == 2) {
            Real beta = procl180_rescale(tol);
            dmk::Prolate0Fun prolate(beta, 10000);
            auto log_windowed = [&beta, &prolate](Real x) {
                return -log_windowed_kernel<Real>(std::sqrt(x), beta, 2, prolate);
            };
            auto fit_func = [&log_windowed](int digits) {
                return make_polyfit_abs_error<Real>(digits, log_windowed, 0.0, 1.0);
            };

            const auto coeffs = coeffs_cache.get<Real>(n_digits, fit_func);

            const auto func_name =
                std::format("void laplace_2d_poly_all_pairs<{}, {}>(int, int, {}, {}, {}, {}, "
                            "int, {} const*, int, {} const*, {} const*, int, {} const*, {}*, int)",
                            T_str, VECWIDTH, T_str, T_str, T_str, T_str, T_str, T_str, T_str, T_str, T_str);

            auto jit_func = RS->compile<void (*)(int, Real, Real, Real, Real, const Real *, int, const Real *,
                                                 const Real *, int, const Real *, Real *)>(
                func_name, {{"n_coeffs", coeffs.size()}, {"n_digits", n_digits}, {"unroll_factor", 3}});

            return [jit_func, coeffs](int nd, Real rsc, Real cen, Real d2max, Real thresh2, int n_src,
                                      const Real *r_src, const Real *charge, int n_trg, const Real *r_trg, Real *pot) {
                jit_func(nd, rsc, cen, d2max, thresh2, coeffs.data(), n_src, r_src, charge, n_trg, r_trg, pot);
            };
        }
        if (n_dim == 3) {
            const Real tol = std::pow(10.0, -n_digits);
            const Real c0 = procl180_rescale(tol);
            dmk::Prolate0Fun prolate_fun(c0, 10000);
            const Real prolate_inf_inv = 1.0 / prolate_fun.int_eval(1.0);
            auto pswf = [&prolate_inf_inv, &prolate_fun](Real x) {
                return Real(1.0 - prolate_inf_inv * prolate_fun.int_eval((x + 1) / 2.0));
            };
            auto fit_func = [&pswf](int digits) { return make_polyfit_abs_error<Real>(digits, pswf, -1.0, 1.0); };

            const auto pswf_coeffs = coeffs_cache.get<Real>(n_digits, fit_func);

            const auto func_name =
                std::format("void laplace_3d_poly_all_pairs<{}, {}>(int, int, {}, {}, {}, {}, "
                            "int, {} const*, int, {} const*, {} const*, int, {} const*, {}*, int)",
                            T_str, VECWIDTH, T_str, T_str, T_str, T_str, T_str, T_str, T_str, T_str, T_str);

            auto jit_func = RS->compile<void (*)(int, Real, Real, Real, Real, const Real *, int, const Real *,
                                                 const Real *, int, const Real *, Real *)>(
                func_name, {{"n_coeffs", pswf_coeffs.size()}, {"n_digits", n_digits}, {"unroll_factor", 3}});

            return [jit_func, pswf_coeffs](int nd, Real rsc, Real cen, Real d2max, Real thresh2, int n_src,
                                           const Real *r_src, const Real *charge, int n_trg, const Real *r_trg,
                                           Real *pot) {
                jit_func(nd, rsc, cen, d2max, thresh2, pswf_coeffs.data(), n_src, r_src, charge, n_trg, r_trg, pot);
            };
        }
    default:
        throw std::runtime_error("Unsupported kernel for direct evaluator");
    }
}

template <typename Real, int DIM>
void yukawa_direct_eval(const ndview<const Real, 2> &r_src, const std::array<std::span<const Real>, DIM> &r_trg,
                        const ndview<const Real, 2> &charges, const ndview<const Real, 1> &coeffs, Real lambda,
                        Real scale, Real center, Real d2max, ndview<Real, 2> &u) {
    const int nsrc = r_src.extent(1);
    const int ntrg = r_trg[0].size();
    const int nd = u.extent(0);
    const int norder = coeffs.extent(0);
    constexpr Real threshq = 1E-30;

    for (int i_trg = 0; i_trg < ntrg; i_trg++) {
        for (int i_src = 0; i_src < nsrc; i_src++) {
            const Real dx = r_trg[0][i_trg] - r_src(0, i_src);
            const Real dy = r_trg[1][i_trg] - r_src(1, i_src);
            Real dd = dx * dx + dy * dy;
            if constexpr (DIM == 3) {
                Real dz = r_trg[2][i_trg] - r_src(2, i_src);
                dd += dz * dz;
            }

            if (dd < threshq || dd > d2max)
                continue;

            const Real r = sqrt(dd);
            const Real xval = r * scale + center;
            const Real fval = chebyshev::evaluate(xval, norder + 1, coeffs.data());
            Real dkval;
            if constexpr (DIM == 2)
                dkval = std::cyl_bessel_k(0, lambda * r);
            if constexpr (DIM == 3)
                dkval = std::exp(-lambda * r) / r;

            const Real factor = dkval + fval;
            for (int i = 0; i < nd; ++i)
                u(i, i_trg) += charges(i, i_src) * factor;
        }
    }
}

template <typename Real, int DIM>
void direct_eval(dmk_ikernel ikernel, const ndview<Real, 2> &r_src, const std::array<std::span<const Real>, DIM> &r_trg,
                 const ndview<Real, 2> &charges, const ndview<Real, 1> &coeffs, const double *kernel_params, Real scale,
                 Real center, Real d2max, ndview<Real, 2> u, int n_digits) {
    constexpr int VECWIDTH = std::is_same_v<Real, float> ? 2 * VECDIM : VECDIM;
    const int nd = charges.extent(0);
    const int ndim = DIM;
    const int nsrc = r_src.extent(1);
    const int ntrg = r_trg[0].size();
    const Real thresh2 = 1E-30; // FIXME
    const Real *trg_ptrs[3] = {r_trg[0].data(), r_trg[1].data(), DIM == 3 ? r_trg[2].data() : nullptr};

    switch (ikernel) {
    case dmk_ikernel::DMK_YUKAWA:
        return yukawa_direct_eval<Real, DIM>(r_src, r_trg, charges, coeffs, *kernel_params, scale, center, d2max, u);
    case dmk_ikernel::DMK_LAPLACE:
        if constexpr (DIM == 2)
            return log_local_kernel_directcp_vec_cpp<Real, VECWIDTH>(
                &nd, &ndim, &n_digits, &scale, &center, &d2max, r_src.data(), &nsrc, charges.data(), trg_ptrs[0],
                trg_ptrs[1], trg_ptrs[2], &ntrg, u.data(), &thresh2);
        if constexpr (DIM == 3)
            return l3d_local_kernel_directcp_vec_cpp<Real, VECWIDTH>(
                &nd, &ndim, &n_digits, &scale, &center, &d2max, r_src.data(), &nsrc, charges.data(), trg_ptrs[0],
                trg_ptrs[1], trg_ptrs[2], &ntrg, u.data(), &thresh2);
    case dmk_ikernel::DMK_SQRT_LAPLACE:
        if constexpr (DIM == 2)
            return l3d_local_kernel_directcp_vec_cpp<Real, VECWIDTH>(
                &nd, &ndim, &n_digits, &scale, &center, &d2max, r_src.data(), &nsrc, charges.data(), trg_ptrs[0],
                trg_ptrs[1], trg_ptrs[2], &ntrg, u.data(), &thresh2);
        if constexpr (DIM == 3)
            return sl3d_local_kernel_directcp_vec_cpp<Real, VECWIDTH>(
                &nd, &ndim, &n_digits, &scale, &center, &d2max, r_src.data(), &nsrc, charges.data(), trg_ptrs[0],
                trg_ptrs[1], trg_ptrs[2], &ntrg, u.data(), &thresh2);
    }
}

template void direct_eval<float, 2>(dmk_ikernel ikernel, const ndview<float, 2> &r_src,
                                    const std::array<std::span<const float>, 2> &r_trg, const ndview<float, 2> &charges,
                                    const ndview<float, 1> &coeffs, const double *kernel_params, float scale,
                                    float center, float d2max, ndview<float, 2> u, int n_digits);
template void direct_eval<float, 3>(dmk_ikernel ikernel, const ndview<float, 2> &r_src,
                                    const std::array<std::span<const float>, 3> &r_trg, const ndview<float, 2> &charges,
                                    const ndview<float, 1> &coeffs, const double *kernel_params, float scale,
                                    float center, float d2max, ndview<float, 2> u, int n_digits);
template void direct_eval<double, 2>(dmk_ikernel ikernel, const ndview<double, 2> &r_src,
                                     const std::array<std::span<const double>, 2> &r_trg,
                                     const ndview<double, 2> &charges, const ndview<double, 1> &coeffs,
                                     const double *kernel_params, double scale, double center, double d2max,
                                     ndview<double, 2> u, int n_digits);
template void direct_eval<double, 3>(dmk_ikernel ikernel, const ndview<double, 2> &r_src,
                                     const std::array<std::span<const double>, 3> &r_trg,
                                     const ndview<double, 2> &charges, const ndview<double, 1> &coeffs,
                                     const double *kernel_params, double scale, double center, double d2max,
                                     ndview<double, 2> u, int n_digits);

template direct_evaluator_func<float> make_evaluator<float>(dmk_ikernel kernel, int n_dim, int n_digits);
template direct_evaluator_func<double> make_evaluator<double>(dmk_ikernel kernel, int n_dim, int n_digits);

} // namespace dmk
