#include <array>
#include <cmath>

#include <dmk/chebychev.hpp>
#include <dmk/direct.hpp>
#include <dmk/util.hpp>
#include <dmk/vector_kernels.hpp>

#define VECDIM 4

#ifdef __AVX512F__
#undef VECDIM
#define VECDIM 8
#endif

namespace dmk {

namespace {

template <typename Real>
inline void eval_polynomial_and_derivative(Real x, const Real *coeffs, int n_coeffs, Real &value, Real &derivative) {
    value = coeffs[n_coeffs - 1];
    derivative = Real{0};
    for (int i = n_coeffs - 2; i >= 0; --i) {
        derivative = derivative * x + value;
        value = value * x + coeffs[i];
    }
}

template <typename Real>
inline void eval_laplace_local_polynomial_3d(int n_digits, Real x, Real &value, Real &derivative) {
    if (n_digits <= 3) {
        static constexpr Real coeffs[] = {1.627823522210361e-01,  -4.553645597616490e-01, 4.171687104204163e-01,
                                          -7.073638602709915e-02, -8.957845614474928e-02, 2.617986644718201e-02,
                                          9.633427876507601e-03};
        return eval_polynomial_and_derivative(x, coeffs, 7, value, derivative);
    }
    if (n_digits <= 6) {
        static constexpr Real coeffs[] = {5.482525801351582e-02,  -2.616592110444692e-01, 4.862652666337138e-01,
                                          -3.894296348642919e-01, 1.638587821812791e-02,  1.870328434198821e-01,
                                          -8.714171086568978e-02, -3.927020727017803e-02, 3.728187607052319e-02,
                                          3.153734425831139e-03,  -8.651313377285847e-03, 1.725110090795567e-04,
                                          1.034762385284044e-03};
        return eval_polynomial_and_derivative(x, coeffs, 13, value, derivative);
    }
    if (n_digits <= 9) {
        static constexpr Real coeffs[] = {
            1.835718730962269e-02,  -1.258015846164503e-01, 3.609487248584408e-01,  -5.314579651112283e-01,
            3.447559412892380e-01,  9.664692318551721e-02,  -3.124274531849053e-01, 1.322460720579388e-01,
            9.773007866584822e-02,  -1.021958831082768e-01, -3.812847450976566e-03, 3.858117355875043e-02,
            -8.728545924521301e-03, -9.401196355382909e-03, 4.024549377076924e-03,  1.512806105865091e-03,
            -9.576734877247042e-04, -1.303457547418901e-04, 1.100385844683190e-04};
        return eval_polynomial_and_derivative(x, coeffs, 19, value, derivative);
    }

    static constexpr Real coeffs[] = {
        6.262472576363448e-03,  -5.605742936112479e-02, 2.185890864792949e-01,  -4.717350304955679e-01,
        5.669680214206270e-01,  -2.511606878849214e-01, -2.744523658778361e-01, 4.582527599363415e-01,
        -1.397724810121539e-01, -2.131762135835757e-01, 1.995489373508990e-01,  1.793390341864239e-02,
        -1.035055132403432e-01, 3.035606831075176e-02,  3.153931762550532e-02,  -2.033178627450288e-02,
        -5.406682731236552e-03, 7.543645573618463e-03,  1.437788047407851e-05,  -1.928370882351732e-03,
        2.891658777328665e-04,  3.332996162099811e-04,  -8.397699195938912e-05, -3.015837377517983e-05,
        9.640642701924662e-06};
    return eval_polynomial_and_derivative(x, coeffs, 25, value, derivative);
}

template <typename Real>
inline void eval_laplace_local_polynomial_2d(int n_digits, Real x, Real &value, Real &derivative) {
    if (n_digits <= 3) {
        static constexpr Real coeffs[] = {3.293312412035785e-01, -4.329140084314137e-01, 1.366683635926240e-01,
                                          -4.309918126794055e-02, 1.041106682948322e-02};
        return eval_polynomial_and_derivative(x, coeffs, 5, value, derivative);
    }
    if (n_digits <= 6) {
        static constexpr Real coeffs[] = {3.449851438836016e-01,  -4.902921365061905e-01, 2.220880572548949e-01,
                                          -1.153716526684871e-01, 5.535102319921498e-02,  -2.281631998557134e-02,
                                          7.843017349311455e-03,  -2.269922867123751e-03, 6.058276012390756e-04,
                                          -1.231943198424746e-04};
        return eval_polynomial_and_derivative(x, coeffs, 10, value, derivative);
    }
    if (n_digits <= 9) {
        static constexpr Real coeffs[] = {3.464285661408188e-01,  -4.987507216024493e-01, 2.448626056577886e-01,
                                          -1.531215433543293e-01, 9.898265049486202e-02,  -6.064807566030946e-02,
                                          3.368003719471507e-02,  -1.657913779091916e-02, 7.167896958658711e-03,
                                          -2.718119907588121e-03, 9.031254984609993e-04,  -2.605021223678894e-04,
                                          6.764515456758602e-05,  -1.806660493741674e-05, 3.640268744220521e-06};
        return eval_polynomial_and_derivative(x, coeffs, 15, value, derivative);
    }

    static constexpr Real coeffs[] = {
        3.465597422118963e-01, -4.998454957461734e-01, 2.491697186708244e-01, -1.637913095741979e-01,
        1.177474435136638e-01, -8.570242499810476e-02, 6.020769882062124e-02, -3.955364604811165e-02,
        2.382498816767467e-02, -1.301290744915523e-02, 6.410015506457686e-03, -2.841967252293358e-03,
        1.135602935662887e-03, -4.109780616023590e-04, 1.339086505665511e-04, -3.822398669202901e-05,
        1.037153217818392e-05, -3.251884408687046e-06, 7.149918161513096e-07};
    return eval_polynomial_and_derivative(x, coeffs, 19, value, derivative);
}

template <typename Real, int DIM>
void laplace_direct_eval(const ndview<const Real, 2> &r_src, const std::array<std::span<const Real>, DIM> &r_trg,
                         const ndview<const Real, 2> &charges, Real scale, Real center, Real d2max, ndview<Real, 2> &u,
                         Real *grad, int n_digits) {
    const int nsrc = r_src.extent(1);
    const int ntrg = r_trg[0].size();
    const int nd = u.extent(0);
    constexpr Real thresh2 = 1E-30;

    for (int i_trg = 0; i_trg < ntrg; ++i_trg) {
        for (int i_src = 0; i_src < nsrc; ++i_src) {
            std::array<Real, DIM> dx;
            Real r2 = Real{0};
            for (int i_dim = 0; i_dim < DIM; ++i_dim) {
                dx[i_dim] = r_trg[i_dim][i_trg] - r_src(i_dim, i_src);
                r2 += dx[i_dim] * dx[i_dim];
            }

            if (r2 < thresh2 || r2 > d2max)
                continue;

            Real factor;
            Real directional_scale;
            if constexpr (DIM == 2) {
                Real poly;
                Real dpoly;
                eval_laplace_local_polynomial_2d(n_digits, r2 * scale + center, poly, dpoly);
                factor = Real{0.5} * std::log(r2 * scale * Real{0.5}) + poly;
                directional_scale = Real{1} / r2 + Real{2} * scale * dpoly;
            } else {
                const Real r = std::sqrt(r2);
                Real poly;
                Real dpoly;
                eval_laplace_local_polynomial_3d(n_digits, (r + center) * scale, poly, dpoly);
                factor = poly / r;
                directional_scale = scale * dpoly / r2 - poly / (r2 * r);
            }

            for (int i = 0; i < nd; ++i) {
                const Real q = charges(i, i_src);
                u(i, i_trg) += q * factor;
                if (grad != nullptr) {
                    for (int i_dim = 0; i_dim < DIM; ++i_dim)
                        grad[i + nd * (i_dim + DIM * i_trg)] += q * dx[i_dim] * directional_scale;
                }
            }
        }
    }
}

} // namespace

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
                dkval = util::cyl_bessel_k(0, lambda * r);
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
                 Real center, Real d2max, ndview<Real, 2> u, Real *grad, int n_digits) {
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
        if (grad != nullptr)
            return laplace_direct_eval<Real, DIM>(r_src, r_trg, charges, scale, center, d2max, u, grad, n_digits);
        if constexpr (DIM == 2)
            return dmk::log_local_kernel_directcp_vec_cpp<Real, VECWIDTH>(
                &nd, &ndim, &n_digits, &scale, &center, &d2max, r_src.data(), &nsrc, charges.data(), trg_ptrs[0],
                trg_ptrs[1], trg_ptrs[2], &ntrg, u.data(), &thresh2);
        if constexpr (DIM == 3)
            return dmk::l3d_local_kernel_directcp_vec_cpp<Real, VECWIDTH>(
                &nd, &ndim, &n_digits, &scale, &center, &d2max, r_src.data(), &nsrc, charges.data(), trg_ptrs[0],
                trg_ptrs[1], trg_ptrs[2], &ntrg, u.data(), &thresh2);
    case dmk_ikernel::DMK_SQRT_LAPLACE:
        if constexpr (DIM == 2)
            return dmk::l3d_local_kernel_directcp_vec_cpp<Real, VECWIDTH>(
                &nd, &ndim, &n_digits, &scale, &center, &d2max, r_src.data(), &nsrc, charges.data(), trg_ptrs[0],
                trg_ptrs[1], trg_ptrs[2], &ntrg, u.data(), &thresh2);
        if constexpr (DIM == 3)
            return dmk::sl3d_local_kernel_directcp_vec_cpp<Real, VECWIDTH>(
                &nd, &ndim, &n_digits, &scale, &center, &d2max, r_src.data(), &nsrc, charges.data(), trg_ptrs[0],
                trg_ptrs[1], trg_ptrs[2], &ntrg, u.data(), &thresh2);
    }
}

template void direct_eval<float, 2>(dmk_ikernel ikernel, const ndview<float, 2> &r_src,
                                    const std::array<std::span<const float>, 2> &r_trg, const ndview<float, 2> &charges,
                                    const ndview<float, 1> &coeffs, const double *kernel_params, float scale,
                                    float center, float d2max, ndview<float, 2> u, float *grad, int n_digits);
template void direct_eval<float, 3>(dmk_ikernel ikernel, const ndview<float, 2> &r_src,
                                    const std::array<std::span<const float>, 3> &r_trg, const ndview<float, 2> &charges,
                                    const ndview<float, 1> &coeffs, const double *kernel_params, float scale,
                                    float center, float d2max, ndview<float, 2> u, float *grad, int n_digits);
template void direct_eval<double, 2>(dmk_ikernel ikernel, const ndview<double, 2> &r_src,
                                     const std::array<std::span<const double>, 2> &r_trg,
                                     const ndview<double, 2> &charges, const ndview<double, 1> &coeffs,
                                     const double *kernel_params, double scale, double center, double d2max,
                                     ndview<double, 2> u, double *grad, int n_digits);
template void direct_eval<double, 3>(dmk_ikernel ikernel, const ndview<double, 2> &r_src,
                                     const std::array<std::span<const double>, 3> &r_trg,
                                     const ndview<double, 2> &charges, const ndview<double, 1> &coeffs,
                                     const double *kernel_params, double scale, double center, double d2max,
                                     ndview<double, 2> u, double *grad, int n_digits);
} // namespace dmk
