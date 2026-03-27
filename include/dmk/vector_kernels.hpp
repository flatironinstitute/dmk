#ifndef VECTOR_KERNELS_NEW_HPP
#define VECTOR_KERNELS_NEW_HPP

//  ___ __  __ ____   ___  ____ _____  _    _   _ _____
// |_ _|  \/  |  _ \ / _ \|  _ \_   _|/ \  | \ | |_   _|
//  | || |\/| | |_) | | | | |_) || | / _ \ |  \| | | |
//  | || |  | |  __/| |_| |  _ < | |/ ___ \| |\  | | |
// |___|_|  |_|_|    \___/|_| \_\|_/_/   \_\_| \_| |_|
//
// There are some strange ways to do things in this file. Largely, the code is written to be
// friendly to JIT compilation of n_digits/n_coeffs while also being friendly to AoT
// compilation. This means that template or runtime arguments can be used (where template will
// override runtime if supplied). This leads to sometimes dumb looking functions. Notably
// runtime dispatch of the rsqrt function, when a template is right there. Or passing
// performance critical parameters as runtime arguments (like n_digits, n_coeffs).

#include <dmk.h>
#include <dmk/util.hpp>

#include <sctl.hpp>

#define DMK_ALWAYS_INLINE __attribute__((always_inline))

template <typename Real>
DMK_ALWAYS_INLINE void shift_scale_polynomial(const Real *coeffs_in, Real a, Real b, Real *coeffs_out, int N) {
    Real tmp[64];
    for (int i = 0; i < N; i++)
        tmp[i] = coeffs_in[i];

    for (int i = 0; i < N - 1; i++) {
        for (int j = N - 2; j >= i; j--) {
            tmp[j] += b * tmp[j + 1];
        }
    }

    Real ak = 1.0;
    for (int k = 0; k < N; k++) {
        coeffs_out[k] = tmp[k] * ak;
        ak *= a;
    }
}

template <class F, class... Args>
DMK_ALWAYS_INLINE auto dispatch_digits(int n_digits, F &&f, Args &&...args) {
    // clang-format off
    switch (n_digits) {
    case 2: return f.template operator()<2>(std::forward<Args>(args)...);
    case 3: return f.template operator()<3>(std::forward<Args>(args)...);
    case 4: return f.template operator()<4>(std::forward<Args>(args)...);
    case 5: return f.template operator()<5>(std::forward<Args>(args)...);
    case 6: return f.template operator()<6>(std::forward<Args>(args)...);
    case 7: return f.template operator()<7>(std::forward<Args>(args)...);
    case 8: return f.template operator()<8>(std::forward<Args>(args)...);
    case 9: return f.template operator()<9>(std::forward<Args>(args)...);
    case 10: return f.template operator()<10>(std::forward<Args>(args)...);
    case 11: return f.template operator()<11>(std::forward<Args>(args)...);
    case 12: return f.template operator()<12>(std::forward<Args>(args)...);
    default: throw std::runtime_error("Unsupported digits");
    }
    // clang-format on
}

// AoT: don't force to reason about dispatch tables/inlinining. just use the rsqrt
// JIT: Dispatch table to correct sctl call
template <int N_DIGITS = -1, typename VecType>
DMK_ALWAYS_INLINE VecType my_approx_rsqrt(const VecType &x, int digits) {
    if constexpr (N_DIGITS > 0) {
        return sctl::approx_rsqrt<N_DIGITS>(x);
    } else {
        auto kernel = [&]<int DIGITS>() -> VecType { return sctl::approx_rsqrt<DIGITS>(x); };
        return dispatch_digits(digits, kernel);
    }
}

template <typename VecType>
DMK_ALWAYS_INLINE VecType horner(const VecType &x, const typename VecType::ScalarType *coeffs, int n_coeffs) {
    VecType poly = coeffs[n_coeffs - 1];
    for (int i = n_coeffs - 2; i >= 0; --i)
        poly = FMA(poly, x, VecType::Load1(&coeffs[i]));
    return poly;
}

// Trick to do evals directly on x2, but have to call on even/odd separately
template <int shift, typename VecType>
DMK_ALWAYS_INLINE VecType horner_split(const VecType &x2, const typename VecType::ScalarType *coeffs, int n_coeffs) {
    const int start_coeff = ((n_coeffs - 1 - shift) & ~1) + shift;
    VecType poly = coeffs[start_coeff];
    for (int i = start_coeff - 2; i >= 0; i -= 2)
        poly = FMA(poly, x2, VecType::Load1(&coeffs[i]));
    return poly;
}

// Simultaneous horner and derivative
template <typename VecType>
DMK_ALWAYS_INLINE void horner_val_deriv(const VecType &x, const typename VecType::ScalarType *coeffs, int n_coeffs,
                                        VecType &value, VecType &derivative) {
    value = VecType::Load1(&coeffs[n_coeffs - 1]);
    derivative = VecType::Zero();
    for (int i = n_coeffs - 2; i >= 0; --i) {
        derivative = FMA(derivative, x, value);
        value = FMA(value, x, VecType::Load1(&coeffs[i]));
    }
}

// Simultaneous split horner val/derivative
template <int shift, typename VecType>
DMK_ALWAYS_INLINE void horner_split_val_deriv(const VecType &x2, const typename VecType::ScalarType *coeffs,
                                              int n_coeffs, VecType &value, VecType &derivative) {
    const int start_coeff = ((n_coeffs - 1 - shift) & ~1) + shift;
    value = VecType::Load1(&coeffs[start_coeff]);
    derivative = VecType::Zero();
    for (int i = start_coeff - 2; i >= shift; i -= 2) {
        derivative = FMA(derivative, x2, value);
        value = FMA(value, x2, VecType::Load1(&coeffs[i]));
    }
}

template <class Real, int KERNEL_INPUT_DIM, int KERNEL_OUTPUT_DIM, int SPATIAL_DIM, int NORMAL_DIM, int VecLen,
          class uKernelEvaluator>
DMK_ALWAYS_INLINE void EvalPairs(int Ns, const Real *__restrict__ r_src, const Real *__restrict__ v_src,
                                 const Real *__restrict__ src_normals, int Nt, const Real *__restrict__ r_trg,
                                 Real *__restrict__ v_trg, uKernelEvaluator uKernel, int unroll_factor, int digits) {
    using namespace sctl;
    using RealVec = Vec<Real, VecLen>;

    Real scale_factor = 1.0;
    if constexpr (requires {
                      { uKernel.scale_factor() };
                  }) {
        scale_factor = uKernel.scale_factor();
    }

    const Long NNt = ((Nt + VecLen - 1) / VecLen) * VecLen;
    if (NNt == VecLen) {
        RealVec xt[SPATIAL_DIM], vt[KERNEL_OUTPUT_DIM], xs[SPATIAL_DIM], ns[NORMAL_DIM], vs[KERNEL_INPUT_DIM];
        for (Integer k = 0; k < KERNEL_OUTPUT_DIM; k++)
            vt[k] = RealVec::Zero();
        for (Integer k = 0; k < SPATIAL_DIM; k++) {
            alignas(sizeof(RealVec)) std::array<Real, VecLen> Xt;
            RealVec::Zero().StoreAligned(&Xt[0]);
            for (Integer i = 0; i < Nt; i++)
                Xt[i] = r_trg[i * SPATIAL_DIM + k];
            xt[k] = RealVec::LoadAligned(&Xt[0]);
        }
        for (Long s = 0; s < Ns; s++) {
            for (Integer k = 0; k < SPATIAL_DIM; k++)
                xs[k] = RealVec::Load1(&r_src[s * SPATIAL_DIM + k]);
            for (Integer k = 0; k < NORMAL_DIM; k++)
                ns[k] = RealVec::Load1(&src_normals[s * NORMAL_DIM + k]);
            for (Integer k = 0; k < KERNEL_INPUT_DIM; k++)
                vs[k] = RealVec::Load1(&v_src[s * KERNEL_INPUT_DIM + k]);

            RealVec dX[SPATIAL_DIM], U[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM];
            for (Integer i = 0; i < SPATIAL_DIM; i++)
                dX[i] = xt[i] - xs[i];
            if constexpr (NORMAL_DIM > 0)
                uKernel(U, dX, ns);
            else
                uKernel(U, dX);

            for (Integer k0 = 0; k0 < KERNEL_INPUT_DIM; k0++) {
                for (Integer k1 = 0; k1 < KERNEL_OUTPUT_DIM; k1++) {
                    vt[k1] = FMA(U[k0][k1], vs[k0], vt[k1]);
                }
            }
        }
        for (Integer k = 0; k < KERNEL_OUTPUT_DIM; k++) {
            alignas(sizeof(RealVec)) std::array<Real, VecLen> out;
            vt[k].StoreAligned(&out[0]);
            for (Long t = 0; t < Nt; t++) {
                v_trg[t * KERNEL_OUTPUT_DIM + k] += out[t] * scale_factor;
            }
        }
    } else {
        const Real *__restrict__ Xs_ = r_src;
        const Real *__restrict__ Ns_ = src_normals;
        const Real *__restrict__ Vs_ = v_src;

        constexpr Integer Nbuff = 16 * 1024;
        constexpr Integer alignment = sizeof(RealVec) / sizeof(Real);

        const Integer Xt_size = SPATIAL_DIM * NNt;
        const Integer required_size = Xt_size + NNt * KERNEL_OUTPUT_DIM;

        dmk::util::StackOrHeapBuffer<Real, Nbuff> buffer(required_size);
        Real *buff = buffer.data();
        Real *__restrict__ Xt_ = buff;
        Real *__restrict__ Vt_ = buff + Xt_size;

        for (Long k = 0; k < SPATIAL_DIM; k++) {
            for (Long i = 0; i < Nt; i++) {
                Xt_[k * NNt + i] = r_trg[i * SPATIAL_DIM + k];
            }
            for (Long i = Nt; i < NNt; i++) {
                Xt_[k * NNt + i] = 0;
            }
        }

        constexpr int MAX_UNROLL = 8;
        Long t = 0;
        for (; t + unroll_factor * VecLen <= NNt; t += unroll_factor * VecLen) {
            RealVec xt[MAX_UNROLL][SPATIAL_DIM];
            RealVec vt[MAX_UNROLL][KERNEL_OUTPUT_DIM];

            for (int u = 0; u < MAX_UNROLL; u++) {
                for (Integer k = 0; k < KERNEL_OUTPUT_DIM; k++) {
                    vt[u][k] = RealVec::Zero();
                }
                for (Integer k = 0; k < SPATIAL_DIM; k++) {
                    xt[u][k] = RealVec::LoadAligned(&Xt_[k * NNt + t + u * VecLen]);
                }
            }

            const Real *__restrict__ src_ptr = r_src;
            const Real *__restrict__ charge_ptr = v_src;
            const Real *__restrict__ normal_ptr = src_normals;
            for (Long s = 0; s < Ns;
                 s++, src_ptr += SPATIAL_DIM, charge_ptr += KERNEL_INPUT_DIM, normal_ptr += NORMAL_DIM) {
                RealVec xs[SPATIAL_DIM], vs[KERNEL_INPUT_DIM], ns[NORMAL_DIM];
                for (Integer k = 0; k < SPATIAL_DIM; k++)
                    xs[k] = RealVec::Load1(&src_ptr[k]);
                for (Integer k = 0; k < NORMAL_DIM; k++)
                    ns[k] = RealVec::Load1(&normal_ptr[k]);
                for (Integer k = 0; k < KERNEL_INPUT_DIM; k++)
                    vs[k] = RealVec::Load1(&charge_ptr[k]);

                for (int u = 0; u < unroll_factor; u++) {
                    RealVec dX[SPATIAL_DIM];
                    RealVec U[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM];
                    for (Integer i = 0; i < SPATIAL_DIM; i++)
                        dX[i] = xt[u][i] - xs[i];
                    if constexpr (NORMAL_DIM > 0)
                        uKernel(U, dX, ns);
                    else
                        uKernel(U, dX);
                    for (Integer k0 = 0; k0 < KERNEL_INPUT_DIM; k0++)
                        for (Integer k1 = 0; k1 < KERNEL_OUTPUT_DIM; k1++)
                            vt[u][k1] = FMA(U[k0][k1], vs[k0], vt[u][k1]);
                }
            }

            for (int u = 0; u < unroll_factor; u++) {
                for (Integer k = 0; k < KERNEL_OUTPUT_DIM; k++) {
                    vt[u][k].StoreAligned(&Vt_[k * NNt + t + u * VecLen]);
                }
            }
        }

        // Remainder loop
        for (; t < NNt; t += VecLen) {
            RealVec xt[SPATIAL_DIM], vt[KERNEL_OUTPUT_DIM];

            for (Integer k = 0; k < KERNEL_OUTPUT_DIM; k++)
                vt[k] = RealVec::Zero();
            for (Integer k = 0; k < SPATIAL_DIM; k++)
                xt[k] = RealVec::LoadAligned(&Xt_[k * NNt + t]);

            for (Long s = 0; s < Ns; s++) {
                RealVec xs[SPATIAL_DIM], vs[KERNEL_INPUT_DIM], ns[NORMAL_DIM];

                for (Integer k = 0; k < SPATIAL_DIM; k++)
                    xs[k] = RealVec::Load1(&r_src[s * SPATIAL_DIM + k]);
                for (Integer k = 0; k < NORMAL_DIM; k++)
                    ns[k] = RealVec::Load1(&src_normals[s * NORMAL_DIM + k]);
                for (Integer k = 0; k < KERNEL_INPUT_DIM; k++)
                    vs[k] = RealVec::Load1(&v_src[s * KERNEL_INPUT_DIM + k]);

                RealVec dX_rem[SPATIAL_DIM], U_rem[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM];
                for (Integer i = 0; i < SPATIAL_DIM; i++)
                    dX_rem[i] = xt[i] - xs[i];
                if constexpr (NORMAL_DIM > 0)
                    uKernel(U_rem, dX_rem, ns);
                else
                    uKernel(U_rem, dX_rem);

                for (Integer k0 = 0; k0 < KERNEL_INPUT_DIM; k0++) {
                    for (Integer k1 = 0; k1 < KERNEL_OUTPUT_DIM; k1++) {
                        vt[k1] = FMA(U_rem[k0][k1], vs[k0], vt[k1]);
                    }
                }
            }

            for (Integer k = 0; k < KERNEL_OUTPUT_DIM; k++)
                vt[k].StoreAligned(&Vt_[k * NNt + t]);
        }

        for (Long k = 0; k < KERNEL_OUTPUT_DIM; k++) {
            for (Long i = 0; i < Nt; i++) {
                v_trg[i * KERNEL_OUTPUT_DIM + k] += Vt_[k * NNt + i] * scale_factor;
            }
        }
    }
}

template <class Real, int MaxVecLen>
struct LaplacePolyEvaluator2D {
    using VecType = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 2;

    VecType thresh2_vec, d2max_vec, rsc_vec, cen_vec, bsizeinv2_vec;
    const Real *coeffs;
    int n_coeffs;
    int n_digits;
    int eval_level;

    static constexpr Real scale_factor() { return Real{1.0}; }

    template <int KERNEL_OUTPUT_DIM>
    DMK_ALWAYS_INLINE void operator()(VecType (&u)[1][KERNEL_OUTPUT_DIM], const VecType (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KERNEL_OUTPUT_DIM == 3);

        const VecType R2 = dX[0] * dX[0] + dX[1] * dX[1];
        const auto mask = (R2 > thresh2_vec) & (R2 < d2max_vec);
        const VecType zero = VecType::Zero();
        const VecType half = Real{0.5};

        if constexpr (!has_grad) {
            const VecType R2sc = R2 * bsizeinv2_vec;
            const VecType ptmp = horner(R2, coeffs, n_coeffs);
            u[0][0] = select(mask, half * sctl::log(R2sc) + ptmp, zero);
        } else {
            const VecType R2sc = R2 * bsizeinv2_vec;
            VecType P, dP;
            horner_val_deriv(R2, coeffs, n_coeffs, P, dP);
            u[0][0] = select(mask, half * sctl::log(R2sc) + P, zero);

            // df/dR2 = 0.5/R2 + P'(R2)
            // Use approx_rsqrt to get Rinv, then R2inv = Rinv^2
            const VecType Rinv = my_approx_rsqrt(R2, n_digits);
            const VecType R2inv = Rinv * Rinv;
            const VecType df_dR2 = half * R2inv + dP;
            const VecType two = Real{2.0};
            for (int i = 0; i < 2; i++)
                u[0][1 + i] = select(mask, two * dX[i] * df_dR2, zero);
        }
    }
};

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS = -1, int EVAL_LEVEL = -1>
void laplace_2d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                               int n_coeffs_rt, const Real *coeffs, int n_src, const Real *r_src, const Real *charge,
                               int n_trg, const Real *r_trg, Real *pot, int unroll_factor) {
    constexpr bool is_static = (N_DIGITS > 0);
    const int n_digits = is_static ? N_DIGITS : n_digits_rt;
    const int n_coeffs = is_static ? N_COEFFS : n_coeffs_rt;
    const int eval_level = (EVAL_LEVEL > 0) ? EVAL_LEVEL : eval_level_rt;

    using VecType = sctl::Vec<Real, MaxVecLen>;
    constexpr int KERNEL_INPUT_DIM = 1;
    constexpr int SPATIAL_DIM = 2;
    constexpr int NORMAL_DIM = 0;

    std::array<Real, 64> coeffs_mod;
    shift_scale_polynomial(coeffs, rsc, cen, coeffs_mod.data(), n_coeffs);

    LaplacePolyEvaluator2D<Real, MaxVecLen> evaluator{
        VecType(thresh2),  VecType(d2max), VecType(rsc), VecType(cen), VecType(Real{0.5} * rsc),
        coeffs_mod.data(), n_coeffs,       n_digits,     eval_level};

    if (eval_level == 1) {
        EvalPairs<Real, KERNEL_INPUT_DIM, 1, SPATIAL_DIM, NORMAL_DIM, MaxVecLen>(
            n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor, n_digits);
    } else {
        EvalPairs<Real, KERNEL_INPUT_DIM, 3, SPATIAL_DIM, NORMAL_DIM, MaxVecLen>(
            n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor, n_digits);
    }
}

template <class Real, int MaxVecLen>
struct LaplacePolyEvaluator3D {
    using VecType = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 3;

    VecType thresh2_vec, d2max_vec, rsc_vec, cen_vec;
    const Real *coeffs;
    int n_coeffs;
    int n_digits;
    int transform_poly;
    int eval_level;

    static constexpr Real scale_factor() { return Real{1.0}; }

    template <int KDIM>
    DMK_ALWAYS_INLINE void operator()(VecType (&u)[1][KDIM], const VecType (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KDIM == 4);
        static_assert(KDIM == 1 || KDIM == 4, "Invalid KDIM");

        const VecType R2 = FMA(dX[0], dX[0], FMA(dX[1], dX[1], dX[2] * dX[2]));
        const auto in_range = (R2 > thresh2_vec) & (R2 < d2max_vec);
        const VecType Rinv = my_approx_rsqrt(R2, n_digits);
        const VecType zero = VecType::Zero();

        if (transform_poly) {
            if constexpr (!has_grad) {
                const VecType E = horner_split<0>(R2, coeffs, n_coeffs);
                const VecType O = horner_split<1>(R2, coeffs, n_coeffs);
                u[0][0] = sctl::select<Real, MaxVecLen>(in_range, FMA(E, Rinv, O), zero);
            } else {
                VecType E, dE, O, dO;
                horner_split_val_deriv<0>(R2, coeffs, n_coeffs, E, dE);
                horner_split_val_deriv<1>(R2, coeffs, n_coeffs, O, dO);

                u[0][0] = sctl::select<Real, MaxVecLen>(in_range, FMA(E, Rinv, O), zero);

                const VecType Rinv3 = Rinv * Rinv * Rinv;
                const VecType half = Real{0.5};
                const VecType df_dR2 = FMA(dE, Rinv, dO) - half * E * Rinv3;
                const VecType two = Real{2.0};
                for (int i = 0; i < 3; i++)
                    u[0][1 + i] = sctl::select<Real, MaxVecLen>(in_range, two * dX[i] * df_dR2, zero);
            }
        } else {
            const VecType xmapped = FMA(R2, Rinv, cen_vec) * rsc_vec;

            if constexpr (!has_grad) {
                const VecType P = horner(xmapped, coeffs, n_coeffs);
                u[0][0] = sctl::select<Real, MaxVecLen>(in_range, P * Rinv, zero);
            } else {
                VecType P, dP;
                horner_val_deriv(xmapped, coeffs, n_coeffs, P, dP);

                u[0][0] = sctl::select<Real, MaxVecLen>(in_range, P * Rinv, zero);

                const VecType Rinv2 = Rinv * Rinv;
                const VecType half = VecType(Real{0.5});
                const VecType df_dR2 = half * Rinv2 * (dP * rsc_vec - P * Rinv);
                const VecType two = VecType(Real{2.0});
                for (int i = 0; i < 3; i++)
                    u[0][1 + i] = sctl::select<Real, MaxVecLen>(in_range, two * dX[i] * df_dR2, zero);
            }
        }
    }
};

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS = -1, int EVAL_LEVEL = -1>
void laplace_3d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                               int n_coeffs_rt, const Real *coeffs, int n_src, const Real *r_src, const Real *charge,
                               int n_trg, const Real *r_trg, Real *pot, int unroll_factor) {
    constexpr bool is_static = (N_DIGITS > 0);
    const int n_digits = is_static ? N_DIGITS : n_digits_rt;
    const int n_coeffs = is_static ? N_COEFFS : n_coeffs_rt;
    const int eval_level = (EVAL_LEVEL > 0) ? EVAL_LEVEL : eval_level_rt;

    using VecType = sctl::Vec<Real, MaxVecLen>;
    constexpr int KERNEL_INPUT_DIM = 1;
    constexpr int SPATIAL_DIM = 3;
    constexpr int NORMAL_DIM = 0;
    const int transform_poly = n_digits < 6;

    Real coeffs_mod[64];
    if (transform_poly) {
        double rsc_pow = 1.0;
        double coeffs_mod_d[64];
        for (int i = 0; i < n_coeffs; ++i) {
            coeffs_mod_d[i] = coeffs[i] * rsc_pow;
            rsc_pow *= rsc;
        }
        for (int i = 0; i < n_coeffs; ++i)
            for (int j = n_coeffs - 1; j > i; --j)
                coeffs_mod_d[j - 1] += cen * coeffs_mod_d[j];
        for (int i = 0; i < n_coeffs; ++i)
            coeffs_mod[i] = coeffs_mod_d[i];
        coeffs = coeffs_mod;
    }

    LaplacePolyEvaluator3D<Real, MaxVecLen> evaluator{VecType(thresh2), VecType(d2max), VecType(rsc),
                                                      VecType(cen),     coeffs,         n_coeffs,
                                                      n_digits,         transform_poly, eval_level};

    if (eval_level == 1) {
        constexpr int KERNEL_OUTPUT_DIM = 1;
        EvalPairs<Real, KERNEL_INPUT_DIM, KERNEL_OUTPUT_DIM, SPATIAL_DIM, NORMAL_DIM, MaxVecLen>(
            n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor, n_digits);
    } else {
        constexpr int KERNEL_OUTPUT_DIM = 4;
        EvalPairs<Real, KERNEL_INPUT_DIM, KERNEL_OUTPUT_DIM, SPATIAL_DIM, NORMAL_DIM, MaxVecLen>(
            n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor, n_digits);
    }
}

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS = -1, int EVAL_LEVEL = -1>
void sqrt_laplace_2d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                                    int n_coeffs_rt, const Real *coeffs, int n_src, const Real *r_src,
                                    const Real *charge, int n_trg, const Real *r_trg, Real *pot, int unroll_factor) {
    constexpr bool is_static = (N_DIGITS > 0);
    const int n_digits = is_static ? N_DIGITS : n_digits_rt;
    const int n_coeffs = is_static ? N_COEFFS : n_coeffs_rt;

    using VecType = sctl::Vec<Real, MaxVecLen>;
    constexpr int KERNEL_INPUT_DIM = 1;
    constexpr int KERNEL_OUTPUT_DIM = 1;
    constexpr int SPATIAL_DIM = 2;
    constexpr int NORMAL_DIM = 0;

    struct Evaluator {
        VecType thresh2_vec, d2max_vec, rsc_vec, cen_vec;
        const Real *coeffs;
        int n_coeffs;
        int n_digits;

        static constexpr Real scale_factor() { return Real{1.0}; }

        Evaluator(VecType thresh2_vec, VecType d2max_vec, VecType rsc_vec, VecType cen_vec, const Real *coeffs,
                  int n_coeffs, int digits)
            : thresh2_vec(thresh2_vec), d2max_vec(d2max_vec), rsc_vec(rsc_vec), cen_vec(cen_vec), coeffs(coeffs),
              n_coeffs(n_coeffs), n_digits(digits) {}

        DMK_ALWAYS_INLINE void operator()(VecType (&u)[1][1], const VecType (&dX)[SPATIAL_DIM]) const {
            const VecType R2 = FMA(dX[0], dX[0], dX[1] * dX[1]);
            const auto mask = (R2 > thresh2_vec) & (R2 < d2max_vec);
            const VecType Rinv = my_approx_rsqrt<N_DIGITS>(R2, n_digits);
            const VecType x = sctl::FMA(R2, Rinv, cen_vec) * rsc_vec;
            u[0][0] = sctl::select(mask, horner(x, coeffs, n_coeffs) * Rinv, VecType::Zero());
        }
    };

    Evaluator evaluator(thresh2, d2max, rsc, cen, coeffs, n_coeffs, n_digits);

    EvalPairs<Real, KERNEL_INPUT_DIM, KERNEL_OUTPUT_DIM, SPATIAL_DIM, NORMAL_DIM, MaxVecLen>(
        n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor, n_digits);
}

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS = -1, int EVAL_LEVEL = -1>
void sqrt_laplace_3d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                                    int n_coeffs_rt, const Real *coeffs, int n_src, const Real *r_src,
                                    const Real *charge, int n_trg, const Real *r_trg, Real *pot, int unroll_factor) {
    constexpr bool is_static = (N_DIGITS > 0);
    const int n_digits = is_static ? N_DIGITS : n_digits_rt;
    const int n_coeffs = is_static ? N_COEFFS : n_coeffs_rt;

    using VecType = sctl::Vec<Real, MaxVecLen>;
    constexpr int KERNEL_INPUT_DIM = 1;
    constexpr int KERNEL_OUTPUT_DIM = 1;
    constexpr int SPATIAL_DIM = 3;
    constexpr int NORMAL_DIM = 0;

    struct Evaluator {
        VecType thresh2_vec, d2max_vec, rsc_vec, cen_vec;
        const Real *coeffs;
        int n_coeffs;
        int n_digits;

        static constexpr Real scale_factor() { return Real{1.0}; }

        Evaluator(VecType thresh2_vec, VecType d2max_vec, VecType rsc_vec, VecType cen_vec, const Real *coeffs,
                  int n_coeffs, int digits)
            : thresh2_vec(thresh2_vec), d2max_vec(d2max_vec), rsc_vec(rsc_vec), cen_vec(cen_vec), coeffs(coeffs),
              n_coeffs(n_coeffs), n_digits(digits) {}

        DMK_ALWAYS_INLINE void operator()(VecType (&u)[1][1], const VecType (&dX)[SPATIAL_DIM]) const {
            const VecType R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
            const auto mask = (R2 > thresh2_vec) & (R2 < d2max_vec);
            const VecType Rinv = my_approx_rsqrt<N_DIGITS>(R2, n_digits);
            const VecType R2inv = Rinv * Rinv;
            u[0][0] = sctl::select(mask, R2inv * horner(R2, coeffs, n_coeffs), VecType::Zero());
        }
    };

    std::array<Real, 64> coeffs_mod;
    shift_scale_polynomial(coeffs, rsc, cen, coeffs_mod.data(), n_coeffs);

    Evaluator evaluator(thresh2, d2max, rsc, cen, coeffs_mod.data(), n_coeffs, n_digits);

    EvalPairs<Real, KERNEL_INPUT_DIM, KERNEL_OUTPUT_DIM, SPATIAL_DIM, NORMAL_DIM, MaxVecLen>(
        n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor, n_digits);
}

#undef DMK_ALWAYS_INLINE

#endif
