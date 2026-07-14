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
#include <stdexcept>

#define DMK_ALWAYS_INLINE __attribute__((always_inline)) inline

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

// AoT: don't force to reason about dispatch tables/inlining. just use the rsqrt
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

// SIMD port of the scalar dmk::bessel::k0: the same two-branch rational
// approximation and coefficient tables, with both branches evaluated for all
// lanes and blended with select on the x<=1 mask. evalpoly maps to horner (both
// ascending-coefficient). Assumes x > 0; callers must mask out r==0 lanes.
template <typename VecType>
DMK_ALWAYS_INLINE VecType vec_bessel_k0(const VecType &x) {
    using Real = typename VecType::ScalarType;

    const Real *P1, *Q1, *P2, *P3, *Q3;
    int nP1, nQ1, nP2, nP3, nQ3;
    Real Y;
    if constexpr (std::is_same_v<Real, double>) {
        using namespace dmk::bessel::detail;
        P1 = P1_k0_d, nP1 = int(sizeof(P1_k0_d) / sizeof(double));
        Q1 = Q1_k0_d, nQ1 = int(sizeof(Q1_k0_d) / sizeof(double));
        P2 = P2_k0_d, nP2 = int(sizeof(P2_k0_d) / sizeof(double));
        P3 = P3_k0_d, nP3 = int(sizeof(P3_k0_d) / sizeof(double));
        Q3 = Q3_k0_d, nQ3 = int(sizeof(Q3_k0_d) / sizeof(double));
        Y = Y_k0_d;
    } else {
        using namespace dmk::bessel::detail;
        P1 = P1_k0_f, nP1 = int(sizeof(P1_k0_f) / sizeof(float));
        Q1 = Q1_k0_f, nQ1 = int(sizeof(Q1_k0_f) / sizeof(float));
        P2 = P2_k0_f, nP2 = int(sizeof(P2_k0_f) / sizeof(float));
        P3 = P3_k0_f, nP3 = int(sizeof(P3_k0_f) / sizeof(float));
        Q3 = Q3_k0_f, nQ3 = int(sizeof(Q3_k0_f) / sizeof(float));
        Y = Y_k0_f;
    }

    const VecType x2 = x * x;

    // small-argument branch (x <= 1): -a*log(x) + P2(x^2)
    const VecType a_s = x2 * Real{0.25};
    const VecType s0 = horner(a_s, P1, nP1) / horner(a_s, Q1, nQ1) + VecType(Y);
    const VecType a1 = FMA(s0, a_s, VecType(Real{1}));
    const VecType small = FMA(-a1, sctl::log(x), horner(x2, P2, nP2));

    // large-argument branch (x > 1): (P3(1/x)/Q3(1/x) + 1) * s^2 / sqrt(x), s = exp(-x/2)
    const VecType xinv = VecType(Real{1}) / x;
    const VecType s = sctl::exp(x * Real{-0.5});
    const VecType rsx = sctl::approx_rsqrt<-1>(x);
    const VecType large = (horner(xinv, P3, nP3) / horner(xinv, Q3, nQ3) + VecType(Real{1})) * s * rsx * s;

    return sctl::select(x <= VecType(Real{1}), small, large);
}

// SIMD port of the scalar dmk::bessel::k1; see vec_bessel_k0 for the structure.
// Assumes x > 0; callers must mask out r==0 lanes.
template <typename VecType>
DMK_ALWAYS_INLINE VecType vec_bessel_k1(const VecType &x) {
    using Real = typename VecType::ScalarType;

    const Real *P1, *Q1, *P2, *Q2, *P3, *Q3;
    int nP1, nQ1, nP2, nQ2, nP3, nQ3;
    Real Y, Y2;
    if constexpr (std::is_same_v<Real, double>) {
        using namespace dmk::bessel::detail;
        P1 = P1_k1_d, nP1 = int(sizeof(P1_k1_d) / sizeof(double));
        Q1 = Q1_k1_d, nQ1 = int(sizeof(Q1_k1_d) / sizeof(double));
        P2 = P2_k1_d, nP2 = int(sizeof(P2_k1_d) / sizeof(double));
        Q2 = Q2_k1_d, nQ2 = int(sizeof(Q2_k1_d) / sizeof(double));
        P3 = P3_k1_d, nP3 = int(sizeof(P3_k1_d) / sizeof(double));
        Q3 = Q3_k1_d, nQ3 = int(sizeof(Q3_k1_d) / sizeof(double));
        Y = Y_k1_d, Y2 = Y2_k1_d;
    } else {
        using namespace dmk::bessel::detail;
        P1 = P1_k1_f, nP1 = int(sizeof(P1_k1_f) / sizeof(float));
        Q1 = Q1_k1_f, nQ1 = int(sizeof(Q1_k1_f) / sizeof(float));
        P2 = P2_k1_f, nP2 = int(sizeof(P2_k1_f) / sizeof(float));
        Q2 = Q2_k1_f, nQ2 = int(sizeof(Q2_k1_f) / sizeof(float));
        P3 = P3_k1_f, nP3 = int(sizeof(P3_k1_f) / sizeof(float));
        Q3 = Q3_k1_f, nQ3 = int(sizeof(Q3_k1_f) / sizeof(float));
        Y = Y_k1_f, Y2 = Y2_k1_f;
    }

    const VecType x2 = x * x;
    const VecType xinv = VecType(Real{1}) / x;

    // small-argument branch (x <= 1): aa*log(x) + (P2(x^2)/Q2(x^2))*x + 1/x
    const VecType a_s = x2 * Real{0.25};
    VecType pq = horner(a_s, P1, nP1) / horner(a_s, Q1, nQ1) + VecType(Y);
    pq = FMA(pq * a_s, a_s, FMA(a_s, VecType(Real{0.5}), VecType(Real{1}))); // pq*a^2 + a/2 + 1
    const VecType aa = pq * x * Real{0.5};
    const VecType t = FMA(horner(x2, P2, nP2) / horner(x2, Q2, nQ2), x, xinv);
    const VecType small = FMA(aa, sctl::log(x), t);

    // large-argument branch (x > 1): (P3(1/x)/Q3(1/x) + Y2) * s^2 / sqrt(x), s = exp(-x/2)
    const VecType s = sctl::exp(x * Real{-0.5});
    const VecType rsx = sctl::approx_rsqrt<-1>(x);
    const VecType large = (horner(xinv, P3, nP3) / horner(xinv, Q3, nQ3) + VecType(Y2)) * s * rsx * s;

    return sctl::select(x <= VecType(Real{1}), small, large);
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

// Simultaneous horner, derivative, and second derivative.
// Recurrence: v_k = c_k + x v_{k+1}, dv_k = v_{k+1} + x dv_{k+1}, d2v_k = 2 dv_{k+1} + x d2v_{k+1}.
template <typename VecType>
DMK_ALWAYS_INLINE void horner_val_deriv2(const VecType &x, const typename VecType::ScalarType *coeffs, int n_coeffs,
                                         VecType &value, VecType &derivative, VecType &derivative2) {
    value = VecType::Load1(&coeffs[n_coeffs - 1]);
    derivative = VecType::Zero();
    derivative2 = VecType::Zero();
    for (int i = n_coeffs - 2; i >= 0; --i) {
        derivative2 = FMA(derivative2, x, derivative + derivative);
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

// clang-format off
template <typename E>
concept KernelEvaluator = requires {
    typename E::scalar_type;
    typename E::vector_type;
    { typename E::scalar_type(E::scale_factor) };
    { int(E::KERNEL_INPUT_DIM) };
    { int(E::SPATIAL_DIM) };
    { int(E::NORMAL_DIM) };
};
// clang-format on

template <int KERNEL_OUTPUT_DIM, KernelEvaluator uKernelEvaluator>
DMK_ALWAYS_INLINE void EvalPairs(int Ns, const typename uKernelEvaluator::scalar_type *__restrict__ r_src,
                                 const typename uKernelEvaluator::scalar_type *__restrict__ v_src,
                                 const typename uKernelEvaluator::scalar_type *__restrict__ src_normals, int Nt,
                                 const typename uKernelEvaluator::scalar_type *__restrict__ r_trg,
                                 typename uKernelEvaluator::scalar_type *__restrict__ v_trg, uKernelEvaluator uKernel,
                                 int unroll_factor) {
    using namespace sctl;
    using Real = typename uKernelEvaluator::scalar_type;
    using RealVec = typename uKernelEvaluator::vector_type;
    constexpr int KERNEL_INPUT_DIM = uKernelEvaluator::KERNEL_INPUT_DIM;
    constexpr int SPATIAL_DIM = uKernelEvaluator::SPATIAL_DIM;
    constexpr int NORMAL_DIM = uKernelEvaluator::NORMAL_DIM;
    constexpr int VecLen = RealVec::Size();
    constexpr Real scale_factor = uKernelEvaluator::scale_factor;

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

// Range-list variant of EvalPairs.  The only difference is the source loop:
// instead of iterating s = 0..Ns-1 contiguously, we iterate over disjoint
// ranges [range_starts[r], range_starts[r] + range_lens[r]).  Everything
// else (target loading, SIMD unrolling, accumulation, output) is identical.
template <int KERNEL_OUTPUT_DIM, KernelEvaluator uKernelEvaluator>
DMK_ALWAYS_INLINE void EvalPairsRanges(int, const typename uKernelEvaluator::scalar_type *__restrict__ r_src,
                                       const typename uKernelEvaluator::scalar_type *__restrict__ v_src,
                                       const typename uKernelEvaluator::scalar_type *__restrict__ src_normals,
                                       int n_ranges, const int *__restrict__ range_starts,
                                       const int *__restrict__ range_lens, int Nt,
                                       const typename uKernelEvaluator::scalar_type *__restrict__ r_trg,
                                       typename uKernelEvaluator::scalar_type *__restrict__ v_trg,
                                       uKernelEvaluator uKernel, int unroll_factor) {
    using namespace sctl;
    using Real = typename uKernelEvaluator::scalar_type;
    using RealVec = typename uKernelEvaluator::vector_type;
    constexpr int KERNEL_INPUT_DIM = uKernelEvaluator::KERNEL_INPUT_DIM;
    constexpr int SPATIAL_DIM = uKernelEvaluator::SPATIAL_DIM;
    constexpr int NORMAL_DIM = uKernelEvaluator::NORMAL_DIM;
    constexpr int VecLen = RealVec::Size();
    constexpr Real scale_factor = uKernelEvaluator::scale_factor;

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
        for (int r = 0; r < n_ranges; r++) {
            const int s0 = range_starts[r];
            const int sn = range_lens[r];
            for (int s = 0; s < sn; s++) {
                for (Integer k = 0; k < SPATIAL_DIM; k++)
                    xs[k] = RealVec::Load1(&r_src[(s0 + s) * SPATIAL_DIM + k]);
                for (Integer k = 0; k < NORMAL_DIM; k++)
                    ns[k] = RealVec::Load1(&src_normals[(s0 + s) * NORMAL_DIM + k]);
                for (Integer k = 0; k < KERNEL_INPUT_DIM; k++)
                    vs[k] = RealVec::Load1(&v_src[(s0 + s) * KERNEL_INPUT_DIM + k]);

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
        }
        for (Integer k = 0; k < KERNEL_OUTPUT_DIM; k++) {
            alignas(sizeof(RealVec)) std::array<Real, VecLen> out;
            vt[k].StoreAligned(&out[0]);
            for (Long t = 0; t < Nt; t++) {
                v_trg[t * KERNEL_OUTPUT_DIM + k] += out[t] * scale_factor;
            }
        }
    } else {
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

            for (int r = 0; r < n_ranges; r++) {
                const Real *__restrict__ src_ptr = r_src + range_starts[r] * SPATIAL_DIM;
                const Real *__restrict__ charge_ptr = v_src + range_starts[r] * KERNEL_INPUT_DIM;
                const Real *__restrict__ normal_ptr = src_normals + range_starts[r] * NORMAL_DIM;
                for (int s = 0; s < range_lens[r];
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

            for (int r = 0; r < n_ranges; r++) {
                const int s0 = range_starts[r];
                const int sn = range_lens[r];
                for (int s = 0; s < sn; s++) {
                    RealVec xs[SPATIAL_DIM], vs[KERNEL_INPUT_DIM], ns[NORMAL_DIM];

                    for (Integer k = 0; k < SPATIAL_DIM; k++)
                        xs[k] = RealVec::Load1(&r_src[(s0 + s) * SPATIAL_DIM + k]);
                    for (Integer k = 0; k < NORMAL_DIM; k++)
                        ns[k] = RealVec::Load1(&src_normals[(s0 + s) * NORMAL_DIM + k]);
                    for (Integer k = 0; k < KERNEL_INPUT_DIM; k++)
                        vs[k] = RealVec::Load1(&v_src[(s0 + s) * KERNEL_INPUT_DIM + k]);

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

template <class Real, int MaxVecLen, int EVAL_LEVEL = DMK_POTENTIAL>
struct YukawaEvaluator3D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == DMK_POTENTIAL ? 1 : 4;

    vector_type lambda;

    template <int KDIM>
    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[1][KDIM], const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KDIM == 4);
        const vector_type R2 = FMA(dX[0], dX[0], FMA(dX[1], dX[1], dX[2] * dX[2]));
        const auto mask = R2 > vector_type::Zero();
        const vector_type Rinv = sctl::approx_rsqrt<-1>(R2, mask);
        const vector_type R = Rinv * R2;
        const vector_type E = sctl::exp(-lambda * R);

        u[0][0] = Rinv * E;
        if constexpr (has_grad) {
            // grad_i = -dX_i * E * Rinv^2 * (lambda + Rinv); Rinv=0 on masked lanes -> 0
            const vector_type g = -E * Rinv * Rinv * (lambda + Rinv);
            for (int i = 0; i < 3; i++)
                u[0][1 + i] = dX[i] * g;
        }
    }
};

// 2D Yukawa (modified Helmholtz) direct evaluator: phi = K0(lambda*r), the 2D
// analog of YukawaEvaluator3D's bare exp(-lambda*r)/r (no 1/2pi prefactor). K0/K1
// are evaluated in SIMD via vec_bessel_k0/k1; the r==0 self-interaction is masked.
template <class Real, int MaxVecLen, int EVAL_LEVEL = DMK_POTENTIAL>
struct YukawaEvaluator2D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == DMK_POTENTIAL ? 1 : 3;

    vector_type lambda;

    template <int KDIM>
    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[1][KDIM], const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KDIM == 3);
        const vector_type R2 = FMA(dX[0], dX[0], dX[1] * dX[1]);
        const auto mask = R2 > vector_type::Zero();
        const vector_type Rinv = sctl::approx_rsqrt<-1>(R2, mask);
        const vector_type R = R2 * Rinv;
        const vector_type arg = lambda * R;
        u[0][0] = sctl::select(mask, vec_bessel_k0(arg), vector_type::Zero());
        if constexpr (has_grad) {
            // grad_i = -dX_i * lambda * K1(lambda*r) * Rinv; select drops the r==0 NaN
            const vector_type g = -lambda * vec_bessel_k1(arg) * Rinv;
            for (int i = 0; i < 2; i++)
                u[0][1 + i] = sctl::select(mask, dX[i] * g, vector_type::Zero());
        }
    }
};

template <class Real, int MaxVecLen, int EVAL_LEVEL>
struct LaplaceEvaluator2D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == DMK_POTENTIAL ? 1 : 3;

    template <int KERNEL_OUTPUT_DIM>
    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[1][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KERNEL_OUTPUT_DIM == 3);
        const vector_type R2 = FMA(dX[0], dX[0], dX[1] * dX[1]);
        const auto mask = R2 > vector_type::Zero();
        const vector_type half = Real{0.5};

        u[0][0] = select(mask, half * sctl::log(R2), vector_type::Zero());
        if constexpr (has_grad) {
            const vector_type Rinv = sctl::approx_rsqrt<-1>(R2, mask);
            const vector_type Rinv2 = Rinv * Rinv;
            for (int i = 0; i < 2; i++)
                u[0][1 + i] = dX[i] * Rinv2;
        }
    }
};

template <class Real, int MaxVecLen, int EVAL_LEVEL>
struct LaplaceEvaluator3D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == DMK_POTENTIAL ? 1 : 4;

    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[1][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KERNEL_OUTPUT_DIM == 4);
        const vector_type R2 = FMA(dX[0], dX[0], FMA(dX[1], dX[1], dX[2] * dX[2]));
        const auto mask = R2 > vector_type::Zero();
        const vector_type Rinv = sctl::approx_rsqrt<-1>(R2, mask);

        u[0][0] = Rinv;
        if constexpr (has_grad) {
            const vector_type Rinv3 = Rinv * Rinv * Rinv;
            for (int i = 0; i < 3; i++)
                u[0][1 + i] = -dX[i] * Rinv3;
        }
    }
};

template <class Real, int MaxVecLen>
struct LaplacePolyEvaluator2D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;

    vector_type thresh2_vec, d2max_vec, rsc_vec, cen_vec, bsizeinv2_vec;
    const Real *coeffs;
    int n_coeffs;
    int n_digits;
    int eval_level;

    template <int KERNEL_OUTPUT_DIM>
    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[1][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KERNEL_OUTPUT_DIM == 3);
        const vector_type R2 = FMA(dX[0], dX[0], dX[1] * dX[1]);
        const auto mask = (R2 > thresh2_vec) & (R2 < d2max_vec);
        const vector_type zero = vector_type::Zero();
        const vector_type half = Real{0.5};

        // Poly is in the scaled variable x = rsc*R2 + cen (avoids forming rsc^k).
        const vector_type x = FMA(R2, rsc_vec, cen_vec);

        if constexpr (!has_grad) {
            const vector_type R2sc = R2 * bsizeinv2_vec;
            const vector_type ptmp = horner(x, coeffs, n_coeffs);
            u[0][0] = select(mask, half * sctl::log(R2sc) + ptmp, zero);
        } else {
            const vector_type R2sc = R2 * bsizeinv2_vec;
            vector_type P, dP;
            horner_val_deriv(x, coeffs, n_coeffs, P, dP);
            dP = dP * rsc_vec; // chain rule: d/dR2 = rsc * d/dx
            u[0][0] = select(mask, half * sctl::log(R2sc) + P, zero);

            // df/dR2 = 0.5/R2 + P'(R2)
            // Use approx_rsqrt to get Rinv, then R2inv = Rinv^2
            const vector_type Rinv = my_approx_rsqrt(R2, n_digits);
            const vector_type R2inv = Rinv * Rinv;
            const vector_type df_dR2 = half * R2inv + dP;
            const vector_type two = Real{2.0};
            for (int i = 0; i < 2; i++)
                u[0][1 + i] = select(mask, two * dX[i] * df_dR2, zero);
        }
    }
};

// Yukawa The residual K0(lambda*r) + C(r) is a log-singular even function of r, so it is fit per
// level as residual = 0.5*log(r^2/bsize^2) * PA(x) + PB(x), where PA(x) ~ -I0(lambda*r)
// carries the log coefficient and PB is the smooth remainder (the windowed
// far-field C(r) and constant shifts fold into PB); both are polynomials in the
// scaled variable x = rsc*r^2 + cen.
template <class Real, int MaxVecLen>
struct YukawaPolyEvaluator2D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;

    vector_type thresh2_vec, d2max_vec, rsc_vec, cen_vec, bsizeinv2_vec;
    const Real *coeffs_log;
    int n_coeffs_log;
    const Real *coeffs_reg;
    int n_coeffs_reg;
    int n_digits;

    template <int KERNEL_OUTPUT_DIM>
    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[1][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KERNEL_OUTPUT_DIM == 3);
        static_assert(KERNEL_OUTPUT_DIM == 1 || KERNEL_OUTPUT_DIM == 3, "Invalid KDIM");
        const vector_type R2 = FMA(dX[0], dX[0], dX[1] * dX[1]);
        const auto in_range = (R2 > thresh2_vec) & (R2 < d2max_vec);
        const vector_type half = Real{0.5};

        const vector_type x = FMA(R2, rsc_vec, cen_vec);
        const vector_type L = half * sctl::log(R2 * bsizeinv2_vec);
        if constexpr (!has_grad) {
            const vector_type PA = horner(x, coeffs_log, n_coeffs_log);
            const vector_type PB = horner(x, coeffs_reg, n_coeffs_reg);
            u[0][0] = sctl::select<Real, MaxVecLen>(in_range, FMA(L, PA, PB), vector_type::Zero());
        } else {
            vector_type PA, dPA, PB, dPB;
            horner_val_deriv(x, coeffs_log, n_coeffs_log, PA, dPA); // d/dx
            horner_val_deriv(x, coeffs_reg, n_coeffs_reg, PB, dPB);
            u[0][0] = sctl::select<Real, MaxVecLen>(in_range, FMA(L, PA, PB), vector_type::Zero());

            // f = L*PA + PB, L = 0.5*log(R2*bsizeinv2), x = rsc*R2 + cen.
            // df/dR2 = 0.5/R2*PA + rsc*(L*dPA + dPB); grad_i = 2*dX_i*df/dR2.
            const vector_type Rinv = my_approx_rsqrt(R2, n_digits);
            const vector_type R2inv = Rinv * Rinv;
            const vector_type df_dR2 = FMA(half * R2inv, PA, rsc_vec * FMA(L, dPA, dPB));
            const vector_type two = Real{2.0};
            for (int i = 0; i < 2; i++)
                u[0][1 + i] = sctl::select<Real, MaxVecLen>(in_range, two * dX[i] * df_dR2, vector_type::Zero());
        }
    }
};

template <class Real, int MaxVecLen>
struct LaplacePolyEvaluator3D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;

    vector_type thresh2_vec, d2max_vec, rsc_vec, cen_vec;
    const Real *coeffs;
    int n_coeffs;
    int n_digits;
    int transform_poly;
    int eval_level;

    template <int KERNEL_OUTPUT_DIM>
    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[1][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KERNEL_OUTPUT_DIM == 4);
        static_assert(KERNEL_OUTPUT_DIM == 1 || KERNEL_OUTPUT_DIM == 4, "Invalid KDIM");

        const vector_type R2 = FMA(dX[0], dX[0], FMA(dX[1], dX[1], dX[2] * dX[2]));
        const auto in_range = (R2 > thresh2_vec) & (R2 < d2max_vec);
        const vector_type Rinv = my_approx_rsqrt(R2, n_digits);
        const vector_type zero = vector_type::Zero();

        if (transform_poly) {
            if constexpr (!has_grad) {
                const vector_type E = horner_split<0>(R2, coeffs, n_coeffs);
                const vector_type O = horner_split<1>(R2, coeffs, n_coeffs);
                u[0][0] = sctl::select<Real, MaxVecLen>(in_range, FMA(E, Rinv, O), zero);
            } else {
                vector_type E, dE, O, dO;
                horner_split_val_deriv<0>(R2, coeffs, n_coeffs, E, dE);
                horner_split_val_deriv<1>(R2, coeffs, n_coeffs, O, dO);

                u[0][0] = sctl::select<Real, MaxVecLen>(in_range, FMA(E, Rinv, O), zero);

                const vector_type df_dR2 = Real{2.0} * FMA(dE, Rinv, dO) - E * Rinv * Rinv * Rinv;
                for (int i = 0; i < 3; i++)
                    u[0][1 + i] = sctl::select<Real, MaxVecLen>(in_range, dX[i] * df_dR2, zero);
            }
        } else {
            const vector_type xmapped = FMA(R2, Rinv, cen_vec) * rsc_vec;

            if constexpr (!has_grad) {
                const vector_type P = horner(xmapped, coeffs, n_coeffs);
                u[0][0] = sctl::select<Real, MaxVecLen>(in_range, P * Rinv, zero);
            } else {
                vector_type P, dP;
                horner_val_deriv(xmapped, coeffs, n_coeffs, P, dP);

                u[0][0] = sctl::select<Real, MaxVecLen>(in_range, P * Rinv, zero);
                const vector_type df_dR2 = Rinv * Rinv * (dP * rsc_vec - P * Rinv);
                for (int i = 0; i < 3; i++)
                    u[0][1 + i] = sctl::select<Real, MaxVecLen>(in_range, dX[i] * df_dR2, zero);
            }
        }
    }
};

// Yukawa 3D short-range residual, unified onto the same polynomial path as the
// other kernels. The residual exp(-lambda*r)/r - W(r) is fit per level as
// Q(r) = r*residual (a smooth, bounded polynomial in the scaled variable
// x = r*rsc + cen); the 1/r singularity is supplied by Rinv at eval time, and
// the exp is folded into the fit (no runtime exp call). Mirrors the Laplace 3D
// non-transform branch (P*Rinv), including the gradient.
template <class Real, int MaxVecLen>
struct YukawaPolyEvaluator3D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;

    vector_type thresh2_vec, d2max_vec, rsc_vec, cen_vec;
    const Real *coeffs;
    int n_coeffs;
    int n_digits;

    template <int KERNEL_OUTPUT_DIM>
    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[1][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KERNEL_OUTPUT_DIM == 4);
        static_assert(KERNEL_OUTPUT_DIM == 1 || KERNEL_OUTPUT_DIM == 4, "Invalid KDIM");

        const vector_type R2 = FMA(dX[0], dX[0], FMA(dX[1], dX[1], dX[2] * dX[2]));
        const auto in_range = (R2 > thresh2_vec) & (R2 < d2max_vec);
        const vector_type Rinv = my_approx_rsqrt(R2, n_digits);
        const vector_type R = R2 * Rinv; // = r

        const vector_type x = FMA(R, rsc_vec, cen_vec); // r*rsc + cen, matching the fit
        if constexpr (!has_grad) {
            const vector_type Q = horner(x, coeffs, n_coeffs);
            u[0][0] = sctl::select<Real, MaxVecLen>(in_range, Q * Rinv, vector_type::Zero());
        } else {
            vector_type Q, dQ;
            horner_val_deriv(x, coeffs, n_coeffs, Q, dQ); // dQ = dQ/dx

            u[0][0] = sctl::select<Real, MaxVecLen>(in_range, Q * Rinv, vector_type::Zero());
            // f = Q*Rinv, x = r*rsc + cen; grad_i = dX_i * Rinv^2 * (dQ*rsc - Q*Rinv)
            const vector_type df = Rinv * Rinv * (dQ * rsc_vec - Q * Rinv);
            for (int i = 0; i < 3; i++)
                u[0][1 + i] = sctl::select<Real, MaxVecLen>(in_range, dX[i] * df, vector_type::Zero());
        }
    }
};

template <typename Real, int MaxVecLen, dmk_eval_type EVAL_LEVEL = DMK_POTENTIAL>
struct LaplaceDipoleEvaluator2D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 2;
    static constexpr int NORMAL_DIM = 0;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == DMK_POTENTIAL ? 1 : 3;
    static constexpr Real scale_factor = 1.0;

    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KERNEL_OUTPUT_DIM > 1);
        const vector_type R2 = FMA(dX[0], dX[0], dX[1] * dX[1]);
        const auto mask = R2 > vector_type::Zero();
        const vector_type Rinv = sctl::approx_rsqrt<-1>(R2, mask);
        const vector_type Rinv2 = Rinv * Rinv;
        // pot per-component: U[k][0] = -dX[k] * Rinv^2  (gives phi = -(d.dX)/R^2)
        for (int k = 0; k < 2; k++)
            u[k][0] = -dX[k] * Rinv2;
        if constexpr (has_grad) {
            // grad[i] = -d_i / R^2 + 2 (d.dX) dX_i / R^4
            // U[k][1+i] = -delta_{ki} Rinv^2 + 2 dX[k] dX[i] Rinv^4
            const vector_type Rinv4 = Rinv2 * Rinv2;
            for (int k = 0; k < 2; k++) {
                for (int i = 0; i < 2; i++) {
                    vector_type val = Real{2} * dX[k] * dX[i] * Rinv4;
                    if (i == k)
                        val = val - Rinv2;
                    u[k][1 + i] = val;
                }
            }
        }
    }
};

template <typename Real, int MaxVecLen, dmk_eval_type EVAL_LEVEL = DMK_POTENTIAL>
struct LaplaceDipoleEvaluator3D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int NORMAL_DIM = 0;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == DMK_POTENTIAL ? 1 : 4;
    static constexpr Real scale_factor = 1.0;

    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KERNEL_OUTPUT_DIM > 1);
        const vector_type R2 = FMA(dX[0], dX[0], FMA(dX[1], dX[1], dX[2] * dX[2]));
        const auto mask = R2 > vector_type::Zero();
        const vector_type Rinv = sctl::approx_rsqrt<-1>(R2, mask);
        const vector_type Rinv3 = Rinv * Rinv * Rinv;
        // pot per-component: U[k][0] = dX[k] * Rinv^3  (gives phi = (d.dX)/R^3)
        for (int k = 0; k < 3; k++)
            u[k][0] = dX[k] * Rinv3;
        if constexpr (has_grad) {
            // grad[i] = d_i / R^3 - 3 (d.dX) dX_i / R^5
            // U[k][1+i] = delta_{ki} Rinv^3 - 3 dX[k] dX[i] Rinv^5
            const vector_type Rinv5 = Rinv3 * Rinv * Rinv;
            for (int k = 0; k < 3; k++) {
                for (int i = 0; i < 3; i++) {
                    vector_type val = -Real{3} * dX[k] * dX[i] * Rinv5;
                    if (i == k)
                        val = val + Rinv3;
                    u[k][1 + i] = val;
                }
            }
        }
    }
};

template <class Real, int MaxVecLen>
struct LaplaceDipolePolyEvaluator2D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 2;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;

    vector_type thresh2_vec, d2max_vec, rsc_vec, cen_vec;
    const Real *coeffs;
    int n_coeffs;
    int n_digits;

    template <int KERNEL_OUTPUT_DIM>
    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KERNEL_OUTPUT_DIM > 1);
        static_assert(KERNEL_OUTPUT_DIM == 1 || KERNEL_OUTPUT_DIM == 3);

        const vector_type R2 = FMA(dX[0], dX[0], dX[1] * dX[1]);
        const auto mask = (R2 > thresh2_vec) & (R2 < d2max_vec);
        const vector_type zero = vector_type::Zero();
        const vector_type Rinv = my_approx_rsqrt(R2, n_digits);
        const vector_type R2inv = Rinv * Rinv;
        const vector_type x = FMA(R2, rsc_vec, cen_vec);

        if constexpr (!has_grad) {
            // F(R^2) = -(R^-2 + 2 dP).  U[k][0] = dX[k] * F.
            vector_type P, dP;
            horner_val_deriv(x, coeffs, n_coeffs, P, dP);
            (void)P;
            dP = dP * rsc_vec; // chain rule: d/dR2 = rsc * d/dx
            const vector_type F = -(R2inv + Real{2} * dP);
            for (int k = 0; k < 2; k++)
                u[k][0] = sctl::select(mask, dX[k] * F, zero);
        } else {
            // F(R^2)  = -(R^-2 + 2 dP)
            // F'(R^2) =  R^-4 - 2 ddP
            // U[k][0]   = dX[k] * F
            // U[k][1+i] = delta_{ki} * F + 2 * dX[k] * dX[i] * F'
            vector_type P, dP, ddP;
            horner_val_deriv2(x, coeffs, n_coeffs, P, dP, ddP);
            dP = dP * rsc_vec;
            ddP = ddP * rsc_vec * rsc_vec;
            const vector_type R4inv = R2inv * R2inv;
            const vector_type F = -(R2inv + Real{2} * dP);
            const vector_type Fprime = R4inv - Real{2} * ddP;
            for (int k = 0; k < 2; k++) {
                u[k][0] = sctl::select(mask, dX[k] * F, zero);
                for (int i = 0; i < 2; i++) {
                    vector_type val = Real{2} * dX[k] * dX[i] * Fprime;
                    if (i == k)
                        val = val + F;
                    u[k][1 + i] = sctl::select(mask, val, zero);
                }
            }
        }
    }
};

template <class Real, int MaxVecLen>
struct LaplaceDipolePolyEvaluator3D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;

    vector_type thresh2_vec, d2max_vec, rsc_vec, cen_vec;
    const Real *coeffs;
    int n_coeffs;
    int n_digits;

    template <int KERNEL_OUTPUT_DIM>
    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KERNEL_OUTPUT_DIM > 1);
        static_assert(KERNEL_OUTPUT_DIM == 1 || KERNEL_OUTPUT_DIM == 4);

        const vector_type R2 = FMA(dX[0], dX[0], FMA(dX[1], dX[1], dX[2] * dX[2]));
        const auto in_range = (R2 > thresh2_vec) & (R2 < d2max_vec);
        const vector_type Rinv = my_approx_rsqrt(R2, n_digits);
        const vector_type xmapped = FMA(R2, Rinv, cen_vec) * rsc_vec;
        const vector_type zero = vector_type::Zero();
        const vector_type Rinv2 = Rinv * Rinv;
        const vector_type Rinv3 = Rinv2 * Rinv;

        if constexpr (!has_grad) {
            // F(R) = P*Rinv^3 - dP*rsc*Rinv^2.  U[k][0] = dX[k] * F.
            vector_type P, dP;
            horner_val_deriv(xmapped, coeffs, n_coeffs, P, dP);
            const vector_type F = P * Rinv3 - dP * rsc_vec * Rinv2;
            for (int k = 0; k < 3; k++)
                u[k][0] = sctl::select<Real, MaxVecLen>(in_range, dX[k] * F, zero);
        } else {
            // F(R)   = P*Rinv^3 - dP*rsc*Rinv^2
            // F'(R)  = 3 dP*rsc*Rinv^3 - 3 P*Rinv^4 - ddP*rsc^2*Rinv^2
            // F'/R   = 3 dP*rsc*Rinv^4 - 3 P*Rinv^5 - ddP*rsc^2*Rinv^3
            // U[k][0]   = dX[k] * F
            // U[k][1+i] = delta_{ki} * F + dX[k] * dX[i] * (F'/R)
            vector_type P, dP, ddP;
            horner_val_deriv2(xmapped, coeffs, n_coeffs, P, dP, ddP);
            const vector_type Rinv4 = Rinv2 * Rinv2;
            const vector_type Rinv5 = Rinv4 * Rinv;
            const vector_type F = P * Rinv3 - dP * rsc_vec * Rinv2;
            const vector_type F_over_R =
                Real{3} * dP * rsc_vec * Rinv4 - Real{3} * P * Rinv5 - ddP * rsc_vec * rsc_vec * Rinv3;
            for (int k = 0; k < 3; k++) {
                u[k][0] = sctl::select<Real, MaxVecLen>(in_range, dX[k] * F, zero);
                for (int i = 0; i < 3; i++) {
                    vector_type val = dX[k] * dX[i] * F_over_R;
                    if (i == k)
                        val = val + F;
                    u[k][1 + i] = sctl::select<Real, MaxVecLen>(in_range, val, zero);
                }
            }
        }
    }
};

template <typename Real, int MaxVecLen, int EVAL_LEVEL = DMK_POTENTIAL>
struct SqrtLaplaceEvaluator2D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == DMK_POTENTIAL ? 1 : 3;

    template <int KDIM>
    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[1][KDIM], const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KDIM == 3);
        const vector_type R2 = FMA(dX[0], dX[0], dX[1] * dX[1]);
        const auto mask = R2 > vector_type::Zero();
        const vector_type Rinv = sctl::approx_rsqrt<-1>(R2, mask); // 0 on masked lanes
        u[0][0] = Rinv;
        if constexpr (has_grad) {
            // grad_i = -dX_i / r^3 = -dX_i * Rinv^3 (0 on masked lanes since Rinv=0)
            const vector_type Rinv3 = Rinv * Rinv * Rinv;
            for (int i = 0; i < 2; i++)
                u[0][1 + i] = -dX[i] * Rinv3;
        }
    }
};

template <typename Real, int MaxVecLen, int EVAL_LEVEL = DMK_POTENTIAL>
struct SqrtLaplaceEvaluator3D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == DMK_POTENTIAL ? 1 : 4;

    template <int KDIM>
    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[1][KDIM], const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KDIM == 4);
        const vector_type R2 = FMA(dX[0], dX[0], FMA(dX[1], dX[1], dX[2] * dX[2]));
        const auto mask = R2 > vector_type::Zero();
        const vector_type Rinv = sctl::approx_rsqrt<-1>(R2, mask); // 0 on masked lanes
        const vector_type Rinv2 = Rinv * Rinv;
        u[0][0] = Rinv2;
        if constexpr (has_grad) {
            // grad_i = -2 dX_i / r^4 = -2 dX_i Rinv^4 (0 on masked lanes since Rinv=0)
            const vector_type g = Real{-2.0} * Rinv2 * Rinv2;
            for (int i = 0; i < 3; i++)
                u[0][1 + i] = dX[i] * g;
        }
    }
};

template <typename Real, int MaxVecLen>
struct SqrtLaplacePolyEvaluator2D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;

    vector_type thresh2_vec, d2max_vec, rsc_vec, cen_vec;
    const Real *coeffs;
    int n_coeffs;
    int n_digits;

    template <int KERNEL_OUTPUT_DIM>
    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[1][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KERNEL_OUTPUT_DIM == 3);
        static_assert(KERNEL_OUTPUT_DIM == 1 || KERNEL_OUTPUT_DIM == 3, "Invalid KDIM");

        const vector_type R2 = FMA(dX[0], dX[0], dX[1] * dX[1]);
        const auto mask = (R2 > thresh2_vec) & (R2 < d2max_vec);
        const vector_type Rinv = my_approx_rsqrt(R2, n_digits);
        const vector_type x = sctl::FMA(R2, Rinv, cen_vec) * rsc_vec; // (r + cen)*rsc
        if constexpr (!has_grad) {
            u[0][0] = sctl::select(mask, horner(x, coeffs, n_coeffs) * Rinv, vector_type::Zero());
        } else {
            vector_type P, dP;
            horner_val_deriv(x, coeffs, n_coeffs, P, dP); // dP = dP/dx
            u[0][0] = sctl::select(mask, P * Rinv, vector_type::Zero());
            // f = P*Rinv, x = (r + cen)*rsc; grad_i = dX_i * Rinv^2 * (dP*rsc - P*Rinv)
            const vector_type df = Rinv * Rinv * (dP * rsc_vec - P * Rinv);
            for (int i = 0; i < 2; i++)
                u[0][1 + i] = sctl::select(mask, dX[i] * df, vector_type::Zero());
        }
    }
};

template <typename Real, int MaxVecLen>
struct SqrtLaplacePolyEvaluator3D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;

    vector_type thresh2_vec, d2max_vec, rsc_vec, cen_vec;
    const Real *coeffs;
    int n_coeffs;
    int n_digits;

    template <int KERNEL_OUTPUT_DIM>
    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[1][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        constexpr bool has_grad = (KERNEL_OUTPUT_DIM == 4);
        static_assert(KERNEL_OUTPUT_DIM == 1 || KERNEL_OUTPUT_DIM == 4, "Invalid KDIM");

        const vector_type R2 = FMA(dX[0], dX[0], FMA(dX[1], dX[1], dX[2] * dX[2]));
        const auto mask = (R2 > thresh2_vec) & (R2 < d2max_vec);
        const vector_type Rinv = my_approx_rsqrt(R2, n_digits);
        const vector_type R2inv = Rinv * Rinv;
        const vector_type x = FMA(R2, rsc_vec, cen_vec); // R2*rsc + cen (poly in R^2)
        if constexpr (!has_grad) {
            u[0][0] = sctl::select(mask, R2inv * horner(x, coeffs, n_coeffs), vector_type::Zero());
        } else {
            vector_type P, dP;
            horner_val_deriv(x, coeffs, n_coeffs, P, dP); // dP = dP/dx
            u[0][0] = sctl::select(mask, R2inv * P, vector_type::Zero());
            // f = P*R2inv, x = R2*rsc + cen; grad_i = 2 dX_i * R2inv * (dP*rsc - P*R2inv)
            const vector_type df = Real{2.0} * R2inv * (dP * rsc_vec - P * R2inv);
            for (int i = 0; i < 3; i++)
                u[0][1 + i] = sctl::select(mask, dX[i] * df, vector_type::Zero());
        }
    }
};

template <typename Real, int MaxVecLen>
struct StokesletEvaluator3D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = 3;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 0.5;

    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        const vector_type R2 = FMA(dX[0], dX[0], FMA(dX[1], dX[1], dX[2] * dX[2]));
        const auto mask = (R2 > vector_type::Zero());
        const vector_type Rinv = sctl::approx_rsqrt<-1>(R2, mask);
        const vector_type Rinv3 = Rinv * Rinv * Rinv;

        for (int i = 0; i < KERNEL_INPUT_DIM; ++i) {
            for (int j = 0; j < KERNEL_OUTPUT_DIM; ++j) {
                vector_type val = dX[j] * dX[i] * Rinv3;
                if (i == j)
                    val += Rinv;
                u[i][j] = val;
            }
        }
    }
};

template <typename Real, int MaxVecLen>
struct StokesletPolyEvaluator3D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = 3;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = 1.0;

    vector_type thresh2_vec, d2max_vec, rsc_vec, cen_vec;
    const Real *coeffs_diag;
    const Real *coeffs_offdiag;
    int n_coeffs_diag;
    int n_coeffs_offdiag;
    int n_digits;

    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM]) const {
        const vector_type R2 = FMA(dX[0], dX[0], FMA(dX[1], dX[1], dX[2] * dX[2]));
        const auto mask = (R2 > thresh2_vec) & (R2 < d2max_vec);
        const vector_type half = Real{0.5};
        const vector_type Rinv = my_approx_rsqrt(R2, n_digits);
        const vector_type Rinv3 = Rinv * Rinv * Rinv;
        const vector_type xtmp = FMA(R2, Rinv, cen_vec) * rsc_vec;
        const vector_type fdiag = (half - horner(xtmp, coeffs_diag, n_coeffs_diag)) * Rinv;
        const vector_type foffd = (half - horner(xtmp, coeffs_offdiag, n_coeffs_offdiag)) * Rinv3;

        for (int i = 0; i < KERNEL_INPUT_DIM; ++i) {
            for (int j = 0; j < KERNEL_OUTPUT_DIM; ++j) {
                vector_type val = foffd * dX[j] * dX[i];
                if (i == j)
                    val = val + fdiag;
                u[i][j] = select(mask, val, vector_type::Zero());
            }
        }
    }
};

template <typename Real, int MaxVecLen>
struct StressletEvaluator3D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = 3;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int NORMAL_DIM = 3;
    static constexpr Real scale_factor = 1.0;

    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM], const vector_type (&ns)[NORMAL_DIM]) const {
        const vector_type R2 = FMA(dX[0], dX[0], FMA(dX[1], dX[1], dX[2] * dX[2]));
        const auto mask = (R2 > vector_type::Zero());
        const vector_type Rinv = sctl::approx_rsqrt<-1>(R2, mask);
        const vector_type Rinv3 = Rinv * Rinv * Rinv;
        const vector_type Rinv5 = Rinv3 * Rinv * Rinv;
        const vector_type rdotn = FMA(dX[0], ns[0], FMA(dX[1], ns[1], dX[2] * ns[2]));

        const vector_type neg3 = Real{-3.0};
        const vector_type factor = neg3 * rdotn * Rinv5;
        for (int j = 0; j < KERNEL_INPUT_DIM; ++j) {
            const vector_type fj = dX[j] * factor;
            for (int i = 0; i < KERNEL_OUTPUT_DIM; ++i)
                u[j][i] = fj * dX[i];
        }
    }
};

template <typename Real, int MaxVecLen>
struct StressletPolyEvaluator3D {
    using scalar_type = Real;
    using vector_type = sctl::Vec<Real, MaxVecLen>;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = 3;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int NORMAL_DIM = 3;
    static constexpr Real scale_factor = 1.0;

    vector_type thresh2_vec, d2max_vec, rsc_vec, cen_vec;
    const Real *coeffs_diag;
    const Real *coeffs_offdiag;
    int n_coeffs_diag;
    int n_coeffs_offdiag;
    int n_digits;

    DMK_ALWAYS_INLINE void operator()(vector_type (&u)[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM],
                                      const vector_type (&dX)[SPATIAL_DIM], const vector_type (&ns)[NORMAL_DIM]) const {
        const vector_type R2 = FMA(dX[0], dX[0], FMA(dX[1], dX[1], dX[2] * dX[2]));
        const auto mask = (R2 > thresh2_vec) & (R2 < d2max_vec);
        const vector_type Rinv = my_approx_rsqrt(R2, n_digits);
        const vector_type Rinv3 = Rinv * Rinv * Rinv;
        const vector_type Rinv5 = Rinv3 * Rinv * Rinv;
        const vector_type xtmp = FMA(R2, Rinv, cen_vec) * rsc_vec;

        const vector_type Fdiag = -horner(xtmp, coeffs_diag, n_coeffs_diag) * Rinv3;
        const vector_type six = Real{6.0};
        const vector_type Foffd = six * horner(xtmp, coeffs_offdiag, n_coeffs_offdiag) * Rinv5;

        const vector_type rdotn = FMA(dX[0], ns[0], FMA(dX[1], ns[1], dX[2] * ns[2]));

        // Stresslet residual kernel (3D), matching Fortran stokes_dmk:
        //   pot_i += Foffd * (r.mu) * (r.nu) * r_i
        //          + Fdiag * (r_i * (mu.nu) + mu_i * (r.nu) + nu_i * (r.mu))
        //
        // U[j][i] = Foffd * dX[j] * rdotn * dX[i]
        //         + Fdiag * (dX[i] * ns[j] + ns[i] * dX[j] + rdotn * delta_{ij})
        const vector_type Fdiag_rdotn = Fdiag * rdotn;
        for (int j = 0; j < KERNEL_INPUT_DIM; ++j) {
            const vector_type foffd_rj_rdotn = Foffd * dX[j] * rdotn;
            const vector_type fdiag_nj = Fdiag * ns[j];
            const vector_type fdiag_rj = Fdiag * dX[j];
            for (int i = 0; i < KERNEL_OUTPUT_DIM; ++i) {
                vector_type val = foffd_rj_rdotn * dX[i] + fdiag_nj * dX[i] + fdiag_rj * ns[i];
                if (i == j)
                    val = val + Fdiag_rdotn;
                u[j][i] = select(mask, val, vector_type::Zero());
            }
        }
    }
};

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS = -1, int EVAL_LEVEL = -1>
void laplace_2d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                               int n_coeffs_rt_0, const Real *coeffs, int n_src, const Real *r_src, const Real *charge,
                               const Real *normals, int n_trg, const Real *r_trg, Real *pot, int unroll_factor) {
    constexpr bool is_static = (N_DIGITS > 0);
    const int n_digits = is_static ? N_DIGITS : n_digits_rt;
    const int n_coeffs = is_static ? N_COEFFS : n_coeffs_rt_0;
    const int eval_level = (EVAL_LEVEL > 0) ? EVAL_LEVEL : eval_level_rt;

    LaplacePolyEvaluator2D<Real, MaxVecLen> evaluator{thresh2, d2max,    rsc,      cen,       Real{0.5} * rsc,
                                                      coeffs,  n_coeffs, n_digits, eval_level};

    if (eval_level == 1) {
        constexpr int KERNEL_OUTPUT_DIM = 1;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    } else {
        constexpr int KERNEL_OUTPUT_DIM = 3;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    }
}

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS = -1, int EVAL_LEVEL = -1>
void laplace_3d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                               int n_coeffs_rt_0, const Real *coeffs, int n_src, const Real *r_src, const Real *charge,
                               const Real *normals, int n_trg, const Real *r_trg, Real *pot, int unroll_factor) {
    constexpr bool is_static = (N_DIGITS > 0);
    const int n_digits = is_static ? N_DIGITS : n_digits_rt;
    const int n_coeffs = is_static ? N_COEFFS : n_coeffs_rt_0;
    const int eval_level = (EVAL_LEVEL > 0) ? EVAL_LEVEL : eval_level_rt;
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

    LaplacePolyEvaluator3D<Real, MaxVecLen> evaluator{thresh2,  d2max,          rsc,       cen, coeffs, n_coeffs,
                                                      n_digits, transform_poly, eval_level};

    if (eval_level == 1) {
        constexpr int KERNEL_OUTPUT_DIM = 1;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    } else {
        constexpr int KERNEL_OUTPUT_DIM = 4;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    }
}

// Range-list twin of laplace_3d_poly_all_pairs.  Only the final EvalPairs call changes.
template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS = -1, int EVAL_LEVEL = -1>
void laplace_3d_poly_all_pairs_ranges(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                                      int n_coeffs_rt_0, const Real *coeffs, int n_ranges, const int *range_starts,
                                      const int *range_lens, int n_src, const Real *r_src, const Real *charge,
                                      const Real *normals, int n_trg, const Real *r_trg, Real *pot, int unroll_factor) {
    constexpr bool is_static = (N_DIGITS > 0);
    const int n_digits = is_static ? N_DIGITS : n_digits_rt;
    const int n_coeffs = is_static ? N_COEFFS : n_coeffs_rt_0;
    const int eval_level = (EVAL_LEVEL > 0) ? EVAL_LEVEL : eval_level_rt;
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

    LaplacePolyEvaluator3D<Real, MaxVecLen> evaluator{thresh2,  d2max,          rsc,       cen, coeffs, n_coeffs,
                                                      n_digits, transform_poly, eval_level};

    if (eval_level == 1) {
        constexpr int KERNEL_OUTPUT_DIM = 1;
        EvalPairsRanges<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_ranges, range_starts, range_lens, n_trg,
                                           r_trg, pot, evaluator, unroll_factor);
    } else {
        constexpr int KERNEL_OUTPUT_DIM = 4;
        EvalPairsRanges<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_ranges, range_starts, range_lens, n_trg,
                                           r_trg, pot, evaluator, unroll_factor);
    }
}

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS = -1, int EVAL_LEVEL = -1>
void yukawa_3d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                              int n_coeffs_rt_0, const Real *coeffs, int n_src, const Real *r_src, const Real *charge,
                              const Real *normals, int n_trg, const Real *r_trg, Real *pot, int unroll_factor) {
    const int n_digits = (N_DIGITS > 0) ? N_DIGITS : n_digits_rt;
    const int n_coeffs = (N_COEFFS > 0) ? N_COEFFS : n_coeffs_rt_0;
    const int eval_level = (EVAL_LEVEL > 0) ? EVAL_LEVEL : eval_level_rt;

    YukawaPolyEvaluator3D<Real, MaxVecLen> evaluator{thresh2, d2max, rsc, cen, coeffs, n_coeffs, n_digits};

    if (eval_level == 1) {
        constexpr int KERNEL_OUTPUT_DIM = 1;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    } else {
        constexpr int KERNEL_OUTPUT_DIM = 4;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    }
}

// 2D Yukawa residual, log-split path. coeffs holds the two concatenated monomial
// polynomials [PA (n_coeffs_log) | PB (n_coeffs_reg)]. rsc/cen use the 2D-Laplace
// mapping (poly in r^2, rsc = 2/bsize^2, cen = -1), so bsizeinv2 = 0.5*rsc = 1/bsize^2.
template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS_LOG = -1, int N_COEFFS_REG = -1,
          int EVAL_LEVEL = -1>
void yukawa_2d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                              int n_coeffs_log_rt, int n_coeffs_reg_rt, const Real *coeffs, int n_src,
                              const Real *r_src, const Real *charge, const Real *normals, int n_trg, const Real *r_trg,
                              Real *pot, int unroll_factor) {
    const int n_digits = (N_DIGITS > 0) ? N_DIGITS : n_digits_rt;
    const int n_log = (N_COEFFS_LOG > 0) ? N_COEFFS_LOG : n_coeffs_log_rt;
    const int n_reg = (N_COEFFS_REG > 0) ? N_COEFFS_REG : n_coeffs_reg_rt;
    const int eval_level = (EVAL_LEVEL > 0) ? EVAL_LEVEL : eval_level_rt;

    YukawaPolyEvaluator2D<Real, MaxVecLen> evaluator{thresh2, d2max,          rsc,   cen,     Real{0.5} * rsc, coeffs,
                                                     n_log,   coeffs + n_log, n_reg, n_digits};

    if (eval_level == 1) {
        constexpr int KERNEL_OUTPUT_DIM = 1;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    } else {
        constexpr int KERNEL_OUTPUT_DIM = 3;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    }
}

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS = -1, int EVAL_LEVEL = -1>
void laplace_dipole_2d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                                      int n_coeffs_rt_0, const Real *coeffs, int n_src, const Real *r_src,
                                      const Real *charge, const Real *normals, int n_trg, const Real *r_trg, Real *pot,
                                      int unroll_factor) {
    constexpr bool is_static = (N_DIGITS > 0);
    const int n_digits = is_static ? N_DIGITS : n_digits_rt;
    const int n_coeffs = is_static ? N_COEFFS : n_coeffs_rt_0;
    const int eval_level = (EVAL_LEVEL > 0) ? EVAL_LEVEL : eval_level_rt;

    LaplaceDipolePolyEvaluator2D<Real, MaxVecLen> evaluator{thresh2, d2max, rsc, cen, coeffs, n_coeffs, n_digits};

    if (eval_level == DMK_POTENTIAL) {
        constexpr int KERNEL_OUTPUT_DIM = 1;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    } else {
        constexpr int KERNEL_OUTPUT_DIM = 3;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    }
}

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS = -1, int EVAL_LEVEL = -1>
void laplace_dipole_3d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                                      int n_coeffs_rt_0, const Real *coeffs, int n_src, const Real *r_src,
                                      const Real *charge, const Real *normals, int n_trg, const Real *r_trg, Real *pot,
                                      int unroll_factor) {
    constexpr bool is_static = (N_DIGITS > 0);
    const int n_digits = is_static ? N_DIGITS : n_digits_rt;
    const int n_coeffs = is_static ? N_COEFFS : n_coeffs_rt_0;
    const int eval_level = (EVAL_LEVEL > 0) ? EVAL_LEVEL : eval_level_rt;

    LaplaceDipolePolyEvaluator3D<Real, MaxVecLen> evaluator{thresh2, d2max, rsc, cen, coeffs, n_coeffs, n_digits};

    if (eval_level == DMK_POTENTIAL) {
        constexpr int KERNEL_OUTPUT_DIM = 1;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    } else {
        constexpr int KERNEL_OUTPUT_DIM = 4;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    }
}

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS = -1, int EVAL_LEVEL = -1>
void sqrt_laplace_2d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                                    int n_coeffs_rt_0, const Real *coeffs, int n_src, const Real *r_src,
                                    const Real *charge, const Real *normals, int n_trg, const Real *r_trg, Real *pot,
                                    int unroll_factor) {
    constexpr bool is_static = (N_DIGITS > 0);
    const int n_digits = is_static ? N_DIGITS : n_digits_rt;
    const int n_coeffs = is_static ? N_COEFFS : n_coeffs_rt_0;
    const int eval_level = (EVAL_LEVEL > 0) ? EVAL_LEVEL : eval_level_rt;

    SqrtLaplacePolyEvaluator2D<Real, MaxVecLen> evaluator{thresh2, d2max, rsc, cen, coeffs, n_coeffs, n_digits};

    if (eval_level == 1) {
        constexpr int KERNEL_OUTPUT_DIM = 1;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    } else {
        constexpr int KERNEL_OUTPUT_DIM = 3;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    }
}

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS = -1, int EVAL_LEVEL = -1>
void sqrt_laplace_3d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                                    int n_coeffs_rt_0, const Real *coeffs, int n_src, const Real *r_src,
                                    const Real *charge, const Real *normals, int n_trg, const Real *r_trg, Real *pot,
                                    int unroll_factor) {
    constexpr bool is_static = (N_DIGITS > 0);
    const int n_digits = is_static ? N_DIGITS : n_digits_rt;
    const int n_coeffs = is_static ? N_COEFFS : n_coeffs_rt_0;
    const int eval_level = (EVAL_LEVEL > 0) ? EVAL_LEVEL : eval_level_rt;

    SqrtLaplacePolyEvaluator3D<Real, MaxVecLen> evaluator{thresh2, d2max, rsc, cen, coeffs, n_coeffs, n_digits};

    if (eval_level == 1) {
        constexpr int KERNEL_OUTPUT_DIM = 1;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    } else {
        constexpr int KERNEL_OUTPUT_DIM = 4;
        EvalPairs<KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
    }
}

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS = -1, int EVAL_LEVEL = -1>
void stokes_2d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                              int n_coeffs_rt_0, int n_coeffs_rt_1, const Real *coeffs, int n_src, const Real *r_src,
                              const Real *charge, const Real *normals, int n_trg, const Real *r_trg, Real *pot,
                              int unroll_factor) {}

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS_0 = -1, int N_COEFFS_1 = -1, int EVAL_LEVEL = -1>
void stokeslet_3d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                                 int n_coeffs_rt_0, int n_coeffs_rt_1, const Real *coeffs, int n_src, const Real *r_src,
                                 const Real *charge, const Real *normals, int n_trg, const Real *r_trg, Real *pot,
                                 int unroll_factor) {
    constexpr bool is_static = (N_DIGITS > 0);
    const int n_digits = is_static ? N_DIGITS : n_digits_rt;
    const int n_coeffs_diag = is_static ? N_COEFFS_0 : n_coeffs_rt_0;
    const int n_coeffs_offdiag = is_static ? N_COEFFS_1 : n_coeffs_rt_1;
    const Real *coeffs_diag = coeffs;
    const Real *coeffs_offdiag = coeffs + n_coeffs_diag;
    using Evaluator = StokesletPolyEvaluator3D<Real, MaxVecLen>;

    Evaluator evaluator{thresh2,          d2max,   rsc, cen, coeffs_diag, coeffs_offdiag, n_coeffs_diag,
                        n_coeffs_offdiag, n_digits};

    EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator, unroll_factor);
}

template <class Real, int MaxVecLen, int N_DIGITS = -1, int N_COEFFS_0 = -1, int N_COEFFS_1 = -1, int EVAL_LEVEL = -1>
void stresslet_3d_poly_all_pairs(int eval_level_rt, int n_digits_rt, Real rsc, Real cen, Real d2max, Real thresh2,
                                 int n_coeffs_rt_0, int n_coeffs_rt_1, const Real *coeffs, int n_src, const Real *r_src,
                                 const Real *charge, const Real *normals, int n_trg, const Real *r_trg, Real *pot,
                                 int unroll_factor) {
    constexpr bool is_static = (N_DIGITS > 0);
    const int n_digits = is_static ? N_DIGITS : n_digits_rt;
    const int n_coeffs_diag = is_static ? N_COEFFS_0 : n_coeffs_rt_0;
    const int n_coeffs_offdiag = is_static ? N_COEFFS_1 : n_coeffs_rt_1;
    const Real *coeffs_diag = coeffs;
    const Real *coeffs_offdiag = coeffs + n_coeffs_diag;
    using Evaluator = StressletPolyEvaluator3D<Real, MaxVecLen>;

    Evaluator evaluator{thresh2,          d2max,   rsc, cen, coeffs_diag, coeffs_offdiag, n_coeffs_diag,
                        n_coeffs_offdiag, n_digits};

    EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, normals, n_trg, r_trg, pot, evaluator, unroll_factor);
}

template <class Real, int MaxVecLen>
inline void yukawa_3d_all_pairs_direct(int n_src, const Real *r_src, const Real *charge, int n_trg, const Real *r_trg,
                                       Real *pot, int unroll_factor, Real lambda, int eval_level) {
    if (eval_level == DMK_POTENTIAL) {
        using Evaluator = YukawaEvaluator3D<Real, MaxVecLen, DMK_POTENTIAL>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot,
                                                       Evaluator{lambda}, unroll_factor);
    }
    if (eval_level == DMK_POTENTIAL_GRAD) {
        using Evaluator = YukawaEvaluator3D<Real, MaxVecLen, DMK_POTENTIAL_GRAD>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot,
                                                       Evaluator{lambda}, unroll_factor);
    }
    throw std::runtime_error("Direct Yukawa evaluator only supports DMK_POTENTIAL/DMK_POTENTIAL_GRAD");
}

template <class Real, int MaxVecLen>
inline void yukawa_2d_all_pairs_direct(int n_src, const Real *r_src, const Real *charge, int n_trg, const Real *r_trg,
                                       Real *pot, int unroll_factor, Real lambda, int eval_level) {
    if (eval_level == DMK_POTENTIAL) {
        using Evaluator = YukawaEvaluator2D<Real, MaxVecLen, DMK_POTENTIAL>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot,
                                                       Evaluator{lambda}, unroll_factor);
    }
    if (eval_level == DMK_POTENTIAL_GRAD) {
        using Evaluator = YukawaEvaluator2D<Real, MaxVecLen, DMK_POTENTIAL_GRAD>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot,
                                                       Evaluator{lambda}, unroll_factor);
    }
    throw std::runtime_error("Direct Yukawa evaluator only supports DMK_POTENTIAL/DMK_POTENTIAL_GRAD");
}

template <class Real, int MaxVecLen>
inline void laplace_2d_all_pairs_direct(int n_src, const Real *r_src, const Real *charge, int n_trg, const Real *r_trg,
                                        Real *pot, int unroll_factor, int eval_level) {
    if (eval_level == DMK_POTENTIAL) {
        using Evaluator = LaplaceEvaluator2D<Real, MaxVecLen, DMK_POTENTIAL>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, Evaluator{},
                                                       unroll_factor);
    }
    if (eval_level == DMK_POTENTIAL_GRAD) {
        using Evaluator = LaplaceEvaluator2D<Real, MaxVecLen, DMK_POTENTIAL_GRAD>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, Evaluator{},
                                                       unroll_factor);
    }
    throw std::runtime_error("Direct Laplace evaluator only supports DMK_POTENTIAL/DMK_POTENTIAL_GRAD");
}

template <class Real, int MaxVecLen>
inline void laplace_3d_all_pairs_direct(int n_src, const Real *r_src, const Real *charge, int n_trg, const Real *r_trg,
                                        Real *pot, int unroll_factor, int eval_level) {
    if (eval_level == DMK_POTENTIAL) {
        using Evaluator = LaplaceEvaluator3D<Real, MaxVecLen, DMK_POTENTIAL>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, Evaluator{},
                                                       unroll_factor);
    }
    if (eval_level == DMK_POTENTIAL_GRAD) {
        using Evaluator = LaplaceEvaluator3D<Real, MaxVecLen, DMK_POTENTIAL_GRAD>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, Evaluator{},
                                                       unroll_factor);
    }
    throw std::runtime_error("Direct Laplace evaluator only supports DMK_POTENTIAL/DMK_POTENTIAL_GRAD");
}

template <class Real, int MaxVecLen>
inline void laplace_dipole_2d_all_pairs_direct(int n_src, const Real *r_src, const Real *charge, int n_trg,
                                               const Real *r_trg, Real *pot, int unroll_factor, int eval_level) {
    if (eval_level == DMK_POTENTIAL) {
        using Evaluator = LaplaceDipoleEvaluator2D<Real, MaxVecLen, DMK_POTENTIAL>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, Evaluator{},
                                                       unroll_factor);
    }
    if (eval_level == DMK_POTENTIAL_GRAD) {
        using Evaluator = LaplaceDipoleEvaluator2D<Real, MaxVecLen, DMK_POTENTIAL_GRAD>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, Evaluator{},
                                                       unroll_factor);
    }
    throw std::runtime_error("Direct Laplace dipole evaluator only supports DMK_POTENTIAL/DMK_POTENTIAL_GRAD");
}

template <class Real, int MaxVecLen>
inline void laplace_dipole_3d_all_pairs_direct(int n_src, const Real *r_src, const Real *charge, int n_trg,
                                               const Real *r_trg, Real *pot, int unroll_factor, int eval_level) {
    if (eval_level == DMK_POTENTIAL) {
        using Evaluator = LaplaceDipoleEvaluator3D<Real, MaxVecLen, DMK_POTENTIAL>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, Evaluator{},
                                                       unroll_factor);
    }
    if (eval_level == DMK_POTENTIAL_GRAD) {
        using Evaluator = LaplaceDipoleEvaluator3D<Real, MaxVecLen, DMK_POTENTIAL_GRAD>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, Evaluator{},
                                                       unroll_factor);
    }
    throw std::runtime_error("Direct Laplace dipole evaluator only supports DMK_POTENTIAL/DMK_POTENTIAL_GRAD");
}

template <class Real, int MaxVecLen>
inline void sqrt_laplace_2d_all_pairs_direct(int n_src, const Real *r_src, const Real *charge, int n_trg,
                                             const Real *r_trg, Real *pot, int unroll_factor, int eval_level) {
    if (eval_level == DMK_POTENTIAL) {
        using Evaluator = SqrtLaplaceEvaluator2D<Real, MaxVecLen, DMK_POTENTIAL>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, Evaluator{},
                                                       unroll_factor);
    }
    if (eval_level == DMK_POTENTIAL_GRAD) {
        using Evaluator = SqrtLaplaceEvaluator2D<Real, MaxVecLen, DMK_POTENTIAL_GRAD>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, Evaluator{},
                                                       unroll_factor);
    }
    throw std::runtime_error("Direct SqrtLaplace evaluator only supports DMK_POTENTIAL/DMK_POTENTIAL_GRAD");
}

template <class Real, int MaxVecLen>
inline void sqrt_laplace_3d_all_pairs_direct(int n_src, const Real *r_src, const Real *charge, int n_trg,
                                             const Real *r_trg, Real *pot, int unroll_factor, int eval_level) {
    if (eval_level == DMK_POTENTIAL) {
        using Evaluator = SqrtLaplaceEvaluator3D<Real, MaxVecLen, DMK_POTENTIAL>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, Evaluator{},
                                                       unroll_factor);
    }
    if (eval_level == DMK_POTENTIAL_GRAD) {
        using Evaluator = SqrtLaplaceEvaluator3D<Real, MaxVecLen, DMK_POTENTIAL_GRAD>;
        return EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, Evaluator{},
                                                       unroll_factor);
    }
    throw std::runtime_error("Direct SqrtLaplace evaluator only supports DMK_POTENTIAL/DMK_POTENTIAL_GRAD");
}

template <class Real, int MaxVecLen>
inline void stokeslet_3d_all_pairs_direct(int n_src, const Real *r_src, const Real *charge, int n_trg,
                                          const Real *r_trg, Real *pot, int unroll_factor) {
    using Evaluator = StokesletEvaluator3D<Real, MaxVecLen>;
    EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, nullptr, n_trg, r_trg, pot, Evaluator{},
                                            unroll_factor);
}

template <class Real, int MaxVecLen>
inline void stresslet_3d_all_pairs_direct(int n_src, const Real *r_src, const Real *charge, const Real *normals,
                                          int n_trg, const Real *r_trg, Real *pot, int unroll_factor) {
    using Evaluator = StressletEvaluator3D<Real, MaxVecLen>;
    EvalPairs<Evaluator::KERNEL_OUTPUT_DIM>(n_src, r_src, charge, normals, n_trg, r_trg, pot, Evaluator{},
                                            unroll_factor);
}

#undef DMK_ALWAYS_INLINE

#endif
