// V2 point-tree near-field (direct) device kernel. The launcher prepends a
// prelude defining Real, the DMK_DIRECT_KERNEL_NAME symbol, the baked-literal
// coefficient struct(s), the DMK_DIRECT_EVALUATOR typedef, and the SRC_TILE /
// BLOCK_SIZE / TARGETS_PER_THREAD constants. Coefficients are compile-time
// literals folded straight into the FMAs (no runtime coeff buffer, no AOT).
//
// Scalar potential kernels (Laplace, Sqrt-Laplace; 2D + 3D), Laplace-dipole
// (3D; 3-vector charge, scalar/gradient output) and Stokeslet / Stresslet
// velocity kernels (3D; two coeff packs diag+offdiag, Stresslet reads
// per-source normals).

#include <dmk/cuda/direct_kernelargs.hpp>

// Evaluator contract: operator() returns false for a pair outside the near-field annulus
// and leaves `u` untouched, so the caller skips the accumulate entirely. thresh2 > 0, so
// being in range already implies R2 > 0 and rsqrt needs no guard of its own.

// rsqrtf() is already the approximate SFU instruction, but ptxas wraps it in a denormal
// guard -- compare against FLT_MIN, scale by 2^24, MUFU.RSQ, scale back -- because it cannot
// prove the argument is normal. Squared distances here sit above thresh2 = 1e-30, so take
// the bare instruction: identical bits for any normal input, five instructions down to one.
// fp64 keeps the library call, since rsqrt.approx.f64 only carries fp32-level accuracy.
__device__ __forceinline__ float dmk_rsqrt(float x) {
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ double dmk_rsqrt(double x) { return rsqrt(x); }

template <typename Coeffs, int I>
__device__ __forceinline__ Real horner_recurse(Real x, Real acc) {
    if constexpr (I == 0) {
        return acc;
    } else {
        return horner_recurse<Coeffs, I - 1>(x, acc * x + Real{Coeffs::at(I - 1)});
    }
}

template <typename Coeffs>
__device__ __forceinline__ Real horner_const(Real x) {
    static_assert(Coeffs::size > 0, "empty coefficient pack");
    return horner_recurse<Coeffs, Coeffs::size - 1>(x, Real{Coeffs::at(Coeffs::size - 1)});
}

template <typename Coeffs, int I>
__device__ __forceinline__ void horner_vd_recurse(Real x, Real &value, Real &deriv) {
    if constexpr (I == Coeffs::size - 1) {
        value = Real{Coeffs::at(I)};
        deriv = Real{0};
    } else {
        horner_vd_recurse<Coeffs, I + 1>(x, value, deriv);
        deriv = deriv * x + value;
        value = value * x + Real{Coeffs::at(I)};
    }
}

template <typename Coeffs>
__device__ __forceinline__ void horner_val_deriv(Real x, Real &value, Real &deriv) {
    static_assert(Coeffs::size > 0, "empty coefficient pack");
    horner_vd_recurse<Coeffs, 0>(x, value, deriv);
}

// Each step consumes the previous iteration's deriv/value, so the assignment
// order is load-bearing.
template <typename Coeffs, int I>
__device__ __forceinline__ void horner_vd2_recurse(Real x, Real &value, Real &deriv, Real &deriv2) {
    if constexpr (I == Coeffs::size - 1) {
        value = Real{Coeffs::at(I)};
        deriv = Real{0};
        deriv2 = Real{0};
    } else {
        horner_vd2_recurse<Coeffs, I + 1>(x, value, deriv, deriv2);
        deriv2 = deriv2 * x + (deriv + deriv);
        deriv = deriv * x + value;
        value = value * x + Real{Coeffs::at(I)};
    }
}

template <typename Coeffs>
__device__ __forceinline__ void horner_val_deriv2(Real x, Real &value, Real &deriv, Real &deriv2) {
    static_assert(Coeffs::size > 0, "empty coefficient pack");
    horner_vd2_recurse<Coeffs, 0>(x, value, deriv, deriv2);
}

template <typename Coeffs, int EVAL_LEVEL>
struct LaplacePolyEvaluator2DCuda {
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real thresh2;
    Real d2max;
    Real rsc;
    Real cen;

    template <bool CHECK_MIN>
    __device__ inline bool operator()(Real (&u)[1][KERNEL_OUTPUT_DIM], const Real (&dX)[2]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        if constexpr (CHECK_MIN) {
            if (!((R2 > thresh2) && (R2 < d2max)))
                return false;
        } else {
            if (!(R2 < d2max))
                return false;
        }
        const Real R2sc = R2 * (Real{0.5} * rsc);
        const Real xmapped = rsc * R2 + cen;
        if constexpr (KERNEL_OUTPUT_DIM == 1) {
            u[0][0] = Real{0.5} * log(R2sc) + horner_const<Coeffs>(xmapped);
        } else {
            Real P, dP;
            horner_val_deriv<Coeffs>(xmapped, P, dP);
            u[0][0] = Real{0.5} * log(R2sc) + P;
            const Real Rinv = dmk_rsqrt(R2);
            const Real df_dR2 = Real{0.5} * (Rinv * Rinv) + rsc * dP;
#pragma unroll
            for (int i = 0; i < 2; ++i)
                u[0][1 + i] = Real{2} * dX[i] * df_dR2;
        }
        return true;
    }
};

template <typename Coeffs, int EVAL_LEVEL>
struct LaplacePolyEvaluator3DCuda {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real thresh2;
    Real d2max;
    Real rsc;
    Real cen;

    template <bool CHECK_MIN>
    __device__ inline bool operator()(Real (&u)[1][KERNEL_OUTPUT_DIM], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const bool in_range = CHECK_MIN ? ((R2 > thresh2) && (R2 < d2max)) : (R2 < d2max);
        if constexpr (KERNEL_OUTPUT_DIM == 1) {
            const Real Rinv = dmk_rsqrt(R2);
            const Real xmapped = (R2 * Rinv + cen) * rsc;
            u[0][0] = in_range ? horner_const<Coeffs>(xmapped) * Rinv : Real{0};
            return true;
        } else {
            if (!in_range)
                return false;
            const Real Rinv = dmk_rsqrt(R2);
            const Real xmapped = (R2 * Rinv + cen) * rsc;
            Real P, dP;
            horner_val_deriv<Coeffs>(xmapped, P, dP);
            u[0][0] = P * Rinv;
            const Real df_dR2 = Rinv * Rinv * (dP * rsc - P * Rinv);
#pragma unroll
            for (int i = 0; i < 3; ++i)
                u[0][1 + i] = dX[i] * df_dR2;
            return true;
        }
    }
};

template <typename Coeffs, int EVAL_LEVEL>
struct SqrtLaplacePolyEvaluator2DCuda {
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real thresh2;
    Real d2max;
    Real rsc;
    Real cen;

    template <bool CHECK_MIN>
    __device__ inline bool operator()(Real (&u)[1][KERNEL_OUTPUT_DIM], const Real (&dX)[2]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        if constexpr (CHECK_MIN) {
            if (!((R2 > thresh2) && (R2 < d2max)))
                return false;
        } else {
            if (!(R2 < d2max))
                return false;
        }
        const Real Rinv = dmk_rsqrt(R2);
        const Real xmapped = (R2 * Rinv + cen) * rsc;
        if constexpr (KERNEL_OUTPUT_DIM == 1) {
            u[0][0] = horner_const<Coeffs>(xmapped) * Rinv;
        } else {
            Real P, dP;
            horner_val_deriv<Coeffs>(xmapped, P, dP);
            u[0][0] = P * Rinv;
            const Real df = Rinv * Rinv * (dP * rsc - P * Rinv);
#pragma unroll
            for (int i = 0; i < 2; ++i)
                u[0][1 + i] = dX[i] * df;
        }
        return true;
    }
};

template <typename Coeffs, int EVAL_LEVEL>
struct SqrtLaplacePolyEvaluator3DCuda {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real thresh2;
    Real d2max;
    Real rsc;
    Real cen;

    template <bool CHECK_MIN>
    __device__ inline bool operator()(Real (&u)[1][KERNEL_OUTPUT_DIM], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        if constexpr (CHECK_MIN) {
            if (!((R2 > thresh2) && (R2 < d2max)))
                return false;
        } else {
            if (!(R2 < d2max))
                return false;
        }
        // rsqrt-then-square, not 1/R2: nvrtc defaults to -prec-div=true, so a literal
        // division expands to a full IEEE sequence.
        const Real Rinv = dmk_rsqrt(R2);
        const Real R2inv = Rinv * Rinv;
        const Real xmapped = rsc * R2 + cen;
        if constexpr (KERNEL_OUTPUT_DIM == 1) {
            u[0][0] = R2inv * horner_const<Coeffs>(xmapped);
        } else {
            Real P, dP;
            horner_val_deriv<Coeffs>(xmapped, P, dP);
            u[0][0] = R2inv * P;
            const Real df = Real{2} * R2inv * (dP * rsc - P * R2inv);
#pragma unroll
            for (int i = 0; i < 3; ++i)
                u[0][1 + i] = dX[i] * df;
        }
        return true;
    }
};

// exp(-lambda*r) is folded into the fit, so there is no runtime exp. Differs from
// Laplace 3D only in the argument mapping: cen is -1 here.
template <typename Coeffs, int EVAL_LEVEL>
struct YukawaPolyEvaluator3DCuda {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real thresh2;
    Real d2max;
    Real rsc;
    Real cen;

    template <bool CHECK_MIN>
    __device__ inline bool operator()(Real (&u)[1][KERNEL_OUTPUT_DIM], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        if constexpr (CHECK_MIN) {
            if (!((R2 > thresh2) && (R2 < d2max)))
                return false;
        } else {
            if (!(R2 < d2max))
                return false;
        }
        const Real Rinv = dmk_rsqrt(R2);
        const Real xmapped = fma(R2 * Rinv, rsc, cen);
        if constexpr (KERNEL_OUTPUT_DIM == 1) {
            u[0][0] = horner_const<Coeffs>(xmapped) * Rinv;
        } else {
            Real P, dP;
            horner_val_deriv<Coeffs>(xmapped, P, dP);
            u[0][0] = P * Rinv;
            const Real df_dR2 = Rinv * Rinv * (dP * rsc - P * Rinv);
#pragma unroll
            for (int i = 0; i < 3; ++i)
                u[0][1 + i] = dX[i] * df_dR2;
        }
        return true;
    }
};

// Packs cover levels [LEVEL0, LEVEL0 + sizeof...(Rest) + 1).
template <int EVAL_LEVEL, int LEVEL0, typename C0, typename... Rest>
struct YukawaLevelsCuda {
    using First = YukawaPolyEvaluator3DCuda<C0, EVAL_LEVEL>;
    static constexpr int SPATIAL_DIM = First::SPATIAL_DIM;
    static constexpr int KERNEL_INPUT_DIM = First::KERNEL_INPUT_DIM;
    static constexpr int KERNEL_OUTPUT_DIM = First::KERNEL_OUTPUT_DIM;
    static constexpr int NORMAL_DIM = First::NORMAL_DIM;
    static constexpr Real scale_factor = First::scale_factor;
};

// u[k][j] is the response of output j to dipole strength component k.
template <typename Coeffs, int EVAL_LEVEL>
struct LaplaceDipolePolyEvaluator3DCuda {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real thresh2;
    Real d2max;
    Real rsc;
    Real cen;

    template <bool CHECK_MIN>
    __device__ inline bool operator()(Real (&u)[3][KERNEL_OUTPUT_DIM], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        if constexpr (CHECK_MIN) {
            if (!((R2 > thresh2) && (R2 < d2max)))
                return false;
        } else {
            if (!(R2 < d2max))
                return false;
        }
        const Real Rinv = dmk_rsqrt(R2);
        const Real Rinv2 = Rinv * Rinv;
        const Real Rinv3 = Rinv2 * Rinv;
        const Real xmapped = (R2 * Rinv + cen) * rsc;
        if constexpr (KERNEL_OUTPUT_DIM == 1) {
            Real P, dP;
            horner_val_deriv<Coeffs>(xmapped, P, dP);
            const Real F = P * Rinv3 - dP * rsc * Rinv2;
#pragma unroll
            for (int k = 0; k < 3; ++k)
                u[k][0] = dX[k] * F;
        } else {
            Real P, dP, ddP;
            horner_val_deriv2<Coeffs>(xmapped, P, dP, ddP);
            const Real Rinv4 = Rinv2 * Rinv2;
            const Real Rinv5 = Rinv4 * Rinv;
            const Real F = P * Rinv3 - dP * rsc * Rinv2;
            const Real F_over_R = Real{3} * dP * rsc * Rinv4 - Real{3} * P * Rinv5 - ddP * rsc * rsc * Rinv3;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                u[k][0] = dX[k] * F;
#pragma unroll
                for (int i = 0; i < 3; ++i)
                    u[k][1 + i] = dX[k] * dX[i] * F_over_R + (i == k ? F : Real{0});
            }
        }
        return true;
    }
};

template <typename CoeffsDiag, typename CoeffsOffdiag>
struct StokesletPolyEvaluator3DCuda {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = 3;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real thresh2;
    Real d2max;
    Real rsc;
    Real cen;

    template <bool CHECK_MIN>
    __device__ inline bool operator()(Real (&u)[3][3], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        if constexpr (CHECK_MIN) {
            if (!((R2 > thresh2) && (R2 < d2max)))
                return false;
        } else {
            if (!(R2 < d2max))
                return false;
        }
        const Real Rinv = dmk_rsqrt(R2);
        const Real Rinv3 = Rinv * Rinv * Rinv;
        const Real xmapped = (R2 * Rinv + cen) * rsc;
        const Real fdiag = horner_const<CoeffsDiag>(xmapped) * Rinv;
        const Real foffd = horner_const<CoeffsOffdiag>(xmapped) * Rinv3;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                Real val = foffd * dX[j] * dX[i];
                if (i == j)
                    val += fdiag;
                u[i][j] = val;
            }
        return true;
    }
};

template <typename CoeffsDiag, typename CoeffsOffdiag>
struct StressletPolyEvaluator3DCuda {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = 3;
    static constexpr int NORMAL_DIM = 3;
    static constexpr Real scale_factor = Real{1};

    Real thresh2;
    Real d2max;
    Real rsc;
    Real cen;

    template <bool CHECK_MIN>
    __device__ inline bool operator()(Real (&u)[3][3], const Real (&dX)[3], const Real (&ns)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        if constexpr (CHECK_MIN) {
            if (!((R2 > thresh2) && (R2 < d2max)))
                return false;
        } else {
            if (!(R2 < d2max))
                return false;
        }
        const Real Rinv = dmk_rsqrt(R2);
        const Real Rinv3 = Rinv * Rinv * Rinv;
        const Real Rinv5 = Rinv3 * Rinv * Rinv;
        const Real xmapped = (R2 * Rinv + cen) * rsc;
        const Real Fdiag = -horner_const<CoeffsDiag>(xmapped) * Rinv3;
        const Real Foffd = Real{6} * horner_const<CoeffsOffdiag>(xmapped) * Rinv5;
        const Real rdotn = dX[0] * ns[0] + dX[1] * ns[1] + dX[2] * ns[2];
        const Real Fdiag_rdotn = Fdiag * rdotn;
        for (int j = 0; j < 3; ++j) {
            const Real foffd_rj_rdotn = Foffd * dX[j] * rdotn;
            const Real fdiag_nj = Fdiag * ns[j];
            const Real fdiag_rj = Fdiag * dX[j];
            for (int i = 0; i < 3; ++i) {
                Real val = foffd_rj_rdotn * dX[i] + fdiag_nj * dX[i] + fdiag_rj * ns[i];
                if (i == j)
                    val += Fdiag_rdotn;
                u[j][i] = val;
            }
        }
        return true;
    }
};

// CHECK_MIN gates the lower cutoff. Only the self box can hold a source coincident with a
// target, so elsewhere `R2 > thresh2` is dead; it must be compile-time, since zeroing a
// runtime thresh2 would still leave the compare. Under PBC a wrapped entry names trg_box too
// but with the coincidence shifted a whole period away, so the gate stays conservative.
template <bool CHECK_MIN, typename Eval>
__device__ __forceinline__ void direct_eval_accumulate(const Eval &evaluator, Real (&vt)[Eval::KERNEL_OUTPUT_DIM],
                                                        const Real (&dX)[Eval::SPATIAL_DIM],
                                                        const Real (&vs)[Eval::KERNEL_INPUT_DIM]) {
    Real U[Eval::KERNEL_INPUT_DIM][Eval::KERNEL_OUTPUT_DIM];
    if (!evaluator.template operator()<CHECK_MIN>(U, dX))
        return;

#pragma unroll
    for (int k0 = 0; k0 < Eval::KERNEL_INPUT_DIM; ++k0) {
#pragma unroll
        for (int k1 = 0; k1 < Eval::KERNEL_OUTPUT_DIM; ++k1) {
            vt[k1] += U[k0][k1] * vs[k0];
        }
    }
}

template <bool CHECK_MIN, typename CoeffsDiag, typename CoeffsOffdiag>
__device__ __forceinline__ void
direct_eval_accumulate(const StokesletPolyEvaluator3DCuda<CoeffsDiag, CoeffsOffdiag> &evaluator, Real (&vt)[3],
                       const Real (&dX)[3], const Real (&vs)[3]) {
    const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
    if constexpr (CHECK_MIN) {
        if (!((R2 > evaluator.thresh2) && (R2 < evaluator.d2max)))
            return;
    } else {
        if (!(R2 < evaluator.d2max))
            return;
    }

    const Real Rinv = dmk_rsqrt(R2);
    const Real Rinv3 = Rinv * Rinv * Rinv;
    const Real xmapped = (R2 * Rinv + evaluator.cen) * evaluator.rsc;
    const Real fdiag = horner_const<CoeffsDiag>(xmapped) * Rinv;
    const Real foffd = horner_const<CoeffsOffdiag>(xmapped) * Rinv3;
    const Real rdotv = dX[0] * vs[0] + dX[1] * vs[1] + dX[2] * vs[2];
    const Real off = foffd * rdotv;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        vt[i] = fma(fdiag, vs[i], vt[i]);
        vt[i] = fma(off, dX[i], vt[i]);
    }
}

template <bool CHECK_MIN, typename CoeffsDiag, typename CoeffsOffdiag>
__device__ __forceinline__ void
direct_eval_accumulate(const StressletPolyEvaluator3DCuda<CoeffsDiag, CoeffsOffdiag> &evaluator, Real (&vt)[3],
                       const Real (&dX)[3], const Real (&vs)[3], const Real (&ns)[3]) {
    const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
    if constexpr (CHECK_MIN) {
        if (!((R2 > evaluator.thresh2) && (R2 < evaluator.d2max)))
            return;
    } else {
        if (!(R2 < evaluator.d2max))
            return;
    }

    const Real Rinv = dmk_rsqrt(R2);
    const Real Rinv3 = Rinv * Rinv * Rinv;
    const Real Rinv5 = Rinv3 * Rinv * Rinv;
    const Real xmapped = (R2 * Rinv + evaluator.cen) * evaluator.rsc;
    const Real Fdiag = -horner_const<CoeffsDiag>(xmapped) * Rinv3;
    const Real Foffd = Real{6} * horner_const<CoeffsOffdiag>(xmapped) * Rinv5;
    const Real rdotn = dX[0] * ns[0] + dX[1] * ns[1] + dX[2] * ns[2];
    const Real rdotv = dX[0] * vs[0] + dX[1] * vs[1] + dX[2] * vs[2];
    const Real ndotv = ns[0] * vs[0] + ns[1] * vs[1] + ns[2] * vs[2];
    const Real r_scale = Foffd * rdotn * rdotv + Fdiag * ndotv;
    const Real n_scale = Fdiag * rdotv;
    const Real v_scale = Fdiag * rdotn;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        vt[i] = fma(r_scale, dX[i], vt[i]);
        vt[i] = fma(n_scale, ns[i], vt[i]);
        vt[i] = fma(v_scale, vs[i], vt[i]);
    }
}

template <int EVAL_LEVEL, int I, typename C0, typename... Rest, typename F>
__device__ __forceinline__ void bind_yukawa_level(int idx, Real thresh2, Real d2max, Real rsc, Real cen, F &&f) {
    using Eval = YukawaPolyEvaluator3DCuda<C0, EVAL_LEVEL>;
    if constexpr (sizeof...(Rest) == 0) {
        f(Eval{thresh2, d2max, rsc, cen});
    } else if (idx == I) {
        f(Eval{thresh2, d2max, rsc, cen});
    } else {
        bind_yukawa_level<EVAL_LEVEL, I + 1, Rest...>(idx, thresh2, d2max, rsc, cen, (F &&)f);
    }
}

// Carries CHECK_MIN into the eval nest, whose lambda is generic: a second `auto` parameter
// instantiates the body once per value, at the cost of duplicating its SASS.
template <bool B>
struct BoolTag {
    static constexpr bool value = B;
};

// Hands `f` the evaluator for a source box's level. Coefficients live in the evaluator
// type, so binding here keeps the level selection out of the per-pair inner loop.
template <typename Eval>
struct LevelBinder {
    template <typename F>
    __device__ __forceinline__ static void bind(int, bool check_min, Real thresh2, Real d2max, Real rsc, Real cen,
                                               F &&f) {
        if (check_min)
            f(Eval{thresh2, d2max, rsc, cen}, BoolTag<true>{});
        else
            f(Eval{thresh2, d2max, rsc, cen}, BoolTag<false>{});
    }
};

template <int EVAL_LEVEL, int LEVEL0, typename C0, typename... Rest>
struct LevelBinder<YukawaLevelsCuda<EVAL_LEVEL, LEVEL0, C0, Rest...>> {
    template <typename F>
    __device__ __forceinline__ static void bind(int level, bool check_min, Real thresh2, Real d2max, Real rsc, Real cen,
                                               F &&f) {
        if (check_min)
            bind_yukawa_level<EVAL_LEVEL, 0, C0, Rest...>(
                level - LEVEL0, thresh2, d2max, rsc, cen,
                [&](const auto &evaluator) { f(evaluator, BoolTag<true>{}); });
        else
            bind_yukawa_level<EVAL_LEVEL, 0, C0, Rest...>(
                level - LEVEL0, thresh2, d2max, rsc, cen,
                [&](const auto &evaluator) { f(evaluator, BoolTag<false>{}); });
    }
};

// Source pre-filtering (PREFILTER != 0).
//
// A warp evaluates one broadcast source against 32 targets in lockstep, so it pays a
// full Horner whenever *any* lane is in range. The useful question is therefore per
// target tile, not per pair: take the tight AABB of a CULL_TILE group of lanes'
// targets, measure each source's squared distance to that box, and drop the sources no
// lane in the group can reach. __ballot_sync + __popc compacts the survivors into
// shared memory so the accumulate loop stays dense and unrolled. The evaluator's own
// R2 < d2max mask remains the exact arbiter; this only removes work.
__device__ __forceinline__ Real cull_min(Real a, Real b) { return a < b ? a : b; }
__device__ __forceinline__ Real cull_max(Real a, Real b) { return a > b ? a : b; }

// Inflating the cutoff keeps the surviving set a superset of what the evaluator would
// accept even if nvrtc contracts the two R2 expressions differently, which is what
// lets PREFILTER output stay bit-identical to PREFILTER=0. A pair dropped at the
// boundary contributes below tolerance by construction, so this can go to zero.
constexpr Real kCullSlack = Real{1e-5};

// Stands in for +/-infinity when a lane holds no target: large enough to fail any
// cutoff, small enough that its square stays finite in fp32.
constexpr Real kCullFar = Real{1e18};

constexpr unsigned kFullWarp = 0xffffffffu;

// 16-byte load unit for the compacted source buffer, so a whole source arrives in one
// LDS.128 instead of SPATIAL_DIM + KERNEL_INPUT_DIM separate scalar loads.
struct alignas(16) CullVec {
    Real v[16 / sizeof(Real)];
};

template <typename Eval, int TILE, int TARGETS, int PREFILTER, int CULL_TILE>
__device__ __forceinline__ void DirectByBoxBody(dmk::cuda::DirectByBoxArgs<Real> a) {
    static_assert(TARGETS > 0, "TARGETS_PER_THREAD must be positive");
    static_assert(TARGETS <= 4, "TARGETS_PER_THREAD must be at most 4");
    static_assert(CULL_TILE > 0 && CULL_TILE <= 32 && (32 % CULL_TILE) == 0,
                  "CULL_TILE must be a power-of-two divisor of the warp size");

    // Cull groups per warp. NSG == 1 is the warp-wide tile.
    constexpr int NSG = 32 / CULL_TILE;

    constexpr int SPATIAL_DIM = Eval::SPATIAL_DIM;
    constexpr int KERNEL_INPUT_DIM = Eval::KERNEL_INPUT_DIM;
    constexpr int KERNEL_OUTPUT_DIM = Eval::KERNEL_OUTPUT_DIM;
    constexpr int NORMAL_DIM = Eval::NORMAL_DIM;
    constexpr Real scale_factor = Eval::scale_factor;

    constexpr int VPS = SPATIAL_DIM + KERNEL_INPUT_DIM + NORMAL_DIM;
    // Compacted sources are padded to a 16-byte multiple so one aligned vector load fetches a
    // whole source: for 3D scalar kernels VPS is already exactly 4 floats (x,y,z,q), turning
    // four scalar LDS in the innermost loop into a single LDS.128.
    constexpr int VEC_N = 16 / sizeof(Real);
    constexpr int VPS_PAD = ((VPS + VEC_N - 1) / VEC_N) * VEC_N;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    Real *smem = reinterpret_cast<Real *>(smem_raw);

    // The compacted survivor buffer is carved first so it structurally inherits smem_raw's
    // 16-byte alignment, which is what makes the vector load above legal -- ptxas cannot
    // prove alignment of a dynamically offset pointer.
    Real *s_cull_dat = nullptr;
    int *s_cull_idx = nullptr;
    if constexpr (PREFILTER == 3) {
        s_cull_dat = smem;
        smem += (blockDim.x / 32) * NSG * 32 * VPS_PAD;
    } else if constexpr (PREFILTER != 0) {
        s_cull_idx = reinterpret_cast<int *>(smem);
        smem += ((blockDim.x / 32) * NSG * 32 * sizeof(int) + sizeof(Real) - 1) / sizeof(Real);
    }

    // With data compaction each source is read once per warp per chunk, not once per target,
    // so the staging tile no longer amortises anything a warp reads twice -- it only shares
    // across the block's warps, which L1 does as well. Skipping it retires both
    // __syncthreads() (the top warp stall) and frees the shared that caps occupancy. The
    // other PREFILTER modes do re-read a source per target, so they always stage.
    constexpr bool STAGE = (PREFILTER != 3) || (STAGE_SRC != 0);

    Real *s_r_src = nullptr;
    Real *s_charge = nullptr;
    Real *s_normal = nullptr;
    if constexpr (STAGE) {
        s_r_src = smem;
        smem += TILE * SPATIAL_DIM;

        s_charge = smem;
        smem += TILE * KERNEL_INPUT_DIM;

        if constexpr (NORMAL_DIM > 0) {
            s_normal = smem;
            smem += TILE * NORMAL_DIM;
        }
    }

    // Cull-group target boxes, [warp][q][group][centre..half_extent]. These live in shared
    // rather than in registers because every lane reads the same box (a broadcast,
    // conflict-free): the register version cost +9 registers per lane at CULL_TILE 32 and +25
    // at 8, which is a larger occupancy loss than the cull saves.
    Real *s_cull_box = nullptr;
    if constexpr (PREFILTER != 0) {
        s_cull_box = smem;
        smem += (blockDim.x / 32) * TARGETS * NSG * 2 * SPATIAL_DIM;
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int cull_group = lane / CULL_TILE;

    const int trg_box_idx = blockIdx.x;
    if (trg_box_idx >= a.n_work) {
        return;
    }

    const int trg_box = a.direct_work[trg_box_idx];
    const int n_targets = a.target_counts[trg_box];
    if (n_targets == 0) {
        return;
    }

    const int trg_level = a.box_levels[trg_box];
    const int n_list1 = a.list1_count[trg_box];

    const Real *__restrict__ r_targets = a.r_target_flat + a.r_target_offsets[trg_box];
    Real *__restrict__ pot_targets = a.pot_flat + a.pot_offsets[trg_box];

    const int target_stride = blockDim.x * TARGETS;
    const int n_target_rounds = (n_targets + target_stride - 1) / target_stride;

    for (int tr = 0; tr < n_target_rounds; ++tr) {
        const int t_base = tr * target_stride + threadIdx.x;

        bool active_target[TARGETS];
        int target_idx[TARGETS];
        bool any_active_target = false;
        Real xt[TARGETS][SPATIAL_DIM];

#pragma unroll
        for (int q = 0; q < TARGETS; ++q) {
            const int t = t_base + q * blockDim.x;
            const bool active = t < n_targets;
            active_target[q] = active;
            target_idx[q] = t;
            any_active_target = any_active_target || active;
            if (active) {
#pragma unroll
                for (int k = 0; k < SPATIAL_DIM; ++k) {
                    xt[q][k] = r_targets[t * SPATIAL_DIM + k];
                }
            } else {
                // The AABB reduce reads every lane's xt, so it must not be garbage.
                if constexpr (PREFILTER != 0) {
#pragma unroll
                    for (int k = 0; k < SPATIAL_DIM; ++k) {
                        xt[q][k] = Real{0};
                    }
                }
            }
        }

        Real vt[TARGETS][KERNEL_OUTPUT_DIM];
#pragma unroll
        for (int q = 0; q < TARGETS; ++q) {
#pragma unroll
            for (int k = 0; k < KERNEL_OUTPUT_DIM; ++k) {
                vt[q][k] = Real{0};
            }
        }

        // Tight AABB of each cull group's targets, published to shared by the group leader.
        // A lane with no target contributes +/-kCullFar so it cannot widen the box; a group
        // that is entirely inactive collapses to an empty box that rejects every source.
        // This belongs here rather than in the source loop: the boxes depend only on the
        // target round, so publishing them per source chunk per list1 entry cost
        // 2*SPATIAL_DIM shuffles thousands of times over.
        //
        // Centre and half-extent, so the per-source test is max(0, |x - c| - h) and the abs
        // rides along as a source modifier. Rounding in c and h moves the effective box by
        // ~1 ulp, far below kCullSlack, so survivors stay a superset.
        if constexpr (PREFILTER != 0) {
#pragma unroll
            for (int q = 0; q < TARGETS; ++q) {
                Real *box = s_cull_box + ((warp * TARGETS + q) * NSG + cull_group) * 2 * SPATIAL_DIM;
#pragma unroll
                for (int k = 0; k < SPATIAL_DIM; ++k) {
                    Real vlo = active_target[q] ? xt[q][k] : kCullFar;
                    Real vhi = active_target[q] ? xt[q][k] : -kCullFar;
#pragma unroll
                    for (int d = CULL_TILE >> 1; d > 0; d >>= 1) {
                        vlo = cull_min(vlo, __shfl_xor_sync(kFullWarp, vlo, d));
                        vhi = cull_max(vhi, __shfl_xor_sync(kFullWarp, vhi, d));
                    }
                    if (lane % CULL_TILE == 0) {
                        box[k] = Real{0.5} * (vhi + vlo);
                        box[SPATIAL_DIM + k] = Real{0.5} * (vhi - vlo);
                    }
                }
            }
            __syncwarp();
        }

        for (int li = 0; li < n_list1; ++li) {
            const int src_box = a.list1_flat[trg_box * a.nlist1_stride + li];

            int src_level = a.box_levels[src_box];
            if (a.ifpwexp[src_box] && src_box == trg_box) {
                src_level = src_level + 1;
            } else if (src_level < trg_level) {
                src_level = trg_level;
            }

            const int n_src = a.src_counts[src_box];
            const Real *__restrict__ r_src = a.r_src_flat + a.r_src_offsets[src_box];
            const Real *__restrict__ charge = a.charge_flat + a.charge_offsets[src_box];

            const Real *__restrict__ normals = nullptr;
            if constexpr (NORMAL_DIM > 0) {
                normals = a.normal_flat + a.normal_offsets[src_box];
            }

            const Real rsc = a.direct_rsc[src_level];
            const Real cen = a.direct_cen[src_level];
            const Real d2max = a.direct_d2max[src_level];

            // PBC: a wrapped list1 entry names its own box, so the source coordinates must
            // be translated into the target's image. Constant over the tile loop.
            Real shift[SPATIAL_DIM] = {};
            if constexpr (PERIODIC) {
                const signed char *sh = a.list1_shift + (trg_box * a.nlist1_stride + li) * SPATIAL_DIM;
                for (int k = 0; k < SPATIAL_DIM; ++k)
                    shift[k] = sh[k];
            }

            LevelBinder<Eval>::bind(src_level, src_box == trg_box, a.thresh2, d2max, rsc, cen,
                                    [&](const auto &evaluator, auto check_min) {
                // Unstaged, the whole source list is one pass: nothing is shared between
                // chunks, so there is no tile to size.
                const int tile_step = STAGE ? TILE : n_src;
                for (int tile0 = 0; tile0 < n_src; tile0 += tile_step) {
                    const int rem = n_src - tile0;
                    const int tile_count = rem < tile_step ? rem : tile_step;

                    if constexpr (STAGE) {
                        // The shared layout matches the global one, so staging is a contiguous
                        // copy; only the periodic shift needs idx split into (source, component).
                        if constexpr (PERIODIC) {
                            for (int idx = threadIdx.x; idx < tile_count * SPATIAL_DIM; idx += blockDim.x)
                                s_r_src[idx] = r_src[tile0 * SPATIAL_DIM + idx] + shift[idx % SPATIAL_DIM];
                        } else {
                            for (int idx = threadIdx.x; idx < tile_count * SPATIAL_DIM; idx += blockDim.x)
                                s_r_src[idx] = r_src[tile0 * SPATIAL_DIM + idx];
                        }
                        for (int idx = threadIdx.x; idx < tile_count * KERNEL_INPUT_DIM; idx += blockDim.x)
                            s_charge[idx] = charge[tile0 * KERNEL_INPUT_DIM + idx];

                        if constexpr (NORMAL_DIM > 0) {
                            for (int idx = threadIdx.x; idx < tile_count * NORMAL_DIM; idx += blockDim.x)
                                s_normal[idx] = normals[tile0 * NORMAL_DIM + idx];
                        }

                        __syncthreads();
                    }

                    if constexpr (PREFILTER == 0) {
                        if (any_active_target) {
#pragma unroll 4
                            for (int ss = 0; ss < tile_count; ++ss) {
                                Real xs[SPATIAL_DIM];
#pragma unroll
                                for (int k = 0; k < SPATIAL_DIM; ++k) {
                                    xs[k] = s_r_src[ss * SPATIAL_DIM + k];
                                }

                                Real vs[KERNEL_INPUT_DIM];
#pragma unroll
                                for (int k = 0; k < KERNEL_INPUT_DIM; ++k) {
                                    vs[k] = s_charge[ss * KERNEL_INPUT_DIM + k];
                                }

                                Real dX[SPATIAL_DIM];
                                if constexpr (NORMAL_DIM > 0) {
                                    Real ns[NORMAL_DIM > 0 ? NORMAL_DIM : 1];
#pragma unroll
                                    for (int k = 0; k < NORMAL_DIM; ++k) {
                                        ns[k] = s_normal[ss * NORMAL_DIM + k];
                                    }
#pragma unroll
                                    for (int q = 0; q < TARGETS; ++q) {
                                        if (active_target[q]) {
#pragma unroll
                                            for (int k = 0; k < SPATIAL_DIM; ++k) {
                                                dX[k] = xt[q][k] - xs[k];
                                            }
                                            direct_eval_accumulate<check_min.value>(evaluator, vt[q], dX, vs, ns);
                                        }
                                    }
                                } else {
#pragma unroll
                                    for (int q = 0; q < TARGETS; ++q) {
                                        if (active_target[q]) {
#pragma unroll
                                            for (int k = 0; k < SPATIAL_DIM; ++k) {
                                                dX[k] = xt[q][k] - xs[k];
                                            }
                                            direct_eval_accumulate<check_min.value>(evaluator, vt[q], dX, vs);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Every lane must reach __ballot_sync, so there is no
                        // any_active_target guard here: an idle lane still tests its
                        // source and just skips the accumulate.
                        // PREFILTER == 2 keeps every source by making the threshold
                        // unreachable rather than by skipping the test, so the distance
                        // computation stays live and the runtime delta against
                        // PREFILTER == 0 is the cull overhead alone, output unchanged.
                        // The bound is derived from a runtime value so it cannot be folded.
                        const Real d2cull = (PREFILTER == 2) ? d2max * kCullFar : d2max * (Real{1} + kCullSlack);

                        // Chunk outer, target inner: the source payload below depends only on
                        // the chunk. The q loop is unrolled so xt/vt stay register-indexed.
                        for (int chunk = 0; chunk < tile_count; chunk += 32) {
                            const int s = chunk + lane;
                            const bool in_tile = s < tile_count;

                            // The source this lane culls, assembled in registers so the
                            // compacting store below is one aligned vector write. Its leading
                            // SPATIAL_DIM entries double as the cull coordinates. A survivor
                            // travels cull_src -> s_cull_dat -> cull_dat.
                            constexpr int CULL_SRC_N = (PREFILTER == 3) ? VPS_PAD : SPATIAL_DIM;
                            alignas(16) Real cull_src[CULL_SRC_N];
                            if (in_tile) {
                                // Unstaged, the periodic shift rides here instead of on the
                                // staging copy. `s` already indexes the whole list, since an
                                // unstaged pass has a single tile.
#pragma unroll
                                for (int k = 0; k < SPATIAL_DIM; ++k) {
                                    if constexpr (STAGE) {
                                        cull_src[k] = s_r_src[s * SPATIAL_DIM + k];
                                    } else if constexpr (PERIODIC) {
                                        cull_src[k] = r_src[s * SPATIAL_DIM + k] + shift[k];
                                    } else {
                                        cull_src[k] = r_src[s * SPATIAL_DIM + k];
                                    }
                                }
                                if constexpr (PREFILTER == 3) {
#pragma unroll
                                    for (int k = 0; k < KERNEL_INPUT_DIM; ++k)
                                        cull_src[SPATIAL_DIM + k] =
                                            STAGE ? s_charge[s * KERNEL_INPUT_DIM + k] : charge[s * KERNEL_INPUT_DIM + k];
                                    if constexpr (NORMAL_DIM > 0) {
#pragma unroll
                                        for (int k = 0; k < NORMAL_DIM; ++k)
                                            cull_src[SPATIAL_DIM + KERNEL_INPUT_DIM + k] =
                                                STAGE ? s_normal[s * NORMAL_DIM + k] : normals[s * NORMAL_DIM + k];
                                    }
                                    // The vector store writes the padding too, so it must not
                                    // be an uninitialised read.
#pragma unroll
                                    for (int k = VPS; k < CULL_SRC_N; ++k)
                                        cull_src[k] = Real{0};
                                }
                            }

#pragma unroll
                            for (int q = 0; q < TARGETS; ++q) {
                                // One ballot per cull group: the whole warp tests its own
                                // source against group g's box, so every group gets its own
                                // survivor list built 32 sources at a time.
                                int my_count = 0;
                                int work_count = 0;
#pragma unroll
                                for (int g = 0; g < NSG; ++g) {
                                    const Real *box =
                                        s_cull_box + ((warp * TARGETS + q) * NSG + g) * 2 * SPATIAL_DIM;
                                    Real d2 = Real{0};
                                    if (in_tile) {
#pragma unroll
                                        for (int k = 0; k < SPATIAL_DIM; ++k) {
                                            const Real gap =
                                                cull_max(Real{0}, fabs(cull_src[k] - box[k]) - box[SPATIAL_DIM + k]);
                                            d2 = fma(gap, gap, d2);
                                        }
                                    }

                                    const bool keep = in_tile && (d2 < d2cull);
                                    const unsigned m = __ballot_sync(kFullWarp, keep);
                                    if (keep) {
                                        const int rank = __popc(m & ((1u << lane) - 1));
                                        if constexpr (PREFILTER == 3) {
                                            // Compact the source DATA, not its index, so the
                                            // evaluation loop addresses off its own counter
                                            // instead of chasing a shared index -- that is a
                                            // dependent LDS -> LDS chain the unroll cannot hide.
                                            Real *dst = s_cull_dat + ((warp * NSG + g) * 32 + rank) * VPS_PAD;
#pragma unroll
                                            for (int v = 0; v < VPS_PAD / VEC_N; ++v)
                                                *reinterpret_cast<CullVec *>(dst + v * VEC_N) =
                                                    *reinterpret_cast<const CullVec *>(&cull_src[v * VEC_N]);
                                        } else {
                                            s_cull_idx[(warp * NSG + g) * 32 + rank] = s;
                                        }
                                    }
                                    const int count = __popc(m);
                                    if (g == cull_group) {
                                        my_count = count;
                                    }

                                    if constexpr (PREFILTER_STATS) {
                                        // The warp iterates max-over-groups times, not
                                        // mean-over-groups, so that max is the real work metric
                                        // and the only thing comparable across CULL_TILE.
                                        if (count > work_count)
                                            work_count = count;
                                        const unsigned m_in = __ballot_sync(kFullWarp, in_tile);
                                        if (lane == 0) {
                                            atomicAdd(&a.cull_stats[0], (unsigned long long)__popc(m_in));
                                            atomicAdd(&a.cull_stats[1], (unsigned long long)count);
                                        }
                                    }
                                }
                                if constexpr (PREFILTER_STATS) {
                                    if (lane == 0)
                                        atomicAdd(&a.cull_stats[5], (unsigned long long)work_count);
                                }
                                __syncwarp();

                                // What does the warp genuinely need? For each source, which
                                // lanes actually have it in range: per cull group (the cull's
                                // own ceiling), per warp (what the production kernel's per-pair
                                // branch already skips for free), and per lane-pair (the floor
                                // any granularity could ever reach).
                                if constexpr (PREFILTER_STATS) {
                                    for (int j = 0; j < 32; ++j) {
                                        const int ss = chunk + j;
                                        bool hit = false;
                                        if (ss < tile_count && active_target[q]) {
                                            Real d2 = Real{0};
#pragma unroll
                                            for (int k = 0; k < SPATIAL_DIM; ++k) {
                                                Real xk = STAGE ? s_r_src[ss * SPATIAL_DIM + k]
                                                                : r_src[ss * SPATIAL_DIM + k];
                                                if constexpr (!STAGE && PERIODIC)
                                                    xk += shift[k];
                                                const Real dd = xt[q][k] - xk;
                                                d2 = fma(dd, dd, d2);
                                            }
                                            hit = (d2 > a.thresh2) && (d2 < d2max);
                                        }
                                        const unsigned m_hit = __ballot_sync(kFullWarp, hit);
                                        if (lane == 0 && ss < tile_count) {
                                            // Per group, so this is directly comparable to
                                            // cull_stats[1] at any CULL_TILE.
                                            int needed_groups = 0;
#pragma unroll
                                            for (int g = 0; g < NSG; ++g) {
                                                const unsigned gmask =
                                                    (CULL_TILE == 32) ? kFullWarp
                                                                      : (((1u << CULL_TILE) - 1u) << (g * CULL_TILE));
                                                needed_groups += (m_hit & gmask) != 0u;
                                            }
                                            atomicAdd(&a.cull_stats[2], (unsigned long long)needed_groups);
                                            atomicAdd(&a.cull_stats[3], (unsigned long long)__popc(m_hit));
                                            if (m_hit != 0u)
                                                atomicAdd(&a.cull_stats[4], 1ull);
                                        }
                                    }
                                }

                                // A divergent trip count is fine: each cull group reads only
                                // its own list and there is no warp-collective op inside.
                                if (active_target[q]) {
#pragma unroll EVAL_UNROLL
                                    for (int j = 0; j < my_count; ++j) {
                                        Real xs[SPATIAL_DIM];
                                        Real vs[KERNEL_INPUT_DIM];
                                        [[maybe_unused]] Real ns[NORMAL_DIM > 0 ? NORMAL_DIM : 1];

                                        if constexpr (PREFILTER == 3) {
                                            const Real *sp =
                                                s_cull_dat + ((warp * NSG + cull_group) * 32 + j) * VPS_PAD;
                                            // Register copy of the shared survivor: aligned, so
                                            // a 3D scalar kernel gets one LDS.128.
                                            alignas(16) Real cull_dat[VPS_PAD];
#pragma unroll
                                            for (int v = 0; v < VPS_PAD / VEC_N; ++v)
                                                *reinterpret_cast<CullVec *>(&cull_dat[v * VEC_N]) =
                                                    *reinterpret_cast<const CullVec *>(sp + v * VEC_N);
#pragma unroll
                                            for (int k = 0; k < SPATIAL_DIM; ++k)
                                                xs[k] = cull_dat[k];
#pragma unroll
                                            for (int k = 0; k < KERNEL_INPUT_DIM; ++k)
                                                vs[k] = cull_dat[SPATIAL_DIM + k];
                                            if constexpr (NORMAL_DIM > 0) {
#pragma unroll
                                                for (int k = 0; k < NORMAL_DIM; ++k)
                                                    ns[k] = cull_dat[SPATIAL_DIM + KERNEL_INPUT_DIM + k];
                                            }
                                        } else {
                                            const int ss = s_cull_idx[(warp * NSG + cull_group) * 32 + j];
#pragma unroll
                                            for (int k = 0; k < SPATIAL_DIM; ++k)
                                                xs[k] = s_r_src[ss * SPATIAL_DIM + k];
#pragma unroll
                                            for (int k = 0; k < KERNEL_INPUT_DIM; ++k)
                                                vs[k] = s_charge[ss * KERNEL_INPUT_DIM + k];
                                            if constexpr (NORMAL_DIM > 0) {
#pragma unroll
                                                for (int k = 0; k < NORMAL_DIM; ++k)
                                                    ns[k] = s_normal[ss * NORMAL_DIM + k];
                                            }
                                        }

                                        Real dX[SPATIAL_DIM];
#pragma unroll
                                        for (int k = 0; k < SPATIAL_DIM; ++k)
                                            dX[k] = xt[q][k] - xs[k];

                                        if constexpr (NORMAL_DIM > 0) {
                                            direct_eval_accumulate<check_min.value>(evaluator, vt[q], dX, vs, ns);
                                        } else {
                                            direct_eval_accumulate<check_min.value>(evaluator, vt[q], dX, vs);
                                        }
                                    }
                                }

                                // The survivor buffer is reused by the next chunk and the next
                                // q, and a lane may write any cull group's slots while another
                                // is still reading them. Independent thread scheduling does not
                                // guarantee the loop above reconverges before those stores.
                                __syncwarp();
                            }
                        }
                    }

                    // Guards the staged tile against the next iteration's overwrite; the
                    // survivor buffers are per-warp and fenced by __syncwarp above.
                    if constexpr (STAGE)
                        __syncthreads();
                }
            });
        }

#pragma unroll
        for (int q = 0; q < TARGETS; ++q) {
            if (active_target[q]) {
#pragma unroll
                for (int k = 0; k < KERNEL_OUTPUT_DIM; ++k) {
                    pot_targets[target_idx[q] * KERNEL_OUTPUT_DIM + k] = vt[q][k] * scale_factor;
                }
            }
        }
    }
}

using Evaluator = DMK_DIRECT_EVALUATOR;
using DirectArgs = dmk::cuda::DirectByBoxArgs<Real>;

// KERNEL_START

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE) DMK_DIRECT_KERNEL_NAME(DirectArgs a) {
    DirectByBoxBody<Evaluator, SRC_TILE, TARGETS_PER_THREAD, PREFILTER, CULL_TILE>(a);
}
