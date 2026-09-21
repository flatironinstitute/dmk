#pragma once
// Residual polynomial evaluators for the NVRTC device paths, shared by pt/direct.cu and
// esp/short_range.cu, which differ only in their coefficients.
//
// Requires `Real` to be already defined; the prelude emits it. NVRTC provides no standard headers,
// so nothing here may use <cmath> or std::.

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
