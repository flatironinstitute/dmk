#include <dmk/cuda/direct_kernelargs.hpp>

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

template <typename Coeffs>
struct LaplacePolyEvaluator2DCuda {
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real thresh2;
    Real d2max;
    Real rsc;
    Real cen;

    __device__ inline void operator()(Real (&u)[1][1], const Real (&dX)[2]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        const bool in_range = (R2 > thresh2) && (R2 < d2max);

        if (!in_range) {
            u[0][0] = Real{0};
            return;
        }

        const Real R2sc = R2 * (Real{0.5} * rsc);
        const Real arg = rsc * R2 + cen;
        const Real ptmp = horner_const<Coeffs>(arg);

        u[0][0] = Real{0.5} * log(R2sc) + ptmp;
    }
};

template <typename Coeffs>
struct LaplacePolyEvaluator3DCuda {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real thresh2;
    Real d2max;
    Real rsc;
    Real cen;

    __device__ inline void operator()(Real (&u)[1][1], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];

        const bool in_range = (R2 > thresh2) && (R2 < d2max);

        if (!in_range) {
            u[0][0] = Real{0};
            return;
        }

        const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
        const Real xmapped = (R2 * Rinv + cen) * rsc;
        const Real P = horner_const<Coeffs>(xmapped);

        u[0][0] = P * Rinv;
    }
};

template <typename Coeffs>
struct SqrtLaplacePolyEvaluator2DCuda {
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real thresh2;
    Real d2max;
    Real rsc;
    Real cen;

    __device__ inline void operator()(Real (&u)[1][1], const Real (&dX)[2]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        const bool in_range = (R2 > thresh2) && (R2 < d2max);

        if (!in_range) {
            u[0][0] = Real{0};
            return;
        }

        const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
        const Real xmapped = (R2 * Rinv + cen) * rsc;

        u[0][0] = horner_const<Coeffs>(xmapped) * Rinv;
    }
};

template <typename Coeffs>
struct SqrtLaplacePolyEvaluator3DCuda {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real thresh2;
    Real d2max;
    Real rsc;
    Real cen;

    __device__ inline void operator()(Real (&u)[1][1], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];

        const bool in_range = (R2 > thresh2) && (R2 < d2max);

        if (!in_range) {
            u[0][0] = Real{0};
            return;
        }

        const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
        const Real R2inv = Rinv * Rinv;
        const Real arg = rsc * R2 + cen;

        u[0][0] = R2inv * horner_const<Coeffs>(arg);
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

    __device__ inline void operator()(Real (&u)[3][3], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];

        const bool in_range = (R2 > thresh2) && (R2 < d2max);

        if (!in_range) {
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 3; ++i) {
                    u[j][i] = Real{0};
                }
            }
            return;
        }

        const Real half = Real{0.5};
        const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
        const Real Rinv3 = Rinv * Rinv * Rinv;
        const Real xtmp = (R2 * Rinv + cen) * rsc;

        const Real fdiag = (half - horner_const<CoeffsDiag>(xtmp)) * Rinv;

        const Real foffd = (half - horner_const<CoeffsOffdiag>(xtmp)) * Rinv3;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Real val = foffd * dX[j] * dX[i];

                if (i == j) {
                    val += fdiag;
                }

                u[i][j] = val;
            }
        }
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

    __device__ inline void operator()(Real (&u)[3][3], const Real (&dX)[3], const Real (&ns)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];

        const bool in_range = (R2 > thresh2) && (R2 < d2max);

        const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
        const Real Rinv3 = Rinv * Rinv * Rinv;
        const Real Rinv5 = Rinv3 * Rinv * Rinv;
        const Real xtmp = (R2 * Rinv + cen) * rsc;

        const Real Fdiag = -horner_const<CoeffsDiag>(xtmp) * Rinv3;

        const Real Foffd = Real{6} * horner_const<CoeffsOffdiag>(xtmp) * Rinv5;

        const Real rdotn = dX[0] * ns[0] + dX[1] * ns[1] + dX[2] * ns[2];

        const Real Fdiag_rdotn = Fdiag * rdotn;

        for (int j = 0; j < 3; ++j) {
            const Real foffd_rj_rdotn = Foffd * dX[j] * rdotn;
            const Real fdiag_nj = Fdiag * ns[j];
            const Real fdiag_rj = Fdiag * dX[j];

            for (int i = 0; i < 3; ++i) {
                Real val = foffd_rj_rdotn * dX[i] + fdiag_nj * dX[i] + fdiag_rj * ns[i];

                if (i == j) {
                    val += Fdiag_rdotn;
                }

                u[j][i] = in_range ? val : Real{0};
            }
        }
    }
};

template <typename Eval>
__device__ __forceinline__ void direct_eval_accumulate(
    const Eval &evaluator,
    Real (&vt)[Eval::KERNEL_OUTPUT_DIM],
    const Real (&dX)[Eval::SPATIAL_DIM],
    const Real (&vs)[Eval::KERNEL_INPUT_DIM]
) {
    Real U[Eval::KERNEL_INPUT_DIM][Eval::KERNEL_OUTPUT_DIM];
    evaluator(U, dX);

#pragma unroll
    for (int k0 = 0; k0 < Eval::KERNEL_INPUT_DIM; ++k0) {
#pragma unroll
        for (int k1 = 0; k1 < Eval::KERNEL_OUTPUT_DIM; ++k1) {
            vt[k1] += U[k0][k1] * vs[k0];
        }
    }
}

template <typename Eval>
__device__ __forceinline__ void direct_eval_accumulate(
    const Eval &evaluator,
    Real (&vt)[Eval::KERNEL_OUTPUT_DIM],
    const Real (&dX)[Eval::SPATIAL_DIM],
    const Real (&vs)[Eval::KERNEL_INPUT_DIM],
    const Real (&ns)[Eval::NORMAL_DIM]
) {
    Real U[Eval::KERNEL_INPUT_DIM][Eval::KERNEL_OUTPUT_DIM];
    evaluator(U, dX, ns);

#pragma unroll
    for (int k0 = 0; k0 < Eval::KERNEL_INPUT_DIM; ++k0) {
#pragma unroll
        for (int k1 = 0; k1 < Eval::KERNEL_OUTPUT_DIM; ++k1) {
            vt[k1] += U[k0][k1] * vs[k0];
        }
    }
}

template <typename CoeffsDiag, typename CoeffsOffdiag>
__device__ __forceinline__ void direct_eval_accumulate(
    const StokesletPolyEvaluator3DCuda<CoeffsDiag, CoeffsOffdiag> &evaluator,
    Real (&vt)[3],
    const Real (&dX)[3],
    const Real (&vs)[3]
) {
    const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
    const bool in_range = (R2 > evaluator.thresh2) && (R2 < evaluator.d2max);

    if (!in_range) {
        return;
    }

    const Real half = Real{0.5};
    const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
    const Real Rinv3 = Rinv * Rinv * Rinv;
    const Real xtmp = (R2 * Rinv + evaluator.cen) * evaluator.rsc;

    const Real fdiag = (half - horner_const<CoeffsDiag>(xtmp)) * Rinv;
    const Real foffd = (half - horner_const<CoeffsOffdiag>(xtmp)) * Rinv3;
    const Real rdotv = dX[0] * vs[0] + dX[1] * vs[1] + dX[2] * vs[2];
    const Real off = foffd * rdotv;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        vt[i] += fdiag * vs[i] + off * dX[i];
    }
}

template <typename CoeffsDiag, typename CoeffsOffdiag>
__device__ __forceinline__ void direct_eval_accumulate(
    const StressletPolyEvaluator3DCuda<CoeffsDiag, CoeffsOffdiag> &evaluator,
    Real (&vt)[3],
    const Real (&dX)[3],
    const Real (&vs)[3],
    const Real (&ns)[3]
) {
    const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
    const bool in_range = (R2 > evaluator.thresh2) && (R2 < evaluator.d2max);

    if (!in_range) {
        return;
    }

    const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
    const Real Rinv3 = Rinv * Rinv * Rinv;
    const Real Rinv5 = Rinv3 * Rinv * Rinv;
    const Real xtmp = (R2 * Rinv + evaluator.cen) * evaluator.rsc;

    const Real Fdiag = -horner_const<CoeffsDiag>(xtmp) * Rinv3;
    const Real Foffd = Real{6} * horner_const<CoeffsOffdiag>(xtmp) * Rinv5;

    const Real rdotn = dX[0] * ns[0] + dX[1] * ns[1] + dX[2] * ns[2];
    const Real rdotv = dX[0] * vs[0] + dX[1] * vs[1] + dX[2] * vs[2];
    const Real ndotv = ns[0] * vs[0] + ns[1] * vs[1] + ns[2] * vs[2];

    const Real r_scale = Foffd * rdotn * rdotv + Fdiag * ndotv;
    const Real n_scale = Fdiag * rdotv;
    const Real v_scale = Fdiag * rdotn;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        vt[i] += r_scale * dX[i] + n_scale * ns[i] + v_scale * vs[i];
    }
}

using Evaluator = DMK_DIRECT_EVALUATOR;

template <typename Eval, int TILE>
__device__ __forceinline__ void DirectByBoxBody(dmk::cuda::DirectByBoxArgs<Real> a) {
    constexpr int SPATIAL_DIM = Eval::SPATIAL_DIM;
    constexpr int KERNEL_INPUT_DIM = Eval::KERNEL_INPUT_DIM;
    constexpr int KERNEL_OUTPUT_DIM = Eval::KERNEL_OUTPUT_DIM;
    constexpr int NORMAL_DIM = Eval::NORMAL_DIM;
    constexpr Real scale_factor = Eval::scale_factor;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    Real *smem = reinterpret_cast<Real *>(smem_raw);

    Real *s_r_src = smem;
    smem += TILE * SPATIAL_DIM;

    Real *s_charge = smem;
    smem += TILE * KERNEL_INPUT_DIM;

    Real *s_normal = nullptr;

    if constexpr (NORMAL_DIM > 0) {
        s_normal = smem;
        smem += TILE * NORMAL_DIM;
    }

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

    const int n_target_rounds = (n_targets + blockDim.x - 1) / blockDim.x;

    for (int tr = 0; tr < n_target_rounds; ++tr) {
        const int t = tr * blockDim.x + threadIdx.x;
        const bool active_target = t < n_targets;

        Real xt[SPATIAL_DIM];

        if (active_target) {
#pragma unroll
            for (int k = 0; k < SPATIAL_DIM; ++k) {
                xt[k] = r_targets[t * SPATIAL_DIM + k];
            }
        }

        Real vt[KERNEL_OUTPUT_DIM];

#pragma unroll
        for (int k = 0; k < KERNEL_OUTPUT_DIM; ++k) {
            vt[k] = Real{0};
        }

        for (int li = 0; li < n_list1; ++li) {
            const int src_box = a.list1_flat[trg_box * a.nlist1_stride + li];

            int src_level = a.box_levels[src_box];

            if (a.ifpwexp[src_box] && src_box == trg_box) {
                src_level = src_level + 1;
            } else if (src_level < trg_level) {
                src_level = trg_level;
            }

            if (src_level >= a.n_levels) {
                src_level = a.n_levels - 1;
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

            Eval evaluator{a.thresh2, d2max, rsc, cen};

            for (int tile0 = 0; tile0 < n_src; tile0 += TILE) {
                const int rem = n_src - tile0;
                const int tile_count = rem < TILE ? rem : TILE;

                for (int idx = threadIdx.x; idx < tile_count * SPATIAL_DIM; idx += blockDim.x) {
                    const int s = idx / SPATIAL_DIM;
                    const int k = idx - s * SPATIAL_DIM;

                    s_r_src[s * SPATIAL_DIM + k] = r_src[(tile0 + s) * SPATIAL_DIM + k];
                }

                for (int idx = threadIdx.x; idx < tile_count * KERNEL_INPUT_DIM; idx += blockDim.x) {
                    const int s = idx / KERNEL_INPUT_DIM;
                    const int k = idx - s * KERNEL_INPUT_DIM;

                    s_charge[s * KERNEL_INPUT_DIM + k] = charge[(tile0 + s) * KERNEL_INPUT_DIM + k];
                }

                if constexpr (NORMAL_DIM > 0) {
                    for (int idx = threadIdx.x; idx < tile_count * NORMAL_DIM; idx += blockDim.x) {
                        const int s = idx / NORMAL_DIM;
                        const int k = idx - s * NORMAL_DIM;

                        s_normal[s * NORMAL_DIM + k] = normals[(tile0 + s) * NORMAL_DIM + k];
                    }
                }

                __syncthreads();

                if (active_target) {
#pragma unroll
                    for (int ss = 0; ss < tile_count; ++ss) {
                        Real xs[SPATIAL_DIM];

#pragma unroll
                        for (int k = 0; k < SPATIAL_DIM; ++k) {
                            xs[k] = s_r_src[ss * SPATIAL_DIM + k];
                        }

                        Real dX[SPATIAL_DIM];

#pragma unroll
                        for (int k = 0; k < SPATIAL_DIM; ++k) {
                            dX[k] = xt[k] - xs[k];
                        }

                        Real vs[KERNEL_INPUT_DIM];

#pragma unroll
                        for (int k = 0; k < KERNEL_INPUT_DIM; ++k) {
                            vs[k] = s_charge[ss * KERNEL_INPUT_DIM + k];
                        }

                        if constexpr (NORMAL_DIM > 0) {
                            Real ns[NORMAL_DIM];

#pragma unroll
                            for (int k = 0; k < NORMAL_DIM; ++k) {
                                ns[k] = s_normal[ss * NORMAL_DIM + k];
                            }

                            direct_eval_accumulate(evaluator, vt, dX, vs, ns);
                        } else {
                            direct_eval_accumulate(evaluator, vt, dX, vs);
                        }
                    }
                }

                __syncthreads();
            }
        }

        if (active_target) {
#pragma unroll
            for (int k = 0; k < KERNEL_OUTPUT_DIM; ++k) {
                pot_targets[t * KERNEL_OUTPUT_DIM + k] += vt[k] * scale_factor;
            }
        }
    }
}

using Evaluator = DMK_DIRECT_EVALUATOR;
using DirectArgs = dmk::cuda::DirectByBoxArgs<Real>;

// KERNEL_START

extern "C" __global__ void DMK_DIRECT_KERNEL_NAME(DirectArgs a) { DirectByBoxBody<Evaluator, SRC_TILE>(a); }
