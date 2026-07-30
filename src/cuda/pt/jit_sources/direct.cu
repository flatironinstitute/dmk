// V2 point-tree near-field (direct) device kernel. The launcher prepends a
// prelude defining Real, the DMK_DIRECT_KERNEL_NAME symbol, the baked-literal
// coefficient struct(s), the DMK_DIRECT_EVALUATOR typedef, and the SRC_TILE /
// BLOCK_SIZE / TARGETS_PER_THREAD constants. Coefficients are compile-time
// literals folded straight into the FMAs (no runtime coeff buffer, no AOT).
//
// Scalar potential kernels (Laplace, Sqrt-Laplace; 2D + 3D).

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

template <typename Eval>
__device__ __forceinline__ void direct_eval_accumulate(const Eval &evaluator, Real (&vt)[Eval::KERNEL_OUTPUT_DIM],
                                                        const Real (&dX)[Eval::SPATIAL_DIM],
                                                        const Real (&vs)[Eval::KERNEL_INPUT_DIM]) {
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

template <typename Eval, int TILE, int TARGETS>
__device__ __forceinline__ void DirectByBoxBody(dmk::cuda::DirectByBoxArgs<Real> a) {
    static_assert(TARGETS > 0, "TARGETS_PER_THREAD must be positive");
    static_assert(TARGETS <= 4, "TARGETS_PER_THREAD must be at most 4");

    constexpr int SPATIAL_DIM = Eval::SPATIAL_DIM;
    constexpr int KERNEL_INPUT_DIM = Eval::KERNEL_INPUT_DIM;
    constexpr int KERNEL_OUTPUT_DIM = Eval::KERNEL_OUTPUT_DIM;
    constexpr Real scale_factor = Eval::scale_factor;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    Real *smem = reinterpret_cast<Real *>(smem_raw);

    Real *s_r_src = smem;
    smem += TILE * SPATIAL_DIM;

    Real *s_charge = smem;
    smem += TILE * KERNEL_INPUT_DIM;

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

            const Real rsc = a.direct_rsc[src_level];
            const Real cen = a.direct_cen[src_level];
            const Real d2max = a.direct_d2max[src_level];

            Eval evaluator{a.thresh2, d2max, rsc, cen};

            for (int tile0 = 0; tile0 < n_src; tile0 += TILE) {
                const int rem = n_src - tile0;
                const int tile_count = rem < TILE ? rem : TILE;

                for (int idx = threadIdx.x; idx < tile_count * SPATIAL_DIM; idx += blockDim.x) {
                    const int ss = idx / SPATIAL_DIM;
                    const int k = idx - ss * SPATIAL_DIM;
                    s_r_src[ss * SPATIAL_DIM + k] = r_src[(tile0 + ss) * SPATIAL_DIM + k];
                }
                for (int idx = threadIdx.x; idx < tile_count * KERNEL_INPUT_DIM; idx += blockDim.x) {
                    const int ss = idx / KERNEL_INPUT_DIM;
                    const int k = idx - ss * KERNEL_INPUT_DIM;
                    s_charge[ss * KERNEL_INPUT_DIM + k] = charge[(tile0 + ss) * KERNEL_INPUT_DIM + k];
                }

                __syncthreads();

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
#pragma unroll
                        for (int q = 0; q < TARGETS; ++q) {
                            if (active_target[q]) {
#pragma unroll
                                for (int k = 0; k < SPATIAL_DIM; ++k) {
                                    dX[k] = xt[q][k] - xs[k];
                                }
                                direct_eval_accumulate(evaluator, vt[q], dX, vs);
                            }
                        }
                    }
                }

                __syncthreads();
            }
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

extern "C" __global__ void DMK_DIRECT_KERNEL_NAME(DirectArgs a) {
    DirectByBoxBody<Evaluator, SRC_TILE, TARGETS_PER_THREAD>(a);
}
