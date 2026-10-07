// Free-space brute-force direct summation device kernel. The launcher prepends a prelude
// defining Real, DMK_DIRECT_KERNEL_NAME, the DMK_DIRECT_EVALUATOR typedef, and the
// BLOCK_SIZE / SRC_TILE constants. Mirrors the free-space evaluators in
// include/dmk/vector_kernels.hpp, contracted with the source strength and accumulated into vt.

#include <dmk/cuda/bessel_device.hpp>
#include <dmk/cuda/direct_freespace_kernelargs.hpp>
#include <dmk/cuda/poly_evaluators_device.hpp>

__device__ __forceinline__ Real dmk_rinv(Real R2) { return R2 > Real{0} ? dmk_rsqrt(R2) : Real{0}; }

template <int EVAL_LEVEL>
struct LaplaceFreeEvaluator2D {
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real lambda;

    __device__ inline void operator()(Real (&vt)[KERNEL_OUTPUT_DIM], const Real (&dX)[2], const Real (&vs)[1]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        vt[0] = fma(R2 > Real{0} ? Real{0.5} * log(R2) : Real{0}, vs[0], vt[0]);
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real Rinv = dmk_rinv(R2);
            const Real g = vs[0] * Rinv * Rinv;
#pragma unroll
            for (int i = 0; i < 2; ++i)
                vt[1 + i] = fma(dX[i], g, vt[1 + i]);
        }
    }
};

template <int EVAL_LEVEL>
struct LaplaceFreeEvaluator3D {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real lambda;

    __device__ inline void operator()(Real (&vt)[KERNEL_OUTPUT_DIM], const Real (&dX)[3], const Real (&vs)[1]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const Real Rinv = dmk_rinv(R2);
        vt[0] = fma(Rinv, vs[0], vt[0]);
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real g = -vs[0] * Rinv * Rinv * Rinv;
#pragma unroll
            for (int i = 0; i < 3; ++i)
                vt[1 + i] = fma(dX[i], g, vt[1 + i]);
        }
    }
};

template <int EVAL_LEVEL>
struct SqrtLaplaceFreeEvaluator2D {
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real lambda;

    __device__ inline void operator()(Real (&vt)[KERNEL_OUTPUT_DIM], const Real (&dX)[2], const Real (&vs)[1]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        const Real Rinv = dmk_rinv(R2);
        vt[0] = fma(Rinv, vs[0], vt[0]);
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real g = -vs[0] * Rinv * Rinv * Rinv;
#pragma unroll
            for (int i = 0; i < 2; ++i)
                vt[1 + i] = fma(dX[i], g, vt[1 + i]);
        }
    }
};

template <int EVAL_LEVEL>
struct SqrtLaplaceFreeEvaluator3D {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real lambda;

    __device__ inline void operator()(Real (&vt)[KERNEL_OUTPUT_DIM], const Real (&dX)[3], const Real (&vs)[1]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const Real Rinv = dmk_rinv(R2);
        const Real Rinv2 = Rinv * Rinv;
        vt[0] = fma(Rinv2, vs[0], vt[0]);
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real g = Real{-2} * vs[0] * Rinv2 * Rinv2;
#pragma unroll
            for (int i = 0; i < 3; ++i)
                vt[1 + i] = fma(dX[i], g, vt[1 + i]);
        }
    }
};

template <int EVAL_LEVEL>
struct YukawaFreeEvaluator3D {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real lambda;

    __device__ inline void operator()(Real (&vt)[KERNEL_OUTPUT_DIM], const Real (&dX)[3], const Real (&vs)[1]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const Real Rinv = dmk_rinv(R2);
        const Real R = Rinv * R2;
        const Real qE = vs[0] * exp(-lambda * R);
        vt[0] = fma(Rinv, qE, vt[0]);
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real g = -qE * Rinv * Rinv * (lambda + Rinv);
#pragma unroll
            for (int i = 0; i < 3; ++i)
                vt[1 + i] = fma(dX[i], g, vt[1 + i]);
        }
    }
};

// phi = K0(lambda*r), the 2D analog of exp(-lambda*r)/r.
template <int EVAL_LEVEL>
struct YukawaFreeEvaluator2D {
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real lambda;

    __device__ inline void operator()(Real (&vt)[KERNEL_OUTPUT_DIM], const Real (&dX)[2], const Real (&vs)[1]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        // K0/K1 at zero are infinite, and the gradient would form inf * 0 before a select
        // could discard it.
        if (!(R2 > Real{0}))
            return;
        const Real Rinv = Real{1} / sqrt(R2);
        const Real arg = lambda * (R2 * Rinv);
        vt[0] = fma(dmk::cuda::bessel::k0(arg), vs[0], vt[0]);
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real g = -lambda * vs[0] * dmk::cuda::bessel::k1(arg) * Rinv;
#pragma unroll
            for (int i = 0; i < 2; ++i)
                vt[1 + i] = fma(dX[i], g, vt[1 + i]);
        }
    }
};

// The sign differs from 3D.
template <int EVAL_LEVEL>
struct LaplaceDipoleFreeEvaluator2D {
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 2;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real lambda;

    __device__ inline void operator()(Real (&vt)[KERNEL_OUTPUT_DIM], const Real (&dX)[2], const Real (&vs)[2]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        const Real Rinv = dmk_rinv(R2);
        const Real Rinv2 = Rinv * Rinv;
        const Real rd = dX[0] * vs[0] + dX[1] * vs[1];
        vt[0] = fma(-Rinv2, rd, vt[0]);
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real g = Real{2} * rd * Rinv2 * Rinv2;
#pragma unroll
            for (int i = 0; i < 2; ++i)
                vt[1 + i] = fma(dX[i], g, fma(-Rinv2, vs[i], vt[1 + i]));
        }
    }
};

template <int EVAL_LEVEL>
struct LaplaceDipoleFreeEvaluator3D {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real lambda;

    __device__ inline void operator()(Real (&vt)[KERNEL_OUTPUT_DIM], const Real (&dX)[3], const Real (&vs)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const Real Rinv = dmk_rinv(R2);
        const Real Rinv2 = Rinv * Rinv;
        const Real Rinv3 = Rinv2 * Rinv;
        const Real rd = dX[0] * vs[0] + dX[1] * vs[1] + dX[2] * vs[2];
        vt[0] = fma(Rinv3, rd, vt[0]);
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real g = Real{-3} * rd * Rinv3 * Rinv2;
#pragma unroll
            for (int i = 0; i < 3; ++i)
                vt[1 + i] = fma(dX[i], g, fma(Rinv3, vs[i], vt[1 + i]));
        }
    }
};

// The 0.5 is the CPU convention: EvalPairs applies it at write-out, not inside the evaluator.
struct StokesletFreeEvaluator3D {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = 3;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{0.5};

    Real lambda;

    __device__ inline void operator()(Real (&vt)[3], const Real (&dX)[3], const Real (&vs)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const Real Rinv = dmk_rinv(R2);
        const Real rf = dX[0] * vs[0] + dX[1] * vs[1] + dX[2] * vs[2];
        const Real t = rf * Rinv * Rinv;
#pragma unroll
        for (int i = 0; i < 3; ++i)
            vt[i] = fma(Rinv, fma(dX[i], t, vs[i]), vt[i]);
    }
};

struct StressletFreeEvaluator3D {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = 3;
    static constexpr int NORMAL_DIM = 3;
    static constexpr Real scale_factor = Real{1};

    Real lambda;

    __device__ inline void operator()(Real (&vt)[3], const Real (&dX)[3], const Real (&vs)[3],
                                      const Real (&ns)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const Real Rinv = dmk_rinv(R2);
        const Real Rinv2 = Rinv * Rinv;
        const Real rdotn = dX[0] * ns[0] + dX[1] * ns[1] + dX[2] * ns[2];
        const Real rf = dX[0] * vs[0] + dX[1] * vs[1] + dX[2] * vs[2];
        const Real g = Real{-3} * rdotn * rf * Rinv2 * Rinv2 * Rinv;
#pragma unroll
        for (int i = 0; i < 3; ++i)
            vt[i] = fma(dX[i], g, vt[i]);
    }
};

// Each block owns BLOCK_SIZE / SPLIT targets, grid-strided; sources stream through shared memory
// in TILE-sized chunks of 16-byte-padded records, and each of SPLIT thread groups takes a TILE /
// SPLIT slice of every chunk. The target loop bound is uniform across the block, so a thread with
// no target still reaches every __syncthreads.
template <typename Eval, int TILE, int SPLIT>
__device__ __forceinline__ void DirectFreespaceBody(dmk::cuda::DirectFreespaceArgs<Real> a) {
    constexpr int DIM = Eval::SPATIAL_DIM;
    constexpr int KID = Eval::KERNEL_INPUT_DIM;
    constexpr int KOD = Eval::KERNEL_OUTPUT_DIM;
    constexpr int ND = Eval::NORMAL_DIM;
    constexpr int VEC = 16 / sizeof(Real);
    constexpr int REC = (DIM + KID + ND + VEC - 1) / VEC * VEC;
    constexpr int TPB = BLOCK_SIZE / SPLIT;
    constexpr int SUB = TILE / SPLIT;
    constexpr int STAGED = TILE / BLOCK_SIZE;
    static_assert(TPB * SPLIT == BLOCK_SIZE && SUB * SPLIT == TILE && STAGED * BLOCK_SIZE == TILE);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    Real *const s_src = reinterpret_cast<Real *>(smem_raw);
    Real *const s_part = s_src + TILE * REC;

    const Eval evaluator{a.lambda};
    const int group = threadIdx.x / TPB;
    const int lt = threadIdx.x % TPB;
    const int stride = TPB * gridDim.x;

    // Next tile, staged in registers while the current one is evaluated.
    Real staged[STAGED][REC];
    auto load_tile = [&](int s0) {
#pragma unroll
        for (int j = 0; j < STAGED; ++j) {
            const int s = s0 + threadIdx.x + j * BLOCK_SIZE;
            if (s < a.n_src) {
#pragma unroll
                for (int k = 0; k < DIM; ++k)
                    staged[j][k] = a.r_src[s * DIM + k];
#pragma unroll
                for (int k = 0; k < KID; ++k)
                    staged[j][DIM + k] = a.charge[s * KID + k];
#pragma unroll
                for (int k = 0; k < ND; ++k)
                    staged[j][DIM + KID + k] = a.normal[s * ND + k];
            }
        }
    };

    for (int t_base = blockIdx.x * TPB; t_base < a.n_trg; t_base += stride) {
        const int t = t_base + lt;
        const bool active = t < a.n_trg;

        Real xt[DIM];
        if (active) {
#pragma unroll
            for (int k = 0; k < DIM; ++k)
                xt[k] = a.r_trg[t * DIM + k];
        }

        Real vt[KOD];
#pragma unroll
        for (int k = 0; k < KOD; ++k)
            vt[k] = Real{0};

        load_tile(0);
        for (int s0 = 0; s0 < a.n_src; s0 += TILE) {
            const int rem = a.n_src - s0;
            const int count = rem < TILE ? rem : TILE;

            __syncthreads();
#pragma unroll
            for (int j = 0; j < STAGED; ++j) {
                const int s = threadIdx.x + j * BLOCK_SIZE;
                if (s < count) {
#pragma unroll
                    for (int k = 0; k < DIM + KID + ND; ++k)
                        s_src[s * REC + k] = staged[j][k];
                }
            }
            __syncthreads();

            if (s0 + TILE < a.n_src)
                load_tile(s0 + TILE);

            if (!active)
                continue;

            const int lo = group * SUB;
            const int hi = lo + SUB < count ? lo + SUB : count;
            for (int s = lo; s < hi; ++s) {
                const Real *rec = s_src + s * REC;
                Real dX[DIM];
#pragma unroll
                for (int k = 0; k < DIM; ++k)
                    dX[k] = xt[k] - rec[k];

                Real vs[KID];
#pragma unroll
                for (int k = 0; k < KID; ++k)
                    vs[k] = rec[DIM + k];

                if constexpr (ND > 0) {
                    Real ns[ND > 0 ? ND : 1];
#pragma unroll
                    for (int k = 0; k < ND; ++k)
                        ns[k] = rec[DIM + KID + k];
                    evaluator(vt, dX, vs, ns);
                } else {
                    evaluator(vt, dX, vs);
                }
            }
        }

        if constexpr (SPLIT > 1) {
#pragma unroll
            for (int k = 0; k < KOD; ++k)
                s_part[(group * KOD + k) * TPB + lt] = vt[k];
            __syncthreads();
            if (group == 0) {
#pragma unroll
                for (int k = 0; k < KOD; ++k) {
#pragma unroll
                    for (int g = 1; g < SPLIT; ++g)
                        vt[k] += s_part[(g * KOD + k) * TPB + lt];
                }
            }
        }

        if (active && group == 0) {
#pragma unroll
            for (int k = 0; k < KOD; ++k)
                a.pot[t * KOD + k] = vt[k] * Eval::scale_factor;
        }
    }
}

using Evaluator = DMK_DIRECT_EVALUATOR;
using DirectArgs = dmk::cuda::DirectFreespaceArgs<Real>;

// KERNEL_START

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE) DMK_DIRECT_KERNEL_NAME(DirectArgs a) {
    DirectFreespaceBody<Evaluator, SRC_TILE, SRC_SPLIT>(a);
}
