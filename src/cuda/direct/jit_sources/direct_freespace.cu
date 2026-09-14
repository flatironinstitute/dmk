// Free-space brute-force direct summation device kernel. The launcher prepends a prelude
// defining Real, DMK_DIRECT_KERNEL_NAME, the DMK_DIRECT_EVALUATOR typedef, and the
// BLOCK_SIZE / SRC_TILE constants. Mirrors the free-space evaluators in
// include/dmk/vector_kernels.hpp.
//
// 1/sqrt(R2) rather than the residual kernel's rsqrt.approx.ftz.f32: this is the reference
// path, and the CPU side refines its rsqrt to full precision.

#include <dmk/cuda/bessel_device.hpp>
#include <dmk/cuda/direct_freespace_kernelargs.hpp>

// Rinv == 0 at a coincident pair, which zeros every product. The 2D log and 2D Yukawa kernels
// are singular at the origin in a way a zero Rinv cannot absorb, so they branch instead.
__device__ __forceinline__ Real dmk_rinv(Real R2) { return R2 > Real{0} ? Real{1} / sqrt(R2) : Real{0}; }

template <int EVAL_LEVEL>
struct LaplaceFreeEvaluator2D {
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real lambda;

    __device__ inline void operator()(Real (&u)[1][KERNEL_OUTPUT_DIM], const Real (&dX)[2]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        u[0][0] = R2 > Real{0} ? Real{0.5} * log(R2) : Real{0};
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real Rinv = dmk_rinv(R2);
            const Real Rinv2 = Rinv * Rinv;
#pragma unroll
            for (int i = 0; i < 2; ++i)
                u[0][1 + i] = dX[i] * Rinv2;
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

    __device__ inline void operator()(Real (&u)[1][KERNEL_OUTPUT_DIM], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const Real Rinv = dmk_rinv(R2);
        u[0][0] = Rinv;
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real Rinv3 = Rinv * Rinv * Rinv;
#pragma unroll
            for (int i = 0; i < 3; ++i)
                u[0][1 + i] = -dX[i] * Rinv3;
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

    __device__ inline void operator()(Real (&u)[1][KERNEL_OUTPUT_DIM], const Real (&dX)[2]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        const Real Rinv = dmk_rinv(R2);
        u[0][0] = Rinv;
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real Rinv3 = Rinv * Rinv * Rinv;
#pragma unroll
            for (int i = 0; i < 2; ++i)
                u[0][1 + i] = -dX[i] * Rinv3;
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

    __device__ inline void operator()(Real (&u)[1][KERNEL_OUTPUT_DIM], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const Real Rinv = dmk_rinv(R2);
        const Real Rinv2 = Rinv * Rinv;
        u[0][0] = Rinv2;
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real g = Real{-2} * Rinv2 * Rinv2;
#pragma unroll
            for (int i = 0; i < 3; ++i)
                u[0][1 + i] = dX[i] * g;
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

    __device__ inline void operator()(Real (&u)[1][KERNEL_OUTPUT_DIM], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const Real Rinv = dmk_rinv(R2);
        const Real R = Rinv * R2;
        const Real E = exp(-lambda * R);
        u[0][0] = Rinv * E;
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real g = -E * Rinv * Rinv * (lambda + Rinv);
#pragma unroll
            for (int i = 0; i < 3; ++i)
                u[0][1 + i] = dX[i] * g;
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

    __device__ inline void operator()(Real (&u)[1][KERNEL_OUTPUT_DIM], const Real (&dX)[2]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        // K0/K1 at zero are infinite, and the gradient would form inf * 0 before a select
        // could discard it.
        if (!(R2 > Real{0})) {
#pragma unroll
            for (int k = 0; k < KERNEL_OUTPUT_DIM; ++k)
                u[0][k] = Real{0};
            return;
        }
        const Real Rinv = Real{1} / sqrt(R2);
        const Real arg = lambda * (R2 * Rinv);
        u[0][0] = dmk::cuda::bessel::k0(arg);
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real g = -lambda * dmk::cuda::bessel::k1(arg) * Rinv;
#pragma unroll
            for (int i = 0; i < 2; ++i)
                u[0][1 + i] = dX[i] * g;
        }
    }
};

// u[k][j] is the response of output j to dipole strength component k. The sign differs from 3D.
template <int EVAL_LEVEL>
struct LaplaceDipoleFreeEvaluator2D {
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 2;
    static constexpr int KERNEL_OUTPUT_DIM = EVAL_LEVEL == 1 ? 1 : SPATIAL_DIM + 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1};

    Real lambda;

    __device__ inline void operator()(Real (&u)[2][KERNEL_OUTPUT_DIM], const Real (&dX)[2]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        const Real Rinv = dmk_rinv(R2);
        const Real Rinv2 = Rinv * Rinv;
#pragma unroll
        for (int k = 0; k < 2; ++k)
            u[k][0] = -dX[k] * Rinv2;
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real Rinv4 = Rinv2 * Rinv2;
#pragma unroll
            for (int k = 0; k < 2; ++k) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    Real val = Real{2} * dX[k] * dX[i] * Rinv4;
                    if (i == k)
                        val = val - Rinv2;
                    u[k][1 + i] = val;
                }
            }
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

    __device__ inline void operator()(Real (&u)[3][KERNEL_OUTPUT_DIM], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const Real Rinv = dmk_rinv(R2);
        const Real Rinv3 = Rinv * Rinv * Rinv;
#pragma unroll
        for (int k = 0; k < 3; ++k)
            u[k][0] = dX[k] * Rinv3;
        if constexpr (KERNEL_OUTPUT_DIM > 1) {
            const Real Rinv5 = Rinv3 * Rinv * Rinv;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    Real val = Real{-3} * dX[k] * dX[i] * Rinv5;
                    if (i == k)
                        val = val + Rinv3;
                    u[k][1 + i] = val;
                }
            }
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

    __device__ inline void operator()(Real (&u)[3][3], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const Real Rinv = dmk_rinv(R2);
        const Real Rinv3 = Rinv * Rinv * Rinv;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                Real val = dX[j] * dX[i] * Rinv3;
                if (i == j)
                    val += Rinv;
                u[i][j] = val;
            }
        }
    }
};

struct StressletFreeEvaluator3D {
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = 3;
    static constexpr int NORMAL_DIM = 3;
    static constexpr Real scale_factor = Real{1};

    Real lambda;

    __device__ inline void operator()(Real (&u)[3][3], const Real (&dX)[3], const Real (&ns)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const Real Rinv = dmk_rinv(R2);
        const Real Rinv5 = Rinv * Rinv * Rinv * Rinv * Rinv;
        const Real rdotn = dX[0] * ns[0] + dX[1] * ns[1] + dX[2] * ns[2];
        const Real factor = Real{-3} * rdotn * Rinv5;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const Real fj = dX[j] * factor;
#pragma unroll
            for (int i = 0; i < 3; ++i)
                u[j][i] = fj * dX[i];
        }
    }
};

template <typename Eval>
__device__ __forceinline__ void free_eval_accumulate(const Eval &evaluator, Real (&vt)[Eval::KERNEL_OUTPUT_DIM],
                                                     const Real (&dX)[Eval::SPATIAL_DIM],
                                                     const Real (&vs)[Eval::KERNEL_INPUT_DIM]) {
    Real U[Eval::KERNEL_INPUT_DIM][Eval::KERNEL_OUTPUT_DIM];
    evaluator(U, dX);
#pragma unroll
    for (int k0 = 0; k0 < Eval::KERNEL_INPUT_DIM; ++k0) {
#pragma unroll
        for (int k1 = 0; k1 < Eval::KERNEL_OUTPUT_DIM; ++k1)
            vt[k1] = fma(U[k0][k1], vs[k0], vt[k1]);
    }
}

template <typename Eval>
__device__ __forceinline__ void
free_eval_accumulate(const Eval &evaluator, Real (&vt)[Eval::KERNEL_OUTPUT_DIM], const Real (&dX)[Eval::SPATIAL_DIM],
                     const Real (&vs)[Eval::KERNEL_INPUT_DIM], const Real (&ns)[Eval::NORMAL_DIM]) {
    Real U[Eval::KERNEL_INPUT_DIM][Eval::KERNEL_OUTPUT_DIM];
    evaluator(U, dX, ns);
#pragma unroll
    for (int k0 = 0; k0 < Eval::KERNEL_INPUT_DIM; ++k0) {
#pragma unroll
        for (int k1 = 0; k1 < Eval::KERNEL_OUTPUT_DIM; ++k1)
            vt[k1] = fma(U[k0][k1], vs[k0], vt[k1]);
    }
}

// One thread per target, grid-strided; sources stream through shared memory in TILE-sized
// chunks. The target loop bound is uniform across the block, so a thread with no target still
// reaches every __syncthreads.
template <typename Eval, int TILE>
__device__ __forceinline__ void DirectFreespaceBody(dmk::cuda::DirectFreespaceArgs<Real> a) {
    constexpr int DIM = Eval::SPATIAL_DIM;
    constexpr int KID = Eval::KERNEL_INPUT_DIM;
    constexpr int KOD = Eval::KERNEL_OUTPUT_DIM;
    constexpr int ND = Eval::NORMAL_DIM;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    Real *smem = reinterpret_cast<Real *>(smem_raw);
    Real *const s_r_src = smem;
    smem += TILE * DIM;
    Real *const s_charge = smem;
    smem += TILE * KID;
    Real *const s_normal = smem;

    const Eval evaluator{a.lambda};
    const int stride = blockDim.x * gridDim.x;

    for (int t_base = blockIdx.x * blockDim.x; t_base < a.n_trg; t_base += stride) {
        const int t = t_base + threadIdx.x;
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

        for (int s0 = 0; s0 < a.n_src; s0 += TILE) {
            const int rem = a.n_src - s0;
            const int count = rem < TILE ? rem : TILE;

            __syncthreads();
            for (int i = threadIdx.x; i < count * DIM; i += blockDim.x)
                s_r_src[i] = a.r_src[s0 * DIM + i];
            for (int i = threadIdx.x; i < count * KID; i += blockDim.x)
                s_charge[i] = a.charge[s0 * KID + i];
            if constexpr (ND > 0) {
                for (int i = threadIdx.x; i < count * ND; i += blockDim.x)
                    s_normal[i] = a.normal[s0 * ND + i];
            }
            __syncthreads();

            if (!active)
                continue;

            for (int s = 0; s < count; ++s) {
                Real dX[DIM];
#pragma unroll
                for (int k = 0; k < DIM; ++k)
                    dX[k] = xt[k] - s_r_src[s * DIM + k];

                Real vs[KID];
#pragma unroll
                for (int k = 0; k < KID; ++k)
                    vs[k] = s_charge[s * KID + k];

                if constexpr (ND > 0) {
                    Real ns[ND > 0 ? ND : 1];
#pragma unroll
                    for (int k = 0; k < ND; ++k)
                        ns[k] = s_normal[s * ND + k];
                    free_eval_accumulate(evaluator, vt, dX, vs, ns);
                } else {
                    free_eval_accumulate(evaluator, vt, dX, vs);
                }
            }
        }

        if (active) {
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
    DirectFreespaceBody<Evaluator, SRC_TILE>(a);
}
