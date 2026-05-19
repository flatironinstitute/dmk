#ifndef DMK_CUDA_DIRECT_KERNELS_CUH
#define DMK_CUDA_DIRECT_KERNELS_CUH

// CUDA direct-evaluation kernels. All pointers passed to the launchers must
// be device-resident.
//
// Coefficient tables ride into the kernel as compile-time *types* (CoeffTag),
// not runtime pointers and not structural-class NTTP values — the latter trip
// a cudafe stub-generation bug in nvcc when used as template args to __global__
// templates. The tag exposes the data via a `constexpr static T at(size_t)`
// function rather than a `constexpr static T data[N]` member because nvcc
// treats class-scope constexpr arrays as host-only; the function form
// constant-folds to immediates on both sides.

#include <dmk/cuda/direct_kernels.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef DMK_CUDA_USE_NVRTC_JIT
#include "cuda/jit/direct_launcher.hpp"
#include <cstdlib>
#endif

namespace dmk::cuda {

template <typename C>
concept CoeffTag = requires {
    typename C::value_type;
    { C::size } -> std::convertible_to<std::size_t>;
    { C::at(0) } -> std::convertible_to<typename C::value_type>;
};

// Template-recursive Horner. With Coeffs as a *type*, each Coeffs::at(I) for
// constexpr I is a constant expression — the unrolled chain becomes a
// sequence of FMAs against literal immediates.
template <CoeffTag Coeffs, std::size_t I, typename Real>
__device__ constexpr Real horner_recurse(Real x, Real acc) {
    if constexpr (I == 0)
        return acc;
    else
        return horner_recurse<Coeffs, I - 1>(x, acc * x + Real{Coeffs::at(I - 1)});
}

template <CoeffTag Coeffs, typename Real>
__device__ constexpr Real horner_const(Real x) {
    static_assert(Coeffs::size > 0, "empty coefficient pack");
    return horner_recurse<Coeffs, Coeffs::size - 1>(x, Real{Coeffs::at(Coeffs::size - 1)});
}

// clang-format off
template <typename E>
concept DeviceKernelEvaluator = requires {
    typename E::scalar_type;
    { int(E::SPATIAL_DIM) };
    { int(E::KERNEL_INPUT_DIM) };
    { int(E::KERNEL_OUTPUT_DIM) };
    { int(E::NORMAL_DIM) };
    { typename E::scalar_type(E::scale_factor) };
};
// clang-format on

template <CoeffTag Coeffs>
struct LaplacePolyEvaluator2DCuda {
    using scalar_type = typename Coeffs::value_type;
    using Real = scalar_type;
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1.0};

    Real thresh2, d2max, rsc, cen;

    __device__ inline void operator()(Real (&u)[1][1], const Real (&dX)[2]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        const bool in_range = (R2 > thresh2) && (R2 < d2max);
        const Real R2sc = R2 * (Real{0.5} * rsc);
        const Real arg = rsc * R2 + cen;
        const Real ptmp = horner_const<Coeffs, Real>(arg);
        u[0][0] = in_range ? Real{0.5} * log(R2sc) + ptmp : Real{0};
    }
};

template <CoeffTag Coeffs>
struct LaplacePolyEvaluator3DCuda {
    using scalar_type = typename Coeffs::value_type;
    using Real = scalar_type;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1.0};
    Real thresh2, d2max, rsc, cen;

    __device__ inline void operator()(Real (&u)[1][1], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const bool in_range = (R2 > thresh2) && (R2 < d2max);
        if (!in_range) {
            u[0][0] = 0.0;
            return;
        }
        const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
        const Real xmapped = (R2 * Rinv + cen) * rsc;
        const Real P = horner_const<Coeffs, Real>(xmapped);
        u[0][0] = P * Rinv;
    }
};

template <CoeffTag Coeffs>
struct SqrtLaplacePolyEvaluator2DCuda {
    using scalar_type = typename Coeffs::value_type;
    using Real = scalar_type;
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1.0};
    Real thresh2, d2max, rsc, cen;

    __device__ inline void operator()(Real (&u)[1][1], const Real (&dX)[2]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1];
        const bool in_range = (R2 > thresh2) && (R2 < d2max);
        const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
        const Real xmapped = (R2 * Rinv + cen) * rsc;
        u[0][0] = in_range ? horner_const<Coeffs, Real>(xmapped) * Rinv : Real{0};
    }
};

template <CoeffTag Coeffs>
struct SqrtLaplacePolyEvaluator3DCuda {
    using scalar_type = typename Coeffs::value_type;
    using Real = scalar_type;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 1;
    static constexpr int KERNEL_OUTPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1.0};
    Real thresh2, d2max, rsc, cen;

    __device__ inline void operator()(Real (&u)[1][1], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const bool in_range = (R2 > thresh2) && (R2 < d2max);
        if (!in_range) {
            u[0][0] = 0.0;
            return;
        }
        const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
        const Real R2inv = Rinv * Rinv;
        const Real arg = rsc * R2 + cen;
        u[0][0] = R2inv * horner_const<Coeffs, Real>(arg);
    }
};

template <CoeffTag CoeffsDiag, CoeffTag CoeffsOffdiag>
struct StokesletPolyEvaluator3DCuda {
    using scalar_type = typename CoeffsDiag::value_type;
    using Real = scalar_type;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = 3;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1.0};
    Real thresh2, d2max, rsc, cen;

    __device__ inline void operator()(Real (&u)[3][3], const Real (&dX)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const bool in_range = (R2 > thresh2) && (R2 < d2max);
        if (!in_range) {
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                    u[j][i] = 0.0;
            return;
        }
        const Real half = Real{0.5};
        const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
        const Real Rinv3 = Rinv * Rinv * Rinv;
        const Real xtmp = (R2 * Rinv + cen) * rsc;
        const Real fdiag = (half - horner_const<CoeffsDiag, Real>(xtmp)) * Rinv;
        const Real foffd = (half - horner_const<CoeffsOffdiag, Real>(xtmp)) * Rinv3;

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                Real val = foffd * dX[j] * dX[i];
                if (i == j)
                    val += fdiag;
                u[i][j] = val;
            }
    }
};

template <CoeffTag CoeffsDiag, CoeffTag CoeffsOffdiag>
struct StressletPolyEvaluator3DCuda {
    using scalar_type = typename CoeffsDiag::value_type;
    using Real = scalar_type;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = 3;
    static constexpr int NORMAL_DIM = 3;
    static constexpr Real scale_factor = Real{1.0};
    Real thresh2, d2max, rsc, cen;

    __device__ inline void operator()(Real (&u)[3][3], const Real (&dX)[3], const Real (&ns)[3]) const {
        const Real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const bool in_range = (R2 > thresh2) && (R2 < d2max);
        const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
        const Real Rinv3 = Rinv * Rinv * Rinv;
        const Real Rinv5 = Rinv3 * Rinv * Rinv;
        const Real xtmp = (R2 * Rinv + cen) * rsc;

        const Real Fdiag = -horner_const<CoeffsDiag, Real>(xtmp) * Rinv3;
        const Real Foffd = Real{6.0} * horner_const<CoeffsOffdiag, Real>(xtmp) * Rinv5;
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
                u[j][i] = in_range ? val : Real{0};
            }
        }
    }
};

// Flat all-pairs driver — one thread per target, source tiles staged into
// shared memory and reused by every thread in the block. Used by the AOT
// residual_evaluator_func getters that test_cuda.cpp uses for raw CPU-vs-GPU
// accuracy comparison. Mirrors host EvalPairs in vector_kernels.hpp.
template <int KERNEL_OUTPUT_DIM, DeviceKernelEvaluator Evaluator>
__global__ void EvalPairsCuda(int n_src, const typename Evaluator::scalar_type *__restrict__ r_src,
                              const typename Evaluator::scalar_type *__restrict__ v_src,
                              const typename Evaluator::scalar_type *__restrict__ src_normals, int n_trg,
                              const typename Evaluator::scalar_type *__restrict__ r_trg,
                              typename Evaluator::scalar_type *__restrict__ v_trg, Evaluator evaluator) {
    using Real = typename Evaluator::scalar_type;
    constexpr int SPATIAL_DIM = Evaluator::SPATIAL_DIM;
    constexpr int KERNEL_INPUT_DIM = Evaluator::KERNEL_INPUT_DIM;
    constexpr int NORMAL_DIM = Evaluator::NORMAL_DIM;
    constexpr int row_size = SPATIAL_DIM + KERNEL_INPUT_DIM + NORMAL_DIM;
    constexpr Real scale_factor = Evaluator::scale_factor;

    extern __shared__ unsigned char shared_raw[];
    Real *shared = reinterpret_cast<Real *>(shared_raw);

    const int i_trg = blockIdx.x * blockDim.x + threadIdx.x;

    Real xt[SPATIAL_DIM];
    Real vt[KERNEL_OUTPUT_DIM];
    if (i_trg < n_trg) {
        for (int k = 0; k < SPATIAL_DIM; ++k)
            xt[k] = r_trg[i_trg * SPATIAL_DIM + k];
        for (int k = 0; k < KERNEL_OUTPUT_DIM; ++k)
            vt[k] = Real{0};
    }

    const int n_tiles = (n_src + blockDim.x - 1) / blockDim.x;
    for (int tile = 0; tile < n_tiles; ++tile) {
        const int i_src = tile * blockDim.x + threadIdx.x;
        const int row = threadIdx.x * row_size;
        if (i_src < n_src) {
            for (int k = 0; k < SPATIAL_DIM; ++k)
                shared[row + k] = r_src[i_src * SPATIAL_DIM + k];
            for (int k = 0; k < KERNEL_INPUT_DIM; ++k)
                shared[row + SPATIAL_DIM + k] = v_src[i_src * KERNEL_INPUT_DIM + k];
            if constexpr (NORMAL_DIM > 0) {
                for (int k = 0; k < NORMAL_DIM; ++k)
                    shared[row + SPATIAL_DIM + KERNEL_INPUT_DIM + k] = src_normals[i_src * NORMAL_DIM + k];
            }
        }
        __syncthreads();

        const int n_local = min((int)blockDim.x, n_src - tile * (int)blockDim.x);
        if (i_trg < n_trg) {
            for (int s = 0; s < n_local; ++s) {
                const Real *r_s = &shared[s * row_size];
                const Real *v_s = r_s + SPATIAL_DIM;
                Real dX[SPATIAL_DIM];
                for (int k = 0; k < SPATIAL_DIM; ++k)
                    dX[k] = xt[k] - r_s[k];

                Real U[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM];
                if constexpr (NORMAL_DIM > 0) {
                    const Real *n_s = v_s + KERNEL_INPUT_DIM;
                    Real ns[NORMAL_DIM];
                    for (int k = 0; k < NORMAL_DIM; ++k)
                        ns[k] = n_s[k];
                    evaluator(U, dX, ns);
                } else {
                    evaluator(U, dX);
                }
                for (int k0 = 0; k0 < KERNEL_INPUT_DIM; ++k0)
                    for (int k1 = 0; k1 < KERNEL_OUTPUT_DIM; ++k1)
                        vt[k1] += U[k0][k1] * v_s[k0];
            }
        }
        __syncthreads();
    }

    if (i_trg < n_trg) {
        for (int k = 0; k < KERNEL_OUTPUT_DIM; ++k)
            v_trg[i_trg * KERNEL_OUTPUT_DIM + k] += vt[k] * scale_factor;
    }
}

namespace detail {
constexpr int default_block_size = 128;

template <typename Evaluator, int KERNEL_OUTPUT_DIM, typename Real>
inline void launch_eval_pairs(const Evaluator &evaluator, int n_src, const Real *r_src, const Real *charge,
                              const Real *normals, int n_trg, const Real *r_trg, Real *pot, cudaStream_t stream) {
    constexpr int block_size = default_block_size;
    const int grid = (n_trg + block_size - 1) / block_size;
    constexpr int row = Evaluator::SPATIAL_DIM + Evaluator::KERNEL_INPUT_DIM + Evaluator::NORMAL_DIM;
    const int smem = block_size * row * sizeof(Real);
    EvalPairsCuda<KERNEL_OUTPUT_DIM, Evaluator>
        <<<grid, block_size, smem, stream>>>(n_src, r_src, charge, normals, n_trg, r_trg, pot, evaluator);
}
} // namespace detail

template <CoeffTag Coeffs, typename Real = typename Coeffs::value_type>
inline void laplace_2d_poly_all_pairs(Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,
                                      const Real *charge, const Real *normals, int n_trg, const Real *r_trg, Real *pot,
                                      cudaStream_t stream = 0) {
    using Evaluator = LaplacePolyEvaluator2DCuda<Coeffs>;
    Evaluator evaluator{thresh2, d2max, rsc, cen};
    detail::launch_eval_pairs<Evaluator, 1>(evaluator, n_src, r_src, charge, normals, n_trg, r_trg, pot, stream);
}

template <CoeffTag Coeffs, typename Real = typename Coeffs::value_type>
inline void laplace_3d_poly_all_pairs(Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,
                                      const Real *charge, const Real *normals, int n_trg, const Real *r_trg, Real *pot,
                                      cudaStream_t stream = 0) {
    using Evaluator = LaplacePolyEvaluator3DCuda<Coeffs>;
    Evaluator evaluator{thresh2, d2max, rsc, cen};
    detail::launch_eval_pairs<Evaluator, 1>(evaluator, n_src, r_src, charge, normals, n_trg, r_trg, pot, stream);
}

template <CoeffTag Coeffs, typename Real = typename Coeffs::value_type>
inline void sqrt_laplace_2d_poly_all_pairs(Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,
                                           const Real *charge, const Real *normals, int n_trg, const Real *r_trg,
                                           Real *pot, cudaStream_t stream = 0) {
    using Evaluator = SqrtLaplacePolyEvaluator2DCuda<Coeffs>;
    Evaluator evaluator{thresh2, d2max, rsc, cen};
    detail::launch_eval_pairs<Evaluator, 1>(evaluator, n_src, r_src, charge, normals, n_trg, r_trg, pot, stream);
}

template <CoeffTag Coeffs, typename Real = typename Coeffs::value_type>
inline void sqrt_laplace_3d_poly_all_pairs(Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,
                                           const Real *charge, const Real *normals, int n_trg, const Real *r_trg,
                                           Real *pot, cudaStream_t stream = 0) {
    using Evaluator = SqrtLaplacePolyEvaluator3DCuda<Coeffs>;
    Evaluator evaluator{thresh2, d2max, rsc, cen};
    detail::launch_eval_pairs<Evaluator, 1>(evaluator, n_src, r_src, charge, normals, n_trg, r_trg, pot, stream);
}

template <CoeffTag CoeffsDiag, CoeffTag CoeffsOffdiag, typename Real = typename CoeffsDiag::value_type>
inline void stokeslet_3d_poly_all_pairs(Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,
                                        const Real *charge, const Real *normals, int n_trg, const Real *r_trg,
                                        Real *pot, cudaStream_t stream = 0) {
    using Evaluator = StokesletPolyEvaluator3DCuda<CoeffsDiag, CoeffsOffdiag>;
    Evaluator evaluator{thresh2, d2max, rsc, cen};
    detail::launch_eval_pairs<Evaluator, Evaluator::KERNEL_OUTPUT_DIM>(evaluator, n_src, r_src, charge, normals, n_trg,
                                                                       r_trg, pot, stream);
}

template <CoeffTag CoeffsDiag, CoeffTag CoeffsOffdiag, typename Real = typename CoeffsDiag::value_type>
inline void stresslet_3d_poly_all_pairs(Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,
                                        const Real *charge, const Real *normals, int n_trg, const Real *r_trg,
                                        Real *pot, cudaStream_t stream = 0) {
    using Evaluator = StressletPolyEvaluator3DCuda<CoeffsDiag, CoeffsOffdiag>;
    Evaluator evaluator{thresh2, d2max, rsc, cen};
    detail::launch_eval_pairs<Evaluator, Evaluator::KERNEL_OUTPUT_DIM>(evaluator, n_src, r_src, charge, normals, n_trg,
                                                                       r_trg, pot, stream);
}

// Per-box driver: one CUDA block per target box, source-box loop inside the
// kernel so each block owns its slice of pot output (no cross-block race).
// Used in production by CudaDirectContext. One launch handles either pot_src
// or pot_trg; cuda/direct.cpp issues two launches per direct evaluation.
//
// No ContactGeometry filtering, no PBC, no Yukawa — caller must reject those.
// Block-load is uneven (big trg boxes dominate), deliberate for now.

template <typename Evaluator, int SRC_TILE>
__global__ void DirectResidualByBoxKernelTiled(DirectByBoxArgs<typename Evaluator::scalar_type> a) {
    using Real = typename Evaluator::scalar_type;

    constexpr int SPATIAL_DIM = Evaluator::SPATIAL_DIM;
    constexpr int KERNEL_INPUT_DIM = Evaluator::KERNEL_INPUT_DIM;
    constexpr int KERNEL_OUTPUT_DIM = Evaluator::KERNEL_OUTPUT_DIM;
    constexpr int NORMAL_DIM = Evaluator::NORMAL_DIM;
    constexpr Real scale_factor = Evaluator::scale_factor;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    Real *smem = reinterpret_cast<Real *>(smem_raw);

    Real *s_r_src = smem;
    smem += SRC_TILE * SPATIAL_DIM;

    Real *s_charge = smem;
    smem += SRC_TILE * KERNEL_INPUT_DIM;

    Real *s_normal = nullptr;
    if constexpr (NORMAL_DIM > 0) {
        s_normal = smem;
        smem += SRC_TILE * NORMAL_DIM;
    }

    const int trg_box_idx = blockIdx.x;
    if (trg_box_idx >= a.n_work)
        return;

    const int trg_box = a.direct_work[trg_box_idx];
    const int n_targets = a.target_counts[trg_box];
    if (n_targets == 0)
        return;

    const int trg_level = a.box_levels[trg_box];
    const int n_list1 = a.list1_count[trg_box];

    const Real *__restrict__ r_targets = a.r_target_flat + a.r_target_offsets[trg_box];
    Real *__restrict__ pot_targets = a.pot_flat + a.pot_offsets[trg_box];

    const int n_target_rounds = (n_targets + blockDim.x - 1) / blockDim.x;

    for (int tr = 0; tr < n_target_rounds; ++tr) {
        const int t = tr * blockDim.x + threadIdx.x;
        const bool active_target = (t < n_targets);

        Real xt[SPATIAL_DIM];
        if (active_target) {
#pragma unroll
            for (int k = 0; k < SPATIAL_DIM; ++k)
                xt[k] = r_targets[t * SPATIAL_DIM + k];
        }

        Real vt[KERNEL_OUTPUT_DIM];
#pragma unroll
        for (int k = 0; k < KERNEL_OUTPUT_DIM; ++k)
            vt[k] = Real{0};

        for (int li = 0; li < n_list1; ++li) {
            const int src_box = a.list1_flat[trg_box * a.nlist1_stride + li];

            int src_level = a.box_levels[src_box];
            if (a.ifpwexp[src_box] && src_box == trg_box)
                src_level = src_level + 1;
            else if (src_level < trg_level)
                src_level = trg_level;
            if (src_level >= a.n_levels)
                src_level = a.n_levels - 1;

            const int n_src = a.src_counts_halo[src_box];

            const Real *__restrict__ r_src = a.r_src_halo_flat + a.r_src_halo_offsets[src_box];
            const Real *__restrict__ charge = a.charge_halo_flat + a.charge_halo_offsets[src_box];

            const Real *__restrict__ normals = nullptr;
            if constexpr (NORMAL_DIM > 0)
                normals = a.normal_halo_flat + a.normal_halo_offsets[src_box];

            const Real rsc = a.direct_rsc[src_level];
            const Real cen = a.direct_cen[src_level];
            const Real d2max = a.direct_d2max[src_level];

            Evaluator evaluator{a.thresh2, d2max, rsc, cen};

            for (int tile0 = 0; tile0 < n_src; tile0 += SRC_TILE) {
                const int rem = n_src - tile0;
                const int tile_count = (rem < SRC_TILE) ? rem : SRC_TILE;

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
                        for (int k = 0; k < SPATIAL_DIM; ++k)
                            xs[k] = s_r_src[ss * SPATIAL_DIM + k];

                        Real dX[SPATIAL_DIM];
#pragma unroll
                        for (int k = 0; k < SPATIAL_DIM; ++k)
                            dX[k] = xt[k] - xs[k];

                        Real vs[KERNEL_INPUT_DIM];
#pragma unroll
                        for (int k = 0; k < KERNEL_INPUT_DIM; ++k)
                            vs[k] = s_charge[ss * KERNEL_INPUT_DIM + k];

                        Real U[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM];

                        if constexpr (NORMAL_DIM > 0) {
                            Real ns[NORMAL_DIM];
#pragma unroll
                            for (int k = 0; k < NORMAL_DIM; ++k)
                                ns[k] = s_normal[ss * NORMAL_DIM + k];
                            evaluator(U, dX, ns);
                        } else {
                            evaluator(U, dX);
                        }

#pragma unroll
                        for (int k0 = 0; k0 < KERNEL_INPUT_DIM; ++k0) {
#pragma unroll
                            for (int k1 = 0; k1 < KERNEL_OUTPUT_DIM; ++k1)
                                vt[k1] += U[k0][k1] * vs[k0];
                        }
                    }
                }

                __syncthreads();
            }
        }

        if (active_target) {
#pragma unroll
            for (int k = 0; k < KERNEL_OUTPUT_DIM; ++k)
                pot_targets[t * KERNEL_OUTPUT_DIM + k] += vt[k] * scale_factor;
        }
    }
}

template <typename Evaluator, typename Real = typename Evaluator::scalar_type, int SRC_TILE = 32>
inline void launch_direct_by_box(const DirectByBoxArgs<Real> &args, cudaStream_t stream = 0) {
    if (args.n_work == 0)
        return;

    constexpr int block_size = 128;

#ifdef DMK_CUDA_USE_NVRTC_JIT
    {
        const char *disable = std::getenv("DMK_DISABLE_DIRECT_JIT");
        const bool use_jit = !(disable && std::string(disable) == "1");

        if (use_jit) {
            static dmk::cuda::jit::JitCache jit_cache;

            dmk::cuda::jit::launch_direct_by_box_jit<Evaluator, Real>(jit_cache, args, stream, SRC_TILE, block_size);

            return;
        }
    }
#endif

    constexpr int SPATIAL_DIM = Evaluator::SPATIAL_DIM;
    constexpr int KERNEL_INPUT_DIM = Evaluator::KERNEL_INPUT_DIM;
    constexpr int NORMAL_DIM = Evaluator::NORMAL_DIM;
    constexpr int values_per_source = SPATIAL_DIM + KERNEL_INPUT_DIM + NORMAL_DIM;

    const std::size_t shared_bytes = SRC_TILE * values_per_source * sizeof(Real);

    DirectResidualByBoxKernelTiled<Evaluator, SRC_TILE><<<args.n_work, block_size, shared_bytes, stream>>>(args);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("launch_direct_by_box: ") + cudaGetErrorString(err) +
                                 " shared_bytes=" + std::to_string(shared_bytes));
    }
}
// template <typename Evaluator, typename Real = typename Evaluator::scalar_type, int SRC_TILE = 32>
// inline void launch_direct_by_box(const DirectByBoxArgs<Real> &args, cudaStream_t stream = 0) {
//     if (args.n_work == 0)
//         return;

//     constexpr int block_size = 128;
//     constexpr int SPATIAL_DIM = Evaluator::SPATIAL_DIM;
//     constexpr int KERNEL_INPUT_DIM = Evaluator::KERNEL_INPUT_DIM;
//     constexpr int NORMAL_DIM = Evaluator::NORMAL_DIM;
//     constexpr int values_per_source = SPATIAL_DIM + KERNEL_INPUT_DIM + NORMAL_DIM;
//     const std::size_t shared_bytes = SRC_TILE * values_per_source * sizeof(Real);

//     DirectResidualByBoxKernelTiled<Evaluator, SRC_TILE><<<args.n_work, block_size, shared_bytes, stream>>>(args);

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//         throw std::runtime_error(std::string("launch_direct_by_box: ") + cudaGetErrorString(err) +
//                                  " shared_bytes=" + std::to_string(shared_bytes));
// }

} // namespace dmk::cuda

#ifdef DMK_CUDA_USE_NVRTC_JIT
#include "cuda/jit/direct_launcher.tpp"
#endif

#endif // DMK_CUDA_DIRECT_KERNELS_CUH
