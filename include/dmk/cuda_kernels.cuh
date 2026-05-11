#ifndef DMK_CUDA_KERNELS_CUH
#define DMK_CUDA_KERNELS_CUH

//   ___ __  __ ____   ___  ____ _____  _    _   _ _____
//  |_ _|  \/  |  _ \ / _ \|  _ \_   _|/ \  | \ | |_   _|
//   | || |\/| | |_) | | | | |_) || | / _ \ |  \| | | |
//   | || |  | |  __/| |_| |  _ < | |/ ___ \| |\  | | |
//  |___|_|  |_|_|    \___/|_| \_\|_/_/   \_\_| \_| |_|
//
// CUDA mirror of vector_kernels.hpp. Coefficient tables ride into the kernel
// as compile-time *types* (CoeffTag) — not as runtime pointers and not as
// structural-class NTTP values (which trip a cudafe stub-generation bug
// in nvcc when used as template args to __global__ templates). A tag is
// just a struct exposing:
//
//   using value_type = Real;
//   static constexpr std::size_t size = N;
//   __host__ __device__ static constexpr Real at(std::size_t i) {
//       constexpr Real v[N] = { ... };
//       return v[i];
//   }
//
// The accessor is a *function*, not a static data member, because nvcc does
// not treat class-scope `static constexpr T data[N]` arrays as device-visible
// (they're host-only declarations even when constexpr). Encoding the values
// as literals inside a dual-annotated constexpr function sidesteps that —
// `Coeffs::at(I)` for constexpr `I` is constant-folded into an immediate on
// both host and device, with no separate device symbol needed.
//
// Benefits over runtime coefficients:
//   * polynomial degree is fixed at instantiation,
//   * horner_const unrolls fully via template recursion (no optimizer
//     dependency for constant folding), and
//   * each Coeffs::at(I) for constexpr I is a constant expression, so
//     coefficient values become PTX/SASS immediates rather than loads.
//
// Each evaluator carries only the runtime fields it actually needs (thresh2,
// d2max, rsc, cen, …) — no coeff pointer, no n_coeffs, no n_digits. The
// AOT-generated cuda_kernels.cu defines one tag struct per
// (kernel, digits, Real) and uses it as the type-template argument.
//
// All r_*, charge, normals, pot pointers passed to the launchers MUST be
// device-resident. No H2D/D2H copies happen here.
//
// Known incompleteness in the bootstrap:
//   * gradient/Hessian eval levels are not implemented (potential only)

#include <cuda_runtime.h>
#include <cstddef>

namespace dmk::cuda {

// =========================================================================
// CoeffTag concept: a type carrying a compile-time coefficient table.
// =========================================================================
template <typename C>
concept CoeffTag = requires {
    typename C::value_type;
    { C::size } -> std::convertible_to<std::size_t>;
    { C::at(0) } -> std::convertible_to<typename C::value_type>;
};

// =========================================================================
// Template-recursive Horner. With Coeffs as a *type*, each Coeffs::at(I)
// for constexpr I is a constant expression — the unrolled chain becomes a
// sequence of FMAs against literal immediates.
// =========================================================================
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

// =========================================================================
// EvalPairsCuda: tiled, shared-memory, all-pairs driver. One thread per
// target. Source tiles are staged into shared memory and reused by every
// thread in the block. Mirrors host EvalPairs in vector_kernels.hpp.
//
// pot_dev is *accumulated into* (matches host residual_evaluator_func contract).
// =========================================================================
template <int KERNEL_OUTPUT_DIM, DeviceKernelEvaluator Evaluator>
__global__ void EvalPairsCuda(int n_src,
                              const typename Evaluator::scalar_type *__restrict__ r_src,
                              const typename Evaluator::scalar_type *__restrict__ v_src,
                              const typename Evaluator::scalar_type *__restrict__ src_normals,
                              int n_trg,
                              const typename Evaluator::scalar_type *__restrict__ r_trg,
                              typename Evaluator::scalar_type *__restrict__ v_trg,
                              Evaluator evaluator) {
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

// =========================================================================
// Per-kernel device evaluators. Each is templated on its CoeffPack(s); the
// scalar type is recovered from the pack's value_type. No coeff pointers,
// no n_coeffs runtime fields — those live in the type.
// =========================================================================

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
        const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
        const Real xmapped = (R2 * Rinv + cen) * rsc;
        const Real P = horner_const<Coeffs, Real>(xmapped);
        u[0][0] = in_range ? P * Rinv : Real{0};
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
        const Real Rinv = R2 > Real{0} ? rsqrt(R2) : Real{0};
        const Real R2inv = Rinv * Rinv;
        const Real arg = rsc * R2 + cen;
        u[0][0] = in_range ? R2inv * horner_const<Coeffs, Real>(arg) : Real{0};
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
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 3; ++i) {
                    u[j][i] = 0.0;
                }
            }
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
                u[i][j] = in_range ? val : Real{0};
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

// =========================================================================
// Host-side launcher templates. All pointers device-resident. Coefficients
// arrive via NTTP and never appear as runtime args.
// =========================================================================

namespace detail {
constexpr int default_block_size = 128;

template <typename Evaluator, int KERNEL_OUTPUT_DIM, typename Real>
inline void launch_eval_pairs(const Evaluator &evaluator, int n_src, const Real *r_src, const Real *charge,
                              const Real *normals, int n_trg, const Real *r_trg, Real *pot, cudaStream_t stream) {
    constexpr int block_size = default_block_size;
    const int grid = (n_trg + block_size - 1) / block_size;
    constexpr int row = Evaluator::SPATIAL_DIM + Evaluator::KERNEL_INPUT_DIM + Evaluator::NORMAL_DIM;
    const int smem = block_size * row * sizeof(Real);
    EvalPairsCuda<KERNEL_OUTPUT_DIM, Evaluator><<<grid, block_size, smem, stream>>>(
        n_src, r_src, charge, normals, n_trg, r_trg, pot, evaluator);
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

} // namespace dmk::cuda

#endif // DMK_CUDA_KERNELS_CUH
