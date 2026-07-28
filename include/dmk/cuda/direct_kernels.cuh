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

#include "cuda/jit/direct_launcher.hpp"

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

// Placeholder: GPU Laplace-dipole direct evaluation is not implemented yet.
// The merge that added the dipole kernel to the AOT list left the CUDA path a
// stub, so this returns zero to keep the dispatch compiling. Wire in the real
// potential (u[k][0] = dX[k] * (P*Rinv^3 - dP*rsc*Rinv^2)) when GPU dipole lands.
template <CoeffTag Coeffs>
struct LaplaceDipolePolyEvaluator3DCuda {
    using scalar_type = typename Coeffs::value_type;
    using Real = scalar_type;
    static constexpr int SPATIAL_DIM = 3;
    static constexpr int KERNEL_INPUT_DIM = 3;
    static constexpr int KERNEL_OUTPUT_DIM = 1;
    static constexpr int NORMAL_DIM = 0;
    static constexpr Real scale_factor = Real{1.0};
    Real thresh2, d2max, rsc, cen;

    __device__ inline void operator()(Real (&u)[3][1], const Real (&dX)[3]) const {
        for (int k = 0; k < 3; ++k)
            u[k][0] = Real{0};
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

template <typename Evaluator, typename Real = typename Evaluator::scalar_type, int SRC_TILE = 32>
inline void launch_direct_by_box(const DirectByBoxArgs<Real> &args, cudaStream_t stream = 0) {
    if (args.n_work == 0)
        return;

    constexpr int block_size = 128;

    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_direct_by_box_jit<Evaluator, Real>(jit_cache, args, stream, SRC_TILE, block_size);
}
} // namespace dmk::cuda

#include "cuda/jit/direct_launcher.tpp"

#endif // DMK_CUDA_DIRECT_KERNELS_CUH
