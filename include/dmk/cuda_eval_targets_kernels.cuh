#ifndef DMK_CUDA_EVAL_TARGETS_KERNELS_CUH
#define DMK_CUDA_EVAL_TARGETS_KERNELS_CUH

// One block per iftensprodeval box; threads stride over the box's owned
// target points. Each thread evaluates the proxy expansion at its target
// via Chebyshev tensor product and accumulates into pot_*_eval[box_offset].
// No race because each box has a disjoint pot slice.
//
// Output layout matches host proxy::eval_targets:
//   pot has shape (N_CHARGE_DIM * OUT_DIM, n_trg) in F_layout
//   where OUT_DIM = 1 for EVAL_LEVEL=1, DIM+1 for EVAL_LEVEL=2.
//   pot[d*OUT_DIM + j + t*POT_STRIDE] with POT_STRIDE = N_CHARGE_DIM*OUT_DIM.
//
// EVAL_LEVEL=2 returns potential + gradient. For DIM=2 that's (pot, gx, gy);
// for DIM=3, (pot, gx, gy, gz). Gradients are scaled by `sc` to match host.

#include <dmk/cuda_eval_targets_kernels.hpp>

#include <cuda_runtime.h>

namespace dmk::cuda {

// T_0 = 1, T_1 = x, T_n = 2x*T_{n-1} - T_{n-2}
template <typename Real, int N>
__device__ inline void chebyshev_fill_dev(Real (&T)[N], Real x, int n_order) {
    if (n_order > 0)
        T[0] = Real{1};
    if (n_order > 1)
        T[1] = x;
    const Real two_x = Real{2} * x;
    for (int i = 2; i < n_order; ++i)
        T[i] = two_x * T[i - 1] - T[i - 2];
}

// T'_0 = 0, T'_1 = 1, T'_n = 2*T_{n-1} + 2x*T'_{n-1} - T'_{n-2}
template <typename Real, int N>
__device__ inline void chebyshev_fill_dev_with_deriv(Real (&T)[N], Real (&dT)[N], Real x, int n_order) {
    if (n_order > 0) {
        T[0] = Real{1};
        dT[0] = Real{0};
    }
    if (n_order > 1) {
        T[1] = x;
        dT[1] = Real{1};
    }
    const Real two_x = Real{2} * x;
    for (int i = 2; i < n_order; ++i) {
        T[i] = two_x * T[i - 1] - T[i - 2];
        dT[i] = Real{2} * T[i - 1] + two_x * dT[i - 1] - dT[i - 2];
    }
}

template <typename Real, int DIM, int EVAL_LEVEL, int N_CHARGE_DIM, int MAX_N_ORDER>
__global__ void EvalTargetsByBoxKernel(EvalTargetsArgs<Real> a) {
    static_assert(DIM == 2 || DIM == 3, "DIM must be 2 or 3");
    static_assert(EVAL_LEVEL == 1 || EVAL_LEVEL == 2, "EVAL_LEVEL must be 1 or 2");
    static_assert(N_CHARGE_DIM == 1 || N_CHARGE_DIM == 3, "N_CHARGE_DIM must be 1 or 3");

    constexpr int OUT_DIM = (EVAL_LEVEL == 1) ? 1 : (DIM + 1);
    constexpr int POT_STRIDE = N_CHARGE_DIM * OUT_DIM;

    const int box_idx = blockIdx.x;
    if (box_idx >= a.n_eval_boxes)
        return;
    const int box = a.eval_targets_box_list[box_idx];
    const int n_target = a.target_counts[box];
    if (n_target == 0)
        return;

    const Real *coeffs = a.proxy_flat + a.proxy_offsets[box];
    const Real *r_target = a.r_target_flat + a.r_target_offsets[box];
    Real *pot = a.pot_flat + a.pot_offsets[box];
    const Real *cen = a.centers + box * DIM;
    const Real sc = a.sc_per_level[a.box_levels[box]];
    const int n_order = a.n_order;
    const int n2 = n_order * n_order;
    const int coeffs_stride_per_dim = (DIM == 2) ? n2 : n2 * n_order;

    for (int t = threadIdx.x; t < n_target; t += blockDim.x) {
        Real Tx[MAX_N_ORDER], Ty[MAX_N_ORDER];
        Real dTx[MAX_N_ORDER], dTy[MAX_N_ORDER];
        // Tz/dTz only used for DIM==3, but always declared since arrays are stack.
        Real Tz[MAX_N_ORDER], dTz[MAX_N_ORDER];

        const Real x = (r_target[t * DIM + 0] - cen[0]) * sc;
        const Real y = (r_target[t * DIM + 1] - cen[1]) * sc;
        if constexpr (EVAL_LEVEL == 1) {
            chebyshev_fill_dev(Tx, x, n_order);
            chebyshev_fill_dev(Ty, y, n_order);
        } else {
            chebyshev_fill_dev_with_deriv(Tx, dTx, x, n_order);
            chebyshev_fill_dev_with_deriv(Ty, dTy, y, n_order);
        }
        if constexpr (DIM == 3) {
            const Real z = (r_target[t * DIM + 2] - cen[2]) * sc;
            if constexpr (EVAL_LEVEL == 1)
                chebyshev_fill_dev(Tz, z, n_order);
            else
                chebyshev_fill_dev_with_deriv(Tz, dTz, z, n_order);
        }

        for (int d = 0; d < N_CHARGE_DIM; ++d) {
            const Real *cd = coeffs + d * coeffs_stride_per_dim;

            Real acc_pot = Real{0};
            Real acc_gx = Real{0}, acc_gy = Real{0}, acc_gz = Real{0};

            if constexpr (DIM == 2) {
                for (int j = 0; j < n_order; ++j) {
                    Real px_pot = Real{0}, px_gx = Real{0};
                    for (int i = 0; i < n_order; ++i) {
                        const Real c = cd[i + j * n_order];
                        px_pot += c * Tx[i];
                        if constexpr (EVAL_LEVEL == 2)
                            px_gx += c * dTx[i];
                    }
                    acc_pot += px_pot * Ty[j];
                    if constexpr (EVAL_LEVEL == 2) {
                        acc_gx += px_gx * Ty[j];
                        acc_gy += px_pot * dTy[j];
                    }
                }
            } else { // DIM == 3
                for (int k = 0; k < n_order; ++k) {
                    Real py_pot = Real{0}, py_gx = Real{0}, py_gy = Real{0};
                    for (int j = 0; j < n_order; ++j) {
                        Real px_pot = Real{0}, px_gx = Real{0};
                        for (int i = 0; i < n_order; ++i) {
                            const Real c = cd[i + j * n_order + k * n2];
                            px_pot += c * Tx[i];
                            if constexpr (EVAL_LEVEL == 2)
                                px_gx += c * dTx[i];
                        }
                        py_pot += px_pot * Ty[j];
                        if constexpr (EVAL_LEVEL == 2) {
                            py_gx += px_gx * Ty[j];
                            py_gy += px_pot * dTy[j];
                        }
                    }
                    acc_pot += py_pot * Tz[k];
                    if constexpr (EVAL_LEVEL == 2) {
                        acc_gx += py_gx * Tz[k];
                        acc_gy += py_gy * Tz[k];
                        acc_gz += py_pot * dTz[k];
                    }
                }
            }

            const int base = d * OUT_DIM + t * POT_STRIDE;
            if constexpr (EVAL_LEVEL == 1) {
                pot[base] += acc_pot;
            } else {
                pot[base + 0] += acc_pot;
                pot[base + 1] += sc * acc_gx;
                pot[base + 2] += sc * acc_gy;
                if constexpr (DIM == 3)
                    pot[base + 3] += sc * acc_gz;
            }
        }
    }
}

template <typename Real, int DIM, int EVAL_LEVEL, int N_CHARGE_DIM, int MAX_N_ORDER>
inline void launch_eval_targets_kernel(const EvalTargetsArgs<Real> &args, cudaStream_t stream) {
    if (args.n_eval_boxes == 0)
        return;
    constexpr int block_size = 512;
    EvalTargetsByBoxKernel<Real, DIM, EVAL_LEVEL, N_CHARGE_DIM, MAX_N_ORDER>
        <<<args.n_eval_boxes, block_size, 0, stream>>>(args);
}

} // namespace dmk::cuda

#endif // DMK_CUDA_EVAL_TARGETS_KERNELS_CUH
