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

#include <dmk/cuda/eval_targets_kernels.hpp>

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cstdint>

namespace dmk::cuda {

template <typename Real>
__device__ __forceinline__ void cooperative_load_16B(
    Real *__restrict__ dst,
    const Real *__restrict__ src,
    int n
) {
    using Vec16 = uint4;
    constexpr int VEC_ELEMS = sizeof(Vec16) / sizeof(Real);

    const std::uintptr_t src_addr = reinterpret_cast<std::uintptr_t>(src);
    const std::uintptr_t dst_addr = reinterpret_cast<std::uintptr_t>(dst);

    if (((src_addr | dst_addr) & 0xF) == 0) {
        const int n_vec = n / VEC_ELEMS;

        const Vec16 *__restrict__ src_v = reinterpret_cast<const Vec16 *>(src);
        Vec16 *__restrict__ dst_v = reinterpret_cast<Vec16 *>(dst);

        for (int q = threadIdx.x; q < n_vec; q += blockDim.x) {
            dst_v[q] = src_v[q];
        }

        for (int r = n_vec * VEC_ELEMS + threadIdx.x; r < n; r += blockDim.x) {
            dst[r] = src[r];
        }
    } else {
        for (int r = threadIdx.x; r < n; r += blockDim.x) {
            dst[r] = src[r];
        }
    }
}

// T_0 = 1, T_1 = x, T_n = 2x*T_{n-1} - T_{n-2}
template <typename Real, int N>
__device__ inline void chebyshev_fill_dev(Real (&T)[N], Real x) {
    if constexpr (N > 0)
        T[0] = Real{1};
    if constexpr (N > 1)
        T[1] = x;
    const Real two_x = Real{2} * x;
    #pragma unroll
    for (int i = 2; i < N; ++i)
        T[i] = two_x * T[i - 1] - T[i - 2];
}

// T'_0 = 0, T'_1 = 1, T'_n = 2*T_{n-1} + 2x*T'_{n-1} - T'_{n-2}
template <typename Real, int N>
__device__ inline void chebyshev_fill_dev_with_deriv(Real (&T)[N], Real (&dT)[N], Real x) {
    if constexpr (N > 0) {
        T[0] = Real{1};
        dT[0] = Real{0};
    }
    if constexpr (N > 1) {
        T[1] = x;
        dT[1] = Real{1};
    }
    const Real two_x = Real{2} * x;
    #pragma unroll
    for (int i = 2; i < N; ++i) {
        T[i] = two_x * T[i - 1] - T[i - 2];
        dT[i] = Real{2} * T[i - 1] + two_x * dT[i - 1] - dT[i - 2];
    }
}


template <typename Real, int DIM, int EVAL_LEVEL, int N_CHARGE_DIM, int N_ORDER>
__global__ void EvalTargetsByBoxKernel(EvalTargetsArgs<Real> a) {
    static_assert(DIM == 2 || DIM == 3, "DIM must be 2 or 3");
    static_assert(EVAL_LEVEL == 1 || EVAL_LEVEL == 2, "EVAL_LEVEL must be 1 or 2");
    static_assert(N_CHARGE_DIM == 1 || N_CHARGE_DIM == 3, "N_CHARGE_DIM must be 1 or 3");

    constexpr int OUT_DIM = (EVAL_LEVEL == 1) ? 1 : (DIM + 1);
    constexpr int POT_STRIDE = N_CHARGE_DIM * OUT_DIM;

    constexpr int n2 = N_ORDER * N_ORDER;
    constexpr int coeffs_stride_per_dim = (DIM == 2) ? n2 : n2 * N_ORDER;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    Real *__restrict__ s_cd = reinterpret_cast<Real *>(smem_raw);

    const int box_idx = blockIdx.x;
    if (box_idx >= a.n_eval_boxes)
        return;

    const int box = a.eval_targets_box_list[box_idx];
    const int n_target = a.target_counts[box];
    if (n_target == 0)
        return;

    const Real *__restrict__ coeffs = a.proxy_flat + a.proxy_offsets[box];

    const Real *__restrict__ r_target = a.r_target_flat + a.r_target_offsets[box];

    Real *__restrict__ pot = a.pot_flat + a.pot_offsets[box];

    const Real *__restrict__ cen = a.centers + box * DIM;

    const Real sc = a.sc_per_level[a.box_levels[box]];

    for (int d = 0; d < N_CHARGE_DIM; ++d) {
        const Real *__restrict__ cd_g = coeffs + d * coeffs_stride_per_dim;

        cooperative_load_16B(s_cd, cd_g, coeffs_stride_per_dim);
        __syncthreads();

        for (int t = threadIdx.x; t < n_target; t += blockDim.x) {
            Real Tx[N_ORDER] = {Real{0}};
            Real dTx[N_ORDER] = {Real{0}};

            const Real x = (r_target[t * DIM + 0] - cen[0]) * sc;
            const Real y = (r_target[t * DIM + 1] - cen[1]) * sc;

            if constexpr (EVAL_LEVEL == 1) {
                chebyshev_fill_dev(Tx, x);
            } else {
                chebyshev_fill_dev_with_deriv(Tx, dTx, x);
            }

            Real acc_pot = Real{0};
            Real acc_gx  = Real{0};
            Real acc_gy  = Real{0};
            Real acc_gz  = Real{0};

            if constexpr (DIM == 2) {
                Real Ty_jm2 = Real{1};
                Real Ty_jm1 = y;

                Real dTy_jm2 = Real{0};
                Real dTy_jm1 = Real{1};

                for (int j = 0; j < N_ORDER; ++j) {
                    Real Tyj;
                    Real dTyj;

                    if (j == 0) {
                        Tyj = Real{1};
                        dTyj = Real{0};
                    } else if (j == 1) {
                        Tyj = y;
                        dTyj = Real{1};
                    } else {
                        const Real two_y = Real{2} * y;

                        Tyj =
                            two_y * Ty_jm1 - Ty_jm2;

                        dTyj =
                            Real{2} * Ty_jm1 +
                            two_y * dTy_jm1 -
                            dTy_jm2;

                        Ty_jm2 = Ty_jm1;
                        Ty_jm1 = Tyj;

                        dTy_jm2 = dTy_jm1;
                        dTy_jm1 = dTyj;
                    }

                    Real px_pot = Real{0};
                    Real px_gx  = Real{0};

                    #pragma unroll
                    for (int i = 0; i < N_ORDER; ++i) {
                        const Real c = s_cd[i + j * N_ORDER];

                        px_pot += c * Tx[i];

                        if constexpr (EVAL_LEVEL == 2) {
                            px_gx += c * dTx[i];
                        }
                    }

                    acc_pot += px_pot * Tyj;

                    if constexpr (EVAL_LEVEL == 2) {
                        acc_gx += px_gx  * Tyj;
                        acc_gy += px_pot * dTyj;
                    }
                }
            } else {
                const Real z = (r_target[t * DIM + 2] - cen[2]) * sc;

                Real Tz_km2 = Real{1};
                Real Tz_km1 = z;

                Real dTz_km2 = Real{0};
                Real dTz_km1 = Real{1};

                for (int k = 0; k < N_ORDER; ++k) {
                    Real Tzk;
                    Real dTzk;

                    if (k == 0) {
                        Tzk = Real{1};
                        dTzk = Real{0};
                    } else if (k == 1) {
                        Tzk = z;
                        dTzk = Real{1};
                    } else {
                        const Real two_z = Real{2} * z;

                        Tzk =
                            two_z * Tz_km1 - Tz_km2;

                        dTzk =
                            Real{2} * Tz_km1 +
                            two_z * dTz_km1 -
                            dTz_km2;

                        Tz_km2 = Tz_km1;
                        Tz_km1 = Tzk;

                        dTz_km2 = dTz_km1;
                        dTz_km1 = dTzk;
                    }

                    Real py_pot = Real{0};
                    Real py_gx  = Real{0};
                    Real py_gy  = Real{0};

                    Real Ty_jm2 = Real{1};
                    Real Ty_jm1 = y;

                    Real dTy_jm2 = Real{0};
                    Real dTy_jm1 = Real{1};

                    #pragma unroll
                    for (int j = 0; j < N_ORDER; ++j) {
                        Real Tyj;
                        Real dTyj;

                        if (j == 0) {
                            Tyj = Real{1};
                            dTyj = Real{0};
                        } else if (j == 1) {
                            Tyj = y;
                            dTyj = Real{1};
                        } else {
                            const Real two_y = Real{2} * y;

                            Tyj =
                                two_y * Ty_jm1 - Ty_jm2;

                            dTyj =
                                Real{2} * Ty_jm1 +
                                two_y * dTy_jm1 -
                                dTy_jm2;

                            Ty_jm2 = Ty_jm1;
                            Ty_jm1 = Tyj;

                            dTy_jm2 = dTy_jm1;
                            dTy_jm1 = dTyj;
                        }

                        Real px_pot = Real{0};
                        Real px_gx  = Real{0};

                        #pragma unroll
                        for (int i = 0; i < N_ORDER; ++i) {
                            const Real c =
                                s_cd[i + j * N_ORDER + k * n2];

                            px_pot += c * Tx[i];

                            if constexpr (EVAL_LEVEL == 2) {
                                px_gx += c * dTx[i];
                            }
                        }

                        py_pot += px_pot * Tyj;

                        if constexpr (EVAL_LEVEL == 2) {
                            py_gx += px_gx  * Tyj;
                            py_gy += px_pot * dTyj;
                        }
                    }

                    acc_pot += py_pot * Tzk;

                    if constexpr (EVAL_LEVEL == 2) {
                        acc_gx += py_gx  * Tzk;
                        acc_gy += py_gy  * Tzk;
                        acc_gz += py_pot * dTzk;
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

                if constexpr (DIM == 3) {
                    pot[base + 3] += sc * acc_gz;
                }
            }
        }

        __syncthreads();
    }
}


template <typename Real, int DIM, int EVAL_LEVEL, int N_CHARGE_DIM, int N_ORDER>
inline void launch_eval_targets_kernel_impl(
    const EvalTargetsArgs<Real> &args,
    cudaStream_t stream
) {
    if (args.n_eval_boxes == 0)
        return;

    constexpr int block_size = 640;
    constexpr int n2 = N_ORDER * N_ORDER;
    constexpr int coeffs_stride_per_dim = (DIM == 2) ? n2 : n2 * N_ORDER;

    const std::size_t shared_bytes = static_cast<std::size_t>(coeffs_stride_per_dim) * sizeof(Real);

    EvalTargetsByBoxKernel<Real, DIM, EVAL_LEVEL, N_CHARGE_DIM, N_ORDER>
        <<<args.n_eval_boxes, block_size, shared_bytes, stream>>>(args);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_eval_targets_kernel: ") +
            cudaGetErrorString(err) +
            " (dim=" + std::to_string(DIM) +
            " eval_level=" + std::to_string(EVAL_LEVEL) +
            " n_charge_dim=" + std::to_string(N_CHARGE_DIM) +
            " n_order=" + std::to_string(args.n_order) +
            " n_eval_boxes=" + std::to_string(args.n_eval_boxes) + ")"
        );
    }
}

template <typename Real, int DIM, int EVAL_LEVEL, int N_CHARGE_DIM>
inline void launch_eval_targets_kernel(
    const EvalTargetsArgs<Real> &args,
    cudaStream_t stream
) {
    if (args.n_eval_boxes == 0)
        return;

#define DISPATCH_N_ORDER(N)                                                   \
    case N:                                                                   \
        launch_eval_targets_kernel_impl<                                      \
            Real, DIM, EVAL_LEVEL, N_CHARGE_DIM, N                            \
        >(args, stream);                                                      \
        break

    switch (args.n_order) {
        DISPATCH_N_ORDER(5);
        DISPATCH_N_ORDER(6);
        DISPATCH_N_ORDER(7);
        DISPATCH_N_ORDER(8);
        DISPATCH_N_ORDER(9);
        DISPATCH_N_ORDER(10);
        DISPATCH_N_ORDER(11);
        DISPATCH_N_ORDER(12);
        DISPATCH_N_ORDER(13);
        DISPATCH_N_ORDER(14);
        DISPATCH_N_ORDER(15);
        DISPATCH_N_ORDER(16);
        DISPATCH_N_ORDER(17);
        DISPATCH_N_ORDER(18);        
        // DISPATCH_N_ORDER(19);
        // DISPATCH_N_ORDER(20);
        // DISPATCH_N_ORDER(21);
        // DISPATCH_N_ORDER(22);
        // DISPATCH_N_ORDER(23);
        // DISPATCH_N_ORDER(24);
        // DISPATCH_N_ORDER(25);
        // DISPATCH_N_ORDER(26);
        // DISPATCH_N_ORDER(27);
        // DISPATCH_N_ORDER(28);
        // DISPATCH_N_ORDER(29);
        // DISPATCH_N_ORDER(30);
        // DISPATCH_N_ORDER(31);
        // DISPATCH_N_ORDER(32);
        // DISPATCH_N_ORDER(33);
        // DISPATCH_N_ORDER(34);
        // DISPATCH_N_ORDER(35);
        // DISPATCH_N_ORDER(36);
        // DISPATCH_N_ORDER(37);
        // DISPATCH_N_ORDER(38);
        // DISPATCH_N_ORDER(39);
        // DISPATCH_N_ORDER(40);
        // DISPATCH_N_ORDER(41);
        // DISPATCH_N_ORDER(42);
        // DISPATCH_N_ORDER(43);
        // DISPATCH_N_ORDER(44);
        // DISPATCH_N_ORDER(45);
        // DISPATCH_N_ORDER(46); //EWWW

        default:
            throw std::runtime_error(
                "launch_eval_targets_kernel: unsupported n_order=" +
                std::to_string(args.n_order)
            );
    }

#undef DISPATCH_N_ORDER
}

} // namespace dmk::cuda

#endif // DMK_CUDA_EVAL_TARGETS_KERNELS_CUH
