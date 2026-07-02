__device__ __forceinline__ void cooperative_load_16B(Real *__restrict__ dst, const Real *__restrict__ src, int n) {
    using Vec16 = uint4;
    constexpr int VEC_ELEMS = sizeof(Vec16) / sizeof(Real);

    const unsigned long long src_addr = reinterpret_cast<unsigned long long>(src);
    const unsigned long long dst_addr = reinterpret_cast<unsigned long long>(dst);

    if (((src_addr | dst_addr) & 0xF) == 0) {
        const int n_vec = n / VEC_ELEMS;
        const Vec16 *__restrict__ src_v = reinterpret_cast<const Vec16 *>(src);
        Vec16 *__restrict__ dst_v = reinterpret_cast<Vec16 *>(dst);

        for (int q = threadIdx.x; q < n_vec; q += blockDim.x)
            dst_v[q] = src_v[q];
        for (int r = n_vec * VEC_ELEMS + threadIdx.x; r < n; r += blockDim.x)
            dst[r] = src[r];
    } else {
        for (int r = threadIdx.x; r < n; r += blockDim.x)
            dst[r] = src[r];
    }
}

__device__ inline void chebyshev_fill_dev(Real (&T)[N_ORDER], Real x) {
    if constexpr (N_ORDER > 0)
        T[0] = Real{1};
    if constexpr (N_ORDER > 1)
        T[1] = x;
    const Real two_x = Real{2} * x;
#pragma unroll
    for (int i = 2; i < N_ORDER; ++i)
        T[i] = two_x * T[i - 1] - T[i - 2];
}

__device__ inline void chebyshev_fill_dev_with_deriv(Real (&T)[N_ORDER], Real (&dT)[N_ORDER], Real x) {
    if constexpr (N_ORDER > 0) {
        T[0] = Real{1};
        dT[0] = Real{0};
    }
    if constexpr (N_ORDER > 1) {
        T[1] = x;
        dT[1] = Real{1};
    }
    const Real two_x = Real{2} * x;
#pragma unroll
    for (int i = 2; i < N_ORDER; ++i) {
        T[i] = two_x * T[i - 1] - T[i - 2];
        dT[i] = Real{2} * T[i - 1] + two_x * dT[i - 1] - dT[i - 2];
    }
}

__device__ __forceinline__ void cheby_step(int j, Real x, Real two_x, Real &Tprev2, Real &Tprev1, Real &dTprev2,
                                           Real &dTprev1, Real &Tj, Real &dTj) {
    if (j == 0) {
        Tj = Real{1};
        dTj = Real{0};
    } else if (j == 1) {
        Tj = x;
        dTj = Real{1};
    } else {
        Tj = two_x * Tprev1 - Tprev2;
        dTj = Real{2} * Tprev1 + two_x * dTprev1 - dTprev2;
    }
    Tprev2 = Tprev1;
    Tprev1 = Tj;
    dTprev2 = dTprev1;
    dTprev1 = dTj;
}

// KERNEL_START

extern "C" __global__ void EvalTargetsByBoxKernel(EvalTargetsArgs<Real> a) {
    constexpr int OUT_DIM = (EVAL_LEVEL == 1) ? 1 : (DIM + 1);
    constexpr int POT_STRIDE = N_CHARGE_DIM * OUT_DIM;
    static_assert(TARGETS_PER_THREAD > 0, "TARGETS_PER_THREAD must be positive");

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

        const int target_stride = blockDim.x * TARGETS_PER_THREAD;
        for (int t_base = threadIdx.x; t_base < n_target; t_base += target_stride) {
            bool active[TARGETS_PER_THREAD];
            int target_idx[TARGETS_PER_THREAD];
            Real x[TARGETS_PER_THREAD];
            Real y[TARGETS_PER_THREAD];
            Real Tx[TARGETS_PER_THREAD][N_ORDER];
            Real dTx[TARGETS_PER_THREAD][N_ORDER];

#pragma unroll
            for (int q = 0; q < TARGETS_PER_THREAD; ++q) {
                const int t = t_base + q * blockDim.x;
                active[q] = t < n_target;
                target_idx[q] = t;

                if (active[q]) {
                    x[q] = (r_target[t * DIM + 0] - cen[0]) * sc;
                    y[q] = (r_target[t * DIM + 1] - cen[1]) * sc;
                } else {
                    x[q] = Real{0};
                    y[q] = Real{0};
                }

                if constexpr (EVAL_LEVEL == 1)
                    chebyshev_fill_dev(Tx[q], x[q]);
                else
                    chebyshev_fill_dev_with_deriv(Tx[q], dTx[q], x[q]);
            }

            Real acc_pot[TARGETS_PER_THREAD] = {};
            Real acc_gx[TARGETS_PER_THREAD] = {};
            Real acc_gy[TARGETS_PER_THREAD] = {};
            Real acc_gz[TARGETS_PER_THREAD] = {};

            if constexpr (DIM == 2) {
                Real Ty_jm2[TARGETS_PER_THREAD];
                Real Ty_jm1[TARGETS_PER_THREAD];
                Real dTy_jm2[TARGETS_PER_THREAD];
                Real dTy_jm1[TARGETS_PER_THREAD];
                Real two_y[TARGETS_PER_THREAD];

#pragma unroll
                for (int q = 0; q < TARGETS_PER_THREAD; ++q) {
                    Ty_jm2[q] = Real{1};
                    Ty_jm1[q] = y[q];
                    dTy_jm2[q] = Real{0};
                    dTy_jm1[q] = Real{1};
                    two_y[q] = Real{2} * y[q];
                }

                for (int j = 0; j < N_ORDER; ++j) {
                    Real Tyj[TARGETS_PER_THREAD];
                    Real dTyj[TARGETS_PER_THREAD];

#pragma unroll
                    for (int q = 0; q < TARGETS_PER_THREAD; ++q)
                        cheby_step(j, y[q], two_y[q], Ty_jm2[q], Ty_jm1[q], dTy_jm2[q], dTy_jm1[q], Tyj[q],
                                   dTyj[q]);

                    Real px_pot[TARGETS_PER_THREAD] = {};
                    Real px_gx[TARGETS_PER_THREAD] = {};

#pragma unroll
                    for (int i = 0; i < N_ORDER; ++i) {
                        const Real c = s_cd[i + j * N_ORDER];

#pragma unroll
                        for (int q = 0; q < TARGETS_PER_THREAD; ++q) {
                            px_pot[q] += c * Tx[q][i];
                            if constexpr (EVAL_LEVEL == 2)
                                px_gx[q] += c * dTx[q][i];
                        }
                    }

#pragma unroll
                    for (int q = 0; q < TARGETS_PER_THREAD; ++q) {
                        acc_pot[q] += px_pot[q] * Tyj[q];
                        if constexpr (EVAL_LEVEL == 2) {
                            acc_gx[q] += px_gx[q] * Tyj[q];
                            acc_gy[q] += px_pot[q] * dTyj[q];
                        }
                    }
                }
            } else {
                Real z[TARGETS_PER_THREAD];
                Real Tz_km2[TARGETS_PER_THREAD];
                Real Tz_km1[TARGETS_PER_THREAD];
                Real dTz_km2[TARGETS_PER_THREAD];
                Real dTz_km1[TARGETS_PER_THREAD];
                Real two_z[TARGETS_PER_THREAD];
                Real two_y[TARGETS_PER_THREAD];

#pragma unroll
                for (int q = 0; q < TARGETS_PER_THREAD; ++q) {
                    if (active[q])
                        z[q] = (r_target[target_idx[q] * DIM + 2] - cen[2]) * sc;
                    else
                        z[q] = Real{0};

                    Tz_km2[q] = Real{1};
                    Tz_km1[q] = z[q];
                    dTz_km2[q] = Real{0};
                    dTz_km1[q] = Real{1};
                    two_z[q] = Real{2} * z[q];
                    two_y[q] = Real{2} * y[q];
                }

                for (int k = 0; k < N_ORDER; ++k) {
                    Real Tzk[TARGETS_PER_THREAD];
                    Real dTzk[TARGETS_PER_THREAD];

#pragma unroll
                    for (int q = 0; q < TARGETS_PER_THREAD; ++q)
                        cheby_step(k, z[q], two_z[q], Tz_km2[q], Tz_km1[q], dTz_km2[q], dTz_km1[q], Tzk[q],
                                   dTzk[q]);

                    Real py_pot[TARGETS_PER_THREAD] = {};
                    Real py_gx[TARGETS_PER_THREAD] = {};
                    Real py_gy[TARGETS_PER_THREAD] = {};

                    Real Ty_jm2[TARGETS_PER_THREAD];
                    Real Ty_jm1[TARGETS_PER_THREAD];
                    Real dTy_jm2[TARGETS_PER_THREAD];
                    Real dTy_jm1[TARGETS_PER_THREAD];

#pragma unroll
                    for (int q = 0; q < TARGETS_PER_THREAD; ++q) {
                        Ty_jm2[q] = Real{1};
                        Ty_jm1[q] = y[q];
                        dTy_jm2[q] = Real{0};
                        dTy_jm1[q] = Real{1};
                    }

#pragma unroll
                    for (int j = 0; j < N_ORDER; ++j) {
                        Real Tyj[TARGETS_PER_THREAD];
                        Real dTyj[TARGETS_PER_THREAD];

#pragma unroll
                        for (int q = 0; q < TARGETS_PER_THREAD; ++q)
                            cheby_step(j, y[q], two_y[q], Ty_jm2[q], Ty_jm1[q], dTy_jm2[q], dTy_jm1[q], Tyj[q],
                                       dTyj[q]);

                        Real px_pot[TARGETS_PER_THREAD] = {};
                        Real px_gx[TARGETS_PER_THREAD] = {};

#pragma unroll
                        for (int i = 0; i < N_ORDER; ++i) {
                            const Real c = s_cd[i + j * N_ORDER + k * n2];

#pragma unroll
                            for (int q = 0; q < TARGETS_PER_THREAD; ++q) {
                                px_pot[q] += c * Tx[q][i];
                                if constexpr (EVAL_LEVEL == 2)
                                    px_gx[q] += c * dTx[q][i];
                            }
                        }

#pragma unroll
                        for (int q = 0; q < TARGETS_PER_THREAD; ++q) {
                            py_pot[q] += px_pot[q] * Tyj[q];
                            if constexpr (EVAL_LEVEL == 2) {
                                py_gx[q] += px_gx[q] * Tyj[q];
                                py_gy[q] += px_pot[q] * dTyj[q];
                            }
                        }
                    }

#pragma unroll
                    for (int q = 0; q < TARGETS_PER_THREAD; ++q) {
                        acc_pot[q] += py_pot[q] * Tzk[q];
                        if constexpr (EVAL_LEVEL == 2) {
                            acc_gx[q] += py_gx[q] * Tzk[q];
                            acc_gy[q] += py_gy[q] * Tzk[q];
                            acc_gz[q] += py_pot[q] * dTzk[q];
                        }
                    }
                }
            }

#pragma unroll
            for (int q = 0; q < TARGETS_PER_THREAD; ++q) {
                if (!active[q]) {
                    continue;
                }

                const int base = d * OUT_DIM + target_idx[q] * POT_STRIDE;

                if constexpr (EVAL_LEVEL == 1) {
                    pot[base] += acc_pot[q];
                } else {
                    pot[base + 0] += acc_pot[q];
                    pot[base + 1] += sc * acc_gx[q];
                    pot[base + 2] += sc * acc_gy[q];
                    if constexpr (DIM == 3)
                        pot[base + 3] += sc * acc_gz[q];
                }
            }
        }

        __syncthreads();
    }
}
