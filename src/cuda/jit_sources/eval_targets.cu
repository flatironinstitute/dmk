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

            if constexpr (EVAL_LEVEL == 1)
                chebyshev_fill_dev(Tx, x);
            else
                chebyshev_fill_dev_with_deriv(Tx, dTx, x);

            Real acc_pot = Real{0};
            Real acc_gx = Real{0};
            Real acc_gy = Real{0};
            Real acc_gz = Real{0};

            if constexpr (DIM == 2) {
                Real Ty_jm2 = Real{1}, Ty_jm1 = y;
                Real dTy_jm2 = Real{0}, dTy_jm1 = Real{1};
                const Real two_y = Real{2} * y;

                for (int j = 0; j < N_ORDER; ++j) {
                    Real Tyj, dTyj;
                    cheby_step(j, y, two_y, Ty_jm2, Ty_jm1, dTy_jm2, dTy_jm1, Tyj, dTyj);

                    Real px_pot = Real{0};
                    Real px_gx = Real{0};

#pragma unroll
                    for (int i = 0; i < N_ORDER; ++i) {
                        const Real c = s_cd[i + j * N_ORDER];
                        px_pot += c * Tx[i];
                        if constexpr (EVAL_LEVEL == 2)
                            px_gx += c * dTx[i];
                    }

                    acc_pot += px_pot * Tyj;
                    if constexpr (EVAL_LEVEL == 2) {
                        acc_gx += px_gx * Tyj;
                        acc_gy += px_pot * dTyj;
                    }
                }
            } else {
                const Real z = (r_target[t * DIM + 2] - cen[2]) * sc;

                Real Tz_km2 = Real{1}, Tz_km1 = z;
                Real dTz_km2 = Real{0}, dTz_km1 = Real{1};
                const Real two_z = Real{2} * z;
                const Real two_y = Real{2} * y;

                for (int k = 0; k < N_ORDER; ++k) {
                    Real Tzk, dTzk;
                    cheby_step(k, z, two_z, Tz_km2, Tz_km1, dTz_km2, dTz_km1, Tzk, dTzk);

                    Real py_pot = Real{0};
                    Real py_gx = Real{0};
                    Real py_gy = Real{0};

                    Real Ty_jm2 = Real{1}, Ty_jm1 = y;
                    Real dTy_jm2 = Real{0}, dTy_jm1 = Real{1};

#pragma unroll
                    for (int j = 0; j < N_ORDER; ++j) {
                        Real Tyj, dTyj;
                        cheby_step(j, y, two_y, Ty_jm2, Ty_jm1, dTy_jm2, dTy_jm1, Tyj, dTyj);

                        Real px_pot = Real{0};
                        Real px_gx = Real{0};

#pragma unroll
                        for (int i = 0; i < N_ORDER; ++i) {
                            const Real c = s_cd[i + j * N_ORDER + k * n2];
                            px_pot += c * Tx[i];
                            if constexpr (EVAL_LEVEL == 2)
                                px_gx += c * dTx[i];
                        }

                        py_pot += px_pot * Tyj;
                        if constexpr (EVAL_LEVEL == 2) {
                            py_gx += px_gx * Tyj;
                            py_gy += px_pot * dTyj;
                        }
                    }

                    acc_pot += py_pot * Tzk;
                    if constexpr (EVAL_LEVEL == 2) {
                        acc_gx += py_gx * Tzk;
                        acc_gy += py_gy * Tzk;
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
                if constexpr (DIM == 3)
                    pot[base + 3] += sc * acc_gz;
            }
        }

        __syncthreads();
    }
}
