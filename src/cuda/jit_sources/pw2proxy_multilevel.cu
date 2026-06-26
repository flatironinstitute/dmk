template <typename Real>
struct alignas(2 * sizeof(Real)) complx {
    Real r;
    Real i;
};

template <typename Real>
__device__ __forceinline__ complx<Real> complx_zero() {
    return complx<Real>{Real{0}, Real{0}};
}

template <typename Real>
__device__ __forceinline__ complx<Real> complx_load(const Real *__restrict__ p, int idx) {
    return reinterpret_cast<const complx<Real> *>(p)[idx];
}

template <typename Real>
__device__ __forceinline__ void complx_madd(complx<Real> &acc, const complx<Real> a, const complx<Real> b) {
    acc.r = fma(a.r, b.r, acc.r);
    acc.r = fma(-a.i, b.i, acc.r);
    acc.i = fma(a.r, b.i, acc.i);
    acc.i = fma(a.i, b.r, acc.i);
}

template <typename Real>
__device__ __forceinline__ Real complx_real_madd(Real acc, const complx<Real> a, const complx<Real> b) {
    acc = fma(a.r, b.r, acc);
    acc = fma(-a.i, b.i, acc);
    return acc;
}

// KERNEL_START

extern "C" __global__ void PwToProxyMultiLevelKernel(const PwToProxyArgs<Real> *__restrict__ args, int n_args) {

    const int box_idx = blockIdx.x;
    const int arg_idx = blockIdx.y;

    if (arg_idx >= n_args)
        return;

    const PwToProxyArgs<Real> a = args[arg_idx];

    if (box_idx >= a.n_boxes_at_level)
        return;

    const int box = a.box_ids[box_idx];

    const long proxy_off = a.proxy_offsets[box];
    if (proxy_off < 0)
        return;

    const int n_pw = N_PW;
    const int n_pw2 = N_PW2;
    const int n_pw_half = n_pw / 2;
    const int n_order = N_ORDER;
    const int n_order2 = n_order * n_order;
    const int n_order3 = n_order2 * n_order;
    const int n_pw_modes = n_pw * n_pw * n_pw2;

    const int k_pad = ((n_order + 3) / 4) * 4;
    const int phase1_cols = n_pw * n_pw2;

    extern __shared__ __align__(16) unsigned char shared_raw[];

    complx<Real> *__restrict__ smem = reinterpret_cast<complx<Real> *>(shared_raw);

    // s_A_T[m, k] = pw2poly(k, m), padded in k for bank/layout control.
    complx<Real> *__restrict__ s_A_T = smem;

    // s_F[c, kr], c = m2 + m3 * n_pw, local k1 fastest for the phase-2 consumer.
    complx<Real> *__restrict__ s_F = s_A_T + n_pw * k_pad;

    // s_G[m3, k2, kr], local k1 fastest for the final proxy store shape.
    complx<Real> *__restrict__ s_G = s_F + phase1_cols * K1_TILE;

    for (int idx = threadIdx.x; idx < n_pw * k_pad; idx += blockDim.x) {
        const int m = idx / k_pad;
        const int k = idx - m * k_pad;

        complx<Real> z = complx_zero<Real>();

        if (k < n_order)
            z = complx_load(a.pw2poly, k * n_pw + m);

        s_A_T[idx] = z;
    }

    __syncthreads();

    const Real *__restrict__ pw_in_box = a.pw_in_pool + box_idx * a.pw_in_stride;
    Real *__restrict__ proxy_box = a.proxy_flat + proxy_off;

    const int k2_tiles = (n_order + K2_TILE - 1) / K2_TILE;
    const int k3_tiles = (n_order + K3_TILE - 1) / K3_TILE;
    const int kr_tiles = (K1_TILE + KR_TILE - 1) / KR_TILE;

    for (int d = 0; d < a.n_charge_dim; ++d) {
        const Real *__restrict__ pw_in_d = pw_in_box + 2 * d * n_pw_modes;
        Real *__restrict__ proxy_d = proxy_box + d * n_order3;

        for (int k1_base = 0; k1_base < n_order; k1_base += K1_TILE) {
            // Phase 1: s_F(m2, m3, kr) = sum_m1 pw(m1, m2, m3) * pw2poly(kr, m1).
            const int col_tiles = (phase1_cols + COL_REG - 1) / COL_REG;

            for (int tile = threadIdx.x; tile < col_tiles; tile += blockDim.x) {
                const int c_base = tile * COL_REG;

                complx<Real> acc[K1_TILE][COL_REG];

#pragma unroll
                for (int kr = 0; kr < K1_TILE; ++kr) {
#pragma unroll
                    for (int cr = 0; cr < COL_REG; ++cr)
                        acc[kr][cr] = complx_zero<Real>();
                }

                for (int m1 = 0; m1 < n_pw; ++m1) {
                    complx<Real> p[COL_REG];

#pragma unroll
                    for (int cr = 0; cr < COL_REG; ++cr) {
                        const int c = c_base + cr;
                        if (c < phase1_cols) {
                            const int pidx = m1 + c * n_pw;
                            p[cr] = complx_load(pw_in_d, pidx);
                        } else {
                            p[cr] = complx_zero<Real>();
                        }
                    }

#pragma unroll
                    for (int kr = 0; kr < K1_TILE; ++kr) {
                        const int k1 = k1_base + kr;
                        complx<Real> a1 = complx_zero<Real>();
                        if (k1 < n_order)
                            a1 = s_A_T[m1 * k_pad + k1];

#pragma unroll
                        for (int cr = 0; cr < COL_REG; ++cr)
                            complx_madd(acc[kr][cr], a1, p[cr]);
                    }
                }

#pragma unroll
                for (int cr = 0; cr < COL_REG; ++cr) {
                    const int c = c_base + cr;
                    if (c < phase1_cols) {
#pragma unroll
                        for (int kr = 0; kr < K1_TILE; ++kr) {
                            const int k1 = k1_base + kr;
                            if (k1 < n_order)
                                s_F[c * K1_TILE + kr] = acc[kr][cr];
                        }
                    }
                }
            }

            __syncthreads();

            // Phase 2: s_G(m3, k2, kr) = sum_m2 s_F(m2, m3, kr) * pw2poly(k2, m2).
            const int phase2_tiles = n_pw2 * k2_tiles * kr_tiles;

            for (int tile = threadIdx.x; tile < phase2_tiles; tile += blockDim.x) {
                int x = tile;
                const int kr_tile = x % kr_tiles;
                x /= kr_tiles;
                const int ktile = x % k2_tiles;
                const int m3 = x / k2_tiles;

                const int kr_base = kr_tile * KR_TILE;
                const int k2_base = ktile * K2_TILE;

                complx<Real> acc[KR_TILE][K2_TILE];

#pragma unroll
                for (int rr = 0; rr < KR_TILE; ++rr) {
#pragma unroll
                    for (int k2r = 0; k2r < K2_TILE; ++k2r)
                        acc[rr][k2r] = complx_zero<Real>();
                }

                for (int m2 = 0; m2 < n_pw; ++m2) {
                    const int c = m2 + m3 * n_pw;

                    complx<Real> f[KR_TILE];

#pragma unroll
                    for (int rr = 0; rr < KR_TILE; ++rr) {
                        const int kr = kr_base + rr;
                        const int k1 = k1_base + kr;
                        if (kr < K1_TILE && k1 < n_order)
                            f[rr] = s_F[c * K1_TILE + kr];
                        else
                            f[rr] = complx_zero<Real>();
                    }

                    complx<Real> a2[K2_TILE];

#pragma unroll
                    for (int k2r = 0; k2r < K2_TILE; ++k2r) {
                        const int k2 = k2_base + k2r;
                        if (k2 < n_order)
                            a2[k2r] = s_A_T[m2 * k_pad + k2];
                        else
                            a2[k2r] = complx_zero<Real>();
                    }

#pragma unroll
                    for (int rr = 0; rr < KR_TILE; ++rr) {
#pragma unroll
                        for (int k2r = 0; k2r < K2_TILE; ++k2r)
                            complx_madd(acc[rr][k2r], a2[k2r], f[rr]);
                    }
                }

                const Real scale = (m3 >= n_pw_half) ? Real{0.5} : Real{1};

#pragma unroll
                for (int k2r = 0; k2r < K2_TILE; ++k2r) {
                    const int k2 = k2_base + k2r;
                    if (k2 < n_order) {
#pragma unroll
                        for (int rr = 0; rr < KR_TILE; ++rr) {
                            const int kr = kr_base + rr;
                            const int k1 = k1_base + kr;

                            if (kr < K1_TILE && k1 < n_order) {
                                complx<Real> v = acc[rr][k2r];
                                v.r *= scale;
                                v.i *= scale;
                                s_G[(m3 * n_order + k2) * K1_TILE + kr] = v;
                            }
                        }
                    }
                }
            }

            __syncthreads();

            // Phase 3: proxy(k1, k2, k3) += 2 * Re(sum_m3 s_G(m3, k2, k1) * pw2poly(k3, m3)).
            const int phase3_tiles = k3_tiles * n_order * K1_TILE;

            for (int tile = threadIdx.x; tile < phase3_tiles; tile += blockDim.x) {
                int x = tile;
                const int kr = x % K1_TILE;
                x /= K1_TILE;
                const int k2 = x % n_order;
                const int k3tile = x / n_order;

                const int k1 = k1_base + kr;
                const int k3_base = k3tile * K3_TILE;

                Real acc[K3_TILE];

#pragma unroll
                for (int k3r = 0; k3r < K3_TILE; ++k3r)
                    acc[k3r] = Real{0};

                for (int m3 = 0; m3 < n_pw2; ++m3) {
                    complx<Real> g = complx_zero<Real>();
                    if (k1 < n_order)
                        g = s_G[(m3 * n_order + k2) * K1_TILE + kr];

                    complx<Real> a3[K3_TILE];

#pragma unroll
                    for (int k3r = 0; k3r < K3_TILE; ++k3r) {
                        const int k3 = k3_base + k3r;
                        if (k3 < n_order && m3 < n_pw)
                            a3[k3r] = s_A_T[m3 * k_pad + k3];
                        else
                            a3[k3r] = complx_zero<Real>();
                    }

#pragma unroll
                    for (int k3r = 0; k3r < K3_TILE; ++k3r)
                        acc[k3r] = complx_real_madd(acc[k3r], g, a3[k3r]);
                }

                if (k1 < n_order) {
#pragma unroll
                    for (int k3r = 0; k3r < K3_TILE; ++k3r) {
                        const int k3 = k3_base + k3r;
                        if (k3 < n_order) {
                            Real *__restrict__ out = proxy_d + k1 + k2 * n_order + k3 * n_order2;
                            *out += Real{2} * acc[k3r];
                        }
                    }
                }
            }

            __syncthreads();
        }
    }
}
