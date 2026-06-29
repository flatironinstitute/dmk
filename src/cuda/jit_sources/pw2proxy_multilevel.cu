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
    const int phase1_cols = n_pw * n_pw;

    extern __shared__ __align__(16) unsigned char shared_raw[];

    complx<Real> *__restrict__ smem = reinterpret_cast<complx<Real> *>(shared_raw);

    // s_A_T[m, k] = pw2poly(k, m), padded in k for bank/layout control.
    complx<Real> *__restrict__ s_A_T = smem;

    // s_F[k3r, xy], xy = m1 + m2 * n_pw, so phase-1 global loads are coalesced.
    complx<Real> *__restrict__ s_F = s_A_T + n_pw * k_pad;

    // s_G[k3r, k2, m1], local m1 fastest for the final contraction.
    complx<Real> *__restrict__ s_G = s_F + phase1_cols * K3_TILE;

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
    const int m1_tiles = (n_pw + KR_TILE - 1) / KR_TILE;

    for (int d = 0; d < a.n_charge_dim; ++d) {
        const Real *__restrict__ pw_in_d = pw_in_box + 2 * d * n_pw_modes;
        Real *__restrict__ proxy_d = proxy_box + d * n_order3;

        for (int k3_base = 0; k3_base < n_order; k3_base += K3_TILE) {
            const int k3_count = (k3_base + K3_TILE <= n_order) ? K3_TILE : (n_order - k3_base);

            // Phase 1: s_F(k3r, m1, m2) = sum_m3 halve(m3) * pw(m1, m2, m3) * pw2poly(k3, m3).
            for (int xy_base = threadIdx.x; xy_base < phase1_cols; xy_base += blockDim.x * COL_REG) {
                complx<Real> acc[K3_TILE][COL_REG];

#pragma unroll
                for (int k3r = 0; k3r < K3_TILE; ++k3r) {
#pragma unroll
                    for (int cr = 0; cr < COL_REG; ++cr)
                        acc[k3r][cr] = complx_zero<Real>();
                }

                for (int m3 = 0; m3 < n_pw2; ++m3) {
                    complx<Real> p[COL_REG];
                    const Real scale = (m3 >= n_pw_half) ? Real{0.5} : Real{1};

#pragma unroll
                    for (int cr = 0; cr < COL_REG; ++cr) {
                        const int xy = xy_base + cr * blockDim.x;
                        if (xy < phase1_cols) {
                            p[cr] = complx_load(pw_in_d, xy + m3 * phase1_cols);
                            p[cr].r *= scale;
                            p[cr].i *= scale;
                        } else {
                            p[cr] = complx_zero<Real>();
                        }
                    }

#pragma unroll
                    for (int k3r = 0; k3r < K3_TILE; ++k3r) {
                        const int k3 = k3_base + k3r;
                        complx<Real> a3 = complx_zero<Real>();
                        if (k3 < n_order)
                            a3 = s_A_T[m3 * k_pad + k3];

#pragma unroll
                        for (int cr = 0; cr < COL_REG; ++cr)
                            complx_madd(acc[k3r][cr], a3, p[cr]);
                    }
                }

#pragma unroll
                for (int k3r = 0; k3r < K3_TILE; ++k3r) {
                    if (k3r < k3_count) {
#pragma unroll
                        for (int cr = 0; cr < COL_REG; ++cr) {
                            const int xy = xy_base + cr * blockDim.x;
                            if (xy < phase1_cols)
                                s_F[k3r * phase1_cols + xy] = acc[k3r][cr];
                        }
                    }
                }
            }

            __syncthreads();

            // Phase 2: s_G(k3r, k2, m1) = sum_m2 s_F(k3r, m1, m2) * pw2poly(k2, m2).
            const int phase2_tiles = k3_count * m1_tiles * k2_tiles;

            for (int tile = threadIdx.x; tile < phase2_tiles; tile += blockDim.x) {
                int x = tile;
                const int ktile = x % k2_tiles;
                x /= k2_tiles;
                const int m1_tile = x % m1_tiles;
                const int k3r = x / m1_tiles;

                const int m1_base = m1_tile * KR_TILE;
                const int k2_base = ktile * K2_TILE;

                complx<Real> acc[KR_TILE][K2_TILE];

#pragma unroll
                for (int rr = 0; rr < KR_TILE; ++rr) {
#pragma unroll
                    for (int k2r = 0; k2r < K2_TILE; ++k2r)
                        acc[rr][k2r] = complx_zero<Real>();
                }

                for (int m2 = 0; m2 < n_pw; ++m2) {
                    complx<Real> f[KR_TILE];

#pragma unroll
                    for (int rr = 0; rr < KR_TILE; ++rr) {
                        const int m1 = m1_base + rr;
                        if (m1 < n_pw)
                            f[rr] = s_F[k3r * phase1_cols + m1 + m2 * n_pw];
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

#pragma unroll
                for (int k2r = 0; k2r < K2_TILE; ++k2r) {
                    const int k2 = k2_base + k2r;
                    if (k2 < n_order) {
#pragma unroll
                        for (int rr = 0; rr < KR_TILE; ++rr) {
                            const int m1 = m1_base + rr;
                            if (m1 < n_pw)
                                s_G[(k3r * n_order + k2) * n_pw + m1] = acc[rr][k2r];
                        }
                    }
                }
            }

            __syncthreads();

            // Phase 3: proxy(k1, k2, k3) += 2 * Re(sum_m1 s_G(k3r, k2, m1) * pw2poly(k1, m1)).
            const int phase3_tiles = k3_count * k2_tiles * n_order;

            for (int tile = threadIdx.x; tile < phase3_tiles; tile += blockDim.x) {
                int x = tile;
                const int k1 = x % n_order;
                x /= n_order;
                const int ktile = x % k2_tiles;
                const int k3r = x / k2_tiles;
                const int k2_base = ktile * K2_TILE;
                const int k3 = k3_base + k3r;

                Real acc[K2_TILE];

#pragma unroll
                for (int k2r = 0; k2r < K2_TILE; ++k2r)
                    acc[k2r] = Real{0};

                for (int m1 = 0; m1 < n_pw; ++m1) {
                    const complx<Real> a1 = s_A_T[m1 * k_pad + k1];

#pragma unroll
                    for (int k2r = 0; k2r < K2_TILE; ++k2r) {
                        const int k2 = k2_base + k2r;
                        if (k2 < n_order) {
                            const complx<Real> g = s_G[(k3r * n_order + k2) * n_pw + m1];
                            acc[k2r] = complx_real_madd(acc[k2r], g, a1);
                        }
                    }
                }

#pragma unroll
                for (int k2r = 0; k2r < K2_TILE; ++k2r) {
                    const int k2 = k2_base + k2r;
                    if (k2 < n_order) {
                        Real *__restrict__ out = proxy_d + k1 + k2 * n_order + k3 * n_order2;
                        *out += Real{2} * acc[k2r];
                    }
                }
            }

            __syncthreads();
        }
    }
}
