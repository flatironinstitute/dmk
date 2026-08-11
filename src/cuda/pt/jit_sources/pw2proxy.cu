// V2 pw2proxy (batched multilevel): projects plane-wave modes back onto proxy
// coefficients (additive). The launcher prepends `using Real` + N_ORDER / N_PW
// / N_PW2 / N_CHARGE_DIM / COL_REG / K2_TILE / K3_TILE / KR_TILE / BLOCK_SIZE.
// One launch covers all levels via a device array of per-level args (n_args ==
// 1 for the windowed root). Accumulates into proxy_flat.

#include <dmk/cuda/pw2proxy_kernelargs.hpp>

using dmk::cuda::PwToProxyArgs;

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

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
    PtPwToProxyMultiLevelKernel(const PwToProxyArgs<Real> *__restrict__ args, int n_args, unsigned char *scratch,
                                long scratch_stride, int box_base) {
    // Only the global-scratch path splits a level across launches, so the offset costs the
    // shared path nothing.
    const int box_idx = SMEM_GLOBAL ? blockIdx.x + box_base : blockIdx.x;
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

    const bool assign = a.assign != 0;

    const int n_pw = N_PW;
    const int n_pw2 = N_PW2;
    const int n_pw_half = n_pw / 2;
    const int n_order = N_ORDER;
    const int n_order2 = n_order * n_order;
    const int n_order3 = n_order2 * n_order;
    const int n_pw_modes = n_pw * n_pw * n_pw2;

    // Live modes sit at the front of the slab, which keeps its full-cube stride. A dead mode
    // reads as zero, which is what its zero kernel FT would have produced.
    const int *__restrict__ pencil = a.pencil_slots;

    const int k_pad = ((n_order + 3) / 4) * 4;
    const int phase1_cols = n_pw * n_pw;

    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    // At the largest expansions the working set exceeds the device's per-block shared limit, and
    // the block runs against a private slice of a global buffer instead. SMEM_GLOBAL is baked per
    // module, so only one of these survives compilation and the shared path keeps its LDS loads.
    unsigned char *__restrict__ shared_raw =
        SMEM_GLOBAL ? scratch + (long(blockIdx.y) * gridDim.x + blockIdx.x) * scratch_stride : &dynamic_smem[0];

    // The pencil table is decoded once per block into shared, already unpacked. In the m3 loop it
    // was a *global* load whose result the data load's address depends on -- two dependent global
    // round-trips per mode, which is what put the long-scoreboard stalls on the multiply -- plus a
    // shift and a mask to unpack, recomputed on all 18 (charge dim, k3 tile) passes even though a
    // column's slot never changes. Unpacked here, the chain is one shared load deep and the shift
    // and mask are gone from the inner loop entirely.
    const int n_pencil = n_pw * n_pw2;
    int4 *__restrict__ s_pencil = reinterpret_cast<int4 *>(shared_raw);

    // A null table means cube order; writing that case out as an identity run keeps one code path
    // in the inner loop.
    const auto decode_pencil = [&](int p) -> int4 {
        int4 v;
        if (pencil) {
            const int2 pv = reinterpret_cast<const int2 *>(pencil)[p];
            v.x = pv.x;
            v.y = pv.y & 0xffff;
            v.z = pv.y >> 16;
        } else {
            v.x = (p % n_pw) * n_pw + (p / n_pw) * phase1_cols;
            v.y = 0;
            v.z = n_pw;
        }
        v.w = 0;
        return v;
    };

    // Shared when the table fits, otherwise decoded per use: at the largest expansions the table
    // is the difference between fitting in shared and not running at all, and it is a latency
    // optimization rather than a requirement.
    const auto pencil_at = [&](int p) -> int4 {
        if constexpr (PENCIL_SMEM)
            return s_pencil[p];
        else
            return decode_pencil(p);
    };

    complx<Real> *__restrict__ smem = reinterpret_cast<complx<Real> *>(s_pencil + (PENCIL_SMEM ? n_pencil : 0));

    complx<Real> *__restrict__ s_A_T = smem;
    complx<Real> *__restrict__ s_F = s_A_T + n_pw * k_pad;
    complx<Real> *__restrict__ s_G = s_F + phase1_cols * K3_TILE;

    for (int idx = threadIdx.x; idx < n_pw * k_pad; idx += blockDim.x) {
        const int m = idx / k_pad;
        const int k = idx - m * k_pad;

        complx<Real> z = complx_zero<Real>();
        if (k < n_order)
            z = complx_load(a.pw2poly, k * n_pw + m);

        s_A_T[idx] = z;
    }

    if constexpr (PENCIL_SMEM) {
        for (int p = threadIdx.x; p < n_pencil; p += blockDim.x)
            s_pencil[p] = decode_pencil(p);
    }

    __syncthreads();

    const Real *__restrict__ pw_in_box = a.pw_in_pool + box_idx * a.pw_in_stride;
    Real *__restrict__ proxy_box = a.proxy_flat + proxy_off;

    const int k2_tiles = (n_order + K2_TILE - 1) / K2_TILE;
    const int m1_tiles = (n_pw + KR_TILE - 1) / KR_TILE;
    const int k1_tiles = (n_order + K1_TILE - 1) / K1_TILE;

    for (int d = 0; d < a.n_charge_dim; ++d) {
        const Real *__restrict__ pw_in_d = pw_in_box + 2 * d * n_pw_modes;
        Real *__restrict__ proxy_d = proxy_box + d * n_order3;

        for (int k3_base = 0; k3_base < n_order; k3_base += K3_TILE) {
            const int k3_count = (k3_base + K3_TILE <= n_order) ? K3_TILE : (n_order - k3_base);

            // Phase 1: s_F(k3r, m1, m2) = sum_m3 halve(m3) * pw(m1, m2, m3) * pw2poly(k3, m3).
            //
            // m3 runs outermost, over an accumulator that spans every column round a thread
            // owns. The pw2poly column is block-uniform and column-independent, so hoisting it
            // above the round loop fetches it once per m3 instead of once per m3 per round.
            // The round count is uniform and compile-time, which is what lets the accumulator
            // and the column indices stay in registers; a round past the last column still
            // runs, it just contributes zero.
            constexpr int P1_ROUNDS = (phase1_cols + BLOCK_SIZE * COL_REG - 1) / (BLOCK_SIZE * COL_REG);

            complx<Real> acc[P1_ROUNDS][K3_TILE][COL_REG];
            int col_m1[P1_ROUNDS][COL_REG], col_m2[P1_ROUNDS][COL_REG];
#pragma unroll
            for (int rd = 0; rd < P1_ROUNDS; ++rd) {
#pragma unroll
                for (int cr = 0; cr < COL_REG; ++cr) {
                    const int xy = threadIdx.x + rd * (BLOCK_SIZE * COL_REG) + cr * BLOCK_SIZE;
                    col_m1[rd][cr] = xy % n_pw;
                    // Clamped so a dead round cannot walk the pencil table off its end; its
                    // column is masked out below regardless of which entry it read.
                    col_m2[rd][cr] = min(xy / n_pw, n_pw - 1);
#pragma unroll
                    for (int k3r = 0; k3r < K3_TILE; ++k3r)
                        acc[rd][k3r][cr] = complx_zero<Real>();
                }
            }

            for (int m3 = 0; m3 < n_pw2; ++m3) {
                const Real scale = (m3 >= n_pw_half) ? Real{0.5} : Real{1};

                complx<Real> a3[K3_TILE];
#pragma unroll
                for (int k3r = 0; k3r < K3_TILE; ++k3r) {
                    const int k3 = k3_base + k3r;
                    a3[k3r] = (k3 < n_order) ? s_A_T[m3 * k_pad + k3] : complx_zero<Real>();
                }

#pragma unroll
                for (int rd = 0; rd < P1_ROUNDS; ++rd) {
#pragma unroll
                    for (int cr = 0; cr < COL_REG; ++cr) {
                        const int xy = threadIdx.x + rd * (BLOCK_SIZE * COL_REG) + cr * BLOCK_SIZE;
                        const int4 pv = pencil_at(col_m2[rd][cr] + m3 * n_pw);
                        const int m1 = col_m1[rd][cr];
                        const int slot = (xy < phase1_cols && m1 >= pv.y && m1 < pv.z) ? pv.x + m1 : -1;

                        complx<Real> p = complx_zero<Real>();
                        if (slot >= 0) {
                            p = complx_load(pw_in_d, slot);
                            p.r *= scale;
                            p.i *= scale;
                        }

#pragma unroll
                        for (int k3r = 0; k3r < K3_TILE; ++k3r)
                            complx_madd(acc[rd][k3r][cr], a3[k3r], p);
                    }
                }
            }

#pragma unroll
            for (int rd = 0; rd < P1_ROUNDS; ++rd) {
#pragma unroll
                for (int k3r = 0; k3r < K3_TILE; ++k3r) {
                    if (k3r < k3_count) {
#pragma unroll
                        for (int cr = 0; cr < COL_REG; ++cr) {
                            const int xy = threadIdx.x + rd * (BLOCK_SIZE * COL_REG) + cr * BLOCK_SIZE;
                            if (xy < phase1_cols)
                                s_F[k3r * phase1_cols + xy] = acc[rd][k3r][cr];
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
            const int phase3_tiles = k3_count * k2_tiles * k1_tiles;

            for (int tile = threadIdx.x; tile < phase3_tiles; tile += blockDim.x) {
                int x = tile;
                const int k1_tile = x % k1_tiles;
                x /= k1_tiles;
                const int ktile = x % k2_tiles;
                const int k3r = x / k2_tiles;
                const int k1_base = k1_tile * K1_TILE;
                const int k2_base = ktile * K2_TILE;
                const int k3 = k3_base + k3r;

                bool k1_ok[K1_TILE];
#pragma unroll
                for (int rr = 0; rr < K1_TILE; ++rr)
                    k1_ok[rr] = (k1_base + rr) < n_order;

                Real acc[K1_TILE][K2_TILE];
#pragma unroll
                for (int rr = 0; rr < K1_TILE; ++rr) {
#pragma unroll
                    for (int k2r = 0; k2r < K2_TILE; ++k2r)
                        acc[rr][k2r] = Real{0};
                }

                for (int m1 = 0; m1 < n_pw; ++m1) {
                    // K1_TILE consecutive k1 share every g, so widening the tile trades shared
                    // loads for registers in the phase that had by far the worst ratio of them.
                    complx<Real> a1[K1_TILE];
#pragma unroll
                    for (int rr = 0; rr < K1_TILE; ++rr)
                        a1[rr] = k1_ok[rr] ? s_A_T[m1 * k_pad + k1_base + rr] : complx_zero<Real>();

#pragma unroll
                    for (int k2r = 0; k2r < K2_TILE; ++k2r) {
                        const int k2 = k2_base + k2r;
                        if (k2 < n_order) {
                            const complx<Real> g = s_G[(k3r * n_order + k2) * n_pw + m1];
#pragma unroll
                            for (int rr = 0; rr < K1_TILE; ++rr)
                                acc[rr][k2r] = complx_real_madd(acc[rr][k2r], g, a1[rr]);
                        }
                    }
                }

#pragma unroll
                for (int k2r = 0; k2r < K2_TILE; ++k2r) {
                    const int k2 = k2_base + k2r;
                    if (k2 < n_order) {
#pragma unroll
                        for (int rr = 0; rr < K1_TILE; ++rr) {
                            if (k1_ok[rr]) {
                                Real *__restrict__ out = proxy_d + k1_base + rr + k2 * n_order + k3 * n_order2;
                                if (assign)
                                    *out = Real{2} * acc[rr][k2r];
                                else
                                    *out += Real{2} * acc[rr][k2r];
                            }
                        }
                    }
                }
            }

            __syncthreads();
        }
    }
}
