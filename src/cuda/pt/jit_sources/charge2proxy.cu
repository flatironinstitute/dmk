// V2 upward charge2proxy device kernel. The launcher prepends `using Real`
// and the N_ORDER / N_CHARGE_DIM / CHUNK / I_TILE / J_TILE / K_TILE constants.
// Additive write into proxy_flat (source boxes -> center box proxy coeffs).

#include <dmk/cuda/charge2proxy_kernelargs.hpp>

template <typename Real>
struct alignas(2 * sizeof(Real)) c2p_real2 {
    Real lo;
    Real hi;
};

template <typename Real>
__device__ __forceinline__ c2p_real2<Real> c2p_load2(const Real *__restrict__ p) {
    return reinterpret_cast<const c2p_real2<Real> *>(p)[0];
}

// The x/y/z basis factors depend only on the spatial tile, so the NC charge components share
// one set of loads and only the charge itself varies with `d`: IT+JT+KT+NC shared loads per
// source pair for 2*NC*IT*JT*KT FMAs.
template <typename Real, int NC, int IT, int JT, int KT, int N, int LD>
__device__ __forceinline__ void
charge2proxy_accum_source_pair(Real (&acc)[NC][KT][IT][JT], const Real *__restrict__ poly_x,
                               const Real *__restrict__ poly_y, const Real *__restrict__ poly_z,
                               const Real *__restrict__ charges_s, int i0, int j0, int k0, int s) {
    c2p_real2<Real> xreg[IT] = {};
    c2p_real2<Real> yreg[JT] = {};
    c2p_real2<Real> zreg[KT] = {};

#pragma unroll
    for (int r = 0; r < IT; ++r) {
        const int i = i0 + r;
        if (i < N)
            xreg[r] = c2p_load2<Real>(poly_x + i * LD + s);
    }

#pragma unroll
    for (int c = 0; c < JT; ++c) {
        const int j = j0 + c;
        if (j < N)
            yreg[c] = c2p_load2<Real>(poly_y + j * LD + s);
    }

#pragma unroll
    for (int kk = 0; kk < KT; ++kk) {
        const int k = k0 + kk;
        if (k < N)
            zreg[kk] = c2p_load2<Real>(poly_z + k * LD + s);
    }

#pragma unroll
    for (int d = 0; d < NC; ++d) {
        const c2p_real2<Real> q = c2p_load2<Real>(charges_s + d * LD + s);

#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
            const Real zq0 = zreg[kk].lo * q.lo;
            const Real zq1 = zreg[kk].hi * q.hi;

#pragma unroll
            for (int c = 0; c < JT; ++c) {
                const Real yzq0 = yreg[c].lo * zq0;
                const Real yzq1 = yreg[c].hi * zq1;

#pragma unroll
                for (int r = 0; r < IT; ++r) {
                    Real a = acc[d][kk][r][c];
                    a = fma(xreg[r].lo, yzq0, a);
                    a = fma(xreg[r].hi, yzq1, a);
                    acc[d][kk][r][c] = a;
                }
            }
        }
    }
}

template <typename Real, int NC, int IT, int JT, int KT, int N, int LD>
__device__ __forceinline__ void
charge2proxy_accum_source_scalar(Real (&acc)[NC][KT][IT][JT], const Real *__restrict__ poly_x,
                                 const Real *__restrict__ poly_y, const Real *__restrict__ poly_z,
                                 const Real *__restrict__ charges_s, int i0, int j0, int k0, int s) {
    Real xreg[IT] = {Real{0}};
    Real yreg[JT] = {Real{0}};
    Real zreg[KT] = {Real{0}};

#pragma unroll
    for (int r = 0; r < IT; ++r) {
        const int i = i0 + r;
        if (i < N)
            xreg[r] = poly_x[i * LD + s];
    }

#pragma unroll
    for (int c = 0; c < JT; ++c) {
        const int j = j0 + c;
        if (j < N)
            yreg[c] = poly_y[j * LD + s];
    }

#pragma unroll
    for (int kk = 0; kk < KT; ++kk) {
        const int k = k0 + kk;
        if (k < N)
            zreg[kk] = poly_z[k * LD + s];
    }

#pragma unroll
    for (int d = 0; d < NC; ++d) {
        const Real q = charges_s[d * LD + s];

#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
            const Real zq = zreg[kk] * q;

#pragma unroll
            for (int c = 0; c < JT; ++c) {
                const Real yzq = yreg[c] * zq;

#pragma unroll
                for (int r = 0; r < IT; ++r) {
                    acc[d][kk][r][c] = fma(xreg[r], yzq, acc[d][kk][r][c]);
                }
            }
        }
    }
}

template <typename Real>
__device__ inline void chebyshev_fill_strided(Real x, Real *out, int n, int stride) {
    Real t0 = Real{1};
    out[0] = t0;
    if (n <= 1)
        return;
    Real t1 = x;
    out[stride] = t1;
    for (int k = 2; k < n; ++k) {
        const Real t2 = Real{2} * x * t1 - t0;
        out[k * stride] = t2;
        t0 = t1;
        t1 = t2;
    }
}

// KERNEL_START

extern "C" __global__ void PtCharge2ProxyKernel(dmk::cuda::Charge2ProxyArgs<Real> a,
                                                 const int *__restrict__ group_perm) {
    constexpr int S_VEC = 2;
    constexpr int LD = CHUNK + S_VEC;
    constexpr int DIM = 3;

    constexpr int N = N_ORDER;
    constexpr int NC = N_CHARGE_DIM;
    constexpr int N2 = N * N;
    constexpr int N3 = N2 * N;
    constexpr int CHUNK_PAIR_END = CHUNK - (CHUNK % S_VEC);

    constexpr int I_TILES = (N + I_TILE - 1) / I_TILE;
    constexpr int J_TILES = (N + J_TILE - 1) / J_TILE;
    constexpr int K_TILES = (N + K_TILE - 1) / K_TILE;

    constexpr int S_TILES = I_TILES * J_TILES * K_TILES;

    extern __shared__ __align__(16) unsigned char shared_raw[];

    Real *poly_x = reinterpret_cast<Real *>(shared_raw);
    Real *poly_y = poly_x + N * LD;
    Real *poly_z = poly_y + N * LD;
    Real *charges_s = poly_z + N * LD;

    const int logical_g = blockIdx.x;
    if (logical_g >= a.n_groups)
        return;

    const int g = group_perm ? group_perm[logical_g] : logical_g;

    const int center_box = a.center_boxes[g];
    const int level = a.levels[g];
    const int sb_off = a.src_box_flat_offsets[g];
    const int n_src_boxes = a.n_src_boxes_per_group[g];

    const Real cx = a.centers[center_box * DIM + 0];
    const Real cy = a.centers[center_box * DIM + 1];
    const Real cz = a.centers[center_box * DIM + 2];

    const Real scale = a.inv_box_scale[level];

    Real *__restrict__ proxy = a.proxy_flat + a.proxy_offsets[center_box];

    // Tile rounds enclose the source loop, so an accumulator survives every chunk of every
    // source box and a coefficient reaches global memory once, as a plain store. The buffers
    // are re-staged per round, which is what that costs.
    //
    // The round count is uniform and compile-time, rather than the tile index being the loop
    // variable, so every thread reaches the barriers below: one past the last tile still
    // stages and still syncs, it just accumulates nothing.
    constexpr int TILE_ROUNDS = (S_TILES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int round = 0; round < TILE_ROUNDS; ++round) {
        const int tile = round * BLOCK_SIZE + threadIdx.x;
        const bool has_tile = tile < S_TILES;

        // The basis is radially compact to tolerance, so coefficients outside a ball in
        // (i,j,k) carry nothing. kTileOrder puts the live tiles first, so this test is uniform
        // across a warp; a dead tile skips the source accumulation but still stores, because
        // the buffer has no memset and relies on every coefficient having a writer that assigns.
        const bool tile_live = has_tile && tile < N_LIVE_SPATIAL_TILES;

        int idx = has_tile ? kTileOrder[tile] : 0;
        const int it = idx % I_TILES;
        idx /= I_TILES;
        const int jt = idx % J_TILES;
        idx /= J_TILES;
        const int kt = idx;

        const int i0 = it * I_TILE;
        const int j0 = jt * J_TILE;
        const int k0 = kt * K_TILE;

        Real acc[NC][K_TILE][I_TILE][J_TILE] = {};

        // Every bound here is group data, never the thread, so all threads run the same number
        // of iterations and reach the barrier below the same number of times.
        for (int sbi = 0; sbi < n_src_boxes; ++sbi) {
            const int sb = a.src_boxes_flat[sb_off + sbi];
            const int n_src = a.src_counts[sb];

            if (n_src == 0)
                continue;

            const Real *__restrict__ r_src = a.r_src + a.r_src_offsets[sb];
            const Real *__restrict__ charge = a.charge + a.charge_offsets[sb];

            for (int s_base = 0; s_base < n_src; s_base += CHUNK) {
                const int n_in_chunk = (s_base + CHUNK > n_src) ? (n_src - s_base) : CHUNK;

                for (int s = threadIdx.x; s < n_in_chunk; s += blockDim.x) {
                    const int sp = s_base + s;
                    const Real x = (r_src[sp * DIM + 0] - cx) * scale;
                    const Real y = (r_src[sp * DIM + 1] - cy) * scale;
                    const Real z = (r_src[sp * DIM + 2] - cz) * scale;
                    chebyshev_fill_strided<Real>(x, poly_x + s, N, LD);
                    chebyshev_fill_strided<Real>(y, poly_y + s, N, LD);
                    chebyshev_fill_strided<Real>(z, poly_z + s, N, LD);
                }

                for (int t = threadIdx.x; t < NC * n_in_chunk; t += blockDim.x) {
                    const int s = t / NC;
                    const int d = t - s * NC;
                    const int sp = s_base + s;
                    charges_s[d * LD + s] = charge[d + sp * NC];
                }

                __syncthreads();

                if (tile_live) {
                    if (n_in_chunk == CHUNK) {
#pragma unroll SRC_UNROLL
                        for (int s = 0; s < CHUNK_PAIR_END; s += S_VEC) {
                            charge2proxy_accum_source_pair<Real, NC, I_TILE, J_TILE, K_TILE, N, LD>(
                                acc, poly_x, poly_y, poly_z, charges_s, i0, j0, k0, s);
                        }
                        if constexpr ((CHUNK % S_VEC) != 0) {
                            charge2proxy_accum_source_scalar<Real, NC, I_TILE, J_TILE, K_TILE, N, LD>(
                                acc, poly_x, poly_y, poly_z, charges_s, i0, j0, k0, CHUNK_PAIR_END);
                        }
                    } else {
                        int s = 0;
                        for (; s + 1 < n_in_chunk; s += S_VEC) {
                            charge2proxy_accum_source_pair<Real, NC, I_TILE, J_TILE, K_TILE, N, LD>(
                                acc, poly_x, poly_y, poly_z, charges_s, i0, j0, k0, s);
                        }
                        if (s < n_in_chunk) {
                            charge2proxy_accum_source_scalar<Real, NC, I_TILE, J_TILE, K_TILE, N, LD>(
                                acc, poly_x, poly_y, poly_z, charges_s, i0, j0, k0, s);
                        }
                    }
                }

                // Guards the staged chunk against the next chunk's overwrite, and -- because it
                // is the last shared access of the round -- against the next round's staging too.
                __syncthreads();
            }
        }

        // Exactly one writer per coefficient, so this assigns and the buffer needs no memset.
        // A dead tile stores its zero accumulator for the same reason.
        if (has_tile) {
#pragma unroll
            for (int d = 0; d < NC; ++d) {
#pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    const int k = k0 + kk;
                    if (k < N) {
#pragma unroll
                        for (int r = 0; r < I_TILE; ++r) {
                            const int i = i0 + r;
                            if (i < N) {
#pragma unroll
                                for (int c = 0; c < J_TILE; ++c) {
                                    const int j = j0 + c;
                                    if (j < N)
                                        proxy[i + j * N + k * N2 + d * N3] = acc[d][kk][r][c];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
