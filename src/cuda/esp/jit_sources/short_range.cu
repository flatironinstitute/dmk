// ESP short-range direct sum. The launcher prepends a prelude defining Real, the
// DMK_ESP_SR_KERNEL_NAME symbol, the baked-literal Coeffs struct, and the BLOCK_SIZE / TILE_WIDTH /
// TARGETS_PER_THREAD / STRATEGY / WANT_FORCE / PRUNE_STATS constants. Coefficients are compile-time
// literals folded into the FMAs -- no runtime coeff buffer.
//
// STRATEGY mirrors GpuSrStrategy: 0 = Dense, 1 = PruneTile, 2 = PruneSource.

#include <dmk/cuda/esp_sr_kernelargs.hpp>

using EspSrArgs = dmk::cuda::EspSrArgs<Real>;

template <int I>
__device__ constexpr Real horner_recurse(Real x, Real acc) {
    if constexpr (I == 0)
        return acc;
    else
        return horner_recurse<I - 1>(x, acc * x + Real(Coeffs::at(I - 1)));
}

template <int UNUSED = 0>
__device__ constexpr Real horner_const(Real x) {
    static_assert(Coeffs::size > 0, "empty coefficient pack");
    return horner_recurse<Coeffs::size - 1>(x, Real(Coeffs::at(Coeffs::size - 1)));
}

// P and dP/dx together, by synthetic division. Coefficients are ascending-order.
template <int I>
__device__ constexpr void horner_recurse_deriv(Real x, Real &P, Real &dP) {
    if constexpr (I == 0) {
        return;
    } else {
        Real old_P = P;
        P = P * x + Real(Coeffs::at(I - 1));
        dP = dP * x + old_P;
        return horner_recurse_deriv<I - 1>(x, P, dP);
    }
}

template <int UNUSED = 0>
__device__ constexpr void horner_const_deriv(Real x, Real &P, Real &dP) {
    static_assert(Coeffs::size > 0, "empty coefficient pack");
    P = Real(Coeffs::at(Coeffs::size - 1));
    dP = Real{0};
    horner_recurse_deriv<Coeffs::size - 1>(x, P, dP);
}

// The short-range math, for one source-target pair.
template <bool WantForce>
__device__ __forceinline__ void eval_esp_pair(
    Real dx, Real dy, Real dz, Real q,
    Real rsc, Real cen, Real r_c_sq,
    Real &pot_acc, Real &gx_acc, Real &gy_acc, Real &gz_acc)
{
    const Real R2 = dx * dx + dy * dy + dz * dz;
    if (R2 <= Real(0) || R2 >= r_c_sq) return; // also masks the R2=0 self-pair

    const Real Rinv = rsqrt(R2); // CUDA's rsqrt()/__drsqrt_rn(), full IEEE precision
    const Real x = (R2 * Rinv + cen) * rsc; // = (R + cen)*rsc, mapped into [-1,1]

    if constexpr (WantForce) {
        Real P, dP;
        horner_const_deriv<>(x, P, dP);
        pot_acc += q * P * Rinv;
        const Real df_dR2 = Rinv * Rinv * (dP * rsc - P * Rinv);
        gx_acc += q * dx * df_dR2;
        gy_acc += q * dy * df_dR2;
        gz_acc += q * dz * df_dR2;
    } else {
        const Real P = horner_const<>(x);
        pot_acc += q * P * Rinv;
    }
}

// Dense: one block per home cell, each thread register-blocking TARGETS targets. Slot q of thread i
// is target t_base + q*blockDim.x -- strided so consecutive threads hold consecutive targets, which
// keeps position loads and result writes coalesced. The 27 neighbour cells are staged through shared
// memory in block-sized tiles so each source is read from global once per block, not once per target.

template <bool WantForce, int TARGETS>
__device__ void short_range_dense_body(const EspSrArgs &a)
{
    const int nc = a.nc;
    const int out_dim = a.out_dim;
    const Real rsc = a.rsc, cen = a.cen, r_c_sq = a.r_c_sq;
    const int *__restrict__ cell_start = a.cell_start;
    const Real *__restrict__ d_xs = a.xs;
    const Real *__restrict__ d_ys = a.ys;
    const Real *__restrict__ d_zs = a.zs;
    const Real *__restrict__ d_qs = a.qs;
    const int *__restrict__ nbc_tab = a.nbc_tab;
    const Real *__restrict__ off_tab = a.off_tab;
    Real *__restrict__ pg_sorted = a.pg_sorted;

    __shared__ Real s_xs[BLOCK_SIZE];
    __shared__ Real s_ys[BLOCK_SIZE];
    __shared__ Real s_zs[BLOCK_SIZE];
    __shared__ Real s_qs[BLOCK_SIZE];

    const int home = blockIdx.x; // 0 .. nc^3-1, row-major (x*nc+y)*nc+z, one block per cell
    const int cx = home / (nc * nc);
    const int cy = (home / nc) % nc;
    const int cz = home % nc;

    const int hbeg = cell_start[home];
    const int n_trg = cell_start[home + 1] - hbeg;
    const int target_stride = blockDim.x * TARGETS;
    const int n_rounds = (n_trg + target_stride - 1) / target_stride;

    for (int round = 0; round < n_rounds; ++round) {
        const int t_base = round * target_stride + threadIdx.x;

        bool active[TARGETS]; //which of this thread's TARGETS target slots actually correspond to a real particle in this round
        int trg_idx[TARGETS];
        bool any_active = false;
        Real xt[TARGETS], yt[TARGETS], zt[TARGETS];
        Real pot_acc[TARGETS] = {}, gx_acc[TARGETS] = {}, gy_acc[TARGETS] = {}, gz_acc[TARGETS] = {};
#pragma unroll
        for (int q = 0; q < TARGETS; ++q) {
            const int t = t_base + q * blockDim.x;
            active[q] = t < n_trg;
            trg_idx[q] = t;
            any_active = any_active || active[q];
            if (active[q]) {
                const int trg = hbeg + t;
                xt[q] = d_xs[trg]; yt[q] = d_ys[trg]; zt[q] = d_zs[trg];
            }
        }

        for (int dxi = 0; dxi < 3; ++dxi) {
            const int nbx = nbc_tab[cx * 3 + dxi];
            const Real ox = off_tab[cx * 3 + dxi];
            for (int dyi = 0; dyi < 3; ++dyi) {
                const int nby = nbc_tab[cy * 3 + dyi];
                const Real oy = off_tab[cy * 3 + dyi];
                for (int dzi = 0; dzi < 3; ++dzi) {
                    const int nbz = nbc_tab[cz * 3 + dzi];
                    const Real oz = off_tab[cz * 3 + dzi];
                    const int nb = (nbx * nc + nby) * nc + nbz;
                    const int sbeg = cell_start[nb], send = cell_start[nb + 1];
                    const int n_tiles = (send - sbeg + BLOCK_SIZE - 1) / BLOCK_SIZE;

                    for (int tile = 0; tile < n_tiles; ++tile) {
                        const int s_idx = sbeg + tile * BLOCK_SIZE + threadIdx.x;
                        if (s_idx < send) {
                            s_xs[threadIdx.x] = d_xs[s_idx] + ox;
                            s_ys[threadIdx.x] = d_ys[s_idx] + oy;
                            s_zs[threadIdx.x] = d_zs[s_idx] + oz;
                            s_qs[threadIdx.x] = d_qs[s_idx];
                        }
                        __syncthreads();

                        const int n_local = min(BLOCK_SIZE, send - sbeg - tile * BLOCK_SIZE);

                        if (any_active) {
#pragma unroll
                            for (int k = 0; k < TARGETS; ++k) {
                                if (!active[k]) continue;
                                for (int s = 0; s < n_local; ++s) {
                                    eval_esp_pair<WantForce>(xt[k] - s_xs[s], yt[k] - s_ys[s], zt[k] - s_zs[s], s_qs[s], rsc, cen, r_c_sq, pot_acc[k], gx_acc[k], gy_acc[k], gz_acc[k]);
                                }
                            }
                        }
                        __syncthreads(); // all threads done reading this tile before it's overwritten
                    }
                }
            }
        }

#pragma unroll
        for (int k = 0; k < TARGETS; ++k) {
            if (!active[k]) continue;
            const int trg = hbeg + trg_idx[k];
            pg_sorted[out_dim * trg + 0] = pot_acc[k];
            if constexpr (WantForce) {
                pg_sorted[out_dim * trg + 1] = gx_acc[k];
                pg_sorted[out_dim * trg + 2] = gy_acc[k];
                pg_sorted[out_dim * trg + 3] = gz_acc[k];
            }
        }
    }
}

template <typename Real>
__device__ __forceinline__ Real warp_reduce_min(Real v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = min(v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}
template <typename Real>
__device__ __forceinline__ Real warp_reduce_max(Real v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = max(v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}

template <bool WantForce>
__device__ void short_range_prune_tile_body(const EspSrArgs &a)
{
    const int nc = a.nc;
    const int out_dim = a.out_dim;
    const Real rsc = a.rsc, cen = a.cen, r_c_sq = a.r_c_sq;
    const int *__restrict__ cell_start = a.cell_start;
    const Real *__restrict__ d_xs = a.xs;
    const Real *__restrict__ d_ys = a.ys;
    const Real *__restrict__ d_zs = a.zs;
    const Real *__restrict__ d_qs = a.qs;
    const int *__restrict__ nbc_tab = a.nbc_tab;
    const Real *__restrict__ off_tab = a.off_tab;
    Real *__restrict__ pg_sorted = a.pg_sorted;
    const int max_tiles = a.max_tiles;

    // Per-tile AABB/shift/base/len table. max_tiles is a cached-with-margin bound, not a guarantee.
    extern __shared__ unsigned char s_raw[];
    Real *s_lo_x = reinterpret_cast<Real *>(s_raw);
    Real *s_lo_y = s_lo_x + max_tiles;
    Real *s_lo_z = s_lo_y + max_tiles;
    Real *s_hi_x = s_lo_z + max_tiles;
    Real *s_hi_y = s_hi_x + max_tiles;
    Real *s_hi_z = s_hi_y + max_tiles;
    Real *s_shift_x = s_hi_z + max_tiles;
    Real *s_shift_y = s_shift_x + max_tiles;
    Real *s_shift_z = s_shift_y + max_tiles;
    int  *s_base = reinterpret_cast<int *>(s_shift_z + max_tiles);
    int  *s_len  = s_base + max_tiles;

    // Per-neighbour-cell metadata (not per-tile); static shared, coexists with the dynamic block.
    __shared__ int  s_nb_lin[27];
    __shared__ int  s_nb_tile0[28]; // exclusive prefix sum of per-neighbor tile counts; [27] = total
    __shared__ Real s_nb_shift_x[27], s_nb_shift_y[27], s_nb_shift_z[27];

    const int home = blockIdx.x; // 0 .. nc^3-1, row-major (x*nc+y)*nc+z, one block per cell
    const int cx = home / (nc * nc);
    const int cy = (home / nc) % nc;
    const int cz = home % nc;

    const int hbeg = cell_start[home];
    const int n_trg = cell_start[home + 1] - hbeg;
    if (n_trg == 0) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane = tid % warpSize;
    const int n_warps = blockDim.x / warpSize;

    if (tid < 27) {
        const int dzi = tid % 3, dyi = (tid / 3) % 3, dxi = tid / 9;
        const int nbx = nbc_tab[cx * 3 + dxi];
        const int nby = nbc_tab[cy * 3 + dyi];
        const int nbz = nbc_tab[cz * 3 + dzi];
        const Real ox = off_tab[cx * 3 + dxi];
        const Real oy = off_tab[cy * 3 + dyi];
        const Real oz = off_tab[cz * 3 + dzi];
        const int nb = (nbx * nc + nby) * nc + nbz;
        s_nb_lin[tid] = nb;
        s_nb_shift_x[tid] = ox;
        s_nb_shift_y[tid] = oy;
        s_nb_shift_z[tid] = oz;
        const int len = cell_start[nb + 1] - cell_start[nb];
        s_nb_tile0[tid] = (len + TILE_WIDTH - 1) / TILE_WIDTH;
    }
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < 27; ++i) {
            const int c = s_nb_tile0[i];
            s_nb_tile0[i] = acc;
            acc += c;
        }
        s_nb_tile0[27] = acc;
    }
    __syncthreads();
    // Clamp: exceeding the margin truncates the tile list (slightly wrong) rather than overflowing
    // shared memory.
    const int n_stiles = min(s_nb_tile0[27], max_tiles);

    // Phase 1: one warp per source tile -- gather, AABB-reduce, store.
    for (int gt = warp_id; gt < n_stiles; gt += n_warps) {
        int nbi = 0;
        while (nbi < 26 && s_nb_tile0[nbi + 1] <= gt)
            ++nbi;
        const int local_tile = gt - s_nb_tile0[nbi];
        const int nb = s_nb_lin[nbi];
        const int nb_beg = cell_start[nb], nb_end = cell_start[nb + 1];
        const int t_base = nb_beg + local_tile * TILE_WIDTH;
        const int t_len = min(TILE_WIDTH, nb_end - t_base);
        const Real ox = s_nb_shift_x[nbi], oy = s_nb_shift_y[nbi], oz = s_nb_shift_z[nbi];

        const bool valid = lane < t_len;
        const Real x = valid ? d_xs[t_base + lane] + ox : Real(0);
        const Real y = valid ? d_ys[t_base + lane] + oy : Real(0);
        const Real z = valid ? d_zs[t_base + lane] + oz : Real(0);
        Real lo_x = warp_reduce_min(valid ? x : Real(INFINITY));
        Real hi_x = warp_reduce_max(valid ? x : Real(-INFINITY));
        Real lo_y = warp_reduce_min(valid ? y : Real(INFINITY));
        Real hi_y = warp_reduce_max(valid ? y : Real(-INFINITY));
        Real lo_z = warp_reduce_min(valid ? z : Real(INFINITY));
        Real hi_z = warp_reduce_max(valid ? z : Real(-INFINITY));

        if (lane == 0) {
            s_lo_x[gt] = lo_x; s_lo_y[gt] = lo_y; s_lo_z[gt] = lo_z;
            s_hi_x[gt] = hi_x; s_hi_y[gt] = hi_y; s_hi_z[gt] = hi_z;
            s_shift_x[gt] = ox; s_shift_y[gt] = oy; s_shift_z[gt] = oz;
            s_base[gt] = t_base; s_len[gt] = t_len;
        }
    }
    __syncthreads(); // every warp's Phase 2 reads the full shared tile table

    // Phase 2: each warp grid-strides over its own 32-target tiles.
    for (int t0 = warp_id * TILE_WIDTH; t0 < n_trg; t0 += n_warps * TILE_WIDTH) {
        const int tlen = min(TILE_WIDTH, n_trg - t0);
        const bool my_active = lane < tlen;
        const int trg = hbeg + t0 + lane;
        const Real xt = my_active ? d_xs[trg] : Real(0);
        const Real yt = my_active ? d_ys[trg] : Real(0);
        const Real zt = my_active ? d_zs[trg] : Real(0);

        const Real tlo_x = warp_reduce_min(my_active ? xt : Real(INFINITY));
        const Real thi_x = warp_reduce_max(my_active ? xt : Real(-INFINITY));
        const Real tlo_y = warp_reduce_min(my_active ? yt : Real(INFINITY));
        const Real thi_y = warp_reduce_max(my_active ? yt : Real(-INFINITY));
        const Real tlo_z = warp_reduce_min(my_active ? zt : Real(INFINITY));
        const Real thi_z = warp_reduce_max(my_active ? zt : Real(-INFINITY));

        Real pot_acc = Real(0), gx_acc = Real(0), gy_acc = Real(0), gz_acc = Real(0);

        for (int st = 0; st < n_stiles; ++st) {
            // Branchless squared box-distance, same formula as the CPU's short_range_prune_tile.
            const Real ddx = max(Real(0), max(s_lo_x[st] - thi_x, tlo_x - s_hi_x[st]));
            const Real ddy = max(Real(0), max(s_lo_y[st] - thi_y, tlo_y - s_hi_y[st]));
            const Real ddz = max(Real(0), max(s_lo_z[st] - thi_z, tlo_z - s_hi_z[st]));
            const bool pruned = ddx * ddx + ddy * ddy + ddz * ddz > r_c_sq;
            if (lane == 0) {
                if constexpr (PRUNE_STATS)
                    atomicAdd(&a.prune_stats[0], 1ull);
                if (!pruned)
                    if constexpr (PRUNE_STATS)
                        atomicAdd(&a.prune_stats[1], 1ull);
            }
            if (pruned)
                continue; // whole tile pruned: no source load, no eval_esp_pair calls

            const int s_base_i = s_base[st], s_len_i = s_len[st];
            const Real shx = s_shift_x[st], shy = s_shift_y[st], shz = s_shift_z[st];
            const bool have_src = lane < s_len_i;
            const Real sx = have_src ? d_xs[s_base_i + lane] + shx : Real(0);
            const Real sy = have_src ? d_ys[s_base_i + lane] + shy : Real(0);
            const Real sz = have_src ? d_zs[s_base_i + lane] + shz : Real(0);
            const Real sq = have_src ? d_qs[s_base_i + lane] : Real(0);

            // Each lane loads one source and broadcasts in turn: no shared staging, no redundant
            // per-lane global reads.
            for (int l = 0; l < s_len_i; ++l) {
                const Real bx = __shfl_sync(0xffffffffu, sx, l);
                const Real by = __shfl_sync(0xffffffffu, sy, l);
                const Real bz = __shfl_sync(0xffffffffu, sz, l);
                const Real bq = __shfl_sync(0xffffffffu, sq, l);
                if (my_active)
                    eval_esp_pair<WantForce>(xt - bx, yt - by, zt - bz, bq, rsc, cen, r_c_sq,
                                                           pot_acc, gx_acc, gy_acc, gz_acc);
            }
        }

        if (my_active) {
            pg_sorted[out_dim * trg + 0] = pot_acc;
            if constexpr (WantForce) {
                pg_sorted[out_dim * trg + 1] = gx_acc;
                pg_sorted[out_dim * trg + 2] = gy_acc;
                pg_sorted[out_dim * trg + 3] = gz_acc;
            }
        }
    }
}

// PruneSource: box-vs-point rather than PruneTile's box-vs-box, so a chunk whose AABB straddles r_c
// can still have most of its individual points culled. Phase 1 is identical to PruneTile's. Phase 2
// keeps the cheap AABB pre-filter, then per-lane tests survivors, warp-ballots them, and compacts
// through shared memory before evaluating.
template <bool WantForce>
__device__ void short_range_prune_source_body(const EspSrArgs &a)
{
    const int nc = a.nc;
    const int out_dim = a.out_dim;
    const Real rsc = a.rsc, cen = a.cen, r_c_sq = a.r_c_sq;
    const int *__restrict__ cell_start = a.cell_start;
    const Real *__restrict__ d_xs = a.xs;
    const Real *__restrict__ d_ys = a.ys;
    const Real *__restrict__ d_zs = a.zs;
    const Real *__restrict__ d_qs = a.qs;
    const int *__restrict__ nbc_tab = a.nbc_tab;
    const Real *__restrict__ off_tab = a.off_tab;
    Real *__restrict__ pg_sorted = a.pg_sorted;
    const int max_tiles = a.max_tiles;

    // Same per-tile table layout as PruneTile.
    extern __shared__ unsigned char s_raw[];
    Real *s_lo_x = reinterpret_cast<Real *>(s_raw);
    Real *s_lo_y = s_lo_x + max_tiles;
    Real *s_lo_z = s_lo_y + max_tiles;
    Real *s_hi_x = s_lo_z + max_tiles;
    Real *s_hi_y = s_hi_x + max_tiles;
    Real *s_hi_z = s_hi_y + max_tiles;
    Real *s_shift_x = s_hi_z + max_tiles;
    Real *s_shift_y = s_shift_x + max_tiles;
    Real *s_shift_z = s_shift_y + max_tiles;
    int  *s_base = reinterpret_cast<int *>(s_shift_z + max_tiles);
    int  *s_len  = s_base + max_tiles;

    __shared__ int  s_nb_lin[27];
    __shared__ int  s_nb_tile0[28];
    __shared__ Real s_nb_shift_x[27], s_nb_shift_y[27], s_nb_shift_z[27];

    // Per-warp compaction staging: one slot per lane. Static -- its size is density-independent.
    constexpr int kMaxWarps = BLOCK_SIZE / TILE_WIDTH;
    __shared__ Real s_stage_x[kMaxWarps][TILE_WIDTH];
    __shared__ Real s_stage_y[kMaxWarps][TILE_WIDTH];
    __shared__ Real s_stage_z[kMaxWarps][TILE_WIDTH];
    __shared__ Real s_stage_q[kMaxWarps][TILE_WIDTH];

    const int home = blockIdx.x;
    const int cx = home / (nc * nc);
    const int cy = (home / nc) % nc;
    const int cz = home % nc;

    const int hbeg = cell_start[home];
    const int n_trg = cell_start[home + 1] - hbeg;
    if (n_trg == 0) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane = tid % warpSize;
    const int n_warps = blockDim.x / warpSize;

    // Phase 1 (identical to PruneTile)
    if (tid < 27) {
        const int dzi = tid % 3, dyi = (tid / 3) % 3, dxi = tid / 9;
        const int nbx = nbc_tab[cx * 3 + dxi];
        const int nby = nbc_tab[cy * 3 + dyi];
        const int nbz = nbc_tab[cz * 3 + dzi];
        const Real ox = off_tab[cx * 3 + dxi];
        const Real oy = off_tab[cy * 3 + dyi];
        const Real oz = off_tab[cz * 3 + dzi];
        const int nb = (nbx * nc + nby) * nc + nbz;
        s_nb_lin[tid] = nb;
        s_nb_shift_x[tid] = ox;
        s_nb_shift_y[tid] = oy;
        s_nb_shift_z[tid] = oz;
        const int len = cell_start[nb + 1] - cell_start[nb];
        s_nb_tile0[tid] = (len + TILE_WIDTH - 1) / TILE_WIDTH;
    }
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < 27; ++i) {
            const int c = s_nb_tile0[i];
            s_nb_tile0[i] = acc;
            acc += c;
        }
        s_nb_tile0[27] = acc;
    }
    __syncthreads();
    const int n_stiles = min(s_nb_tile0[27], max_tiles);

    for (int gt = warp_id; gt < n_stiles; gt += n_warps) {
        int nbi = 0;
        while (nbi < 26 && s_nb_tile0[nbi + 1] <= gt)
            ++nbi;
        const int local_tile = gt - s_nb_tile0[nbi];
        const int nb = s_nb_lin[nbi];
        const int nb_beg = cell_start[nb], nb_end = cell_start[nb + 1];
        const int t_base = nb_beg + local_tile * TILE_WIDTH;
        const int t_len = min(TILE_WIDTH, nb_end - t_base);
        const Real ox = s_nb_shift_x[nbi], oy = s_nb_shift_y[nbi], oz = s_nb_shift_z[nbi];

        const bool valid = lane < t_len;
        const Real x = valid ? d_xs[t_base + lane] + ox : Real(0);
        const Real y = valid ? d_ys[t_base + lane] + oy : Real(0);
        const Real z = valid ? d_zs[t_base + lane] + oz : Real(0);
        Real lo_x = warp_reduce_min(valid ? x : Real(INFINITY));
        Real hi_x = warp_reduce_max(valid ? x : Real(-INFINITY));
        Real lo_y = warp_reduce_min(valid ? y : Real(INFINITY));
        Real hi_y = warp_reduce_max(valid ? y : Real(-INFINITY));
        Real lo_z = warp_reduce_min(valid ? z : Real(INFINITY));
        Real hi_z = warp_reduce_max(valid ? z : Real(-INFINITY));

        if (lane == 0) {
            s_lo_x[gt] = lo_x; s_lo_y[gt] = lo_y; s_lo_z[gt] = lo_z;
            s_hi_x[gt] = hi_x; s_hi_y[gt] = hi_y; s_hi_z[gt] = hi_z;
            s_shift_x[gt] = ox; s_shift_y[gt] = oy; s_shift_z[gt] = oz;
            s_base[gt] = t_base; s_len[gt] = t_len;
        }
    }
    __syncthreads();

    // Phase 2: per warp, box-vs-point pruning + compaction
    for (int t0 = warp_id * TILE_WIDTH; t0 < n_trg; t0 += n_warps * TILE_WIDTH) {
        const int tlen = min(TILE_WIDTH, n_trg - t0);
        const bool my_active = lane < tlen;
        const int trg = hbeg + t0 + lane;
        const Real xt = my_active ? d_xs[trg] : Real(0);
        const Real yt = my_active ? d_ys[trg] : Real(0);
        const Real zt = my_active ? d_zs[trg] : Real(0);

        const Real tlo_x = warp_reduce_min(my_active ? xt : Real(INFINITY));
        const Real thi_x = warp_reduce_max(my_active ? xt : Real(-INFINITY));
        const Real tlo_y = warp_reduce_min(my_active ? yt : Real(INFINITY));
        const Real thi_y = warp_reduce_max(my_active ? yt : Real(-INFINITY));
        const Real tlo_z = warp_reduce_min(my_active ? zt : Real(INFINITY));
        const Real thi_z = warp_reduce_max(my_active ? zt : Real(-INFINITY));

        Real pot_acc = Real(0), gx_acc = Real(0), gy_acc = Real(0), gz_acc = Real(0);

        for (int st = 0; st < n_stiles; ++st) {
            // Box-vs-box pre-filter on the Phase-1 AABBs: a chunk fully out of range needs no
            // per-point work.
            const Real ddx = max(Real(0), max(s_lo_x[st] - thi_x, tlo_x - s_hi_x[st]));
            const Real ddy = max(Real(0), max(s_lo_y[st] - thi_y, tlo_y - s_hi_y[st]));
            const Real ddz = max(Real(0), max(s_lo_z[st] - thi_z, tlo_z - s_hi_z[st]));
            const bool box_pruned = ddx * ddx + ddy * ddy + ddz * ddz > r_c_sq;

            const int s_len_i = s_len[st];
            if (lane == 0)
                if constexpr (PRUNE_STATS)
                    atomicAdd(&a.prune_stats[2], (unsigned long long)s_len_i);
            if (box_pruned)
                continue;

            const int s_base_i = s_base[st];
            const Real shx = s_shift_x[st], shy = s_shift_y[st], shz = s_shift_z[st];
            const bool have_src = lane < s_len_i;
            const Real sx = have_src ? d_xs[s_base_i + lane] + shx : Real(0);
            const Real sy = have_src ? d_ys[s_base_i + lane] + shy : Real(0);
            const Real sz = have_src ? d_zs[s_base_i + lane] + shz : Real(0);
            const Real sq = have_src ? d_qs[s_base_i + lane] : Real(0);

            // Same formula as above with the source side degenerated to a point (lo=hi=s).
            const Real pdx = max(Real(0), max(tlo_x - sx, sx - thi_x));
            const Real pdy = max(Real(0), max(tlo_y - sy, sy - thi_y));
            const Real pdz = max(Real(0), max(tlo_z - sz, sz - thi_z));
            const bool in_range = have_src && (pdx * pdx + pdy * pdy + pdz * pdz <= r_c_sq);

            const unsigned mask = __ballot_sync(0xffffffffu, in_range);
            if (lane == 0)
                if constexpr (PRUNE_STATS)
                    atomicAdd(&a.prune_stats[3], (unsigned long long)__popc(mask));
            if (mask == 0)
                continue; // chunk's box test didn't clear it, but no individual survivors either

            if (in_range) {
                const int slot = __popc(mask & ((1u << lane) - 1u));
                s_stage_x[warp_id][slot] = sx;
                s_stage_y[warp_id][slot] = sy;
                s_stage_z[warp_id][slot] = sz;
                s_stage_q[warp_id][slot] = sq;
            }
            __syncwarp(); // scatters visible before any lane reads the staging buffer

            const int m = __popc(mask);
            for (int i = 0; i < m; ++i) {
                const Real bx = s_stage_x[warp_id][i];
                const Real by = s_stage_y[warp_id][i];
                const Real bz = s_stage_z[warp_id][i];
                const Real bq = s_stage_q[warp_id][i];
                if (my_active)
                    eval_esp_pair<WantForce>(xt - bx, yt - by, zt - bz, bq, rsc, cen, r_c_sq,
                                                           pot_acc, gx_acc, gy_acc, gz_acc);
            }
            __syncwarp(); // all reads of this chunk's staging buffer done before the next chunk scatters
        }

        if (my_active) {
            pg_sorted[out_dim * trg + 0] = pot_acc;
            if constexpr (WantForce) {
                pg_sorted[out_dim * trg + 1] = gx_acc;
                pg_sorted[out_dim * trg + 2] = gy_acc;
                pg_sorted[out_dim * trg + 3] = gz_acc;
            }
        }
    }
}
// KERNEL_START

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE) DMK_ESP_SR_KERNEL_NAME(EspSrArgs a) {
    constexpr bool want_force = (WANT_FORCE != 0);
    if constexpr (STRATEGY == 0)
        short_range_dense_body<want_force, TARGETS_PER_THREAD>(a);
    else if constexpr (STRATEGY == 1)
        short_range_prune_tile_body<want_force>(a);
    else
        short_range_prune_source_body<want_force>(a);
}
