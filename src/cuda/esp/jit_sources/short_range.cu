// ESP short-range direct sum. The launcher prepends Real, DMK_ESP_SR_KERNEL_NAME, the coefficient
// packs, DMK_ESP_SR_EVALUATOR and the BLOCK_SIZE / TILE_WIDTH / TARGETS_PER_THREAD / STRATEGY /
// PRUNE_STATS / KERNEL constants. STRATEGY mirrors GpuSrStrategy: 0 Dense, 1 PruneTile, 2 PruneSource.

#include <dmk/cuda/esp_sr_kernelargs.hpp>
#include <dmk/cuda/poly_evaluators_device.hpp>

using EspSrArgs = dmk::cuda::EspSrArgs<Real>;
using Evaluator = DMK_ESP_SR_EVALUATOR;

constexpr int IN_DIM = Evaluator::KERNEL_INPUT_DIM;
constexpr int OUT_DIM = Evaluator::KERNEL_OUTPUT_DIM;
constexpr int NRM_DIM = Evaluator::NORMAL_DIM;
constexpr int NRM_SLOTS = NRM_DIM > 0 ? NRM_DIM : 1; // zero-length arrays break CUDA < 12.5

// NVRTC has no standard headers, so INFINITY does not exist.
__device__ __forceinline__ float dmk_inf(float) { return __int_as_float(0x7f800000); }
__device__ __forceinline__ double dmk_inf(double) { return __longlong_as_double(0x7ff0000000000000LL); }

// thresh2 = 0: R2 > 0 is what drops the self pair, as on the CPU.
__device__ __forceinline__ Evaluator esp_evaluator(const EspSrArgs &a) {
    return Evaluator{Real(0), a.r_c_sq, a.rsc, a.cen};
}

// Templated on Eval so the if constexpr condition is dependent and the untaken branch is discarded.
template <typename Eval>
__device__ __forceinline__ void esp_accumulate(const Eval &ev, Real (&acc)[Eval::KERNEL_OUTPUT_DIM], Real dx, Real dy,
                                               Real dz, const Real (&vs)[Eval::KERNEL_INPUT_DIM],
                                               const Real (&ns)[Eval::NORMAL_DIM > 0 ? Eval::NORMAL_DIM : 1]) {
    const Real dX[3] = {dx, dy, dz};
    if constexpr (Eval::NORMAL_DIM > 0)
        direct_eval_accumulate<true>(ev, acc, dX, vs, ns);
    else
        direct_eval_accumulate<true>(ev, acc, dX, vs);
}

// Dense: one block per home cell, each thread register-blocking TARGETS strided targets. The 27
// neighbour cells are staged through shared memory in block-sized tiles.

template <int TARGETS>
__device__ void short_range_dense_body(const EspSrArgs &a)
{
    const int nc = a.nc;
    const Evaluator ev = esp_evaluator(a);
    const int *__restrict__ cell_start = a.cell_start;
    const Real *__restrict__ d_xs = a.xs;
    const Real *__restrict__ d_ys = a.ys;
    const Real *__restrict__ d_zs = a.zs;
    const Real *__restrict__ d_qs = a.qs;
    const Real *__restrict__ d_ns = a.ns;
    const int n_sorted = a.n_sorted;
    const int *__restrict__ nbc_tab = a.nbc_tab;
    const Real *__restrict__ off_tab = a.off_tab;
    Real *__restrict__ pg_sorted = a.pg_sorted;

    __shared__ Real s_xs[BLOCK_SIZE];
    __shared__ Real s_ys[BLOCK_SIZE];
    __shared__ Real s_zs[BLOCK_SIZE];
    __shared__ Real s_qs[IN_DIM][BLOCK_SIZE];
    __shared__ Real s_ns[NRM_SLOTS][BLOCK_SIZE];

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
        Real acc[TARGETS][OUT_DIM] = {};
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

        // -1 is a free-space neighbour outside the box. Block-uniform, so __syncthreads stays
        // collective.
        for (int dxi = 0; dxi < 3; ++dxi) {
            const int nbx = nbc_tab[cx * 3 + dxi];
            if (nbx < 0)
                continue;
            const Real ox = off_tab[cx * 3 + dxi];
            for (int dyi = 0; dyi < 3; ++dyi) {
                const int nby = nbc_tab[cy * 3 + dyi];
                if (nby < 0)
                    continue;
                const Real oy = off_tab[cy * 3 + dyi];
                for (int dzi = 0; dzi < 3; ++dzi) {
                    const int nbz = nbc_tab[cz * 3 + dzi];
                    if (nbz < 0)
                        continue;
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
#pragma unroll
                            for (int c = 0; c < IN_DIM; ++c)
                                s_qs[c][threadIdx.x] = d_qs[c * n_sorted + s_idx];
                            if constexpr (NRM_DIM > 0) {
#pragma unroll
                                for (int c = 0; c < NRM_DIM; ++c)
                                    s_ns[c][threadIdx.x] = d_ns[c * n_sorted + s_idx];
                            }
                        }
                        __syncthreads();

                        const int n_local = min(BLOCK_SIZE, send - sbeg - tile * BLOCK_SIZE);

                        if (any_active) {
#pragma unroll
                            for (int k = 0; k < TARGETS; ++k) {
                                if (!active[k]) continue;
                                for (int s = 0; s < n_local; ++s) {
                                    Real vs[IN_DIM], ns[NRM_SLOTS] = {};
#pragma unroll
                                    for (int c = 0; c < IN_DIM; ++c)
                                        vs[c] = s_qs[c][s];
                                    if constexpr (NRM_DIM > 0) {
#pragma unroll
                                        for (int c = 0; c < NRM_DIM; ++c)
                                            ns[c] = s_ns[c][s];
                                    }
                                    esp_accumulate(ev, acc[k], xt[k] - s_xs[s], yt[k] - s_ys[s], zt[k] - s_zs[s], vs,
                                                   ns);
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
#pragma unroll
            for (int c = 0; c < OUT_DIM; ++c)
                pg_sorted[OUT_DIM * trg + c] = acc[k][c];
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

template <int UNUSED = 0>
__device__ void short_range_prune_tile_body(const EspSrArgs &a)
{
    const int nc = a.nc;
    const Real r_c_sq = a.r_c_sq;
    const Evaluator ev = esp_evaluator(a);
    const int *__restrict__ cell_start = a.cell_start;
    const Real *__restrict__ d_xs = a.xs;
    const Real *__restrict__ d_ys = a.ys;
    const Real *__restrict__ d_zs = a.zs;
    const Real *__restrict__ d_qs = a.qs;
    const Real *__restrict__ d_ns = a.ns;
    const int n_sorted = a.n_sorted;
    const int *__restrict__ nbc_tab = a.nbc_tab;
    const Real *__restrict__ off_tab = a.off_tab;
    Real *__restrict__ pg_sorted = a.pg_sorted;
    const int max_tiles = a.max_tiles;

    // Per-tile AABB/shift/base/len table.
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

    // Per-neighbour-cell metadata; static shared.
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
        // -1 means outside the box; zero tiles drops it from the prefix sum.
        const bool valid_nb = (nbx >= 0) && (nby >= 0) && (nbz >= 0);
        const int nb = valid_nb ? (nbx * nc + nby) * nc + nbz : 0;
        s_nb_lin[tid] = nb;
        s_nb_shift_x[tid] = ox;
        s_nb_shift_y[tid] = oy;
        s_nb_shift_z[tid] = oz;
        const int len = valid_nb ? cell_start[nb + 1] - cell_start[nb] : 0;
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
    // Exceeding the margin truncates the tile list rather than overflowing shared memory.
    const int n_stiles = min(s_nb_tile0[27], max_tiles);

    // Phase 1: one warp per source tile.
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
        Real lo_x = warp_reduce_min(valid ? x : dmk_inf(Real(0)));
        Real hi_x = warp_reduce_max(valid ? x : -dmk_inf(Real(0)));
        Real lo_y = warp_reduce_min(valid ? y : dmk_inf(Real(0)));
        Real hi_y = warp_reduce_max(valid ? y : -dmk_inf(Real(0)));
        Real lo_z = warp_reduce_min(valid ? z : dmk_inf(Real(0)));
        Real hi_z = warp_reduce_max(valid ? z : -dmk_inf(Real(0)));

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

        const Real tlo_x = warp_reduce_min(my_active ? xt : dmk_inf(Real(0)));
        const Real thi_x = warp_reduce_max(my_active ? xt : -dmk_inf(Real(0)));
        const Real tlo_y = warp_reduce_min(my_active ? yt : dmk_inf(Real(0)));
        const Real thi_y = warp_reduce_max(my_active ? yt : -dmk_inf(Real(0)));
        const Real tlo_z = warp_reduce_min(my_active ? zt : dmk_inf(Real(0)));
        const Real thi_z = warp_reduce_max(my_active ? zt : -dmk_inf(Real(0)));

        Real acc[OUT_DIM] = {};

        for (int st = 0; st < n_stiles; ++st) {

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
                continue; // whole tile pruned: no source load, no evaluator calls

            const int s_base_i = s_base[st], s_len_i = s_len[st];
            const Real shx = s_shift_x[st], shy = s_shift_y[st], shz = s_shift_z[st];
            const bool have_src = lane < s_len_i;
            const Real sx = have_src ? d_xs[s_base_i + lane] + shx : Real(0);
            const Real sy = have_src ? d_ys[s_base_i + lane] + shy : Real(0);
            const Real sz = have_src ? d_zs[s_base_i + lane] + shz : Real(0);
            Real sq[IN_DIM], sn[NRM_SLOTS] = {};
#pragma unroll
            for (int c = 0; c < IN_DIM; ++c)
                sq[c] = have_src ? d_qs[c * n_sorted + s_base_i + lane] : Real(0);
            if constexpr (NRM_DIM > 0) {
#pragma unroll
                for (int c = 0; c < NRM_DIM; ++c)
                    sn[c] = have_src ? d_ns[c * n_sorted + s_base_i + lane] : Real(0);
            }

            // Each lane loads one source and broadcasts in turn.
            for (int l = 0; l < s_len_i; ++l) {
                const Real bx = __shfl_sync(0xffffffffu, sx, l);
                const Real by = __shfl_sync(0xffffffffu, sy, l);
                const Real bz = __shfl_sync(0xffffffffu, sz, l);
                Real vs[IN_DIM], ns[NRM_SLOTS] = {};
#pragma unroll
                for (int c = 0; c < IN_DIM; ++c)
                    vs[c] = __shfl_sync(0xffffffffu, sq[c], l);
                if constexpr (NRM_DIM > 0) {
#pragma unroll
                    for (int c = 0; c < NRM_DIM; ++c)
                        ns[c] = __shfl_sync(0xffffffffu, sn[c], l);
                }
                if (my_active)
                    esp_accumulate(ev, acc, xt - bx, yt - by, zt - bz, vs, ns);
            }
        }

        if (my_active) {
#pragma unroll
            for (int c = 0; c < OUT_DIM; ++c)
                pg_sorted[OUT_DIM * trg + c] = acc[c];
        }
    }
}

// PruneSource: box-vs-point rather than box-vs-box, so a chunk straddling r_c can still have most
// of its points culled. Survivors are warp-ballotted and compacted through shared memory.
template <int UNUSED = 0>
__device__ void short_range_prune_source_body(const EspSrArgs &a)
{
    const int nc = a.nc;
    const Real r_c_sq = a.r_c_sq;
    const Evaluator ev = esp_evaluator(a);
    const int *__restrict__ cell_start = a.cell_start;
    const Real *__restrict__ d_xs = a.xs;
    const Real *__restrict__ d_ys = a.ys;
    const Real *__restrict__ d_zs = a.zs;
    const Real *__restrict__ d_qs = a.qs;
    const Real *__restrict__ d_ns = a.ns;
    const int n_sorted = a.n_sorted;
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

    // Per-warp compaction staging: one slot per lane.
    constexpr int kMaxWarps = BLOCK_SIZE / TILE_WIDTH;
    __shared__ Real s_stage_x[kMaxWarps][TILE_WIDTH];
    __shared__ Real s_stage_y[kMaxWarps][TILE_WIDTH];
    __shared__ Real s_stage_z[kMaxWarps][TILE_WIDTH];
    __shared__ Real s_stage_q[kMaxWarps][IN_DIM][TILE_WIDTH];
    __shared__ Real s_stage_n[kMaxWarps][NRM_SLOTS][TILE_WIDTH];

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

    // Phase 1, as PruneTile
    if (tid < 27) {
        const int dzi = tid % 3, dyi = (tid / 3) % 3, dxi = tid / 9;
        const int nbx = nbc_tab[cx * 3 + dxi];
        const int nby = nbc_tab[cy * 3 + dyi];
        const int nbz = nbc_tab[cz * 3 + dzi];
        const Real ox = off_tab[cx * 3 + dxi];
        const Real oy = off_tab[cy * 3 + dyi];
        const Real oz = off_tab[cz * 3 + dzi];
        const bool valid_nb = (nbx >= 0) && (nby >= 0) && (nbz >= 0);
        const int nb = valid_nb ? (nbx * nc + nby) * nc + nbz : 0;
        s_nb_lin[tid] = nb;
        s_nb_shift_x[tid] = ox;
        s_nb_shift_y[tid] = oy;
        s_nb_shift_z[tid] = oz;
        const int len = valid_nb ? cell_start[nb + 1] - cell_start[nb] : 0;
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
        Real lo_x = warp_reduce_min(valid ? x : dmk_inf(Real(0)));
        Real hi_x = warp_reduce_max(valid ? x : -dmk_inf(Real(0)));
        Real lo_y = warp_reduce_min(valid ? y : dmk_inf(Real(0)));
        Real hi_y = warp_reduce_max(valid ? y : -dmk_inf(Real(0)));
        Real lo_z = warp_reduce_min(valid ? z : dmk_inf(Real(0)));
        Real hi_z = warp_reduce_max(valid ? z : -dmk_inf(Real(0)));

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

        const Real tlo_x = warp_reduce_min(my_active ? xt : dmk_inf(Real(0)));
        const Real thi_x = warp_reduce_max(my_active ? xt : -dmk_inf(Real(0)));
        const Real tlo_y = warp_reduce_min(my_active ? yt : dmk_inf(Real(0)));
        const Real thi_y = warp_reduce_max(my_active ? yt : -dmk_inf(Real(0)));
        const Real tlo_z = warp_reduce_min(my_active ? zt : dmk_inf(Real(0)));
        const Real thi_z = warp_reduce_max(my_active ? zt : -dmk_inf(Real(0)));

        Real acc[OUT_DIM] = {};

        for (int st = 0; st < n_stiles; ++st) {
    
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
            Real sq[IN_DIM], sn[NRM_SLOTS] = {};
#pragma unroll
            for (int c = 0; c < IN_DIM; ++c)
                sq[c] = have_src ? d_qs[c * n_sorted + s_base_i + lane] : Real(0);
            if constexpr (NRM_DIM > 0) {
#pragma unroll
                for (int c = 0; c < NRM_DIM; ++c)
                    sn[c] = have_src ? d_ns[c * n_sorted + s_base_i + lane] : Real(0);
            }


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
#pragma unroll
                for (int c = 0; c < IN_DIM; ++c)
                    s_stage_q[warp_id][c][slot] = sq[c];
                if constexpr (NRM_DIM > 0) {
#pragma unroll
                    for (int c = 0; c < NRM_DIM; ++c)
                        s_stage_n[warp_id][c][slot] = sn[c];
                }
            }
            __syncwarp(); // scatters visible before any lane reads the staging buffer

            const int m = __popc(mask);
            for (int i = 0; i < m; ++i) {
                const Real bx = s_stage_x[warp_id][i];
                const Real by = s_stage_y[warp_id][i];
                const Real bz = s_stage_z[warp_id][i];
                Real vs[IN_DIM], ns[NRM_SLOTS] = {};
#pragma unroll
                for (int c = 0; c < IN_DIM; ++c)
                    vs[c] = s_stage_q[warp_id][c][i];
                if constexpr (NRM_DIM > 0) {
#pragma unroll
                    for (int c = 0; c < NRM_DIM; ++c)
                        ns[c] = s_stage_n[warp_id][c][i];
                }
                if (my_active)
                    esp_accumulate(ev, acc, xt - bx, yt - by, zt - bz, vs, ns);
            }
            __syncwarp(); // all reads of this chunk's staging buffer done before the next chunk scatters
        }

        if (my_active) {
#pragma unroll
            for (int c = 0; c < OUT_DIM; ++c)
                pg_sorted[OUT_DIM * trg + c] = acc[c];
        }
    }
}
// KERNEL_START

// KERNEL only separates modules whose baked coefficients coincide; TILE_WIDTH and PRUNE_STATS are
// read only by the pruned bodies. Named here so NVRTC does not warn they are unused.
static_assert(KERNEL >= 0, "KERNEL must be a dmk_ikernel value");
static_assert(TILE_WIDTH == 32, "the pruned strategies assume one warp per tile");
static_assert(PRUNE_STATS == 0 || PRUNE_STATS == 1, "PRUNE_STATS is a flag");

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE) DMK_ESP_SR_KERNEL_NAME(EspSrArgs a) {
    if constexpr (STRATEGY == 0)
        short_range_dense_body<TARGETS_PER_THREAD>(a);
    else if constexpr (STRATEGY == 1)
        short_range_prune_tile_body<>(a);
    else
        short_range_prune_source_body<>(a);
}
