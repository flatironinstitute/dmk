// The kernels behind metadata.hpp. nvcc compiles this one unit; it takes the tree's node arrays as
// raw device pointers, so it needs nothing from SCTL but Morton and Long.
//
// Two passes, each a handful of launches: box_status turns the per-box particle counts into
// subtree sums, leafness, the two expansion flags and the particle offsets; compute_device_metadata
// evaluates every remaining per-box predicate in one kernel, scans them together, and fills every
// work list in one more. The host reads back one scalar (the depth) between the two, and one block
// (the join) at the end.

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <cub/block/block_scan.cuh>
#include "metadata.hpp"

#include "dmk/cuda/helpers.hpp" // DMK_CHECK_CUDA, pool_alloc/pool_free

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

namespace dmk::cuda::pt {
namespace {

using Attr = device_tree::NodeAttr;

constexpr int BLOCK = 256;
constexpr unsigned FULL = 0xffffffffu;
constexpr int kMaxChildren = 8;
constexpr long kDirectGroup = 1L << 62; ///< added to the direct sort keys so they sort ahead of the c2p keys

template <int DIM>
constexpr int n_nbr_slots() {
    int v = 1;
    for (int d = 0; d < DIM; ++d)
        v *= 3;
    return v;
}
template <int DIM>
constexpr int n_child_slots() {
    return 1 << DIM;
}

/// Hands thrust's temporaries pooled device blocks instead of cudaMalloc/cudaFree per call, because
/// cudaFree synchronizes the whole device. Paired with par_nosync so a pass never joins on its own.
struct PoolAlloc {
    using value_type = char;
    char *allocate(std::ptrdiff_t n) { return static_cast<char *>(dmk::cuda_helpers::pool_alloc((std::size_t)n, 0)); }
    void deallocate(char *p, std::size_t n) { dmk::cuda_helpers::pool_free(p, n, 0); }
};

inline auto policy(PoolAlloc &alloc) { return thrust::cuda::par_nosync(alloc).on(0); }

/// A box at depth d has edge 2^-d; exact in either precision.
template <typename Real>
__device__ Real level_size(int lvl) {
    return Real(1) / Real(1L << lvl);
}

// ---------------------------------------------------------------------------------------------
// box_status
// ---------------------------------------------------------------------------------------------

/// The three per-box particle counts, carried through one scan instead of three.
struct Counts3 {
    long halo, own, trg;
};
struct Counts3Add {
    __host__ __device__ Counts3 operator()(const Counts3 &a, const Counts3 &b) const {
        return {a.halo + b.halo, a.own + b.own, a.trg + b.trg};
    }
};

/// n_boxes+1 threads: the tail slot is the scan's total. Leafness comes from the node's own
/// attribute at one rank (nothing refined a node elsewhere); with several ranks the reduced flags
/// were uploaded already and `attr` is null. The owned counts are the halo counts inside the
/// owned node range, which is what adopt_device_tree used to copy on the host.
__global__ void k_pack_counts(const sctl::Long *__restrict__ halo, const sctl::Long *__restrict__ trg,
                              const Attr *__restrict__ attr, long n_boxes, long owned_begin, long owned_end,
                              Counts3 *__restrict__ out, unsigned char *__restrict__ leaf,
                              unsigned long long *__restrict__ depth_max) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i > n_boxes)
        return;
    if (!i)
        *depth_max = 0;
    if (i == n_boxes) {
        out[i] = {0, 0, 0};
        return;
    }
    if (attr)
        leaf[i] = attr[i].Leaf ? 1 : 0;
    const long h = halo[i];
    out[i] = {h, (i >= owned_begin && i < owned_end) ? h : 0, trg ? trg[i] : 0};
}

/// Subtree sums from the scanned counts, the depth, and the plane-wave flag. The run ends at the
/// next sibling of the box or of its nearest ancestor that is not a last child, which is what the
/// host walk finds level by level. A box carries a plane-wave expansion unless it and all its
/// neighbours are leaves; the root always does.
template <int DIM>
__global__ void k_subtree_flags(const sctl::Morton<DIM> *__restrict__ mid, const sctl::Long *__restrict__ parent,
                                const sctl::Long *__restrict__ child, const sctl::Long *__restrict__ nbr,
                                const Counts3 *__restrict__ off, const unsigned char *__restrict__ leaf,
                                long n_boxes, int *__restrict__ src_halo, int *__restrict__ src_own,
                                int *__restrict__ trg_own, unsigned char *__restrict__ ifpwexp,
                                unsigned long long *__restrict__ depth_max) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    const unsigned depth = __reduce_max_sync(FULL, i < n_boxes ? (unsigned)mid[i].Depth() : 0u);
    if (!(threadIdx.x & 31))
        atomicMax(depth_max, (unsigned long long)depth);
    if (i >= n_boxes)
        return;
    constexpr int NC = n_child_slots<DIM>();
    constexpr int NN = n_nbr_slots<DIM>();
    long j = i;
    while (j > 0 && mid[j].Path2Node() == NC - 1)
        j = parent[j];
    const Counts3 end = j > 0 ? off[child[parent[j] * NC + mid[j].Path2Node() + 1]] : off[n_boxes];
    const Counts3 beg = off[i];
    src_halo[i] = (int)(end.halo - beg.halo);
    src_own[i] = (int)(end.own - beg.own);
    trg_own[i] = (int)(end.trg - beg.trg);

    unsigned char v = leaf[i] ? 0 : 1;
    for (int k = 0; k < NN && !v; ++k) {
        const sctl::Long nb = nbr[i * NN + k];
        if (nb >= 0 && !leaf[nb])
            v = 1;
    }
    ifpwexp[i] = i == 0 ? 1 : v;
}

/// The eval flag and the particle offsets. A box takes its own flag when it carries a plane-wave
/// expansion and its parent's otherwise, and those two cases are disjoint, so the host's level
/// ordering carries no information: this is a pure per-box predicate. The offsets are the scans
/// times a component count each.
template <int DIM>
__global__ void k_eval_flags_offsets(const sctl::Long *__restrict__ parent, const sctl::Long *__restrict__ child,
                                     const unsigned char *__restrict__ ifpwexp, const int *__restrict__ src_own,
                                     const int *__restrict__ trg_own, const Counts3 *__restrict__ off, long n_boxes,
                                     int charge_dof, int n_tables_up, int dof_src, int dof_trg,
                                     unsigned char *__restrict__ out, long *__restrict__ offsets,
                                     int *__restrict__ zero, int n_zero) {
    const long b = blockIdx.x * (long)blockDim.x + threadIdx.x;
    for (long i = b; i < n_zero; i += (long)gridDim.x * blockDim.x)
        zero[i] = 0;
    if (b >= n_boxes)
        return;
    constexpr int NC = n_child_slots<DIM>();
    const auto any_pwexp_child = [&](long p) {
        for (int c = 0; c < NC; ++c) {
            const sctl::Long cb = child[p * NC + c];
            if (cb >= 0 && ifpwexp[cb])
                return true;
        }
        return false;
    };

    unsigned char v = 0;
    if (src_own[b] + trg_own[b]) {
        if (ifpwexp[b]) {
            v = any_pwexp_child(b) ? 0 : 1;
        } else {
            const sctl::Long p = parent[b];
            if (p >= 0 && ifpwexp[p] && (src_own[p] + trg_own[p]) && any_pwexp_child(p))
                v = 1;
        }
    }
    out[b] = v;

    const Counts3 s = off[b];
    offsets[off_r_src_halo * n_boxes + b] = DIM * s.halo;
    offsets[off_charge_halo * n_boxes + b] = charge_dof * s.halo;
    offsets[off_normal_halo * n_boxes + b] = DIM * s.halo;
    offsets[off_r_trg * n_boxes + b] = DIM * s.trg;
    offsets[off_pot_trg * n_boxes + b] = dof_trg * s.trg;
    offsets[off_pot_src * n_boxes + b] = dof_src * s.own;
    offsets[off_r_src_owned * n_boxes + b] = DIM * s.own;
    offsets[off_charge_owned * n_boxes + b] = n_tables_up * s.own;
}

// ---------------------------------------------------------------------------------------------
// compute_device_metadata
// ---------------------------------------------------------------------------------------------

/// Every per-box quantity that is later compacted, carried through one scan. c2p packs the group
/// count in the high half and the source count in the low half; neither total can reach 2^32
/// because a box is at most one group and one group's source.
struct ScanIn {
    long up, down, c2p, eval, zero, pwout;
};
struct ScanAdd {
    __host__ __device__ ScanIn operator()(const ScanIn &a, const ScanIn &b) const {
        return {a.up + b.up, a.down + b.down, a.c2p + b.c2p, a.eval + b.eval, a.zero + b.zero, a.pwout + b.pwout};
    }
};

/// Per-level counters the predicate pass accumulates and the level scan turns into offsets. The
/// counters are zeroed by the scan afterwards so the fill pass can use them as cursors.
struct LevelScratch {
    int *tp;          ///< [nl] downward tensorprod pairs
    int *tpup;        ///< [nl] upward tensorprod pairs
    int *bucket;      ///< [nl*(NCMAX+1)] upward parents by child count
    int *bucket_base; ///< [nl*(NCMAX+1)]
    int *pw_tot;      ///< [kMaxLevels*2] plane-wave list totals per level, from the chunk table
    static constexpr int NCB = kMaxChildren + 1;
    __host__ __device__ static constexpr int size(int nl) { return nl * (2 + 2 * NCB) + 2 * kMaxLevels; }
    __host__ __device__ explicit LevelScratch(int *base, int nl)
        : tp(base), tpup(base + nl), bucket(base + 2 * nl), bucket_base(base + 2 * nl + nl * NCB),
          pw_tot(base + 2 * nl + 2 * nl * NCB) {}
};

/// The per-level tables in the join block.
struct JoinTables {
    long *tp_off, *tp_up_off, *tp_up_par_off, *pw_eval_off, *pw_form_off, *group_base;
    __host__ __device__ JoinTables(long *join, int nl)
        : tp_off(join + join_tables), tp_up_off(tp_off + nl + 1), tp_up_par_off(tp_up_off + nl + 1),
          pw_eval_off(tp_up_par_off + nl + 1), pw_form_off(pw_eval_off + nl + 1), group_base(pw_form_off + nl + 1) {}
};

/// One charge2proxy group per box that contributes charges directly to a proxy expansion.
/// A box with any proxy-bearing child gathers only its non-proxy-bearing children; one with
/// none gathers itself. Mirrors DMKPtTree::build_upward_pass_work_lists. Every group has at
/// least one source box with sources, so every group launches.
template <int DIM>
__device__ bool c2p_group_of(const sctl::Long *__restrict__ child, const int *__restrict__ src_owned,
                             const unsigned char *__restrict__ ifpwexp, long box, int *src_boxes, int &n_src) {
    constexpr int NC = n_child_slots<DIM>();
    n_src = 0;
    if (!(src_owned[box] > 0 && ifpwexp[box]))
        return false;

    bool from_children = false;
    for (int c = 0; c < NC; ++c) {
        const sctl::Long cb = child[box * NC + c];
        if (cb >= 0 && src_owned[cb] > 0 && ifpwexp[cb]) {
            from_children = true;
            break;
        }
    }
    if (!from_children) {
        src_boxes[n_src++] = (int)box;
        return true;
    }
    for (int c = 0; c < NC; ++c) {
        const sctl::Long cb = child[box * NC + c];
        if (cb >= 0 && src_owned[cb] > 0 && !ifpwexp[cb])
            src_boxes[n_src++] = (int)cb;
    }
    return n_src > 0;
}

/// Tensorprod pairs: a parent that carries a plane-wave expansion, holds points and is not itself
/// an eval leaf pairs with each of its non-empty children.
template <int DIM>
__device__ int tp_child_count(const sctl::Long *__restrict__ child, const int *__restrict__ src_own,
                              const int *__restrict__ trg_own, long b) {
    constexpr int NC = n_child_slots<DIM>();
    int n = 0;
    for (int c = 0; c < NC; ++c) {
        const sctl::Long cb = child[b * NC + c];
        if (cb >= 0 && (src_own[cb] + trg_own[cb]))
            ++n;
    }
    return n;
}

__device__ bool tp_is_parent(const unsigned char *__restrict__ ifpwexp, const unsigned char *__restrict__ iftpeval,
                             const int *__restrict__ src_own, const int *__restrict__ trg_own, long b) {
    return ifpwexp[b] && (src_own[b] + trg_own[b]) && !iftpeval[b];
}

/// Upward tensorprod pairs: a parent carrying sources and a plane-wave expansion pairs with each
/// child that does the same.
template <int DIM>
__device__ int tpup_child_count(const sctl::Long *__restrict__ child, const int *__restrict__ src_own,
                                const unsigned char *__restrict__ ifpwexp, long b) {
    constexpr int NC = n_child_slots<DIM>();
    if (!(src_own[b] > 0 && ifpwexp[b]))
        return 0;
    int n = 0;
    for (int c = 0; c < NC; ++c) {
        const sctl::Long cb = child[b * NC + c];
        if (cb >= 0 && src_own[cb] > 0 && ifpwexp[cb])
            ++n;
    }
    return n;
}

/// The two per-level plane-wave lists. A box has an upward proxy slot iff ifpwexp and sources
/// under it on some rank, which is what proxy_off_up != -1 used to encode.
__device__ bool pw_eval_pred(const unsigned char *ifpwexp, const int *src_own, const int *trg_own, long b) {
    return ifpwexp[b] && (src_own[b] + trg_own[b]) > 0;
}
__device__ bool has_proxy_up(const unsigned char *ifpwexp, const int *src_global, long b) {
    return ifpwexp[b] && src_global[b] > 0;
}
__device__ bool pw_form_pred(const unsigned char *ifpwexp, const int *src_global, bool skip_root, long b) {
    return has_proxy_up(ifpwexp, src_global, b) && !(skip_root && b == 0);
}

/// Everything the tree's node arrays decide box by box, in one pass: geometry, the scan inputs,
/// the per-level pair counts, the sort-array initialisation, and a per-(chunk, level) count of the
/// plane-wave lists that lets the fill place each box in Morton order within its level without a
/// serial sweep. One block is one chunk of BLOCK boxes; n_boxes+1 threads, the tail seeding the
/// scan total and the summary slots.
template <typename Real, int DIM>
__global__ void k_box_predicates(const sctl::Morton<DIM> *__restrict__ mid, const sctl::Long *__restrict__ child,
                                 const unsigned char *__restrict__ ifpwexp, const unsigned char *__restrict__ iftpeval,
                                 const int *__restrict__ src_own, const int *__restrict__ trg_own,
                                 const int *__restrict__ src_global, long n_boxes, int n_levels, int n_coeffs_up,
                                 int n_coeffs_down, bool skip_root, Real *__restrict__ centers,
                                 int *__restrict__ levels, ScanIn *__restrict__ scan, long *__restrict__ counts_up,
                                 int *__restrict__ sorted, long *__restrict__ keys, unsigned char *__restrict__ is_c2p,
                                 unsigned char *__restrict__ is_tpup, int *__restrict__ level_scratch,
                                 int *__restrict__ chunk_table, long *__restrict__ join) {
    constexpr int NC = n_child_slots<DIM>();
    __shared__ int s_cnt[kMaxLevels * 2];
    for (int t = threadIdx.x; t < kMaxLevels * 2; t += blockDim.x)
        s_cnt[t] = 0;
    __syncthreads();

    const long b = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (b == n_boxes) {
        scan[b] = {0, 0, 0, 0, 0, 0};
        join[join_n_direct_work] = 0;
        join[join_min_direct_level] = n_levels;
        for (int k = join_direct_src; k <= join_direct_trg_sq; ++k)
            join[k] = 0;
    }
    if (b < n_boxes) {
        LevelScratch ls(level_scratch, n_levels);
        const int lvl = mid[b].Depth();
        levels[b] = lvl;
        Real origin[DIM];
        mid[b].Coord(origin);
        const Real half = Real(0.5) * level_size<Real>(lvl);
        for (int d = 0; d < DIM; ++d)
            centers[b * DIM + d] = origin[d] + half;

        const bool proxy_up = has_proxy_up(ifpwexp, src_global, b);
        const long up = proxy_up ? n_coeffs_up : 0;
        const long down = ((ifpwexp[b] || iftpeval[b]) && (src_own[b] + trg_own[b])) ? n_coeffs_down : 0;
        counts_up[b] = up;

        int src_boxes[NC];
        int n_src = 0;
        const bool live = c2p_group_of<DIM>(child, src_own, ifpwexp, b, src_boxes, n_src);
        is_c2p[b] = live;

        sorted[b] = (int)b;
        sorted[n_boxes + b] = (int)b;
        keys[n_boxes + b] = -1;

        if (tp_is_parent(ifpwexp, iftpeval, src_own, trg_own, b)) {
            const int n = tp_child_count<DIM>(child, src_own, trg_own, b);
            if (n)
                atomicAdd(&ls.tp[lvl], n);
        }
        const int k = tpup_child_count<DIM>(child, src_own, ifpwexp, b);
        is_tpup[b] = k > 0;
        if (k) {
            atomicAdd(&ls.tpup[lvl], k);
            atomicAdd(&ls.bucket[lvl * LevelScratch::NCB + (NC - k)], 1);
        }

        const bool pe = pw_eval_pred(ifpwexp, src_own, trg_own, b);
        const bool pf = pw_form_pred(ifpwexp, src_global, skip_root, b);
        if (pe)
            atomicAdd(&s_cnt[lvl * 2], 1);
        if (pf)
            atomicAdd(&s_cnt[lvl * 2 + 1], 1);

        // form_outgoing launches the root unconditionally, and the list omits it under PBC;
        // anything read without a writer that assigns first would carry the previous eval's values.
        const bool read_set = pf || (b == 0 && proxy_up);
        const long zero = (read_set && !(live || k > 0)) ? 1 : 0;
        // A pw_out slab only where proxy2pw writes one: shift_pw skips a box without a slab, which
        // is what an unwritten zero slab contributed.
        const long pwout = pf ? 1 : 0;
        scan[b] = {up, down, live ? ((1L << 32) | (long)n_src) : 0L, iftpeval[b] ? 1L : 0L, zero, pwout};
    }
    __syncthreads();
    for (int t = threadIdx.x; t < kMaxLevels * 2; t += blockDim.x)
        chunk_table[blockIdx.x * (kMaxLevels * 2) + t] = s_cnt[t];
}

/// Exclusive scan of one column of the chunk table, one block per (level, list); the column total
/// is that level's list length.
__global__ void k_chunk_scan(int *__restrict__ table, long n_chunks, int *__restrict__ pw_tot) {
    using Scan = cub::BlockScan<int, BLOCK>;
    __shared__ typename Scan::TempStorage tmp;
    const int col = blockIdx.x;
    int base = 0;
    for (long t = 0; t < n_chunks; t += BLOCK) {
        const long c = t + threadIdx.x;
        const int v = c < n_chunks ? table[c * (kMaxLevels * 2) + col] : 0;
        int pre = 0, tot = 0;
        Scan(tmp).ExclusiveSum(v, pre, tot);
        if (c < n_chunks)
            table[c * (kMaxLevels * 2) + col] = base + pre;
        base += tot;
        __syncthreads();
    }
    if (!threadIdx.x)
        pw_tot[col] = base;
}

/// Per-level prefix sums, the scan totals and the summary scalars. A few hundred entries: staged
/// through shared memory and walked by one thread, which beats a device scan's launches.
template <int DIM>
__global__ void k_level_scan(int *__restrict__ level_scratch, int n_levels,
                             const ScanIn *__restrict__ scan, long n_boxes, long n_pw_per_box,
                             long *__restrict__ join, long *__restrict__ proxy_totals) {
    constexpr int NC = n_child_slots<DIM>();
    constexpr int NCB = LevelScratch::NCB;
    __shared__ int s[LevelScratch::size(kMaxLevels)];
    const int n = LevelScratch::size(n_levels);
    for (int t = threadIdx.x; t < n; t += blockDim.x)
        s[t] = level_scratch[t];
    __syncthreads();
    if (!threadIdx.x) {
        LevelScratch ls(s, n_levels);
        JoinTables T(join, n_levels);
        long tp = 0, pair = 0, par = 0, eval = 0, form = 0, max_form = 0;
        for (int L = 0; L < n_levels; ++L) {
            T.tp_off[L] = tp;
            tp += ls.tp[L];
            ls.tp[L] = 0;
            T.tp_up_off[L] = pair;
            pair += ls.tpup[L];
            ls.tpup[L] = 0;
            T.tp_up_par_off[L] = par;
            for (int c = 0; c <= NC; ++c) {
                const int i = L * NCB + c;
                ls.bucket_base[i] = (int)par;
                par += ls.bucket[i];
                ls.bucket[i] = 0;
            }
            T.pw_eval_off[L] = eval;
            T.pw_form_off[L] = form;
            const int ne = ls.pw_tot[L * 2], nf = ls.pw_tot[L * 2 + 1];
            eval += ne;
            form += nf;
            max_form = nf > max_form ? nf : max_form;
        }
        T.tp_off[n_levels] = tp;
        T.tp_up_off[n_levels] = pair;
        T.tp_up_par_off[n_levels] = par;
        T.pw_eval_off[n_levels] = eval;
        T.pw_form_off[n_levels] = form;
        join[join_max_pw_form] = max_form;
        const ScanIn tot = scan[n_boxes];
        join[join_n_c2p_groups] = tot.c2p >> 32;
        join[join_n_eval] = tot.eval;
        join[join_n_zero] = tot.zero;
        join[join_pw_out_size] = tot.pwout * n_pw_per_box;
        proxy_totals[0] = tot.up;
        proxy_totals[1] = tot.down;
    }
    __syncthreads();
    for (int t = threadIdx.x; t < n; t += blockDim.x)
        level_scratch[t] = s[t];
}

/// Every work list, filled from the scans. The plane-wave lists keep box order within a level --
/// shift_pw merges consecutive entries and relies on them being Morton-adjacent -- so a box's slot
/// is its level's base, plus its chunk's prefix from the chunk table, plus its rank inside the
/// chunk: lanes of one level are matched within the warp and counted below the lane, then the
/// warps ahead in the block are summed. Everything else is grouped by parent and otherwise
/// unordered: a box is the child of exactly one parent, so every pair writes a distinct slot.
template <int DIM>
__global__ void k_box_fill(const sctl::Long *__restrict__ child, const unsigned char *__restrict__ ifpwexp,
                           const unsigned char *__restrict__ iftpeval, const int *__restrict__ src_own,
                           const int *__restrict__ trg_own, const int *__restrict__ src_global,
                           const int *__restrict__ levels, const unsigned char *__restrict__ is_c2p, long n_boxes,
                           int n_levels, bool skip_root, long n_pw_per_box, const ScanIn *__restrict__ scan,
                           const int *__restrict__ chunk_table, int *__restrict__ level_scratch,
                           long *__restrict__ join, long *__restrict__ off_up, long *__restrict__ off_down,
                           int *__restrict__ c2p_center, int *__restrict__ c2p_level, int *__restrict__ c2p_off,
                           int *__restrict__ c2p_count, int *__restrict__ c2p_flat, long *__restrict__ keys,
                           int *__restrict__ tp_par, int *__restrict__ tp_chi, int *__restrict__ tp_oct,
                           int *__restrict__ tp_dst, int *__restrict__ eval_list, int *__restrict__ up_src,
                           int *__restrict__ up_dst, int *__restrict__ up_oct, int *__restrict__ up_assign,
                           int *__restrict__ up_par, int *__restrict__ up_beg, int *__restrict__ up_cnt,
                           int *__restrict__ eval_flat, int *__restrict__ form_flat, int *__restrict__ zero_boxes,
                           long *__restrict__ pw_out_off) {
    constexpr int NC = n_child_slots<DIM>();
    constexpr int WARPS = BLOCK / 32;
    __shared__ int s_w[WARPS * kMaxLevels * 2];
    for (int t = threadIdx.x; t < WARPS * kMaxLevels * 2; t += blockDim.x)
        s_w[t] = 0;
    __syncthreads();

    const long b = blockIdx.x * (long)blockDim.x + threadIdx.x;
    const bool valid = b < n_boxes;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int lvl = valid ? levels[b] : -1;
    const bool pe = valid && pw_eval_pred(ifpwexp, src_own, trg_own, b);
    const bool pf = valid && pw_form_pred(ifpwexp, src_global, skip_root, b);

    const unsigned same = __match_any_sync(FULL, lvl);
    const unsigned be = __ballot_sync(FULL, pe);
    const unsigned bf = __ballot_sync(FULL, pf);
    const unsigned below = (1u << lane) - 1;
    if (lvl >= 0 && (__ffs(same) - 1) == lane) {
        s_w[(warp * kMaxLevels + lvl) * 2] = __popc(same & be);
        s_w[(warp * kMaxLevels + lvl) * 2 + 1] = __popc(same & bf);
    }
    __syncthreads();
    const JoinTables T(join, n_levels);
    if (pe || pf) {
        int ahead_e = 0, ahead_f = 0;
        for (int w = 0; w < warp; ++w) {
            ahead_e += s_w[(w * kMaxLevels + lvl) * 2];
            ahead_f += s_w[(w * kMaxLevels + lvl) * 2 + 1];
        }
        const int *chunk = chunk_table + blockIdx.x * (kMaxLevels * 2) + lvl * 2;
        if (pe)
            eval_flat[T.pw_eval_off[lvl] + chunk[0] + ahead_e + __popc(same & be & below)] = (int)b;
        if (pf)
            form_flat[T.pw_form_off[lvl] + chunk[1] + ahead_f + __popc(same & bf & below)] = (int)b;
    }
    if (!valid)
        return;

    LevelScratch ls(level_scratch, n_levels);
    const ScanIn s0 = scan[b], s1 = scan[b + 1];
    off_up[b] = s1.up > s0.up ? s0.up : -1;
    off_down[b] = s1.down > s0.down ? s0.down : -1;
    pw_out_off[b] = s1.pwout > s0.pwout ? s0.pwout * n_pw_per_box : -1;
    if (s1.zero > s0.zero)
        zero_boxes[s0.zero] = (int)b;
    if (iftpeval[b])
        eval_list[s0.eval] = (int)b;

    { // charge2proxy group, with the same work key the host used: sources, then 32-wide chunks
        int src_boxes[NC];
        int n_src = 0;
        if (c2p_group_of<DIM>(child, src_own, ifpwexp, b, src_boxes, n_src)) {
            constexpr int CHUNK = 32;
            const int g = (int)(s0.c2p >> 32);
            const int off = (int)(s0.c2p & 0xffffffffL);
            long total_sources = 0, total_chunks = 0;
            for (int k = 0; k < n_src; ++k) {
                const int n = src_own[src_boxes[k]];
                total_sources += n;
                total_chunks += (n + CHUNK - 1) / CHUNK;
                c2p_flat[off + k] = src_boxes[k];
            }
            c2p_center[g] = (int)b;
            c2p_level[g] = lvl;
            c2p_off[g] = off;
            c2p_count[g] = n_src;
            keys[n_boxes + g] = total_sources * 1024L + total_chunks;
        }
    }

    if (tp_is_parent(ifpwexp, iftpeval, src_own, trg_own, b)) {
        const int n = tp_child_count<DIM>(child, src_own, trg_own, b);
        if (n) {
            int at = T.tp_off[lvl] + atomicAdd(&ls.tp[lvl], n);
            for (int c = 0; c < NC; ++c) {
                const sctl::Long cb = child[b * NC + c];
                if (cb < 0 || !(src_own[cb] + trg_own[cb]))
                    continue;
                tp_par[at] = (int)b;
                tp_chi[at] = (int)cb;
                tp_oct[at] = c;
                // pw2proxy covers a box iff ifpwexp && (src+trg) > 0, and the child already
                // satisfies the latter, so without ifpwexp this pair is its only writer.
                tp_dst[at] = ifpwexp[cb] ? 0 : 1;
                ++at;
            }
        }
    }

    const int k = tpup_child_count<DIM>(child, src_own, ifpwexp, b);
    if (k) {
        const int rel = atomicAdd(&ls.tpup[lvl], k); // level-relative, as the launcher indexes it
        int at = T.tp_up_off[lvl] + rel;
        for (int c = 0; c < NC; ++c) {
            const sctl::Long cb = child[b * NC + c];
            if (cb < 0 || !(src_own[cb] > 0 && ifpwexp[cb]))
                continue;
            up_src[at] = (int)cb;
            up_dst[at] = (int)b;
            up_oct[at] = c;
            // charge2proxy writes this box first where it has a group; otherwise the first child
            // owns the first write. Exactly one pair per parent may be marked.
            up_assign[at] = (at == T.tp_up_off[lvl] + rel && !is_c2p[b]) ? 1 : 0;
            ++at;
        }
        // Parents with the most children first, so the launcher's early blocks carry the long tails.
        const int bk = lvl * LevelScratch::NCB + (NC - k);
        const int slot = ls.bucket_base[bk] + atomicAdd(&ls.bucket[bk], 1);
        up_par[slot] = (int)b;
        up_beg[slot] = rel;
        up_cnt[slot] = k;
    }
}

template <typename Real, int DIM>
__device__ void shift_from_slot(int k, Real bsize, const Real *c_box, const Real *c_nbr, int *shift) {
    for (int d = 0; d < DIM; d++) {
        const int dir = (k % 3) - 1;
        k /= 3;
        const double expected = (double)c_box[d] + dir * (double)bsize;
        shift[d] = (int)round(expected - (double)c_nbr[d]);
    }
}

/// Chebyshev-ball touching test on box centers, shifted source included.
template <typename Real, int DIM>
__device__ bool within_cutoff(const Real *c_trg, const Real *c_src, const int *shift, double cutoff) {
    for (int d = 0; d < DIM; d++)
        if (fabs((double)c_trg[d] - ((double)c_src[d] + shift[d])) > cutoff)
            return false;
    return true;
}

/// One warp per box, one lane per neighbour slot: the lane screens that slot at the box's level
/// (the leaf neighbour, or the children of a refined one that touch the box) and again at the
/// parent's level. Two warp scans place the survivors in the order the serial walk produced:
/// same-level slots first, then the parent's.
template <typename Real, int DIM>
__global__ void k_list1(const sctl::Morton<DIM> *mid, const Attr *attr, const unsigned char *leaf,
                        const sctl::Long *parent, const sctl::Long *child, const sctl::Long *nbr, const int *src_cnt,
                        const Real *centers, long n_boxes, int stride, bool periodic, int *list1, int *count,
                        signed char *shift_out) {
    constexpr int NN = n_nbr_slots<DIM>();
    constexpr int NC = n_child_slots<DIM>();
    const long box = (blockIdx.x * (long)blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (box >= n_boxes)
        return;
    const bool active = leaf[box] && !attr[box].Ghost;
    if (!active) {
        if (!lane)
            count[box] = 0;
        return;
    }

    const int lvl = mid[box].Depth();
    const Real bsize = level_size<Real>(lvl);
    const double cutoff_child = 1.05 * 0.75 * (double)bsize;
    const double cutoff_parent_nbr = 1.5 * 1.05 * (double)bsize;
    const Real *c_box = centers + box * DIM;

    int cand[NC];
    int n_same = 0;
    int shift[DIM] = {};
    if (lane < NN) {
        const sctl::Long nb = nbr[box * NN + lane];
        if (nb >= 0) {
            if (periodic)
                shift_from_slot<Real, DIM>(lane, bsize, c_box, centers + nb * DIM, shift);
            if (leaf[nb]) {
                if (src_cnt[nb])
                    cand[n_same++] = (int)nb;
            } else {
                for (int c = 0; c < NC; c++) {
                    const sctl::Long ch = child[nb * NC + c];
                    if (ch >= 0 && src_cnt[ch] &&
                        within_cutoff<Real, DIM>(c_box, centers + ch * DIM, shift, cutoff_child))
                        cand[n_same++] = (int)ch;
                }
            }
        }
    }
    int par_cand = -1;
    int par_shift[DIM] = {};
    if (lvl != 0 && lane < NN) {
        const sctl::Long par = parent[box];
        const sctl::Long nb = nbr[par * NN + lane];
        if (nb >= 0 && leaf[nb] && src_cnt[nb]) {
            if (periodic)
                shift_from_slot<Real, DIM>(lane, level_size<Real>(lvl - 1), centers + par * DIM, centers + nb * DIM,
                                           par_shift);
            if (within_cutoff<Real, DIM>(c_box, centers + nb * DIM, par_shift, cutoff_parent_nbr))
                par_cand = (int)nb;
        }
    }

    int pos = n_same;
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(FULL, pos, o);
        if (lane >= o)
            pos += v;
    }
    const int total_same = __shfl_sync(FULL, pos, 31);
    pos -= n_same;
    int ppos = par_cand >= 0 ? 1 : 0;
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(FULL, ppos, o);
        if (lane >= o)
            ppos += v;
    }
    const int total = total_same + __shfl_sync(FULL, ppos, 31);
    ppos = total_same + ppos - (par_cand >= 0 ? 1 : 0);

    const auto put = [&](int k, int nb, const int *sh) {
        list1[box * stride + k] = nb;
        if (shift_out)
            for (int d = 0; d < DIM; d++)
                shift_out[(box * (long)stride + k) * DIM + d] = (signed char)sh[d];
    };
    for (int i = 0; i < n_same; ++i)
        put(pos + i, cand[i], shift);
    if (par_cand >= 0)
        put(ppos, par_cand, par_shift);
    if (!lane)
        count[box] = total;
}

/// Per-box near-field cost as a sort key, offset so the direct keys sort ahead of the c2p keys
/// that share the array. Excluded boxes take -1 and so land past the end of the work list.
__global__ void k_direct_work_cost(const Attr *attr, const unsigned char *leaf, const int *src_cnt_halo,
                                   const int *src_cnt_owned, const int *trg_cnt_owned, const int *list1,
                                   const int *count, long n_boxes, int stride, long *keys) {
    const long box = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (box >= n_boxes)
        return;
    const long own = src_cnt_owned[box] + trg_cnt_owned[box];
    long cost = -1;
    if (leaf[box] && !attr[box].Ghost && count[box] > 0 && own > 0) {
        long src = 0;
        for (int i = 0; i < count[box]; i++)
            src += src_cnt_halo[list1[box * stride + i]];
        cost = own * src;
    }
    keys[box] = cost + kDirectGroup;
}

/// After the sort: the work-list length and its shallowest level, the self-correction factor
/// parallel to the list, and the occupancy sums of the direct tuning key. The self-correction
/// factor must match the level the direct residual uses.
template <typename Real>
__global__ void k_direct_summary(const long *__restrict__ keys, const int *__restrict__ direct_work,
                                 const int *__restrict__ levels, const unsigned char *__restrict__ ifpwexp,
                                 const int *__restrict__ src_cnt_owned, const int *__restrict__ trg_cnt_owned,
                                 LevelTable<Real> w0, bool has_w0, long n_boxes, long *__restrict__ join,
                                 Real *__restrict__ self_corr) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long occ[4] = {};
    const bool work = i < n_boxes && keys[i] >= kDirectGroup;
    const int box = work ? direct_work[i] : 0;
    const unsigned n_work = __popc(__ballot_sync(FULL, work));
    const unsigned min_level = __reduce_min_sync(FULL, work ? (unsigned)levels[box] : ~0u);
    if (!(threadIdx.x & 31) && n_work) {
        atomicAdd(reinterpret_cast<unsigned long long *>(join + join_n_direct_work), (unsigned long long)n_work);
        atomicMin(reinterpret_cast<unsigned long long *>(join + join_min_direct_level), (unsigned long long)min_level);
    }
    if (work) {
        self_corr[i] = has_w0 ? w0.v[levels[box] + ifpwexp[box]] : Real{0};
        const long s = src_cnt_owned[box], t = trg_cnt_owned[box];
        occ[0] = s;
        occ[1] = s * s;
        occ[2] = t;
        occ[3] = t * t;
    }
    for (int k = 0; k < 4; ++k) {
        for (int o = 16; o; o >>= 1)
            occ[k] += __shfl_down_sync(FULL, occ[k], o);
        if (!(threadIdx.x & 31) && occ[k])
            atomicAdd(reinterpret_cast<unsigned long long *>(join + join_direct_src + k), (unsigned long long)occ[k]);
    }
}

// ---- shift_pw groups -------------------------------------------------------------------------

/// A shift group is the plane-wave eval boxes of one parent. Siblings are consecutive in each
/// level's eval list, and a level's first box never shares a parent with the previous level's
/// last, so a group starts wherever the parent changes. One flag per eval entry, 0 past the list.
__global__ void k_group_heads(const int *__restrict__ eval_flat, const sctl::Long *__restrict__ parent,
                              const long *__restrict__ eval_off, int n_levels, long n_boxes, int *__restrict__ heads) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i > n_boxes)
        return;
    heads[i] = i < eval_off[n_levels] && (i == 0 || parent[eval_flat[i]] != parent[eval_flat[i - 1]]);
}

/// From the scanned flags: each group's first eval entry, the list length after the last group,
/// and each level's first group in the join block.
__global__ void k_group_begin(const int *__restrict__ eval_flat, const sctl::Long *__restrict__ parent,
                              const long *__restrict__ eval_off, int n_levels, long n_boxes,
                              const int *__restrict__ gid, int *__restrict__ group_begin,
                              long *__restrict__ group_base) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i > n_boxes)
        return;
    const long n_eval = eval_off[n_levels];
    if (i < n_eval ? i == 0 || parent[eval_flat[i]] != parent[eval_flat[i - 1]] : i == n_eval)
        group_begin[gid[i]] = (int)i;
    if (i <= n_levels)
        group_base[i] = gid[eval_off[i]];
}

struct HaloArgs {
    const int *eval_flat;
    const sctl::Long *nbr;
    const unsigned char *leaf;
    const long *pw_out_offsets;
    const int *group_begin;
    const long *eval_off; ///< [n_levels+1] device, in the join block
    int n_levels;
    int n_groups;
};

/// The 3^DIM-1-k digits of a neighbour slot k, most significant first, minus one: the slot's
/// offset in the order the shift table is built.
template <int DIM>
__device__ void slot_offset(int k, int *o) {
    int ind = n_nbr_slots<DIM>() - 1 - k;
    for (int a = DIM - 1; a >= 0; --a) {
        o[a] = ind % 3 - 1;
        ind /= 3;
    }
}

constexpr int kHaloWarps = 4;

/// One warp per group. A member's position is where it sits in the first member's neighbour list,
/// with the frame then shifted so every member lands in {0,1}^DIM. The guards are the ones the
/// translation always had: a neighbour slot holding the box itself is only ever the centre one,
/// and leaf-leaf pairs belong to the direct pass.
template <int DIM>
__global__ void k_shift_halo(HaloArgs a, ShiftPwHalo *__restrict__ out) {
    constexpr int NN = n_nbr_slots<DIM>();
    __shared__ long s_off[kHaloWarps][64];
    __shared__ unsigned s_take[kHaloWarps][64];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = blockIdx.x * kHaloWarps + warp;
    if (g >= a.n_groups)
        return;
    const int begin = a.group_begin[g];
    const int n_mem = a.group_begin[g + 1] - begin;
    int level = 0;
    while (level + 1 < a.n_levels && a.eval_off[level + 1] <= begin)
        ++level;
    const sctl::Long box0 = a.eval_flat[begin];

    int rel[DIM] = {};
    if (lane < n_mem) {
        const sctl::Long box = a.eval_flat[begin + lane];
        for (int k = 0; k < NN; ++k)
            if (a.nbr[box0 * NN + k] == box) {
                slot_offset<DIM>(k, rel);
                break;
            }
    }
    int p[DIM];
    int slot = 0;
    for (int ax = 0; ax < DIM; ++ax) {
        int lo = lane < n_mem ? rel[ax] : 1;
        for (int o = 16; o; o >>= 1)
            lo = min(lo, __shfl_xor_sync(FULL, lo, o));
        p[ax] = rel[ax] - lo;
        slot = 2 * slot + p[ax];
    }

    for (int h = lane; h < 64; h += 32) {
        s_off[warp][h] = -1;
        s_take[warp][h] = 0;
    }
    __syncwarp();
    for (int base = 0; base < n_mem * NN; base += 32) {
        const int idx = base + lane;
        const int t = idx < n_mem * NN ? idx / NN : 0;
        const int t_slot = __shfl_sync(FULL, slot, t);
        int tp[DIM];
        for (int ax = 0; ax < DIM; ++ax)
            tp[ax] = __shfl_sync(FULL, p[ax], t);
        if (idx < n_mem * NN) {
            const int k = idx - t * NN;
            const sctl::Long box = a.eval_flat[begin + t];
            const sctl::Long nb = a.nbr[box * NN + k];
            const sctl::Long src = k == NN / 2 ? box : nb;
            const bool take = k == NN / 2
                                  ? a.pw_out_offsets[box] >= 0
                                  : nb >= 0 && nb != box && !(a.leaf[box] && a.leaf[nb]) && a.pw_out_offsets[nb] >= 0;
            if (take) {
                int o[DIM];
                slot_offset<DIM>(k, o);
                int h = 0;
                for (int ax = 0; ax < DIM; ++ax)
                    h = 4 * h + tp[ax] + o[ax] + 1;
                s_off[warp][h] = a.pw_out_offsets[src];
                atomicOr(&s_take[warp][h], 1u << t_slot);
            }
        }
    }
    __syncwarp();
    ShiftPwHalo &dst = out[g];
    for (int h = lane; h < 64; h += 32) {
        dst.pw_off[h] = s_off[warp][h];
        dst.take[h] = (unsigned char)s_take[warp][h];
    }
    if (lane < 8)
        dst.member[lane] = -1;
    __syncwarp();
    if (lane < n_mem)
        dst.member[slot] = begin + lane - (int)a.eval_off[level];
}

} // namespace

std::size_t scan_scratch_bytes(long n_boxes) { return (std::size_t)(n_boxes + 1) * sizeof(ScanIn); }
long n_chunks(long n_boxes) { return (n_boxes + 1 + BLOCK - 1) / BLOCK; }
int level_scratch_size(int n_levels) { return LevelScratch::size(n_levels); }

template <int DIM>
void box_status(const BoxStatusInputs<DIM> &in, BoxStatusOutputs &out) {
    const long n_boxes = in.n_boxes;
    if (!n_boxes)
        return;
    const long grid = (n_boxes + BLOCK - 1) / BLOCK;
    const std::size_t nb = (std::size_t)n_boxes;

    // Staging either direction through page-locked memory measured as a wash: it trades the
    // per-call cost for a host copy of the same bytes, and at one thread that copy is not free.
    const std::size_t b_up = 2 * nb * sizeof(sctl::Long);
    auto *const d_halo_in = static_cast<sctl::Long *>(dmk::cuda_helpers::pool_alloc(b_up, 0));
    auto *const d_trg_in = d_halo_in + nb;
    auto *const d_depth = static_cast<unsigned long long *>(dmk::cuda_helpers::pool_alloc(sizeof(long long), 0));
    DMK_CHECK_CUDA(
        cudaMemcpyAsync(d_halo_in, in.h_src_cnt_with_halo, nb * sizeof(sctl::Long), cudaMemcpyHostToDevice, 0));
    if (in.h_trg_cnt_owned)
        DMK_CHECK_CUDA(
            cudaMemcpyAsync(d_trg_in, in.h_trg_cnt_owned, nb * sizeof(sctl::Long), cudaMemcpyHostToDevice, 0));
    if (in.h_is_global_leaf)
        DMK_CHECK_CUDA(cudaMemcpyAsync(out.d_is_global_leaf, in.h_is_global_leaf, nb, cudaMemcpyHostToDevice, 0));

    auto *const d_cnt = reinterpret_cast<Counts3 *>(out.d_count_scan);
    PoolAlloc alloc;
    k_pack_counts<<<(n_boxes + 1 + BLOCK - 1) / BLOCK, BLOCK>>>(
        d_halo_in, in.h_trg_cnt_owned ? d_trg_in : nullptr, in.h_is_global_leaf ? nullptr : in.d_attr, n_boxes,
        in.owned_begin, in.owned_end, d_cnt, out.d_is_global_leaf, d_depth);
    thrust::exclusive_scan(policy(alloc), d_cnt, d_cnt + n_boxes + 1, d_cnt, Counts3{0, 0, 0}, Counts3Add());
    k_subtree_flags<DIM><<<grid, BLOCK>>>(in.d_mid, in.d_parent, in.d_child, in.d_nbr, d_cnt, out.d_is_global_leaf,
                                          n_boxes, out.d_src_counts_with_halo, out.d_src_counts_owned,
                                          out.d_trg_counts_owned, out.d_ifpwexp, d_depth);
    k_eval_flags_offsets<DIM><<<grid, BLOCK>>>(in.d_parent, in.d_child, out.d_ifpwexp, out.d_src_counts_owned,
                                               out.d_trg_counts_owned, d_cnt, n_boxes, in.charge_dof, in.n_tables_up,
                                               in.dof_src, in.dof_trg, out.d_iftensprodeval, out.d_offsets,
                                               out.d_zero, out.n_zero);
    DMK_CHECK_CUDA(cudaMemcpyAsync(out.h_max_depth, d_depth, sizeof(long), cudaMemcpyDeviceToHost, 0));
    DMK_CHECK_CUDA(cudaEventRecord(out.ready, 0));
    dmk::cuda_helpers::pool_free(d_depth, sizeof(long long), 0);
    dmk::cuda_helpers::pool_free(d_halo_in, b_up, 0);
}

template <typename Real, int DIM>
void compute_device_metadata(const DeviceMetadataInputs<Real, DIM> &in, DeviceMetadataOutputs<Real> &out) {
    const long n_boxes = in.n_boxes;
    out.n_boxes = n_boxes;
    out.n_levels_ = in.n_levels;
    if (!n_boxes)
        return;
    const int nl = in.n_levels;
    const long grid = (n_boxes + BLOCK - 1) / BLOCK;
    const long chunks = n_chunks(n_boxes);
    auto *const scan = static_cast<ScanIn *>(out.d_scan);
    PoolAlloc alloc;

    k_box_predicates<Real, DIM><<<chunks, BLOCK>>>(
        in.d_mid, in.d_child, in.d_ifpwexp, in.d_iftensprodeval, in.d_src_cnt_owned, in.d_trg_cnt_owned,
        in.d_src_cnt_global, n_boxes, nl, in.n_coeffs_up, in.n_coeffs_down, in.skip_root_form, out.d_centers,
        out.d_box_levels, scan, out.d_proxy_counts_up, out.d_sorted, out.d_sort_keys, out.d_is_c2p_center,
        out.d_is_tp_up_par, out.d_level_scratch, out.d_chunk_table, out.d_join);
    // Issued here and read only at AddData, well before the join; the destination is page-locked
    // so this does not drain the stream on the way out.
    DMK_CHECK_CUDA(cudaMemcpyAsync(out.h_proxy_counts_up, out.d_proxy_counts_up, n_boxes * sizeof(long),
                                   cudaMemcpyDeviceToHost, 0));
    thrust::exclusive_scan(policy(alloc), scan, scan + n_boxes + 1, scan, ScanIn{0, 0, 0, 0, 0, 0}, ScanAdd());
    k_chunk_scan<<<kMaxLevels * 2, BLOCK>>>(out.d_chunk_table, chunks,
                                            LevelScratch(out.d_level_scratch, nl).pw_tot);
    k_level_scan<DIM><<<1, BLOCK>>>(out.d_level_scratch, nl, scan, n_boxes, in.n_pw_per_box,
                                    out.d_join, out.d_proxy_totals);
    DMK_CHECK_CUDA(
        cudaMemcpyAsync(out.h_proxy_totals, out.d_proxy_totals, 2 * sizeof(long), cudaMemcpyDeviceToHost, 0));
    DMK_CHECK_CUDA(cudaEventRecord(out.proxy_ready, 0));

    k_box_fill<DIM><<<grid, BLOCK>>>(
        in.d_child, in.d_ifpwexp, in.d_iftensprodeval, in.d_src_cnt_owned, in.d_trg_cnt_owned, in.d_src_cnt_global,
        out.d_box_levels, out.d_is_c2p_center, n_boxes, nl, in.skip_root_form, in.n_pw_per_box, scan,
        out.d_chunk_table, out.d_level_scratch, out.d_join, out.d_proxy_off_up, out.d_proxy_off_down,
        out.d_c2p_center_boxes, out.d_c2p_levels, out.d_c2p_src_box_flat_offsets, out.d_c2p_n_src_boxes_per_group,
        out.d_c2p_src_boxes_flat, out.d_sort_keys, out.d_tp_parents, out.d_tp_children, out.d_tp_octants,
        out.d_tp_assign_dst, out.d_eval_targets, out.d_tp_up_src, out.d_tp_up_dst, out.d_tp_up_octants,
        out.d_tp_up_assign, out.d_tp_up_par, out.d_tp_up_par_begin, out.d_tp_up_par_count, out.d_pw_eval_flat,
        out.d_pw_form_flat, out.d_proxy_zero_boxes, out.d_pw_out_offsets);

    k_list1<Real, DIM><<<(n_boxes * 32 + BLOCK - 1) / BLOCK, BLOCK>>>(
        in.d_mid, in.d_attr, in.d_is_global_leaf, in.d_parent, in.d_child, in.d_nbr, in.d_src_cnt_with_halo,
        out.d_centers, n_boxes, in.nlist1_stride, in.periodic, out.d_list1, out.d_list1_count,
        in.periodic ? out.d_list1_shift : nullptr);
    k_direct_work_cost<<<grid, BLOCK>>>(in.d_attr, in.d_is_global_leaf, in.d_src_cnt_with_halo, in.d_src_cnt_owned,
                                        in.d_trg_cnt_owned, out.d_list1, out.d_list1_count, n_boxes, in.nlist1_stride,
                                        out.d_sort_keys);

    // Heaviest first, so a launch's early blocks carry the long tails. The direct keys sit above
    // every c2p key, so one sort orders both lists: [0, n) is the direct work list with the excluded
    // boxes past its end, [n, 2n) the group permutation.
    thrust::stable_sort_by_key(policy(alloc), out.d_sort_keys, out.d_sort_keys + 2 * n_boxes, out.d_sorted,
                               thrust::greater<long>());
    const bool has_w0 = in.self_mode != SelfCorrectionMode::zero;
    k_direct_summary<Real><<<grid, BLOCK>>>(out.d_sort_keys, out.d_sorted, out.d_box_levels, in.d_ifpwexp,
                                            in.d_src_cnt_owned, in.d_trg_cnt_owned, in.w0, has_w0, n_boxes, out.d_join,
                                            out.d_self_correction_work);

    { // shift_pw groups; their halos wait for the join to size their storage
        const JoinTables T(out.d_join, nl);
        k_group_heads<<<(n_boxes + 1 + BLOCK - 1) / BLOCK, BLOCK>>>(out.d_pw_eval_flat, in.d_parent, T.pw_eval_off, nl,
                                                                    n_boxes, out.d_group_heads);
        thrust::exclusive_scan(policy(alloc), out.d_group_heads, out.d_group_heads + n_boxes + 1, out.d_group_heads);
        k_group_begin<<<(n_boxes + 1 + BLOCK - 1) / BLOCK, BLOCK>>>(out.d_pw_eval_flat, in.d_parent, T.pw_eval_off, nl,
                                                                    n_boxes, out.d_group_heads, out.d_group_begin,
                                                                    T.group_base);
    }
}

template <typename Real>
void finish_proxy_metadata(const DeviceMetadataOutputs<Real> &out) {
    if (out.proxy_ready && out.n_boxes)
        DMK_CHECK_CUDA(cudaEventSynchronize(out.proxy_ready));
}

template <typename Real>
void finish_device_metadata(DeviceMetadataOutputs<Real> &out, int n_levels) {
    out.n_levels_ = n_levels;
    if (!out.n_boxes) {
        out.min_direct_level = n_levels;
        std::fill(out.h_join, out.h_join + join_size(n_levels), 0L);
        return;
    }
    DMK_CHECK_CUDA(
        cudaMemcpyAsync(out.h_join, out.d_join, join_size(n_levels) * sizeof(long), cudaMemcpyDeviceToHost, 0));
    DMK_CHECK_CUDA(cudaStreamSynchronize(0));
    const long *j = out.h_join;
    out.n_direct_work = (int)j[join_n_direct_work];
    out.min_direct_level = (int)j[join_min_direct_level];
    out.n_c2p_groups = (int)j[join_n_c2p_groups];
    out.n_eval_boxes = (int)j[join_n_eval];
    out.max_pw_form_per_level = (int)j[join_max_pw_form];
    out.n_proxy_zero_boxes = (int)j[join_n_zero];
    out.pw_out_size = j[join_pw_out_size];
    out.direct_occupancy_src = j[join_direct_src] ? (int)(j[join_direct_src_sq] / j[join_direct_src]) : 0;
    out.direct_occupancy_trg = j[join_direct_trg] ? (int)(j[join_direct_trg_sq] / j[join_direct_trg]) : 0;
    out.n_shift_groups = (int)out.h_table(5)[n_levels];
}

template <typename Real, int DIM>
void fill_shift_halos(const DeviceMetadataInputs<Real, DIM> &in, const DeviceMetadataOutputs<Real> &out,
                      ShiftPwHalo *d_halo) {
    if (!out.n_shift_groups)
        return;
    const HaloArgs a{out.d_pw_eval_flat,   in.d_nbr,          in.d_is_global_leaf,
                     out.d_pw_out_offsets, out.d_group_begin, JoinTables(out.d_join, in.n_levels).pw_eval_off,
                     in.n_levels,          out.n_shift_groups};
    k_shift_halo<DIM><<<(out.n_shift_groups + kHaloWarps - 1) / kHaloWarps, kHaloWarps * 32>>>(a, d_halo);
}

#define DMK_INSTANTIATE_METADATA(Real, DIM)                                                                            \
    template void compute_device_metadata<Real, DIM>(const DeviceMetadataInputs<Real, DIM> &,                          \
                                                     DeviceMetadataOutputs<Real> &);                                   \
    template void fill_shift_halos<Real, DIM>(const DeviceMetadataInputs<Real, DIM> &,                                 \
                                              const DeviceMetadataOutputs<Real> &, ShiftPwHalo *);

#define DMK_INSTANTIATE_FINISH(Real)                                                                                   \
    template void finish_device_metadata<Real>(DeviceMetadataOutputs<Real> &, int);                                    \
    template void finish_proxy_metadata<Real>(const DeviceMetadataOutputs<Real> &);
DMK_INSTANTIATE_FINISH(float)
DMK_INSTANTIATE_FINISH(double)

DMK_INSTANTIATE_METADATA(float, 2)
DMK_INSTANTIATE_METADATA(float, 3)
DMK_INSTANTIATE_METADATA(double, 2)
DMK_INSTANTIATE_METADATA(double, 3)

template void box_status<2>(const BoxStatusInputs<2> &, BoxStatusOutputs &);
template void box_status<3>(const BoxStatusInputs<3> &, BoxStatusOutputs &);

} // namespace dmk::cuda::pt
