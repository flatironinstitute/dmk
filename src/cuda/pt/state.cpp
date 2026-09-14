#include <dmk/cuda/pt/state.hpp>

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"

#include <dmk.h>
#include <dmk/cuda/helpers.hpp>
#include <dmk/direct.hpp>
#include <dmk/tree.hpp>

#include <cuda_runtime.h>
#include <sctl.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <string>
#include <utility>
#include <vector>

#include "gpu_tree_build.hpp"

namespace dmk::cuda::pt {

namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

// Charge/potential scatter helpers, JIT-compiled from pt/shared_state.cu. Fixed
// block, one launch each.
template <typename Real>
void launch_scatter_forward(const Real *in, Real *out, const long *scatter_index, long n_particles, int dof,
                            cudaStream_t stream) {
    if (n_particles == 0)
        return;
    constexpr int BLOCK = 256;
    static JitCache cache;
    JitKey key;
    key.name = "PtScatterForwardKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {{"BLOCK_SIZE", BLOCK}};
    auto kernel = cache.get_kernel_from_source(
        key, [&] { return make_stage_source("pt/shared_state.cu", key, "", "SharedState"); });
    const long grid = (n_particles + BLOCK - 1) / BLOCK;
    kernel->launch(dim3(grid, 1, 1), dim3(BLOCK, 1, 1), 0, stream, in, out, scatter_index, n_particles, dof);
}

template <typename Real>
void launch_scatter_forward_stresslet(const Real *densities, const Real *normals, Real *out, const long *scatter_index,
                                      long n_particles, int dim, cudaStream_t stream) {
    if (n_particles == 0)
        return;
    constexpr int BLOCK = 256;
    static JitCache cache;
    JitKey key;
    key.name = "PtScatterForwardStressletKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {{"BLOCK_SIZE", BLOCK}};
    auto kernel = cache.get_kernel_from_source(
        key, [&] { return make_stage_source("pt/shared_state.cu", key, "", "SharedState"); });
    const long grid = (n_particles + BLOCK - 1) / BLOCK;
    kernel->launch(dim3(grid, 1, 1), dim3(BLOCK, 1, 1), 0, stream, densities, normals, out, scatter_index, n_particles,
                   dim);
}

// Tree-order helpers for the path where the tree owns the particle scatter.
template <typename Real>
void launch_accumulate(Real *out, const Real *a, const Real *b, long n_values, cudaStream_t stream) {
    if (n_values == 0)
        return;
    constexpr int BLOCK = 256;
    static JitCache cache;
    JitKey key;
    key.name = "PtAccumulateKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {{"BLOCK_SIZE", BLOCK}};
    auto kernel = cache.get_kernel_from_source(
        key, [&] { return make_stage_source("pt/shared_state.cu", key, "", "SharedState"); });
    const long grid = (n_values + BLOCK - 1) / BLOCK;
    kernel->launch(dim3(grid, 1, 1), dim3(BLOCK, 1, 1), 0, stream, out, a, b, n_values);
}

template <typename Real>
void launch_outer_product(const Real *densities, const Real *normals, Real *out, long n_particles, int dim,
                          cudaStream_t stream) {
    if (n_particles == 0)
        return;
    constexpr int BLOCK = 256;
    static JitCache cache;
    JitKey key;
    key.name = "PtOuterProductKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {{"BLOCK_SIZE", BLOCK}};
    auto kernel = cache.get_kernel_from_source(
        key, [&] { return make_stage_source("pt/shared_state.cu", key, "", "SharedState"); });
    const long grid = (n_particles + BLOCK - 1) / BLOCK;
    kernel->launch(dim3(grid, 1, 1), dim3(BLOCK, 1, 1), 0, stream, densities, normals, out, n_particles, dim);
}

template <typename Real>
void launch_accumulate_and_scatter(Real *out, const Real *pot_eval, const Real *pot_extra, const long *scatter_index,
                                   int dof, long n_particles, cudaStream_t stream) {
    if (n_particles == 0)
        return;
    constexpr int BLOCK = 256;
    static JitCache cache;
    JitKey key;
    key.name = "PtAccumulateAndScatterKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {{"BLOCK_SIZE", BLOCK}};
    auto kernel = cache.get_kernel_from_source(
        key, [&] { return make_stage_source("pt/shared_state.cu", key, "", "SharedState"); });
    const long grid = (n_particles + BLOCK - 1) / BLOCK;
    kernel->launch(dim3(grid, 1, 1), dim3(BLOCK, 1, 1), 0, stream, out, pot_eval, pot_extra, scatter_index, dof,
                   n_particles);
}

// Upload any host container (std::vector or std::span) to a device buffer.
// Empty sources leave the buffer unallocated.
template <typename T, typename Src>
void up(DeviceBuffer<T> &d, const Src &s) {
    if (!s.empty())
        d.upload(s.data(), s.size());
}

// Batches the State constructor's transfers into one device allocation. Most of its forty-odd
// uploads are a few kilobytes, and at that size a transfer costs far more in per-call overhead than
// in bytes moved, so the small ones are staged through one page-locked buffer and cross together.
// Large entries are sent straight from their own storage: staging one costs about what its transfer
// does, and -- measurably worse -- evicts the host working set the next build's metadata pass needs.
class UploadBatch {
  public:
    template <typename T, typename Src>
    void add(DeviceBuffer<T> &dst, const Src &src) {
        if (src.empty())
            return;
        items_.push_back({reinterpret_cast<const char *>(src.data()), 0, src.size() * sizeof(T), src.size(), &dst,
                          +[](void *d, const char *p, std::size_t n) {
                              static_cast<DeviceBuffer<T> *>(d)->adopt(reinterpret_cast<const T *>(p), n);
                          }});
    }

    void flush(DeviceBuffer<char> &arena) {
        if (items_.empty())
            return;
        constexpr std::size_t kPackMax = 32 * 1024;
        const auto align = [](std::size_t x) { return (x + 255) & ~(std::size_t)255; }; // any element type

        // Staged entries first and contiguous, so they cross as one transfer.
        std::size_t total = 0, packed = 0;
        for (auto &it : items_)
            if (it.bytes < kPackMax) {
                it.off = total;
                total = align(total + it.bytes);
                packed = total;
            }
        for (auto &it : items_)
            if (it.bytes >= kPackMax) {
                it.off = total;
                total = align(total + it.bytes);
            }
        arena.resize(total);

        const auto pad = packed ? cuda_helpers::pinned_alloc(packed) : std::pair<char *, std::size_t>{nullptr, 0};
        for (const auto &it : items_) {
            if (pad.first && it.bytes < kPackMax)
                std::memcpy(pad.first + it.off, it.src, it.bytes);
            else
                DMK_CHECK_CUDA(cudaMemcpy(arena.data() + it.off, it.src, it.bytes, cudaMemcpyHostToDevice));
        }
        if (pad.first) {
            DMK_CHECK_CUDA(cudaMemcpy(arena.data(), pad.first, packed, cudaMemcpyHostToDevice));
            cuda_helpers::pinned_free(pad.first, pad.second);
        }
        for (const auto &it : items_)
            it.bind(it.dst, arena.data() + it.off, it.n);
    }

  private:
    struct Item {
        const char *src;
        std::size_t off, bytes, n;
        void *dst;
        void (*bind)(void *, const char *, std::size_t);
    };

    std::vector<Item> items_;
};

// Non-owning span helpers over the tree's flat host arrays. `long` matches
// sctl::Long's ABI, so the reinterpret_cast in long_span is safe.
template <typename Real, typename V>
std::span<const Real> real_span(const V &v) {
    return v.Dim() ? std::span<const Real>(&v[0], v.Dim()) : std::span<const Real>();
}
template <typename V>
std::span<const int> int_span(const V &v) {
    return v.Dim() ? std::span<const int>(&v[0], v.Dim()) : std::span<const int>();
}
template <typename V>
std::span<const long> long_span(const V &v) {
    return v.Dim() ? std::span<const long>(reinterpret_cast<const long *>(&v[0]), v.Dim()) : std::span<const long>();
}

// Per-level flat box list filtered by `pred`. Fills offset ([n_levels+1]),
// count ([n_levels]), running max, and the flattened box ids.
template <typename Real, int DIM, typename Pred>
void build_per_level_box_list(DMKPtTree<Real, DIM> &tree, int n_levels, std::vector<int> &offset,
                              std::vector<int> &count, int &max_per_level, std::vector<int> &flat, Pred pred) {
    offset.assign(n_levels + 1, 0);
    count.assign(n_levels, 0);
    flat.clear();
    flat.reserve(tree.n_boxes());
    for (int L = 0; L < n_levels; ++L) {
        offset[L] = flat.size();
        for (int idx = 0; idx < tree.level_indices[L].Dim(); ++idx) {
            const int b = tree.level_indices[L][idx];
            if (pred(b)) {
                flat.push_back(b);
                count[L]++;
            }
        }
        max_per_level = std::max(max_per_level, count[L]);
    }
    offset[n_levels] = flat.size();
}

template <typename Real, int DIM>
void build_charge2proxy_groups(BuildInputs<Real, DIM> &in, DMKPtTree<Real, DIM> &tree) {
    auto &w = in.worklists;
    w.n_c2p_groups = tree.charge2proxy_groups.size();
    w.c2p_center_boxes.reserve(w.n_c2p_groups);
    w.c2p_levels.reserve(w.n_c2p_groups);
    w.c2p_src_box_flat_offsets.reserve(w.n_c2p_groups);
    w.c2p_n_src_boxes_per_group.reserve(w.n_c2p_groups);
    w.c2p_src_boxes_flat.reserve((std::size_t)w.n_c2p_groups * 2);
    for (const auto &g : tree.charge2proxy_groups) {
        w.c2p_center_boxes.push_back(g.center_box);
        w.c2p_levels.push_back(g.level);
        w.c2p_src_box_flat_offsets.push_back(w.c2p_src_boxes_flat.size());
        w.c2p_n_src_boxes_per_group.push_back(g.n_src_boxes);
        for (int k = 0; k < g.n_src_boxes; ++k)
            w.c2p_src_boxes_flat.push_back(g.src_boxes[k]);
    }

    // Group ordering: heaviest source work first so heavy groups grab CTAs
    // early. Work key matches the device kernel's formula (CHUNK=32 tiebreaker).
    if (w.n_c2p_groups) {
        constexpr int CHUNK = 32;
        std::vector<std::pair<long long, int>> work_perm;
        work_perm.reserve(w.n_c2p_groups);
        for (int g = 0; g < w.n_c2p_groups; ++g) {
            long long total_sources = 0;
            long long total_chunks = 0;
            const auto &grp = tree.charge2proxy_groups[g];
            for (int sbi = 0; sbi < grp.n_src_boxes; ++sbi) {
                const int n_src = tree.src_counts_owned[grp.src_boxes[sbi]];
                total_sources += n_src;
                total_chunks += (n_src + CHUNK - 1) / CHUNK;
            }
            work_perm.emplace_back(total_sources * 1024LL + total_chunks, g);
        }
        std::sort(work_perm.begin(), work_perm.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        w.c2p_group_perm.resize(w.n_c2p_groups);
        w.n_c2p_active_groups = 0;
        for (int i = 0; i < w.n_c2p_groups; ++i) {
            w.c2p_group_perm[i] = work_perm[i].second;
            if (work_perm[i].first > 0)
                ++w.n_c2p_active_groups;
        }
    }
}

template <typename Real, int DIM>
void build_tp_up_pair_lists(BuildInputs<Real, DIM> &in, DMKPtTree<Real, DIM> &tree) {
    auto &w = in.worklists;
    const int n_levels = in.topology.n_levels;
    const auto &node_lists = tree.box_lists();
    constexpr int n_children = 1 << DIM;

    // build_charge2proxy_groups already ran, so take the centers from the groups that actually
    // launch rather than re-deriving the predicate that produced them.
    std::vector<char> is_c2p_center(tree.n_boxes(), 0);
    for (int i = 0; i < w.n_c2p_active_groups; ++i)
        is_c2p_center[w.c2p_center_boxes[w.c2p_group_perm[i]]] = 1;
    w.tp_up_offset.assign(n_levels + 1, 0);
    w.tp_up_count.assign(n_levels, 0);
    w.tp_up_par_offset.assign(n_levels + 1, 0);
    w.tp_up_par_count.assign(n_levels, 0);
    // At most one parent entry and one child entry per box, so this is an upper bound rather than a
    // guess; growing these by doubling is otherwise the bulk of the pass.
    const std::size_t n_box = tree.n_boxes();
    for (auto *v : {&w.tp_up_src, &w.tp_up_dst, &w.tp_up_octants, &w.tp_up_assign, &w.tp_up_par,
                    &w.tp_up_par_child_begin, &w.tp_up_par_child_count})
        v->reserve(n_box);
    std::vector<int> order, par, beg, cnt;
    for (int L = 0; L < n_levels; ++L) {
        w.tp_up_offset[L] = w.tp_up_src.size();
        w.tp_up_par_offset[L] = w.tp_up_par.size();
        for (int idx = 0; idx < tree.level_indices[L].Dim(); ++idx) {
            const int parent = tree.level_indices[L][idx];
            if (!(tree.src_counts_owned[parent] > 0 && tree.ifpwexp[parent]))
                continue;
            const int child_begin = static_cast<int>(w.tp_up_src.size());
            for (int ic = 0; ic < n_children; ++ic) {
                const int child = node_lists[parent].child[ic];
                if (child < 0)
                    continue;
                if (!(tree.src_counts_owned[child] > 0 && tree.ifpwexp[child]))
                    continue;
                w.tp_up_src.push_back(child);
                w.tp_up_dst.push_back(parent);
                w.tp_up_octants.push_back(ic);
                w.tp_up_assign.push_back(0);
                w.tp_up_count[L]++;
            }
            // The children just pushed are contiguous, which is what lets one block own
            // the parent and walk them without atomics.
            const int n_child = static_cast<int>(w.tp_up_src.size()) - child_begin;
            if (n_child == 0)
                continue;
            w.tp_up_par.push_back(parent);
            // Level-relative, to match the level-offset src_boxes the launcher passes.
            w.tp_up_par_child_begin.push_back(child_begin - w.tp_up_offset[L]);
            w.tp_up_par_child_count.push_back(n_child);
            w.tp_up_par_count[L]++;

            // charge2proxy writes this box first if it has a group; where it does not, the first
            // child owns the first write. Exactly one pair per parent may be marked -- a second
            // would assign over the first child's contribution.
            if (!is_c2p_center[parent])
                w.tp_up_assign[child_begin] = 1;
        }
        // Blocks carry 1..8 children, so order the level heaviest first and let the scheduler
        // absorb the imbalance early. Only the parent entries move; the child arrays they index
        // stay put. Matches what build_charge2proxy_groups does with its work permutation.
        const int par_lo = w.tp_up_par_offset[L], par_hi = static_cast<int>(w.tp_up_par.size());
        order.resize(par_hi - par_lo);
        for (int i = 0; i < static_cast<int>(order.size()); ++i)
            order[i] = par_lo + i;
        std::stable_sort(order.begin(), order.end(),
                         [&](int x, int y) { return w.tp_up_par_child_count[x] > w.tp_up_par_child_count[y]; });
        par.resize(order.size());
        beg.resize(order.size());
        cnt.resize(order.size());
        for (int i = 0; i < static_cast<int>(order.size()); ++i) {
            par[i] = w.tp_up_par[order[i]];
            beg[i] = w.tp_up_par_child_begin[order[i]];
            cnt[i] = w.tp_up_par_child_count[order[i]];
        }
        std::copy(par.begin(), par.end(), w.tp_up_par.begin() + par_lo);
        std::copy(beg.begin(), beg.end(), w.tp_up_par_child_begin.begin() + par_lo);
        std::copy(cnt.begin(), cnt.end(), w.tp_up_par_child_count.begin() + par_lo);

        w.max_tp_up_per_level = std::max(w.max_tp_up_per_level, w.tp_up_count[L]);
    }
    w.tp_up_offset[n_levels] = w.tp_up_src.size();
    w.tp_up_par_offset[n_levels] = w.tp_up_par.size();
}

} // namespace

template <typename Real, int DIM>
BuildInputs<Real, DIM> to_build_inputs(DMKPtTree<Real, DIM> &tree, GpuTreeMetadata<Real> *md) {
    BuildInputs<Real, DIM> in;
    in.device_metadata = md;

    sctl::Profile::Tic("tbi_topology", &tree.comm());
    // --- Topology ---
    auto &topo = in.topology;
    const int n_boxes = tree.n_boxes();
    const int n_levels = tree.n_levels();
    topo.n_boxes = n_boxes;
    topo.n_levels = n_levels;
    // One source of truth: the device kernel indexes the tree's own rows, so the stride must be
    // the tree's, not a second copy of the formula.
    topo.nlist1_stride = DMKPtTree<Real, DIM>::list1_stride();
    topo.n_neighbors = sctl::pow<DIM>(3);

    if (!md)
        topo.direct_work = std::span<const int>(tree.direct_work.data(), tree.direct_work.size());

    sctl::Profile::Tic("tbi_list1", &tree.comm());
    // list1 already sits in the tree in exactly this layout, one stride-nlist1_stride row per box.
    if (!md) {
        topo.list1_flat = tree.list1_flat();
        topo.list1_count = tree.nlist1();
    }
    // Under PBC a wrapped neighbor appears in list1 as its own box id, so the image shift
    // is the only thing distinguishing the entries: a single-level tree has the root
    // listed 3^DIM times with 3^DIM distinct shifts. boxsize[0] is 1, so shifts are +-1.
    if (tree.params.use_periodic && !md) {
        topo.list1_shift_flat.assign((std::size_t)n_boxes * topo.nlist1_stride * DIM, 0);
#pragma omp parallel for schedule(static)
        for (int b = 0; b < n_boxes; ++b) {
            const auto sh = tree.list1_shift(b);
            for (std::size_t k = 0; k < sh.size(); ++k)
                for (int d = 0; d < DIM; ++d)
                    topo.list1_shift_flat[((std::size_t)b * topo.nlist1_stride + k) * DIM + d] = sh[k][d];
        }
    }

    sctl::Profile::Toc();

    sctl::Profile::Tic("tbi_perbox", &tree.comm());
    const auto &node_mid = tree.box_mid();
    const auto &node_lists = tree.box_lists();
    topo.box_levels.resize(n_boxes);
    topo.ifpwexp.resize(n_boxes);
    topo.shift_nbr_offsets.resize(n_boxes + 1);

    // Per-box scalars and the shift-list prefilter in one sweep over the node lists. shift_pw's
    // neighbor loop rejected empty slots, self, leaf-leaf pairs, and neighbors without an outgoing
    // expansion; all four depend only on the tree, so they are resolved once here and the device
    // loop runs branch-free over survivors. Survivors go to a per-thread buffer and are stitched
    // afterwards: counting them first would mean a second pass over the same predicate, and the
    // neighbor ids are a strided read no cache keeps between passes.
    const int n_nbr_thread = std::max(1, omp_get_max_threads());
    std::vector<std::vector<ShiftPwNeighbor>> t_nbr(n_nbr_thread);
    std::vector<std::size_t> nbr_base(n_nbr_thread + 1, 0);
#pragma omp parallel num_threads(n_nbr_thread)
    {
        const int t = omp_get_thread_num();
        const int lo = (int)((long)n_boxes * t / n_nbr_thread);
        const int hi = (int)((long)n_boxes * (t + 1) / n_nbr_thread);
        auto &out = t_nbr[t];
        out.reserve((std::size_t)(hi - lo) * 8);
        for (int b = lo; b < hi; ++b) {
            topo.box_levels[b] = node_mid[b].Depth();
            topo.ifpwexp[b] = tree.ifpwexp[b] ? 1 : 0;
            const bool leaf_b = tree.is_global_leaf[b];
            const int n0 = (int)out.size();
            for (int k = 0; k < topo.n_neighbors; ++k) {
                const int nbr = node_lists[b].nbr[k];
                if (nbr < 0 || nbr == b)
                    continue;
                if (leaf_b && tree.is_global_leaf[nbr])
                    continue;
                const long pw_off = tree.pw_out_offsets[nbr];
                if (pw_off < 0)
                    continue;
                out.push_back({pw_off, topo.n_neighbors - 1 - k, 0});
            }
            topo.shift_nbr_offsets[b] = (int)out.size() - n0;
        }
#pragma omp barrier
#pragma omp single
        {
            for (int i = 0; i < n_nbr_thread; ++i)
                nbr_base[i + 1] = nbr_base[i] + t_nbr[i].size();
            topo.shift_nbr.resize(nbr_base[n_nbr_thread]);
            // Counts -> CSR offsets, with each thread's block starting at its own base.
            int at = 0;
            for (int i = 0; i < n_nbr_thread; ++i) {
                at = (int)nbr_base[i];
                const int blo = (int)((long)n_boxes * i / n_nbr_thread);
                const int bhi = (int)((long)n_boxes * (i + 1) / n_nbr_thread);
                for (int b = blo; b < bhi; ++b) {
                    const int n = topo.shift_nbr_offsets[b];
                    topo.shift_nbr_offsets[b] = at;
                    at += n;
                }
            }
            topo.shift_nbr_offsets[n_boxes] = (int)nbr_base[n_nbr_thread];
        }
        std::copy(out.begin(), out.end(), topo.shift_nbr.begin() + nbr_base[t]);
    }
    sctl::Profile::Toc();

    sctl::Profile::Toc();

    sctl::Profile::Tic("tbi_particles", &tree.comm());
    // --- Particles ---
    auto &part = in.particles;
    part.is_stresslet = tree.params.kernel == DMK_STRESSLET;
    part.r_src = real_span<Real>(tree.r_src_sorted_owned);
    part.r_trg = real_span<Real>(tree.r_trg_sorted_owned);
    part.src_counts = int_span(tree.src_counts_owned);
    part.trg_counts = int_span(tree.trg_counts_owned);
    part.r_src_offsets = long_span(tree.r_src_offsets_owned);
    part.r_trg_offsets = long_span(tree.r_trg_offsets_owned);
    part.scatter_index_src = long_span(tree.scatter_idx_src);
    part.scatter_index_trg = long_span(tree.scatter_idx_trg);
    part.n_src = tree.n_src_sorted;
    part.n_trg = tree.n_trg_sorted;
    part.d_r_src = tree.d_r_src_sorted;
    part.d_r_trg = tree.d_r_trg_sorted;
#ifdef DMK_GPU_OFFLOAD
    part.gpu_tree = tree.gpu_tree;
#endif
    if (part.is_stresslet) {
        part.charge_offsets = long_span(tree.density_offsets_with_halo);
        part.normal_offsets = long_span(tree.normal_offsets_with_halo);
        part.charge_outer_offsets = long_span(tree.charge_offsets_owned);
    } else {
        part.charge_offsets = long_span(tree.charge_offsets_owned);
    }

    sctl::Profile::Toc();

    if (md) {
        // The device metadata was only enqueued; everything above this point is host work that ran
        // behind it. Its two host-visible scalars are read from here on.
        sctl::Profile::Tic("tbi_md_finish", &tree.comm());
        gpu_tree_metadata_finish<Real, DIM>(tree.gpu_tree, *md);
        sctl::Profile::Toc();
    }

    sctl::Profile::Tic("tbi_fourier", &tree.comm());
    // --- Fourier + per-level geometry ---
    auto &fou = in.fourier;
    fou.n_pw = tree.expansion_constants.n_pw_diff;
    fou.n_pw2 = (fou.n_pw + 1) / 2;
    if constexpr (DIM == 3)
        fou.n_pw_modes = fou.n_pw * fou.n_pw * fou.n_pw2;
    else
        fou.n_pw_modes = fou.n_pw * fou.n_pw2;
    fou.n_charge_dim = tree.n_tables_down;
    fou.n_tables_up = tree.n_tables_up;
    fou.n_order = tree.expansion_constants.n_order;
    // Root plane-wave grid: the periodic root kernel lives on the reciprocal lattice and
    // uses n_pw_periodic modes, so the `_win` fields carry whichever the tree built.
    fou.n_pw_win =
        tree.params.use_periodic ? tree.expansion_constants.n_pw_periodic : tree.expansion_constants.n_pw_win;
    fou.n_pw2_win = (fou.n_pw_win + 1) / 2;
    if constexpr (DIM == 3)
        fou.n_pw_modes_win = fou.n_pw_win * fou.n_pw_win * fou.n_pw2_win;
    else
        fou.n_pw_modes_win = fou.n_pw_win * fou.n_pw2_win;
    fou.hpw_win = (Real)tree.expansion_constants.hpw_win;
    fou.n_digits = tree.n_digits;
    fou.beta = tree.expansion_constants.beta;
    fou.fparam = tree.params.fparam;

    fou.p2c = real_span<Real>(tree.p2c);
    fou.c2p = real_span<Real>(tree.c2p);
    fou.centers = real_span<Real>(tree.centers);
    fou.direct_rsc = real_span<Real>(tree.direct_rsc);
    fou.direct_cen = real_span<Real>(tree.direct_cen);
    fou.direct_d2max = real_span<Real>(tree.direct_d2max);

    // Yukawa's residual fit depends on lambda*boxsize, so it needs one polynomial per
    // level rather than a single scale-invariant pack. src_level spans [min target-box
    // depth, n_levels]; skipping shallower levels also avoids a fit that would throw
    // at large lambda.
    if (tree.params.kernel == DMK_YUKAWA) {
        int level0 = n_levels;
        if (md) {
            level0 = md->min_direct_level;
        } else {
            for (int b : topo.direct_work)
                level0 = std::min(level0, topo.box_levels[b]);
        }
        fou.direct_coeffs_level0 = level0;
        for (int L = level0; L <= n_levels; ++L) {
            const auto c = tree.fourier_data.local_correction_coeffs(L, tree.n_digits);
            fou.direct_coeffs_by_level.emplace_back(c.reg_poly.begin(), c.reg_poly.end());
        }
    }

    fou.pw2poly_per_level_reals = 2 * fou.n_pw * fou.n_order;
    fou.poly2pw_per_level_reals = 2 * fou.n_pw * fou.n_order;
    fou.radialft_per_level_reals = fou.n_pw_modes;
    fou.wpwshift_per_level_reals = 2 * topo.n_neighbors * fou.n_pw_modes;

    fou.pw2poly_flat.resize((std::size_t)n_levels * fou.pw2poly_per_level_reals);
    fou.poly2pw_flat.resize((std::size_t)n_levels * fou.poly2pw_per_level_reals);
    fou.radialft_flat.resize((std::size_t)n_levels * fou.radialft_per_level_reals);
    fou.wpwshift_flat.resize((std::size_t)n_levels * fou.wpwshift_per_level_reals);
    fou.inv_box_scale.resize(n_levels);
    fou.hpw_per_level.resize(n_levels);
    for (int L = 0; L < n_levels; ++L) {
        const auto &dfd = tree.difference_fourier_data[L];
        const Real *pw2poly = reinterpret_cast<const Real *>(&dfd.pw2poly[0]);
        const Real *poly2pw = reinterpret_cast<const Real *>(&dfd.poly2pw[0]);
        const Real *wpwshift = reinterpret_cast<const Real *>(&dfd.wpwshift[0]);
        std::copy(pw2poly, pw2poly + fou.pw2poly_per_level_reals,
                  &fou.pw2poly_flat[(std::size_t)L * fou.pw2poly_per_level_reals]);
        std::copy(poly2pw, poly2pw + fou.poly2pw_per_level_reals,
                  &fou.poly2pw_flat[(std::size_t)L * fou.poly2pw_per_level_reals]);
        std::copy(&dfd.radialft[0], &dfd.radialft[0] + fou.radialft_per_level_reals,
                  &fou.radialft_flat[(std::size_t)L * fou.radialft_per_level_reals]);
        std::copy(wpwshift, wpwshift + fou.wpwshift_per_level_reals,
                  &fou.wpwshift_flat[(std::size_t)L * fou.wpwshift_per_level_reals]);
        fou.inv_box_scale[L] = Real{2} / (Real)tree.boxsize[L];
        fou.hpw_per_level[L] = (Real)tree.expansion_constants.hpw_diff / (Real)tree.boxsize[L];
    }

    // The kernel FT is tabulated against integer radius j1^2+j2^2+j3^2 and gathered onto the mode
    // cube, so modes past the band edge fall outside the prolate window's support and the multiply
    // sets them to exactly zero. Taking the mask from the tabulated values rather than from the
    // ball geometry means kernels that build the FT by quadrature, with no hard gate, keep every
    // mode and simply see no pruning.
    //
    // Live modes move to the front of every slab so shift_pw and multiply walk a dense prefix.
    // Slab size and every pw offset stay as the tree built them; only the order within a slab
    // changes, and the dead tail is never read or written. DMK_PW_MASK=0 disables.
    const bool prune = [] {
        const char *v = std::getenv("DMK_PW_MASK");
        return !v || std::atoi(v) != 0;
    }();

    std::vector<char> live(fou.n_pw_modes, !prune);
    for (int m = 0; m < fou.n_pw_modes && prune; ++m)
        for (int L = 0; L < n_levels && !live[m]; ++L)
            live[m] = fou.radialft_flat[(std::size_t)L * fou.radialft_per_level_reals + m] != Real{0};

    // For a radial mask the live m1 at fixed (m2, m3) form a single contiguous run, which lets
    // proxy2pw and pw2proxy get a slot by arithmetic off an n_pw*n_pw2 table. Verify rather than
    // assume it: a gap anywhere makes that arithmetic wrong, so fall back to no pruning.
    const int n_pencil = fou.n_pw_modes / fou.n_pw;
    fou.pencil_slots.assign(2 * (std::size_t)n_pencil, 0);
    fou.full_of_compact.clear();
    bool contiguous = true;
    for (int p = 0; p < n_pencil && contiguous; ++p) {
        const char *row = &live[(std::size_t)p * fou.n_pw];
        int lo = 0;
        while (lo < fou.n_pw && !row[lo])
            ++lo;
        int hi = lo;
        while (hi < fou.n_pw && row[hi])
            ++hi;
        for (int i = hi; i < fou.n_pw; ++i)
            contiguous = contiguous && !row[i];

        fou.pencil_slots[2 * p] = (int)fou.full_of_compact.size() - lo;
        fou.pencil_slots[2 * p + 1] = lo | (hi << 16);
        for (int m1 = lo; m1 < hi; ++m1)
            fou.full_of_compact.push_back(p * fou.n_pw + m1);
    }

    const int n_live = contiguous ? (int)fou.full_of_compact.size() : fou.n_pw_modes;
    fou.n_pw_live = n_live;

    if (n_live == fou.n_pw_modes) {
        // Nothing to prune -- either the FT is nonzero everywhere, as for the quadrature-built
        // kernels, or a pencil had a gap. Drop the tables so the kernels take their cube-order
        // path; the permutation below would be a no-op.
        fou.pencil_slots.clear();
        fou.full_of_compact.clear();
    } else {
        // Permute the two per-mode tables into slab order. A wpwshift neighbour block is split
        // real-then-imaginary, so each half gathers separately.
        std::vector<Real> rad_new(fou.n_pw_modes);
        std::vector<Real> ws_new(fou.wpwshift_per_level_reals);
        for (int L = 0; L < n_levels; ++L) {
            Real *rad = &fou.radialft_flat[(std::size_t)L * fou.radialft_per_level_reals];
            std::fill(rad_new.begin(), rad_new.end(), Real{0});
            for (int c = 0; c < n_live; ++c)
                rad_new[c] = rad[fou.full_of_compact[c]];
            std::copy(rad_new.begin(), rad_new.end(), rad);

            Real *ws = &fou.wpwshift_flat[(std::size_t)L * fou.wpwshift_per_level_reals];
            std::fill(ws_new.begin(), ws_new.end(), Real{0});
            for (int ind = 0; ind < topo.n_neighbors; ++ind) {
                const Real *src = ws + (std::size_t)ind * 2 * fou.n_pw_modes;
                Real *dst = &ws_new[(std::size_t)ind * 2 * fou.n_pw_modes];
                for (int c = 0; c < n_live; ++c) {
                    dst[c] = src[fou.full_of_compact[c]];
                    dst[fou.n_pw_modes + c] = src[fou.n_pw_modes + fou.full_of_compact[c]];
                }
            }
            std::copy(ws_new.begin(), ws_new.end(), ws);
        }
    }

    // The proxy basis is radially compact to tolerance, so like the plane-wave grid its tensor
    // cube carries nothing in the corners. 110% of n_order-1 measured free (l2 unchanged to three
    // figures) across laplace/yukawa/stokeslet/stresslet, eps 1e-3..1e-6, both precisions and
    // n_order 9..21, while dropping ~35% of the coefficients at every one of those orders; 100%
    // costs up to 50% more error. Only charge2proxy exploits it: eval_targets' inner block is
    // tight enough that any control flow to skip dead coefficients costs more than it saves.
    // DMK_PROXY_BALL overrides the percentage, 0 disables.
    const int ball_pct = [] {
        const char *v = std::getenv("DMK_PROXY_BALL");
        return v ? std::atoi(v) : 110;
    }();
    fou.proxy_ball_r2 = 0;
    if (ball_pct > 0 && fou.n_order > 0) {
        const double r = (fou.n_order - 1) * ball_pct / 100.0;
        fou.proxy_ball_r2 = (int)(r * r);
    }

    if (fou.n_pw_win) {
        const auto &wfd = tree.window_fourier_data;
        const Real *pw2poly = reinterpret_cast<const Real *>(&wfd.pw2poly[0]);
        const Real *poly2pw = reinterpret_cast<const Real *>(&wfd.poly2pw[0]);
        fou.window_pw2poly.assign(pw2poly, pw2poly + 2 * fou.n_pw_win * fou.n_order);
        fou.window_poly2pw.assign(poly2pw, poly2pw + 2 * fou.n_pw_win * fou.n_order);
        fou.window_radialft.assign(&wfd.radialft[0], &wfd.radialft[0] + fou.n_pw_modes_win);
    }

    sctl::Profile::Toc();

    sctl::Profile::Tic("tbi_worklists", &tree.comm());
    // --- Worklists ---
    auto &w = in.worklists;
    sctl::Profile::Tic("wl_c2p");
    build_charge2proxy_groups(in, tree);
    sctl::Profile::Toc();
    sctl::Profile::Tic("wl_tpup");
    build_tp_up_pair_lists(in, tree);
    sctl::Profile::Toc();
    sctl::Profile::Tic("wl_lists");

    w.tp_offset.assign(n_levels + 1, 0);
    w.tp_count.assign(n_levels, 0);
    for (auto *v : {&w.tp_parents, &w.tp_children, &w.tp_octants, &w.tp_assign_dst})
        v->reserve(tree.n_boxes());
    for (int L = 0; L < n_levels; ++L) {
        w.tp_offset[L] = w.tp_parents.size();
        for (const auto &p : tree.tensorprod_pairs_per_level[L]) {
            w.tp_parents.push_back(p.parent);
            w.tp_children.push_back(p.child);
            w.tp_octants.push_back(p.child_octant);
            // pw2proxy covers a box iff ifpwexp && (src+trg) > 0, and the child already
            // satisfies the latter here, so without ifpwexp this pair is its only writer.
            w.tp_assign_dst.push_back(tree.ifpwexp[p.child] ? 0 : 1);
            w.tp_count[L]++;
        }
        w.max_tp_per_level = std::max(w.max_tp_per_level, w.tp_count[L]);
    }
    w.tp_offset[n_levels] = w.tp_parents.size();

    // pw_eval per-level max is unused; discard it.
    int pw_eval_max_discard = 0;
    build_per_level_box_list(
        tree, n_levels, w.pw_eval_box_offset, w.pw_eval_box_count, pw_eval_max_discard, w.pw_eval_box_flat,
        [&](int b) { return tree.ifpwexp[b] && (tree.src_counts_owned[b] + tree.trg_counts_owned[b]) > 0; });

    // Under periodic the level-0 difference kernel is skipped: the periodic root kernel
    // already contains W_0+D_0, so pw_out(0) must stay zero.
    const bool skip_root_form = tree.params.use_periodic;
    build_per_level_box_list(
        tree, n_levels, w.pw_form_box_offset, w.pw_form_box_count, w.max_pw_form_per_level, w.pw_form_box_flat,
        [&](int b) { return tree.ifpwexp[b] && tree.proxy_coeffs_offsets[b] != -1 && !(skip_root_form && b == 0); });

    // Nothing zeroes the upward buffer, so any box read without a writer that assigns first would
    // carry the previous eval's values. Take the difference outright rather than re-deriving a
    // predicate: read set is the proxy2pw list plus the root, which form_outgoing launches
    // unconditionally and which the list omits under PBC. Written set is the launched
    // charge2proxy centers plus every tensorprod parent. Expected to come out empty on the
    // current tree, but it is the check that keeps that true.
    {
        std::vector<char> written(tree.n_boxes(), 0);
        for (int i = 0; i < w.n_c2p_active_groups; ++i)
            written[w.c2p_center_boxes[w.c2p_group_perm[i]]] = 1;
        for (int b : w.tp_up_par)
            written[b] = 1;

        std::vector<char> read_set(tree.n_boxes(), 0);
        for (int b : w.pw_form_box_flat)
            read_set[b] = 1;
        if (tree.proxy_coeffs_offsets[0] != -1)
            read_set[0] = 1;

        for (int b = 0; b < static_cast<int>(tree.n_boxes()); ++b)
            if (read_set[b] && !written[b])
                w.proxy_zero_boxes.push_back(b);
    }

    // Merge each level's box list into groups and union their source sets. Consecutive
    // entries are Morton-ordered, hence spatially adjacent, so their 3^DIM stencils
    // overlap: a group of 8 spans a 4x4x4 region of sources instead of 8 separate 3x3x3
    // ones, and shift_pw fetches each source's plane-wave slab once rather than once per
    // target. That slab traffic is the whole cost of the kernel.
    sctl::Profile::Toc();
    sctl::Profile::Tic("wl_shiftgroup");
    const int shift_group = shift_group_size(fou.n_charge_dim);
    // Enumerate the groups first: each one's union reads only its own members' neighbour lists, so
    // with the group list in hand the unions are independent and run in parallel. Every insertion
    // rescans the group's entries linearly, which made this the largest single cost in
    // to_build_inputs.
    struct GroupSpan {
        int box_off, g0, n_mem;
    };
    std::vector<GroupSpan> groups;
    groups.reserve(w.pw_eval_box_flat.size() / shift_group + n_levels);
    w.shift_group_base.assign(n_levels, 0);
    for (int L = 0; L < n_levels; ++L) {
        w.shift_group_base[L] = static_cast<int>(groups.size());
        const int n_box = w.pw_eval_box_count[L];
        const int box_off = w.pw_eval_box_offset[L];
        for (int g0 = 0; g0 < n_box; g0 += shift_group)
            groups.push_back({box_off, g0, std::min(shift_group, n_box - g0)});
    }

    const int n_grp = static_cast<int>(groups.size());
    const int n_thread = std::max(1, omp_get_max_threads());
    std::vector<std::vector<ShiftPwGroupSrc>> t_src(n_thread);
    std::vector<std::vector<int>> t_count(n_thread);
#pragma omp parallel num_threads(n_thread)
    {
        const int t = omp_get_thread_num();
        // Contiguous group ranges, assigned by hand: the flat output has to come out in group
        // order, so which groups a thread owns cannot be left to the schedule.
        const int lo = static_cast<int>((long)n_grp * t / n_thread);
        const int hi = static_cast<int>((long)n_grp * (t + 1) / n_thread);
        auto &out = t_src[t];
        auto &cnt = t_count[t];
        out.reserve((std::size_t)(hi - lo) * 80);
        cnt.reserve(hi - lo);
        for (int g = lo; g < hi; ++g) {
            const auto &grp = groups[g];
            const int grp_begin = static_cast<int>(out.size());
            for (int m = 0; m < grp.n_mem; ++m) {
                const int box = w.pw_eval_box_flat[grp.box_off + grp.g0 + m];
                for (int e = topo.shift_nbr_offsets[box]; e < topo.shift_nbr_offsets[box + 1]; ++e) {
                    const ShiftPwNeighbor &nb = topo.shift_nbr[e];
                    // Linear over this group's entries only; the union is ~64 wide. Merging
                    // requires the slot to be free for *this* member, not just to match the
                    // source: under PBC one box wraps into several neighbor slots of the same
                    // target with different shifts, and collapsing those would drop a term.
                    int slot = -1;
                    for (int q = grp_begin; q < static_cast<int>(out.size()); ++q)
                        if (out[q].pw_off == nb.pw_off && out[q].shift_ind[m] < 0) {
                            slot = q;
                            break;
                        }
                    if (slot < 0) {
                        slot = static_cast<int>(out.size());
                        ShiftPwGroupSrc gs;
                        gs.pw_off = nb.pw_off;
                        for (int u = 0; u < kShiftGroupMax; ++u)
                            gs.shift_ind[u] = -1;
                        out.push_back(gs);
                    }
                    out[slot].shift_ind[m] = static_cast<signed char>(nb.shift_ind);
                }
            }
            cnt.push_back(static_cast<int>(out.size()) - grp_begin);
        }
    }

    std::vector<std::size_t> t_base(n_thread + 1, 0);
    for (int t = 0; t < n_thread; ++t)
        t_base[t + 1] = t_base[t] + t_src[t].size();
    w.shift_group_offsets.clear();
    w.shift_group_offsets.reserve(n_grp + 1);
    for (int t = 0; t < n_thread; ++t) {
        std::size_t at = t_base[t];
        for (const int c : t_count[t]) {
            w.shift_group_offsets.push_back(static_cast<int>(at));
            at += c;
        }
    }
    w.shift_group_src.resize(t_base[n_thread]);
#pragma omp parallel for schedule(static, 1) num_threads(n_thread)
    for (int t = 0; t < n_thread; ++t)
        std::copy(t_src[t].begin(), t_src[t].end(), w.shift_group_src.begin() + t_base[t]);
    w.shift_group_offsets.push_back(static_cast<int>(w.shift_group_src.size()));

    sctl::Profile::Toc();
    w.pw_in_pool_base.assign(n_levels, 0);
    long total_slots = 0;
    for (int L = 0; L < n_levels; ++L) {
        w.pw_in_pool_base[L] = total_slots;
        total_slots += w.pw_eval_box_count[L];
    }

    w.eval_targets_box_list = tree.eval_targets_box_list;
    if (!md)
        w.self_correction_work = tree.self_correction_work;

    sctl::Profile::Toc();

    sctl::Profile::Tic("tbi_scratch", &tree.comm());
    // --- Scratch strides / sizes ---
    auto &sc = in.scratch;
    sc.tensorprod_scratch_stride_reals = 2L * fou.n_order * fou.n_order * fou.n_order; // 2 * n_order^3 ping-pong slab
    sc.pw_in_stride_reals = 2L * fou.n_charge_dim * fou.n_pw_modes;
    sc.pw_form_stride_reals = (fou.n_tables_up != fou.n_charge_dim) ? 2L * fou.n_tables_up * fou.n_pw_modes : 0;
    sc.proxy_coeffs_upward_dim = tree.proxy_coeffs_upward_size;
    sc.proxy_coeffs_downward_dim = tree.proxy_coeffs_downward_size;
    sc.pw_out_dim = tree.pw_out_size; // sized by init_planewave_data (called before to_build_inputs)
    sc.proxy_offsets_upward = long_span(tree.proxy_coeffs_offsets);
    sc.proxy_offsets_downward = long_span(tree.proxy_coeffs_offsets_downward);
    sc.pw_out_offsets = long_span(tree.pw_out_offsets);

    sctl::Profile::Toc();

    sctl::Profile::Tic("tbi_outputs", &tree.comm());
    // --- Outputs ---
    auto &out = in.outputs;
    out.kernel = tree.params.kernel;
    out.eval_src = tree.params.eval_src;
    out.eval_trg = tree.params.eval_trg;
    out.pot_src_dof = tree.kernel_output_dim_src;
    out.pot_trg_dof = tree.kernel_output_dim_trg;
    out.pot_src_size = in.particles.n_src * out.pot_src_dof;
    out.pot_trg_size = in.particles.n_trg * out.pot_trg_dof;
    out.pot_src_offsets = long_span(tree.pot_src_offsets);
    out.pot_trg_offsets = long_span(tree.pot_trg_offsets);
    sctl::Profile::Toc();

    return in;
}

template <typename Real, int DIM>
State<Real, DIM>::State(const BuildInputs<Real, DIM> &in) {
    UploadBatch batch;
    kernel = in.outputs.kernel;
    n_boxes = in.topology.n_boxes;
    n_levels = in.topology.n_levels;

    sctl::Profile::Tic("sc_topology");
    // --- Topology ---
    topology.nlist1_stride = in.topology.nlist1_stride;
    topology.n_neighbors = in.topology.n_neighbors;
    if (const auto *md = in.device_metadata) {
        // Derived on the device from the tree's own node lists, and owned by the tree, which
        // outlives this State. box_levels and ifpwexp still come from the host: metadata routines
        // that have not moved yet read them there.
        const std::size_t nb = in.topology.n_boxes;
        const std::size_t rows = nb * in.topology.nlist1_stride;
        topology.d_direct_work.adopt(md->d_direct_work, md->n_direct_work);
        topology.d_list1_flat.adopt(md->d_list1_flat, rows);
        topology.d_list1_count.adopt(md->d_list1_count, nb);
        if (md->d_list1_shift)
            topology.d_list1_shift.adopt(md->d_list1_shift, rows * DIM);
    } else {
        batch.add(topology.d_direct_work, in.topology.direct_work);
        batch.add(topology.d_list1_flat, in.topology.list1_flat);
        batch.add(topology.d_list1_count, in.topology.list1_count);
        batch.add(topology.d_list1_shift, in.topology.list1_shift_flat);
    }
    batch.add(topology.d_box_levels, in.topology.box_levels);
    batch.add(topology.d_ifpwexp, in.topology.ifpwexp);

    sctl::Profile::Toc();
    sctl::Profile::Tic("sc_particles");
    // --- Particles ---
    const auto &pi = in.particles;
    particles.n_src = pi.n_src;
    particles.n_trg = pi.n_trg;
    particles.gpu_tree = pi.gpu_tree;
    // Upload the coordinates only if the tree did not already leave them on the device.
    if (pi.d_r_src) {
        particles.r_src_ptr = pi.d_r_src;
    } else {
        up(particles.d_r_src, pi.r_src);
        particles.r_src_ptr = particles.d_r_src.data();
    }
    if (pi.d_r_trg) {
        particles.r_trg_ptr = pi.d_r_trg;
    } else {
        up(particles.d_r_trg, pi.r_trg);
        particles.r_trg_ptr = particles.d_r_trg.data();
    }
    batch.add(particles.d_src_counts, pi.src_counts);
    batch.add(particles.d_trg_counts, pi.trg_counts);
    batch.add(particles.d_r_src_offsets, pi.r_src_offsets);
    batch.add(particles.d_r_trg_offsets, pi.r_trg_offsets);
    batch.add(particles.d_charge_offsets, pi.charge_offsets);
    batch.add(particles.d_scatter_index_src, pi.scatter_index_src);
    batch.add(particles.d_scatter_index_trg, pi.scatter_index_trg);
    if (pi.is_stresslet) {
        batch.add(particles.d_normal_offsets, pi.normal_offsets);
        batch.add(particles.d_charge_outer_offsets, pi.charge_outer_offsets);
    }

    sctl::Profile::Toc();
    sctl::Profile::Tic("sc_fourier");
    // --- Fourier (scalars mirror BuildInputs; buffers uploaded verbatim) ---
    const auto &fi = in.fourier;
    fourier.n_pw = fi.n_pw;
    fourier.n_pw2 = fi.n_pw2;
    fourier.n_pw_modes = fi.n_pw_modes;
    fourier.n_charge_dim = fi.n_charge_dim;
    fourier.n_tables_up = fi.n_tables_up;
    fourier.n_order = fi.n_order;
    fourier.n_pw_win = fi.n_pw_win;
    fourier.n_pw2_win = fi.n_pw2_win;
    fourier.n_pw_modes_win = fi.n_pw_modes_win;
    fourier.hpw_win = fi.hpw_win;
    fourier.hpw_per_level = fi.hpw_per_level;
    fourier.direct_coeffs_by_level = fi.direct_coeffs_by_level;
    fourier.direct_coeffs_level0 = fi.direct_coeffs_level0;
    fourier.n_digits = fi.n_digits;
    fourier.beta = fi.beta;
    fourier.fparam = fi.fparam;
    fourier.pw2poly_per_level_reals = fi.pw2poly_per_level_reals;
    fourier.poly2pw_per_level_reals = fi.poly2pw_per_level_reals;
    fourier.radialft_per_level_reals = fi.radialft_per_level_reals;
    fourier.wpwshift_per_level_reals = fi.wpwshift_per_level_reals;
    batch.add(fourier.d_pw2poly_flat, fi.pw2poly_flat);
    batch.add(fourier.d_poly2pw_flat, fi.poly2pw_flat);
    batch.add(fourier.d_radialft_flat, fi.radialft_flat);
    batch.add(fourier.d_wpwshift_flat, fi.wpwshift_flat);
    batch.add(fourier.d_pencil_slots, fi.pencil_slots);
    fourier.proxy_ball_r2 = fi.proxy_ball_r2;
    batch.add(fourier.d_full_of_compact, fi.full_of_compact);
    fourier.n_pw_live = fi.n_pw_live;
    batch.add(fourier.d_window_pw2poly, fi.window_pw2poly);
    batch.add(fourier.d_window_poly2pw, fi.window_poly2pw);
    batch.add(fourier.d_window_radialft, fi.window_radialft);
    batch.add(fourier.d_p2c, fi.p2c);
    batch.add(fourier.d_c2p, fi.c2p);
    // Box centers come from the device metadata when the tree derived them there; the host array is
    // then never built.
    if (in.device_metadata)
        fourier.d_centers.adopt(in.device_metadata->d_centers, (std::size_t)n_boxes * DIM);
    else
        batch.add(fourier.d_centers, fi.centers);
    batch.add(fourier.d_inv_box_scale, fi.inv_box_scale);
    batch.add(fourier.d_direct_rsc, fi.direct_rsc);
    batch.add(fourier.d_direct_cen, fi.direct_cen);
    batch.add(fourier.d_direct_d2max, fi.direct_d2max);

    sctl::Profile::Toc();
    sctl::Profile::Tic("sc_worklists");
    // --- Worklists ---
    const auto &wi = in.worklists;
    worklists.n_c2p_groups = wi.n_c2p_groups;
    worklists.n_c2p_active_groups = wi.n_c2p_active_groups;
    batch.add(worklists.d_c2p_center_boxes, wi.c2p_center_boxes);
    batch.add(worklists.d_c2p_levels, wi.c2p_levels);
    batch.add(worklists.d_c2p_src_box_flat_offsets, wi.c2p_src_box_flat_offsets);
    batch.add(worklists.d_c2p_n_src_boxes_per_group, wi.c2p_n_src_boxes_per_group);
    batch.add(worklists.d_c2p_src_boxes_flat, wi.c2p_src_boxes_flat);
    batch.add(worklists.d_c2p_group_perm, wi.c2p_group_perm);
    batch.add(worklists.d_tp_parents, wi.tp_parents);
    batch.add(worklists.d_tp_children, wi.tp_children);
    batch.add(worklists.d_tp_octants, wi.tp_octants);
    batch.add(worklists.d_tp_assign_dst, wi.tp_assign_dst);
    batch.add(worklists.d_tp_up_src_boxes, wi.tp_up_src);
    batch.add(worklists.d_tp_up_dst_boxes, wi.tp_up_dst);
    batch.add(worklists.d_tp_up_octants, wi.tp_up_octants);
    batch.add(worklists.d_tp_up_assign, wi.tp_up_assign);
    batch.add(worklists.d_proxy_zero_boxes, wi.proxy_zero_boxes);
    worklists.n_proxy_zero_boxes = static_cast<int>(wi.proxy_zero_boxes.size());
    batch.add(worklists.d_tp_up_par, wi.tp_up_par);
    batch.add(worklists.d_tp_up_par_child_begin, wi.tp_up_par_child_begin);
    batch.add(worklists.d_tp_up_par_child_count, wi.tp_up_par_child_count);
    batch.add(worklists.d_pw_eval_box_flat, wi.pw_eval_box_flat);
    batch.add(worklists.d_shift_group_src, wi.shift_group_src);
    batch.add(worklists.d_shift_group_offsets, wi.shift_group_offsets);
    worklists.shift_group_base_h = wi.shift_group_base;
    batch.add(worklists.d_pw_form_box_flat, wi.pw_form_box_flat);
    worklists.n_eval_boxes = static_cast<int>(wi.eval_targets_box_list.size());
    batch.add(worklists.d_eval_targets_box_list, wi.eval_targets_box_list);
    if (const auto *md = in.device_metadata)
        worklists.d_self_correction_work.adopt(md->d_self_correction_work, md->n_direct_work);
    else
        batch.add(worklists.d_self_correction_work, wi.self_correction_work);
    worklists.pw_in_pool_base_h = wi.pw_in_pool_base;
    worklists.tp_offset_h = wi.tp_offset;
    worklists.tp_count_h = wi.tp_count;
    worklists.tp_up_offset_h = wi.tp_up_offset;
    worklists.tp_up_count_h = wi.tp_up_count;
    worklists.tp_up_par_offset_h = wi.tp_up_par_offset;
    worklists.tp_up_par_count_h = wi.tp_up_par_count;
    worklists.pw_eval_box_offset_h = wi.pw_eval_box_offset;
    worklists.pw_eval_box_count_h = wi.pw_eval_box_count;
    worklists.pw_form_box_offset_h = wi.pw_form_box_offset;
    worklists.pw_form_box_count_h = wi.pw_form_box_count;

    sctl::Profile::Toc();
    sctl::Profile::Tic("sc_scratch");
    // --- Scratch (buffers sized/zeroed here; contents produced by the passes) ---
    const auto &si = in.scratch;
    scratch.tensorprod_scratch_stride_reals = si.tensorprod_scratch_stride_reals;
    scratch.pw_in_stride_reals = si.pw_in_stride_reals;
    scratch.pw_form_stride_reals = si.pw_form_stride_reals;
    if (si.proxy_coeffs_upward_dim) {
        scratch.d_proxy_coeffs_upward.resize(si.proxy_coeffs_upward_dim);
        scratch.d_proxy_coeffs_upward.zero_async();
    }
    if (si.proxy_coeffs_downward_dim) {
        scratch.d_proxy_coeffs_downward.resize(si.proxy_coeffs_downward_dim);
        scratch.d_proxy_coeffs_downward.zero_async();
    }
    batch.add(scratch.d_proxy_offsets_upward, si.proxy_offsets_upward);
    batch.add(scratch.d_proxy_offsets_downward, si.proxy_offsets_downward);
    batch.add(scratch.d_pw_out_offsets, si.pw_out_offsets);
    scratch.d_pw_out.resize(2 * si.pw_out_dim);

    const int max_tp_any = std::max(wi.max_tp_per_level, wi.max_tp_up_per_level);
    if (max_tp_any && scratch.tensorprod_scratch_stride_reals)
        scratch.d_tensorprod_scratch.resize((std::size_t)max_tp_any * scratch.tensorprod_scratch_stride_reals);

    long total_slots = 0;
    for (int L = 0; L < n_levels; ++L)
        total_slots += wi.pw_eval_box_count[L];
    if (total_slots && scratch.pw_in_stride_reals)
        scratch.d_pw_in_pool.resize((std::size_t)total_slots * scratch.pw_in_stride_reals);

    const bool split_up_down = fourier.n_tables_up != fourier.n_charge_dim;
    if (split_up_down && wi.max_pw_form_per_level && scratch.pw_form_stride_reals)
        scratch.d_pw_form_pool.resize((std::size_t)wi.max_pw_form_per_level * scratch.pw_form_stride_reals);

    if (fourier.n_pw_modes_win) {
        scratch.d_window_pw_form_in.resize(2 * (std::size_t)fourier.n_tables_up * fourier.n_pw_modes_win);
        if (split_up_down)
            scratch.d_window_pw_form_out.resize(2 * (std::size_t)fourier.n_charge_dim * fourier.n_pw_modes_win);
    }
    const int zero_int = 0;
    scratch.d_box0_id.upload(&zero_int, 1);

    sctl::Profile::Toc();
    sctl::Profile::Tic("sc_outputs");
    // --- Outputs ---
    outputs.eval_src = in.outputs.eval_src;
    outputs.eval_trg = in.outputs.eval_trg;
    outputs.pot_src_dof = in.outputs.pot_src_dof;
    outputs.pot_trg_dof = in.outputs.pot_trg_dof;
    outputs.pot_src_size = in.outputs.pot_src_size;
    outputs.pot_trg_size = in.outputs.pot_trg_size;
    batch.add(outputs.d_pot_src_offsets, in.outputs.pot_src_offsets);
    batch.add(outputs.d_pot_trg_offsets, in.outputs.pot_trg_offsets);
    outputs.d_pot_direct_src.resize(outputs.pot_src_size);
    outputs.d_pot_direct_trg.resize(outputs.pot_trg_size);
    outputs.d_pot_eval_src.resize(outputs.pot_src_size);
    outputs.d_pot_eval_trg.resize(outputs.pot_trg_size);
    if (particles.gpu_tree) {
        // The tree maps these back to the caller's order in desort_potentials, so no user-order
        // device buffer is needed here at all.
        if (outputs.pot_src_size)
            outputs.pot_src_tree =
                gpu_tree_reserve_data<Real, DIM>(particles.gpu_tree, "pdmk_pot_src", "pdmk_src", outputs.pot_src_dof);
        if (outputs.pot_trg_size)
            outputs.pot_trg_tree =
                gpu_tree_reserve_data<Real, DIM>(particles.gpu_tree, "pdmk_pot_trg", "pdmk_trg", outputs.pot_trg_dof);
    } else {
        outputs.d_pot_src_final.resize(outputs.pot_src_size);
        outputs.d_pot_trg_final.resize(outputs.pot_trg_size);
    }

    batch.flush(upload_arena);

    direct_stream = cuda_helpers::DeviceStream::non_blocking();
    // The near-field kernel's blocks would otherwise hold every SM until they tail off, starving
    // this chain for most of the eval even though it is enqueued at the same time.
    downward_stream = cuda_helpers::DeviceStream::non_blocking_priority();

    scratch.d_pw_out.zero_async(downward_stream.get());
    // shift_pw writes only the live prefix of each slab, so the dead tail must read zero and stay
    // that way: nothing else ever writes this pool.
    scratch.d_pw_in_pool.zero_async(downward_stream.get());
    outputs.d_pot_eval_src.zero_async(downward_stream.get());
    outputs.d_pot_eval_trg.zero_async(downward_stream.get());
    outputs.d_pot_direct_src.zero_async(direct_stream.get());
    outputs.d_pot_direct_trg.zero_async(direct_stream.get());
    sctl::Profile::Toc();
}

template <typename Real, int DIM>
void State<Real, DIM>::upload_and_sort_charges(const Real *charges, const Real *normals, long n_src) {
    const int charge_dof = get_kernel_input_dim(DIM, kernel);

    if (particles.gpu_tree) {
        // AddParticleData sorts the host values into tree order on the device -- exactly what the
        // scatter kernels below do by hand, but with no permutation on the host.
        particles.charge_ptr =
            gpu_tree_add_data<Real, DIM>(particles.gpu_tree, "pdmk_charge", "pdmk_src", charges, charge_dof);
        if (kernel == DMK_STRESSLET) {
            particles.normal_ptr =
                gpu_tree_add_data<Real, DIM>(particles.gpu_tree, "pdmk_normal", "pdmk_src", normals, DIM);
            particles.d_charge_outer.resize(particles.n_src * DIM * DIM);
            particles.charge_outer_ptr = particles.d_charge_outer.data();
            // Both operands are already in tree order, so the outer product needs no index.
            launch_outer_product<Real>(particles.charge_ptr, particles.normal_ptr, particles.d_charge_outer.data(),
                                       particles.n_src, DIM, direct_stream.get());
        }
        direct_stream.sync();
        return;
    }

    DeviceBuffer<Real> d_charge_input;
    d_charge_input.upload_async(charges, n_src * charge_dof, direct_stream.get());
    particles.d_charge.resize(n_src * charge_dof);
    launch_scatter_forward(d_charge_input.data(), particles.d_charge.data(), particles.d_scatter_index_src.data(),
                           n_src, charge_dof, direct_stream.get());

    if (kernel == DMK_STRESSLET) {
        DeviceBuffer<Real> d_normal_input;
        d_normal_input.upload_async(normals, n_src * DIM, direct_stream.get());
        particles.d_normal.resize(n_src * DIM);
        launch_scatter_forward(d_normal_input.data(), particles.d_normal.data(), particles.d_scatter_index_src.data(),
                               n_src, DIM, direct_stream.get());
        particles.d_charge_outer.resize(n_src * DIM * DIM);
        launch_scatter_forward_stresslet(d_charge_input.data(), d_normal_input.data(), particles.d_charge_outer.data(),
                                         particles.d_scatter_index_src.data(), n_src, DIM, direct_stream.get());
        particles.normal_ptr = particles.d_normal.data();
        particles.charge_outer_ptr = particles.d_charge_outer.data();
        particles.charge_ptr = particles.d_charge.data();
        direct_stream.sync();
        return;
    }
    particles.charge_ptr = particles.d_charge.data();
    direct_stream.sync();
}

template <typename Real, int DIM>
void State<Real, DIM>::finalize() {
    // direct_stream must see the eval-side writes (queued on downward_stream)
    // before the accumulate; direct_stream is already serial with the direct
    // pass's own writes to d_pot_direct_*.
    auto eval_done = cuda_helpers::DeviceEvent::disable_timing();
    DMK_CHECK_CUDA(cudaEventRecord(eval_done, downward_stream.get()));
    DMK_CHECK_CUDA(cudaStreamWaitEvent(direct_stream.get(), eval_done, 0));

    if (particles.gpu_tree) {
        // Sum in tree order; desort_potentials hands the result to the tree to reorder.
        if (outputs.pot_src_size)
            launch_accumulate<Real>(outputs.pot_src_tree, outputs.d_pot_eval_src.data(),
                                    outputs.d_pot_direct_src.data(), outputs.pot_src_size, direct_stream.get());
        if (outputs.pot_trg_size)
            launch_accumulate<Real>(outputs.pot_trg_tree, outputs.d_pot_eval_trg.data(),
                                    outputs.d_pot_direct_trg.data(), outputs.pot_trg_size, direct_stream.get());
        direct_stream.sync();
        return;
    }

    if (outputs.pot_src_size) {
        const long n = static_cast<long>(outputs.pot_src_size / outputs.pot_src_dof);
        launch_accumulate_and_scatter<Real>(outputs.d_pot_src_final.data(), outputs.d_pot_eval_src.data(),
                                            outputs.d_pot_direct_src.data(), particles.d_scatter_index_src.data(),
                                            outputs.pot_src_dof, n, direct_stream.get());
    }
    if (outputs.pot_trg_size) {
        const long n = static_cast<long>(outputs.pot_trg_size / outputs.pot_trg_dof);
        launch_accumulate_and_scatter<Real>(outputs.d_pot_trg_final.data(), outputs.d_pot_eval_trg.data(),
                                            outputs.d_pot_direct_trg.data(), particles.d_scatter_index_trg.data(),
                                            outputs.pot_trg_dof, n, direct_stream.get());
    }
    direct_stream.sync();
}

template <typename Real, int DIM>
void State<Real, DIM>::dump(DMKPtTree<Real, DIM> &tree) {
    const std::string prefix = "gpu_v2/";
    tree.dump(prefix);
    auto write = [&](const std::string &name, const Real *d_ptr, std::size_t n) {
        const std::string path = prefix + name + "." + std::to_string(tree.comm().Size()) + "." +
                                 std::to_string(tree.comm().Rank()) + ".dat";
        cuda_helpers::dump_device_buffer_to_file<Real>(path, d_ptr, n);
    };
    write("dmk_proxy_coeffs_downward", scratch.d_proxy_coeffs_downward.data(), scratch.d_proxy_coeffs_downward.size());
    write("dmk_proxy_coeffs", scratch.d_proxy_coeffs_upward.data(), scratch.d_proxy_coeffs_upward.size());
}

template BuildInputs<float, 2> to_build_inputs<float, 2>(DMKPtTree<float, 2> &, GpuTreeMetadata<float> *);
template BuildInputs<float, 3> to_build_inputs<float, 3>(DMKPtTree<float, 3> &, GpuTreeMetadata<float> *);
template BuildInputs<double, 2> to_build_inputs<double, 2>(DMKPtTree<double, 2> &, GpuTreeMetadata<double> *);
template BuildInputs<double, 3> to_build_inputs<double, 3>(DMKPtTree<double, 3> &, GpuTreeMetadata<double> *);

template struct State<float, 2>;
template struct State<float, 3>;
template struct State<double, 2>;
template struct State<double, 3>;

} // namespace dmk::cuda::pt
