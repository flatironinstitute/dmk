#include <dmk/cuda/pt/tree.hpp>

#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/pt/passes.hpp>
#include <dmk/direct.hpp>
#include <dmk/error.hpp>
#include <dmk/logger.h>
#include <dmk/nvtx_wrapper.h>

#include "metadata.hpp"

#include <cstdlib>
#include <string>

namespace dmk::cuda::pt {
namespace {

/// The device this rank should drive when the caller did not name one: one per rank among the
/// ranks sharing a node, so a four-rank node with four GPUs uses all of them.
int deviceForRank(const sctl::Comm &comm, int log_level) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1)
        throw std::runtime_error("dmk::cuda::pt: no CUDA device available");

    int local_rank = 0;
    int local_size = 1;
#ifdef DMK_HAVE_MPI
    { // rank among the ranks sharing this node
        MPI_Comm node = MPI_COMM_NULL;
        MPI_Comm_split_type(comm.GetMPI_Comm(), MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
        MPI_Comm_rank(node, &local_rank);
        MPI_Comm_size(node, &local_size);
        MPI_Comm_free(&node);
    }
#endif
    const int dev = local_rank % n_dev;
    const auto logger = dmk::get_rank_logger(comm, log_level);
    if (local_size > n_dev)
        logger->warn("{} ranks share this node's {} device(s), so they share GPUs", local_size, n_dev);
    logger->info("rank {} of {} on this node uses device {} of {}", local_rank, local_size, dev, n_dev);
    return dev;
}

/// Bind before any other CUDA call creates a context.
void selectDevice(const sctl::Comm &comm, int log_level, int requested) {
    static const bool done = [&comm, log_level, requested] {
        bind_gpu_device(requested >= 0 ? requested : deviceForRank(comm, log_level));
        return true;
    }();
    (void)done;
}

// A pass range that closes on its own device work. Pushed enqueue-side it would be
// a few microseconds wide and every kernel would land under the join in `finalize`,
// so DMK_NVTX_SYNC=1 syncs the pass's stream before the pop. That serializes the
// pipeline -- it is a measurement mode, and the per-pass numbers it reports are
// uncontended ones, not the times those passes take in a real eval.
bool nvtx_sync() {
    static const bool on = [] {
        const char *v = std::getenv("DMK_NVTX_SYNC");
        return v && std::atoi(v) != 0;
    }();
    return on;
}

struct NvtxPass {
    NvtxPass(const char *name, cudaStream_t stream) : stream_(stream) { nvtxRangePush(name); }
    ~NvtxPass() {
        if (nvtx_sync() && stream_)
            cudaStreamSynchronize(stream_);
        nvtxRangePop();
    }
    NvtxPass(const NvtxPass &) = delete;
    NvtxPass &operator=(const NvtxPass &) = delete;

    cudaStream_t stream_;
};

/// Copies the device tree's node topology and per-node particle counts into the host tree.
template <class Real, int DIM>
void copyTreeCounts(device_tree::PtTree<Real, DIM> &dev_tree, bool has_trg, DMKPtTree<Real, DIM> &out) {
    sctl::Vector<sctl::Long> cnt;
    dev_tree.Data("pdmk_src", cnt);
    out.r_src_cnt_with_halo = cnt;
    if (has_trg) {
        dev_tree.Data("pdmk_trg", cnt);
        out.r_trg_cnt_owned = cnt;
    } else {
        out.r_trg_cnt_owned.ReInit(cnt.Dim());
        out.r_trg_cnt_owned.SetZero();
    }
}

template <typename Real, int DIM>
void copyTreeToHost(device_tree::PtTree<Real, DIM> &dev_tree, bool has_trg, DMKPtTree<Real, DIM> &out) {
    constexpr int n_child = 1 << DIM;
    constexpr int n_nbr = sctl::pow<DIM>(3);

    const auto mid = dev_tree.NodeMID();
    const sctl::Long n_nodes = mid.size();
    const auto attr = dev_tree.NodeAttrs();
    const auto lists = dev_tree.Lists();
    SCTL_ASSERT(attr.size() == (std::size_t)n_nodes);
    SCTL_ASSERT(lists.parent.size() == (std::size_t)n_nodes);
    SCTL_ASSERT(lists.child.size() == (std::size_t)(n_nodes * n_child));
    SCTL_ASSERT(lists.nbr.size() == (std::size_t)(n_nodes * n_nbr));

    out.node_mid_host.ReInit(n_nodes);
    out.node_attr_host.ReInit(n_nodes);
    out.node_lst_host.ReInit(n_nodes);

    // One page-locked staging block for the whole topology, rather than a pageable buffer per
    // array: pageable D2H runs at a fraction of the link's rate, and the repack below reads the
    // staging copy either way. The five transfers are enqueued async so they overlap, then joined
    // once. If page-locking fails the staging is pageable, which makes each copy synchronous but
    // still correct.
    const auto pad = [](std::size_t x) { return (x + 255) & ~(std::size_t)255; };
    const std::size_t b_mid = n_nodes * sizeof(sctl::Morton<DIM>);
    const std::size_t b_attr = n_nodes * sizeof(device_tree::NodeAttr);
    const std::size_t b_par = n_nodes * sizeof(sctl::Long);
    const std::size_t b_child = (std::size_t)n_nodes * n_child * sizeof(sctl::Long);
    const std::size_t o_attr = pad(b_mid), o_par = pad(o_attr + b_attr), o_child = pad(o_par + b_par),
                      total = pad(o_child + b_child);

    const auto stage = n_nodes ? cuda_helpers::pinned_alloc(total) : std::pair<char *, std::size_t>{nullptr, 0};
    // sctl::Long elements so the fallback is at least as aligned as the arrays read out of it
    sctl::ScratchBuf<sctl::Long> fallback(!n_nodes || stage.first ? 0 : (sctl::Long)(total / sizeof(sctl::Long)));
    char *const buf = stage.first ? stage.first : (n_nodes ? (char *)&fallback[0] : nullptr);

    if (n_nodes) {
        const auto grab = [&](std::size_t off, const void *src, std::size_t bytes) {
            DMK_CHECK_CUDA(cudaMemcpyAsync(buf + off, src, bytes, cudaMemcpyDeviceToHost, 0));
        };
        grab(0, mid.data(), b_mid);
        grab(o_attr, attr.data(), b_attr);
        grab(o_par, lists.parent.data(), b_par);
        grab(o_child, lists.child.data(), b_child);
        DMK_CHECK_CUDA(cudaStreamSynchronize(0));

        const auto *h_mid = reinterpret_cast<const sctl::Morton<DIM> *>(buf);
        const auto *h_attr = reinterpret_cast<const device_tree::NodeAttr *>(buf + o_attr);
        const auto *h_par = reinterpret_cast<const sctl::Long *>(buf + o_par);
        const auto *h_child = reinterpret_cast<const sctl::Long *>(buf + o_child);

// SoA on the device, one struct per node on the host; one pass over the staging block.
#pragma omp parallel for schedule(static)
        for (sctl::Long i = 0; i < n_nodes; ++i) {
            out.node_mid_host[i] = h_mid[i];
            out.node_attr_host[i].Leaf = h_attr[i].Leaf;
            out.node_attr_host[i].Ghost = h_attr[i].Ghost;
            auto &l = out.node_lst_host[i];
            l.p2n = (i ? h_mid[i].Path2Node() : 0);
            l.parent = h_par[i];
            for (int k = 0; k < n_child; ++k)
                l.child[k] = h_child[i * n_child + k];
        }
    }
    if (stage.first)
        cuda_helpers::pinned_free(stage.first, stage.second);

    copyTreeCounts<Real, DIM>(dev_tree, has_trg, out);
}

} // namespace

void bind_gpu_device(int device_id) {
    if (device_id < 0) // the tree resolves it against its communicator
        return;

    int n_devices = 0;
    DMK_CHECK_CUDA(cudaGetDeviceCount(&n_devices));
    if (device_id >= n_devices)
        throw api_error(DMK_ERR_INVALID_ARGUMENT, "gpu_device_id " + std::to_string(device_id) + " is out of range: " +
                                                      std::to_string(n_devices) + " CUDA device(s) visible");
    DMK_CHECK_CUDA(cudaSetDevice(device_id));

    // Initialized by the first call, so `bound` is the device this process committed to.
    static const int bound = device_id;
    if (bound != device_id)
        throw api_error(DMK_ERR_INVALID_ARGUMENT, "this process is already using CUDA device " + std::to_string(bound) +
                                                      "; DMK drives one device per process (requested " +
                                                      std::to_string(device_id) + ")");
}

/// One device allocation per element type, handed out in slices: the metadata is some fifty
/// per-box arrays, and fifty pool lookups per build is host time the tree build can feel.
template <typename T>
struct Arena {
    cuda_helpers::DeviceBuffer<T> buf;
    std::size_t used = 0;
    void reserve(std::size_t n) {
        buf.resize(n);
        used = 0;
    }
    T *take(std::size_t n) {
        T *const p = buf.data() + used;
        used += n;
        return p;
    }
};

/// Owns the device metadata buffers for as long as the State that adopted them. Sized once, before
/// the depth is known, against the deepest tree the Morton ids can represent.
template <typename Real, int DIM>
struct DeviceMetadata {
    Arena<int> ints, scratch_ints;
    Arena<long> longs, scratch_longs;
    Arena<unsigned char> bytes;
    Arena<signed char> shifts;
    Arena<Real> reals;
    cuda_helpers::DeviceBuffer<ShiftPwGroupSrc> shift_src;
    /// max_depth | proxy counts | proxy totals | join block | the two per-node count arrays on their way up
    std::pair<char *, std::size_t> pinned{nullptr, 0};
    long *h_counts = nullptr;
    BoxStatusOutputs bo;
    DeviceMetadataInputs<Real, DIM> in;
    DeviceMetadataOutputs<Real> out;
    ~DeviceMetadata() {
        if (pinned.first)
            cuda_helpers::pinned_free(pinned.first, pinned.second);
        if (bo.ready)
            cudaEventDestroy(bo.ready);
        if (out.proxy_ready)
            cudaEventDestroy(out.proxy_ready);
    }

    void allocate(long nb, int stride, bool periodic, int shift_group) {
        constexpr int nl = kMaxLevels;
        const std::size_t n = nb;
        const long bound = shift_groups_bound(nb, nl, shift_group);
        out.n_groups_bound = bound;

        ints.reserve(n * (28 + stride) + 2 * (bound + 1)); // the last n is the multi-rank global count
        bo.d_src_counts_with_halo = ints.take(n);
        bo.d_src_counts_owned = ints.take(n);
        bo.d_trg_counts_owned = ints.take(n);
        out.d_box_levels = ints.take(n);
        out.d_list1 = ints.take(n * stride);
        out.d_list1_count = ints.take(n);
        out.d_sorted = ints.take(2 * n);
        out.d_c2p_center_boxes = ints.take(n);
        out.d_c2p_levels = ints.take(n);
        out.d_c2p_src_box_flat_offsets = ints.take(n);
        out.d_c2p_n_src_boxes_per_group = ints.take(n);
        out.d_c2p_src_boxes_flat = ints.take(n);
        out.d_tp_parents = ints.take(n);
        out.d_tp_children = ints.take(n);
        out.d_tp_octants = ints.take(n);
        out.d_tp_assign_dst = ints.take(n);
        out.d_eval_targets = ints.take(n);
        out.d_tp_up_src = ints.take(n);
        out.d_tp_up_dst = ints.take(n);
        out.d_tp_up_octants = ints.take(n);
        out.d_tp_up_assign = ints.take(n);
        out.d_tp_up_par = ints.take(n);
        out.d_tp_up_par_begin = ints.take(n);
        out.d_tp_up_par_count = ints.take(n);
        out.d_pw_eval_flat = ints.take(n);
        out.d_pw_form_flat = ints.take(n);
        out.d_proxy_zero_boxes = ints.take(n);
        out.d_shift_counts = ints.take(bound + 1);
        out.d_shift_offsets = ints.take(bound + 1);

        longs.reserve(3 * (n + 1) + n_offset_kinds * n + 4 * n + 2 + join_size(nl));
        bo.d_count_scan = longs.take(3 * (n + 1));
        bo.d_offsets = longs.take(n_offset_kinds * n);
        out.d_proxy_off_up = longs.take(n);
        out.d_proxy_off_down = longs.take(n);
        out.d_proxy_counts_up = longs.take(n);
        out.d_pw_out_offsets = longs.take(n);
        out.d_proxy_totals = longs.take(2);
        out.d_join = longs.take(join_size(nl));

        bytes.reserve(5 * n);
        bo.d_is_global_leaf = bytes.take(n);
        bo.d_ifpwexp = bytes.take(n);
        bo.d_iftensprodeval = bytes.take(n);
        out.d_is_c2p_center = bytes.take(n);
        out.d_is_tp_up_par = bytes.take(n);
        shifts.reserve(periodic ? n * stride * DIM : 0);
        out.d_list1_shift = periodic ? shifts.take(n * stride * DIM) : nullptr;

        reals.reserve(n * DIM + n);
        out.d_centers = reals.take(n * DIM);
        out.d_self_correction_work = reals.take(n);

        scratch_longs.reserve(2 * n + scan_scratch_bytes(nb) / sizeof(long));
        out.d_sort_keys = scratch_longs.take(2 * n);
        out.d_scan = scratch_longs.take(scan_scratch_bytes(nb) / sizeof(long));
        scratch_ints.reserve(n_chunks(nb) * kMaxLevels * 2 + level_scratch_size(nl));
        out.d_chunk_table = scratch_ints.take(n_chunks(nb) * kMaxLevels * 2);
        out.d_level_scratch = scratch_ints.take(level_scratch_size(nl));

        pinned = cuda_helpers::pinned_alloc((1 + n + 2 + join_size(nl) + 2 * n) * sizeof(long));
        long *const h = reinterpret_cast<long *>(pinned.first);
        bo.h_max_depth = h;
        out.h_proxy_counts_up = h + 1;
        out.h_proxy_totals = h + 1 + n;
        out.h_join = h + 3 + n;
        h_counts = h + 3 + n + join_size(nl);
        bo.d_zero = out.d_level_scratch;
        bo.n_zero = level_scratch_size(nl);
        DMK_CHECK_CUDA(cudaEventCreateWithFlags(&bo.ready, cudaEventDisableTiming));
        DMK_CHECK_CUDA(cudaEventCreateWithFlags(&out.proxy_ready, cudaEventDisableTiming));

        out.d_is_global_leaf = bo.d_is_global_leaf;
        out.d_ifpwexp = bo.d_ifpwexp;
        out.d_src_counts_with_halo = bo.d_src_counts_with_halo;
        out.d_src_counts_owned = bo.d_src_counts_owned;
        out.d_trg_counts_owned = bo.d_trg_counts_owned;
        out.d_offsets = bo.d_offsets;
    }

    /// The pass is joined; what only it read can go back to the pool.
    void release_scratch() {
        scratch_ints.buf.reset();
        scratch_longs.buf.reset();
    }
};

template <typename Real, int DIM>
Tree<Real, DIM>::Tree(const sctl::Comm &comm, const pdmk_params &params, const sctl::Vector<Real> &r_src,
                      const sctl::Vector<Real> &charge, const sctl::Vector<Real> &normal,
                      const sctl::Vector<Real> &r_trg) {
    const cudaStream_t stream = 0;
    NvtxPass treeprof("build_tree", stream);
    selectDevice(comm, params.log_level, params.gpu_device_id);
    n_src_local_ = r_src.Dim() / DIM;
    n_trg_local_ = r_trg.Dim() / DIM;
    const sctl::Long n_trg = n_trg_local_;
    charge_dof_ = get_kernel_input_dim(DIM, params.kernel);
    stresslet_ = params.kernel == DMK_STRESSLET;
    stage_charges(charge.Dim() ? &charge[0] : nullptr, normal.Dim() ? &normal[0] : nullptr);

    { // build the device tree
        sctl::Profile::Scoped profile("build_device_tree", &comm);
        nvtxRangePush("build_device_tree");
        using MemSpace = device_tree::MemSpace;
        constexpr bool balance21 = true;
        constexpr int halo = 0;
        const auto periodicity = params.use_periodic ? sctl::Periodicity::XYZ : sctl::Periodicity::NONE;
        const sctl::Long n_src = n_src_local_;

        dev_tree_ = std::make_unique<device_tree::PtTree<Real, DIM>>(comm);
        { // one upload serves the refinement and the source set
            cuda_helpers::DeviceBuffer<Real> d_src;
            d_src.upload(n_src ? &r_src[0] : nullptr, r_src.Dim());
            dev_tree_->UpdateRefinement(d_src.data(), n_src, MemSpace::Device, params.n_per_leaf, balance21,
                                        periodicity, halo);
            dev_tree_->AddParticles("pdmk_src", d_src.data(), n_src, MemSpace::Device);
        }
        if (n_trg)
            dev_tree_->AddParticles("pdmk_trg", &r_trg[0], n_trg, MemSpace::Host);

        { // registered before the source broadcast, so sized to the owned sources
            const int dof_src = get_kernel_output_dim(DIM, params.kernel, params.eval_src);
            const int dof_trg = get_kernel_output_dim(DIM, params.kernel, params.eval_trg);
            if (dof_src)
                dev_tree_->AddParticleData("pdmk_pot_src", "pdmk_src", dof_src);
            if (n_trg && dof_trg)
                dev_tree_->AddParticleData("pdmk_pot_trg", "pdmk_trg", dof_trg);
        }
        dev_tree_->Broadcast("pdmk_src");
        nvtxRangePop();
    }

    // host precompute only, over the device tree's per-node counts
    nvtxRangePush("cpu_tree_init");
    tree_ = std::make_unique<DMKPtTree<Real, DIM>>(comm, params, r_src, charge, normal, r_trg);
    nvtxRangePop();
    {
        NvtxPass gprof("copy_tree_to_host", stream);
        sctl::Profile::Scoped profile("copy_tree_to_host", &comm);
        // The AoS node arrays are only needed by the multi-rank node_status walk and by the tree
        // dump; everything else on this path reads the device tree directly.
        const bool want_dump = (params.debug_flags & DMK_DEBUG_DUMP_TREE) || util::env_is_set("DMK_DEBUG_DUMP_TREE");
        if (comm.Size() > 1 || want_dump)
            copyTreeToHost<Real, DIM>(*dev_tree_, n_trg != 0, *tree_);
        else
            copyTreeCounts<Real, DIM>(*dev_tree_, n_trg != 0, *tree_);
    }
    sctl::Long node_begin = 0;
    sctl::Long node_end = 0;
    { // adopt its nodes
        NvtxPass gprof("adopt_nodes", stream);
        sctl::Profile::Scoped profile("adopt_nodes", &comm);
        dev_tree_->OwnedRange(node_begin, node_end);
        tree_->adopt_device_tree(node_begin, node_end);
    }
    if (comm.Size() > 1) { // box_status derives both on the device for a single rank
        NvtxPass gprof("node_status", stream);
        sctl::Profile::Scoped profile("node_status", &comm);
        const auto &attr = tree_->GetNodeAttr();
        const sctl::Long n_boxes = attr.Dim();
        sctl::Vector<bool> leaf(n_boxes);
#pragma omp parallel for schedule(static)
        for (sctl::Long i = 0; i < n_boxes; ++i)
            leaf[i] = attr[i].Leaf;

        sctl::Vector<sctl::Long> under(n_boxes);
        { // sources this rank owns under each box: a subtree's counts are one run in Morton order
            const auto &nl = tree_->GetNodeLists();
            const auto &cnt = tree_->r_src_cnt_with_halo;
            constexpr int n_child = 1 << DIM;
            sctl::ScratchBuf<sctl::Long> off(n_boxes + 1);
            off[0] = 0;
            for (sctl::Long i = 0; i < n_boxes; ++i)
                off[i + 1] = off[i] + (i >= node_begin && i < node_end ? cnt[i] : 0);
#pragma omp parallel for schedule(static)
            for (sctl::Long i = 0; i < n_boxes; ++i) {
                sctl::Long j = i; // the run ends at the next sibling of i or of its nearest ancestor with one
                while (j > 0 && nl[j].p2n == n_child - 1)
                    j = nl[j].parent;
                const sctl::Long end = j > 0 ? off[nl[nl[j].parent].child[nl[j].p2n + 1]] : off[n_boxes];
                under[i] = end - off[i];
            }
        }
        { // reduced over every holder: refined here, and sources owned under
            sctl::ScratchBuf<sctl::Long> ones(n_boxes);
#pragma omp parallel for schedule(static)
            for (sctl::Long i = 0; i < n_boxes; ++i)
                ones[i] = 1;
            dev_tree_->AddData("node_status", 2, sctl::Vector<sctl::Long>(n_boxes, ones.begin(), false));
            sctl::Vector<sctl::Long> cnt;
            const auto span = dev_tree_->Data("node_status", cnt);
            SCTL_ASSERT_MSG(span.size() == 2 * (std::size_t)n_boxes, "pt::Tree: two values per box");
            sctl::ScratchBuf<Real> v(2 * n_boxes);
#pragma omp parallel for schedule(static)
            for (sctl::Long i = 0; i < n_boxes; ++i) {
                v[2 * i] = leaf[i] ? Real(0) : Real(1);
                v[2 * i + 1] = (Real)under[i];
            }
            DMK_CHECK_CUDA(cudaMemcpy(span.data(), &v[0], 2 * n_boxes * sizeof(Real), cudaMemcpyHostToDevice));

            dev_tree_->ReduceBroadcast("node_status");

            sctl::Vector<sctl::Long> cnt2;
            const auto out = dev_tree_->Data("node_status", cnt2);
            sctl::ScratchBuf<Real> total(out.size());
            if (out.size())
                DMK_CHECK_CUDA(cudaMemcpy(&total[0], out.data(), out.size() * sizeof(Real), cudaMemcpyDeviceToHost));
            sctl::ScratchBuf<sctl::Long> pos(n_boxes);
            sctl::omp_par::scan(cnt2.begin(), pos.begin(), n_boxes, sctl::Long(0));
#pragma omp parallel for schedule(static)
            for (sctl::Long i = 0; i < n_boxes; ++i) {
                // a node outside the exchange is a coarse fill node for another rank's subtree
                const bool reached = cnt2[i] > 0;
                leaf[i] = reached && total[2 * pos[i]] == Real(0);          // ranks that refined the node
                under[i] = reached ? (sctl::Long)total[2 * pos[i] + 1] : 0; // owned sources under it, all ranks
            }
        }
        tree_->is_global_leaf.Swap(leaf);
        tree_->src_counts_global.Swap(under);
    }

    const sctl::Long nb = tree_->n_boxes();
    const int stride = DMKPtTree<Real, DIM>::list1_stride();
    const int shift_group = shift_group_size(tree_->n_tables_down);
    md_ = std::make_unique<DeviceMetadata<Real, DIM>>();
    { // subtree counts, both expansion flags and the particle offsets, enqueued only
        NvtxPass gprof("box_status", stream);
        sctl::Profile::Scoped profile("box_status", &comm);
        md_->allocate(nb, stride, params.use_periodic, shift_group);
        if (nb) {
            const auto lists = dev_tree_->Lists();
            BoxStatusInputs<DIM> bi;
            bi.d_mid = dev_tree_->NodeMID().data();
            bi.d_parent = lists.parent.data();
            bi.d_child = lists.child.data();
            bi.d_nbr = lists.nbr.data();
            bi.d_attr = dev_tree_->NodeAttrs().data();
            // Single rank: leafness is the node's own attribute, so box_status derives it and
            // node_status above never ran.
            bi.h_is_global_leaf =
                comm.Size() > 1 ? reinterpret_cast<const unsigned char *>(&tree_->is_global_leaf[0]) : nullptr;
            // Through the page-locked block: from pageable memory the runtime would first drain the
            // stream, which now holds the whole tree build.
            std::copy(&tree_->r_src_cnt_with_halo[0], &tree_->r_src_cnt_with_halo[0] + nb, md_->h_counts);
            bi.h_src_cnt_with_halo = md_->h_counts;
            if (tree_->r_trg_cnt_owned.Dim()) {
                std::copy(&tree_->r_trg_cnt_owned[0], &tree_->r_trg_cnt_owned[0] + nb, md_->h_counts + nb);
                bi.h_trg_cnt_owned = md_->h_counts + nb;
            }
            bi.owned_begin = node_begin;
            bi.owned_end = node_end;
            bi.n_boxes = nb;
            bi.charge_dof = tree_->kernel_input_dim;
            bi.n_tables_up = tree_->n_tables_up;
            bi.dof_src = tree_->kernel_output_dim_src;
            bi.dof_trg = tree_->kernel_output_dim_trg;
            box_status<DIM>(bi, md_->bo);
        }
    }
    { // sorts the staged charges; needs only the tree, so it overlaps the pass above
        NvtxPass gprof("scatter_charges", stream);
        sctl::Profile::Scoped profile("scatter_charges", &comm);
        scatter_charges();
    }
    { // the depth is the one thing the host needs from the device before it can size its tables
        NvtxPass gprof("metadata_for_gpu", stream);
        sctl::Profile::Scoped profile("metadata_for_gpu", &comm);
        int max_depth = 1;
        if (nb) {
            DMK_CHECK_CUDA(cudaEventSynchronize(md_->bo.ready));
            max_depth = (int)*md_->bo.h_max_depth + 1;
        }
        tree_->metadata_for_gpu(max_depth);
    }

    { // per-box geometry, interaction lists and work lists, on the device
        NvtxPass gprof("device_metadata", stream);
        sctl::Profile::Scoped profile("device_metadata", &comm);
        auto &m = *md_;
        if (nb && comm.Size() > 1) { // reduced across ranks, so not the device's own subtree sums
            int *const d_global = m.ints.take(nb);
            std::vector<int> global(nb);
            for (long i = 0; i < nb; ++i)
                global[i] = (int)tree_->src_counts_global[i];
            DMK_CHECK_CUDA(cudaMemcpyAsync(d_global, global.data(), nb * sizeof(int), cudaMemcpyHostToDevice, 0));
            DMK_CHECK_CUDA(cudaStreamSynchronize(0));
            m.in.d_src_cnt_global = d_global;
        } else {
            m.in.d_src_cnt_global = m.bo.d_src_counts_owned;
        }

        auto &mi = m.in;
        mi.d_mid = dev_tree_->NodeMID().data();
        mi.d_attr = dev_tree_->NodeAttrs().data();
        const auto lists = dev_tree_->Lists();
        mi.d_parent = lists.parent.data();
        mi.d_child = lists.child.data();
        mi.d_nbr = lists.nbr.data();
        mi.d_is_global_leaf = m.bo.d_is_global_leaf;
        mi.d_ifpwexp = m.bo.d_ifpwexp;
        mi.d_iftensprodeval = m.bo.d_iftensprodeval;
        mi.d_src_cnt_with_halo = m.bo.d_src_counts_with_halo;
        mi.d_src_cnt_owned = m.bo.d_src_counts_owned;
        mi.d_trg_cnt_owned = m.bo.d_trg_counts_owned;
        mi.n_coeffs_up = tree_->n_tables_up * sctl::pow<DIM>(tree_->expansion_constants.n_order);
        mi.n_coeffs_down = tree_->n_tables_down * sctl::pow<DIM>(tree_->expansion_constants.n_order);
        mi.n_boxes = nb;
        mi.n_levels = tree_->n_levels();
        mi.nlist1_stride = stride;
        mi.periodic = params.use_periodic;
        mi.skip_root_form = params.use_periodic;
        mi.shift_group = shift_group;
        {
            const int n_pw = tree_->expansion_constants.n_pw_diff;
            mi.n_pw_per_box = (long)sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2) * tree_->n_tables_down;
        }
        // Same choice build_self_correction_work_list makes on the host path.
        if (params.kernel == DMK_STRESSLET)
            mi.self_mode = SelfCorrectionMode::zero;
        else if (params.kernel == DMK_LAPLACE_DIPOLE)
            mi.self_mode =
                params.eval_src >= DMK_POTENTIAL_GRAD ? SelfCorrectionMode::w0_grad : SelfCorrectionMode::zero;
        else
            mi.self_mode = SelfCorrectionMode::w0;
        const auto &w0 = mi.self_mode == SelfCorrectionMode::w0_grad ? tree_->self_w0_grad : tree_->self_w0;
        for (sctl::Long i = 0; i < w0.Dim() && i <= kMaxLevels; ++i)
            mi.w0.v[i] = w0[i];
        compute_device_metadata<Real, DIM>(mi, m.out);
    }

    { // only the proxy readbacks, which the pass enqueues first; the rest of it is still running
        finish_proxy_metadata<Real>(md_->out);
        if (nb) {
            tree_->proxy_coeffs_counts.ReInit(nb, sctl::Ptr2Itr<sctl::Long>(md_->out.h_proxy_counts_up, nb), false);
            tree_->proxy_coeffs_downward_dim = md_->out.h_proxy_totals[1];
        }
    }
    dev_tree_->AddData("proxy_coeffs", 1, tree_->proxy_coeffs_counts); // the tree sums these across ranks
    {
        NvtxPass gprof("device_metadata_join", stream);
        sctl::Profile::Scoped profile("device_metadata_join", &comm);
        finish_device_metadata<Real>(md_->out, tree_->n_levels());
        tree_->pw_out_size = md_->out.pw_out_size;
        md_->shift_src.resize(md_->out.n_shift_src);
        fill_shift_groups<Real, DIM>(md_->in, md_->out, md_->shift_src.data());
        md_->release_scratch();
    }

    { // build the device state
        sctl::Profile::Scoped profile("state_ctor", &comm);
        NvtxPass gprof("state_ctor", stream);
        auto inputs = to_build_inputs(*tree_, md_->out);
        inputs.topology.d_nbr = reinterpret_cast<const long *>(dev_tree_->Lists().nbr.data());
        inputs.worklists.d_shift_group_src = md_->shift_src.data();
        inputs.worklists.d_shift_group_offsets = md_->out.d_shift_offsets;
        { // views into the tree's storage
            sctl::Vector<sctl::Long> cnt;
            inputs.scratch.d_proxy_coeffs = dev_tree_->Data("proxy_coeffs", cnt);
            inputs.particles.d_r_src = dev_tree_->Data("pdmk_src", cnt);
            if (n_trg)
                inputs.particles.d_r_trg = dev_tree_->Data("pdmk_trg", cnt);
        }
        state_ = std::make_unique<State<Real, DIM>>(inputs);
    }

    {
        sctl::Profile::Scoped profile("bind_charges", &comm);
        NvtxPass gprof("bind_charges", stream);
        bind_charges();
    }
    cuda_helpers::check_device_errors("tree create");
    nvtxRangePop();
}

template <typename Real, int DIM>
Tree<Real, DIM>::~Tree() {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const std::string msg = std::string("CUDA error at tree destroy: ") + cudaGetErrorString(err);
        dmk::set_last_error(msg);
        dmk::get_logger()->error("{}", msg);
    }
}

template <typename Real, int DIM>
void Tree<Real, DIM>::eval() {
    // The near-field `direct` runs concurrently on direct_stream with the
    // upward -> form_outgoing -> downward -> eval_targets chain on
    // downward_stream; `finalize` joins them (direct_stream waits on the
    // downward-stream eval writes), sums the near+far potentials, descatters to
    // user order in d_pot_*_final, and syncs.
    const auto ds = state_->direct_stream;
    const auto ws = state_->downward_stream;
    {
        // Neither proxy buffer is zeroed: every box either has a writer that assigns before it
        // accumulates, or is cleared per-slab inside the pass that reads it.
        NvtxPass r("pt_upward", ws);
        pt::upward(*state_, ws);
    }
    if (GetComm().Size() > 1) { // sum each box's partial expansions over the ranks holding it
        DMK_CHECK_CUDA(cudaStreamSynchronize(ws));
        dev_tree_->ReduceBroadcast("proxy_coeffs");
        sctl::Vector<sctl::Long> cnt;
        const auto span = dev_tree_->Data("proxy_coeffs", cnt);
        state_->scratch.d_proxy_coeffs_upward = span;

        // the exchange relays the data set, so the per-box offsets are rebuilt from its counts
        const sctl::Long n_boxes = cnt.Dim();
        sctl::ScratchBuf<sctl::Long> off(n_boxes);
        sctl::omp_par::scan(cnt.begin(), off.begin(), n_boxes, sctl::Long(0));
        SCTL_ASSERT_MSG((std::size_t)(off[n_boxes - 1] + cnt[n_boxes - 1]) == span.size(),
                        "pt::Tree: proxy counts disagree with the exchanged storage");
#pragma omp parallel for schedule(static)
        for (sctl::Long b = 0; b < n_boxes; ++b)
            if (!cnt[b])
                off[b] = -1;
        state_->scratch.d_proxy_offsets_upward.upload(&off[0], n_boxes);
    }
    {
        NvtxPass r("pt_form_outgoing", ws);
        pt::form_outgoing(*state_, ws);
    }
    {
        NvtxPass r("pt_downward", ws);
        pt::downward(*state_, ws);
    }
    {
        NvtxPass r("pt_eval_targets", ws);
        pt::eval_targets(*state_, ws);
    }
    {
        NvtxPass r("pt_direct", ds);
        pt::direct(*state_, ds);
    }
    {
        NvtxPass r("pt_self_correction", ds);
        pt::self_correction(*state_, ds);
    }
    {
        NvtxPass r("pt_finalize", nullptr);
        state_->finalize();
    }
    cuda_helpers::check_device_errors("tree eval");
}

template <typename Real, int DIM>
void Tree<Real, DIM>::desort_potentials(Real *pot_src, Real *pot_trg) {
    // the sums are in tree order; the tree writes them back in the caller's order
    using MemSpace = device_tree::MemSpace;
    const auto &o = state_->outputs;
    const auto give_back = [this](const char *name, const Real *d_sorted, std::size_t n_sorted, Real *out,
                                  sctl::Long n_out) {
        sctl::Vector<sctl::Long> cnt;
        const auto view = dev_tree_->Data(name, cnt);
        SCTL_ASSERT_MSG(view.size() == n_sorted, "pt::Tree::desort_potentials: the tree holds a different count");
        DMK_CHECK_CUDA(cudaMemcpy(view.data(), d_sorted, n_sorted * sizeof(Real), cudaMemcpyDeviceToDevice));
        dev_tree_->GetParticleData(name, out, n_out, MemSpace::Host);
    };
    if (o.pot_src_size)
        give_back("pdmk_pot_src", o.d_pot_src_final.data(), o.pot_src_size, pot_src, n_src_local_ * o.pot_src_dof);
    if (o.pot_trg_size)
        give_back("pdmk_pot_trg", o.d_pot_trg_final.data(), o.pot_trg_size, pot_trg, n_trg_local_ * o.pot_trg_dof);
}

template <typename Real, int DIM>
void Tree<Real, DIM>::stage_charges(const Real *charge, const Real *normal) {
    const cudaStream_t s = cuda_helpers::upload_stream();
    if (charge_consumed_.get()) // the previous scatter is still reading the buffers being refilled
        DMK_CHECK_CUDA(cudaStreamWaitEvent(s, charge_consumed_, 0));
    d_charge_in_.upload_async(charge, n_src_local_ * charge_dof_, s);
    if (stresslet_)
        d_normal_in_.upload_async(normal, n_src_local_ * DIM, s);
    if (!charge_ready_.get())
        charge_ready_ = cuda_helpers::DeviceEvent::disable_timing();
    DMK_CHECK_CUDA(cudaEventRecord(charge_ready_, s));
}

template <typename Real, int DIM>
void Tree<Real, DIM>::scatter_charges() {
    using MemSpace = device_tree::MemSpace;
    DMK_CHECK_CUDA(cudaStreamWaitEvent(0, charge_ready_, 0));
    // the near field reads a box's sources on every rank holding the box, hence the broadcast
    dev_tree_->AddParticleData("pdmk_charge", "pdmk_src", d_charge_in_.data(), d_charge_in_.size(), MemSpace::Device);
    dev_tree_->Broadcast("pdmk_charge");
    if (stresslet_) {
        dev_tree_->AddParticleData("pdmk_normal", "pdmk_src", d_normal_in_.data(), d_normal_in_.size(),
                                   MemSpace::Device);
        dev_tree_->Broadcast("pdmk_normal");
    }
    if (!charge_consumed_.get())
        charge_consumed_ = cuda_helpers::DeviceEvent::disable_timing();
    DMK_CHECK_CUDA(cudaEventRecord(charge_consumed_, 0));
}

template <typename Real, int DIM>
void Tree<Real, DIM>::bind_charges() {
    // Re-queried rather than carried from the scatter: any node data registered in between may
    // have moved the particle arena.
    sctl::Vector<sctl::Long> cnt;
    const auto charge_span = dev_tree_->Data("pdmk_charge", cnt);
    Real *d_normal = nullptr;
    sctl::Long n_normal = 0;
    if (tree_->params.kernel == DMK_STRESSLET) {
        const auto normal_span = dev_tree_->Data("pdmk_normal", cnt);
        d_normal = normal_span.data();
        n_normal = normal_span.size();
    }
    state_->set_charges(charge_span.data(), charge_span.size(), d_normal, n_normal);
}

template <typename Real, int DIM>
void Tree<Real, DIM>::update_charges(const Real *charge, const Real *normal) {
    stage_charges(charge, normal);
    scatter_charges();
    bind_charges();
    cuda_helpers::check_device_errors("tree update_charges");
}

template class Tree<float, 2>;
template class Tree<float, 3>;
template class Tree<double, 2>;
template class Tree<double, 3>;

} // namespace dmk::cuda::pt
