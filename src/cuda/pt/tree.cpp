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
    const std::size_t b_nbr = (std::size_t)n_nodes * n_nbr * sizeof(sctl::Long);
    const std::size_t o_attr = pad(b_mid), o_par = pad(o_attr + b_attr), o_child = pad(o_par + b_par),
                      o_nbr = pad(o_child + b_child), total = pad(o_nbr + b_nbr);

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
        grab(o_nbr, lists.nbr.data(), b_nbr);
        DMK_CHECK_CUDA(cudaStreamSynchronize(0));

        const auto *h_mid = reinterpret_cast<const sctl::Morton<DIM> *>(buf);
        const auto *h_attr = reinterpret_cast<const device_tree::NodeAttr *>(buf + o_attr);
        const auto *h_par = reinterpret_cast<const sctl::Long *>(buf + o_par);
        const auto *h_child = reinterpret_cast<const sctl::Long *>(buf + o_child);
        const auto *h_nbr = reinterpret_cast<const sctl::Long *>(buf + o_nbr);

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
            for (int k = 0; k < n_nbr; ++k)
                l.nbr[k] = h_nbr[i * n_nbr + k];
        }
    }
    if (stage.first)
        cuda_helpers::pinned_free(stage.first, stage.second);

    { // per-node particle counts
        sctl::Vector<sctl::Long> cnt;
        dev_tree.Data("pdmk_src", cnt);
        out.r_src_cnt_with_halo = cnt;

        if (has_trg) {
            dev_tree.Data("pdmk_trg", cnt);
            out.r_trg_cnt_owned = cnt;
        } else {
            out.r_trg_cnt_owned.ReInit(n_nodes);
            out.r_trg_cnt_owned.SetZero();
        }
    }
}

/// Replaces a particle data set with `data`; returns a device pointer to it in tree order.
template <class Tree, class Real>
Real *scatterParticleData(Tree &tree, const std::string &name, const std::string &group, const Real *data, sctl::Long n,
                          sctl::Long &n_out) {
    using MemSpace = device_tree::MemSpace;
    tree.AddParticleData(name, group, data, n, MemSpace::Host);
    tree.Broadcast(name); // the near field reads a box's sources on every rank holding the box
    sctl::Vector<sctl::Long> cnt;
    const auto span = tree.Data(name, cnt);
    n_out = span.size();
    return span.data();
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

/// Owns the device metadata buffers for as long as the State that adopted them.
template <typename Real, int DIM>
struct DeviceMetadata {
    cuda_helpers::DeviceBuffer<Real> centers, self_corr;
    cuda_helpers::DeviceBuffer<int> box_levels, list1, list1_count, direct_work, summary;
    cuda_helpers::DeviceBuffer<signed char> list1_shift;
    cuda_helpers::DeviceBuffer<long> cost;
    DeviceMetadataOutputs<Real> out;
};

template <typename Real, int DIM>
Tree<Real, DIM>::Tree(const sctl::Comm &comm, const pdmk_params &params, const sctl::Vector<Real> &r_src,
                      const sctl::Vector<Real> &charge, const sctl::Vector<Real> &normal,
                      const sctl::Vector<Real> &r_trg) {
    selectDevice(comm, params.log_level, params.gpu_device_id);
    n_src_local_ = r_src.Dim() / DIM;
    n_trg_local_ = r_trg.Dim() / DIM;
    const sctl::Long n_trg = n_trg_local_;

    { // build the device tree
        sctl::Profile::Scoped profile("build_device_tree", &comm);
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
    }

    // host precompute only, over a copy of the device tree's nodes
    tree_ = std::make_unique<DMKPtTree<Real, DIM>>(comm, params, r_src, charge, normal, r_trg);
    {
        sctl::Profile::Scoped profile("copy_tree_to_host", &comm);
        copyTreeToHost<Real, DIM>(*dev_tree_, n_trg != 0, *tree_);
    }
    { // adopt its nodes
        sctl::Profile::Scoped profile("adopt_nodes", &comm);
        sctl::Long node_begin = 0;
        sctl::Long node_end = 0;
        dev_tree_->OwnedRange(node_begin, node_end);
        tree_->adopt_device_tree(node_begin, node_end);
    }
    { // global leaf status and the sources under each box, over all ranks
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
            const auto &cnt = tree_->r_src_cnt_owned;
            constexpr int n_child = 1 << DIM;
            sctl::ScratchBuf<sctl::Long> off(n_boxes + 1);
            sctl::omp_par::scan(cnt.begin(), off.begin(), n_boxes, sctl::Long(0));
            off[n_boxes] = off[n_boxes - 1] + cnt[n_boxes - 1];
#pragma omp parallel for schedule(static)
            for (sctl::Long i = 0; i < n_boxes; ++i) {
                sctl::Long j = i; // the run ends at the next sibling of i or of its nearest ancestor with one
                while (j > 0 && nl[j].p2n == n_child - 1)
                    j = nl[j].parent;
                const sctl::Long end = j > 0 ? off[nl[nl[j].parent].child[nl[j].p2n + 1]] : off[n_boxes];
                under[i] = end - off[i];
            }
        }
        if (comm.Size() > 1) { // reduced over every holder: refined here, and sources owned under
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
    tree_->device_metadata = true; // the passes guarded by it run on the device just below
    tree_->generate_metadata_for_gpu();
    tree_->init_planewave_data();

    { // per-box geometry, interaction lists and direct-work order, on the device
        sctl::Profile::Scoped profile("device_metadata", &comm);
        const long nb = (long)tree_->n_boxes();
        const long stride = DMKPtTree<Real, DIM>::list1_stride();
        md_ = std::make_unique<DeviceMetadata<Real, DIM>>();
        auto &m = *md_;
        m.centers.resize(nb * DIM);
        m.box_levels.resize(nb);
        m.list1.resize(nb * stride);
        m.list1_count.resize(nb);
        m.direct_work.resize(nb);
        m.self_corr.resize(nb);
        m.cost.resize(nb);
        m.summary.resize(2);
        if (params.use_periodic)
            m.list1_shift.resize(nb * stride * DIM);

        DeviceMetadataInputs<Real, DIM> mi;
        mi.d_mid = dev_tree_->NodeMID().data();
        mi.d_attr = dev_tree_->NodeAttrs().data();
        const auto lists = dev_tree_->Lists();
        mi.d_parent = lists.parent.data();
        mi.d_child = lists.child.data();
        mi.d_nbr = lists.nbr.data();
        // sctl::Vector<bool> stores a byte per element
        mi.is_global_leaf = nb ? reinterpret_cast<const unsigned char *>(&tree_->is_global_leaf[0]) : nullptr;
        mi.ifpwexp = nb ? reinterpret_cast<const unsigned char *>(&tree_->ifpwexp[0]) : nullptr;
        mi.src_cnt_with_halo = &tree_->src_counts_with_halo[0];
        mi.src_cnt_owned = &tree_->src_counts_owned[0];
        mi.trg_cnt_owned = &tree_->trg_counts_owned[0];
        mi.boxsize = &tree_->boxsize[0];
        mi.w0 = tree_->self_w0.Dim() ? &tree_->self_w0[0] : nullptr;
        mi.w0_grad = tree_->self_w0_grad.Dim() ? &tree_->self_w0_grad[0] : nullptr;
        mi.n_boxes = nb;
        mi.n_boxsize = tree_->boxsize.Dim();
        mi.n_w0 = tree_->self_w0.Dim();
        mi.n_levels = tree_->n_levels();
        mi.nlist1_stride = (int)stride;
        mi.periodic = params.use_periodic;
        // Same choice build_self_correction_work_list makes on the host path.
        if (params.kernel == DMK_STRESSLET)
            mi.self_mode = SelfCorrectionMode::zero;
        else if (params.kernel == DMK_LAPLACE_DIPOLE)
            mi.self_mode =
                params.eval_src >= DMK_POTENTIAL_GRAD ? SelfCorrectionMode::w0_grad : SelfCorrectionMode::zero;
        else
            mi.self_mode = SelfCorrectionMode::w0;

        auto &mo = m.out;
        mo.d_centers = m.centers.data();
        mo.d_box_levels = m.box_levels.data();
        mo.d_list1 = m.list1.data();
        mo.d_list1_count = m.list1_count.data();
        mo.d_list1_shift = params.use_periodic ? m.list1_shift.data() : nullptr;
        mo.d_direct_work = m.direct_work.data();
        mo.d_self_correction_work = m.self_corr.data();
        mo.d_cost_scratch = m.cost.data();
        mo.d_summary = m.summary.data();
        compute_device_metadata<Real, DIM>(mi, mo);
    }

    dev_tree_->AddData("proxy_coeffs", 1, tree_->proxy_coeffs_counts); // the tree sums these across ranks
    {                                                                  // build the device state
        sctl::Profile::Scoped profile("state_ctor", &comm);
        auto inputs = to_build_inputs(*tree_, &md_->out);
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
        sctl::Profile::Scoped profile("upload_and_sort_charges", &comm);
        update_charges(charge.Dim() ? &charge[0] : nullptr,
                       (params.kernel == DMK_STRESSLET && normal.Dim()) ? &normal[0] : nullptr);
    }
    cuda_helpers::check_device_errors("tree create");
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
    const auto ds = state_->direct_stream.get();
    const auto ws = state_->downward_stream.get();
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
void Tree<Real, DIM>::update_charges(const Real *charge, const Real *normal) {
    const sctl::Long n_src = n_src_local_;
    const int charge_dof = tree_->kernel_input_dim;
    sctl::Long n_charge = 0;
    sctl::Long n_normal = 0;
    Real *d_charge = scatterParticleData(*dev_tree_, "pdmk_charge", "pdmk_src", charge, n_src * charge_dof, n_charge);
    Real *d_normal = nullptr;
    if (tree_->params.kernel == DMK_STRESSLET)
        d_normal = scatterParticleData(*dev_tree_, "pdmk_normal", "pdmk_src", normal, n_src * DIM, n_normal);
    state_->set_charges(d_charge, n_charge, d_normal, n_normal);
    cuda_helpers::check_device_errors("tree update_charges");
}

template class Tree<float, 2>;
template class Tree<float, 3>;
template class Tree<double, 2>;
template class Tree<double, 3>;

} // namespace dmk::cuda::pt
