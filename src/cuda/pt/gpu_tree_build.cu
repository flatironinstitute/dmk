// The device-resident tree build: the only nvcc-compiled translation unit on the DMK point-tree
// path. It sees thrust and SCTL's experimental GPU tree and nothing of DMK -- nvcc crashes in
// cudafe++ on <sctl.hpp>, which every dmk header pulls in. Do not add a dmk include here; put
// anything that needs one on the far side of gpu_tree_build.hpp.
//
// Particle data rides the tree rather than DMK's own scatter kernels: AddParticleData sorts host
// values into tree order on the device and GetParticleData maps them back, so the permutation
// never has to leave the tree.

#include <cstddef>
#include <set>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <sctl/experimental/gpu-tree.hpp>
#include <sctl/profile.hpp>

#include "gpu_tree_build.hpp"

namespace dmk::cuda::pt {

namespace {

template <typename Real, int DIM>
using DeviceTree = gpu_tree::PtTree<Real, DIM, gpu_tree::DeviceVector>;

/// Copy a device container into host storage, element for element.
template <typename HostVec, typename DevVec>
void to_host(HostVec &dst, const DevVec &src) {
    dst.ReInit((sctl::Long)src.size());
    if (src.size())
        thrust::copy(src.begin(), src.end(), &dst[0]);
}

/// 3^DIM neighbor slots and 2^DIM child slots, the node-list row strides.
template <int DIM>
constexpr int n_nbr_slots() {
    return DIM == 2 ? 9 : 27;
}
template <int DIM>
constexpr int n_child_slots() {
    return 1 << DIM;
}

constexpr int kBlock = 256;

int grid_for(long n) { return (int)((n + kBlock - 1) / kBlock); }

template <typename T>
T *raw(gpu_tree::DeviceVector<T> &v) {
    return thrust::raw_pointer_cast(v.data());
}

/// Hands thrust's temporaries the same pooled device blocks gpu_tree's own vectors use. The default
/// policy cudaMallocs and cudaFrees each one, and cudaFree synchronizes the whole device -- which
/// turns an enqueue-only pass into a blocking one.
struct PoolAlloc {
    using value_type = char;
    char *allocate(std::ptrdiff_t n) {
        return static_cast<char *>(gpu_tree::detail::device_block_alloc((std::size_t)n));
    }
    void deallocate(char *p, std::size_t n) { gpu_tree::detail::device_block_free(p, n); }
};

/// A page-locked host block. Page-locking a few megabytes costs far more than the one transfer it
/// serves, so blocks are pooled and returned on tree destroy instead of freed. Not thread-safe,
/// like SCTL's own device scratch pool; the GPU path is single-rank and builds one tree at a time.
struct PinnedBlock {
    char *p = nullptr;
    std::size_t bytes = 0;
};

std::vector<PinnedBlock> &pinned_pool() {
    static std::vector<PinnedBlock> pool;
    return pool;
}

PinnedBlock pinned_acquire(std::size_t bytes) {
    auto &pool = pinned_pool();
    // Fits without wasting more than half of itself, so an eight-byte request cannot walk off with
    // the multi-megabyte topology block and force it to be page-locked again.
    for (auto it = pool.begin(); it != pool.end(); ++it)
        if (it->bytes >= bytes && it->bytes <= 2 * bytes) {
            const PinnedBlock b = *it;
            pool.erase(it);
            return b;
        }
    PinnedBlock b;
    if (cudaMallocHost((void **)&b.p, bytes) != cudaSuccess)
        return {};
    b.bytes = bytes;
    return b;
}

void pinned_release(PinnedBlock b) {
    if (b.p)
        pinned_pool().push_back(b);
}

/// Rebind `v` as a non-owning view of `n` elements at `p`. sctl::Vector's move-assign falls back to
/// a copy when either side is a view, so the view has to be built by the constructor -- and the
/// point here is to not copy.
template <typename T>
void rebind(sctl::Vector<T> &v, char *p, sctl::Long n) {
    v.~Vector();
    new (&v) sctl::Vector<T>(n, sctl::Ptr2Itr<T>(p, n), /*own_data=*/false);
}

/// Transpose the tree's SoA node lists into the array-of-struct layout sctl::Tree uses, on the
/// device. sctl::Tree<DIM>::NodeLists is POD, so the result copies straight out.
template <int DIM, typename Attr>
__global__ void k_mirror(const sctl::Morton<DIM> *mid, const Attr *attr, const sctl::Long *parent,
                         const sctl::Long *child, const sctl::Long *nbr, long n,
                         typename sctl::Tree<DIM>::NodeLists *out_lists, typename sctl::Tree<DIM>::NodeAttr *out_attr) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    constexpr int NC = n_child_slots<DIM>();
    constexpr int NN = n_nbr_slots<DIM>();
    out_attr[i].Leaf = attr[i].Leaf;
    out_attr[i].Ghost = attr[i].Ghost;
    auto &L = out_lists[i];
    // The GPU tree does not carry p2n. Path2Node() gives it, except at the root, where it
    // returns 0 while sctl::Tree uses -1 -- no parent, so no index among siblings.
    L.p2n = (mid[i].Depth() == 0) ? -1 : mid[i].Path2Node();
    L.parent = parent[i];
    for (int k = 0; k < NC; k++)
        L.child[k] = child[i * NC + k];
    for (int k = 0; k < NN; k++)
        L.nbr[k] = nbr[i * NN + k];
}

} // namespace

template <typename Real, int DIM>
struct GpuTree {
    DeviceTree<Real, DIM> tree;
    gpu_tree::DeviceVector<Real> scratch; ///< staging for a host->device data attach
    /// Destination for the reverse scatter. Persistent because gpu_tree_get_data runs inside the
    /// timed eval, and a cudaMalloc/cudaFree pair per call costs more than the copy itself.
    gpu_tree::DeviceVector<Real> gather;
    /// Data sets attached so far. AddParticleData rejects a name that already exists and
    /// DeleteParticleData rejects one that does not, so the attach has to know which it is --
    /// update_charges re-attaches the same name on every call.
    std::set<std::string> data_names;

    /// Device metadata, owned here because State only aliases it and the tree outlives State.
    gpu_tree::DeviceVector<Real> md_centers, md_self_corr;
    gpu_tree::DeviceVector<int> md_box_levels, md_list1, md_list1_count, md_direct_work;
    gpu_tree::DeviceVector<unsigned char> md_ifpwexp;
    gpu_tree::DeviceVector<signed char> md_list1_shift;
    /// The host AoS topology, assembled on the device before a single copy out.
    gpu_tree::DeviceVector<typename sctl::Tree<DIM>::NodeLists> md_mirror_lists;
    gpu_tree::DeviceVector<typename sctl::Tree<DIM>::NodeAttr> md_mirror_attr;
    /// Staged host inputs to the metadata kernels: per-level geometry and per-box counts.
    gpu_tree::DeviceVector<Real> md_boxsize, md_w0;
    gpu_tree::DeviceVector<int> md_src_cnt, md_trg_cnt;
    /// Scratch for the direct-work sort. Persistent because a fresh device_vector per build means a
    /// cudaMalloc plus a cudaFree, and cudaFree synchronizes the whole device.
    gpu_tree::DeviceVector<long> md_cost;
    gpu_tree::DeviceVector<int> md_summary; ///< {work-list length, shallowest level}, device side
    /// Landing pad for the two scalars. Page-locked because a copy into pageable memory is
    /// synchronous however it is spelled, which would undo the split.
    PinnedBlock summary;
    /// Page-locked landing pad for the topology copy-out. The host mirror is a view onto it, so it
    /// is held for the tree's whole life, not just the transfer.
    PinnedBlock pinned;

    GpuTree() : tree(sctl::Comm::Self()) { summary = pinned_acquire(2 * sizeof(int)); }
    ~GpuTree() {
        pinned_release(pinned);
        pinned_release(summary);
    }
    int *summary_host() { return reinterpret_cast<int *>(summary.p); }

    void drop(const std::string &name) {
        if (data_names.erase(name))
            tree.DeleteParticleData(name);
    }
};

template <typename Real, int DIM>
GpuTree<Real, DIM> *gpu_tree_create(const Real *r_src, sctl::Long n_src, const Real *r_trg, sctl::Long n_trg,
                                    sctl::Long n_per_leaf, bool periodic, GpuTreeTopology<DIM> &topology,
                                    GpuTreeParticles<Real> &particles) {
    auto *self = new GpuTree<Real, DIM>();
    auto &tree = self->tree;

    sctl::Profile::Tic("gtc_coord_alloc");
    gpu_tree::DeviceVector<Real> d_src, d_trg;
    d_src.resize(n_src * DIM);
    d_trg.resize(n_trg * DIM);
    sctl::Profile::Toc();
    sctl::Profile::Tic("gtc_coord_h2d");
    // One cudaMemcpy per side rather than thrust::copy over device_vector iterators, which runs
    // this well under the link's pageable rate.
    if (n_src)
        cudaMemcpy(raw(d_src), r_src, n_src * DIM * sizeof(Real), cudaMemcpyHostToDevice);
    if (n_trg)
        cudaMemcpy(raw(d_trg), r_trg, n_trg * DIM * sizeof(Real), cudaMemcpyHostToDevice);
    sctl::Profile::Toc();
    const sctl::Periodicity per = periodic ? sctl::all_periodic(DIM) : sctl::Periodicity::NONE;

    // Refinement first, then the groups, which is the order gpu_tree::PtTree::test documents; the
    // sources drive the refinement exactly as they do on the host path.
    sctl::Profile::Tic("gtc_refine");
    tree.UpdateRefinement(d_src, n_per_leaf, /*balance21=*/true, per, /*halo_size=*/0);
    sctl::Profile::Toc();
    sctl::Profile::Tic("gtc_add_particles");
    tree.AddParticles("pdmk_src", d_src);
    tree.AddParticles("pdmk_trg", d_trg);
    sctl::Profile::Toc();

    { // topology, into the array-of-struct layout DMK's metadata routines read
        using Mid = sctl::Morton<DIM>;
        using Attr = typename sctl::Tree<DIM>::NodeAttr;
        using Lists = typename sctl::Tree<DIM>::NodeLists;
        const sctl::Long n = (sctl::Long)tree.GetNodeMID().size();

        sctl::Profile::Tic("gtc_topo_repack");
        self->md_mirror_lists.resize(n);
        self->md_mirror_attr.resize(n);
        const auto &NL = tree.GetNodeLists();
        k_mirror<DIM, typename DeviceTree<Real, DIM>::NodeAttr><<<grid_for(n), kBlock>>>(
            thrust::raw_pointer_cast(tree.GetNodeMID().data()), thrust::raw_pointer_cast(tree.GetNodeAttr().data()),
            thrust::raw_pointer_cast(NL.parent.data()), thrust::raw_pointer_cast(NL.child.data()),
            thrust::raw_pointer_cast(NL.nbr.data()), n, raw(self->md_mirror_lists), raw(self->md_mirror_attr));
        sctl::Profile::Toc();

        sctl::Profile::Tic("gtc_topo_d2h");
        // One page-locked landing pad for all three arrays, which the host mirror then views in
        // place. A pageable copy of this (~300 B/box) runs at a third of link speed and cannot be
        // asynchronous, and copying out of the pad again would cost as much as the transfer.
        const auto align = [](std::size_t x) { return (x + 255) & ~(std::size_t)255; };
        const std::size_t off_mid = 0;
        const std::size_t off_attr = align(off_mid + n * sizeof(Mid));
        const std::size_t off_lists = align(off_attr + n * sizeof(Attr));
        const std::size_t need = align(off_lists + n * sizeof(Lists));
        if (self->pinned.bytes < need) {
            pinned_release(self->pinned);
            self->pinned = pinned_acquire(need);
        }
        if (self->pinned.p) {
            char *const pad = self->pinned.p;
            cudaMemcpyAsync(pad + off_mid, thrust::raw_pointer_cast(tree.GetNodeMID().data()), n * sizeof(Mid),
                            cudaMemcpyDeviceToHost);
            cudaMemcpyAsync(pad + off_attr, raw(self->md_mirror_attr), n * sizeof(Attr), cudaMemcpyDeviceToHost);
            cudaMemcpyAsync(pad + off_lists, raw(self->md_mirror_lists), n * sizeof(Lists), cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(0);
            rebind(topology.node_mid, pad + off_mid, n);
            rebind(topology.node_attr, pad + off_attr, n);
            rebind(topology.node_lists, pad + off_lists, n);
        } else { // page-locking failed; the pageable path is slower but correct
            to_host(topology.node_mid, tree.GetNodeMID());
            to_host(topology.node_attr, self->md_mirror_attr);
            to_host(topology.node_lists, self->md_mirror_lists);
        }
        sctl::Profile::Toc();

        tree.GetOwnedRange(topology.owned_begin, topology.owned_end);
    }

    sctl::Profile::Tic("gtc_particle_counts");
    { // particles: coordinates stay on the device, only the per-box counts come back
        gpu_tree::DataView<Real, gpu_tree::DeviceVector> view;
        tree.GetData(view, particles.src_cnt, "pdmk_src");
        particles.d_r_src = view.data();
        particles.n_src = n_src;
        tree.GetData(view, particles.trg_cnt, "pdmk_trg");
        particles.d_r_trg = view.data();
        particles.n_trg = n_trg;
    }
    sctl::Profile::Toc();

    return self;
}

template <typename Real, int DIM>
void gpu_tree_destroy(GpuTree<Real, DIM> *tree) {
    delete tree;
}

template <typename Real, int DIM>
Real *gpu_tree_add_data(GpuTree<Real, DIM> *tree, const std::string &name, const std::string &group,
                        const Real *host_values, int dof) {
    gpu_tree::DataView<Real, gpu_tree::DeviceVector> view;
    sctl::Vector<sctl::Long> cnt;
    tree->tree.GetData(view, cnt, group); // the group's own coordinates, to size the attach
    const sctl::Long n = sctl::omp_par::reduce(cnt.begin(), cnt.Dim());

    tree->scratch.resize(n * dof);
    cudaMemcpy(raw(tree->scratch), host_values, n * dof * sizeof(Real), cudaMemcpyHostToDevice);
    tree->drop(name);
    tree->tree.AddParticleData(name, group, tree->scratch);
    tree->data_names.insert(name);
    return gpu_tree_data<Real, DIM>(tree, name);
}

template <typename Real, int DIM>
Real *gpu_tree_reserve_data(GpuTree<Real, DIM> *tree, const std::string &name, const std::string &group, int dof) {
    tree->drop(name);
    tree->tree.AddParticleData(name, group, (sctl::Long)dof);
    tree->data_names.insert(name);
    return gpu_tree_data<Real, DIM>(tree, name);
}

template <typename Real, int DIM>
Real *gpu_tree_data(GpuTree<Real, DIM> *tree, const std::string &name) {
    gpu_tree::DataView<Real, gpu_tree::DeviceVector> view;
    sctl::Vector<sctl::Long> cnt;
    tree->tree.GetData(view, cnt, name);
    return view.data();
}

template <typename Real, int DIM>
void gpu_tree_delete_data(GpuTree<Real, DIM> *tree, const std::string &name) {
    tree->drop(name); // tolerant: a name never attached is simply not there
}

template <typename Real, int DIM>
void gpu_tree_get_data(GpuTree<Real, DIM> *tree, const std::string &name, Real *host_out) {
    tree->tree.GetParticleData(tree->gather, name);
    if (tree->gather.size())
        thrust::copy(tree->gather.begin(), tree->gather.end(), host_out);
}

// ---------------------------------------------------------------------------------------------
// Device metadata. These reproduce five DMKPtTree host routines (see gpu_tree_metadata) against
// the tree's own device node lists, so the per-box traversals never round-trip through the host.
// The list1 rows are built in the same order as the host loops, which keeps them comparable.

namespace {

/// Upload `n` host values, sizing the device buffer to match.
template <typename T>
const T *stage(gpu_tree::DeviceVector<T> &dst, const T *src, long n) {
    if (!n)
        return nullptr;
    dst.resize(n);
    cudaMemcpy(raw(dst), src, n * sizeof(T), cudaMemcpyHostToDevice);
    return raw(dst);
}

/// The periodic image offset of neighbor slot `k`, from the slot's direction and the two centers.
/// Mirrors compute_periodic_shift_from_slot; double throughout, as the host does.
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

template <typename Real, int DIM>
__global__ void k_centers_levels(const sctl::Morton<DIM> *mid, long n_boxes, const Real *boxsize, Real *centers,
                                 int *levels) {
    const long b = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (b >= n_boxes)
        return;
    const int lvl = mid[b].Depth();
    levels[b] = lvl;
    Real origin[DIM];
    mid[b].Coord(origin);
    const Real half = Real(0.5) * boxsize[lvl];
    for (int d = 0; d < DIM; d++)
        centers[b * DIM + d] = origin[d] + half;
}

template <int DIM, typename Attr>
__global__ void k_ifpwexp(const sctl::Long *nbr, const Attr *attr, long n_boxes, unsigned char *ifpwexp) {
    const long b = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (b >= n_boxes)
        return;
    constexpr int NN = n_nbr_slots<DIM>();
    if (!attr[b].Leaf) {
        ifpwexp[b] = 1;
        return;
    }
    // The root always carries an expansion, which is why the host sets it before its loop.
    unsigned char v = (b == 0) ? 1 : 0;
    for (int k = 0; k < NN; k++) {
        const sctl::Long nb = nbr[b * NN + k];
        if (nb >= 0 && !attr[nb].Leaf) {
            v = 1;
            break;
        }
    }
    ifpwexp[b] = v;
}

template <typename Real, int DIM, typename Attr>
__global__ void k_list1(const sctl::Morton<DIM> *mid, const Attr *attr, const sctl::Long *parent,
                        const sctl::Long *child, const sctl::Long *nbr, const int *src_cnt, const Real *centers,
                        const Real *boxsize, long n_boxes, int stride, bool periodic, int *list1, int *count,
                        signed char *shift_out) {
    const long box = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (box >= n_boxes)
        return;
    constexpr int NN = n_nbr_slots<DIM>();
    constexpr int NC = n_child_slots<DIM>();

    count[box] = 0;
    if (!attr[box].Leaf || attr[box].Ghost)
        return;

    const int lvl = mid[box].Depth();
    const Real bsize = boxsize[lvl];
    const double cutoff_child = 1.05 * 0.75 * (double)bsize;
    const double cutoff_parent_nbr = 1.5 * 1.05 * (double)bsize;
    const Real *c_box = centers + box * DIM;

    int k = 0;
    auto add = [&](sctl::Long nb, const int *shift) {
        list1[box * stride + k] = (int)nb;
        if (shift_out)
            for (int d = 0; d < DIM; d++)
                shift_out[(box * (long)stride + k) * DIM + d] = (signed char)shift[d];
        k++;
    };

    // Same-level neighbors: leaf neighbors directly; otherwise their children that touch box.
    for (int nk = 0; nk < NN; nk++) {
        const sctl::Long nb = nbr[box * NN + nk];
        if (nb < 0)
            continue;
        int shift[DIM] = {};
        if (periodic)
            shift_from_slot<Real, DIM>(nk, bsize, c_box, centers + nb * DIM, shift);
        if (attr[nb].Leaf) {
            if (src_cnt[nb])
                add(nb, shift);
        } else {
            for (int c = 0; c < NC; c++) {
                const sctl::Long ch = child[nb * NC + c];
                if (ch >= 0 && src_cnt[ch] && within_cutoff<Real, DIM>(c_box, centers + ch * DIM, shift, cutoff_child))
                    add(ch, shift);
            }
        }
    }

    // Parent's neighbors: coarser leaf boxes that cannot appear as same-level neighbors.
    if (lvl != 0) {
        const sctl::Long par = parent[box];
        for (int nk = 0; nk < NN; nk++) {
            const sctl::Long nb = nbr[par * NN + nk];
            if (nb < 0 || !attr[nb].Leaf || !src_cnt[nb])
                continue;
            int shift[DIM] = {};
            if (periodic)
                shift_from_slot<Real, DIM>(nk, boxsize[lvl - 1], centers + par * DIM, centers + nb * DIM, shift);
            if (within_cutoff<Real, DIM>(c_box, centers + nb * DIM, shift, cutoff_parent_nbr))
                add(nb, shift);
        }
    }
    count[box] = k;
}

/// Per-box near-field cost, and whether the box has near-field work at all.
template <int DIM, typename Attr>
__global__ void k_direct_work_cost(const Attr *attr, const int *src_cnt, const int *trg_cnt, const int *list1,
                                   const int *count, long n_boxes, int stride, long *cost) {
    const long box = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (box >= n_boxes)
        return;
    const long own = src_cnt[box] + trg_cnt[box];
    if (!(attr[box].Leaf && !attr[box].Ghost && count[box] > 0 && own > 0)) {
        // Excluded boxes take a cost below every real one, so the descending sort leaves them past
        // the end of the work list and no host round-trip is needed to compact first.
        cost[box] = -1;
        return;
    }
    long src = 0;
    for (int i = 0; i < count[box]; i++)
        src += src_cnt[list1[box * stride + i]];
    cost[box] = own * src;
}

/// Work-list length and its shallowest level, in one block so both come back in a single transfer.
__global__ void k_direct_work_summary(const long *cost, const int *box_levels, const int *direct_work, long n_boxes,
                                      int n_levels, int *out) {
    __shared__ int s_n[256], s_lvl[256];
    int n = 0, lvl = n_levels;
    for (long i = threadIdx.x; i < n_boxes; i += blockDim.x)
        if (cost[i] >= 0) {
            n++;
            const int l = box_levels[direct_work[i]];
            lvl = l < lvl ? l : lvl;
        }
    s_n[threadIdx.x] = n;
    s_lvl[threadIdx.x] = lvl;
    __syncthreads();
    for (int half = blockDim.x / 2; half; half >>= 1) {
        if (threadIdx.x < half) {
            s_n[threadIdx.x] += s_n[threadIdx.x + half];
            s_lvl[threadIdx.x] = min(s_lvl[threadIdx.x], s_lvl[threadIdx.x + half]);
        }
        __syncthreads();
    }
    if (!threadIdx.x) {
        out[0] = s_n[0];
        out[1] = s_lvl[0];
    }
}

template <typename Real, int DIM>
__global__ void k_self_correction(const int *direct_work, long n_dw, const sctl::Morton<DIM> *mid,
                                  const unsigned char *ifpwexp, const Real *w0, Real *out) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n_dw)
        return;
    const int box = direct_work[i];
    // Must match the level the direct residual uses.
    out[i] = w0 ? w0[mid[box].Depth() + ifpwexp[box]] : Real{0};
}

} // namespace

template <typename Real, int DIM>
void gpu_tree_metadata(GpuTree<Real, DIM> *self, const GpuTreeMetadataParams<Real> &params,
                       GpuTreeMetadata<Real> &out) {
    auto &tree = self->tree;
    const long n_boxes = (long)tree.GetNodeMID().size();
    const int stride = params.nlist1_stride;
    using Attr = typename DeviceTree<Real, DIM>::NodeAttr;

    sctl::Profile::Tic("md_stage");
    const Real *d_boxsize = stage(self->md_boxsize, params.boxsize, params.n_boxsize);
    const int *d_src_cnt = stage(self->md_src_cnt, params.src_cnt, n_boxes);
    const int *d_trg_cnt = stage(self->md_trg_cnt, params.trg_cnt, n_boxes);

    sctl::Profile::Toc();
    sctl::Profile::Tic("md_kernels");
    const auto *mid = thrust::raw_pointer_cast(tree.GetNodeMID().data());
    const auto *attr = thrust::raw_pointer_cast(tree.GetNodeAttr().data());
    const auto &L = tree.GetNodeLists();
    const auto *parent = thrust::raw_pointer_cast(L.parent.data());
    const auto *child = thrust::raw_pointer_cast(L.child.data());
    const auto *nbr = thrust::raw_pointer_cast(L.nbr.data());

    self->md_centers.resize(n_boxes * DIM);
    self->md_box_levels.resize(n_boxes);
    k_centers_levels<Real, DIM>
        <<<grid_for(n_boxes), kBlock>>>(mid, n_boxes, d_boxsize, raw(self->md_centers), raw(self->md_box_levels));

    self->md_ifpwexp.resize(n_boxes);
    k_ifpwexp<DIM, Attr><<<grid_for(n_boxes), kBlock>>>(nbr, attr, n_boxes, raw(self->md_ifpwexp));

    self->md_list1.resize(n_boxes * (long)stride);
    self->md_list1_count.resize(n_boxes);
    self->md_list1_shift.resize(params.periodic ? n_boxes * (long)stride * DIM : 0);
    k_list1<Real, DIM, Attr><<<grid_for(n_boxes), kBlock>>>(
        mid, attr, parent, child, nbr, d_src_cnt, raw(self->md_centers), d_boxsize, n_boxes, stride, params.periodic,
        raw(self->md_list1), raw(self->md_list1_count), params.periodic ? raw(self->md_list1_shift) : nullptr);

    sctl::Profile::Toc();
    sctl::Profile::Tic("md_directwork");
    { // direct work list, heaviest first
        auto &cost = self->md_cost;
        cost.resize(n_boxes);
        k_direct_work_cost<DIM, Attr><<<grid_for(n_boxes), kBlock>>>(
            attr, d_src_cnt, d_trg_cnt, raw(self->md_list1), raw(self->md_list1_count), n_boxes, stride, raw(cost));
        // Sort every box rather than compacting first: compaction would have to report its length to
        // the host, and that round-trip is the only thing standing between this routine and running
        // entirely behind to_build_inputs. Excluded boxes carry cost -1 and land past the end.
        //
        // The host sorts with std::sort, which is unstable, so only the descending order itself is
        // part of the contract -- ties may land in either order on either path.
        self->md_direct_work.resize(n_boxes);
        PoolAlloc alloc;
        thrust::sequence(thrust::cuda::par(alloc), self->md_direct_work.begin(), self->md_direct_work.end());
        thrust::stable_sort_by_key(thrust::cuda::par(alloc), cost.begin(), cost.end(), self->md_direct_work.begin(),
                                   thrust::greater<long>());

        self->md_summary.resize(2);
        k_direct_work_summary<<<1, 256>>>(raw(cost), raw(self->md_box_levels), raw(self->md_direct_work), n_boxes,
                                          params.n_levels, raw(self->md_summary));
    }

    sctl::Profile::Toc();

    sctl::Profile::Tic("md_selfcorr");
    { // self-correction factor, in direct_work order
        self->md_self_corr.resize(n_boxes);
        const Real *w0_host = params.self_mode == SelfCorrectionMode::w0
                                  ? params.w0
                                  : (params.self_mode == SelfCorrectionMode::w0_grad ? params.w0_grad : nullptr);
        const Real *d_w0 = stage(self->md_w0, w0_host, w0_host ? params.n_w0 : 0);
        // Over every box, not just the work list: its length is still on the device, and the entries
        // past the end are never read.
        k_self_correction<Real, DIM><<<grid_for(n_boxes), kBlock>>>(
            raw(self->md_direct_work), n_boxes, mid, raw(self->md_ifpwexp), d_w0, raw(self->md_self_corr));
    }

    // One 8-byte readback, started here and waited for in gpu_tree_metadata_finish. Everything above
    // is enqueued, so the whole routine costs launch time only.
    if (self->summary_host())
        cudaMemcpyAsync(self->summary_host(), raw(self->md_summary), 2 * sizeof(int), cudaMemcpyDeviceToHost);
    sctl::Profile::Toc();

    out.d_centers = raw(self->md_centers);
    out.d_box_levels = raw(self->md_box_levels);
    out.d_ifpwexp = raw(self->md_ifpwexp);
    out.d_list1_flat = raw(self->md_list1);
    out.d_list1_count = raw(self->md_list1_count);
    out.d_list1_shift = params.periodic ? raw(self->md_list1_shift) : nullptr;
    out.d_direct_work = raw(self->md_direct_work);
    out.d_self_correction_work = raw(self->md_self_corr);
}

template <typename Real, int DIM>
void gpu_tree_metadata_finish(GpuTree<Real, DIM> *self, GpuTreeMetadata<Real> &out) {
    cudaStreamSynchronize(0);
    if (!self->summary_host()) { // page-locking failed at construction; read it the slow way
        int h[2] = {0, 0};
        cudaMemcpy(h, thrust::raw_pointer_cast(self->md_summary.data()), 2 * sizeof(int), cudaMemcpyDeviceToHost);
        out.n_direct_work = h[0];
        out.min_direct_level = h[1];
        return;
    }
    out.n_direct_work = self->summary_host()[0];
    out.min_direct_level = self->summary_host()[1];
}

// DMKPtTree is built for DIM=2 and DIM=3 in both precisions, so all four are linker-required.
#define DMK_INSTANTIATE_GPU_TREE(Real, DIM)                                                                            \
    template GpuTree<Real, DIM> *gpu_tree_create<Real, DIM>(const Real *, sctl::Long, const Real *, sctl::Long,        \
                                                            sctl::Long, bool, GpuTreeTopology<DIM> &,                  \
                                                            GpuTreeParticles<Real> &);                                 \
    template void gpu_tree_destroy<Real, DIM>(GpuTree<Real, DIM> *);                                                   \
    template Real *gpu_tree_add_data<Real, DIM>(GpuTree<Real, DIM> *, const std::string &, const std::string &,        \
                                                const Real *, int);                                                    \
    template Real *gpu_tree_reserve_data<Real, DIM>(GpuTree<Real, DIM> *, const std::string &, const std::string &,    \
                                                    int);                                                              \
    template Real *gpu_tree_data<Real, DIM>(GpuTree<Real, DIM> *, const std::string &);                                \
    template void gpu_tree_delete_data<Real, DIM>(GpuTree<Real, DIM> *, const std::string &);                          \
    template void gpu_tree_get_data<Real, DIM>(GpuTree<Real, DIM> *, const std::string &, Real *);                     \
    template void gpu_tree_metadata<Real, DIM>(GpuTree<Real, DIM> *, const GpuTreeMetadataParams<Real> &,              \
                                               GpuTreeMetadata<Real> &);                                               \
    template void gpu_tree_metadata_finish<Real, DIM>(GpuTree<Real, DIM> *, GpuTreeMetadata<Real> &);

DMK_INSTANTIATE_GPU_TREE(float, 2)
DMK_INSTANTIATE_GPU_TREE(float, 3)
DMK_INSTANTIATE_GPU_TREE(double, 2)
DMK_INSTANTIATE_GPU_TREE(double, 3)

#undef DMK_INSTANTIATE_GPU_TREE

} // namespace dmk::cuda::pt
