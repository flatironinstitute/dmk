// The kernels behind metadata.hpp. nvcc compiles this one unit; it takes the tree's node arrays as
// raw device pointers, so it needs nothing from SCTL but Morton and Long.

#include "metadata.hpp"

#include <sctl/experimental/gpu-vector.hpp> // gpu_tree::detail's device block pool

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#define DMK_CUDA_OK(call)                                                                                              \
    do {                                                                                                               \
        const cudaError_t err_ = (call);                                                                               \
        if (err_ != cudaSuccess) {                                                                                     \
            std::fprintf(stderr, "dmk metadata: %s failed: %s\n", #call, cudaGetErrorString(err_));                    \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

namespace dmk::cuda::pt {
namespace {

using Attr = device_tree::NodeAttr;

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
/// cudaFree synchronizes the whole device.
struct PoolAlloc {
    using value_type = char;
    char *allocate(std::ptrdiff_t n) {
        return static_cast<char *>(gpu_tree::detail::device_block_alloc((std::size_t)n));
    }
    void deallocate(char *p, std::size_t n) { gpu_tree::detail::device_block_free(p, n); }
};

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

template <typename Real, int DIM>
__global__ void k_list1(const sctl::Morton<DIM> *mid, const Attr *attr, const unsigned char *leaf,
                        const sctl::Long *parent, const sctl::Long *child, const sctl::Long *nbr, const int *src_cnt,
                        const Real *centers, const Real *boxsize, long n_boxes, int stride, bool periodic, int *list1,
                        int *count, signed char *shift_out) {
    const long box = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (box >= n_boxes)
        return;
    constexpr int NN = n_nbr_slots<DIM>();
    constexpr int NC = n_child_slots<DIM>();

    count[box] = 0;
    if (!leaf[box] || attr[box].Ghost)
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
        if (leaf[nb]) {
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
            if (nb < 0 || !leaf[nb] || !src_cnt[nb])
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
__global__ void k_direct_work_cost(const Attr *attr, const unsigned char *leaf, const int *src_cnt_halo,
                                   const int *src_cnt_owned, const int *trg_cnt_owned, const int *list1,
                                   const int *count, long n_boxes, int stride, long *cost) {
    const long box = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (box >= n_boxes)
        return;
    const long own = src_cnt_owned[box] + trg_cnt_owned[box];
    if (!(leaf[box] && !attr[box].Ghost && count[box] > 0 && own > 0)) {
        // Excluded boxes take a cost below every real one, so the descending sort leaves them past
        // the end of the work list and no host round-trip is needed to compact first.
        cost[box] = -1;
        return;
    }
    long src = 0;
    for (int i = 0; i < count[box]; i++)
        src += src_cnt_halo[list1[box * stride + i]];
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

/// Upload a host array into a scratch device block held for the duration of the call.
template <typename T>
struct Scratch {
    T *p = nullptr;
    std::size_t bytes = 0;
    explicit Scratch(const T *host, long n) {
        if (!host || !n)
            return;
        bytes = (std::size_t)n * sizeof(T);
        p = static_cast<T *>(gpu_tree::detail::device_block_alloc(bytes));
        DMK_CUDA_OK(cudaMemcpyAsync(p, host, bytes, cudaMemcpyHostToDevice, 0));
    }
    ~Scratch() {
        if (p)
            gpu_tree::detail::device_block_free(p, bytes);
    }
    Scratch(const Scratch &) = delete;
    Scratch &operator=(const Scratch &) = delete;
};

} // namespace

template <typename Real, int DIM>
void compute_device_metadata(const DeviceMetadataInputs<Real, DIM> &in, DeviceMetadataOutputs<Real> &out) {
    const long n_boxes = in.n_boxes;
    if (!n_boxes) {
        out.n_direct_work = 0;
        out.min_direct_level = in.n_levels;
        return;
    }
    const int stride = in.nlist1_stride;
    constexpr int BLOCK = 256;
    const long grid = (n_boxes + BLOCK - 1) / BLOCK;

    // The per-box host quantities the kernels need. Small next to the node arrays, which are
    // already resident and never leave the device.
    const Scratch<Real> boxsize(in.boxsize, in.n_boxsize);
    const Scratch<unsigned char> leaf(in.is_global_leaf, n_boxes);
    const Scratch<unsigned char> ifpwexp(in.ifpwexp, n_boxes);
    const Scratch<int> src_halo(in.src_cnt_with_halo, n_boxes);
    const Scratch<int> src_owned(in.src_cnt_owned, n_boxes);
    const Scratch<int> trg_owned(in.trg_cnt_owned, n_boxes);

    const Real *w0_host = in.self_mode == SelfCorrectionMode::w0
                              ? in.w0
                              : (in.self_mode == SelfCorrectionMode::w0_grad ? in.w0_grad : nullptr);
    const Scratch<Real> w0(w0_host, w0_host ? in.n_w0 : 0);

    k_centers_levels<Real, DIM><<<grid, BLOCK>>>(in.d_mid, n_boxes, boxsize.p, out.d_centers, out.d_box_levels);
    k_list1<Real, DIM><<<grid, BLOCK>>>(in.d_mid, in.d_attr, leaf.p, in.d_parent, in.d_child, in.d_nbr, src_halo.p,
                                        out.d_centers, boxsize.p, n_boxes, stride, in.periodic, out.d_list1,
                                        out.d_list1_count, in.periodic ? out.d_list1_shift : nullptr);
    k_direct_work_cost<<<grid, BLOCK>>>(in.d_attr, leaf.p, src_halo.p, src_owned.p, trg_owned.p, out.d_list1,
                                        out.d_list1_count, n_boxes, stride, out.d_cost_scratch);

    // Heaviest box first, so the launch's early blocks carry the long tails. Excluded boxes sort
    // past the end by their -1 cost, so the prefix of length n_direct_work is the work list.
    PoolAlloc alloc;
    thrust::sequence(thrust::cuda::par(alloc).on(0), out.d_direct_work, out.d_direct_work + n_boxes);
    thrust::stable_sort_by_key(thrust::cuda::par(alloc).on(0), out.d_cost_scratch, out.d_cost_scratch + n_boxes,
                               out.d_direct_work, thrust::greater<long>());

    k_direct_work_summary<<<1, 256>>>(out.d_cost_scratch, out.d_box_levels, out.d_direct_work, n_boxes, in.n_levels,
                                      out.d_summary);

    int summary[2] = {0, in.n_levels};
    DMK_CUDA_OK(cudaMemcpyAsync(summary, out.d_summary, 2 * sizeof(int), cudaMemcpyDeviceToHost, 0));
    DMK_CUDA_OK(cudaStreamSynchronize(0));
    out.n_direct_work = summary[0];
    out.min_direct_level = summary[1];

    if (out.n_direct_work) {
        const long grid_dw = (out.n_direct_work + BLOCK - 1) / BLOCK;
        k_self_correction<Real, DIM><<<grid_dw, BLOCK>>>(out.d_direct_work, out.n_direct_work, in.d_mid, ifpwexp.p,
                                                         w0.p, out.d_self_correction_work);
    }
    DMK_CUDA_OK(cudaStreamSynchronize(0));
}

#define DMK_INSTANTIATE_METADATA(Real, DIM)                                                                            \
    template void compute_device_metadata<Real, DIM>(const DeviceMetadataInputs<Real, DIM> &,                          \
                                                     DeviceMetadataOutputs<Real> &);

DMK_INSTANTIATE_METADATA(float, 2)
DMK_INSTANTIATE_METADATA(float, 3)
DMK_INSTANTIATE_METADATA(double, 2)
DMK_INSTANTIATE_METADATA(double, 3)

} // namespace dmk::cuda::pt
