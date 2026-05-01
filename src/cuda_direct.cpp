// Orchestration for the GPU offload of direct (near-field residual)
// interactions. See include/dmk/cuda_direct.hpp for the lifecycle.
//
// This file is plain C++: it allocates/copies via the CUDA runtime API and
// dispatches the kernel through cuda::launch_direct_by_box_dispatch (defined
// in src/cuda_kernels.cu, compiled by nvcc). No <<<>>> launch syntax here.

#include <dmk/cuda_direct.hpp>
#include <dmk/cuda_direct_kernels.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/tree.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace dmk {

namespace {

#define DMK_CHECK_CUDA(expr)                                                                                           \
    do {                                                                                                               \
        cudaError_t _e = (expr);                                                                                       \
        if (_e != cudaSuccess)                                                                                         \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e));                            \
    } while (0)

template <typename T>
T *device_alloc(std::size_t n) {
    if (n == 0)
        return nullptr;
    T *d = nullptr;
    DMK_CHECK_CUDA(cudaMalloc(&d, n * sizeof(T)));
    return d;
}

template <typename T>
T *device_alloc_and_zero(std::size_t n) {
    T *d = device_alloc<T>(n);
    if (d)
        DMK_CHECK_CUDA(cudaMemsetAsync(d, 0, n * sizeof(T)));
    return d;
}

template <typename T>
T *device_upload(const T *src_host, std::size_t n) {
    T *d = device_alloc<T>(n);
    if (d)
        DMK_CHECK_CUDA(cudaMemcpy(d, src_host, n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

void device_free(void *p) {
    if (p)
        cudaFree(p);
}

template <typename SctlVecInt>
std::vector<int> sctl_int_vec_to_std(const SctlVecInt &v) {
    std::vector<int> out(v.Dim());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = (int)v[i];
    return out;
}

} // namespace

template <typename Real, int DIM>
struct CudaDirectContext<Real, DIM>::Impl {
    DMKPtTree<Real, DIM> &tree;

    // Topology + per-box metadata
    int *d_direct_work = nullptr;
    int *d_list1_flat = nullptr;
    int *d_list1_count = nullptr;
    int *d_box_levels = nullptr;
    unsigned char *d_ifpwexp = nullptr;

    // Per-level direct-eval params
    Real *d_direct_rsc = nullptr;
    Real *d_direct_cen = nullptr;
    Real *d_direct_d2max = nullptr;

    // Source data (with halo)
    Real *d_r_src_halo = nullptr;
    long *d_r_src_halo_offsets = nullptr;
    int *d_src_counts_halo = nullptr;

    // For non-stresslet kernels this holds charges; for stresslet it holds densities
    Real *d_charge_halo = nullptr;
    long *d_charge_halo_offsets = nullptr;

    // Stresslet only
    Real *d_normal_halo = nullptr;
    long *d_normal_halo_offsets = nullptr;

    // Owned source positions (target points for the pot_src side)
    Real *d_r_src_owned = nullptr;
    long *d_r_src_owned_offsets = nullptr;
    int *d_src_counts_owned = nullptr;

    // Owned target positions (target points for the pot_trg side)
    Real *d_r_trg_owned = nullptr;
    long *d_r_trg_owned_offsets = nullptr;
    int *d_trg_counts_owned = nullptr;

    // Outputs
    Real *d_pot_src = nullptr;
    long *d_pot_src_offsets = nullptr;
    Real *d_pot_trg = nullptr;
    long *d_pot_trg_offsets = nullptr;

    std::size_t pot_src_size = 0;
    std::size_t pot_trg_size = 0;

    int nlist1_stride = 0;
    int n_work = 0;
    int n_levels = 0;
    bool launched = false;

    explicit Impl(DMKPtTree<Real, DIM> &t) : tree(t) {
        // Mirrors the formula in tree.hpp's nlist1_max_.
        nlist1_stride = (1 << (2 * DIM)) - (1 << DIM) + 1;
    }

    ~Impl() {
        device_free(d_direct_work);
        device_free(d_list1_flat);
        device_free(d_list1_count);
        device_free(d_box_levels);
        device_free(d_ifpwexp);
        device_free(d_direct_rsc);
        device_free(d_direct_cen);
        device_free(d_direct_d2max);
        device_free(d_r_src_halo);
        device_free(d_r_src_halo_offsets);
        device_free(d_src_counts_halo);
        device_free(d_charge_halo);
        device_free(d_charge_halo_offsets);
        device_free(d_normal_halo);
        device_free(d_normal_halo_offsets);
        device_free(d_r_src_owned);
        device_free(d_r_src_owned_offsets);
        device_free(d_src_counts_owned);
        device_free(d_r_trg_owned);
        device_free(d_r_trg_owned_offsets);
        device_free(d_trg_counts_owned);
        device_free(d_pot_src);
        device_free(d_pot_src_offsets);
        device_free(d_pot_trg);
        device_free(d_pot_trg_offsets);
    }
};

template <typename Real, int DIM>
CudaDirectContext<Real, DIM>::CudaDirectContext(DMKPtTree<Real, DIM> &tree) : pimpl_(std::make_unique<Impl>(tree)) {}

template <typename Real, int DIM>
CudaDirectContext<Real, DIM>::~CudaDirectContext() = default;

template <typename Real, int DIM>
void CudaDirectContext<Real, DIM>::launch() {
    auto &t = pimpl_->tree;
    auto &im = *pimpl_;

    if (t.params.use_periodic)
        throw std::runtime_error("CUDA direct: periodic boundary conditions are not yet supported");
    if (t.params.kernel == DMK_YUKAWA)
        throw std::runtime_error("CUDA direct: Yukawa kernel is not yet supported on the GPU path");

    const auto &node_mid = t.GetNodeMID();
    const std::size_t n_boxes = t.n_boxes();

    // ---------- host-side metadata buffers ----------
    std::vector<int> direct_work_h(t.direct_work.begin(), t.direct_work.end());
    im.n_work = (int)direct_work_h.size();
    im.n_levels = t.boxsize.Dim();

    std::vector<int> list1_flat_h(n_boxes * im.nlist1_stride, -1);
    std::vector<int> list1_count_h(n_boxes, 0);
    for (std::size_t b = 0; b < n_boxes; ++b) {
        const auto sp = t.list1((int)b);
        list1_count_h[b] = (int)sp.size();
        for (std::size_t k = 0; k < sp.size(); ++k)
            list1_flat_h[b * im.nlist1_stride + k] = sp[k];
    }

    std::vector<int> box_levels_h(n_boxes);
    std::vector<unsigned char> ifpwexp_h(n_boxes);
    for (std::size_t b = 0; b < n_boxes; ++b) {
        box_levels_h[b] = node_mid[b].Depth();
        ifpwexp_h[b] = t.ifpwexp[b] ? 1 : 0;
    }

    // ---------- uploads ----------
    im.d_direct_work = device_upload(direct_work_h.data(), direct_work_h.size());
    im.d_list1_flat = device_upload(list1_flat_h.data(), list1_flat_h.size());
    im.d_list1_count = device_upload(list1_count_h.data(), list1_count_h.size());
    im.d_box_levels = device_upload(box_levels_h.data(), box_levels_h.size());
    im.d_ifpwexp = device_upload(ifpwexp_h.data(), ifpwexp_h.size());

    im.d_direct_rsc = device_upload(&t.direct_rsc[0], t.direct_rsc.Dim());
    im.d_direct_cen = device_upload(&t.direct_cen[0], t.direct_cen.Dim());
    im.d_direct_d2max = device_upload(&t.direct_d2max[0], t.direct_d2max.Dim());

    if (t.r_src_sorted_with_halo.Dim())
        im.d_r_src_halo = device_upload(&t.r_src_sorted_with_halo[0], t.r_src_sorted_with_halo.Dim());
    im.d_r_src_halo_offsets =
        device_upload((const long *)&t.r_src_offsets_with_halo[0], t.r_src_offsets_with_halo.Dim());
    {
        auto h = sctl_int_vec_to_std(t.src_counts_with_halo);
        im.d_src_counts_halo = device_upload(h.data(), h.size());
    }

    const bool is_stresslet = t.params.kernel == DMK_STRESSLET;
    if (is_stresslet) {
        if (t.density_sorted_with_halo.Dim())
            im.d_charge_halo = device_upload(&t.density_sorted_with_halo[0], t.density_sorted_with_halo.Dim());
        im.d_charge_halo_offsets =
            device_upload((const long *)&t.density_offsets_with_halo[0], t.density_offsets_with_halo.Dim());
        if (t.normal_sorted_with_halo.Dim())
            im.d_normal_halo = device_upload(&t.normal_sorted_with_halo[0], t.normal_sorted_with_halo.Dim());
        im.d_normal_halo_offsets =
            device_upload((const long *)&t.normal_offsets_with_halo[0], t.normal_offsets_with_halo.Dim());
    } else {
        if (t.charge_sorted_with_halo.Dim())
            im.d_charge_halo = device_upload(&t.charge_sorted_with_halo[0], t.charge_sorted_with_halo.Dim());
        im.d_charge_halo_offsets =
            device_upload((const long *)&t.charge_offsets_with_halo[0], t.charge_offsets_with_halo.Dim());
    }

    if (t.r_src_sorted_owned.Dim())
        im.d_r_src_owned = device_upload(&t.r_src_sorted_owned[0], t.r_src_sorted_owned.Dim());
    im.d_r_src_owned_offsets = device_upload((const long *)&t.r_src_offsets_owned[0], t.r_src_offsets_owned.Dim());
    {
        auto h = sctl_int_vec_to_std(t.src_counts_owned);
        im.d_src_counts_owned = device_upload(h.data(), h.size());
    }

    if (t.r_trg_sorted_owned.Dim())
        im.d_r_trg_owned = device_upload(&t.r_trg_sorted_owned[0], t.r_trg_sorted_owned.Dim());
    im.d_r_trg_owned_offsets = device_upload((const long *)&t.r_trg_offsets_owned[0], t.r_trg_offsets_owned.Dim());
    {
        auto h = sctl_int_vec_to_std(t.trg_counts_owned);
        im.d_trg_counts_owned = device_upload(h.data(), h.size());
    }

    im.pot_src_size = t.pot_src_sorted.Dim();
    im.pot_trg_size = t.pot_trg_sorted.Dim();
    im.d_pot_src = device_alloc_and_zero<Real>(im.pot_src_size);
    im.d_pot_trg = device_alloc_and_zero<Real>(im.pot_trg_size);
    im.d_pot_src_offsets = device_upload((const long *)&t.pot_src_offsets[0], t.pot_src_offsets.Dim());
    im.d_pot_trg_offsets = device_upload((const long *)&t.pot_trg_offsets[0], t.pot_trg_offsets.Dim());

    // ---------- launch (two passes: pot_src side, then pot_trg side) ----------
    cuda::DirectByBoxArgs<Real> args;
    args.n_work = im.n_work;
    args.n_levels = im.n_levels;
    args.nlist1_stride = im.nlist1_stride;
    args.thresh2 = Real{1e-30};
    args.direct_work = im.d_direct_work;
    args.list1_flat = im.d_list1_flat;
    args.list1_count = im.d_list1_count;
    args.box_levels = im.d_box_levels;
    args.ifpwexp = im.d_ifpwexp;
    args.direct_rsc = im.d_direct_rsc;
    args.direct_cen = im.d_direct_cen;
    args.direct_d2max = im.d_direct_d2max;
    args.r_src_halo_flat = im.d_r_src_halo;
    args.r_src_halo_offsets = im.d_r_src_halo_offsets;
    args.src_counts_halo = im.d_src_counts_halo;
    args.charge_halo_flat = im.d_charge_halo;
    args.charge_halo_offsets = im.d_charge_halo_offsets;
    args.normal_halo_flat = im.d_normal_halo;
    args.normal_halo_offsets = im.d_normal_halo_offsets;

    // pot_src side: target points are the trg_box's owned sources.
    args.r_target_flat = im.d_r_src_owned;
    args.r_target_offsets = im.d_r_src_owned_offsets;
    args.target_counts = im.d_src_counts_owned;
    args.pot_flat = im.d_pot_src;
    args.pot_offsets = im.d_pot_src_offsets;
    cuda::launch_direct_by_box_dispatch<Real>(t.params.kernel, DIM, t.n_digits, args, /*stream=*/0);

    // pot_trg side: target points are the trg_box's owned targets.
    args.r_target_flat = im.d_r_trg_owned;
    args.r_target_offsets = im.d_r_trg_owned_offsets;
    args.target_counts = im.d_trg_counts_owned;
    args.pot_flat = im.d_pot_trg;
    args.pot_offsets = im.d_pot_trg_offsets;
    cuda::launch_direct_by_box_dispatch<Real>(t.params.kernel, DIM, t.n_digits, args, /*stream=*/0);

    im.launched = true;
}

template <typename Real, int DIM>
void CudaDirectContext<Real, DIM>::merge_into_host() {
    auto &t = pimpl_->tree;
    auto &im = *pimpl_;

    if (!im.launched)
        return;

    DMK_CHECK_CUDA(cudaDeviceSynchronize());

    if (im.pot_src_size) {
        std::vector<Real> tmp(im.pot_src_size);
        DMK_CHECK_CUDA(cudaMemcpy(tmp.data(), im.d_pot_src, im.pot_src_size * sizeof(Real), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < im.pot_src_size; ++i)
            t.pot_src_sorted[i] += tmp[i];
    }
    if (im.pot_trg_size) {
        std::vector<Real> tmp(im.pot_trg_size);
        DMK_CHECK_CUDA(cudaMemcpy(tmp.data(), im.d_pot_trg, im.pot_trg_size * sizeof(Real), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < im.pot_trg_size; ++i)
            t.pot_trg_sorted[i] += tmp[i];
    }
}

template class CudaDirectContext<float, 2>;
template class CudaDirectContext<float, 3>;
template class CudaDirectContext<double, 2>;
template class CudaDirectContext<double, 3>;

} // namespace dmk
