// Construction / destruction of CudaSharedDeviceState — uploads all
// read-only inputs + topology that GPU offload operations need.
//
// Plain C++; no <<<>>> launch syntax. Compiled into DMKOBJS_CUDA.

#include <dmk/cuda_shared_state.hpp>
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
CudaSharedDeviceState<Real, DIM>::CudaSharedDeviceState(DMKPtTree<Real, DIM> &tree) {
    if (tree.params.use_periodic)
        throw std::runtime_error("CUDA offload: periodic boundary conditions are not yet supported");
    if (tree.params.kernel == DMK_YUKAWA)
        throw std::runtime_error("CUDA offload: Yukawa kernel is not yet supported on the GPU path");

    const auto &node_mid = tree.GetNodeMID();
    n_boxes = (int)tree.n_boxes();
    n_levels = tree.boxsize.Dim();
    nlist1_stride = (1 << (2 * DIM)) - (1 << DIM) + 1;

    // ---------- host-side topology buffers ----------
    std::vector<int> direct_work_h(tree.direct_work.begin(), tree.direct_work.end());
    n_direct_work = (int)direct_work_h.size();

    std::vector<int> list1_flat_h((std::size_t)n_boxes * nlist1_stride, -1);
    std::vector<int> list1_count_h(n_boxes, 0);
    for (int b = 0; b < n_boxes; ++b) {
        const auto sp = tree.list1(b);
        list1_count_h[b] = (int)sp.size();
        for (std::size_t k = 0; k < sp.size(); ++k)
            list1_flat_h[(std::size_t)b * nlist1_stride + k] = sp[k];
    }

    std::vector<int> box_levels_h(n_boxes);
    std::vector<unsigned char> ifpwexp_h(n_boxes);
    for (int b = 0; b < n_boxes; ++b) {
        box_levels_h[b] = node_mid[b].Depth();
        ifpwexp_h[b] = tree.ifpwexp[b] ? 1 : 0;
    }

    // ---------- uploads ----------
    d_direct_work = device_upload(direct_work_h.data(), direct_work_h.size());
    d_list1_flat = device_upload(list1_flat_h.data(), list1_flat_h.size());
    d_list1_count = device_upload(list1_count_h.data(), list1_count_h.size());
    d_box_levels = device_upload(box_levels_h.data(), box_levels_h.size());
    d_ifpwexp = device_upload(ifpwexp_h.data(), ifpwexp_h.size());

    d_direct_rsc = device_upload(&tree.direct_rsc[0], tree.direct_rsc.Dim());
    d_direct_cen = device_upload(&tree.direct_cen[0], tree.direct_cen.Dim());
    d_direct_d2max = device_upload(&tree.direct_d2max[0], tree.direct_d2max.Dim());

    if (tree.r_src_sorted_with_halo.Dim())
        d_r_src_halo = device_upload(&tree.r_src_sorted_with_halo[0], tree.r_src_sorted_with_halo.Dim());
    d_r_src_halo_offsets =
        device_upload((const long *)&tree.r_src_offsets_with_halo[0], tree.r_src_offsets_with_halo.Dim());
    {
        auto h = sctl_int_vec_to_std(tree.src_counts_with_halo);
        d_src_counts_halo = device_upload(h.data(), h.size());
    }

    const bool is_stresslet = tree.params.kernel == DMK_STRESSLET;
    if (is_stresslet) {
        if (tree.density_sorted_with_halo.Dim())
            d_charge_halo = device_upload(&tree.density_sorted_with_halo[0], tree.density_sorted_with_halo.Dim());
        d_charge_halo_offsets =
            device_upload((const long *)&tree.density_offsets_with_halo[0], tree.density_offsets_with_halo.Dim());
        if (tree.normal_sorted_with_halo.Dim())
            d_normal_halo = device_upload(&tree.normal_sorted_with_halo[0], tree.normal_sorted_with_halo.Dim());
        d_normal_halo_offsets =
            device_upload((const long *)&tree.normal_offsets_with_halo[0], tree.normal_offsets_with_halo.Dim());
    } else {
        if (tree.charge_sorted_with_halo.Dim())
            d_charge_halo = device_upload(&tree.charge_sorted_with_halo[0], tree.charge_sorted_with_halo.Dim());
        d_charge_halo_offsets =
            device_upload((const long *)&tree.charge_offsets_with_halo[0], tree.charge_offsets_with_halo.Dim());
    }

    if (tree.r_src_sorted_owned.Dim())
        d_r_src_owned = device_upload(&tree.r_src_sorted_owned[0], tree.r_src_sorted_owned.Dim());
    d_r_src_owned_offsets = device_upload((const long *)&tree.r_src_offsets_owned[0], tree.r_src_offsets_owned.Dim());
    {
        auto h = sctl_int_vec_to_std(tree.src_counts_owned);
        d_src_counts_owned = device_upload(h.data(), h.size());
    }

    if (tree.r_trg_sorted_owned.Dim())
        d_r_trg_owned = device_upload(&tree.r_trg_sorted_owned[0], tree.r_trg_sorted_owned.Dim());
    d_r_trg_owned_offsets = device_upload((const long *)&tree.r_trg_offsets_owned[0], tree.r_trg_offsets_owned.Dim());
    {
        auto h = sctl_int_vec_to_std(tree.trg_counts_owned);
        d_trg_counts_owned = device_upload(h.data(), h.size());
    }

    pot_src_size = tree.pot_src_sorted.Dim();
    pot_trg_size = tree.pot_trg_sorted.Dim();
    d_pot_src_offsets = device_upload((const long *)&tree.pot_src_offsets[0], tree.pot_src_offsets.Dim());
    d_pot_trg_offsets = device_upload((const long *)&tree.pot_trg_offsets[0], tree.pot_trg_offsets.Dim());

    // Downward proxy buffer: allocated zero-initialized; populated later (by
    // host upload from eval_targets, or by GPU planewave_to_proxy / tensorprod
    // kernels once those are in place).
    proxy_size = tree.proxy_coeffs_downward.Dim();
    if (proxy_size) {
        d_proxy_coeffs_downward = device_alloc<Real>(proxy_size);
        DMK_CHECK_CUDA(cudaMemset(d_proxy_coeffs_downward, 0, proxy_size * sizeof(Real)));
    }
    d_proxy_offsets_downward =
        device_upload((const long *)&tree.proxy_coeffs_offsets_downward[0], tree.proxy_coeffs_offsets_downward.Dim());
}

template <typename Real, int DIM>
CudaSharedDeviceState<Real, DIM>::~CudaSharedDeviceState() {
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
    device_free(d_pot_src_offsets);
    device_free(d_pot_trg_offsets);
    device_free(d_proxy_coeffs_downward);
    device_free(d_proxy_offsets_downward);
}

template struct CudaSharedDeviceState<float, 2>;
template struct CudaSharedDeviceState<float, 3>;
template struct CudaSharedDeviceState<double, 2>;
template struct CudaSharedDeviceState<double, 3>;

} // namespace dmk
