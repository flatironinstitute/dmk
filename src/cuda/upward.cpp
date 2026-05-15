// GPU upward pass: zero d_proxy_coeffs_upward, run charge2proxy across all
// groups, then per-level (deepest first) tensorprod (child→parent with c2p).

#include <cuda_runtime.h>
#include <dmk/cuda/charge2proxy_kernels.hpp>
#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/shared_state.hpp>
#include <dmk/cuda/tensorprod_kernels.hpp>
#include <dmk/cuda/upward.hpp>
#include <dmk/nvtx_wrapper.h>
#include <dmk/tree.hpp>

#include <stdexcept>

namespace dmk {

template <typename Real, int DIM>
struct CudaUpwardContext<Real, DIM>::Impl {
    DMKPtTree<Real, DIM> &tree;
    CudaSharedDeviceState<Real, DIM> &shared;

    Impl(DMKPtTree<Real, DIM> &t, CudaSharedDeviceState<Real, DIM> &s) : tree(t), shared(s) {
        if (DIM != 3)
            throw std::runtime_error("CUDA upward: only DIM=3 supported");
        if (!s.d_charge_owned || !s.d_charge_owned_offsets)
            throw std::runtime_error("CUDA upward: owned charges not uploaded by shared state");
    }
};

template <typename Real, int DIM>
CudaUpwardContext<Real, DIM>::CudaUpwardContext(DMKPtTree<Real, DIM> &tree, CudaSharedDeviceState<Real, DIM> &shared)
    : pimpl_(std::make_unique<Impl>(tree, shared)) {}

template <typename Real, int DIM>
CudaUpwardContext<Real, DIM>::~CudaUpwardContext() = default;

template <typename Real, int DIM>
void CudaUpwardContext<Real, DIM>::run() {
    auto &s = pimpl_->shared;

    s.d_proxy_coeffs_upward.zero_async(s.downward_stream);

    if (s.n_c2p_groups) {
        cuda::Charge2ProxyArgs<Real> a;
        a.n_groups = s.n_c2p_groups;
        a.n_order = s.n_order;
        a.n_charge_dim = s.n_tables_up;
        a.center_boxes = s.d_c2p_center_boxes.data();
        a.levels = s.d_c2p_levels.data();
        a.src_box_flat_offsets = s.d_c2p_src_box_flat_offsets.data();
        a.n_src_boxes_per_group = s.d_c2p_n_src_boxes_per_group.data();
        a.src_boxes_flat = s.d_c2p_src_boxes_flat.data();
        a.centers = s.d_centers.data();
        a.inv_box_scale = s.d_inv_box_scale.data();
        a.r_src_owned = s.d_r_src_owned.data();
        a.r_src_owned_offsets = s.d_r_src_owned_offsets.data();
        a.src_counts_owned = s.d_src_counts_owned.data();
        a.charge_owned = s.d_charge_owned.data();
        a.charge_owned_offsets = s.d_charge_owned_offsets.data();
        a.proxy_flat = s.d_proxy_coeffs_upward.data();
        a.proxy_offsets = s.d_proxy_offsets_upward.data();
        cuda::launch_charge2proxy_dispatch<Real>(DIM, a, s.downward_stream);
    }

    // Per-level tensorprod (deepest level first) — accumulates children's
    // proxies into their parent's slot.
    for (int L = s.n_levels - 1; L >= 0; --L) {
        const int n_pairs = s.tp_up_count_h[L];
        if (n_pairs == 0)
            continue;
        auto range_string = "tensorprod_dispatch level: " + std::to_string(L);
        nvtxRangePush(range_string.c_str());
        const int off = s.tp_up_offset_h[L];
        cuda::TensorprodArgs<Real> ta;
        ta.n_pairs = n_pairs;
        ta.n_order = s.n_order;
        ta.n_charge_dim = s.n_tables_up;
        ta.src_boxes = s.d_tp_up_src_boxes.data() + off;
        ta.dst_boxes = s.d_tp_up_dst_boxes.data() + off;
        ta.child_octants = s.d_tp_up_octants.data() + off;
        ta.proxy_flat = s.d_proxy_coeffs_upward.data();
        ta.proxy_offsets = s.d_proxy_offsets_upward.data();
        ta.umat_flat = s.d_c2p.data();
        ta.scratch = s.d_tensorprod_scratch.data();
        ta.scratch_stride = s.tensorprod_scratch_stride_reals;
        ta.additive_atomic = true; // multiple children of the same parent → race without atomics
        cuda::launch_tensorprod_dispatch<Real>(DIM, ta, s.downward_stream);
        nvtxRangePop();
    }

    s.proxy_upward_resident_on_device = true;
}

template class CudaUpwardContext<float, 2>;
template class CudaUpwardContext<float, 3>;
template class CudaUpwardContext<double, 2>;
template class CudaUpwardContext<double, 3>;

} // namespace dmk
