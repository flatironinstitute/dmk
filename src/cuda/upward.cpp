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
CudaUpwardContext<Real, DIM>::CudaUpwardContext(DMKPtTree<Real, DIM> &tree, CudaSharedDeviceState<Real, DIM> &shared)
    : tree_(tree), shared_(shared) {
    if (DIM != 3)
        throw std::runtime_error("CUDA upward: only DIM=3 supported");
}

template <typename Real, int DIM>
void CudaUpwardContext<Real, DIM>::run() {
    auto &s = shared_;

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
        a.r_src = s.d_r_src.data();
        a.r_src_offsets = s.d_r_src_offsets.data();
        a.src_counts = s.d_src_counts.data();
        const bool is_stresslet = tree_.params.kernel == DMK_STRESSLET;
        a.charge = is_stresslet ? s.d_charge_outer.data() : s.d_charge.data();
        a.charge_offsets = is_stresslet ? s.d_charge_outer_offsets.data() : s.d_charge_offsets.data();
        a.proxy_flat = s.d_proxy_coeffs_upward.data();
        a.proxy_offsets = s.d_proxy_offsets_upward.data();
        a.group_perm = s.d_c2p_group_perm.data();
        a.n_active_groups = s.n_c2p_active_groups;
        cuda::launch_charge2proxy<Real, DIM>(a, s.downward_stream);
    }

    // Per-level tensorprod (deepest level first) — accumulates children's
    // proxies into their parent's slot.
    for (int L = s.n_levels - 1; L >= 0; --L) {
        const int n_pairs = s.tp_up_count_h[L];
        if (n_pairs == 0)
            continue;
        auto range_string = "tensorprod up: level " + std::to_string(L);
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
        cuda::launch_tensorprod<Real, DIM>(ta, s.downward_stream);
        nvtxRangePop();
    }
}

template class CudaUpwardContext<float, 2>;
template class CudaUpwardContext<float, 3>;
template class CudaUpwardContext<double, 2>;
template class CudaUpwardContext<double, 3>;

} // namespace dmk
