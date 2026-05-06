// Per-level downward GPU orchestration. Issues shift_pw, pw_to_proxy, and
// tensorprod kernels on the shared state's downward stream.

#include <dmk/cuda_downward.hpp>
#include <dmk/cuda_pw_to_proxy_kernels.hpp>
#include <dmk/cuda_shared_state.hpp>
#include <dmk/cuda_shift_pw_kernels.hpp>
#include <dmk/cuda_tensorprod_kernels.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/tree.hpp>

#include <cuda_runtime.h>

#include <stdexcept>

namespace dmk {

template <typename Real, int DIM>
CudaDownwardContext<Real, DIM>::CudaDownwardContext(DMKPtTree<Real, DIM> &tree,
                                                    CudaSharedDeviceState<Real, DIM> &shared)
    : tree_(tree), shared_(shared) {
    // Bootstrap scope: only 3D downward kernels exist. Throwing here lets
    // tree.cpp's upward_pass catch reset cuda_downward_ctx_ and fall back to
    // the CPU level loop without aborting the run.
    if (DIM != 3)
        throw std::runtime_error("CUDA downward: only DIM=3 supported");
}

template <typename Real, int DIM>
void CudaDownwardContext<Real, DIM>::upload_pw_out() {
    shared_.upload_pw_out(tree_);
}

template <typename Real, int DIM>
void CudaDownwardContext<Real, DIM>::run_level(int level) {
    auto &s = shared_;
    const int n_pw_eval = s.pw_eval_box_count_h[level];

    if (n_pw_eval > 0) {
        const int box_offset = s.pw_eval_box_offset_h[level];

        // 1. shift_pw: pw_out → pw_in_pool.
        cuda::ShiftPwArgs<Real> sa;
        sa.n_boxes_at_level = n_pw_eval;
        sa.n_neighbors = s.n_neighbors;
        sa.n_charge_dim = s.n_charge_dim;
        sa.n_pw_modes = s.n_pw_modes;
        sa.pw_in_stride = s.pw_in_stride_reals;
        sa.box_ids = s.d_pw_eval_box_flat + box_offset;
        sa.neighbors = s.d_neighbors;
        sa.pw_out_offsets = s.d_pw_out_offsets;
        sa.is_global_leaf = s.d_is_global_leaf;
        sa.pw_out_flat = s.d_pw_out;
        sa.wpwshift = s.d_wpwshift_flat + (long)level * s.wpwshift_per_level_reals;
        sa.pw_in_pool = s.d_pw_in_pool;
        cuda::launch_shift_pw_dispatch<Real>(DIM, sa, s.downward_stream);

        // 2. pw_to_proxy: pw_in_pool → d_proxy_coeffs_downward (additive).
        cuda::PwToProxyArgs<Real> pa;
        pa.n_boxes_at_level = n_pw_eval;
        pa.n_order = s.n_order;
        pa.n_pw = s.n_pw;
        pa.n_pw2 = s.n_pw2;
        pa.n_charge_dim = s.n_charge_dim;
        pa.pw_in_stride = s.pw_in_stride_reals;
        pa.box_ids = s.d_pw_eval_box_flat + box_offset;
        pa.pw_in_pool = s.d_pw_in_pool;
        pa.pw2poly = s.d_pw2poly_flat + (long)level * s.pw2poly_per_level_reals;
        pa.proxy_flat = s.d_proxy_coeffs_downward;
        pa.proxy_offsets = s.d_proxy_offsets_downward;
        cuda::launch_pw_to_proxy_dispatch<Real>(DIM, pa, s.downward_stream);
    }

    // 3. tensorprod: parent's proxy → child's proxy (additive).
    const int n_tp = s.tp_count_h[level];
    if (n_tp > 0) {
        const int tp_offset = s.tp_offset_h[level];
        cuda::TensorprodArgs<Real> ta;
        ta.n_pairs = n_tp;
        ta.n_order = s.n_order;
        ta.n_charge_dim = s.n_charge_dim;
        ta.parents = s.d_tp_parents + tp_offset;
        ta.children = s.d_tp_children + tp_offset;
        ta.child_octants = s.d_tp_octants + tp_offset;
        ta.proxy_flat = s.d_proxy_coeffs_downward;
        ta.proxy_offsets = s.d_proxy_offsets_downward;
        ta.p2c_flat = s.d_p2c;
        ta.scratch = s.d_tensorprod_scratch;
        ta.scratch_stride = s.tensorprod_scratch_stride_reals;
        cuda::launch_tensorprod_dispatch<Real>(DIM, ta, s.downward_stream);
    }
}

template <typename Real, int DIM>
void CudaDownwardContext<Real, DIM>::mark_proxy_resident() {
    shared_.proxy_resident_on_device = true;
}

template class CudaDownwardContext<float, 2>;
template class CudaDownwardContext<float, 3>;
template class CudaDownwardContext<double, 2>;
template class CudaDownwardContext<double, 3>;

} // namespace dmk
