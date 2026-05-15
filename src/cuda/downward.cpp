// Per-level downward GPU orchestration. Issues shift_pw, pw_to_proxy, and
// tensorprod kernels on the shared state's downward stream.

#include <cuda_runtime.h>
#include <dmk/cuda/downward.hpp>
#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/pw_to_proxy_kernels.hpp>
#include <dmk/cuda/shared_state.hpp>
#include <dmk/cuda/shift_pw_kernels.hpp>
#include <dmk/cuda/tensorprod_kernels.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/nvtx_wrapper.h>
#include <dmk/tree.hpp>
#include <stdexcept>
#include <vector>

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
void CudaDownwardContext<Real, DIM>::run() {
    nvtxRangePush("cuda_downward: multilevel local + levelwise tensorprod");

    auto &s = shared_;

    const int n_levels = s.n_levels;

    std::vector<cuda::ShiftPwArgs<Real>> shift_args_h;
    std::vector<cuda::PwToProxyArgs<Real>> pw_to_proxy_args_h;

    // pw_in_pool layout (per-level disjoint slabs) was precomputed in shared
    // state — just index into it here.
    for (int level = 0; level < n_levels; ++level) {
        const int n_pw_eval = s.pw_eval_box_count_h[level];
        if (n_pw_eval <= 0)
            continue;

        const int box_offset = s.pw_eval_box_offset_h[level];
        Real *level_pw_in_pool = s.d_pw_in_pool.data() + s.pw_in_pool_base_h[level] * s.pw_in_stride_reals;

        cuda::ShiftPwArgs<Real> sa;
        sa.n_boxes_at_level = n_pw_eval;
        sa.n_neighbors = s.n_neighbors;
        sa.n_charge_dim = s.n_charge_dim;
        sa.n_pw_modes = s.n_pw_modes;
        sa.pw_in_stride = s.pw_in_stride_reals;
        sa.box_ids = s.d_pw_eval_box_flat.data() + box_offset;
        sa.neighbors = s.d_neighbors.data();
        sa.pw_out_offsets = s.d_pw_out_offsets.data();
        sa.is_global_leaf = s.d_is_global_leaf.data();
        sa.pw_out_flat = s.d_pw_out.data();
        sa.wpwshift = s.d_wpwshift_flat.data() + (long)level * s.wpwshift_per_level_reals;
        sa.pw_in_pool = level_pw_in_pool;
        shift_args_h.push_back(sa);

        cuda::PwToProxyArgs<Real> pa;
        pa.n_boxes_at_level = n_pw_eval;
        pa.n_order = s.n_order;
        pa.n_pw = s.n_pw;
        pa.n_pw2 = s.n_pw2;
        pa.n_charge_dim = s.n_charge_dim;
        pa.pw_in_stride = s.pw_in_stride_reals;
        pa.box_ids = s.d_pw_eval_box_flat.data() + box_offset;
        pa.pw_in_pool = level_pw_in_pool;
        pa.pw2poly = s.d_pw2poly_flat.data() + (long)level * s.pw2poly_per_level_reals;
        pa.proxy_flat = s.d_proxy_coeffs_downward.data();
        pa.proxy_offsets = s.d_proxy_offsets_downward.data();
        pw_to_proxy_args_h.push_back(pa);
    }

    if (!shift_args_h.empty()) {
        nvtxRangePush("shift_pw: multilevel");
        cuda::launch_shift_pw_multilevel<Real, DIM>(shift_args_h, s.d_shift_pw_args.data(), s.downward_stream);
        nvtxRangePop();
    }

    if (!pw_to_proxy_args_h.empty()) {
        nvtxRangePush("pw_to_proxy: multilevel");
        cuda::launch_pw_to_proxy_multilevel<Real, DIM>(pw_to_proxy_args_h, s.d_pw_to_proxy_args.data(),
                                                       s.downward_stream);
        nvtxRangePop();
    }

    // 3. tensorprod level by level
    for (int level = 0; level < n_levels; ++level) {
        const int n_tp = s.tp_count_h[level];

        if (n_tp <= 0)
            continue;

        auto range_string = "tensorprod: level " + std::to_string(level);

        nvtxRangePush(range_string.c_str());

        const int tp_offset = s.tp_offset_h[level];

        cuda::TensorprodArgs<Real> ta;
        ta.n_pairs = n_tp;
        ta.n_order = s.n_order;
        ta.n_charge_dim = s.n_charge_dim;
        ta.src_boxes = s.d_tp_parents.data() + tp_offset;
        ta.dst_boxes = s.d_tp_children.data() + tp_offset;
        ta.child_octants = s.d_tp_octants.data() + tp_offset;
        ta.proxy_flat = s.d_proxy_coeffs_downward.data();
        ta.proxy_offsets = s.d_proxy_offsets_downward.data();
        ta.umat_flat = s.d_p2c.data();
        ta.scratch = s.d_tensorprod_scratch.data();
        ta.scratch_stride = s.tensorprod_scratch_stride_reals;

        cuda::launch_tensorprod<Real, DIM>(ta, s.downward_stream);

        nvtxRangePop();
    }

    s.proxy_resident_on_device = true;
    nvtxRangePop();
}

template class CudaDownwardContext<float, 2>;
template class CudaDownwardContext<float, 3>;
template class CudaDownwardContext<double, 2>;
template class CudaDownwardContext<double, 3>;

} // namespace dmk
