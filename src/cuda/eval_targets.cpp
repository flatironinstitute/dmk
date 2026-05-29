#include <dmk/cuda/eval_targets.hpp>
#include <dmk/cuda/eval_targets_kernels.hpp>
#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/shared_state.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/tree.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace dmk {

namespace {

template <int DIM>
int n_charge_dim_for(dmk_ikernel kernel) {
    switch (kernel) {
    case DMK_LAPLACE:
        return 1;
    case DMK_SQRT_LAPLACE:
        return 1;
    case DMK_STOKESLET:
        return DIM;
    case DMK_STRESSLET:
        return DIM;
    default:
        throw std::runtime_error("CUDA eval_targets: unsupported kernel");
    }
}

int eval_level_for(dmk_eval_type ev) {
    switch (ev) {
    case DMK_POTENTIAL:
        return 1;
    case DMK_VELOCITY:
        return 1;
    case DMK_POTENTIAL_GRAD:
        return 2;
    default:
        throw std::runtime_error("CUDA eval_targets: unsupported eval_type");
    }
}

} // namespace

template <typename Real, int DIM>
CudaEvalTargetsContext<Real, DIM>::CudaEvalTargetsContext(DMKPtTree<Real, DIM> &tree,
                                                          CudaSharedDeviceState<Real, DIM> &shared)
    : tree_(tree), shared_(shared) {
    n_charge_dim_ = n_charge_dim_for<DIM>(tree.params.kernel);
    eval_level_src_ = eval_level_for(tree.params.eval_src);
    eval_level_trg_ = eval_level_for(tree.params.eval_trg);
    stream_ = cuda_helpers::DeviceStream::non_blocking();

    n_eval_boxes_ = tree.eval_targets_box_list.size();
    if (n_eval_boxes_)
        d_eval_targets_box_list_.upload(tree.eval_targets_box_list.data(), tree.eval_targets_box_list.size());

    n_order_ = tree.expansion_constants.n_order;

    d_pot_src_eval_.resize(shared.pot_src_size);
    d_pot_trg_eval_.resize(shared.pot_trg_size);

    n_input_dim_ = tree.kernel_input_dim;
    pot_stride_ = tree.kernel_output_dim_src;
    if (!tree.self_correction_work.empty())
        d_self_correction_work_.upload(tree.self_correction_work.data(), tree.self_correction_work.size());
}

template <typename Real, int DIM>
void CudaEvalTargetsContext<Real, DIM>::launch() {
    auto &t = tree_;
    auto &shared = shared_;

    d_pot_src_eval_.zero_async(stream_);
    d_pot_trg_eval_.zero_async(stream_);

    if (n_eval_boxes_ == 0)
        return;

    // If the GPU downward pass already populated d_proxy_coeffs_downward, we
    // skip the host→device upload entirely. Otherwise (eval_targets-only
    // path) we upload the CPU-built proxy_coeffs_downward into the shared
    // buffer.
    if (!shared.proxy_resident_on_device && shared.d_proxy_coeffs_downward) {
        DMK_CHECK_CUDA(cudaMemcpyAsync(shared.d_proxy_coeffs_downward.data(), &t.proxy_coeffs_downward[0],
                                       shared.d_proxy_coeffs_downward.size() * sizeof(Real), cudaMemcpyHostToDevice,
                                       stream_));
    } else if (shared.proxy_resident_on_device) {
        // Downward kernels wrote to d_proxy on shared.downward_stream. Make
        // eval_targets' stream wait for downward to complete before reading.
        auto evt = cuda_helpers::DeviceEvent::disable_timing();
        DMK_CHECK_CUDA(cudaEventRecord(evt, shared.downward_stream));
        DMK_CHECK_CUDA(cudaStreamWaitEvent(stream_, evt, 0));
    }

    cuda::EvalTargetsArgs<Real> args;
    args.n_eval_boxes = n_eval_boxes_;
    args.n_order = n_order_;
    args.eval_targets_box_list = d_eval_targets_box_list_.data();
    args.box_levels = shared.d_box_levels.data();
    args.sc_per_level = shared.d_inv_box_scale.data();
    args.proxy_flat = shared.d_proxy_coeffs_downward.data();
    args.proxy_offsets = shared.d_proxy_offsets_downward.data();
    args.centers = shared.d_centers.data();

    // pot_src side
    args.r_target_flat = shared.d_r_src.data();
    args.r_target_offsets = shared.d_r_src_offsets.data();
    args.target_counts = shared.d_src_counts.data();
    args.pot_flat = d_pot_src_eval_.data();
    args.pot_offsets = shared.d_pot_src_offsets.data();
    cuda::launch_eval_targets<Real, DIM>(eval_level_src_, n_charge_dim_, args, stream_);

    // pot_trg side
    args.r_target_flat = shared.d_r_trg.data();
    args.r_target_offsets = shared.d_r_trg_offsets.data();
    args.target_counts = shared.d_trg_counts.data();
    args.pot_flat = d_pot_trg_eval_.data();
    args.pot_offsets = shared.d_pot_trg_offsets.data();
    cuda::launch_eval_targets<Real, DIM>(eval_level_trg_, n_charge_dim_, args, stream_);

    // Self-correction modifies pot_src_eval in sorted layout; commutes with
    // direct's contribution, so it can run here on stream_ (before the merge).
    if (d_self_correction_work_ && shared.pot_src_size) {
        cuda::SelfCorrectionArgs<Real> sc_args;
        sc_args.direct_work = shared.d_direct_work.data();
        sc_args.correction_factors = d_self_correction_work_.data();
        sc_args.src_counts = shared.d_src_counts.data();
        sc_args.charge = shared.d_charge.data();
        sc_args.charge_offsets = shared.d_charge_offsets.data();
        sc_args.pot_src = d_pot_src_eval_.data();
        sc_args.pot_src_offsets = shared.d_pot_src_offsets.data();
        sc_args.n_direct_work = shared.n_direct_work;
        sc_args.n_input_dim = n_input_dim_;
        sc_args.pot_stride = pot_stride_;
        cuda::launch_self_correction(sc_args, stream_);
    }
}

template class CudaEvalTargetsContext<float, 2>;
template class CudaEvalTargetsContext<float, 3>;
template class CudaEvalTargetsContext<double, 2>;
template class CudaEvalTargetsContext<double, 3>;

} // namespace dmk
