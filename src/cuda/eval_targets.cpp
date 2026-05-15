// Orchestration for the GPU offload of proxy::eval_targets. Plain C++;
// kernel launches go through cuda::launch_eval_targets (defined in
// src/cuda/eval_targets_kernels.cu).

#include <dmk/cuda/eval_targets.hpp>
#include <dmk/cuda/eval_targets_kernels.hpp>
#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/shared_state.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/tree.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace dmk {

namespace {

int n_charge_dim_for(dmk_ikernel kernel) {
    switch (kernel) {
    case DMK_LAPLACE:
    case DMK_SQRT_LAPLACE:
        return 1;
    case DMK_STOKESLET:
    case DMK_STRESSLET:
        return 3;
    default:
        throw std::runtime_error("CUDA eval_targets: unsupported kernel");
    }
}

int eval_level_for(dmk_eval_type ev) {
    switch (ev) {
    case DMK_POTENTIAL:
    case DMK_VELOCITY:
        return 1;
    case DMK_POTENTIAL_GRAD:
    case DMK_VELOCITY_PRESSURE:
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
    // Validate before allocating anything. Throwing here means the tree
    // sees a null cuda_eval_targets_ctx_ and falls back to the
    // already-parallel CPU eval_targets in form_eval_expansions.
    // Supported combos must match the dispatch in eval_targets_kernels.cu.
    const int ncd = n_charge_dim_for(tree.params.kernel);
    const int el_src = eval_level_for(tree.params.eval_src);
    const int el_trg = eval_level_for(tree.params.eval_trg);
    auto supported = [](int dim, int el, int ncd) {
        if (ncd == 1)
            return (dim == 2 || dim == 3) && (el == 1 || el == 2);
        if (ncd == 3)
            return dim == 3 && el == 1;
        return false;
    };
    if (!supported(DIM, el_src, ncd) || !supported(DIM, el_trg, ncd))
        throw std::runtime_error("CUDA eval_targets: unsupported (DIM=" + std::to_string(DIM) + ", eval_src_level=" +
                                 std::to_string(el_src) + ", eval_trg_level=" + std::to_string(el_trg) +
                                 ", n_charge_dim=" + std::to_string(ncd) + ")");

    stream_ = cuda_helpers::DeviceStream::non_blocking();

    n_eval_boxes_ = (int)tree.eval_targets_box_list.size();
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

    d_pot_src_eval_.zero_async(stream_);
    d_pot_trg_eval_.zero_async(stream_);

    const int n_charge_dim = n_charge_dim_for(t.params.kernel);
    const int eval_level_src = eval_level_for(t.params.eval_src);
    const int eval_level_trg = eval_level_for(t.params.eval_trg);

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
    args.r_target_flat = shared.d_r_src_owned.data();
    args.r_target_offsets = shared.d_r_src_owned_offsets.data();
    args.target_counts = shared.d_src_counts_owned.data();
    args.pot_flat = d_pot_src_eval_.data();
    args.pot_offsets = shared.d_pot_src_offsets.data();
    cuda::launch_eval_targets<Real, DIM>(eval_level_src, n_charge_dim, args, stream_);

    // pot_trg side
    args.r_target_flat = shared.d_r_trg_owned.data();
    args.r_target_offsets = shared.d_r_trg_owned_offsets.data();
    args.target_counts = shared.d_trg_counts_owned.data();
    args.pot_flat = d_pot_trg_eval_.data();
    args.pot_offsets = shared.d_pot_trg_offsets.data();
    cuda::launch_eval_targets<Real, DIM>(eval_level_trg, n_charge_dim, args, stream_);

    launched_ = true;
}

template <typename Real, int DIM>
void CudaEvalTargetsContext<Real, DIM>::finalize_gpu_only(Real *d_extra_src, Real *d_extra_trg) {
    auto &t = tree_;
    auto &shared = shared_;

    // No eval kernels ran (n_eval_boxes == 0): direct's output is the final
    // answer. Drain direct's stream and download.
    if (!launched_) {
        DMK_CHECK_CUDA(cudaStreamSynchronize(shared.direct_stream));
        if (shared.pot_src_size && d_extra_src)
            DMK_CHECK_CUDA(cudaMemcpy(&t.pot_src_sorted[0], d_extra_src, shared.pot_src_size * sizeof(Real),
                                      cudaMemcpyDeviceToHost));
        if (shared.pot_trg_size && d_extra_trg)
            DMK_CHECK_CUDA(cudaMemcpy(&t.pot_trg_sorted[0], d_extra_trg, shared.pot_trg_size * sizeof(Real),
                                      cudaMemcpyDeviceToHost));
        return;
    }

    // Make eval_targets' stream wait for direct's kernels to finish before
    // the accumulate kernels below read direct's output buffers.
    auto direct_done = cuda_helpers::DeviceEvent::disable_timing();
    DMK_CHECK_CUDA(cudaEventRecord(direct_done, shared.direct_stream));
    DMK_CHECK_CUDA(cudaStreamWaitEvent(stream_, direct_done, 0));

    // Sum the direct device buffers into the eval buffers in-place on GPU,
    // then copy the combined result straight to host — no temps, no CPU loops.
    if (shared.pot_src_size && d_extra_src)
        cuda::launch_inplace_accumulate(d_pot_src_eval_.data(), d_extra_src, shared.pot_src_size, stream_);
    if (shared.pot_trg_size && d_extra_trg)
        cuda::launch_inplace_accumulate(d_pot_trg_eval_.data(), d_extra_trg, shared.pot_trg_size, stream_);

    if (d_self_correction_work_ && shared.pot_src_size) {
        cuda::SelfCorrectionArgs<Real> sc_args;
        sc_args.direct_work = shared.d_direct_work.data();
        sc_args.correction_factors = d_self_correction_work_.data();
        sc_args.src_counts_owned = shared.d_src_counts_owned.data();
        sc_args.src_counts_halo = shared.d_src_counts_halo.data();
        sc_args.charge_halo = shared.d_charge_halo.data();
        sc_args.charge_halo_offsets = shared.d_charge_halo_offsets.data();
        sc_args.pot_src = d_pot_src_eval_.data();
        sc_args.pot_src_offsets = shared.d_pot_src_offsets.data();
        sc_args.n_direct_work = shared.n_direct_work;
        sc_args.n_input_dim = n_input_dim_;
        sc_args.pot_stride = pot_stride_;
        cuda::launch_self_correction(sc_args, stream_);
    }

    DMK_CHECK_CUDA(cudaStreamSynchronize(stream_));

    if (shared.pot_src_size)
        DMK_CHECK_CUDA(cudaMemcpy(&t.pot_src_sorted[0], d_pot_src_eval_.data(), shared.pot_src_size * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
    if (shared.pot_trg_size)
        DMK_CHECK_CUDA(cudaMemcpy(&t.pot_trg_sorted[0], d_pot_trg_eval_.data(), shared.pot_trg_size * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
}

template class CudaEvalTargetsContext<float, 2>;
template class CudaEvalTargetsContext<float, 3>;
template class CudaEvalTargetsContext<double, 2>;
template class CudaEvalTargetsContext<double, 3>;

} // namespace dmk
