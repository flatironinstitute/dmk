// Orchestration for the GPU offload of proxy::eval_targets. Plain C++;
// kernel launches go through cuda::launch_eval_targets_dispatch (defined in
// src/cuda_eval_targets_kernels.cu).

#include <dmk/cuda_eval_targets.hpp>
#include <dmk/cuda_eval_targets_kernels.hpp>
#include <dmk/cuda_helpers.hpp>
#include <dmk/cuda_shared_state.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/tree.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace dmk {

using cuda_helpers::device_alloc;
using cuda_helpers::device_free;
using cuda_helpers::device_upload;

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
struct CudaEvalTargetsContext<Real, DIM>::Impl {
    DMKPtTree<Real, DIM> &tree;
    CudaSharedDeviceState<Real, DIM> &shared;

    // Constructed once at upward_pass start; reused by launch().
    // Note: d_proxy_coeffs_downward + offsets now live in shared state (so
    // future GPU planewave_to_proxy / tensorprod can write to them directly
    // and eval_targets just reads).
    Real *d_centers = nullptr;
    int *d_eval_targets_box_list = nullptr;
    Real *d_sc_per_level = nullptr;

    // Output buffers; allocated once at construction and zeroed at the start
    // of each launch() so the context can be reused across evals.
    Real *d_pot_src_eval = nullptr;
    Real *d_pot_trg_eval = nullptr;

    Real *d_self_correction_work = nullptr;
    int n_input_dim = 0;
    int pot_stride = 0;

    cudaStream_t stream = nullptr;
    int n_eval_boxes = 0;
    int n_order = 0;
    bool launched = false;

    Impl(DMKPtTree<Real, DIM> &t, CudaSharedDeviceState<Real, DIM> &s) : tree(t), shared(s) {
        // Validate before allocating anything. Throwing here means the tree
        // sees a null cuda_eval_targets_ctx_ and falls back to the
        // already-parallel CPU eval_targets in form_eval_expansions.
        // Supported combos must match the dispatch in cuda_eval_targets_kernels.cu.
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
            throw std::runtime_error("CUDA eval_targets: unsupported (DIM=" + std::to_string(DIM) +
                                     ", eval_src_level=" + std::to_string(el_src) + ", eval_trg_level=" +
                                     std::to_string(el_trg) + ", n_charge_dim=" + std::to_string(ncd) + ")");

        DMK_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        const int n_lvl = tree.boxsize.Dim();
        std::vector<Real> sc_h(n_lvl);
        for (int l = 0; l < n_lvl; ++l)
            sc_h[l] = Real{2} / tree.boxsize[l];
        d_sc_per_level = device_upload(sc_h.data(), sc_h.size());

        if (tree.centers.Dim())
            d_centers = device_upload(&tree.centers[0], tree.centers.Dim());

        n_eval_boxes = (int)tree.eval_targets_box_list.size();
        if (n_eval_boxes)
            d_eval_targets_box_list =
                device_upload(tree.eval_targets_box_list.data(), tree.eval_targets_box_list.size());

        n_order = tree.expansion_constants.n_order;

        d_pot_src_eval = device_alloc<Real>(shared.pot_src_size);
        d_pot_trg_eval = device_alloc<Real>(shared.pot_trg_size);

        n_input_dim = tree.kernel_input_dim;
        pot_stride = tree.kernel_output_dim_src;
        if (!tree.self_correction_work.empty())
            d_self_correction_work = device_upload(tree.self_correction_work.data(), tree.self_correction_work.size());
    }

    ~Impl() {
        device_free(d_centers);
        device_free(d_eval_targets_box_list);
        device_free(d_sc_per_level);
        device_free(d_pot_src_eval);
        device_free(d_pot_trg_eval);
        device_free(d_self_correction_work);
        if (stream)
            cudaStreamDestroy(stream);
    }
};

template <typename Real, int DIM>
CudaEvalTargetsContext<Real, DIM>::CudaEvalTargetsContext(DMKPtTree<Real, DIM> &tree,
                                                          CudaSharedDeviceState<Real, DIM> &shared)
    : pimpl_(std::make_unique<Impl>(tree, shared)) {}

template <typename Real, int DIM>
CudaEvalTargetsContext<Real, DIM>::~CudaEvalTargetsContext() = default;

template <typename Real, int DIM>
void CudaEvalTargetsContext<Real, DIM>::launch() {
    auto &t = pimpl_->tree;
    auto &shared = pimpl_->shared;
    auto &im = *pimpl_;

    if (im.n_eval_boxes == 0)
        return;

    // If the GPU downward pass already populated d_proxy_coeffs_downward, we
    // skip the host→device upload entirely. Otherwise (eval_targets-only
    // path) we upload the CPU-built proxy_coeffs_downward into the shared
    // buffer.
    if (!shared.proxy_resident_on_device && shared.proxy_size) {
        DMK_CHECK_CUDA(cudaMemcpyAsync(shared.d_proxy_coeffs_downward, &t.proxy_coeffs_downward[0],
                                       shared.proxy_size * sizeof(Real), cudaMemcpyHostToDevice, im.stream));
    } else if (shared.proxy_resident_on_device) {
        // Downward kernels wrote to d_proxy on shared.downward_stream. Make
        // eval_targets' stream wait for downward to complete before reading.
        cudaEvent_t evt;
        DMK_CHECK_CUDA(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
        DMK_CHECK_CUDA(cudaEventRecord(evt, shared.downward_stream));
        DMK_CHECK_CUDA(cudaStreamWaitEvent(im.stream, evt, 0));
        cudaEventDestroy(evt);
    }

    if (im.d_pot_src_eval)
        DMK_CHECK_CUDA(cudaMemsetAsync(im.d_pot_src_eval, 0, shared.pot_src_size * sizeof(Real), im.stream));
    if (im.d_pot_trg_eval)
        DMK_CHECK_CUDA(cudaMemsetAsync(im.d_pot_trg_eval, 0, shared.pot_trg_size * sizeof(Real), im.stream));

    const int n_charge_dim = n_charge_dim_for(t.params.kernel);
    const int eval_level_src = eval_level_for(t.params.eval_src);
    const int eval_level_trg = eval_level_for(t.params.eval_trg);

    cuda::EvalTargetsArgs<Real> args;
    args.n_eval_boxes = im.n_eval_boxes;
    args.n_order = im.n_order;
    args.eval_targets_box_list = im.d_eval_targets_box_list;
    args.box_levels = shared.d_box_levels;
    args.sc_per_level = im.d_sc_per_level;
    args.proxy_flat = shared.d_proxy_coeffs_downward;
    args.proxy_offsets = shared.d_proxy_offsets_downward;
    args.centers = im.d_centers;

    // pot_src side
    args.r_target_flat = shared.d_r_src_owned;
    args.r_target_offsets = shared.d_r_src_owned_offsets;
    args.target_counts = shared.d_src_counts_owned;
    args.pot_flat = im.d_pot_src_eval;
    args.pot_offsets = shared.d_pot_src_offsets;
    cuda::launch_eval_targets_dispatch<Real>(DIM, eval_level_src, n_charge_dim, args, im.stream);

    // pot_trg side
    args.r_target_flat = shared.d_r_trg_owned;
    args.r_target_offsets = shared.d_r_trg_owned_offsets;
    args.target_counts = shared.d_trg_counts_owned;
    args.pot_flat = im.d_pot_trg_eval;
    args.pot_offsets = shared.d_pot_trg_offsets;
    cuda::launch_eval_targets_dispatch<Real>(DIM, eval_level_trg, n_charge_dim, args, im.stream);

    im.launched = true;
}

template <typename Real, int DIM>
void CudaEvalTargetsContext<Real, DIM>::finalize_gpu_only(Real *d_extra_src, Real *d_extra_trg) {
    auto &t = pimpl_->tree;
    auto &shared = pimpl_->shared;
    auto &im = *pimpl_;

    // Caller already called CudaDirectContext::sync() (cudaDeviceSynchronize),
    // so both direct and eval kernels are complete. If eval was not launched
    // (n_eval_boxes == 0), skip straight to downloading only the direct result.
    if (!im.launched) {
        if (shared.pot_src_size && d_extra_src)
            DMK_CHECK_CUDA(cudaMemcpy(&t.pot_src_sorted[0], d_extra_src, shared.pot_src_size * sizeof(Real),
                                      cudaMemcpyDeviceToHost));
        if (shared.pot_trg_size && d_extra_trg)
            DMK_CHECK_CUDA(cudaMemcpy(&t.pot_trg_sorted[0], d_extra_trg, shared.pot_trg_size * sizeof(Real),
                                      cudaMemcpyDeviceToHost));
        return;
    }

    // Sum the direct device buffers into the eval buffers in-place on GPU,
    // then copy the combined result straight to host — no temps, no CPU loops.
    if (shared.pot_src_size && d_extra_src)
        cuda::launch_inplace_accumulate(im.d_pot_src_eval, d_extra_src, shared.pot_src_size, im.stream);
    if (shared.pot_trg_size && d_extra_trg)
        cuda::launch_inplace_accumulate(im.d_pot_trg_eval, d_extra_trg, shared.pot_trg_size, im.stream);

    if (im.d_self_correction_work && shared.pot_src_size) {
        cuda::SelfCorrectionArgs<Real> sc_args;
        sc_args.direct_work = shared.d_direct_work;
        sc_args.correction_factors = im.d_self_correction_work;
        sc_args.src_counts_owned = shared.d_src_counts_owned;
        sc_args.src_counts_halo = shared.d_src_counts_halo;
        sc_args.charge_halo = shared.d_charge_halo;
        sc_args.charge_halo_offsets = shared.d_charge_halo_offsets;
        sc_args.pot_src = im.d_pot_src_eval;
        sc_args.pot_src_offsets = shared.d_pot_src_offsets;
        sc_args.n_direct_work = shared.n_direct_work;
        sc_args.n_input_dim = im.n_input_dim;
        sc_args.pot_stride = im.pot_stride;
        cuda::launch_self_correction(sc_args, im.stream);
    }

    DMK_CHECK_CUDA(cudaStreamSynchronize(im.stream));

    if (shared.pot_src_size)
        DMK_CHECK_CUDA(cudaMemcpy(&t.pot_src_sorted[0], im.d_pot_src_eval, shared.pot_src_size * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
    if (shared.pot_trg_size)
        DMK_CHECK_CUDA(cudaMemcpy(&t.pot_trg_sorted[0], im.d_pot_trg_eval, shared.pot_trg_size * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
}

template class CudaEvalTargetsContext<float, 2>;
template class CudaEvalTargetsContext<float, 3>;
template class CudaEvalTargetsContext<double, 2>;
template class CudaEvalTargetsContext<double, 3>;

} // namespace dmk
