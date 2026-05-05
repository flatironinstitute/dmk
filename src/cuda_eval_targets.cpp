// Orchestration for the GPU offload of proxy::eval_targets. Plain C++;
// kernel launches go through cuda::launch_eval_targets_dispatch (defined in
// src/cuda_eval_targets_kernels.cu).

#include <dmk/cuda_eval_targets.hpp>
#include <dmk/cuda_eval_targets_kernels.hpp>
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

    // Allocated/zeroed at launch.
    Real *d_pot_src_eval = nullptr;
    Real *d_pot_trg_eval = nullptr;

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
    }

    ~Impl() {
        device_free(d_centers);
        device_free(d_eval_targets_box_list);
        device_free(d_sc_per_level);
        device_free(d_pot_src_eval);
        device_free(d_pot_trg_eval);
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

    // Upload host proxy_coeffs_downward into shared buffer. Once the GPU
    // tensorprod / planewave_to_proxy stages are wired in, this upload goes
    // away — the buffer will already be populated GPU-side.
    if (shared.proxy_size)
        DMK_CHECK_CUDA(cudaMemcpyAsync(shared.d_proxy_coeffs_downward, &t.proxy_coeffs_downward[0],
                                       shared.proxy_size * sizeof(Real), cudaMemcpyHostToDevice, im.stream));

    im.d_pot_src_eval = device_alloc<Real>(shared.pot_src_size);
    im.d_pot_trg_eval = device_alloc<Real>(shared.pot_trg_size);
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
void CudaEvalTargetsContext<Real, DIM>::merge_into_host() {
    auto &t = pimpl_->tree;
    auto &shared = pimpl_->shared;
    auto &im = *pimpl_;

    if (!im.launched)
        return;

    DMK_CHECK_CUDA(cudaStreamSynchronize(im.stream));

    if (shared.pot_src_size) {
        std::vector<Real> tmp(shared.pot_src_size);
        DMK_CHECK_CUDA(
            cudaMemcpy(tmp.data(), im.d_pot_src_eval, shared.pot_src_size * sizeof(Real), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < shared.pot_src_size; ++i)
            t.pot_src_sorted[i] += tmp[i];
    }
    if (shared.pot_trg_size) {
        std::vector<Real> tmp(shared.pot_trg_size);
        DMK_CHECK_CUDA(
            cudaMemcpy(tmp.data(), im.d_pot_trg_eval, shared.pot_trg_size * sizeof(Real), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < shared.pot_trg_size; ++i)
            t.pot_trg_sorted[i] += tmp[i];
    }
}

template class CudaEvalTargetsContext<float, 2>;
template class CudaEvalTargetsContext<float, 3>;
template class CudaEvalTargetsContext<double, 2>;
template class CudaEvalTargetsContext<double, 3>;

} // namespace dmk
