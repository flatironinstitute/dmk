#include <dmk/cuda/pt/tree.hpp>

#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/pt/passes.hpp>
#include <dmk/error.hpp>
#include <dmk/logger.h>
#include <dmk/nvtx_wrapper.h>

#include <cstdlib>
#include <string>

namespace dmk::cuda::pt {
namespace {

// A pass range that closes on its own device work. Pushed enqueue-side it would be
// a few microseconds wide and every kernel would land under the join in `finalize`,
// so DMK_NVTX_SYNC=1 syncs the pass's stream before the pop. That serializes the
// pipeline -- it is a measurement mode, and the per-pass numbers it reports are
// uncontended ones, not the times those passes take in a real eval.
bool nvtx_sync() {
    static const bool on = [] {
        const char *v = std::getenv("DMK_NVTX_SYNC");
        return v && std::atoi(v) != 0;
    }();
    return on;
}

struct NvtxPass {
    NvtxPass(const char *name, cudaStream_t stream) : stream_(stream) { nvtxRangePush(name); }
    ~NvtxPass() {
        if (nvtx_sync() && stream_)
            cudaStreamSynchronize(stream_);
        nvtxRangePop();
    }
    NvtxPass(const NvtxPass &) = delete;
    NvtxPass &operator=(const NvtxPass &) = delete;

    cudaStream_t stream_;
};

} // namespace

void bind_gpu_device(int device_id) {
    int n_devices = 0;
    DMK_CHECK_CUDA(cudaGetDeviceCount(&n_devices));
    if (device_id < 0 || device_id >= n_devices)
        throw api_error(DMK_ERR_INVALID_ARGUMENT, "gpu_device_id " + std::to_string(device_id) + " is out of range: " +
                                                      std::to_string(n_devices) + " CUDA device(s) visible");

    // Initialized by the first call, so `bound` is the device this process committed to.
    static const int bound = device_id;
    if (bound != device_id)
        throw api_error(DMK_ERR_INVALID_ARGUMENT, "this process is already using CUDA device " + std::to_string(bound) +
                                                      "; DMK drives one device per process (requested " +
                                                      std::to_string(device_id) + ")");
}

template <typename Real, int DIM>
Tree<Real, DIM>::Tree(const sctl::Comm &comm, const pdmk_params &params, const sctl::Vector<Real> &r_src,
                      const sctl::Vector<Real> &charge, const sctl::Vector<Real> &normal,
                      const sctl::Vector<Real> &r_trg)
    : device_id_(params.gpu_device_id) {
    bind_gpu_device(device_id_);
    cuda_helpers::ScopedDevice device_scope(device_id_);

    // The owned tree runs the GPU host precompute only (tree build, metadata,
    // and plane-wave layout); all device state lives in state_.
    tree_ = std::make_unique<DMKPtTree<Real, DIM>>(comm, params, r_src, charge, normal, r_trg);
    tree_->init_planewave_data();

    state_ = std::make_unique<State<Real, DIM>>(to_build_inputs(*tree_));
    const long n_src = r_src.Dim() / DIM;
    const Real *charge_ptr = charge.Dim() ? &charge[0] : nullptr;
    const Real *normal_ptr = (params.kernel == DMK_STRESSLET && normal.Dim()) ? &normal[0] : nullptr;
    state_->upload_and_sort_charges(charge_ptr, normal_ptr, n_src);
    cuda_helpers::check_device_errors("tree create");
}

template <typename Real, int DIM>
Tree<Real, DIM>::~Tree() {
    cuda_helpers::ScopedDevice device_scope(device_id_);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const std::string msg = std::string("CUDA error at tree destroy: ") + cudaGetErrorString(err);
        dmk::set_last_error(msg);
        dmk::get_logger()->error("{}", msg);
    }
}

template <typename Real, int DIM>
void Tree<Real, DIM>::eval() {
    cuda_helpers::ScopedDevice device_scope(device_id_);
    // The near-field `direct` runs concurrently on direct_stream with the
    // upward -> form_outgoing -> downward -> eval_targets chain on
    // downward_stream; `finalize` joins them (direct_stream waits on the
    // downward-stream eval writes), sums the near+far potentials, descatters to
    // user order in d_pot_*_final, and syncs.
    const auto ds = state_->direct_stream.get();
    const auto ws = state_->downward_stream.get();
    {
        // Neither proxy buffer is zeroed: every box either has a writer that assigns before it
        // accumulates, or is cleared per-slab inside the pass that reads it.
        NvtxPass r("pt_upward", ws);
        pt::upward(*state_, ws);
    }
    {
        NvtxPass r("pt_form_outgoing", ws);
        pt::form_outgoing(*state_, ws);
    }
    {
        NvtxPass r("pt_downward", ws);
        pt::downward(*state_, ws);
    }
    {
        NvtxPass r("pt_eval_targets", ws);
        pt::eval_targets(*state_, ws);
    }
    {
        NvtxPass r("pt_direct", ds);
        pt::direct(*state_, ds);
    }
    {
        NvtxPass r("pt_self_correction", ds);
        pt::self_correction(*state_, ds);
    }
    {
        NvtxPass r("pt_finalize", nullptr);
        state_->finalize();
    }
    cuda_helpers::check_device_errors("tree eval");
}

template <typename Real, int DIM>
void Tree<Real, DIM>::desort_potentials(Real *pot_src, Real *pot_trg) {
    cuda_helpers::ScopedDevice device_scope(device_id_);
    // finalize wrote the descattered (user-order) result into d_pot_*_final and
    // synced; one D2H per side.
    const auto &o = state_->outputs;
    if (o.pot_src_size)
        DMK_CHECK_CUDA(
            cudaMemcpy(pot_src, o.d_pot_src_final.data(), o.pot_src_size * sizeof(Real), cudaMemcpyDeviceToHost));
    if (o.pot_trg_size)
        DMK_CHECK_CUDA(
            cudaMemcpy(pot_trg, o.d_pot_trg_final.data(), o.pot_trg_size * sizeof(Real), cudaMemcpyDeviceToHost));
}

template <typename Real, int DIM>
void Tree<Real, DIM>::update_charges(const Real *charge, const Real *normal) {
    cuda_helpers::ScopedDevice device_scope(device_id_);
    state_->upload_and_sort_charges(charge, normal, tree_->r_src_sorted_owned.Dim() / DIM);
    cuda_helpers::check_device_errors("tree update_charges");
}

template class Tree<float, 2>;
template class Tree<float, 3>;
template class Tree<double, 2>;
template class Tree<double, 3>;

} // namespace dmk::cuda::pt
