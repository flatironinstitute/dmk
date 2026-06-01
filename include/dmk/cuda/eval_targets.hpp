#ifndef DMK_CUDA_EVAL_TARGETS_HPP
#define DMK_CUDA_EVAL_TARGETS_HPP

// GPU offload of proxy::eval_targets at iftensprodeval boxes. launch()
// uploads proxy_coeffs_downward (or skips when the GPU downward pass already
// populated it), queues the eval kernel, and applies self-correction in
// place — all on stream(). Sorted-layout outputs live in d_pot_*_eval; the
// merge with direct's output (and the descatter to user order) happens in
// CudaSharedDeviceState::finalize_pot().

#include <dmk/cuda/helpers.hpp>

namespace dmk {

template <typename Real, int DIM>
struct DMKPtTree;

template <typename Real, int DIM>
struct CudaSharedDeviceState;

template <typename Real, int DIM>
class CudaEvalTargetsContext {
  public:
    CudaEvalTargetsContext(DMKPtTree<Real, DIM> &tree, CudaSharedDeviceState<Real, DIM> &shared);
    CudaEvalTargetsContext(const CudaEvalTargetsContext &) = delete;
    CudaEvalTargetsContext &operator=(const CudaEvalTargetsContext &) = delete;

    /// Zero d_pot_*_eval, queue the eval kernel (if there are eval boxes),
    /// then apply self-correction. All async on stream().
    void launch();

    /// Sorted-layout output buffers; consumed by shared.finalize_pot().
    Real *device_pot_src() { return d_pot_src_eval_.data(); }
    Real *device_pot_trg() { return d_pot_trg_eval_.data(); }
    cudaStream_t stream() const { return stream_; }

  private:
    DMKPtTree<Real, DIM> &tree_;
    CudaSharedDeviceState<Real, DIM> &shared_;

    cuda_helpers::DeviceBuffer<int> d_eval_targets_box_list_;
    cuda_helpers::DeviceBuffer<Real> d_pot_src_eval_;
    cuda_helpers::DeviceBuffer<Real> d_pot_trg_eval_;
    cuda_helpers::DeviceBuffer<Real> d_self_correction_work_;

    int n_input_dim_ = 0;
    int pot_stride_ = 0;
    int n_eval_boxes_ = 0;
    int n_order_ = 0;
    int n_charge_dim_ = 0; // 1 for laplace/sqrt_laplace, 3 for stokeslet/stresslet
    int eval_level_src_ = 0;
    int eval_level_trg_ = 0;

    cuda_helpers::DeviceStream stream_;
};

} // namespace dmk

#endif // DMK_CUDA_EVAL_TARGETS_HPP
