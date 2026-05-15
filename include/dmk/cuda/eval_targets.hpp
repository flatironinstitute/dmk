#ifndef DMK_CUDA_EVAL_TARGETS_HPP
#define DMK_CUDA_EVAL_TARGETS_HPP

// GPU offload of proxy::eval_targets at iftensprodeval boxes.
//
// Lifecycle, driven by DMKPtTree::upward_pass / downward_pass:
//
//   1. upward_pass() constructs the context (allocates per-context device
//      buffers + creates a non-blocking stream).
//   2. downward_pass() runs the multilevel work on the CPU. The CPU's
//      `proxy::eval_targets` calls are skipped when the context is live.
//   3. End of downward_pass(): launch() uploads proxy_coeffs_downward,
//      zeroes the per-context output buffers, and queues the kernel on the
//      eval_targets stream.
//   4. finalize_gpu_only(d_extra_src, d_extra_trg): accumulates the direct
//      device buffers into d_pot_*_eval in-place on GPU, then copies the
//      combined result directly to tree.pot_*_sorted — no temporaries, no
//      CPU accumulation.

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

    /// Upload proxy_coeffs_downward, allocate output buffers, queue kernel.
    void launch();

    /// Accumulate d_extra_{src,trg} into d_pot_*_eval on GPU, then copy the
    /// combined result directly to tree.pot_*_sorted.
    void finalize_gpu_only(Real *d_extra_src, Real *d_extra_trg);

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
    bool launched_ = false;

    cuda_helpers::DeviceStream stream_;
};

} // namespace dmk

#endif // DMK_CUDA_EVAL_TARGETS_HPP
