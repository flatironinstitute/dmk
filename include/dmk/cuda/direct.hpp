#ifndef DMK_CUDA_DIRECT_HPP
#define DMK_CUDA_DIRECT_HPP

// GPU offload of the direct (near-field residual) interactions.
//
// Lifecycle, driven by DMKPtTree::upward_pass / downward_pass when
// DMK_GPU_OFFLOAD is enabled at configure time:
//
//   1. upward_pass() constructs CudaSharedDeviceState (uploads inputs +
//      topology) and CudaDirectContext, then calls launch():
//        - zeroes the device pot_src_direct / pot_trg_direct (allocated at
//          construction time)
//        - dispatches the per-box residual kernel on the shared direct stream.
//   2. downward_pass() runs multilevel work on the CPU. The GPU work is in
//      flight on the direct stream throughout.
//   3. End of downward_pass(): the caller passes device_pot_src() /
//      device_pot_trg() to CudaEvalTargetsContext::finalize_gpu_only() which
//      sums both GPU results in-place and downloads the combined answer in
//      one pass.
//
// Simplifications still in force:
//   * no ContactGeometry filtering — all owned src / trg points evaluated
//     against all halo sources of every list1 source box.
//   * no PBC support (CudaSharedDeviceState ctor throws).
//   * no Yukawa support (same).

#include <dmk/cuda/helpers.hpp>

namespace dmk {

template <typename Real, int DIM>
struct DMKPtTree;

template <typename Real, int DIM>
struct CudaSharedDeviceState;

template <typename Real, int DIM>
class CudaDirectContext {
  public:
    CudaDirectContext(DMKPtTree<Real, DIM> &tree, CudaSharedDeviceState<Real, DIM> &shared);
    CudaDirectContext(const CudaDirectContext &) = delete;
    CudaDirectContext &operator=(const CudaDirectContext &) = delete;

    /// Zero output buffers and queue the residual kernel on the shared
    /// direct stream. Returns immediately.
    void launch();

    /// Device pointers to the direct potential outputs.
    Real *device_pot_src() { return d_pot_src_direct_.data(); }
    Real *device_pot_trg() { return d_pot_trg_direct_.data(); }

  private:
    DMKPtTree<Real, DIM> &tree_;
    CudaSharedDeviceState<Real, DIM> &shared_;
    cuda_helpers::DeviceBuffer<Real> d_pot_src_direct_;
    cuda_helpers::DeviceBuffer<Real> d_pot_trg_direct_;
};

} // namespace dmk

#endif // DMK_CUDA_DIRECT_HPP
