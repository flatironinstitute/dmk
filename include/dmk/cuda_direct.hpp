#ifndef DMK_CUDA_DIRECT_HPP
#define DMK_CUDA_DIRECT_HPP

// GPU offload of the direct (near-field residual) interactions.
//
// Lifecycle, driven by DMKPtTree::upward_pass / downward_pass when
// DMK_GPU_OFFLOAD is enabled at configure time:
//
//   1. upward_pass() constructs CudaSharedDeviceState (uploads inputs +
//      topology) and CudaDirectContext, then calls launch():
//        - allocate + zero device pot_src_direct / pot_trg_direct
//        - dispatch the per-box residual kernel on the shared direct stream.
//   2. downward_pass() runs multilevel work on the CPU. The GPU work is in
//      flight on the direct stream throughout.
//   3. End of downward_pass(): the caller
//      then passes device_pot_src()/device_pot_trg() to
//      CudaEvalTargetsContext::finalize_gpu_only() which sums both GPU results
//      in-place and downloads the combined answer in one pass.
//
// Simplifications still in force:
//   * no ContactGeometry filtering — all owned src / trg points evaluated
//     against all halo sources of every list1 source box.
//   * no PBC support (CudaSharedDeviceState ctor throws).
//   * no Yukawa support (same).

#include <memory>

namespace dmk {

template <typename Real, int DIM>
struct DMKPtTree;

template <typename Real, int DIM>
struct CudaSharedDeviceState;

template <typename Real, int DIM>
class CudaDirectContext {
  public:
    CudaDirectContext(DMKPtTree<Real, DIM> &tree, CudaSharedDeviceState<Real, DIM> &shared);
    ~CudaDirectContext();
    CudaDirectContext(const CudaDirectContext &) = delete;
    CudaDirectContext &operator=(const CudaDirectContext &) = delete;

    /// Allocate output buffers and queue the residual kernel on the shared
    /// direct stream. Returns immediately.
    void launch();

    /// Device pointers to the direct potential outputs
    Real *device_pot_src() const;
    Real *device_pot_trg() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace dmk

#endif // DMK_CUDA_DIRECT_HPP
