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
//   3. End of downward_pass(): merge_into_host():
//        - cudaDeviceSynchronize()
//        - download d_pot_*_direct and accumulate into the tree's host
//          pot_*_sorted arrays (which already hold multilevel far-field
//          contributions).
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

    /// Synchronize, copy the device pot buffers back, and accumulate into
    /// tree.pot_src_sorted / pot_trg_sorted.
    void merge_into_host();

  private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace dmk

#endif // DMK_CUDA_DIRECT_HPP
