#ifndef DMK_CUDA_UPWARD_HPP
#define DMK_CUDA_UPWARD_HPP

// GPU equivalent of DMKPtTree::upward_pass()'s body (charge2proxy +
// per-level child→parent tensorprod). Replaces the CPU loops that fill
// proxy_coeffs_upward; the MPI ReduceBroadcast that follows in the CPU path
// is currently incompatible with this orchestrator (CMake forbids
// DMK_GPU_OFFLOAD + DMK_HAVE_MPI for now).
//
// Lifecycle:
//   - Constructed alongside the other GPU contexts in upward_pass.
//   - run() called once per upward_pass after the work lists and host
//     buffers are ready, on shared.downward_stream so subsequent downward
//     kernels chain naturally on the same stream.

#include <memory>

namespace dmk {

template <typename Real, int DIM>
struct DMKPtTree;
template <typename Real, int DIM>
struct CudaSharedDeviceState;

template <typename Real, int DIM>
class CudaUpwardContext {
  public:
    CudaUpwardContext(DMKPtTree<Real, DIM> &tree, CudaSharedDeviceState<Real, DIM> &shared);
    ~CudaUpwardContext();
    CudaUpwardContext(const CudaUpwardContext &) = delete;
    CudaUpwardContext &operator=(const CudaUpwardContext &) = delete;

    void run();

  private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace dmk

#endif // DMK_CUDA_UPWARD_HPP
