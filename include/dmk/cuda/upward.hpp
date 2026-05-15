#ifndef DMK_CUDA_UPWARD_HPP
#define DMK_CUDA_UPWARD_HPP

// GPU equivalent of DMKPtTree::upward_pass()'s body: charge2proxy + per-level
// child→parent tensorprod, populating proxy_coeffs_upward. Runs on
// shared.downward_stream so the subsequent downward kernels chain naturally.
// MPI ReduceBroadcast is not yet wired here; CMake forbids DMK_GPU_OFFLOAD +
// DMK_HAVE_MPI.

namespace dmk {

template <typename Real, int DIM>
struct DMKPtTree;
template <typename Real, int DIM>
struct CudaSharedDeviceState;

template <typename Real, int DIM>
class CudaUpwardContext {
  public:
    CudaUpwardContext(DMKPtTree<Real, DIM> &tree, CudaSharedDeviceState<Real, DIM> &shared);
    CudaUpwardContext(const CudaUpwardContext &) = delete;
    CudaUpwardContext &operator=(const CudaUpwardContext &) = delete;

    void run();

  private:
    DMKPtTree<Real, DIM> &tree_;
    CudaSharedDeviceState<Real, DIM> &shared_;
};

} // namespace dmk

#endif // DMK_CUDA_UPWARD_HPP
