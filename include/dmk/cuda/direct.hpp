#ifndef DMK_CUDA_DIRECT_HPP
#define DMK_CUDA_DIRECT_HPP

// GPU offload of the direct (near-field residual) interactions.

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
