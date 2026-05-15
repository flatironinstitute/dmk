#ifndef DMK_CUDA_DOWNWARD_HPP
#define DMK_CUDA_DOWNWARD_HPP

// GPU equivalent of the form_eval_expansions inner loop: shift_pw +
// pw_to_proxy + tensorprod, all on shared.downward_stream so they chain in
// order. After run(), d_proxy_coeffs_downward is final on the device.

namespace dmk {

template <typename Real, int DIM>
struct DMKPtTree;

template <typename Real, int DIM>
struct CudaSharedDeviceState;

template <typename Real, int DIM>
class CudaDownwardContext {
  public:
    CudaDownwardContext(DMKPtTree<Real, DIM> &tree, CudaSharedDeviceState<Real, DIM> &shared);
    CudaDownwardContext(const CudaDownwardContext &) = delete;
    CudaDownwardContext &operator=(const CudaDownwardContext &) = delete;

    /// Issue shift_pw + pw_to_proxy + per-level tensorprod on the downward
    /// stream, then mark proxy as resident on the device so eval_targets
    /// skips its upload.
    void run();

  private:
    DMKPtTree<Real, DIM> &tree_;
    CudaSharedDeviceState<Real, DIM> &shared_;
};

} // namespace dmk

#endif // DMK_CUDA_DOWNWARD_HPP
