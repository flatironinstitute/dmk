#ifndef DMK_CUDA_DOWNWARD_HPP
#define DMK_CUDA_DOWNWARD_HPP

// Per-level downward-pass orchestration on the GPU. Replaces the CPU
// form_eval_expansions inner loop with three kernel launches per level on
// the shared `downward_stream`:
//
//   shift_pw_kernel        — pw_out + wpwshift → pw_in_pool
//   pw_to_proxy_kernel     — pw_in_pool → d_proxy_coeffs_downward (additive)
//   tensorprod_kernel      — d_proxy[parent] → d_proxy[child] (additive)
//
// All on the same stream → strict ordering. After all levels run, the
// shared d_proxy_coeffs_downward is final on the device, and eval_targets
// can read it directly with no H2D upload.

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
