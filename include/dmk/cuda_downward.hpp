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
    ~CudaDownwardContext() = default;
    CudaDownwardContext(const CudaDownwardContext &) = delete;
    CudaDownwardContext &operator=(const CudaDownwardContext &) = delete;

    /// Issue the three GPU kernels for one level on the downward stream.
    /// Caller is responsible for invoking levels in order 0..n_levels-1.
    void run_level(int level);

    /// Mark proxy as resident on the device so eval_targets skips its upload.
    /// Called after all levels have been issued.
    void mark_proxy_resident();

  private:
    DMKPtTree<Real, DIM> &tree_;
    CudaSharedDeviceState<Real, DIM> &shared_;
};

} // namespace dmk

#endif // DMK_CUDA_DOWNWARD_HPP
