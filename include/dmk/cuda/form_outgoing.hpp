#ifndef DMK_CUDA_FORM_OUTGOING_HPP
#define DMK_CUDA_FORM_OUTGOING_HPP

// GPU equivalent of DMKPtTree::form_outgoing_expansions(). Computes pw_out
// for every box that does PW work, plus the windowed-kernel root
// contribution into proxy_coeffs_downward[0].

namespace dmk {

template <typename Real, int DIM>
struct DMKPtTree;
template <typename Real, int DIM>
struct CudaSharedDeviceState;

template <typename Real, int DIM>
class CudaFormOutgoingContext {
  public:
    CudaFormOutgoingContext(DMKPtTree<Real, DIM> &tree, CudaSharedDeviceState<Real, DIM> &shared);
    CudaFormOutgoingContext(const CudaFormOutgoingContext &) = delete;
    CudaFormOutgoingContext &operator=(const CudaFormOutgoingContext &) = delete;

    void run();

  private:
    DMKPtTree<Real, DIM> &tree_;
    CudaSharedDeviceState<Real, DIM> &shared_;
};

} // namespace dmk

#endif // DMK_CUDA_FORM_OUTGOING_HPP
