#ifndef DMK_CUDA_PROXY2PW_KERNELS_HPP
#define DMK_CUDA_PROXY2PW_KERNELS_HPP

// Per-block GPU proxycharge2pw: one block per box, projects the box's upward
// proxy expansion onto plane-wave modes via 3 axis-wise complex GEMMs with
// poly2pw. Output is interleaved complex.

#include <cuda_runtime.h>

#include <vector>
namespace dmk::cuda {

template <typename Real>
struct Proxy2PwArgs {
    int n_boxes_at_level = 0; // gridDim.x
    int n_order = 0;
    int n_pw = 0;
    int n_pw2 = 0;        // (n_pw + 1) / 2
    int n_charge_dim = 0; // = n_tables_up

    const int *box_ids = nullptr;        // [n_boxes_at_level]
    const Real *proxy_flat = nullptr;    // d_proxy_coeffs_upward (real)
    const long *proxy_offsets = nullptr; // [n_boxes]; -1 = no upward proxy
    const Real *poly2pw = nullptr;       // (n_pw rows, n_order cols), F-major, interleaved complex

    Real *dst_flat = nullptr;          // interleaved complex
    const long *dst_offsets = nullptr; // in COMPLEX units
    long dst_stride_complex = 0;       // if dst_offsets is null, use box_idx * dst_stride_complex
};

template <typename Real>
void launch_proxy2pw_dispatch(int dim, const Proxy2PwArgs<Real> &args, cudaStream_t stream);

template <typename Real>
void launch_proxy2pw_multilevel_dispatch(
    int dim,
    const std::vector<Proxy2PwArgs<Real> > &pa_h,
    cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_PROXY2PW_KERNELS_HPP
