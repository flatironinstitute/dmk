// include/dmk/cuda/pw_to_proxy_kernelargs.hpp
#pragma once

namespace dmk::cuda {

template <typename Real>
struct PwToProxyArgs {
    int n_boxes_at_level = 0;
    int n_order = 0;
    int n_pw = 0;
    int n_pw2 = 0;
    int n_charge_dim = 0;
    long pw_in_stride = 0;

    const int *box_ids = nullptr;
    const Real *pw_in_pool = nullptr;
    const Real *pw2poly = nullptr;

    Real *proxy_flat = nullptr;
    const long *proxy_offsets = nullptr;
};

} // namespace dmk::cuda