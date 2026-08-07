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

    /// Store with `=` instead of `+=`. Each element of a box is stored exactly once per
    /// block, so this is a complete write wherever the launch is the box's first writer.
    int assign = 0;
    // Per (m2,m3) pencil: the slot m1 == 0 would land on, then the live m1 range as lo | hi<<16.
    // Null means the slab is in cube order.
    const int *pencil_slots = nullptr;
};

} // namespace dmk::cuda