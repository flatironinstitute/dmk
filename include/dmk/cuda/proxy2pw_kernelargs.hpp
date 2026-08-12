#pragma once

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

    // Fused kernel-FT multiply: 0 off (a separate multiply pass follows), 1 scalar scaling by
    // radialft[slot], 2 the Stokeslet projector f*(k^2 delta - kk) across the three charge dims.
    int multiply_mode = 0;
    const Real *radialft = nullptr; // [n_pw_live], slab order
    Real hpw = 0;                   // mode spacing, for the k vector
    int is_windowed = 0;            // root: Stokeslet zero-mode correction
    // Per (m2,m3) pencil: the slot m1 == 0 would land on, then the live m1 range as lo | hi<<16.
    // Null means the slab is in cube order.
    const int *pencil_slots = nullptr;
};

} // namespace dmk::cuda