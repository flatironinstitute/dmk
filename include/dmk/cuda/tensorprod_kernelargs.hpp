#pragma once

namespace dmk::cuda {

template <typename Real>
struct TensorprodArgs {
    int n_pairs = 0;
    int n_order = 0;
    int n_charge_dim = 0; // n_tables_down (downward) or n_tables_up (upward)

    // Per-pair (uploaded by the orchestration code per level).
    const int *src_boxes = nullptr;     // [n_pairs] box id read from
    const int *dst_boxes = nullptr;     // [n_pairs] box id written to (additive)
    const int *child_octants = nullptr; // [n_pairs] 0..(2^DIM - 1)
    // [n_pairs] nonzero where this pair is the first writer of its dst_box; phase 3 then
    // stores with `=`. A box has one parent, so at most one pair can claim it.
    const int *assign_dst = nullptr;

    // Gather form: one block per parent, walking that parent's children, which are
    // contiguous in src_boxes/child_octants. The block owns dst_box outright, so the
    // adds are block-local and need no atomics. Leave par_boxes null for the per-pair
    // form above, where dst_boxes drives the block. assign_dst still governs the store,
    // so a parent that charge2proxy also writes must not be marked.
    const int *par_boxes = nullptr;       // [n_par] parent box ids
    const int *par_child_begin = nullptr; // [n_par] first child in src_boxes
    const int *par_child_count = nullptr; // [n_par]
    int n_par = 0;

    // Shared-state device pointers.
    Real *proxy_flat = nullptr;          // d_proxy_coeffs_(up|down)ward (read+write)
    const long *proxy_offsets = nullptr; // [n_boxes]

    // umat matrices, layout: [octant][axis][k_out, k_in] in F-major n_order×n_order.
    // Total length = n_octants * DIM * n_order * n_order. Pass d_p2c for
    // downward, d_c2p for upward.
    const Real *umat_flat = nullptr;

    // Per-block global scratch for the ff/ff2 ping-pong buffers. Block uses
    // scratch + blockIdx.x * scratch_stride; ff occupies the first N3 reals,
    // ff2 the next N3.
    Real *scratch = nullptr;
    long scratch_stride = 0; // reals; = 2 * n_order^3
};

} // namespace dmk::cuda