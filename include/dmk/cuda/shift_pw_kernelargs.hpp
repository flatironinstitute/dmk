#pragma once

namespace dmk::cuda {

/// One surviving source box in a target box's shift list. `pw_off` is that box's
/// entry in pw_out (complex units, as pw_out_offsets stores it) and `shift_ind`
/// indexes wpwshift. 16 bytes so the device fetches an entry in one load.
struct alignas(16) ShiftPwNeighbor {
    long pw_off;
    int shift_ind;
    int pad;
};

template <typename Real>
struct ShiftPwArgs {
    int n_boxes_at_level = 0;
    int n_neighbors = 0;
    int n_charge_dim = 0;
    int n_pw_modes = 0;

    long pw_in_stride = 0;

    const int *box_ids = nullptr;
    const long *pw_out_offsets = nullptr;

    // Pre-filtered shift lists, CSR by absolute box id: box b owns entries
    // shift_nbr[shift_nbr_offsets[b] .. shift_nbr_offsets[b + 1]).
    const ShiftPwNeighbor *shift_nbr = nullptr;
    const int *shift_nbr_offsets = nullptr;

    const Real *pw_out_flat = nullptr;
    const Real *wpwshift = nullptr;

    Real *pw_in_pool = nullptr;
};
} // namespace dmk::cuda
