#pragma once

namespace dmk::cuda {

/// Target boxes per shift_pw block. A level's box list is Morton-ordered, so
/// consecutive entries are spatial neighbours and their source sets overlap
/// heavily: one block covering 8 of them visits each distinct source once
/// instead of once per target. Set to 1 to recover the one-box-per-block kernel.
constexpr int kShiftGroup = 8;

/// One surviving source box in a target box's shift list. `pw_off` is that box's
/// entry in pw_out (complex units, as pw_out_offsets stores it) and `shift_ind`
/// indexes wpwshift. 16 bytes so the device fetches an entry in one load.
struct alignas(16) ShiftPwNeighbor {
    long pw_off;
    int shift_ind;
    int pad;
};

/// One distinct source box for a whole group, with the shift index each group
/// member needs for it; -1 where that member is not a neighbour of this source.
/// 16 bytes at kShiftGroup == 8, so again one load per entry.
struct alignas(16) ShiftPwGroupSrc {
    long pw_off;
    signed char shift_ind[kShiftGroup];
};

template <typename Real>
struct ShiftPwArgs {
    int n_boxes_at_level = 0;
    int n_groups_at_level = 0;
    int n_neighbors = 0;
    int n_charge_dim = 0;
    int n_pw_modes = 0;

    long pw_in_stride = 0;

    const int *box_ids = nullptr;
    const long *pw_out_offsets = nullptr;

    // Merged shift lists, CSR by level-local group index: group g owns entries
    // group_src[group_offsets[g] .. group_offsets[g + 1]). Group g's members are
    // level-local boxes g*kShiftGroup .. +kShiftGroup, clipped to n_boxes_at_level.
    const ShiftPwGroupSrc *group_src = nullptr;
    const int *group_offsets = nullptr;

    const Real *pw_out_flat = nullptr;
    const Real *wpwshift = nullptr;

    Real *pw_in_pool = nullptr;
};
} // namespace dmk::cuda
