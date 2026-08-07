#pragma once

#ifndef __CUDACC_RTC__
#include <cstdlib>
#endif

namespace dmk::cuda {

/// Upper bound on targets per shift_pw block, and the capacity of
/// ShiftPwGroupSrc::shift_ind. 8 keeps that struct at 16 bytes.
constexpr int kShiftGroupMax = 8;

/// Target boxes per shift_pw block; the merge loop in state.cpp covers the
/// grouping itself. A bigger group cuts source-slab traffic but the block holds
/// acc[group][n_charge_dim] complex accumulators, and the kernel is latency-bound
/// with occupancy limited by registers, so cap that accumulator at 16 registers.
/// Powers of two only, since a group is a Morton run.
/// DMK_SHIFT_GROUP overrides; 1 recovers the one-box-per-block kernel. Host-only,
/// as NVRTC rejects unannotated functions and the kernel takes the chosen size as
/// its SHIFT_GROUP define.
#ifndef __CUDACC_RTC__
inline int shift_group_size(int n_charge_dim) {
    static const int forced = [] {
        const char *v = std::getenv("DMK_SHIFT_GROUP");
        return v ? std::atoi(v) : 0;
    }();
    if (forced > 0)
        return forced < kShiftGroupMax ? forced : kShiftGroupMax;
    int g = kShiftGroupMax;
    while (g > 1 && 2 * g * n_charge_dim > 16)
        g >>= 1;
    return g;
}
#endif

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
/// 16 bytes, so again one load per entry. Entries past the group's size are unused.
struct alignas(16) ShiftPwGroupSrc {
    long pw_off;
    signed char shift_ind[kShiftGroupMax];
};

template <typename Real>
struct ShiftPwArgs {
    int n_boxes_at_level = 0;
    int n_groups_at_level = 0;
    int n_neighbors = 0;
    int n_charge_dim = 0;
    int n_pw_modes = 0;
    int n_pw_live = 0;

    long pw_in_stride = 0;

    const int *box_ids = nullptr;
    const long *pw_out_offsets = nullptr;

    // Merged shift lists, CSR by level-local group index: group g owns entries
    // group_src[group_offsets[g] .. group_offsets[g + 1]). Group g's members are
    // level-local boxes g*SHIFT_GROUP .. +SHIFT_GROUP, clipped to n_boxes_at_level.
    const ShiftPwGroupSrc *group_src = nullptr;
    const int *group_offsets = nullptr;

    const Real *pw_out_flat = nullptr;
    const Real *wpwshift = nullptr;

    Real *pw_in_pool = nullptr;
};
} // namespace dmk::cuda
