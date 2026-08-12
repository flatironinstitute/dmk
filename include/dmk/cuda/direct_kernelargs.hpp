#pragma once

namespace dmk::cuda {

template <typename Real>
struct DirectByBoxArgs {
    int n_work = 0;
    int nlist1_stride = 0;

    Real thresh2 = Real{0};

    const int *direct_work = nullptr;
    const int *target_counts = nullptr;
    const int *box_levels = nullptr;
    const int *list1_count = nullptr;
    const int *list1_flat = nullptr;
    const signed char *list1_shift = nullptr;

    const unsigned char *ifpwexp = nullptr;

    const int *src_counts = nullptr;

    const Real *r_target_flat = nullptr;
    const long *r_target_offsets = nullptr;

    Real *pot_flat = nullptr;
    const long *pot_offsets = nullptr;

    const Real *r_src_flat = nullptr;
    const long *r_src_offsets = nullptr;

    const Real *charge_flat = nullptr;
    const long *charge_offsets = nullptr;

    const Real *normal_flat = nullptr;
    const long *normal_offsets = nullptr;

    const Real *direct_rsc = nullptr;
    const Real *direct_cen = nullptr;
    const Real *direct_d2max = nullptr;

    // Prefilter selectivity counters, only written under PREFILTER_STATS. [0]-[2] are per
    // (source, cull group) so they are directly comparable at any CULL_TILE:
    // [0] scanned, [1] survived the cull, [2] genuinely needed (>=1 lane of the group in
    // range), [3] in-range lane-pairs (the floor), [4] sources needed by the warp as a whole
    // (what the per-pair branch already skips for free).
    unsigned long long *cull_stats = nullptr;
};

} // namespace dmk::cuda
