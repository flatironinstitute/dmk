#pragma once

namespace dmk::cuda {

// Shared verbatim between the host launcher and the NVRTC device source, which is what guarantees
// layout agreement -- so no includes and no std:: here.
//
// Positions/charges are the cell-sorted SoA arrays; pg_sorted is the interleaved
// [pot, d/dx, d/dy, d/dz] accumulator in that same order, un-permuted afterwards. max_tiles sizes
// the pruned strategies' dynamic shared-memory table; prune_stats is read only when PRUNE_STATS.
template <typename Real>
struct EspSrArgs {
    int nc = 0;
    int out_dim = 0;

    Real rsc = Real{0};
    Real cen = Real{0};
    Real r_c_sq = Real{0};

    const int *cell_start = nullptr;
    const Real *xs = nullptr;
    const Real *ys = nullptr;
    const Real *zs = nullptr;
    const Real *qs = nullptr;

    const int *nbc_tab = nullptr;
    const Real *off_tab = nullptr;

    Real *pg_sorted = nullptr;

    int max_tiles = 0;
    unsigned long long *prune_stats = nullptr;
};

} // namespace dmk::cuda
