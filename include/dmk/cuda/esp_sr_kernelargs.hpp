#pragma once

namespace dmk::cuda {

// Shared verbatim with the NVRTC device source, so no includes and no std:: here.
//
// Cell-sorted throughout: qs holds KERNEL_INPUT_DIM charge planes and ns NORMAL_DIM normal planes,
// both of stride n_sorted; pg_sorted is the KERNEL_OUTPUT_DIM-interleaved accumulator.
template <typename Real>
struct EspSrArgs {
    int nc = 0;
    int n_sorted = 0; // plane stride for qs/ns

    Real rsc = Real{0};
    Real cen = Real{0};
    Real r_c_sq = Real{0};

    const int *cell_start = nullptr;
    const Real *xs = nullptr;
    const Real *ys = nullptr;
    const Real *zs = nullptr;
    const Real *qs = nullptr;
    const Real *ns = nullptr;

    const int *nbc_tab = nullptr;
    const Real *off_tab = nullptr;

    Real *pg_sorted = nullptr;

    int max_tiles = 0;
    unsigned long long *prune_stats = nullptr;
};

} // namespace dmk::cuda
