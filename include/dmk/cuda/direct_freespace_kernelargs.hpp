#pragma once

namespace dmk::cuda {

// Brute-force free-space direct summation: one flat source list against one flat target list.
// `lambda` is read only by Yukawa, and is a runtime field so one module serves every lambda.
template <typename Real>
struct DirectFreespaceArgs {
    int n_src = 0;
    int n_trg = 0;

    Real lambda = Real{0};

    const Real *r_src = nullptr;
    const Real *charge = nullptr;
    const Real *normal = nullptr;

    const Real *r_trg = nullptr;
    Real *pot = nullptr;
};

} // namespace dmk::cuda
