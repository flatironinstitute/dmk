#pragma once
// The ESP GPU short-range direct sum (esp/short_range.cpp).

#include "state.hpp"

namespace dmk::cuda::esp {

// Generates the residual coefficients at runtime and bakes them into the NVRTC module.
template <typename Real>
void short_range_gpu(GpuState &gpu, int n, const Real *d_pos_aos, const Real *d_charges, Real *d_pot, Real *d_fx,
                     Real *d_fy, Real *d_fz);

} // namespace dmk::cuda::esp
