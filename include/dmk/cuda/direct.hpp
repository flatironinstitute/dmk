#pragma once

#include <dmk.h>

namespace dmk::cuda {

/// Brute-force free-space direct summation on the GPU. Reads n_dim, kernel and fparam from
/// `params`; `eval` selects the output field. All pointers are host pointers and `pot` is
/// overwritten; the caller selects and scopes the CUDA device.
template <typename Real>
void direct_freespace(const pdmk_params &params, dmk_eval_type eval, int n_src, const Real *r_src, const Real *charge,
                      const Real *normal, int n_trg, const Real *r_trg, Real *pot);

} // namespace dmk::cuda
