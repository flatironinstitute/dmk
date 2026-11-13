#pragma once

/// @file
/// V2 point-tree pipeline passes. Each is a free function operating on a
/// `pt::State` (mirroring the CPU two-tier pattern: state owner + free
/// functions on views). The stream/event DAG that orders them lives in
/// `pt::Tree::eval`.

#include <cuda_runtime.h>

#include <dmk/cuda/pt/state.hpp>

namespace dmk::cuda::pt {

/// Near-field (direct) residual. Writes the sorted-order source and target
/// potentials into `state.outputs.d_pot_direct_{src,trg}` on `stream`.
template <typename Real, int DIM>
void direct(State<Real, DIM> &state, cudaStream_t stream);

/// Upward pass: charge2proxy (leaf sources -> proxy) + per-level child->parent
/// tensorprod. Populates `state.scratch.d_proxy_coeffs_upward` on `stream`.
template <typename Real, int DIM>
void upward(State<Real, DIM> &state, cudaStream_t stream);

/// Form-outgoing pass: per-level proxy2pw + kernel-FT multiply (-> the outgoing PW field in
/// d_pw_out_proxy_down) and the windowed root (-> d_window_pw_form_*). Reads d_proxy_coeffs_upward.
template <typename Real, int DIM>
void form_outgoing(State<Real, DIM> &state, cudaStream_t stream);

/// Downward pass: per-level shift_pw (neighbor pw_out -> pw_in), then pw2proxy for the windowed
/// root and every level (pw_in -> proxy), then per-level parent->child tensorprod. The downward
/// proxies overwrite the outgoing PW field in d_pw_out_proxy_down once shift_pw has read it.
template <typename Real, int DIM>
void downward(State<Real, DIM> &state, cudaStream_t stream);

/// Eval-targets pass: evaluates the downward proxy at each eval box's sources
/// and targets. Writes the far-field contribution into d_pot_eval_{src,trg} on
/// `stream`.
template <typename Real, int DIM>
void eval_targets(State<Real, DIM> &state, cudaStream_t stream);

/// Subtracts the on-particle self term from the source-side potential, over the
/// direct work list. Reads and writes `state.outputs.d_pot_direct_src`, so it
/// belongs on the direct stream after `direct`.
template <typename Real, int DIM>
void self_correction(State<Real, DIM> &state, cudaStream_t stream);

} // namespace dmk::cuda::pt
