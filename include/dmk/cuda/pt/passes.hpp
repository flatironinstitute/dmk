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

/// Form-outgoing pass: per-level proxy2pw + kernel-FT multiply (-> d_pw_out) and
/// the windowed root (-> d_proxy_coeffs_downward[0]). Reads d_proxy_coeffs_upward.
template <typename Real, int DIM>
void form_outgoing(State<Real, DIM> &state, cudaStream_t stream);

/// Downward pass: per-level shift_pw (neighbor pw_out -> pw_in) + pw2proxy
/// (pw_in -> proxy) then per-level parent->child tensorprod. Reads d_pw_out,
/// accumulates into d_proxy_coeffs_downward on `stream`.
template <typename Real, int DIM>
void downward(State<Real, DIM> &state, cudaStream_t stream);

} // namespace dmk::cuda::pt
