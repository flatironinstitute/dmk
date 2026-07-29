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

} // namespace dmk::cuda::pt
