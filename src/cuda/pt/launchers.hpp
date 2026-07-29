#pragma once

// Thin, stage-agnostic launcher helpers over the shared JIT core. Each V2 pass
// launcher composes these instead of duplicating the source-assembly + tune
// boilerplate:
//   - emit_params:      key.params -> `constexpr int NAME = VALUE;` lines
//   - make_stage_source: prelude + emit_params + load_split(<file>)
//   - autotune_config:   grid tune (shared tune_grid) + in-process cache

#include "../jit/autotune.hpp"
#include "../jit/jit_types.hpp"

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace dmk::cuda::pt {

using jit::JitKey;
using jit::TuningParameter;
using jit::TuningParams;

/// Emit each integer key param as a `constexpr int`. Replaces the per-stage
/// make_*_specialization_constants helpers.
std::string emit_params(const JitKey &key);

/// Assemble a full JIT translation unit: a stage-specific `prelude` (types,
/// baked coeffs, evaluator typedef), then emit_params(key), then the split
/// device source loaded from `filename` (header + kernel around // KERNEL_START).
std::string make_stage_source(std::string_view filename, const JitKey &key, const std::string &prelude,
                              std::string_view label);

/// Grid-tune over `space` and return the winning params. Wraps the shared
/// tune_grid (persistent JSON cache + DMK_JIT_AUTOTUNE_* env controls) with an
/// in-process cache keyed by device + `tune_key` so repeat evals don't re-tune.
/// `benchmark` returns a runtime in ms for a candidate; `constraint` rejects
/// infeasible candidates.
TuningParams autotune_config(const std::string &tune_key, const std::string &kernel_label,
                             const std::vector<TuningParameter> &space, const TuningParams &defaults,
                             const std::function<bool(const TuningParams &)> &constraint,
                             const std::function<double(const TuningParams &)> &benchmark);

} // namespace dmk::cuda::pt
