#pragma once

// Thin, stage-agnostic launcher helpers over the shared JIT core. Each V2 pass
// launcher composes these instead of duplicating the source-assembly + tune
// boilerplate:
//   - emit_params:      key.params -> `constexpr int NAME = VALUE;` lines
//   - make_stage_source: prelude + emit_params + load_split(<file>)
//   - autotune_config:   grid tune (shared tune_grid) + in-process cache

#include "../jit/autotune.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_types.hpp"

#include <cuda_runtime.h>

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
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

/// In-process lookup of an already-tuned config for `tune_key` (the same cache
/// autotune_config populates). Returns nullopt on first encounter.
std::optional<TuningParams> autotune_cached(const std::string &tune_key);

/// Grid-tune over `space` and return the winning params. Wraps the shared
/// tune_grid (persistent JSON cache + DMK_JIT_AUTOTUNE_* env controls) with an
/// in-process cache keyed by device + `tune_key` so repeat evals don't re-tune.
/// `benchmark` returns a runtime in ms for a candidate; `constraint` rejects
/// infeasible candidates.
TuningParams autotune_config(const std::string &tune_key, const std::string &kernel_label,
                             const std::vector<TuningParameter> &space, const TuningParams &defaults,
                             const std::function<bool(const TuningParams &)> &constraint,
                             const std::function<double(const TuningParams &)> &benchmark,
                             const std::function<void(const TuningParams &)> &precompile,
                             const std::function<TuningParams(TuningParams)> &canonicalize);

/// Clamp each tiling parameter to the extent it tiles, so tiles wider than their loop bound
/// collapse onto it. Only valid where the extent is fixed at compile time.
TuningParams clamp_tiles(TuningParams params, const std::vector<std::pair<const char *, int>> &tile_extents);

/// Static __shared__ bytes a compiled kernel uses; shares the per-block budget with the dynamic
/// request, so the opt-in threshold has to include it.
std::size_t static_smem(const jit::JitKernel &kernel);

/// Opt in to >48KB shared memory for a compiled kernel (no-op below the static limit). Must be
/// called before launching with that shared_bytes.
void set_max_dynamic_smem(const jit::JitKernel &kernel, std::size_t shared_bytes);

/// Properties of the current device, queried once and cached. GPU eval is
/// single-device (single rank).
const cudaDeviceProp &device_prop();

/// Max opt-in dynamic shared memory per block for the current device.
std::size_t device_max_shared_bytes();

/// Blocks per SM a launch can reach given its shared-memory footprint and block
/// size. Feed this to `__launch_bounds__` as a baked parameter on any kernel
/// whose occupancy shared memory caps: left to itself ptxas budgets registers
/// for an occupancy it will never get and pays for the shortfall by serializing
/// loads.
int resident_blocks_per_sm(std::size_t shared_bytes, int block_size);

/// Tune `launch_one` over `space` then run the winning config on `stream`.
/// For additive kernels (whose re-runs corrupt their output), pass the output
/// buffer base + element count so the tuner snapshots/restores it around each
/// timed run; pass `snapshot_base == nullptr` for idempotent kernels. The final
/// launch runs on the pre-tune buffer state, so the caller sets that state
/// (e.g. zero / prior-stage result) before calling.
template <typename Real, typename LaunchOne>
void autotuned_launch(const std::string &tune_key, const std::string &kernel_label,
                      const std::vector<TuningParameter> &space, const TuningParams &defaults,
                      const std::function<bool(const TuningParams &)> &constraint, LaunchOne &&launch_one,
                      Real *snapshot_base, std::size_t snapshot_count, cudaStream_t stream,
                      const std::function<TuningParams(TuningParams)> &canonicalize) {
    jit::AutotuneDeviceRangeSnapshots<Real> snap;
    bool have_snap = false;
    const std::function<double(const TuningParams &)> benchmark = [&](const TuningParams &p) -> double {
        if (snapshot_base && !have_snap) {
            std::vector<std::pair<Real *, long>> ranges{{snapshot_base, 0}};
            snap = jit::make_device_range_snapshots<Real>(std::move(ranges), snapshot_count, stream);
            have_snap = true;
        }
        if (have_snap)
            jit::restore_device_range_snapshots(snap, stream);
        return jit::benchmark_cuda_ms(stream, jit::CudaBenchmarkOptions{},
                                      [&](cudaStream_t s) { launch_one(p, s, false); });
    };
    const std::function<void(const TuningParams &)> precompile = [&](const TuningParams &p) {
        launch_one(p, cudaStream_t{}, true);
    };
    const TuningParams config =
        autotune_config(tune_key, kernel_label, space, defaults, constraint, benchmark, precompile, canonicalize);
    if (have_snap)
        jit::restore_device_range_snapshots(snap, stream);
    launch_one(config, stream, false);
}

} // namespace dmk::cuda::pt
