#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dmk::cuda::jit {

using TuningParams = std::map<std::string, int>;

struct TuningParameter {
    std::string name;
    std::vector<int> values;
};

struct CachedTuneResult {
    std::string key;
    std::string kernel;
    std::string device;
    double runtime_ms = 0.0;
    TuningParams params;
};

struct CudaBenchmarkOptions {
    int warmup = 2;
    /// Timed as one block; timing launches individually lets the GPU idle down between them.
    int batch_launches = 5;
};

struct GridTuneOptions {
    std::string kernel;
    std::string key;
    std::filesystem::path cache_path;
    bool force = false;
    bool disable = false;
    bool fallback_to_default_on_failure = true;
    /// Build a candidate's kernel without launching it. Optional.
    std::function<void(const TuningParams &)> precompile;
    /// Collapse candidates that generate identical code. Applied before the constraint.
    std::function<TuningParams(TuningParams)> canonicalize;
    /// Feasible-candidate count at or below which the search is exhaustive.
    std::size_t exhaustive_below = 32;
    /// Candidates timed between re-measurements of a control config, whose change over the
    /// sweep is the drift every other measurement is scaled by. Power-limited kernels drift
    /// ~50% across their own sweep, dwarfing the differences being ranked. 0 disables.
    int control_interval = 4;
    /// Repeats of the first candidate until two agree, so the sweep is referenced against the
    /// device's sustained state rather than its ramp into one.
    int max_settle_rounds = 20;
    double settle_tolerance = 0.01;
    /// Margin a candidate must beat the incumbent by to replace it. An identical config
    /// re-times ~2% apart, so anything below this is a coin flip.
    double improvement_threshold = 0.005;
};

struct GridTuneDecision {
    TuningParams params;
    double runtime_ms = 0.0;
    bool from_cache = false;
    bool tuned = false;
};

class JsonTuningCache {
  public:
    explicit JsonTuningCache(std::filesystem::path path);

    std::optional<CachedTuneResult> get(const std::string &key);
    void put(const CachedTuneResult &result);

  private:
    std::filesystem::path path_;
};

std::filesystem::path default_tuning_cache_path();
std::string current_cuda_device_key();
bool env_flag_enabled(const char *name);
std::vector<TuningParams> expand_grid(const std::vector<TuningParameter> &space);
std::string tuning_params_to_string(const TuningParams &params);

/// Compile every candidate across host threads, overridable with DMK_JIT_COMPILE_THREADS.
/// Returns seconds spent.
double precompile_candidates(const std::string &kernel, const std::function<void(const TuningParams &)> &precompile,
                             const std::vector<TuningParams> &candidates);

/// Defined out of line so this header does not drag spdlog into every launcher.
void log_tune_cache_hit(const std::string &kernel, const TuningParams &params, double runtime_ms);
void log_tune_candidate(const std::string &kernel, const TuningParams &params, double runtime_ms, double wall_ms);
void log_tune_defaults(const std::string &kernel, const TuningParams &params);
void log_tune_drift(const std::string &kernel, double control_first_ms, double control_latest_ms);
void log_tune_settle(const std::string &kernel, int rounds, double runtime_ms);
void log_tune_result(const std::string &kernel, const TuningParams &params, double runtime_ms, std::size_t n_candidates,
                     std::size_t n_raw, double compile_s, double bench_s);

void check_cuda(cudaError_t err, const char *where);

template <typename Real>
class AutotuneDeviceRangeSnapshot {
  public:
    AutotuneDeviceRangeSnapshot(Real *base, long offset, std::size_t count, Real *data)
        : base_(base), offset_(offset), count_(count), data_(data) {}

    AutotuneDeviceRangeSnapshot(const AutotuneDeviceRangeSnapshot &) = delete;
    AutotuneDeviceRangeSnapshot &operator=(const AutotuneDeviceRangeSnapshot &) = delete;

    AutotuneDeviceRangeSnapshot(AutotuneDeviceRangeSnapshot &&other) noexcept
        : base_(other.base_), offset_(other.offset_), count_(other.count_), data_(std::exchange(other.data_, nullptr)) {
    }

    AutotuneDeviceRangeSnapshot &operator=(AutotuneDeviceRangeSnapshot &&other) noexcept {
        if (this != &other) {
            release();
            base_ = other.base_;
            offset_ = other.offset_;
            count_ = other.count_;
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }

    ~AutotuneDeviceRangeSnapshot() { release(); }

    void restore(cudaStream_t stream) const {
        check_cuda(cudaMemcpyAsync(base_ + offset_, data_, count_ * sizeof(Real), cudaMemcpyDeviceToDevice, stream),
                   "AutotuneDeviceRangeSnapshot restore");
    }

  private:
    Real *base_ = nullptr;
    long offset_ = 0;
    std::size_t count_ = 0;
    Real *data_ = nullptr;

    void release() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
        }
    }
};

template <typename Real>
using AutotuneDeviceRangeSnapshots = std::vector<AutotuneDeviceRangeSnapshot<Real>>;

template <typename Real>
AutotuneDeviceRangeSnapshots<Real> make_device_range_snapshots(std::vector<std::pair<Real *, long>> ranges,
                                                               std::size_t count, cudaStream_t stream) {
    ranges.erase(std::remove_if(ranges.begin(), ranges.end(),
                                [](const auto &range) { return range.first == nullptr || range.second < 0; }),
                 ranges.end());

    std::sort(ranges.begin(), ranges.end(), [](const auto &a, const auto &b) {
        const auto ap = reinterpret_cast<std::uintptr_t>(a.first);
        const auto bp = reinterpret_cast<std::uintptr_t>(b.first);
        return ap < bp || (ap == bp && a.second < b.second);
    });
    ranges.erase(std::unique(ranges.begin(), ranges.end()), ranges.end());

    AutotuneDeviceRangeSnapshots<Real> snapshots;
    snapshots.reserve(ranges.size());

    for (const auto &[base, offset] : ranges) {
        void *raw = nullptr;
        check_cuda(cudaMalloc(&raw, count * sizeof(Real)), "make_device_range_snapshots cudaMalloc");

        Real *saved = static_cast<Real *>(raw);
        try {
            check_cuda(cudaMemcpyAsync(saved, base + offset, count * sizeof(Real), cudaMemcpyDeviceToDevice, stream),
                       "make_device_range_snapshots cudaMemcpyAsync");
        } catch (...) {
            cudaFree(saved);
            throw;
        }

        snapshots.emplace_back(base, offset, count, saved);
    }

    check_cuda(cudaStreamSynchronize(stream), "make_device_range_snapshots sync");
    return snapshots;
}

template <typename Real>
void restore_device_range_snapshots(const AutotuneDeviceRangeSnapshots<Real> &snapshots, cudaStream_t stream) {
    for (const auto &snapshot : snapshots) {
        snapshot.restore(stream);
    }
    check_cuda(cudaStreamSynchronize(stream), "restore_device_range_snapshots sync");
}

template <class Launch>
void invoke_cuda_launch(Launch &launch, cudaStream_t stream) {
    if constexpr (std::is_invocable_v<Launch &, cudaStream_t>) {
        launch(stream);
    } else {
        launch();
    }
}

/// Mean over one batch. Taking the fastest of several batches instead rewards whichever
/// candidate boosts highest, which is not how these kernels run.
template <class Launch>
double benchmark_cuda_ms(cudaStream_t stream, const CudaBenchmarkOptions &options, Launch &&launch) {
    if (options.warmup < 0 || options.batch_launches <= 0) {
        throw std::runtime_error("benchmark_cuda_ms: invalid benchmark options");
    }

    for (int i = 0; i < options.warmup; ++i) {
        invoke_cuda_launch(launch, stream);
    }
    check_cuda(cudaStreamSynchronize(stream), "benchmark_cuda_ms warmup sync");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    float elapsed_ms = 0.0f;
    try {
        check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
        for (int i = 0; i < options.batch_launches; ++i) {
            invoke_cuda_launch(launch, stream);
        }
        check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    } catch (...) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        throw;
    }

    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");

    return static_cast<double>(elapsed_ms) / options.batch_launches;
}

template <class Constraint, class Benchmark>
GridTuneDecision tune_grid(const GridTuneOptions &options, const std::vector<TuningParameter> &space,
                           const TuningParams &default_params, Constraint &&constraint, Benchmark &&benchmark) {
    if (options.disable || env_flag_enabled("DMK_JIT_AUTOTUNE_DISABLE")) {
        return GridTuneDecision{default_params, 0.0, false, false};
    }

    const bool force = options.force || env_flag_enabled("DMK_JIT_AUTOTUNE_FORCE");
    const std::filesystem::path cache_path =
        options.cache_path.empty() ? default_tuning_cache_path() : options.cache_path;
    const std::string device_key = current_cuda_device_key();
    const std::string cache_key = device_key + "|" + options.key;

    JsonTuningCache cache(cache_path);
    if (!force) {
        if (auto cached = cache.get(cache_key)) {
            log_tune_cache_hit(options.kernel, cached->params, cached->runtime_ms);
            return GridTuneDecision{cached->params, cached->runtime_ms, true, false};
        }
    }

    const auto canonical = [&](TuningParams p) {
        return options.canonicalize ? options.canonicalize(std::move(p)) : p;
    };
    const auto feasible = [&](const TuningParams &p) {
        try {
            return bool(std::invoke(constraint, p));
        } catch (...) {
            return false;
        }
    };

    double compile_s = 0.0;
    double bench_s = 0.0;
    std::optional<GridTuneDecision> best;
    std::map<TuningParams, double> timed;

    std::optional<TuningParams> control;
    double control_first = 0.0;
    double control_latest = 0.0;
    int since_control = 0;

    const auto settle = [&](const TuningParams &p) {
        double previous = 0.0;
        for (int round = 0; round < std::max(1, options.max_settle_rounds); ++round) {
            const double runtime_ms = std::invoke(benchmark, p);
            if (previous > 0.0 && std::abs(runtime_ms - previous) <= options.settle_tolerance * previous) {
                log_tune_settle(options.kernel, round + 1, runtime_ms);
                return runtime_ms;
            }
            previous = runtime_ms;
        }
        log_tune_settle(options.kernel, options.max_settle_rounds, previous);
        return previous;
    };

    const auto run_batch = [&](const std::vector<TuningParams> &batch) {
        std::vector<TuningParams> fresh;
        for (const TuningParams &p : batch) {
            if (!timed.count(p)) {
                fresh.push_back(p);
            }
        }
        if (fresh.empty()) {
            return;
        }
        if (options.precompile && !fresh.empty()) {
            compile_s += precompile_candidates(options.kernel, options.precompile, fresh);
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (const TuningParams &p : fresh) {
            double runtime_ms = std::numeric_limits<double>::infinity();
            const auto candidate_start = std::chrono::steady_clock::now();
            try {
                if (control && options.control_interval > 0 && since_control >= options.control_interval) {
                    const double control_ms = std::invoke(benchmark, *control);
                    since_control = 0;
                    if (control_ms > 0.0 && control_ms < std::numeric_limits<double>::infinity()) {
                        control_latest = control_ms;
                        log_tune_drift(options.kernel, control_first, control_latest);
                    }
                }
                if (!control) {
                    runtime_ms = settle(p);
                    control = p;
                    control_first = runtime_ms;
                    control_latest = runtime_ms;
                } else {
                    runtime_ms = std::invoke(benchmark, p);
                    if (control_first > 0.0 && control_latest > 0.0) {
                        runtime_ms *= control_first / control_latest;
                    }
                }
                ++since_control;
                log_tune_candidate(
                    options.kernel, p, runtime_ms,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - candidate_start)
                        .count());
                if (!best || runtime_ms < best->runtime_ms * (1.0 - options.improvement_threshold)) {
                    best = GridTuneDecision{p, runtime_ms, false, true};
                }
            } catch (...) {
            }
            timed.emplace(p, runtime_ms);
        }
        bench_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    // Sized on surviving candidates, not the raw product: constraints can cut a space by an
    // order of magnitude, and filtering costs no compiles.
    std::vector<TuningParams> all_feasible;
    {
        std::set<TuningParams> distinct;
        for (TuningParams params : expand_grid(space)) {
            params = canonical(std::move(params));
            if (distinct.insert(params).second && feasible(params)) {
                all_feasible.push_back(std::move(params));
            }
        }
    }
    const std::size_t grid_size = all_feasible.size();

    if (env_flag_enabled("DMK_JIT_AUTOTUNE_EXHAUSTIVE") || grid_size <= options.exhaustive_below) {
        run_batch(all_feasible);
    } else {
        // Coordinate descent: sum(|axis|) per pass rather than the product. The full grid
        // stays available via DMK_JIT_AUTOTUNE_EXHAUSTIVE.
        const auto descend = [&](TuningParams current) {
            if (!feasible(current)) {
                return;
            }
            run_batch({current});
            if (!timed.count(current)) {
                return;
            }
            constexpr int max_passes = 3;
            for (int pass = 0; pass < max_passes; ++pass) {
                const TuningParams pass_start = current;

                // One batch per pass: an axis at a time offers 2-3 candidates and leaves most
                // compile threads idle. Speculative compiles are cheap by comparison.
                if (options.precompile) {
                    std::vector<TuningParams> pass_batch;
                    std::set<TuningParams> distinct;
                    for (const TuningParameter &axis : space) {
                        for (int value : axis.values) {
                            TuningParams p = current;
                            p[axis.name] = value;
                            p = canonical(std::move(p));
                            if (distinct.insert(p).second && !timed.count(p) && feasible(p)) {
                                pass_batch.push_back(std::move(p));
                            }
                        }
                    }
                    if (pass_batch.size() > 1) {
                        compile_s += precompile_candidates(options.kernel, options.precompile, pass_batch);
                    }
                }

                for (const TuningParameter &axis : space) {
                    std::vector<TuningParams> batch;
                    std::set<TuningParams> distinct;
                    for (int value : axis.values) {
                        TuningParams p = current;
                        p[axis.name] = value;
                        p = canonical(std::move(p));
                        if (distinct.insert(p).second && feasible(p)) {
                            batch.push_back(std::move(p));
                        }
                    }
                    run_batch(batch);
                    for (const TuningParams &p : batch) {
                        const auto it = timed.find(p);
                        if (it != timed.end() &&
                            it->second < timed.at(current) * (1.0 - options.improvement_threshold)) {
                            current = p;
                        }
                    }
                }
                if (current == pass_start) {
                    break;
                }
            }
        };

        // Corners as well as defaults: a one-axis-at-a-time descent cannot cross an
        // interaction, and optima needing several axes to move together sit near them.
        TuningParams lo = default_params;
        TuningParams hi = default_params;
        for (const TuningParameter &axis : space) {
            if (axis.values.empty()) {
                continue;
            }
            const auto extremes = std::minmax_element(axis.values.begin(), axis.values.end());
            lo[axis.name] = *extremes.first;
            hi[axis.name] = *extremes.second;
        }
        // A rejected corner is a lost start, and the bound rejecting it is often what
        // separates it from the optimum. Walk it back to feasibility instead of dropping it.
        const auto repair = [&](TuningParams p) {
            for (const TuningParameter &axis : space) {
                if (feasible(p)) {
                    break;
                }
                const auto it = default_params.find(axis.name);
                if (it != default_params.end()) {
                    p[axis.name] = it->second;
                }
                p = canonical(std::move(p));
            }
            return p;
        };

        descend(canonical(default_params));
        descend(repair(canonical(std::move(lo))));
        descend(repair(canonical(std::move(hi))));

        // Defaults do not fit this device.
        if (!best && !all_feasible.empty()) {
            descend(all_feasible.front());
        }
    }

    if (!best) {
        if (options.fallback_to_default_on_failure) {
            log_tune_defaults(options.kernel, default_params);
            return GridTuneDecision{default_params, 0.0, false, false};
        }
        throw std::runtime_error("tune_grid: no valid configuration for " + options.kernel);
    }

    log_tune_result(options.kernel, best->params, best->runtime_ms, timed.size(), grid_size, compile_s, bench_s);

    cache.put(CachedTuneResult{
        cache_key,
        options.kernel,
        device_key,
        best->runtime_ms,
        best->params,
    });

    return *best;
}

} // namespace dmk::cuda::jit
