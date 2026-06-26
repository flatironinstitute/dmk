#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
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
    int repeats = 5;
};

struct GridTuneOptions {
    std::string kernel;
    std::string key;
    std::filesystem::path cache_path;
    CudaBenchmarkOptions benchmark;
    bool force = false;
    bool disable = false;
    bool verbose = false;
    bool fallback_to_default_on_failure = true;
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

    std::optional<CachedTuneResult> get(const std::string& key);
    void put(const CachedTuneResult& result);

  private:
    std::filesystem::path path_;
};

std::filesystem::path default_tuning_cache_path();
std::string current_cuda_device_key();
bool env_flag_enabled(const char* name);
std::vector<TuningParams> expand_grid(const std::vector<TuningParameter>& space);
std::string tuning_params_to_string(const TuningParams& params);

void check_cuda(cudaError_t err, const char* where);

template <typename Real>
class AutotuneDeviceRangeSnapshot {
  public:
    AutotuneDeviceRangeSnapshot(Real* base, long offset, std::size_t count, Real* data)
        : base_(base), offset_(offset), count_(count), data_(data) {}

    AutotuneDeviceRangeSnapshot(const AutotuneDeviceRangeSnapshot&) = delete;
    AutotuneDeviceRangeSnapshot& operator=(const AutotuneDeviceRangeSnapshot&) = delete;

    AutotuneDeviceRangeSnapshot(AutotuneDeviceRangeSnapshot&& other) noexcept
        : base_(other.base_),
          offset_(other.offset_),
          count_(other.count_),
          data_(std::exchange(other.data_, nullptr)) {}

    AutotuneDeviceRangeSnapshot& operator=(AutotuneDeviceRangeSnapshot&& other) noexcept {
        if (this != &other) {
            release();
            base_ = other.base_;
            offset_ = other.offset_;
            count_ = other.count_;
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }

    ~AutotuneDeviceRangeSnapshot() {
        release();
    }

    void restore(cudaStream_t stream) const {
        check_cuda(
            cudaMemcpyAsync(
                base_ + offset_,
                data_,
                count_ * sizeof(Real),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "AutotuneDeviceRangeSnapshot restore"
        );
    }

  private:
    Real* base_ = nullptr;
    long offset_ = 0;
    std::size_t count_ = 0;
    Real* data_ = nullptr;

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
AutotuneDeviceRangeSnapshots<Real> make_device_range_snapshots(
    std::vector<std::pair<Real*, long>> ranges,
    std::size_t count,
    cudaStream_t stream
) {
    ranges.erase(
        std::remove_if(
            ranges.begin(),
            ranges.end(),
            [](const auto& range) {
                return range.first == nullptr || range.second < 0;
            }
        ),
        ranges.end()
    );

    std::sort(
        ranges.begin(),
        ranges.end(),
        [](const auto& a, const auto& b) {
            const auto ap = reinterpret_cast<std::uintptr_t>(a.first);
            const auto bp = reinterpret_cast<std::uintptr_t>(b.first);
            return ap < bp || (ap == bp && a.second < b.second);
        }
    );
    ranges.erase(std::unique(ranges.begin(), ranges.end()), ranges.end());

    AutotuneDeviceRangeSnapshots<Real> snapshots;
    snapshots.reserve(ranges.size());

    for (const auto& [base, offset] : ranges) {
        void* raw = nullptr;
        check_cuda(
            cudaMalloc(&raw, count * sizeof(Real)),
            "make_device_range_snapshots cudaMalloc"
        );

        Real* saved = static_cast<Real*>(raw);
        try {
            check_cuda(
                cudaMemcpyAsync(
                    saved,
                    base + offset,
                    count * sizeof(Real),
                    cudaMemcpyDeviceToDevice,
                    stream
                ),
                "make_device_range_snapshots cudaMemcpyAsync"
            );
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
void restore_device_range_snapshots(
    const AutotuneDeviceRangeSnapshots<Real>& snapshots,
    cudaStream_t stream
) {
    for (const auto& snapshot : snapshots) {
        snapshot.restore(stream);
    }
    check_cuda(cudaStreamSynchronize(stream), "restore_device_range_snapshots sync");
}

template <class Launch>
void invoke_cuda_launch(Launch& launch, cudaStream_t stream) {
    if constexpr (std::is_invocable_v<Launch&, cudaStream_t>) {
        launch(stream);
    } else {
        launch();
    }
}

template <class Launch>
double benchmark_cuda_ms(cudaStream_t stream, const CudaBenchmarkOptions& options, Launch&& launch) {
    if (options.warmup < 0 || options.repeats <= 0) {
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
        for (int i = 0; i < options.repeats; ++i) {
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

    return static_cast<double>(elapsed_ms) / static_cast<double>(options.repeats);
}

template <class Constraint, class Benchmark>
GridTuneDecision tune_grid(
    const GridTuneOptions& options,
    const std::vector<TuningParameter>& space,
    const TuningParams& default_params,
    Constraint&& constraint,
    Benchmark&& benchmark
) {
    if (options.disable || env_flag_enabled("DMK_JIT_AUTOTUNE_DISABLE")) {
        return GridTuneDecision{default_params, 0.0, false, false};
    }

    const bool force = options.force || env_flag_enabled("DMK_JIT_AUTOTUNE_FORCE");
    const bool verbose = options.verbose || env_flag_enabled("DMK_JIT_AUTOTUNE_VERBOSE");
    const std::filesystem::path cache_path =
        options.cache_path.empty() ? default_tuning_cache_path() : options.cache_path;
    const std::string device_key = current_cuda_device_key();
    const std::string cache_key = device_key + "|" + options.key;

    JsonTuningCache cache(cache_path);
    if (!force) {
        if (auto cached = cache.get(cache_key)) {
            if (verbose) {
                std::cerr << "[dmk jit autotune] cache hit kernel=" << options.kernel
                          << " params=" << tuning_params_to_string(cached->params)
                          << " runtime_ms=" << cached->runtime_ms << "\n";
            }
            return GridTuneDecision{cached->params, cached->runtime_ms, true, false};
        }
    }

    std::optional<GridTuneDecision> best;
    for (const TuningParams& params : expand_grid(space)) {
        bool accepted = false;
        try {
            accepted = std::invoke(constraint, params);
        } catch (...) {
            accepted = false;
        }
        if (!accepted) {
            continue;
        }

        try {
            const double runtime_ms = std::invoke(benchmark, params);
            if (verbose) {
                std::cout << "[dmk jit autotune] candidate kernel=" << options.kernel
                          << " params=" << tuning_params_to_string(params)
                          << " runtime_ms=" << runtime_ms << "\n";
            }
            if (!best || runtime_ms < best->runtime_ms) {
                best = GridTuneDecision{params, runtime_ms, false, true};
            }
        } catch (...) {
            continue;
        }
    }

    if (!best) {
        if (options.fallback_to_default_on_failure) {
            if (verbose) {
                std::cerr << "[dmk jit autotune] no valid config for kernel=" << options.kernel
                          << "; using defaults params=" << tuning_params_to_string(default_params) << "\n";
            }
            return GridTuneDecision{default_params, 0.0, false, false};
        }
        throw std::runtime_error("tune_grid: no valid configuration for " + options.kernel);
    }

    if (verbose) {
        std::cerr << "[dmk jit autotune] tuned kernel=" << options.kernel
                  << " params=" << tuning_params_to_string(best->params)
                  << " runtime_ms=" << best->runtime_ms << "\n";
    }

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
