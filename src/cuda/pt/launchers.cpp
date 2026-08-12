#include "launchers.hpp"

#include "../jit/jit_source_utils.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dmk::cuda::pt {

namespace {

std::mutex &tune_cache_mutex() {
    static std::mutex m;
    return m;
}
std::map<std::string, TuningParams> &tune_cache() {
    static std::map<std::string, TuningParams> c;
    return c;
}

} // namespace

std::string emit_params(const JitKey &key) {
    std::ostringstream ss;
    for (const auto &[name, value] : key.params) {
        ss << "constexpr int " << name << " = " << value << ";\n";
    }
    ss << "\n";
    return ss.str();
}

std::string make_stage_source(std::string_view filename, const JitKey &key, const std::string &prelude,
                              std::string_view label) {
    const jit::SplitSource split = jit::load_split_jit_source(filename, label);
    std::ostringstream ss;
    ss << "using Real = " << key.real << ";\n\n";
    ss << prelude;
    ss << emit_params(key);
    ss << split.header << "\n";
    ss << split.kernel << "\n";
    return ss.str();
}

const cudaDeviceProp &device_prop() {
    static const cudaDeviceProp prop = [] {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p, device);
        return p;
    }();
    return prop;
}

std::size_t device_max_shared_bytes() {
    const cudaDeviceProp &p = device_prop();
    return p.sharedMemPerBlockOptin > 0 ? std::size_t(p.sharedMemPerBlockOptin) : std::size_t(p.sharedMemPerBlock);
}

int resident_blocks_per_sm(std::size_t shared_bytes, int block_size) {
    const cudaDeviceProp &p = device_prop();
    const int by_shared =
        shared_bytes ? int(std::size_t(p.sharedMemPerMultiprocessor) / shared_bytes) : p.maxBlocksPerMultiProcessor;
    return std::max(1, std::min(by_shared, p.maxThreadsPerMultiProcessor / block_size));
}

void set_max_dynamic_smem(const jit::JitKernel &kernel, std::size_t shared_bytes) {
    if (shared_bytes <= 48 * 1024)
        return;
    const CUresult res = cuFuncSetAttribute(kernel.function(), CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                            static_cast<int>(shared_bytes));
    if (res != CUDA_SUCCESS) {
        const char *name = nullptr;
        cuGetErrorName(res, &name);
        throw std::runtime_error(std::string("set_max_dynamic_smem: cuFuncSetAttribute failed: ") +
                                 (name ? name : "<unknown>") + " (" + std::to_string(shared_bytes) + " bytes)");
    }
}

std::optional<TuningParams> autotune_cached(const std::string &tune_key) {
    std::lock_guard<std::mutex> lock(tune_cache_mutex());
    const auto it = tune_cache().find(tune_key);
    if (it == tune_cache().end())
        return std::nullopt;
    return it->second;
}

TuningParams clamp_tiles(TuningParams params, const std::vector<std::pair<const char *, int>> &tile_extents) {
    for (const auto &[name, extent] : tile_extents) {
        const auto it = params.find(name);
        if (it != params.end() && extent > 0) {
            it->second = std::min(it->second, extent);
        }
    }
    return params;
}

TuningParams autotune_config(const std::string &tune_key, const std::string &kernel_label,
                             const std::vector<TuningParameter> &space, const TuningParams &defaults,
                             const std::function<bool(const TuningParams &)> &constraint,
                             const std::function<double(const TuningParams &)> &benchmark,
                             const std::function<void(const TuningParams &)> &precompile,
                             const std::function<TuningParams(TuningParams)> &canonicalize) {
    {
        std::lock_guard<std::mutex> lock(tune_cache_mutex());
        const auto it = tune_cache().find(tune_key);
        if (it != tune_cache().end()) {
            return it->second;
        }
    }

    // The persisted key also carries the shape of the tuning space: widening a space or renaming
    // a parameter leaves stale winners that the backfill below would silently accept.
    std::ostringstream space_sig;
    for (const auto &param : space) {
        space_sig << param.name << '=';
        for (int v : param.values)
            space_sig << v << '.';
        space_sig << ';';
    }

    jit::GridTuneOptions options;
    options.kernel = kernel_label;
    options.key = tune_key + "|sp=" + std::to_string(std::hash<std::string>{}(space_sig.str()));
    options.precompile = precompile;
    options.canonicalize = canonicalize;

    jit::GridTuneDecision decision = jit::tune_grid(options, space, defaults, constraint, benchmark);

    // A persisted cache entry from an older, narrower tuning space can lack keys
    // the current space defines; backfill them from defaults so a launcher's
    // p.at(name) never throws map::at on a stale cache.
    for (const auto &param : space)
        decision.params.emplace(param.name, defaults.at(param.name));

    // tune_grid falls back to `defaults` when nothing passes `constraint`, and returns
    // persisted entries unchecked, so an infeasible config can reach a launch.
    if (!constraint(decision.params))
        throw std::runtime_error(std::string("autotune_config: no feasible tuning config for ") + kernel_label +
                                 " (key " + tune_key +
                                 "); the tuning space likely cannot satisfy this device's shared-memory limit");

    std::lock_guard<std::mutex> lock(tune_cache_mutex());
    tune_cache()[tune_key] = decision.params;
    return decision.params;
}

} // namespace dmk::cuda::pt
