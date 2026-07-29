#include "launchers.hpp"

#include "../jit/jit_source_utils.hpp"

#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <sstream>

namespace dmk::cuda::pt {

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
    ss << prelude;
    ss << emit_params(key);
    ss << split.header << "\n";
    ss << split.kernel << "\n";
    return ss.str();
}

TuningParams autotune_config(const std::string &tune_key, const std::string &kernel_label,
                             const std::vector<TuningParameter> &space, const TuningParams &defaults,
                             const std::function<bool(const TuningParams &)> &constraint,
                             const std::function<double(const TuningParams &)> &benchmark) {
    static std::mutex mutex;
    static std::map<std::string, TuningParams> cache;

    int device = 0;
    cudaGetDevice(&device);
    const std::string in_process_key = std::to_string(device) + "|" + tune_key;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = cache.find(in_process_key);
        if (it != cache.end()) {
            return it->second;
        }
    }

    jit::GridTuneOptions options;
    options.kernel = kernel_label;
    options.key = tune_key;
    options.benchmark = jit::CudaBenchmarkOptions{2, 5};

    const jit::GridTuneDecision decision = jit::tune_grid(options, space, defaults, constraint, benchmark);

    std::lock_guard<std::mutex> lock(mutex);
    cache[in_process_key] = decision.params;
    return decision.params;
}

} // namespace dmk::cuda::pt
