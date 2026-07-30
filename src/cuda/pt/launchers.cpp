#include "launchers.hpp"

#include "../jit/jit_source_utils.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

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

    jit::GridTuneDecision decision = jit::tune_grid(options, space, defaults, constraint, benchmark);

    // A persisted cache entry from an older, narrower tuning space can lack keys
    // the current space defines; backfill them from defaults so a launcher's
    // p.at(name) never throws map::at on a stale cache.
    for (const auto &param : space)
        decision.params.emplace(param.name, defaults.at(param.name));

    std::lock_guard<std::mutex> lock(mutex);
    cache[in_process_key] = decision.params;
    return decision.params;
}

} // namespace dmk::cuda::pt
