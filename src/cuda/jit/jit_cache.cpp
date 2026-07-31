#include "jit_cache.hpp"
#include <dmk_jit_config.hpp>

namespace dmk::cuda::jit {

std::string JitKey::to_string() const {
    std::string s;
    s.reserve(name.size() + real.size() + 24 + params.size() * 16);
    s += name;
    s += "|real=";
    s += real;
    s += "|sm=";
    s += std::to_string(sm_major);
    s += std::to_string(sm_minor);
    for (const auto &[k, v] : params) {
        s += '|';
        s += k;
        s += '=';
        s += std::to_string(v);
    }
    return s;
}

JitCache::JitCache() {
    CUresult res = cuInit(0);

    if (res != CUDA_SUCCESS) {
        const char *name = nullptr;
        const char *msg = nullptr;
        cuGetErrorName(res, &name);
        cuGetErrorString(res, &msg);

        throw std::runtime_error(std::string("cuInit failed: ") + (name ? name : "<unknown>") + ": " +
                                 (msg ? msg : "<no message>"));
    }
    include_dirs_.push_back(DMK_JIT_INCLUDE_DIR);
    include_dirs_.push_back(DMK_JIT_GENERATED_INCLUDE_DIR);

    {
        auto err = cudaGetDevice(&device_);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(err));
        }
    }

    cudaDeviceProp prop{};
    auto err = cudaGetDevice(&device_);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(err));
    }

    err = cudaGetDeviceProperties(&prop, device_);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(err));
    }

    sm_major_ = prop.major;
    sm_minor_ = prop.minor;
}

JitCache::JitCache(std::vector<std::string> include_dirs) : JitCache() { include_dirs_ = std::move(include_dirs); }

std::vector<std::string> JitCache::make_nvrtc_options() const {
    std::vector<std::string> opts = extra_options_;

    for (const auto &dir : include_dirs_) {
        opts.push_back("-I" + dir);
    }

    return opts;
}

std::shared_ptr<JitKernel> JitCache::get_kernel_from_source(const JitKey &key,
                                                            const std::function<std::string()> &source_fn,
                                                            const std::string &name_expression) {
    {
        const std::string cache_key = key.to_string();
        std::lock_guard<std::mutex> guard(mutex_);
        auto it = cache_.find(cache_key);
        if (it != cache_.end())
            return it->second;
    }
    // Miss: build the source (outside the fast path) and compile via the eager
    // overload, which re-checks the cache under lock to absorb a compile race.
    return get_kernel_from_source(key, source_fn(), name_expression);
}

std::shared_ptr<JitKernel> JitCache::get_kernel_from_source(const JitKey &key, const std::string &source,
                                                            const std::string &name_expression) {
    const std::string cache_key = key.to_string();

    std::lock_guard<std::mutex> guard(mutex_);

    auto it = cache_.find(cache_key);

    if (it != cache_.end()) {
        return it->second;
    }

    CompiledBinary bin =
        compiler_.compile(source, key.name + ".cu", key.sm_major, key.sm_minor, make_nvrtc_options(), name_expression);
    CUmodule module = nullptr;

    CUresult res = cuModuleLoadData(&module, static_cast<const void *>(bin.image.data()));

    if (res != CUDA_SUCCESS) {
        const char *name = nullptr;
        const char *msg = nullptr;

        cuGetErrorName(res, &name);
        cuGetErrorString(res, &msg);

        throw std::runtime_error(std::string("cuModuleLoadData failed: ") + (name ? name : "<unknown>") + ": " +
                                 (msg ? msg : "<no message>"));
    }

    const std::string function_name = !bin.lowered_name.empty() ? bin.lowered_name : key.name;

    CUfunction function = nullptr;

    res = cuModuleGetFunction(&function, module, function_name.c_str());

    if (res != CUDA_SUCCESS) {
        cuModuleUnload(module);

        const char *name = nullptr;
        const char *msg = nullptr;

        cuGetErrorName(res, &name);
        cuGetErrorString(res, &msg);

        throw std::runtime_error(std::string("cuModuleGetFunction failed for ") + key.name + ": " +
                                 (name ? name : "<unknown>") + ": " + (msg ? msg : "<no message>"));
    }

    auto kernel = std::make_shared<JitKernel>(module, function);

    cache_.emplace(cache_key, kernel);

    return kernel;
}

} // namespace dmk::cuda::jit
