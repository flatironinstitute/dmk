#include "jit_cache.hpp"
#include <dmk_jit_config.hpp>
#include <sstream>

namespace dmk::cuda::jit{

std::string make_charge2proxy_source(const JitKey& key);

std::string JitKey::to_string() const {
    std::ostringstream os;

    os << name << "|real="<< real << "|sm=" << sm_major << sm_minor;

    for (const auto& [k, v] : params) {
        os << "|" << k << "=" << v;
    }

    return os.str();

}

JitCache::JitCache() {
    CUresult res = cuInit(0);

    if (res != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* msg = nullptr;
        cuGetErrorName(res, &name);
        cuGetErrorString(res, &msg);

        throw std::runtime_error(
            std::string("cuInit failed: ") +
            (name ? name : "<unknown>") + ": " +
            (msg ? msg : "<no message>")
        );
    }
    include_dirs_.push_back(DMK_JIT_INCLUDE_DIR);
    include_dirs_.push_back(DMK_JIT_GENERATED_INCLUDE_DIR);
}

JitCache::JitCache(std::vector<std::string> include_dirs) : JitCache() {
    include_dirs_ = std::move(include_dirs);
}

std::vector<std::string> JitCache::make_nvrtc_options() const {
    std::vector<std::string> opts = extra_options_;

    for (const auto& dir : include_dirs_) {
        opts.push_back("-I" + dir);
    }

    return opts;
}

std::shared_ptr<JitKernel> JitCache::get_kernel(const JitKey& key) {
    const std::string cache_key = key.to_string();

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(cache_key);
    if (it != cache_.end()) {
        return it->second;
    }

    auto kernel = compile_and_load(key);
    cache_.emplace(cache_key, kernel);

    return kernel;
}

std::shared_ptr<JitKernel> JitCache::compile_and_load(const JitKey& key) {
    const std::string source = make_source(key);
    
    CompiledBinary bin = compiler_.compile(
        source,
        key.name + ".cu",
        key.sm_major,
        key.sm_minor,
        make_nvrtc_options()
    );

    CUmodule module = nullptr;

    CUresult res = cuModuleLoadData(&module, static_cast<const void*>(bin.image.data()));

    if (res != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* msg = nullptr;
        cuGetErrorName(res, &name);
        cuGetErrorString(res, &msg);

        throw std::runtime_error(
            std::string("cuModuleLoadData failed: ") +
            (name ? name : "<unknown>") + ": " +
            (msg ? msg: "<no message>")
        );
    }

    CUfunction fn = nullptr;

    res = cuModuleGetFunction(
        &fn,
        module,
        key.name.c_str()
    );

    if (res != CUDA_SUCCESS) {
        cuModuleUnload(module);

        const char* name = nullptr;
        const char* msg = nullptr;
        cuGetErrorName(res, &name);
        cuGetErrorString(res, &msg);

        throw std::runtime_error(
            std::string("cuModuleGetFunction failed for ") +
            key.name + ": " +
            (name ? name : "<unknown>") + ": " +
            (msg ? msg: "<no message>")
        );
    }

    return std::make_shared<JitKernel>(module, fn);
}

std::string JitCache::make_source(const JitKey& key) const {
    if (key.name == "Charge2ProxyKernel") {
        return make_charge2proxy_source(key);
    }

    throw std::runtime_error("Unknown JIT kernel family: " + key.name);
}

} //namespace dmk::cuda::jit