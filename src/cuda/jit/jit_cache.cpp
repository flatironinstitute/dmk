#include "jit_cache.hpp"
#include <dmk_jit_config.hpp>
#include <sstream>

namespace dmk::cuda::jit {

std::string make_charge2proxy_source(const JitKey &key);
std::string make_tensorprod_source(const JitKey &key);
std::string make_proxy2pw_source(const JitKey &key);
std::string make_pw2proxy_source(const JitKey &key);
std::string make_shift_pw_source(const JitKey &key);
std::string make_direct_by_box_source(const JitKey& key);
std::string make_eval_targets_source(const JitKey& key);
std::string make_self_correction_source(const JitKey& key);
std::string make_multiply_kernelft_source(const JitKey& key);
std::string make_shared_state_source(const JitKey& key);

std::string JitKey::to_string() const {
    std::ostringstream os;

    os << name << "|real=" << real << "|sm=" << sm_major << sm_minor;

    for (const auto &[k, v] : params) {
        os << "|" << k << "=" << v;
    }

    return os.str();
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

    cudaDeviceProp prop{};
    auto err = cudaGetDeviceProperties(&prop, device_);
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

std::shared_ptr<JitKernel> JitCache::get_kernel(const JitKey &key) {
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

std::shared_ptr<JitKernel> JitCache::compile_and_load(const JitKey &key) {
    const std::string source = make_source(key);

    CompiledBinary bin = compiler_.compile(source, key.name + ".cu", key.sm_major, key.sm_minor, make_nvrtc_options());

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

    CUfunction fn = nullptr;

    res = cuModuleGetFunction(&fn, module, key.name.c_str());

    if (res != CUDA_SUCCESS) {
        cuModuleUnload(module);

        const char *name = nullptr;
        const char *msg = nullptr;
        cuGetErrorName(res, &name);
        cuGetErrorString(res, &msg);

        throw std::runtime_error(std::string("cuModuleGetFunction failed for ") + key.name + ": " +
                                 (name ? name : "<unknown>") + ": " + (msg ? msg : "<no message>"));
    }

    return std::make_shared<JitKernel>(module, fn);
}

std::string JitCache::make_source(const JitKey &key) const {
    if (key.name == "Charge2ProxyKernel") {
        return make_charge2proxy_source(key);
    }
    if (key.name == "TensorprodKernel") {
        return make_tensorprod_source(key);
    }
    if (key.name == "Proxy2PwKernel") {
        return make_proxy2pw_source(key);
    }
    if (key.name == "Proxy2PwMultiLevelKernel") {
        return make_proxy2pw_source(key);
    }
    if (key.name == "PwToProxyKernel" || key.name == "PwToProxyMultiLevelKernel") {
        return make_pw2proxy_source(key);
    }
    if (key.name == "ShiftPwKernel") {
        return make_shift_pw_source(key);
    }
    if (key.name.rfind("DirectByBoxKernel_", 0) == 0) {
        return make_direct_by_box_source(key);
    }
    if (key.name == "EvalTargetsByBoxKernel") {
        return make_eval_targets_source(key);
    }
    if (key.name == "SelfCorrectionKernel") {
        return make_self_correction_source(key);
    }
    if (key.name == "MultiplyCd2pByBoxKernel" ||
        key.name == "MultiplyStokeslet3DByBoxKernel" ||
        key.name == "MultiplyStresslet3DByBoxKernel") {
        return make_multiply_kernelft_source(key);
    }
    if (key.name == "AccumulateAndScatterKernel") {
        return make_shared_state_source(key);
    }
    throw std::runtime_error("Unknown JIT kernel family: " + key.name);
}

} // namespace dmk::cuda::jit
