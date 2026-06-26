#include "proxy2pw_launcher.hpp"

#include "autotune.hpp"
#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/proxy2pw_kernels.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda::jit {
namespace {

constexpr int PROXY2PW_Z_TILE = 4;
constexpr int PROXY2PW_I_TILE = 4;
constexpr int PROXY2PW_M1_TILE = 4;
constexpr int PROXY2PW_M2_TILE = 4;
constexpr int PROXY2PW_BLOCK_SIZE = 128;

struct Proxy2PwLaunchConfig {
    int blocksize = PROXY2PW_BLOCK_SIZE;
    int z_tile = PROXY2PW_Z_TILE;
    int i_tile = PROXY2PW_I_TILE;
    int m1_tile = PROXY2PW_M1_TILE;
    int m2_tile = PROXY2PW_M2_TILE;
};

Proxy2PwLaunchConfig default_proxy2pw_config(int blocksize = PROXY2PW_BLOCK_SIZE) {
    return Proxy2PwLaunchConfig{
        blocksize,
        PROXY2PW_Z_TILE,
        PROXY2PW_I_TILE,
        PROXY2PW_M1_TILE,
        PROXY2PW_M2_TILE,
    };
}

int tuning_param_or(const TuningParams& params, const char* name, int fallback) {
    const auto it = params.find(name);
    return it == params.end() ? fallback : it->second;
}

TuningParams proxy2pw_tuning_params(const Proxy2PwLaunchConfig& config) {
    return TuningParams{
        {"BLOCK_SIZE", config.blocksize},
        {"Z_TILE", config.z_tile},
        {"I_TILE", config.i_tile},
        {"M1_TILE", config.m1_tile},
        {"M2_TILE", config.m2_tile},
    };
}

Proxy2PwLaunchConfig proxy2pw_config_from_params(const TuningParams& params) {
    const Proxy2PwLaunchConfig defaults = default_proxy2pw_config();
    return Proxy2PwLaunchConfig{
        tuning_param_or(params, "BLOCK_SIZE", defaults.blocksize),
        tuning_param_or(params, "Z_TILE", defaults.z_tile),
        tuning_param_or(params, "I_TILE", defaults.i_tile),
        tuning_param_or(params, "M1_TILE", defaults.m1_tile),
        tuning_param_or(params, "M2_TILE", defaults.m2_tile),
    };
}

std::string make_specialization_constants(const JitKey& key) {
    const int blocksize    = required_int_param(key, "BLOCK_SIZE", "Proxy2Pw");
    const int n_order      = required_int_param(key, "N_ORDER", "Proxy2Pw");
    const int n_pw         = required_int_param(key, "N_PW", "Proxy2Pw");
    const int n_pw2        = required_int_param(key, "N_PW2", "Proxy2Pw");
    const int n_charge_dim = required_int_param(key, "N_CHARGE_DIM", "Proxy2Pw");
    const int z_tile       = required_int_param(key, "Z_TILE", "Proxy2Pw");
    const int i_tile       = required_int_param(key, "I_TILE", "Proxy2Pw");
    const int m1_tile      = required_int_param(key, "M1_TILE", "Proxy2Pw");
    const int m2_tile      = required_int_param(key, "M2_TILE", "Proxy2Pw");
    const std::string real_type = key.real;

    std::ostringstream ss;

    ss << "#include <dmk/cuda/proxy2pw_kernelargs.hpp>\n";
    ss << "using dmk::cuda::Proxy2PwArgs;\n\n";
    ss << "constexpr int N_ORDER      = " << n_order << ";\n";
    ss << "constexpr int N_PW         = " << n_pw << ";\n";
    ss << "constexpr int N_PW2        = " << n_pw2 << ";\n";
    ss << "constexpr int N_CHARGE_DIM = " << n_charge_dim << ";\n";
    ss << "constexpr int BLOCK_SIZE   = " << blocksize << ";\n";
    ss << "constexpr int PROXY2PW_Z_TILE  = " << z_tile << ";\n";
    ss << "constexpr int PROXY2PW_I_TILE  = " << i_tile << ";\n";
    ss << "constexpr int PROXY2PW_M1_TILE = " << m1_tile << ";\n";
    ss << "constexpr int PROXY2PW_M2_TILE = " << m2_tile << ";\n";
    ss << "using Real = " << real_type << ";\n\n";

    return ss.str();
}

std::size_t proxy2pw_shared_bytes(
    int n_order,
    int n_pw,
    int z_tile,
    std::size_t sizeof_real
) {
    const std::size_t complex_count =
        std::size_t(z_tile) *
        (
            std::size_t(n_order) * std::size_t(n_order) +
            std::size_t(n_order) * std::size_t(n_pw)
        ) +
        std::size_t(n_order) * std::size_t(n_pw);

    return std::size_t{2} * complex_count * sizeof_real;
}

template <typename Real>
void check_proxy2pw_shape_or_throw(
    const dmk::cuda::Proxy2PwArgs<Real>& a
) {
    if (a.n_order <= 0 || a.n_pw <= 0 || a.n_pw2 <= 0 || a.n_charge_dim <= 0) {
        throw std::runtime_error(
            "Proxy2Pw JIT: invalid shape"
        );
    }
}

void set_dynamic_smem_if_needed(
    const JitKernel& kernel,
    std::size_t shared_bytes,
    const char* label
) {
    if (shared_bytes <= 48 * 1024) {
        return;
    }

    CUresult res = cuFuncSetAttribute(
        kernel.function(),
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        static_cast<int>(shared_bytes)
    );

    if (res != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* msg = nullptr;

        cuGetErrorName(res, &name);
        cuGetErrorString(res, &msg);

        throw std::runtime_error(
            std::string(label) +
            ": cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) failed: " +
            (name ? name : "<unknown>") + ": " +
            (msg ? msg : "<no message>") +
            " shared_bytes=" + std::to_string(shared_bytes)
        );
    }
}

std::map<std::string, Proxy2PwLaunchConfig>& proxy2pw_config_cache() {
    static std::map<std::string, Proxy2PwLaunchConfig> cache;
    return cache;
}

std::mutex& proxy2pw_config_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

template <typename Real, int DIM>
std::string proxy2pw_tuning_key(
    const dmk::cuda::Proxy2PwArgs<Real>& args
) {
    std::ostringstream ss;
    ss << "Proxy2PwKernel"
       << "|real=" << jit_real_name<Real>()
       << "|dim=" << DIM
       << "|n_order=" << args.n_order
       << "|n_pw=" << args.n_pw
       << "|n_pw2=" << args.n_pw2
       << "|n_charge_dim=" << args.n_charge_dim
       << "|n_boxes=" << args.n_boxes_at_level;
    return ss.str();
}

template <typename Real, int DIM>
std::string proxy2pw_multilevel_tuning_key(
    const std::vector<dmk::cuda::Proxy2PwArgs<Real>>& args_h,
    int max_boxes
) {
    std::ostringstream ss;
    ss << "Proxy2PwMultiLevelKernel"
       << "|real=" << jit_real_name<Real>()
       << "|dim=" << DIM
       << "|n_args=" << args_h.size()
       << "|max_boxes=" << max_boxes;

    if (!args_h.empty()) {
        ss << "|n_order=" << args_h[0].n_order
           << "|n_pw=" << args_h[0].n_pw
           << "|n_pw2=" << args_h[0].n_pw2
           << "|n_charge_dim=" << args_h[0].n_charge_dim;
    }

    return ss.str();
}


} // namespace


std::string make_proxy2pw_source(const JitKey& key) {
    std::filesystem::path filename;

    if (key.name == "Proxy2PwKernel") {
        filename = "proxy2pw.cu";
    } else if (key.name == "Proxy2PwMultiLevelKernel") {
        filename = "proxy2pw_multilevel.cu";
    } else {
        throw std::runtime_error(
            "Proxy2Pw JIT: unknown kernel name: " + key.name
        );
    }

    const SplitSource split = load_split_jit_source(filename.string(), "Proxy2Pw");

    std::ostringstream generated;

    generated << make_specialization_constants(key);

    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real>
void launch_proxy2pw_jit_config(
    JitCache& cache,
    const dmk::cuda::Proxy2PwArgs<Real>& args,
    cudaStream_t stream,
    const Proxy2PwLaunchConfig& config
) {
    if (args.n_boxes_at_level == 0) {
        return;
    }

    check_proxy2pw_shape_or_throw(args);

    JitKey key;
    key.name = "Proxy2PwKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"BLOCK_SIZE", config.blocksize},
        {"N_ORDER", args.n_order},
        {"N_CHARGE_DIM", args.n_charge_dim},
        {"N_PW", args.n_pw},
        {"N_PW2", args.n_pw2},
        {"Z_TILE", config.z_tile},
        {"I_TILE", config.i_tile},
        {"M1_TILE", config.m1_tile},
        {"M2_TILE", config.m2_tile}
    };

    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes =
        proxy2pw_shared_bytes(
            args.n_order,
            args.n_pw,
            config.z_tile,
            sizeof(Real)
        );

    set_dynamic_smem_if_needed(
        *kernel,
        shared_bytes,
        "launch_proxy2pw_jit"
    );

    kernel->launch(
        dim3(args.n_boxes_at_level, 1, 1),
        dim3(config.blocksize, 1, 1),
        shared_bytes,
        stream,
        args
    );

}

template <typename Real>
void launch_proxy2pw_jit(
    JitCache& cache,
    const dmk::cuda::Proxy2PwArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
) {
    launch_proxy2pw_jit_config<Real>(
        cache,
        args,
        stream,
        default_proxy2pw_config(blocksize)
    );
}

template <typename Real>
void launch_proxy2pw_multilevel_jit_config(
    JitCache& cache,
    const std::vector<dmk::cuda::Proxy2PwArgs<Real>>& pa_h,
    dmk::cuda::Proxy2PwArgs<Real>* d_args_scratch,
    cudaStream_t stream,
    const Proxy2PwLaunchConfig& config
) {
    if (pa_h.empty()) {
        return;
    }

    int max_boxes = 0;
    int max_n_order = 0;
    int max_n_pw = 0;

    for (const auto& pa : pa_h) {
        max_boxes = std::max(max_boxes, pa.n_boxes_at_level);
        max_n_order = std::max(max_n_order, pa.n_order);
        max_n_pw = std::max(max_n_pw, pa.n_pw);
    }

    if (max_boxes == 0) {
        return;
    }

    DMK_CHECK_CUDA(
        cudaMemcpyAsync(
            d_args_scratch,
            pa_h.data(),
            pa_h.size() * sizeof(dmk::cuda::Proxy2PwArgs<Real>),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    JitKey key;
    key.name = "Proxy2PwMultiLevelKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"BLOCK_SIZE", config.blocksize},
        {"N_ORDER", pa_h[0].n_order},
        {"N_CHARGE_DIM", pa_h[0].n_charge_dim},
        {"N_PW", pa_h[0].n_pw},
        {"N_PW2", pa_h[0].n_pw2},
        {"Z_TILE", config.z_tile},
        {"I_TILE", config.i_tile},
        {"M1_TILE", config.m1_tile},
        {"M2_TILE", config.m2_tile}
    };

    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes =
        proxy2pw_shared_bytes(
            max_n_order,
            max_n_pw,
            config.z_tile,
            sizeof(Real)
        );

    set_dynamic_smem_if_needed(
        *kernel,
        shared_bytes,
        "launch_proxy2pw_multilevel_jit"
    );

    const int n_args = static_cast<int>(pa_h.size());

    kernel->launch(
        dim3(max_boxes, n_args, 1),
        dim3(config.blocksize, 1, 1),
        shared_bytes,
        stream,
        d_args_scratch,
        n_args
    );
}

template <typename Real>
void launch_proxy2pw_multilevel_jit(
    JitCache& cache,
    const std::vector<dmk::cuda::Proxy2PwArgs<Real>>& pa_h,
    dmk::cuda::Proxy2PwArgs<Real>* d_args_scratch,
    cudaStream_t stream,
    int blocksize
) {
    launch_proxy2pw_multilevel_jit_config<Real>(
        cache,
        pa_h,
        d_args_scratch,
        stream,
        default_proxy2pw_config(blocksize)
    );
}

template void launch_proxy2pw_jit<float>(
    JitCache&,
    const dmk::cuda::Proxy2PwArgs<float>&,
    cudaStream_t,
    int
);

template void launch_proxy2pw_jit<double>(
    JitCache&,
    const dmk::cuda::Proxy2PwArgs<double>&,
    cudaStream_t,
    int
);

template void launch_proxy2pw_multilevel_jit<float>(
    JitCache&,
    const std::vector<dmk::cuda::Proxy2PwArgs<float>>&,
    dmk::cuda::Proxy2PwArgs<float>*,
    cudaStream_t,
    int
);

template void launch_proxy2pw_multilevel_jit<double>(
    JitCache&,
    const std::vector<dmk::cuda::Proxy2PwArgs<double>>&,
    dmk::cuda::Proxy2PwArgs<double>*,
    cudaStream_t,
    int
);

template <typename Real, int DIM>
Proxy2PwLaunchConfig tune_proxy2pw_launch_config(
    JitCache& cache,
    const dmk::cuda::Proxy2PwArgs<Real>& args,
    cudaStream_t stream
) {
    const Proxy2PwLaunchConfig defaults = default_proxy2pw_config();
    if (env_flag_enabled("DMK_JIT_AUTOTUNE_DISABLE")) {
        return defaults;
    }

    const std::string tune_key = proxy2pw_tuning_key<Real, DIM>(args);
    const bool force = env_flag_enabled("DMK_JIT_AUTOTUNE_FORCE");
    int device = 0;
    check_cuda(cudaGetDevice(&device), "Proxy2Pw tune cudaGetDevice");
    const std::string in_process_key =
        tune_key + "|device=" + std::to_string(device);

    std::unique_lock<std::mutex> config_lock(proxy2pw_config_cache_mutex());
    const auto it = proxy2pw_config_cache().find(in_process_key);
    if (it != proxy2pw_config_cache().end()) {
        return it->second;
    }

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "Proxy2Pw tune cudaGetDeviceProperties");
    const std::size_t max_shared_bytes =
        prop.sharedMemPerBlockOptin > 0
            ? static_cast<std::size_t>(prop.sharedMemPerBlockOptin)
            : static_cast<std::size_t>(prop.sharedMemPerBlock);

    GridTuneOptions options;
    options.kernel = "Proxy2PwKernel";
    options.key = tune_key;
    options.force = force;
    options.benchmark = CudaBenchmarkOptions{
        2, // warmup
        5, // repeats
    };

    const std::vector<TuningParameter> space{
        {"BLOCK_SIZE", {128, 256}},
        {"Z_TILE", {2, 4}},
        {"I_TILE", {2, 4}},
        {"M1_TILE", {2, 4, 6}},
        {"M2_TILE", {2, 4}},
    };

    const auto constraint = [&](const TuningParams& params) {
        const Proxy2PwLaunchConfig config = proxy2pw_config_from_params(params);
        if (config.blocksize <= 0 || config.blocksize > prop.maxThreadsPerBlock ||
            config.blocksize % 32 != 0) {
            return false;
        }
        if (config.z_tile <= 0 || config.z_tile > args.n_pw2 ||
            config.i_tile <= 0 || config.i_tile > args.n_order ||
            config.m1_tile <= 0 || config.m1_tile > args.n_pw ||
            config.m2_tile <= 0 || config.m2_tile > args.n_pw) {
            return false;
        }
        if (config.i_tile * config.m2_tile > 24 ||
            config.m1_tile * config.m2_tile > 24) {
            return false;
        }

        const std::size_t shared_bytes =
            proxy2pw_shared_bytes(
                args.n_order,
                args.n_pw,
                config.z_tile,
                sizeof(Real)
            );
        return shared_bytes <= max_shared_bytes;
    };

    const auto benchmark = [&](const TuningParams& params) {
        const Proxy2PwLaunchConfig config = proxy2pw_config_from_params(params);
        return benchmark_cuda_ms(stream, options.benchmark, [&](cudaStream_t bench_stream) {
            launch_proxy2pw_jit_config<Real>(
                cache,
                args,
                bench_stream,
                config
            );
        });
    };

    const GridTuneDecision decision =
        tune_grid(
            options,
            space,
            proxy2pw_tuning_params(defaults),
            constraint,
            benchmark
        );

    const Proxy2PwLaunchConfig tuned_config =
        proxy2pw_config_from_params(decision.params);
    proxy2pw_config_cache()[in_process_key] = tuned_config;
    return tuned_config;
}

template <typename Real, int DIM>
void launch_proxy2pw_autotuned(
    JitCache& cache,
    const dmk::cuda::Proxy2PwArgs<Real>& args,
    cudaStream_t stream
) {
    Proxy2PwLaunchConfig config = default_proxy2pw_config();

    try {
        config = tune_proxy2pw_launch_config<Real, DIM>(
            cache,
            args,
            stream
        );
    } catch (const std::exception& e) {
        if (env_flag_enabled("DMK_JIT_AUTOTUNE_VERBOSE")) {
            std::cerr << "[dmk jit autotune] Proxy2PwKernel tuning failed; "
                      << "using defaults: " << e.what() << "\n";
        }
    }

    launch_proxy2pw_jit_config<Real>(
        cache,
        args,
        stream,
        config
    );
}

template <typename Real, int DIM>
Proxy2PwLaunchConfig tune_proxy2pw_multilevel_launch_config(
    JitCache& cache,
    const std::vector<dmk::cuda::Proxy2PwArgs<Real>>& pa_h,
    dmk::cuda::Proxy2PwArgs<Real>* d_args_scratch,
    cudaStream_t stream
) {
    const Proxy2PwLaunchConfig defaults = default_proxy2pw_config();
    if (env_flag_enabled("DMK_JIT_AUTOTUNE_DISABLE")) {
        return defaults;
    }

    int max_boxes = 0;
    int max_n_order = 0;
    int max_n_pw = 0;
    int max_n_pw2 = 0;

    for (const auto& pa : pa_h) {
        max_boxes = std::max(max_boxes, pa.n_boxes_at_level);
        max_n_order = std::max(max_n_order, pa.n_order);
        max_n_pw = std::max(max_n_pw, pa.n_pw);
        max_n_pw2 = std::max(max_n_pw2, pa.n_pw2);
    }

    if (max_boxes == 0 || pa_h.empty()) {
        return defaults;
    }

    const std::string tune_key =
        proxy2pw_multilevel_tuning_key<Real, DIM>(pa_h, max_boxes);
    const bool force = env_flag_enabled("DMK_JIT_AUTOTUNE_FORCE");
    int device = 0;
    check_cuda(cudaGetDevice(&device), "Proxy2Pw multilevel tune cudaGetDevice");
    const std::string in_process_key =
        tune_key + "|device=" + std::to_string(device);

    std::unique_lock<std::mutex> config_lock(proxy2pw_config_cache_mutex());
    const auto it = proxy2pw_config_cache().find(in_process_key);
    if (it != proxy2pw_config_cache().end()) {
        return it->second;
    }

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "Proxy2Pw multilevel tune cudaGetDeviceProperties");
    const std::size_t max_shared_bytes =
        prop.sharedMemPerBlockOptin > 0
            ? static_cast<std::size_t>(prop.sharedMemPerBlockOptin)
            : static_cast<std::size_t>(prop.sharedMemPerBlock);

    GridTuneOptions options;
    options.kernel = "Proxy2PwMultiLevelKernel";
    options.key = tune_key;
    options.force = force;
    options.benchmark = CudaBenchmarkOptions{
        2, // warmup
        5, // repeats
    };

    const std::vector<TuningParameter> space{
        {"BLOCK_SIZE", {128, 256}},
        {"Z_TILE", {2, 4}},
        {"I_TILE", {2, 4}},
        {"M1_TILE", {2, 4, 6}},
        {"M2_TILE", {2, 4}},
    };

    const auto constraint = [&](const TuningParams& params) {
        const Proxy2PwLaunchConfig config = proxy2pw_config_from_params(params);
        if (config.blocksize <= 0 || config.blocksize > prop.maxThreadsPerBlock ||
            config.blocksize % 32 != 0) {
            return false;
        }
        if (config.z_tile <= 0 || config.z_tile > max_n_pw2 ||
            config.i_tile <= 0 || config.i_tile > max_n_order ||
            config.m1_tile <= 0 || config.m1_tile > max_n_pw ||
            config.m2_tile <= 0 || config.m2_tile > max_n_pw) {
            return false;
        }
        if (config.i_tile * config.m2_tile > 24 ||
            config.m1_tile * config.m2_tile > 24) {
            return false;
        }

        const std::size_t shared_bytes =
            proxy2pw_shared_bytes(
                max_n_order,
                max_n_pw,
                config.z_tile,
                sizeof(Real)
            );
        return shared_bytes <= max_shared_bytes;
    };

    const auto benchmark = [&](const TuningParams& params) {
        const Proxy2PwLaunchConfig config = proxy2pw_config_from_params(params);
        return benchmark_cuda_ms(stream, options.benchmark, [&](cudaStream_t bench_stream) {
            launch_proxy2pw_multilevel_jit_config<Real>(
                cache,
                pa_h,
                d_args_scratch,
                bench_stream,
                config
            );
        });
    };

    const GridTuneDecision decision =
        tune_grid(
            options,
            space,
            proxy2pw_tuning_params(defaults),
            constraint,
            benchmark
        );

    const Proxy2PwLaunchConfig tuned_config =
        proxy2pw_config_from_params(decision.params);
    proxy2pw_config_cache()[in_process_key] = tuned_config;
    return tuned_config;
}

template <typename Real, int DIM>
void launch_proxy2pw_multilevel_autotuned(
    JitCache& cache,
    const std::vector<dmk::cuda::Proxy2PwArgs<Real>>& pa_h,
    dmk::cuda::Proxy2PwArgs<Real>* d_args_scratch,
    cudaStream_t stream
) {
    Proxy2PwLaunchConfig config = default_proxy2pw_config();

    try {
        config = tune_proxy2pw_multilevel_launch_config<Real, DIM>(
            cache,
            pa_h,
            d_args_scratch,
            stream
        );
    } catch (const std::exception& e) {
        if (env_flag_enabled("DMK_JIT_AUTOTUNE_VERBOSE")) {
            std::cerr << "[dmk jit autotune] Proxy2PwMultiLevelKernel tuning failed; "
                      << "using defaults: " << e.what() << "\n";
        }
    }

    launch_proxy2pw_multilevel_jit_config<Real>(
        cache,
        pa_h,
        d_args_scratch,
        stream,
        config
    );
}

} // namespace dmk::cuda::jit

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_proxy2pw(const Proxy2PwArgs<Real>& args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_proxy2pw_autotuned<Real, DIM>(
        jit_cache,
        args,
        stream
    );
}

template <typename Real, int DIM>
void launch_proxy2pw_multilevel(
    const std::vector<Proxy2PwArgs<Real>>& pa_h,
    Proxy2PwArgs<Real>* d_args_scratch,
    cudaStream_t stream
) {
    if (pa_h.empty())
        return;

    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_proxy2pw_multilevel_autotuned<Real, DIM>(
        jit_cache,
        pa_h,
        d_args_scratch,
        stream
    );
}

template void launch_proxy2pw<float, 2>(const Proxy2PwArgs<float>&, cudaStream_t);
template void launch_proxy2pw<float, 3>(const Proxy2PwArgs<float>&, cudaStream_t);
template void launch_proxy2pw<double, 2>(const Proxy2PwArgs<double>&, cudaStream_t);
template void launch_proxy2pw<double, 3>(const Proxy2PwArgs<double>&, cudaStream_t);

template void launch_proxy2pw_multilevel<float, 2>(const std::vector<Proxy2PwArgs<float>>&, Proxy2PwArgs<float>*,
                                                   cudaStream_t);
template void launch_proxy2pw_multilevel<float, 3>(const std::vector<Proxy2PwArgs<float>>&, Proxy2PwArgs<float>*,
                                                   cudaStream_t);
template void launch_proxy2pw_multilevel<double, 2>(const std::vector<Proxy2PwArgs<double>>&, Proxy2PwArgs<double>*,
                                                    cudaStream_t);
template void launch_proxy2pw_multilevel<double, 3>(const std::vector<Proxy2PwArgs<double>>&, Proxy2PwArgs<double>*,
                                                    cudaStream_t);

} // namespace dmk::cuda
