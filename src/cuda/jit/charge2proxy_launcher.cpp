#include "autotune.hpp"
#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"
#include <dmk/cuda/charge2proxy_kernels.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace dmk::cuda::jit {

namespace {

constexpr int CHARGE2PROXY_CHUNK = 128;
constexpr int CHARGE2PROXY_I_TILE = 3;
constexpr int CHARGE2PROXY_J_TILE = 3;
constexpr int CHARGE2PROXY_K_TILE = 4;
constexpr int CHARGE2PROXY_BLOCK_SIZE = 128;

struct Charge2ProxyLaunchConfig {
    int chunk = CHARGE2PROXY_CHUNK;
    int i_tile = CHARGE2PROXY_I_TILE;
    int j_tile = CHARGE2PROXY_J_TILE;
    int k_tile = CHARGE2PROXY_K_TILE;
    int blocksize = CHARGE2PROXY_BLOCK_SIZE;
};

Charge2ProxyLaunchConfig default_charge2proxy_config() {
    return Charge2ProxyLaunchConfig{};
}

int tuning_param_or(const TuningParams& params, const char* name, int fallback) {
    const auto it = params.find(name);
    return it == params.end() ? fallback : it->second;
}

TuningParams charge2proxy_tuning_params(const Charge2ProxyLaunchConfig& config) {
    return TuningParams{
        {"CHUNK", config.chunk},
        {"I_TILE", config.i_tile},
        {"J_TILE", config.j_tile},
        {"K_TILE", config.k_tile},
        {"BLOCK_SIZE", config.blocksize},
    };
}

Charge2ProxyLaunchConfig charge2proxy_config_from_params(const TuningParams& params) {
    const Charge2ProxyLaunchConfig defaults = default_charge2proxy_config();
    return Charge2ProxyLaunchConfig{
        tuning_param_or(params, "CHUNK", defaults.chunk),
        tuning_param_or(params, "I_TILE", defaults.i_tile),
        tuning_param_or(params, "J_TILE", defaults.j_tile),
        tuning_param_or(params, "K_TILE", defaults.k_tile),
        tuning_param_or(params, "BLOCK_SIZE", defaults.blocksize),
    };
}

std::size_t charge2proxy_coeff_count(int n_order, int n_charge_dim) {
    const std::size_t n = static_cast<std::size_t>(n_order);
    return n * n * n * static_cast<std::size_t>(n_charge_dim);
}

std::string make_specialization_constants(const JitKey& key) {
    const int n_order      = required_int_param(key, "N_ORDER", "Charge2Proxy");
    const int n_charge_dim = required_int_param(key, "N_CHARGE_DIM", "Charge2Proxy");
    const int chunk        = required_int_param(key, "CHUNK", "Charge2Proxy");
    const int i_tile       = required_int_param(key, "I_TILE", "Charge2Proxy");
    const int j_tile       = required_int_param(key, "J_TILE", "Charge2Proxy");
    const int k_tile       = required_int_param(key, "K_TILE", "Charge2Proxy");
    const std::string real_type = key.real;
    std::ostringstream ss;
    ss << "#include <dmk/cuda/charge2proxy_kernelargs.hpp>\n"; 
    ss << "using dmk::cuda::Charge2ProxyArgs;\n\n";
    ss << "constexpr int N_ORDER      = " << n_order << ";\n";
    ss << "constexpr int N_CHARGE_DIM = " << n_charge_dim << ";\n";
    ss << "constexpr int CHUNK        = " << chunk << ";\n";
    ss << "constexpr int I_TILE       = " << i_tile << ";\n";
    ss << "constexpr int J_TILE       = " << j_tile << ";\n";
    ss << "constexpr int K_TILE       = " << k_tile << ";\n";
    ss << "using Real = " << real_type << "; \n\n";
    return ss.str();

}


std::size_t charge2proxy_shared_bytes(int n_order, int n_charge_dim, int chunk, std::size_t sizeof_real) {
    const int ld = chunk + 2;

    return (std::size_t{3} * std::size_t(n_order) * std::size_t(ld) + std::size_t(n_charge_dim) * std::size_t(ld)) * sizeof_real;
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

template <typename Real>
AutotuneDeviceRangeSnapshots<Real> make_charge2proxy_output_snapshots(
    const dmk::cuda::Charge2ProxyArgs<Real>& args,
    const int* group_perm,
    int n_launch_groups,
    cudaStream_t stream
) {
    std::vector<int> groups(static_cast<std::size_t>(n_launch_groups));
    if (group_perm != nullptr) {
        check_cuda(
            cudaMemcpyAsync(
                groups.data(),
                group_perm,
                groups.size() * sizeof(int),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "Charge2Proxy copy group perm"
        );
        check_cuda(cudaStreamSynchronize(stream), "Charge2Proxy sync group perm");
    } else {
        for (int i = 0; i < n_launch_groups; ++i) {
            groups[static_cast<std::size_t>(i)] = i;
        }
    }

    std::vector<int> center_boxes(groups.size());
    for (std::size_t i = 0; i < groups.size(); ++i) {
        check_cuda(
            cudaMemcpyAsync(
                &center_boxes[i],
                args.center_boxes + groups[i],
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "Charge2Proxy copy center box"
        );
    }
    check_cuda(cudaStreamSynchronize(stream), "Charge2Proxy sync center boxes");

    std::sort(center_boxes.begin(), center_boxes.end());
    center_boxes.erase(std::unique(center_boxes.begin(), center_boxes.end()), center_boxes.end());

    std::vector<std::pair<Real*, long>> ranges;
    ranges.reserve(center_boxes.size());
    for (int box : center_boxes) {
        long offset = -1;
        check_cuda(
            cudaMemcpyAsync(
                &offset,
                args.proxy_offsets + box,
                sizeof(long),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "Charge2Proxy copy proxy offset"
        );
        check_cuda(cudaStreamSynchronize(stream), "Charge2Proxy sync proxy offset");
        ranges.emplace_back(args.proxy_flat, offset);
    }

    return make_device_range_snapshots<Real>(
        std::move(ranges),
        charge2proxy_coeff_count(args.n_order, args.n_charge_dim),
        stream
    );
}

std::map<std::string, Charge2ProxyLaunchConfig>& charge2proxy_config_cache() {
    static std::map<std::string, Charge2ProxyLaunchConfig> cache;
    return cache;
}

std::mutex& charge2proxy_config_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

template <typename Real>
std::string charge2proxy_tuning_key(
    const dmk::cuda::Charge2ProxyArgs<Real>& args,
    int n_launch_groups
) {
    std::ostringstream ss;
    ss << "Charge2ProxyKernel"
       << "|real=" << jit_real_name<Real>()
       << "|dim=3"
       << "|n_order=" << args.n_order
       << "|n_charge_dim=" << args.n_charge_dim
       << "|n_groups=" << args.n_groups
       << "|n_launch_groups=" << n_launch_groups;
    return ss.str();
}

} // namespace


std::string make_charge2proxy_source(const JitKey& key) {
    const SplitSource split = load_split_jit_source("charge2proxy.cu", "Charge2Proxy");
    std::ostringstream generated;
    generated << split.header << "\n";

    generated << make_specialization_constants(key);

    generated << split.kernel << "\n";


    return generated.str();
}

template <typename Real>
void launch_charge2proxy_jit(
    JitCache& cache,
    const dmk::cuda::Charge2ProxyArgs<Real>& args,
    const int* group_perm,
    int n_launch_groups,
    cudaStream_t stream,
    int chunk,
    int i_tile,
    int j_tile,
    int k_tile,
    int blocksize
) {
    if (args.n_groups == 0 || n_launch_groups == 0) {
        return;
    }

    JitKey key;
    key.name = "Charge2ProxyKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"N_ORDER", args.n_order},
        {"N_CHARGE_DIM", args.n_charge_dim},
        {"CHUNK", chunk},
        {"I_TILE", i_tile},
        {"J_TILE", j_tile},
        {"K_TILE", k_tile},
        {"BLOCK_SIZE", blocksize},
    };
    
    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes = charge2proxy_shared_bytes(args.n_order, args.n_charge_dim, chunk, sizeof(Real));

    set_dynamic_smem_if_needed(
        *kernel,
        shared_bytes,
        "launch_charge2proxy_jit"
    );
    
    kernel->launch(
        dim3(n_launch_groups, 1, 1),
        dim3(blocksize, 1, 1),
        shared_bytes,
        stream,
        args,
        group_perm
    );
}

template <typename Real>
Charge2ProxyLaunchConfig tune_charge2proxy_launch_config(
    JitCache& cache,
    const dmk::cuda::Charge2ProxyArgs<Real>& args,
    const int* group_perm,
    int n_launch_groups,
    cudaStream_t stream
) {
    const Charge2ProxyLaunchConfig defaults = default_charge2proxy_config();
    if (env_flag_enabled("DMK_JIT_AUTOTUNE_DISABLE")) {
        return defaults;
    }

    const std::string tune_key = charge2proxy_tuning_key(args, n_launch_groups);
    const bool force = env_flag_enabled("DMK_JIT_AUTOTUNE_FORCE");
    int device = 0;
    check_cuda(cudaGetDevice(&device), "Charge2Proxy tune cudaGetDevice");
    const std::string in_process_key =
        tune_key + "|device=" + std::to_string(device);

    std::unique_lock<std::mutex> config_lock(charge2proxy_config_cache_mutex());
    const auto it = charge2proxy_config_cache().find(in_process_key);
    if (it != charge2proxy_config_cache().end()) {
        return it->second;
    }

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "Charge2Proxy tune cudaGetDeviceProperties");
    const std::size_t max_shared_bytes =
        prop.sharedMemPerBlockOptin > 0
            ? static_cast<std::size_t>(prop.sharedMemPerBlockOptin)
            : static_cast<std::size_t>(prop.sharedMemPerBlock);

    std::optional<AutotuneDeviceRangeSnapshots<Real>> snapshots;

    GridTuneOptions options;
    options.kernel = "Charge2ProxyKernel";
    options.key = tune_key;
    options.force = force;
    options.benchmark = CudaBenchmarkOptions{
        2, // warmup
        5, // repeats
    };

    const std::vector<TuningParameter> space{
        {"CHUNK", {64, 128}},
        {"I_TILE", {2, 3, 4}},
        {"J_TILE", {2, 3, 4}},
        {"K_TILE", {2, 4}},
        {"BLOCK_SIZE", {128, 256}},
    };

    const auto constraint = [&](const TuningParams& params) {
        const Charge2ProxyLaunchConfig config = charge2proxy_config_from_params(params);
        if (config.blocksize <= 0 || config.blocksize > prop.maxThreadsPerBlock ||
            config.blocksize % 32 != 0) {
            return false;
        }
        if (config.chunk <= 0 ||
            config.i_tile <= 0 || config.i_tile > args.n_order ||
            config.j_tile <= 0 || config.j_tile > args.n_order ||
            config.k_tile <= 0 || config.k_tile > args.n_order) {
            return false;
        }
        if (config.i_tile * config.j_tile * config.k_tile > 48) {
            return false;
        }

        const std::size_t shared_bytes =
            charge2proxy_shared_bytes(
                args.n_order,
                args.n_charge_dim,
                config.chunk,
                sizeof(Real)
            );
        return shared_bytes <= max_shared_bytes;
    };

    const auto benchmark = [&](const TuningParams& params) {
        const Charge2ProxyLaunchConfig config = charge2proxy_config_from_params(params);
        if (!snapshots) {
            snapshots.emplace(
                make_charge2proxy_output_snapshots(
                    args,
                    group_perm,
                    n_launch_groups,
                    stream
                )
            );
        }

        restore_device_range_snapshots(*snapshots, stream);
        try {
            const double runtime_ms =
                benchmark_cuda_ms(stream, options.benchmark, [&](cudaStream_t bench_stream) {
                    launch_charge2proxy_jit<Real>(
                        cache,
                        args,
                        group_perm,
                        n_launch_groups,
                        bench_stream,
                        config.chunk,
                        config.i_tile,
                        config.j_tile,
                        config.k_tile,
                        config.blocksize
                    );
                });
            restore_device_range_snapshots(*snapshots, stream);
            return runtime_ms;
        } catch (...) {
            restore_device_range_snapshots(*snapshots, stream);
            throw;
        }
    };

    const GridTuneDecision decision =
        tune_grid(
            options,
            space,
            charge2proxy_tuning_params(defaults),
            constraint,
            benchmark
        );

    if (snapshots) {
        restore_device_range_snapshots(*snapshots, stream);
    }

    const Charge2ProxyLaunchConfig tuned_config =
        charge2proxy_config_from_params(decision.params);
    charge2proxy_config_cache()[in_process_key] = tuned_config;
    return tuned_config;
}

template <typename Real>
void launch_charge2proxy_autotuned(
    JitCache& cache,
    const dmk::cuda::Charge2ProxyArgs<Real>& args,
    const int* group_perm,
    int n_launch_groups,
    cudaStream_t stream
) {
    Charge2ProxyLaunchConfig config = default_charge2proxy_config();

    try {
        config = tune_charge2proxy_launch_config<Real>(
            cache,
            args,
            group_perm,
            n_launch_groups,
            stream
        );
    } catch (const std::exception& e) {
        if (env_flag_enabled("DMK_JIT_AUTOTUNE_VERBOSE")) {
            std::cerr << "[dmk jit autotune] Charge2ProxyKernel tuning failed; "
                      << "using defaults: " << e.what() << "\n";
        }
    }

    launch_charge2proxy_jit<Real>(
        cache,
        args,
        group_perm,
        n_launch_groups,
        stream,
        config.chunk,
        config.i_tile,
        config.j_tile,
        config.k_tile,
        config.blocksize
    );
}

template void launch_charge2proxy_jit<float>(
    JitCache&,
    const dmk::cuda::Charge2ProxyArgs<float>&,
    const int*,
    int,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int
);

template void launch_charge2proxy_jit<double>(
    JitCache&,
    const dmk::cuda::Charge2ProxyArgs<double>&,
    const int*,
    int,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int
);

} // namespace dmk::cuda::jit

namespace dmk::cuda {
namespace {

template <typename Real>
void launch_charge2proxy_3d_impl(const Charge2ProxyArgs<Real> &args, const int *group_perm, int n_launch_groups,
                                 cudaStream_t stream) {
    if (args.n_groups == 0 || n_launch_groups == 0)
        return;

    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_charge2proxy_autotuned<Real>(
        jit_cache,
        args,
        group_perm,
        n_launch_groups,
        stream
    );
}

} // namespace

template <typename Real, int DIM>
void launch_charge2proxy(const Charge2ProxyArgs<Real> &args, cudaStream_t stream) {
    if constexpr (DIM != 3) {
        throw std::runtime_error("CUDA charge2proxy: only DIM=3 supported for now");
    } else {
        launch_charge2proxy_3d_impl<Real>(args, args.group_perm, args.n_active_groups, stream);
    }
}

template void launch_charge2proxy<float, 2>(const Charge2ProxyArgs<float> &, cudaStream_t);
template void launch_charge2proxy<float, 3>(const Charge2ProxyArgs<float> &, cudaStream_t);
template void launch_charge2proxy<double, 2>(const Charge2ProxyArgs<double> &, cudaStream_t);
template void launch_charge2proxy<double, 3>(const Charge2ProxyArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
