#include "shift_pw_launcher.hpp"

#include "autotune.hpp"
#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/shift_pw_kernels.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda::jit {
namespace {

struct ShiftPwLaunchConfig {
    int blocksize = 0;
    int neighbor_unroll = 0; // 0 means CUDA's default/full unroll.
};

constexpr int kDefaultNeighborUnroll = 0;

int tuning_param_or(const TuningParams &params, const char *name, int fallback) {
    const auto it = params.find(name);
    return it == params.end() ? fallback : it->second;
}

TuningParams shift_pw_tuning_params(const ShiftPwLaunchConfig &config) {
    return TuningParams{
        {"BLOCK_SIZE", config.blocksize},
        {"NEIGHBOR_UNROLL", config.neighbor_unroll},
    };
}

ShiftPwLaunchConfig shift_pw_config_from_params(const TuningParams &params, const ShiftPwLaunchConfig &fallback) {
    return ShiftPwLaunchConfig{
        tuning_param_or(params, "BLOCK_SIZE", fallback.blocksize),
        tuning_param_or(params, "NEIGHBOR_UNROLL", fallback.neighbor_unroll),
    };
}

std::string make_specialization_constants(const JitKey &key) {
    const int blocksize = required_int_param(key, "BLOCK_SIZE", "ShiftPw");
    const int n_pw_modes = required_int_param(key, "N_PW_MODES", "ShiftPw");
    const int n_charge_dim = required_int_param(key, "N_CHARGE_DIM", "ShiftPw");
    const int n_neighbors = required_int_param(key, "N_NEIGHBORS", "ShiftPw");
    const int neighbor_unroll = required_int_param(key, "NEIGHBOR_UNROLL", "ShiftPw");
    std::ostringstream ss;

    ss << "#include <dmk/cuda/shift_pw_kernelargs.hpp>\n";

    ss << "using dmk::cuda::ShiftPwArgs;\n";
    ss << "#define SHIFT_PW_NEIGHBOR_UNROLL " << neighbor_unroll << "\n";

    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n";
    ss << "constexpr int N_PW_MODES   = " << n_pw_modes << ";\n";
    ss << "constexpr int N_CHARGE_DIM = " << n_charge_dim << ";\n";
    ss << "constexpr int N_NEIGHBORS  = " << n_neighbors << ";\n";
    ss << "using Real = " << key.real << ";\n\n";

    return ss.str();
}

} // namespace

namespace {

std::map<std::string, ShiftPwLaunchConfig> &shift_pw_config_cache() {
    static std::map<std::string, ShiftPwLaunchConfig> cache;
    return cache;
}

std::mutex &shift_pw_config_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

template <typename Real, int DIM>
std::string shift_pw_tuning_key(const dmk::cuda::ShiftPwArgs<Real> &args) {
    std::ostringstream ss;
    ss << "ShiftPwByBoxKernel"
       << "|real=" << jit_real_name<Real>() << "|dim=" << DIM << "|n_boxes=" << args.n_boxes_at_level
       << "|n_neighbors=" << args.n_neighbors << "|n_pw_modes=" << args.n_pw_modes
       << "|n_charge_dim=" << args.n_charge_dim;
    return ss.str();
}

template <typename Real, int DIM>
std::string shift_pw_multilevel_tuning_key(const std::vector<dmk::cuda::ShiftPwArgs<Real>> &args_h, int max_boxes) {
    std::ostringstream ss;
    ss << "ShiftPwKernel"
       << "|real=" << jit_real_name<Real>() << "|dim=" << DIM << "|n_args=" << args_h.size()
       << "|max_boxes=" << max_boxes;

    if (!args_h.empty()) {
        ss << "|n_neighbors=" << args_h[0].n_neighbors << "|n_pw_modes=" << args_h[0].n_pw_modes
           << "|n_charge_dim=" << args_h[0].n_charge_dim;
    }

    return ss.str();
}

} // namespace

std::string make_shift_pw_source(const JitKey &key) {
    const SplitSource split = load_split_jit_source("shiftpw.cu", "ShiftPw");

    std::ostringstream generated;

    generated << make_specialization_constants(key);
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real>
void launch_shift_pw_jit(JitCache &cache, const dmk::cuda::ShiftPwArgs<Real> &args, cudaStream_t stream, int blocksize,
                         int neighbor_unroll) {
    if (args.n_boxes_at_level == 0) {
        return;
    }

    JitKey key;
    key.name = "ShiftPwByBoxKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"BLOCK_SIZE", blocksize},         {"N_CHARGE_DIM", args.n_charge_dim},
        {"N_NEIGHBORS", args.n_neighbors}, {"NEIGHBOR_UNROLL", neighbor_unroll},
        {"N_PW_MODES", args.n_pw_modes},
    };

    auto kernel = cache.get_kernel(key);

    kernel->launch(dim3(args.n_boxes_at_level, 1, 1), dim3(blocksize, 1, 1), 0, stream, args);
}

template <typename Real>
void launch_shift_pw_multilevel_jit(JitCache &cache, const std::vector<dmk::cuda::ShiftPwArgs<Real>> &args_h,
                                    dmk::cuda::ShiftPwArgs<Real> *d_args_scratch, cudaStream_t stream, int blocksize,
                                    int neighbor_unroll) {
    if (args_h.empty()) {
        return;
    }

    int max_boxes = 0;

    for (const auto &a : args_h) {
        if (a.n_boxes_at_level == 0) {
            continue;
        }

        max_boxes = std::max(max_boxes, a.n_boxes_at_level);
    }

    if (max_boxes == 0) {
        return;
    }

    DMK_CHECK_CUDA(cudaMemcpyAsync(d_args_scratch, args_h.data(), args_h.size() * sizeof(dmk::cuda::ShiftPwArgs<Real>),
                                   cudaMemcpyHostToDevice, stream));

    JitKey key;
    key.name = "ShiftPwKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"BLOCK_SIZE", blocksize},
        {"N_CHARGE_DIM", args_h[0].n_charge_dim},
        {"N_NEIGHBORS", args_h[0].n_neighbors},
        {"NEIGHBOR_UNROLL", neighbor_unroll},
        {"N_PW_MODES", args_h[0].n_pw_modes},
    };

    auto kernel = cache.get_kernel(key);

    const int n_args = static_cast<int>(args_h.size());

    kernel->launch(dim3(max_boxes, n_args, 1), dim3(blocksize, 1, 1), 0, stream, d_args_scratch, n_args);
}

template void launch_shift_pw_jit<float>(JitCache &, const dmk::cuda::ShiftPwArgs<float> &, cudaStream_t, int, int);

template void launch_shift_pw_jit<double>(JitCache &, const dmk::cuda::ShiftPwArgs<double> &, cudaStream_t, int, int);

template void launch_shift_pw_multilevel_jit<float>(JitCache &, const std::vector<dmk::cuda::ShiftPwArgs<float>> &,
                                                    dmk::cuda::ShiftPwArgs<float> *, cudaStream_t, int, int);

template void launch_shift_pw_multilevel_jit<double>(JitCache &, const std::vector<dmk::cuda::ShiftPwArgs<double>> &,
                                                     dmk::cuda::ShiftPwArgs<double> *, cudaStream_t, int, int);

template <typename Real, int DIM>
ShiftPwLaunchConfig tune_shift_pw_config(JitCache &cache, const dmk::cuda::ShiftPwArgs<Real> &args, cudaStream_t stream,
                                         ShiftPwLaunchConfig defaults) {
    if (env_flag_enabled("DMK_JIT_AUTOTUNE_DISABLE")) {
        return defaults;
    }

    const std::string tune_key = shift_pw_tuning_key<Real, DIM>(args);
    const bool force = env_flag_enabled("DMK_JIT_AUTOTUNE_FORCE");
    int device = 0;
    check_cuda(cudaGetDevice(&device), "ShiftPw tune cudaGetDevice");
    const std::string in_process_key = tune_key + "|device=" + std::to_string(device);

    std::unique_lock<std::mutex> config_lock(shift_pw_config_cache_mutex());
    const auto it = shift_pw_config_cache().find(in_process_key);
    if (it != shift_pw_config_cache().end()) {
        return it->second;
    }

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "ShiftPw tune cudaGetDeviceProperties");

    GridTuneOptions options;
    options.kernel = "ShiftPwByBoxKernel";
    options.key = tune_key;
    options.force = force;
    options.benchmark = CudaBenchmarkOptions{
        2, // warmup
        5, // repeats
    };

    const std::vector<TuningParameter> space{
        {"BLOCK_SIZE", {64, 128, 256, 512, 768}},
        {"NEIGHBOR_UNROLL", {1, 2, 3, 4, 6, 9, 0}},
    };

    const auto constraint = [&](const TuningParams &params) {
        const ShiftPwLaunchConfig config = shift_pw_config_from_params(params, defaults);
        return config.blocksize > 0 && config.blocksize <= prop.maxThreadsPerBlock && config.blocksize % 32 == 0 &&
               (config.neighbor_unroll == 0 ||
                (config.neighbor_unroll > 0 && config.neighbor_unroll <= args.n_neighbors));
    };

    const auto benchmark = [&](const TuningParams &params) {
        const ShiftPwLaunchConfig config = shift_pw_config_from_params(params, defaults);
        return benchmark_cuda_ms(stream, options.benchmark, [&](cudaStream_t bench_stream) {
            launch_shift_pw_jit<Real>(cache, args, bench_stream, config.blocksize, config.neighbor_unroll);
        });
    };

    const GridTuneDecision decision =
        tune_grid(options, space, shift_pw_tuning_params(defaults), constraint, benchmark);

    const ShiftPwLaunchConfig tuned_config = shift_pw_config_from_params(decision.params, defaults);
    shift_pw_config_cache()[in_process_key] = tuned_config;
    return tuned_config;
}

template <typename Real, int DIM>
void launch_shift_pw_autotuned(JitCache &cache, const dmk::cuda::ShiftPwArgs<Real> &args, cudaStream_t stream,
                               int default_blocksize) {
    ShiftPwLaunchConfig config{default_blocksize, kDefaultNeighborUnroll};

    try {
        config = tune_shift_pw_config<Real, DIM>(cache, args, stream, config);
    } catch (const std::exception &e) {
        if (env_flag_enabled("DMK_JIT_AUTOTUNE_VERBOSE")) {
            std::cerr << "[dmk jit autotune] ShiftPwByBoxKernel tuning failed; "
                      << "using defaults: " << e.what() << "\n";
        }
    }

    launch_shift_pw_jit<Real>(cache, args, stream, config.blocksize, config.neighbor_unroll);
}

template <typename Real, int DIM>
ShiftPwLaunchConfig tune_shift_pw_multilevel_config(JitCache &cache,
                                                    const std::vector<dmk::cuda::ShiftPwArgs<Real>> &args_h,
                                                    dmk::cuda::ShiftPwArgs<Real> *d_args_scratch, cudaStream_t stream,
                                                    ShiftPwLaunchConfig defaults) {
    if (env_flag_enabled("DMK_JIT_AUTOTUNE_DISABLE")) {
        return defaults;
    }

    int max_boxes = 0;
    for (const auto &args : args_h) {
        max_boxes = std::max(max_boxes, args.n_boxes_at_level);
    }
    if (max_boxes == 0 || args_h.empty()) {
        return defaults;
    }

    const std::string tune_key = shift_pw_multilevel_tuning_key<Real, DIM>(args_h, max_boxes);
    const bool force = env_flag_enabled("DMK_JIT_AUTOTUNE_FORCE");
    int device = 0;
    check_cuda(cudaGetDevice(&device), "ShiftPw multilevel tune cudaGetDevice");
    const std::string in_process_key = tune_key + "|device=" + std::to_string(device);

    std::unique_lock<std::mutex> config_lock(shift_pw_config_cache_mutex());
    const auto it = shift_pw_config_cache().find(in_process_key);
    if (it != shift_pw_config_cache().end()) {
        return it->second;
    }

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "ShiftPw multilevel tune cudaGetDeviceProperties");

    GridTuneOptions options;
    options.kernel = "ShiftPwKernel";
    options.key = tune_key;
    options.force = force;
    options.benchmark = CudaBenchmarkOptions{
        2, // warmup
        5, // repeats
    };

    const std::vector<TuningParameter> space{
        {"BLOCK_SIZE", {64, 128, 256, 512, 768}},
        {"NEIGHBOR_UNROLL", {1, 2, 3, 4, 6, 9, 0}},
    };

    const auto constraint = [&](const TuningParams &params) {
        const ShiftPwLaunchConfig config = shift_pw_config_from_params(params, defaults);
        return config.blocksize > 0 && config.blocksize <= prop.maxThreadsPerBlock && config.blocksize % 32 == 0 &&
               (config.neighbor_unroll == 0 ||
                (config.neighbor_unroll > 0 && config.neighbor_unroll <= args_h[0].n_neighbors));
    };

    const auto benchmark = [&](const TuningParams &params) {
        const ShiftPwLaunchConfig config = shift_pw_config_from_params(params, defaults);
        return benchmark_cuda_ms(stream, options.benchmark, [&](cudaStream_t bench_stream) {
            launch_shift_pw_multilevel_jit<Real>(cache, args_h, d_args_scratch, bench_stream, config.blocksize,
                                                 config.neighbor_unroll);
        });
    };

    const GridTuneDecision decision =
        tune_grid(options, space, shift_pw_tuning_params(defaults), constraint, benchmark);

    const ShiftPwLaunchConfig tuned_config = shift_pw_config_from_params(decision.params, defaults);
    shift_pw_config_cache()[in_process_key] = tuned_config;
    return tuned_config;
}

template <typename Real, int DIM>
void launch_shift_pw_multilevel_autotuned(JitCache &cache, const std::vector<dmk::cuda::ShiftPwArgs<Real>> &args_h,
                                          dmk::cuda::ShiftPwArgs<Real> *d_args_scratch, cudaStream_t stream,
                                          int default_blocksize) {
    ShiftPwLaunchConfig config{default_blocksize, kDefaultNeighborUnroll};

    try {
        config = tune_shift_pw_multilevel_config<Real, DIM>(cache, args_h, d_args_scratch, stream, config);
    } catch (const std::exception &e) {
        if (env_flag_enabled("DMK_JIT_AUTOTUNE_VERBOSE")) {
            std::cerr << "[dmk jit autotune] ShiftPwKernel tuning failed; "
                      << "using defaults: " << e.what() << "\n";
        }
    }

    launch_shift_pw_multilevel_jit<Real>(cache, args_h, d_args_scratch, stream, config.blocksize,
                                         config.neighbor_unroll);
}

} // namespace dmk::cuda::jit

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_shift_pw(const ShiftPwArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 512;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_shift_pw_autotuned<Real, DIM>(jit_cache, args, stream, block_size);
}

template void launch_shift_pw<float, 2>(const ShiftPwArgs<float> &, cudaStream_t);
template void launch_shift_pw<float, 3>(const ShiftPwArgs<float> &, cudaStream_t);
template void launch_shift_pw<double, 2>(const ShiftPwArgs<double> &, cudaStream_t);
template void launch_shift_pw<double, 3>(const ShiftPwArgs<double> &, cudaStream_t);

template <typename Real, int DIM>
void launch_shift_pw_multilevel(const std::vector<ShiftPwArgs<Real>> &args_h, ShiftPwArgs<Real> *d_args_scratch,
                                cudaStream_t stream) {
    if (args_h.empty())
        return;

    constexpr int block_size = 256;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_shift_pw_multilevel_autotuned<Real, DIM>(jit_cache, args_h, d_args_scratch, stream,
                                                                    block_size);
}

template void launch_shift_pw_multilevel<float, 2>(const std::vector<ShiftPwArgs<float>> &, ShiftPwArgs<float> *,
                                                   cudaStream_t);
template void launch_shift_pw_multilevel<float, 3>(const std::vector<ShiftPwArgs<float>> &, ShiftPwArgs<float> *,
                                                   cudaStream_t);
template void launch_shift_pw_multilevel<double, 2>(const std::vector<ShiftPwArgs<double>> &, ShiftPwArgs<double> *,
                                                    cudaStream_t);
template void launch_shift_pw_multilevel<double, 3>(const std::vector<ShiftPwArgs<double>> &, ShiftPwArgs<double> *,
                                                    cudaStream_t);

} // namespace dmk::cuda
