#include "eval_targets_launcher.hpp"

#include "autotune.hpp"
#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/eval_targets_kernels.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dmk::cuda::jit {
namespace {

struct EvalTargetsLaunchConfig {
    int blocksize = 640;
    int targets_per_thread = 1;
};

struct EvalTargetsWorkSummary {
    std::vector<int> boxes;
    std::vector<int> target_counts;
    std::vector<long> pot_offsets;
    int nonempty_boxes = 0;
    long total_targets = 0;
    int max_targets_per_box = 0;
};

EvalTargetsLaunchConfig default_eval_targets_config() { return EvalTargetsLaunchConfig{}; }

int tuning_param_or(const TuningParams &params, const char *name, int fallback) {
    const auto it = params.find(name);
    return it == params.end() ? fallback : it->second;
}

TuningParams eval_targets_tuning_params(const EvalTargetsLaunchConfig &config) {
    return TuningParams{{"BLOCK_SIZE", config.blocksize}, {"TARGETS_PER_THREAD", config.targets_per_thread}};
}

EvalTargetsLaunchConfig eval_targets_config_from_params(const TuningParams &params) {
    const EvalTargetsLaunchConfig defaults = default_eval_targets_config();
    return EvalTargetsLaunchConfig{tuning_param_or(params, "BLOCK_SIZE", defaults.blocksize),
                                   tuning_param_or(params, "TARGETS_PER_THREAD", defaults.targets_per_thread)};
}

std::string make_specialization_constants(const JitKey &key) {
    const int dim = required_int_param(key, "DIM", "EvalTargets");
    const int eval_level = required_int_param(key, "EVAL_LEVEL", "EvalTargets");
    const int n_charge_dim = required_int_param(key, "N_CHARGE_DIM", "EvalTargets");
    const int n_order = required_int_param(key, "N_ORDER", "EvalTargets");
    const int blocksize = required_int_param(key, "BLOCK_SIZE", "EvalTargets");
    const int targets_per_thread = required_int_param(key, "TARGETS_PER_THREAD", "EvalTargets");

    std::ostringstream ss;

    ss << "#include <dmk/cuda/eval_targets_kernelargs.hpp>\n";
    ss << "using dmk::cuda::EvalTargetsArgs;\n\n";
    ss << "using Real = " << key.real << ";\n\n";
    ss << "constexpr int DIM          = " << dim << ";\n";
    ss << "constexpr int EVAL_LEVEL   = " << eval_level << ";\n";
    ss << "constexpr int N_CHARGE_DIM = " << n_charge_dim << ";\n";
    ss << "constexpr int N_ORDER      = " << n_order << ";\n";
    ss << "constexpr int BLOCK_SIZE   = " << blocksize << ";\n\n";
    ss << "constexpr int TARGETS_PER_THREAD = " << targets_per_thread << ";\n\n";

    return ss.str();
}

std::string make_self_correction_specialization_constants(const JitKey &key) {
    const int blocksize = required_int_param(key, "BLOCK_SIZE", "SelfCorrection");

    std::ostringstream ss;

    ss << "#include <dmk/cuda/eval_targets_kernelargs.hpp>\n";
    ss << "using dmk::cuda::SelfCorrectionArgs;\n\n";
    ss << "using Real = " << key.real << ";\n\n";
    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n\n";

    return ss.str();
}

std::size_t eval_targets_shared_bytes(int dim, int n_order, std::size_t sizeof_real) {
    const int n2 = n_order * n_order;
    const int coeffs_stride_per_dim = (dim == 2) ? n2 : n2 * n_order;

    return std::size_t(coeffs_stride_per_dim) * sizeof_real;
}

int eval_targets_pot_stride(int dim, int eval_level, int n_charge_dim) {
    const int out_dim = (eval_level == 1) ? 1 : (dim + 1);
    return n_charge_dim * out_dim;
}

void check_eval_targets_shape_or_throw(int dim, int eval_level, int n_charge_dim, const char *real, int n_order,
                                       int n_eval_boxes) {
    const bool supported_eval_shape = (n_charge_dim == 1 && (eval_level == 1 || eval_level == 2)) ||
                                      (dim == 3 && n_charge_dim == 3 && eval_level == 1);

    if (!(dim == 2 || dim == 3) || !supported_eval_shape || n_order <= 0) {
        throw std::runtime_error(
            std::string("EvalTargets JIT: unsupported shape") + " real=" + real + " dim=" + std::to_string(dim) +
            " eval_level=" + std::to_string(eval_level) + " n_charge_dim=" + std::to_string(n_charge_dim) +
            " n_order=" + std::to_string(n_order) + " n_eval_boxes=" + std::to_string(n_eval_boxes));
    }
}

template <typename Real>
EvalTargetsWorkSummary make_eval_targets_work_summary(const dmk::cuda::EvalTargetsArgs<Real> &args,
                                                       cudaStream_t stream) {
    EvalTargetsWorkSummary summary;
    summary.boxes.resize(static_cast<std::size_t>(args.n_eval_boxes));

    check_cuda(cudaMemcpyAsync(summary.boxes.data(), args.eval_targets_box_list,
                               summary.boxes.size() * sizeof(int), cudaMemcpyDeviceToHost, stream),
               "EvalTargets copy eval box list");
    check_cuda(cudaStreamSynchronize(stream), "EvalTargets sync eval box list");

    std::sort(summary.boxes.begin(), summary.boxes.end());
    summary.boxes.erase(std::unique(summary.boxes.begin(), summary.boxes.end()), summary.boxes.end());
    if (summary.boxes.empty()) {
        return summary;
    }

    const int max_box = summary.boxes.back();
    std::vector<int> target_counts(static_cast<std::size_t>(max_box) + 1);
    std::vector<long> pot_offsets(static_cast<std::size_t>(max_box) + 1);

    check_cuda(cudaMemcpyAsync(target_counts.data(), args.target_counts, target_counts.size() * sizeof(int),
                               cudaMemcpyDeviceToHost, stream),
               "EvalTargets copy target counts");
    check_cuda(cudaMemcpyAsync(pot_offsets.data(), args.pot_offsets, pot_offsets.size() * sizeof(long),
                               cudaMemcpyDeviceToHost, stream),
               "EvalTargets copy pot offsets");
    check_cuda(cudaStreamSynchronize(stream), "EvalTargets sync work summary");

    summary.target_counts.reserve(summary.boxes.size());
    summary.pot_offsets.reserve(summary.boxes.size());
    for (int box : summary.boxes) {
        const int count = target_counts[box];
        summary.target_counts.push_back(count);
        summary.pot_offsets.push_back(pot_offsets[box]);
        if (count <= 0) {
            continue;
        }
        ++summary.nonempty_boxes;
        summary.total_targets += count;
        summary.max_targets_per_box = std::max(summary.max_targets_per_box, count);
    }

    return summary;
}

template <typename Real>
AutotuneDeviceRangeSnapshots<Real>
make_eval_targets_output_snapshots(const dmk::cuda::EvalTargetsArgs<Real> &args, int dim, int eval_level,
                                   int n_charge_dim, const EvalTargetsWorkSummary &work, cudaStream_t stream) {
    AutotuneDeviceRangeSnapshots<Real> snapshots;
    snapshots.reserve(work.boxes.size());

    const int pot_stride = eval_targets_pot_stride(dim, eval_level, n_charge_dim);

    for (std::size_t i = 0; i < work.boxes.size(); ++i) {
        const int n_target = work.target_counts[i];
        if (n_target <= 0) {
            continue;
        }

        const long offset = work.pot_offsets[i];
        if (offset < 0) {
            continue;
        }

        const std::size_t count = static_cast<std::size_t>(n_target) * static_cast<std::size_t>(pot_stride);
        void *raw = nullptr;
        check_cuda(cudaMalloc(&raw, count * sizeof(Real)), "EvalTargets output snapshot cudaMalloc");

        Real *saved = static_cast<Real *>(raw);
        try {
            check_cuda(cudaMemcpyAsync(saved, args.pot_flat + offset, count * sizeof(Real), cudaMemcpyDeviceToDevice,
                                       stream),
                       "EvalTargets output snapshot cudaMemcpyAsync");
        } catch (...) {
            cudaFree(saved);
            throw;
        }

        snapshots.emplace_back(args.pot_flat, offset, count, saved);
    }

    check_cuda(cudaStreamSynchronize(stream), "EvalTargets output snapshots sync");
    return snapshots;
}

std::map<std::string, EvalTargetsLaunchConfig> &eval_targets_config_cache() {
    static std::map<std::string, EvalTargetsLaunchConfig> cache;
    return cache;
}

std::mutex &eval_targets_config_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

template <typename Real, int DIM>
std::string eval_targets_tuning_key(int eval_level, int n_charge_dim, const dmk::cuda::EvalTargetsArgs<Real> &args) {
    std::ostringstream ss;
    ss << "EvalTargetsByBoxKernel"
       << "|real=" << jit_real_name<Real>() << "|dim=" << DIM << "|eval_level=" << eval_level
       << "|n_charge_dim=" << n_charge_dim << "|n_order=" << args.n_order
       << "|n_eval_boxes=" << args.n_eval_boxes << "|tuning=block_tpt_v1";
    return ss.str();
}

void set_dynamic_smem_if_needed(const JitKernel &kernel, std::size_t shared_bytes, const char *label) {
    if (shared_bytes <= 48 * 1024) {
        return;
    }

    CUresult res = cuFuncSetAttribute(kernel.function(), CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      static_cast<int>(shared_bytes));

    if (res != CUDA_SUCCESS) {
        const char *name = nullptr;
        const char *msg = nullptr;

        cuGetErrorName(res, &name);
        cuGetErrorString(res, &msg);

        throw std::runtime_error(std::string(label) + ": cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) failed: " +
                                 (name ? name : "<unknown>") + ": " + (msg ? msg : "<no message>") +
                                 " shared_bytes=" + std::to_string(shared_bytes));
    }
}

} // namespace

std::string make_eval_targets_source(const JitKey &key) {
    const SplitSource split = load_split_jit_source("eval_targets.cu", "EvalTargets");

    std::ostringstream generated;

    generated << make_specialization_constants(key) << "\n";
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

std::string make_self_correction_source(const JitKey &key) {
    const SplitSource split = load_split_jit_source("self_correction.cu", "SelfCorrection");

    std::ostringstream generated;

    generated << make_self_correction_specialization_constants(key) << "\n";
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real, int DIM>
void launch_eval_targets_jit(JitCache &cache, int eval_level, int n_charge_dim,
                             const dmk::cuda::EvalTargetsArgs<Real> &args, cudaStream_t stream, int blocksize,
                             int targets_per_thread) {
    if (args.n_eval_boxes == 0) {
        return;
    }

    check_eval_targets_shape_or_throw(DIM, eval_level, n_charge_dim, jit_real_name<Real>(), args.n_order,
                                      args.n_eval_boxes);

    JitKey key;
    key.name = "EvalTargetsByBoxKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"DIM", DIM},
        {"EVAL_LEVEL", eval_level},
        {"N_CHARGE_DIM", n_charge_dim},
        {"N_ORDER", args.n_order},
        {"BLOCK_SIZE", blocksize},
        {"TARGETS_PER_THREAD", targets_per_thread},
    };

    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes = eval_targets_shared_bytes(DIM, args.n_order, sizeof(Real));

    set_dynamic_smem_if_needed(*kernel, shared_bytes, "launch_eval_targets_jit");

    kernel->launch(dim3(args.n_eval_boxes, 1, 1), dim3(blocksize, 1, 1), shared_bytes, stream, args);
}

template <typename Real>
void launch_self_correction_jit(JitCache &cache, const dmk::cuda::SelfCorrectionArgs<Real> &args, cudaStream_t stream,
                                int blocksize) {
    if (args.n_direct_work == 0) {
        return;
    }

    JitKey key;
    key.name = "SelfCorrectionKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    kernel->launch(dim3(args.n_direct_work, 1, 1), dim3(blocksize, 1, 1), 0, stream, args);
}

template void launch_eval_targets_jit<float, 2>(JitCache &, int, int, const dmk::cuda::EvalTargetsArgs<float> &,
                                                cudaStream_t, int, int);

template void launch_eval_targets_jit<float, 3>(JitCache &, int, int, const dmk::cuda::EvalTargetsArgs<float> &,
                                                cudaStream_t, int, int);

template void launch_eval_targets_jit<double, 2>(JitCache &, int, int, const dmk::cuda::EvalTargetsArgs<double> &,
                                                 cudaStream_t, int, int);

template void launch_eval_targets_jit<double, 3>(JitCache &, int, int, const dmk::cuda::EvalTargetsArgs<double> &,
                                                 cudaStream_t, int, int);

template void launch_self_correction_jit<float>(JitCache &, const dmk::cuda::SelfCorrectionArgs<float> &, cudaStream_t,
                                                int);

template void launch_self_correction_jit<double>(JitCache &, const dmk::cuda::SelfCorrectionArgs<double> &,
                                                 cudaStream_t, int);

template <typename Real, int DIM>
EvalTargetsLaunchConfig tune_eval_targets_launch_config(JitCache &cache, int eval_level, int n_charge_dim,
                                                        const dmk::cuda::EvalTargetsArgs<Real> &args,
                                                        cudaStream_t stream) {
    const EvalTargetsLaunchConfig defaults = default_eval_targets_config();
    if (env_flag_enabled("DMK_JIT_AUTOTUNE_DISABLE")) {
        return defaults;
    }

    check_eval_targets_shape_or_throw(DIM, eval_level, n_charge_dim, jit_real_name<Real>(), args.n_order,
                                      args.n_eval_boxes);

    const std::string tune_key = eval_targets_tuning_key<Real, DIM>(eval_level, n_charge_dim, args);
    const bool force = env_flag_enabled("DMK_JIT_AUTOTUNE_FORCE");
    int device = 0;
    check_cuda(cudaGetDevice(&device), "EvalTargets tune cudaGetDevice");
    const std::string in_process_key = tune_key + "|device=" + std::to_string(device);

    std::unique_lock<std::mutex> config_lock(eval_targets_config_cache_mutex());
    const auto it = eval_targets_config_cache().find(in_process_key);
    if (!force && it != eval_targets_config_cache().end()) {
        return it->second;
    }

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "EvalTargets tune cudaGetDeviceProperties");
    const std::size_t max_shared_bytes = prop.sharedMemPerBlockOptin > 0
                                             ? static_cast<std::size_t>(prop.sharedMemPerBlockOptin)
                                             : static_cast<std::size_t>(prop.sharedMemPerBlock);

    std::optional<AutotuneDeviceRangeSnapshots<Real>> snapshots;
    std::optional<EvalTargetsWorkSummary> work;

    GridTuneOptions options;
    options.kernel = "EvalTargetsByBoxKernel";
    options.key = tune_key;
    options.force = force;
    options.benchmark = CudaBenchmarkOptions{
        2, // warmup
        5, // repeats
    };

    const std::vector<TuningParameter> space{
        {"BLOCK_SIZE", {128, 256, 512}},
        {"TARGETS_PER_THREAD", {1, 2, 3, 4}},
    };

    const auto constraint = [&](const TuningParams &params) {
        const EvalTargetsLaunchConfig config = eval_targets_config_from_params(params);
        if (config.blocksize <= 0 || config.blocksize > prop.maxThreadsPerBlock || config.blocksize % 32 != 0) {
            return false;
        }
        if (config.targets_per_thread <= 0) {
            return false;
        }

        const std::size_t shared_bytes = eval_targets_shared_bytes(DIM, args.n_order, sizeof(Real));
        return shared_bytes <= max_shared_bytes;
    };

    const auto benchmark = [&](const TuningParams &params) {
        const EvalTargetsLaunchConfig config = eval_targets_config_from_params(params);
        if (!work) {
            work.emplace(make_eval_targets_work_summary(args, stream));
        }
        if (!snapshots) {
            snapshots.emplace(make_eval_targets_output_snapshots(args, DIM, eval_level, n_charge_dim, *work, stream));
        }

        restore_device_range_snapshots(*snapshots, stream);
        try {
            const double runtime_ms = benchmark_cuda_ms(stream, options.benchmark, [&](cudaStream_t bench_stream) {
                launch_eval_targets_jit<Real, DIM>(cache, eval_level, n_charge_dim, args, bench_stream,
                                                   config.blocksize, config.targets_per_thread);
            });
            restore_device_range_snapshots(*snapshots, stream);
            return runtime_ms;
        } catch (...) {
            restore_device_range_snapshots(*snapshots, stream);
            throw;
        }
    };

    const GridTuneDecision decision =
        tune_grid(options, space, eval_targets_tuning_params(defaults), constraint, benchmark);

    if (snapshots) {
        restore_device_range_snapshots(*snapshots, stream);
    }

    const EvalTargetsLaunchConfig tuned_config = eval_targets_config_from_params(decision.params);
    eval_targets_config_cache()[in_process_key] = tuned_config;
    return tuned_config;
}

template <typename Real, int DIM>
void launch_eval_targets_autotuned(JitCache &cache, int eval_level, int n_charge_dim,
                                   const dmk::cuda::EvalTargetsArgs<Real> &args, cudaStream_t stream) {
    EvalTargetsLaunchConfig config = default_eval_targets_config();

    try {
        config = tune_eval_targets_launch_config<Real, DIM>(cache, eval_level, n_charge_dim, args, stream);
    } catch (const std::exception &e) {
        if (env_flag_enabled("DMK_JIT_AUTOTUNE_VERBOSE")) {
            std::cerr << "[dmk jit autotune] EvalTargetsByBoxKernel tuning failed; "
                      << "using defaults: " << e.what() << "\n";
        }
    }

    launch_eval_targets_jit<Real, DIM>(cache, eval_level, n_charge_dim, args, stream, config.blocksize,
                                       config.targets_per_thread);
}

} // namespace dmk::cuda::jit

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_eval_targets(int eval_level, int n_charge_dim, const EvalTargetsArgs<Real> &args, cudaStream_t stream) {
    if (args.n_eval_boxes == 0)
        return;

    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_eval_targets_autotuned<Real, DIM>(jit_cache, eval_level, n_charge_dim, args, stream);
}

template void launch_eval_targets<float, 2>(int, int, const EvalTargetsArgs<float> &, cudaStream_t);
template void launch_eval_targets<float, 3>(int, int, const EvalTargetsArgs<float> &, cudaStream_t);
template void launch_eval_targets<double, 2>(int, int, const EvalTargetsArgs<double> &, cudaStream_t);
template void launch_eval_targets<double, 3>(int, int, const EvalTargetsArgs<double> &, cudaStream_t);

template <typename Real>
void launch_self_correction(const SelfCorrectionArgs<Real> &args, cudaStream_t stream) {
    if (args.n_direct_work == 0)
        return;

    constexpr int block = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_self_correction_jit<Real>(jit_cache, args, stream, block);
}

template void launch_self_correction<float>(const SelfCorrectionArgs<float> &, cudaStream_t);
template void launch_self_correction<double>(const SelfCorrectionArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
