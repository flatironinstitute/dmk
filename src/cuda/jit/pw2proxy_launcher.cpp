#include "pw2proxy_launcher.hpp"

#include "autotune.hpp"
#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/pw2proxy_kernels.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iterator>
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

struct PwToProxyLaunchConfig {
    int k1_tile = 6;
    int col_reg = 2;
    int k2_tile = 2;
    int k3_tile = 3;
    int kr_tile = 6;
    int blocksize = 128;
};

PwToProxyLaunchConfig default_pw_to_proxy_config(bool multilevel) {
    if (multilevel) {
        return PwToProxyLaunchConfig{
            18, // K1_TILE
            1,  // COL_REG
            2,  // K2_TILE
            3,  // K3_TILE
            9,  // KR_TILE
            256 // blocksize
        };
    }

    return PwToProxyLaunchConfig{};
}

int tuning_param_or(const TuningParams& params, const char* name, int fallback) {
    const auto it = params.find(name);
    return it == params.end() ? fallback : it->second;
}

TuningParams pw_to_proxy_tuning_params(const PwToProxyLaunchConfig& config) {
    return TuningParams{
        {"K1_TILE", config.k1_tile},
        {"COL_REG", config.col_reg},
        {"K2_TILE", config.k2_tile},
        {"K3_TILE", config.k3_tile},
        {"KR_TILE", config.kr_tile},
        {"BLOCK_SIZE", config.blocksize},
    };
}

PwToProxyLaunchConfig pw_to_proxy_config_from_params(
    const TuningParams& params,
    const PwToProxyLaunchConfig& defaults
) {
    return PwToProxyLaunchConfig{
        tuning_param_or(params, "K1_TILE", defaults.k1_tile),
        tuning_param_or(params, "COL_REG", defaults.col_reg),
        tuning_param_or(params, "K2_TILE", defaults.k2_tile),
        tuning_param_or(params, "K3_TILE", defaults.k3_tile),
        tuning_param_or(params, "KR_TILE", defaults.kr_tile),
        tuning_param_or(params, "BLOCK_SIZE", defaults.blocksize),
    };
}

std::size_t pw_to_proxy_coeff_count(int n_order, int n_charge_dim) {
    const std::size_t n = static_cast<std::size_t>(n_order);
    return n * n * n * static_cast<std::size_t>(n_charge_dim);
}

std::string make_prelude(const JitKey& key) {
    const int k1_tile   = required_int_param(key, "K1_TILE", "PwToProxy");
    const int col_reg   = required_int_param(key, "COL_REG", "PwToProxy");
    const int k2_tile   = required_int_param(key, "K2_TILE", "PwToProxy");
    const int k3_tile   = required_int_param(key, "K3_TILE", "PwToProxy");
    const int kr_tile   = required_int_param(key, "KR_TILE", "PwToProxy");
    const int blocksize = required_int_param(key, "BLOCK_SIZE", "PwToProxy");
    const int n_order      = required_int_param(key, "N_ORDER", "PwToProxy");
    const int n_pw         = required_int_param(key, "N_PW", "PwToProxy");
    const int n_pw2        = required_int_param(key, "N_PW2", "PwToProxy");
    const int n_charge_dim = required_int_param(key, "N_CHARGE_DIM", "PwToProxy");

    std::ostringstream ss;

    ss << "#include <dmk/cuda/pw2proxy_kernelargs.hpp>\n\n";

    ss << "using dmk::cuda::PwToProxyArgs;\n";

    ss << "using Real = " << key.real << ";\n\n";

    ss << "constexpr int K1_TILE   = " << k1_tile << ";\n";
    ss << "constexpr int COL_REG   = " << col_reg << ";\n";
    ss << "constexpr int K2_TILE   = " << k2_tile << ";\n";
    ss << "constexpr int K3_TILE   = " << k3_tile << ";\n";
    ss << "constexpr int KR_TILE   = " << kr_tile << ";\n";
    ss << "constexpr int N_ORDER      = " << n_order << ";\n";
    ss << "constexpr int N_PW         = " << n_pw << ";\n";
    ss << "constexpr int N_PW2        = " << n_pw2 << ";\n";
    ss << "constexpr int N_CHARGE_DIM = " << n_charge_dim << ";\n";
    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n\n";

    return ss.str();
}

std::size_t pw_to_proxy_shared_bytes(
    int max_n_pw,
    int max_n_pw2,
    int max_n_order,
    int k1_tile,
    std::size_t complex_size
) {
    const int max_k_pad = ((max_n_order + 3) / 4) * 4;
    const int max_phase1_cols = max_n_pw * max_n_pw2;

    const std::size_t complex_count =
        std::size_t(max_n_pw) * std::size_t(max_k_pad) +
        std::size_t(k1_tile) * std::size_t(max_phase1_cols) +
        std::size_t(k1_tile) * std::size_t(max_n_pw2) * std::size_t(max_n_order);

    return complex_count * complex_size;
}

template <typename Real>
void check_pw_to_proxy_shape_or_throw(
    const dmk::cuda::PwToProxyArgs<Real>& a,
    const char* where
) {
    auto fail = [&](const char* field) {
        throw std::runtime_error(
            std::string("PwToProxy JIT: invalid args/shape in ") +
            where +
            " field=" + field +
            " n_boxes_at_level=" + std::to_string(a.n_boxes_at_level) +
            " n_order=" + std::to_string(a.n_order) +
            " n_pw=" + std::to_string(a.n_pw) +
            " n_pw2=" + std::to_string(a.n_pw2) +
            " n_charge_dim=" + std::to_string(a.n_charge_dim) +
            " pw_in_stride=" + std::to_string(a.pw_in_stride) +
            " box_ids=" + std::to_string(reinterpret_cast<std::uintptr_t>(a.box_ids)) +
            " pw_in_pool=" + std::to_string(reinterpret_cast<std::uintptr_t>(a.pw_in_pool)) +
            " pw2poly=" + std::to_string(reinterpret_cast<std::uintptr_t>(a.pw2poly)) +
            " proxy_flat=" + std::to_string(reinterpret_cast<std::uintptr_t>(a.proxy_flat)) +
            " proxy_offsets=" + std::to_string(reinterpret_cast<std::uintptr_t>(a.proxy_offsets))
        );
    };

    if (a.n_boxes_at_level < 0) fail("n_boxes_at_level");
    if (a.n_order <= 0) fail("n_order");
    if (a.n_pw <= 0) fail("n_pw");
    if (a.n_pw2 <= 0) fail("n_pw2");
    if (a.n_charge_dim <= 0) fail("n_charge_dim");
    if (a.n_boxes_at_level > 1 && a.pw_in_stride <= 0) {
        fail("pw_in_stride");
    }
    if (a.box_ids == nullptr) fail("box_ids");
    if (a.pw_in_pool == nullptr) fail("pw_in_pool");
    if (a.pw2poly == nullptr) fail("pw2poly");
    if (a.proxy_flat == nullptr) fail("proxy_flat");
    if (a.proxy_offsets == nullptr) fail("proxy_offsets");
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
AutotuneDeviceRangeSnapshots<Real> make_pw_to_proxy_output_snapshots(
    const dmk::cuda::PwToProxyArgs<Real>& args,
    cudaStream_t stream
) {
    std::vector<int> box_ids(static_cast<std::size_t>(args.n_boxes_at_level));
    check_cuda(
        cudaMemcpyAsync(
            box_ids.data(),
            args.box_ids,
            box_ids.size() * sizeof(int),
            cudaMemcpyDeviceToHost,
            stream
        ),
        "PwToProxy copy box ids"
    );
    check_cuda(cudaStreamSynchronize(stream), "PwToProxy sync box ids");

    std::sort(box_ids.begin(), box_ids.end());
    box_ids.erase(std::unique(box_ids.begin(), box_ids.end()), box_ids.end());

    std::vector<std::pair<Real*, long>> ranges;
    ranges.reserve(box_ids.size());
    for (int box : box_ids) {
        long offset = -1;
        check_cuda(
            cudaMemcpyAsync(
                &offset,
                args.proxy_offsets + box,
                sizeof(long),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "PwToProxy copy proxy offset"
        );
        check_cuda(cudaStreamSynchronize(stream), "PwToProxy sync proxy offset");
        ranges.emplace_back(args.proxy_flat, offset);
    }

    return make_device_range_snapshots<Real>(
        std::move(ranges),
        pw_to_proxy_coeff_count(args.n_order, args.n_charge_dim),
        stream
    );
}

template <typename Real>
AutotuneDeviceRangeSnapshots<Real> make_pw_to_proxy_multilevel_output_snapshots(
    const std::vector<dmk::cuda::PwToProxyArgs<Real>>& args_h,
    cudaStream_t stream
) {
    AutotuneDeviceRangeSnapshots<Real> snapshots;

    for (const auto& args : args_h) {
        if (args.n_boxes_at_level == 0) {
            continue;
        }

        std::vector<int> box_ids(static_cast<std::size_t>(args.n_boxes_at_level));
        check_cuda(
            cudaMemcpyAsync(
                box_ids.data(),
                args.box_ids,
                box_ids.size() * sizeof(int),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "PwToProxy multilevel copy box ids"
        );
        check_cuda(cudaStreamSynchronize(stream), "PwToProxy multilevel sync box ids");

        std::sort(box_ids.begin(), box_ids.end());
        box_ids.erase(std::unique(box_ids.begin(), box_ids.end()), box_ids.end());

        std::vector<std::pair<Real*, long>> ranges;
        ranges.reserve(box_ids.size());
        for (int box : box_ids) {
            long offset = -1;
            check_cuda(
                cudaMemcpyAsync(
                    &offset,
                    args.proxy_offsets + box,
                    sizeof(long),
                    cudaMemcpyDeviceToHost,
                    stream
                ),
                "PwToProxy multilevel copy proxy offset"
            );
            check_cuda(cudaStreamSynchronize(stream), "PwToProxy multilevel sync proxy offset");
            ranges.emplace_back(args.proxy_flat, offset);
        }

        auto part = make_device_range_snapshots<Real>(
            std::move(ranges),
            pw_to_proxy_coeff_count(args.n_order, args.n_charge_dim),
            stream
        );
        snapshots.insert(
            snapshots.end(),
            std::make_move_iterator(part.begin()),
            std::make_move_iterator(part.end())
        );
    }

    return snapshots;
}

std::map<std::string, PwToProxyLaunchConfig>& pw_to_proxy_config_cache() {
    static std::map<std::string, PwToProxyLaunchConfig> cache;
    return cache;
}

std::mutex& pw_to_proxy_config_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

template <typename Real, int DIM>
std::string pw_to_proxy_tuning_key(
    const dmk::cuda::PwToProxyArgs<Real>& args
) {
    std::ostringstream ss;
    ss << "PwToProxyKernel"
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
std::string pw_to_proxy_multilevel_tuning_key(
    const std::vector<dmk::cuda::PwToProxyArgs<Real>>& args_h,
    int max_boxes
) {
    std::ostringstream ss;
    ss << "PwToProxyMultiLevelKernel"
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

std::string make_pw2proxy_source(const JitKey& key) {
    std::filesystem::path filename;

    if (key.name == "PwToProxyKernel") {
        filename = "pw2proxy.cu";
    } else if (key.name == "PwToProxyMultiLevelKernel") {
        filename = "pw2proxy_multilevel.cu";
    } else {
        throw std::runtime_error(
            "PwToProxy JIT: unknown kernel name: " + key.name
        );
    }

    const SplitSource split = load_split_jit_source(filename.string(), "PwToProxy");

    std::ostringstream generated;

    generated << make_prelude(key);
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real>
void launch_pw_to_proxy_jit(
    JitCache& cache,
    const dmk::cuda::PwToProxyArgs<Real>& args,
    cudaStream_t stream,
    int k1_tile,
    int col_reg,
    int k2_tile,
    int k3_tile,
    int kr_tile,
    int blocksize
) {
    if (args.n_boxes_at_level == 0) {
        return;
    }

    check_pw_to_proxy_shape_or_throw(args, "single");

    JitKey key;
    key.name = "PwToProxyKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"K1_TILE", k1_tile},
        {"COL_REG", col_reg},
        {"K2_TILE", k2_tile},
        {"K3_TILE", k3_tile},
        {"KR_TILE", kr_tile},
        {"N_ORDER", args.n_order},
        {"N_CHARGE_DIM", args.n_charge_dim},
        {"N_PW", args.n_pw},
        {"N_PW2", args.n_pw2},
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes =
        pw_to_proxy_shared_bytes(
            args.n_pw,
            args.n_pw2,
            args.n_order,
            k1_tile,
            sizeof(dmk::cuda_helpers::complx<Real>)
        );

    set_dynamic_smem_if_needed(
        *kernel,
        shared_bytes,
        "launch_pw_to_proxy_jit"
    );

    kernel->launch(
        dim3(args.n_boxes_at_level, 1, 1),
        dim3(blocksize, 1, 1),
        shared_bytes,
        stream,
        args
    );
}

template <typename Real>
void launch_pw_to_proxy_multilevel_jit(
    JitCache& cache,
    const std::vector<dmk::cuda::PwToProxyArgs<Real>>& args_h,
    dmk::cuda::PwToProxyArgs<Real>* d_args_scratch,
    cudaStream_t stream,
    int k1_tile,
    int col_reg,
    int k2_tile,
    int k3_tile,
    int kr_tile,
    int blocksize
) {
    if (args_h.empty()) {
        return;
    }

    int max_boxes = 0;
    int max_n_pw = 0;
    int max_n_pw2 = 0;
    int max_n_order = 0;

    for (const auto& a : args_h) {
        if (a.n_boxes_at_level == 0) {
            continue;
        }

        check_pw_to_proxy_shape_or_throw(a, "multilevel");

        max_boxes = std::max(max_boxes, a.n_boxes_at_level);
        max_n_pw = std::max(max_n_pw, a.n_pw);
        max_n_pw2 = std::max(max_n_pw2, a.n_pw2);
        max_n_order = std::max(max_n_order, a.n_order);
    }

    if (max_boxes == 0) {
        return;
    }

    DMK_CHECK_CUDA(
        cudaMemcpyAsync(
            d_args_scratch,
            args_h.data(),
            args_h.size() * sizeof(dmk::cuda::PwToProxyArgs<Real>),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    JitKey key;
    key.name = "PwToProxyMultiLevelKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"K1_TILE", k1_tile},
        {"COL_REG", col_reg},
        {"K2_TILE", k2_tile},
        {"K3_TILE", k3_tile},
        {"KR_TILE", kr_tile},
        {"N_ORDER", args_h[0].n_order},
        {"N_CHARGE_DIM", args_h[0].n_charge_dim},
        {"N_PW", args_h[0].n_pw},
        {"N_PW2", args_h[0].n_pw2},
        {"BLOCK_SIZE", blocksize},

    };

    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes =
        pw_to_proxy_shared_bytes(
            max_n_pw,
            max_n_pw2,
            max_n_order,
            k1_tile,
            sizeof(dmk::cuda_helpers::complx<Real>)
        );

    set_dynamic_smem_if_needed(
        *kernel,
        shared_bytes,
        "launch_pw_to_proxy_multilevel_jit"
    );

    const int n_args = static_cast<int>(args_h.size());

    kernel->launch(
        dim3(max_boxes, n_args, 1),
        dim3(blocksize, 1, 1),
        shared_bytes,
        stream,
        d_args_scratch,
        n_args
    );
}

template void launch_pw_to_proxy_jit<float>(
    JitCache&,
    const dmk::cuda::PwToProxyArgs<float>&,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int,
    int
);

template void launch_pw_to_proxy_jit<double>(
    JitCache&,
    const dmk::cuda::PwToProxyArgs<double>&,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int,
    int
);

template void launch_pw_to_proxy_multilevel_jit<float>(
    JitCache&,
    const std::vector<dmk::cuda::PwToProxyArgs<float>>&,
    dmk::cuda::PwToProxyArgs<float>*,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int,
    int
);

template void launch_pw_to_proxy_multilevel_jit<double>(
    JitCache&,
    const std::vector<dmk::cuda::PwToProxyArgs<double>>&,
    dmk::cuda::PwToProxyArgs<double>*,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int,
    int
);

template <typename Real, int DIM>
PwToProxyLaunchConfig tune_pw_to_proxy_launch_config(
    JitCache& cache,
    const dmk::cuda::PwToProxyArgs<Real>& args,
    cudaStream_t stream
) {
    const PwToProxyLaunchConfig defaults = default_pw_to_proxy_config(false);
    if (env_flag_enabled("DMK_JIT_AUTOTUNE_DISABLE")) {
        return defaults;
    }

    const std::string tune_key = pw_to_proxy_tuning_key<Real, DIM>(args);
    const bool force = env_flag_enabled("DMK_JIT_AUTOTUNE_FORCE");
    int device = 0;
    check_cuda(cudaGetDevice(&device), "PwToProxy tune cudaGetDevice");
    const std::string in_process_key =
        tune_key + "|device=" + std::to_string(device);

    std::unique_lock<std::mutex> config_lock(pw_to_proxy_config_cache_mutex());
    const auto it = pw_to_proxy_config_cache().find(in_process_key);
    if (it != pw_to_proxy_config_cache().end()) {
        return it->second;
    }

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "PwToProxy tune cudaGetDeviceProperties");
    const std::size_t max_shared_bytes =
        prop.sharedMemPerBlockOptin > 0
            ? static_cast<std::size_t>(prop.sharedMemPerBlockOptin)
            : static_cast<std::size_t>(prop.sharedMemPerBlock);

    std::optional<AutotuneDeviceRangeSnapshots<Real>> snapshots;

    GridTuneOptions options;
    options.kernel = "PwToProxyKernel";
    options.key = tune_key;
    options.force = force;
    options.benchmark = CudaBenchmarkOptions{
        2, // warmup
        5, // repeats
    };

    const std::vector<TuningParameter> space{
        {"K1_TILE", {4, 6, 8}},
        {"COL_REG", {1, 2}},
        {"K2_TILE", {2, 3}},
        {"K3_TILE", {3}},
        {"KR_TILE", {4, 6}},
        {"BLOCK_SIZE", {128, 256}},
    };

    const auto constraint = [&](const TuningParams& params) {
        const PwToProxyLaunchConfig config =
            pw_to_proxy_config_from_params(params, defaults);

        if (config.blocksize <= 0 || config.blocksize > prop.maxThreadsPerBlock ||
            config.blocksize % 32 != 0) {
            return false;
        }
        if (config.k1_tile <= 0 || config.k1_tile > args.n_order ||
            config.col_reg <= 0 ||
            config.k2_tile <= 0 || config.k2_tile > args.n_order ||
            config.k3_tile <= 0 || config.k3_tile > args.n_order ||
            config.kr_tile <= 0 || config.kr_tile > config.k1_tile) {
            return false;
        }
        if (config.k1_tile * config.col_reg > 24 ||
            config.kr_tile * config.k2_tile > 24) {
            return false;
        }

        const std::size_t shared_bytes =
            pw_to_proxy_shared_bytes(
                args.n_pw,
                args.n_pw2,
                args.n_order,
                config.k1_tile,
                sizeof(dmk::cuda_helpers::complx<Real>)
            );
        return shared_bytes <= max_shared_bytes;
    };

    const auto benchmark = [&](const TuningParams& params) {
        const PwToProxyLaunchConfig config =
            pw_to_proxy_config_from_params(params, defaults);

        if (!snapshots) {
            snapshots.emplace(make_pw_to_proxy_output_snapshots(args, stream));
        }

        restore_device_range_snapshots(*snapshots, stream);
        try {
            const double runtime_ms =
                benchmark_cuda_ms(stream, options.benchmark, [&](cudaStream_t bench_stream) {
                    launch_pw_to_proxy_jit<Real>(
                        cache,
                        args,
                        bench_stream,
                        config.k1_tile,
                        config.col_reg,
                        config.k2_tile,
                        config.k3_tile,
                        config.kr_tile,
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
            pw_to_proxy_tuning_params(defaults),
            constraint,
            benchmark
        );

    if (snapshots) {
        restore_device_range_snapshots(*snapshots, stream);
    }

    const PwToProxyLaunchConfig tuned_config =
        pw_to_proxy_config_from_params(decision.params, defaults);
    pw_to_proxy_config_cache()[in_process_key] = tuned_config;
    return tuned_config;
}

template <typename Real, int DIM>
void launch_pw_to_proxy_autotuned(
    JitCache& cache,
    const dmk::cuda::PwToProxyArgs<Real>& args,
    cudaStream_t stream
) {
    PwToProxyLaunchConfig config = default_pw_to_proxy_config(false);

    try {
        config = tune_pw_to_proxy_launch_config<Real, DIM>(
            cache,
            args,
            stream
        );
    } catch (const std::exception& e) {
        if (env_flag_enabled("DMK_JIT_AUTOTUNE_VERBOSE")) {
            std::cerr << "[dmk jit autotune] PwToProxyKernel tuning failed; "
                      << "using defaults: " << e.what() << "\n";
        }
    }

    launch_pw_to_proxy_jit<Real>(
        cache,
        args,
        stream,
        config.k1_tile,
        config.col_reg,
        config.k2_tile,
        config.k3_tile,
        config.kr_tile,
        config.blocksize
    );
}

template <typename Real, int DIM>
PwToProxyLaunchConfig tune_pw_to_proxy_multilevel_launch_config(
    JitCache& cache,
    const std::vector<dmk::cuda::PwToProxyArgs<Real>>& args_h,
    dmk::cuda::PwToProxyArgs<Real>* d_args_scratch,
    cudaStream_t stream
) {
    const PwToProxyLaunchConfig defaults = default_pw_to_proxy_config(true);
    if (env_flag_enabled("DMK_JIT_AUTOTUNE_DISABLE")) {
        return defaults;
    }

    int max_boxes = 0;
    int max_n_pw = 0;
    int max_n_pw2 = 0;
    int max_n_order = 0;

    for (const auto& a : args_h) {
        max_boxes = std::max(max_boxes, a.n_boxes_at_level);
        max_n_pw = std::max(max_n_pw, a.n_pw);
        max_n_pw2 = std::max(max_n_pw2, a.n_pw2);
        max_n_order = std::max(max_n_order, a.n_order);
    }

    if (max_boxes == 0 || args_h.empty()) {
        return defaults;
    }

    const std::string tune_key =
        pw_to_proxy_multilevel_tuning_key<Real, DIM>(args_h, max_boxes);
    const bool force = env_flag_enabled("DMK_JIT_AUTOTUNE_FORCE");
    int device = 0;
    check_cuda(cudaGetDevice(&device), "PwToProxy multilevel tune cudaGetDevice");
    const std::string in_process_key =
        tune_key + "|device=" + std::to_string(device);

    std::unique_lock<std::mutex> config_lock(pw_to_proxy_config_cache_mutex());
    const auto it = pw_to_proxy_config_cache().find(in_process_key);
    if (it != pw_to_proxy_config_cache().end()) {
        return it->second;
    }

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "PwToProxy multilevel tune cudaGetDeviceProperties");
    const std::size_t max_shared_bytes =
        prop.sharedMemPerBlockOptin > 0
            ? static_cast<std::size_t>(prop.sharedMemPerBlockOptin)
            : static_cast<std::size_t>(prop.sharedMemPerBlock);

    std::optional<AutotuneDeviceRangeSnapshots<Real>> snapshots;

    GridTuneOptions options;
    options.kernel = "PwToProxyMultiLevelKernel";
    options.key = tune_key;
    options.force = force;
    options.benchmark = CudaBenchmarkOptions{
        2, // warmup
        5, // repeats
    };

    const std::vector<TuningParameter> space{
        {"K1_TILE", {12, 18}},
        {"COL_REG", {1, 2}},
        {"K2_TILE", {2, 3}},
        {"K3_TILE", {3}},
        {"KR_TILE", {6, 9}},
        {"BLOCK_SIZE", {128, 256}},
    };

    const auto constraint = [&](const TuningParams& params) {
        const PwToProxyLaunchConfig config =
            pw_to_proxy_config_from_params(params, defaults);

        if (config.blocksize <= 0 || config.blocksize > prop.maxThreadsPerBlock ||
            config.blocksize % 32 != 0) {
            return false;
        }
        if (config.k1_tile <= 0 || config.k1_tile > max_n_order ||
            config.col_reg <= 0 ||
            config.k2_tile <= 0 || config.k2_tile > max_n_order ||
            config.k3_tile <= 0 || config.k3_tile > max_n_order ||
            config.kr_tile <= 0 || config.kr_tile > config.k1_tile) {
            return false;
        }
        if (config.k1_tile * config.col_reg > 36 ||
            config.kr_tile * config.k2_tile > 36) {
            return false;
        }

        const std::size_t shared_bytes =
            pw_to_proxy_shared_bytes(
                max_n_pw,
                max_n_pw2,
                max_n_order,
                config.k1_tile,
                sizeof(dmk::cuda_helpers::complx<Real>)
            );
        return shared_bytes <= max_shared_bytes;
    };

    const auto benchmark = [&](const TuningParams& params) {
        const PwToProxyLaunchConfig config =
            pw_to_proxy_config_from_params(params, defaults);

        if (!snapshots) {
            snapshots.emplace(
                make_pw_to_proxy_multilevel_output_snapshots(args_h, stream)
            );
        }

        restore_device_range_snapshots(*snapshots, stream);
        try {
            const double runtime_ms =
                benchmark_cuda_ms(stream, options.benchmark, [&](cudaStream_t bench_stream) {
                    launch_pw_to_proxy_multilevel_jit<Real>(
                        cache,
                        args_h,
                        d_args_scratch,
                        bench_stream,
                        config.k1_tile,
                        config.col_reg,
                        config.k2_tile,
                        config.k3_tile,
                        config.kr_tile,
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
            pw_to_proxy_tuning_params(defaults),
            constraint,
            benchmark
        );

    if (snapshots) {
        restore_device_range_snapshots(*snapshots, stream);
    }

    const PwToProxyLaunchConfig tuned_config =
        pw_to_proxy_config_from_params(decision.params, defaults);
    pw_to_proxy_config_cache()[in_process_key] = tuned_config;
    return tuned_config;
}

template <typename Real, int DIM>
void launch_pw_to_proxy_multilevel_autotuned(
    JitCache& cache,
    const std::vector<dmk::cuda::PwToProxyArgs<Real>>& args_h,
    dmk::cuda::PwToProxyArgs<Real>* d_args_scratch,
    cudaStream_t stream
) {
    PwToProxyLaunchConfig config = default_pw_to_proxy_config(true);

    try {
        config = tune_pw_to_proxy_multilevel_launch_config<Real, DIM>(
            cache,
            args_h,
            d_args_scratch,
            stream
        );
    } catch (const std::exception& e) {
        if (env_flag_enabled("DMK_JIT_AUTOTUNE_VERBOSE")) {
            std::cerr << "[dmk jit autotune] PwToProxyMultiLevelKernel tuning failed; "
                      << "using defaults: " << e.what() << "\n";
        }
    }

    launch_pw_to_proxy_multilevel_jit<Real>(
        cache,
        args_h,
        d_args_scratch,
        stream,
        config.k1_tile,
        config.col_reg,
        config.k2_tile,
        config.k3_tile,
        config.kr_tile,
        config.blocksize
    );
}

} // namespace dmk::cuda::jit

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_pw_to_proxy(const PwToProxyArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_pw_to_proxy_autotuned<Real, DIM>(
        jit_cache,
        args,
        stream
    );
}

template void launch_pw_to_proxy<float, 2>(const PwToProxyArgs<float> &, cudaStream_t);
template void launch_pw_to_proxy<float, 3>(const PwToProxyArgs<float> &, cudaStream_t);
template void launch_pw_to_proxy<double, 2>(const PwToProxyArgs<double> &, cudaStream_t);
template void launch_pw_to_proxy<double, 3>(const PwToProxyArgs<double> &, cudaStream_t);

template <typename Real, int DIM>
void launch_pw_to_proxy_multilevel(
    const std::vector<PwToProxyArgs<Real>> &args_h,
    PwToProxyArgs<Real> *d_args_scratch,
    cudaStream_t stream
) {
    if (args_h.empty())
        return;

    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_pw_to_proxy_multilevel_autotuned<Real, DIM>(
        jit_cache,
        args_h,
        d_args_scratch,
        stream
    );
}

template void launch_pw_to_proxy_multilevel<float, 2>(const std::vector<PwToProxyArgs<float>> &,
                                                      PwToProxyArgs<float> *, cudaStream_t);
template void launch_pw_to_proxy_multilevel<float, 3>(const std::vector<PwToProxyArgs<float>> &,
                                                      PwToProxyArgs<float> *, cudaStream_t);
template void launch_pw_to_proxy_multilevel<double, 2>(const std::vector<PwToProxyArgs<double>> &,
                                                       PwToProxyArgs<double> *, cudaStream_t);
template void launch_pw_to_proxy_multilevel<double, 3>(const std::vector<PwToProxyArgs<double>> &,
                                                       PwToProxyArgs<double> *, cudaStream_t);

} // namespace dmk::cuda
