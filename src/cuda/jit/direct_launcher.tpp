#pragma once

#include "autotune.hpp"
#include "direct_source_registry.hpp"
#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dmk::cuda::jit {
namespace detail {

template <typename Coeffs>
std::string emit_coeff_tag(const char *name) {
    using Real = typename Coeffs::value_type;

    std::ostringstream ss;
    ss << std::setprecision(std::numeric_limits<Real>::max_digits10);

    ss << "struct " << name << " {\n";
    ss << "    static constexpr int size = " << static_cast<int>(Coeffs::size) << ";\n";
    ss << "    __device__ static constexpr Real at(int i) {\n";
    ss << "        constexpr Real data[size] = {\n";

    for (std::size_t i = 0; i < Coeffs::size; ++i) {
        ss << "            Real{" << Real{Coeffs::at(i)} << "}";

        if (i + 1 < Coeffs::size) {
            ss << ",";
        }

        ss << "\n";
    }

    ss << "        };\n";
    ss << "        return data[i];\n";
    ss << "    }\n";
    ss << "};\n\n";

    return ss.str();
}

template <typename Evaluator>
struct DirectJitTraits;

template <typename Coeffs>
struct DirectJitTraits<dmk::cuda::LaplacePolyEvaluator2DCuda<Coeffs>> {
    using Real = typename Coeffs::value_type;

    static const char *family() { return "Laplace2D"; }

    static std::string coeff_prelude() { return emit_coeff_tag<Coeffs>("Coeff0"); }

    static std::string evaluator_expr() { return "LaplacePolyEvaluator2DCuda<Coeff0>"; }
};

template <typename Coeffs>
struct DirectJitTraits<dmk::cuda::LaplacePolyEvaluator3DCuda<Coeffs>> {
    using Real = typename Coeffs::value_type;

    static const char *family() { return "Laplace3D"; }

    static std::string coeff_prelude() { return emit_coeff_tag<Coeffs>("Coeff0"); }

    static std::string evaluator_expr() { return "LaplacePolyEvaluator3DCuda<Coeff0>"; }
};

template <typename Coeffs>
struct DirectJitTraits<dmk::cuda::SqrtLaplacePolyEvaluator2DCuda<Coeffs>> {
    using Real = typename Coeffs::value_type;

    static const char *family() { return "SqrtLaplace2D"; }

    static std::string coeff_prelude() { return emit_coeff_tag<Coeffs>("Coeff0"); }

    static std::string evaluator_expr() { return "SqrtLaplacePolyEvaluator2DCuda<Coeff0>"; }
};

template <typename Coeffs>
struct DirectJitTraits<dmk::cuda::SqrtLaplacePolyEvaluator3DCuda<Coeffs>> {
    using Real = typename Coeffs::value_type;

    static const char *family() { return "SqrtLaplace3D"; }

    static std::string coeff_prelude() { return emit_coeff_tag<Coeffs>("Coeff0"); }

    static std::string evaluator_expr() { return "SqrtLaplacePolyEvaluator3DCuda<Coeff0>"; }
};

template <typename CoeffsDiag, typename CoeffsOffdiag>
struct DirectJitTraits<dmk::cuda::StokesletPolyEvaluator3DCuda<CoeffsDiag, CoeffsOffdiag>> {
    using Real = typename CoeffsDiag::value_type;

    static const char *family() { return "Stokeslet3D"; }

    static std::string coeff_prelude() {
        return emit_coeff_tag<CoeffsDiag>("CoeffDiag") + emit_coeff_tag<CoeffsOffdiag>("CoeffOffdiag");
    }

    static std::string evaluator_expr() { return "StokesletPolyEvaluator3DCuda<CoeffDiag, CoeffOffdiag>"; }
};

template <typename CoeffsDiag, typename CoeffsOffdiag>
struct DirectJitTraits<dmk::cuda::StressletPolyEvaluator3DCuda<CoeffsDiag, CoeffsOffdiag>> {
    using Real = typename CoeffsDiag::value_type;

    static const char *family() { return "Stresslet3D"; }

    static std::string coeff_prelude() {
        return emit_coeff_tag<CoeffsDiag>("CoeffDiag") + emit_coeff_tag<CoeffsOffdiag>("CoeffOffdiag");
    }

    static std::string evaluator_expr() { return "StressletPolyEvaluator3DCuda<CoeffDiag, CoeffOffdiag>"; }
};

template <typename Evaluator>
DirectSourceDescriptor make_direct_descriptor() {
    using Traits = DirectJitTraits<Evaluator>;

    return DirectSourceDescriptor{Traits::coeff_prelude(), Traits::evaluator_expr()};
}

template <typename Evaluator>
const std::string &direct_kernel_name_for_type() {
    using Traits = DirectJitTraits<Evaluator>;
    using Real = typename Traits::Real;

    static const std::string name = [] {
        const int id = next_direct_kernel_id();

        return std::string("DirectByBoxKernel_") + jit_real_name<Real>() + "_" + Traits::family() + "_" +
               std::to_string(id);
    }();

    return name;
}

template <typename Evaluator>
void ensure_direct_descriptor_registered() {
    static const bool registered = [] {
        const std::string &kernel_name = direct_kernel_name_for_type<Evaluator>();

        register_direct_source_descriptor(kernel_name, make_direct_descriptor<Evaluator>());

        return true;
    }();

    (void)registered;
}

struct DirectLaunchConfig {
    int src_tile = 32;
    int blocksize = 128;
};

inline int tuning_param_or(const TuningParams& params, const char* name, int fallback) {
    const auto it = params.find(name);
    return it == params.end() ? fallback : it->second;
}

inline TuningParams direct_tuning_params(const DirectLaunchConfig& config) {
    return TuningParams{
        {"SRC_TILE", config.src_tile},
        {"BLOCK_SIZE", config.blocksize},
    };
}

inline DirectLaunchConfig direct_config_from_params(
    const TuningParams& params,
    const DirectLaunchConfig& defaults
) {
    return DirectLaunchConfig{
        tuning_param_or(params, "SRC_TILE", defaults.src_tile),
        tuning_param_or(params, "BLOCK_SIZE", defaults.blocksize),
    };
}

inline std::string fnv1a64_hex(const std::string& text) {
    std::uint64_t h = 14695981039346656037ull;
    for (unsigned char c : text) {
        h ^= static_cast<std::uint64_t>(c);
        h *= 1099511628211ull;
    }

    std::ostringstream ss;
    ss << std::hex << h;
    return ss.str();
}

template <typename Evaluator>
std::string direct_evaluator_hash() {
    static const std::string hash = [] {
        const DirectSourceDescriptor descriptor = make_direct_descriptor<Evaluator>();
        return fnv1a64_hex(
            std::string(DirectJitTraits<Evaluator>::family()) +
            "\n" +
            descriptor.evaluator_expr +
            "\n" +
            descriptor.coeff_prelude
        );
    }();

    return hash;
}

template <typename Evaluator, typename Real>
std::string direct_tuning_key(
    const dmk::cuda::DirectByBoxArgs<Real>& args
) {
    std::ostringstream ss;
    ss << "DirectByBoxKernel"
       << "|real=" << jit_real_name<Real>()
       << "|family=" << DirectJitTraits<Evaluator>::family()
       << "|eval=" << direct_evaluator_hash<Evaluator>()
       << "|spatial_dim=" << Evaluator::SPATIAL_DIM
       << "|input_dim=" << Evaluator::KERNEL_INPUT_DIM
       << "|output_dim=" << Evaluator::KERNEL_OUTPUT_DIM
       << "|normal_dim=" << Evaluator::NORMAL_DIM
       << "|n_work=" << args.n_work
       << "|n_levels=" << args.n_levels
       << "|nlist1_stride=" << args.nlist1_stride;
    return ss.str();
}

inline std::map<std::string, DirectLaunchConfig>& direct_config_cache() {
    static std::map<std::string, DirectLaunchConfig> cache;
    return cache;
}

inline std::mutex& direct_config_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

template <typename Evaluator, typename Real>
AutotuneDeviceRangeSnapshots<Real> make_direct_output_snapshots(
    const dmk::cuda::DirectByBoxArgs<Real>& args,
    cudaStream_t stream
) {
    std::vector<int> boxes(static_cast<std::size_t>(args.n_work));
    check_cuda(
        cudaMemcpyAsync(
            boxes.data(),
            args.direct_work,
            boxes.size() * sizeof(int),
            cudaMemcpyDeviceToHost,
            stream
        ),
        "DirectByBox copy direct work"
    );
    check_cuda(cudaStreamSynchronize(stream), "DirectByBox sync direct work");

    std::sort(boxes.begin(), boxes.end());
    boxes.erase(std::unique(boxes.begin(), boxes.end()), boxes.end());

    AutotuneDeviceRangeSnapshots<Real> snapshots;
    snapshots.reserve(boxes.size());

    for (int box : boxes) {
        int n_targets = 0;
        long offset = -1;

        check_cuda(
            cudaMemcpyAsync(
                &n_targets,
                args.target_counts + box,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "DirectByBox copy target count"
        );
        check_cuda(
            cudaMemcpyAsync(
                &offset,
                args.pot_offsets + box,
                sizeof(long),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "DirectByBox copy pot offset"
        );
        check_cuda(cudaStreamSynchronize(stream), "DirectByBox sync output metadata");

        if (n_targets <= 0 || offset < 0) {
            continue;
        }

        const std::size_t count =
            static_cast<std::size_t>(n_targets) *
            static_cast<std::size_t>(Evaluator::KERNEL_OUTPUT_DIM);

        void* raw = nullptr;
        check_cuda(
            cudaMalloc(&raw, count * sizeof(Real)),
            "DirectByBox allocate output snapshot"
        );

        Real* saved = static_cast<Real*>(raw);
        try {
            check_cuda(
                cudaMemcpyAsync(
                    saved,
                    args.pot_flat + offset,
                    count * sizeof(Real),
                    cudaMemcpyDeviceToDevice,
                    stream
                ),
                "DirectByBox copy output snapshot"
            );
        } catch (...) {
            cudaFree(saved);
            throw;
        }

        snapshots.emplace_back(args.pot_flat, offset, count, saved);
    }

    check_cuda(cudaStreamSynchronize(stream), "DirectByBox sync output snapshots");
    return snapshots;
}

template <typename Evaluator, typename Real>
void launch_direct_by_box_jit_config(
    JitCache& cache,
    const dmk::cuda::DirectByBoxArgs<Real>& args,
    cudaStream_t stream,
    const DirectLaunchConfig& config
) {
    if (args.n_work == 0) {
        return;
    }

    static_assert(std::is_same_v<Real, typename Evaluator::scalar_type>,
                  "DirectByBox JIT: Real must match Evaluator::scalar_type");

    ensure_direct_descriptor_registered<Evaluator>();

    JitKey key;
    key.name = direct_kernel_name_for_type<Evaluator>();
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"SRC_TILE", config.src_tile},
        {"BLOCK_SIZE", config.blocksize},
    };

    auto kernel = cache.get_kernel(key);

    constexpr int values_per_source =
        Evaluator::SPATIAL_DIM +
        Evaluator::KERNEL_INPUT_DIM +
        Evaluator::NORMAL_DIM;

    const std::size_t shared_bytes =
        std::size_t(config.src_tile) *
        std::size_t(values_per_source) *
        sizeof(Real);

    kernel->launch(
        dim3(args.n_work, 1, 1),
        dim3(config.blocksize, 1, 1),
        shared_bytes,
        stream,
        args
    );
}

template <typename Evaluator, typename Real>
DirectLaunchConfig tune_direct_launch_config(
    JitCache& cache,
    const dmk::cuda::DirectByBoxArgs<Real>& args,
    cudaStream_t stream,
    const DirectLaunchConfig& defaults
) {
    if (env_flag_enabled("DMK_JIT_AUTOTUNE_DISABLE")) {
        return defaults;
    }

    const std::string tune_key = direct_tuning_key<Evaluator>(args);
    const bool force = env_flag_enabled("DMK_JIT_AUTOTUNE_FORCE");
    int device = 0;
    check_cuda(cudaGetDevice(&device), "DirectByBox tune cudaGetDevice");
    const std::string in_process_key =
        tune_key + "|device=" + std::to_string(device);

    std::unique_lock<std::mutex> config_lock(direct_config_cache_mutex());
    const auto it = direct_config_cache().find(in_process_key);
    if (it != direct_config_cache().end()) {
        return it->second;
    }

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "DirectByBox tune cudaGetDeviceProperties");
    const std::size_t max_shared_bytes =
        prop.sharedMemPerBlockOptin > 0
            ? static_cast<std::size_t>(prop.sharedMemPerBlockOptin)
            : static_cast<std::size_t>(prop.sharedMemPerBlock);

    std::optional<AutotuneDeviceRangeSnapshots<Real>> snapshots;

    GridTuneOptions options;
    options.kernel = "DirectByBoxKernel";
    options.key = tune_key;
    options.force = force;
    options.benchmark = CudaBenchmarkOptions{
        2, // warmup
        5, // repeats
    };

    const std::vector<TuningParameter> space{
        {"SRC_TILE", {16, 32, 64, 96, 128, 192, 256}},
        {"BLOCK_SIZE", {64, 128, 256, 512}},
    };

    constexpr int values_per_source =
        Evaluator::SPATIAL_DIM +
        Evaluator::KERNEL_INPUT_DIM +
        Evaluator::NORMAL_DIM;

    const auto constraint = [&](const TuningParams& params) {
        const DirectLaunchConfig config =
            direct_config_from_params(params, defaults);

        if (config.src_tile <= 0 ||
            config.blocksize <= 0 ||
            config.blocksize > prop.maxThreadsPerBlock ||
            config.blocksize % 32 != 0) {
            return false;
        }

        const std::size_t shared_bytes =
            std::size_t(config.src_tile) *
            std::size_t(values_per_source) *
            sizeof(Real);

        return shared_bytes <= max_shared_bytes;
    };

    const auto benchmark = [&](const TuningParams& params) {
        const DirectLaunchConfig config =
            direct_config_from_params(params, defaults);

        if (!snapshots) {
            snapshots.emplace(
                make_direct_output_snapshots<Evaluator>(args, stream)
            );
        }

        restore_device_range_snapshots(*snapshots, stream);
        try {
            const double runtime_ms =
                benchmark_cuda_ms(stream, options.benchmark, [&](cudaStream_t bench_stream) {
                    launch_direct_by_box_jit_config<Evaluator, Real>(
                        cache,
                        args,
                        bench_stream,
                        config
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
            direct_tuning_params(defaults),
            constraint,
            benchmark
        );

    if (snapshots) {
        restore_device_range_snapshots(*snapshots, stream);
    }

    const DirectLaunchConfig tuned_config =
        direct_config_from_params(decision.params, defaults);
    direct_config_cache()[in_process_key] = tuned_config;
    return tuned_config;
}

} // namespace detail

template <typename Evaluator, typename Real>
void launch_direct_by_box_jit(JitCache &cache, const dmk::cuda::DirectByBoxArgs<Real> &args, cudaStream_t stream,
                              int src_tile, int blocksize) {
    if (args.n_work == 0) {
        return;
    }

    detail::DirectLaunchConfig config{src_tile, blocksize};

    try {
        config = detail::tune_direct_launch_config<Evaluator, Real>(
            cache,
            args,
            stream,
            config
        );
    } catch (const std::exception& e) {
        if (env_flag_enabled("DMK_JIT_AUTOTUNE_VERBOSE")) {
            std::cerr << "[dmk jit autotune] DirectByBoxKernel tuning failed; "
                      << "using defaults: " << e.what() << "\n";
        }
    }

    detail::launch_direct_by_box_jit_config<Evaluator, Real>(
        cache,
        args,
        stream,
        config
    );
}

} // namespace dmk::cuda::jit
