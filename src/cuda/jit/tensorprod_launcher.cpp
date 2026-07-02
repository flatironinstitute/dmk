#include "tensorprod_launcher.hpp"

#include "autotune.hpp"
#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/tensorprod_kernels.hpp>

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

constexpr int TENSOR_Z_TILE = 2;
constexpr int TENSOR_I_TILE = 2;
constexpr int TENSOR_J_TILE = 4;
constexpr int TENSOR_BLOCK_SIZE = 512;

struct TensorprodLaunchConfig {
    int blocksize = TENSOR_BLOCK_SIZE;
    int z_tile = TENSOR_Z_TILE;
    int i_tile = TENSOR_I_TILE;
    int j_tile = TENSOR_J_TILE;
};

TensorprodLaunchConfig default_tensorprod_config(int blocksize = TENSOR_BLOCK_SIZE) {
    return TensorprodLaunchConfig{
        blocksize,
        TENSOR_Z_TILE,
        TENSOR_I_TILE,
        TENSOR_J_TILE,
    };
}

int tuning_param_or(const TuningParams &params, const char *name, int fallback) {
    const auto it = params.find(name);
    return it == params.end() ? fallback : it->second;
}

TuningParams tensorprod_tuning_params(const TensorprodLaunchConfig &config) {
    return TuningParams{
        {"BLOCK_SIZE", config.blocksize},
        {"Z_TILE", config.z_tile},
        {"I_TILE", config.i_tile},
        {"J_TILE", config.j_tile},
    };
}

TensorprodLaunchConfig tensorprod_config_from_params(const TuningParams &params) {
    const TensorprodLaunchConfig defaults = default_tensorprod_config();
    return TensorprodLaunchConfig{
        tuning_param_or(params, "BLOCK_SIZE", defaults.blocksize),
        tuning_param_or(params, "Z_TILE", defaults.z_tile),
        tuning_param_or(params, "I_TILE", defaults.i_tile),
        tuning_param_or(params, "J_TILE", defaults.j_tile),
    };
}

std::string make_specialization_constants(const JitKey &key) {
    const int n_order = required_int_param(key, "N_ORDER", "Tensorprod");
    const int blocksize = required_int_param(key, "BLOCK_SIZE", "Tensorprod");
    const int n_charge_dim = required_int_param(key, "N_CHARGE_DIM", "Tensorprod");
    const int z_tile = required_int_param(key, "Z_TILE", "Tensorprod");
    const int i_tile = required_int_param(key, "I_TILE", "Tensorprod");
    const int j_tile = required_int_param(key, "J_TILE", "Tensorprod");
    const std::string real_type = key.real;

    std::ostringstream ss;

    ss << "#include <dmk/cuda/tensorprod_kernelargs.hpp>\n";
    ss << "using dmk::cuda::TensorprodArgs;\n\n";

    ss << "constexpr int N_ORDER   = " << n_order << ";\n";
    ss << "constexpr int N_CHARGE_DIM  = " << n_charge_dim << ";\n";
    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n";
    ss << "constexpr int TENSOR_Z_TILE = " << z_tile << ";\n";
    ss << "constexpr int TENSOR_I_TILE = " << i_tile << ";\n";
    ss << "constexpr int TENSOR_J_TILE = " << j_tile << ";\n";
    ss << "using Real = " << real_type << ";\n\n";

    return ss.str();
}

std::size_t tensorprod_shared_bytes(int n_order, int z_tile, std::size_t sizeof_real) {
    const std::size_t n2 = std::size_t(n_order) * std::size_t(n_order);

    const std::size_t real_count = std::size_t(2 * z_tile + 3) * n2;

    return real_count * sizeof_real;
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

std::size_t tensorprod_coeff_count(int n_order, int n_charge_dim) {
    const std::size_t n = static_cast<std::size_t>(n_order);
    return n * n * n * static_cast<std::size_t>(n_charge_dim);
}

template <typename Real>
class DeviceRangeSnapshot {
  public:
    DeviceRangeSnapshot(long offset, std::size_t count, Real *data) : offset_(offset), count_(count), data_(data) {}

    DeviceRangeSnapshot(const DeviceRangeSnapshot &) = delete;
    DeviceRangeSnapshot &operator=(const DeviceRangeSnapshot &) = delete;

    DeviceRangeSnapshot(DeviceRangeSnapshot &&other) noexcept
        : offset_(other.offset_), count_(other.count_), data_(std::exchange(other.data_, nullptr)) {}

    DeviceRangeSnapshot &operator=(DeviceRangeSnapshot &&other) noexcept {
        if (this != &other) {
            release();
            offset_ = other.offset_;
            count_ = other.count_;
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }

    ~DeviceRangeSnapshot() { release(); }

    void restore(Real *dst, cudaStream_t stream) const {
        check_cuda(cudaMemcpyAsync(dst + offset_, data_, count_ * sizeof(Real), cudaMemcpyDeviceToDevice, stream),
                   "Tensorprod output snapshot restore");
    }

  private:
    long offset_ = 0;
    std::size_t count_ = 0;
    Real *data_ = nullptr;

    void release() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
        }
    }
};

template <typename Real>
using TensorprodOutputSnapshots = std::vector<DeviceRangeSnapshot<Real>>;

template <typename Real>
TensorprodOutputSnapshots<Real> make_tensorprod_output_snapshots(const dmk::cuda::TensorprodArgs<Real> &args,
                                                                 cudaStream_t stream) {
    std::vector<int> dst_boxes(static_cast<std::size_t>(args.n_pairs));
    check_cuda(cudaMemcpyAsync(dst_boxes.data(), args.dst_boxes, dst_boxes.size() * sizeof(int), cudaMemcpyDeviceToHost,
                               stream),
               "Tensorprod copy dst boxes");
    check_cuda(cudaStreamSynchronize(stream), "Tensorprod sync dst boxes");

    std::sort(dst_boxes.begin(), dst_boxes.end());
    dst_boxes.erase(std::unique(dst_boxes.begin(), dst_boxes.end()), dst_boxes.end());

    std::vector<long> dst_offsets(dst_boxes.size());
    for (std::size_t i = 0; i < dst_boxes.size(); ++i) {
        check_cuda(cudaMemcpyAsync(&dst_offsets[i], args.proxy_offsets + dst_boxes[i], sizeof(long),
                                   cudaMemcpyDeviceToHost, stream),
                   "Tensorprod copy proxy offset");
    }
    check_cuda(cudaStreamSynchronize(stream), "Tensorprod sync proxy offsets");

    std::sort(dst_offsets.begin(), dst_offsets.end());
    dst_offsets.erase(std::unique(dst_offsets.begin(), dst_offsets.end()), dst_offsets.end());

    const std::size_t coeff_count = tensorprod_coeff_count(args.n_order, args.n_charge_dim);

    TensorprodOutputSnapshots<Real> snapshots;
    snapshots.reserve(dst_offsets.size());

    for (long offset : dst_offsets) {
        if (offset < 0) {
            continue;
        }

        void *raw = nullptr;
        check_cuda(cudaMalloc(&raw, coeff_count * sizeof(Real)), "Tensorprod allocate output snapshot");

        Real *saved = static_cast<Real *>(raw);
        try {
            check_cuda(cudaMemcpyAsync(saved, args.proxy_flat + offset, coeff_count * sizeof(Real),
                                       cudaMemcpyDeviceToDevice, stream),
                       "Tensorprod copy output snapshot");
        } catch (...) {
            cudaFree(saved);
            throw;
        }
        snapshots.emplace_back(offset, coeff_count, saved);
    }

    check_cuda(cudaStreamSynchronize(stream), "Tensorprod sync output snapshots");
    return snapshots;
}

template <typename Real>
void restore_tensorprod_output_snapshots(const TensorprodOutputSnapshots<Real> &snapshots, Real *proxy_flat,
                                         cudaStream_t stream) {
    for (const auto &snapshot : snapshots) {
        snapshot.restore(proxy_flat, stream);
    }
    check_cuda(cudaStreamSynchronize(stream), "Tensorprod sync output snapshot restore");
}

template <typename Real, int DIM>
std::string tensorprod_tuning_key(const dmk::cuda::TensorprodArgs<Real> &args) {
    std::ostringstream ss;
    ss << "TensorprodKernel"
       << "|real=" << jit_real_name<Real>() << "|dim=" << DIM << "|n_order=" << args.n_order
       << "|n_charge_dim=" << args.n_charge_dim << "|n_pairs=" << args.n_pairs
       << "|atomic=" << (args.additive_atomic ? 1 : 0);
    return ss.str();
}

std::map<std::string, TensorprodLaunchConfig> &tensorprod_config_cache() {
    static std::map<std::string, TensorprodLaunchConfig> cache;
    return cache;
}

std::mutex &tensorprod_config_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

} // namespace

std::string make_tensorprod_source(const JitKey &key) {
    const SplitSource split = load_split_jit_source("tensorproduct.cu", "Tensorprod");

    std::ostringstream generated;

    generated << split.header << "\n";
    generated << make_specialization_constants(key);
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real>
void launch_tensorprod_jit_config(JitCache &cache, const dmk::cuda::TensorprodArgs<Real> &args, cudaStream_t stream,
                                  const TensorprodLaunchConfig &config) {
    if (args.n_pairs == 0) {
        return;
    }

    JitKey key;
    key.name = "TensorprodKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"N_ORDER", args.n_order}, {"N_CHARGE_DIM", args.n_charge_dim}, {"BLOCK_SIZE", config.blocksize},
        {"Z_TILE", config.z_tile}, {"I_TILE", config.i_tile},           {"J_TILE", config.j_tile},
    };

    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes = tensorprod_shared_bytes(args.n_order, config.z_tile, sizeof(Real));

    set_dynamic_smem_if_needed(*kernel, shared_bytes, "launch_tensorprod_jit");

    kernel->launch(dim3(args.n_pairs, 1, 1), dim3(config.blocksize, 1, 1), shared_bytes, stream, args);
}

template <typename Real>
void launch_tensorprod_jit(JitCache &cache, const dmk::cuda::TensorprodArgs<Real> &args, cudaStream_t stream,
                           int blocksize) {
    launch_tensorprod_jit_config<Real>(cache, args, stream, default_tensorprod_config(blocksize));
}

template void launch_tensorprod_jit<float>(JitCache &, const dmk::cuda::TensorprodArgs<float> &, cudaStream_t, int);

template void launch_tensorprod_jit<double>(JitCache &, const dmk::cuda::TensorprodArgs<double> &, cudaStream_t, int);

template <typename Real, int DIM>
TensorprodLaunchConfig tune_tensorprod_launch_config(JitCache &cache, const dmk::cuda::TensorprodArgs<Real> &args,
                                                     cudaStream_t stream) {
    const TensorprodLaunchConfig defaults = default_tensorprod_config();
    if (env_flag_enabled("DMK_JIT_AUTOTUNE_DISABLE")) {
        return defaults;
    }

    const std::string tune_key = tensorprod_tuning_key<Real, DIM>(args);
    const bool force = env_flag_enabled("DMK_JIT_AUTOTUNE_FORCE");
    int device = 0;
    check_cuda(cudaGetDevice(&device), "Tensorprod tune cudaGetDevice");
    const std::string in_process_key = tune_key + "|device=" + std::to_string(device);

    std::unique_lock<std::mutex> config_lock(tensorprod_config_cache_mutex());
    const auto it = tensorprod_config_cache().find(in_process_key);
    if (it != tensorprod_config_cache().end()) {
        return it->second;
    }

    std::optional<TensorprodOutputSnapshots<Real>> snapshots;

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "Tensorprod tune cudaGetDeviceProperties");

    const std::size_t max_shared_bytes = prop.sharedMemPerBlockOptin > 0
                                             ? static_cast<std::size_t>(prop.sharedMemPerBlockOptin)
                                             : static_cast<std::size_t>(prop.sharedMemPerBlock);

    GridTuneOptions options;
    options.kernel = "TensorprodKernel";
    options.key = tune_key;
    options.force = force;
    options.benchmark = CudaBenchmarkOptions{
        2, // warmup
        5, // repeats
    };

    const std::vector<TuningParameter> space{
        {"BLOCK_SIZE", {128, 256, 512}},
        {"Z_TILE", {1, 2, 4}},
        {"I_TILE", {1, 2, 3, 4}},
        {"J_TILE", {2, 4, 6}},
    };

    const auto constraint = [&](const TuningParams &params) {
        const TensorprodLaunchConfig config = tensorprod_config_from_params(params);
        if (config.blocksize <= 0 || config.blocksize > prop.maxThreadsPerBlock || config.blocksize % 32 != 0) {
            return false;
        }
        if (config.z_tile <= 0 || config.z_tile > args.n_order || config.i_tile <= 0 || config.i_tile > args.n_order ||
            config.j_tile <= 0 || config.j_tile > args.n_order) {
            return false;
        }
        if (config.i_tile * config.j_tile > 16) {
            return false;
        }

        const std::size_t shared_bytes = tensorprod_shared_bytes(args.n_order, config.z_tile, sizeof(Real));
        return shared_bytes <= max_shared_bytes;
    };

    const auto benchmark = [&](const TuningParams &params) {
        const TensorprodLaunchConfig config = tensorprod_config_from_params(params);
        if (!snapshots) {
            snapshots.emplace(make_tensorprod_output_snapshots(args, stream));
        }

        restore_tensorprod_output_snapshots(*snapshots, args.proxy_flat, stream);
        try {
            const double runtime_ms = benchmark_cuda_ms(stream, options.benchmark, [&](cudaStream_t bench_stream) {
                launch_tensorprod_jit_config<Real>(cache, args, bench_stream, config);
            });
            restore_tensorprod_output_snapshots(*snapshots, args.proxy_flat, stream);
            return runtime_ms;
        } catch (...) {
            restore_tensorprod_output_snapshots(*snapshots, args.proxy_flat, stream);
            throw;
        }
    };

    const GridTuneDecision decision =
        tune_grid(options, space, tensorprod_tuning_params(defaults), constraint, benchmark);

    if (snapshots) {
        restore_tensorprod_output_snapshots(*snapshots, args.proxy_flat, stream);
    }

    const TensorprodLaunchConfig tuned_config = tensorprod_config_from_params(decision.params);
    tensorprod_config_cache()[in_process_key] = tuned_config;
    return tuned_config;
}

template <typename Real, int DIM>
void launch_tensorprod_autotuned(JitCache &cache, const dmk::cuda::TensorprodArgs<Real> &args, cudaStream_t stream) {
    TensorprodLaunchConfig config = default_tensorprod_config();

    try {
        config = tune_tensorprod_launch_config<Real, DIM>(cache, args, stream);
    } catch (const std::exception &e) {
        if (env_flag_enabled("DMK_JIT_AUTOTUNE_VERBOSE")) {
            std::cerr << "[dmk jit autotune] TensorprodKernel tuning failed; "
                      << "using defaults: " << e.what() << "\n";
        }
    }

    launch_tensorprod_jit_config<Real>(cache, args, stream, config);
}

} // namespace dmk::cuda::jit

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_tensorprod(const TensorprodArgs<Real> &args, cudaStream_t stream) {
    if (args.n_pairs == 0)
        return;

    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_tensorprod_autotuned<Real, DIM>(jit_cache, args, stream);
}

template void launch_tensorprod<float, 2>(const TensorprodArgs<float> &, cudaStream_t);
template void launch_tensorprod<float, 3>(const TensorprodArgs<float> &, cudaStream_t);
template void launch_tensorprod<double, 2>(const TensorprodArgs<double> &, cudaStream_t);
template void launch_tensorprod<double, 3>(const TensorprodArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
