#include "eval_targets_launcher.hpp"

#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_types.hpp"

#ifdef DMK_CUDA_USE_NVRTC_JIT
#include <dmk_jit_config.hpp>
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dmk::cuda::jit {
namespace {

template <typename Real>
const char* real_name();

template <>
const char* real_name<float>() {
    return "float";
}

template <>
const char* real_name<double>() {
    return "double";
}

int get_required_param(const JitKey& key, const char* name) {
    const auto it = key.params.find(name);

    if (it == key.params.end()) {
        throw std::runtime_error(
            std::string("EvalTargets JIT key missing parameter: ") + name
        );
    }

    return it->second;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "EvalTargets JIT: failed to open source file: " + path.string()
        );
    }

    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct SplitSource {
    std::string header;
    std::string kernel;
};

SplitSource split_at_kernel_start(const std::string& source) {
    constexpr const char* marker = "// KERNEL_START";

    const std::size_t pos = source.find(marker);

    if (pos == std::string::npos) {
        throw std::runtime_error(
            "EvalTargets JIT source is missing // KERNEL_START marker"
        );
    }

    return SplitSource{
        source.substr(0, pos),
        source.substr(pos)
    };
}

std::filesystem::path jit_source_root() {
#ifdef DMK_JIT_SOURCE_DIR
    return std::filesystem::path(DMK_JIT_SOURCE_DIR);
#else
    if (const char* env = std::getenv("DMK_JIT_SOURCE_DIR")) {
        return std::filesystem::path(env);
    }

    return std::filesystem::path("src/cuda/jit_sources");
#endif
}

std::string make_specialization_constants(const JitKey& key) {
    const int dim          = get_required_param(key, "DIM");
    const int eval_level   = get_required_param(key, "EVAL_LEVEL");
    const int n_charge_dim = get_required_param(key, "N_CHARGE_DIM");
    const int n_order      = get_required_param(key, "N_ORDER");
    const int blocksize    = get_required_param(key, "BLOCK_SIZE");

    std::ostringstream ss;

    ss << "#include <dmk/cuda/eval_targets_kernelargs.hpp>\n";
    ss << "using dmk::cuda::EvalTargetsArgs;\n\n";
    ss << "using Real = " << key.real << ";\n\n";
    ss << "constexpr int DIM          = " << dim << ";\n";
    ss << "constexpr int EVAL_LEVEL   = " << eval_level << ";\n";
    ss << "constexpr int N_CHARGE_DIM = " << n_charge_dim << ";\n";
    ss << "constexpr int N_ORDER      = " << n_order << ";\n";
    ss << "constexpr int BLOCK_SIZE   = " << blocksize << ";\n\n";

    return ss.str();
}

std::size_t eval_targets_shared_bytes(int dim, int n_order, std::size_t sizeof_real) {
    const int n2 = n_order * n_order;
    const int coeffs_stride_per_dim = (dim == 2) ? n2 : n2 * n_order;

    return std::size_t(coeffs_stride_per_dim) * sizeof_real;
}

void check_eval_targets_shape_or_throw(
    int dim,
    int eval_level,
    int n_charge_dim,
    int blocksize,
    const char* real,
    int n_order,
    int n_eval_boxes
) {
    const bool supported_eval_shape =
        (n_charge_dim == 1 && (eval_level == 1 || eval_level == 2)) ||
        (dim == 3 && n_charge_dim == 3 && eval_level == 1);

    if (!(dim == 2 || dim == 3) ||
        !supported_eval_shape ||
        n_order <= 0 ||
        blocksize <= 0) {
        throw std::runtime_error(
            std::string("EvalTargets JIT: unsupported shape") +
            " real=" + real +
            " dim=" + std::to_string(dim) +
            " eval_level=" + std::to_string(eval_level) +
            " n_charge_dim=" + std::to_string(n_charge_dim) +
            " n_order=" + std::to_string(n_order) +
            " n_eval_boxes=" + std::to_string(n_eval_boxes) +
            " blocksize=" + std::to_string(blocksize)
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

} // namespace

std::string make_eval_targets_source(const JitKey& key) {
    const auto source_path = jit_source_root() / "eval_targets.cu";

    const std::string file_source = read_text_file(source_path);
    const SplitSource split = split_at_kernel_start(file_source);

    std::ostringstream generated;

    generated << make_specialization_constants(key) << "\n";
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real, int DIM>
void launch_eval_targets_jit(
    JitCache& cache,
    int eval_level,
    int n_charge_dim,
    const dmk::cuda::EvalTargetsArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
) {
    if (args.n_eval_boxes == 0) {
        return;
    }

    check_eval_targets_shape_or_throw(
        DIM,
        eval_level,
        n_charge_dim,
        blocksize,
        real_name<Real>(),
        args.n_order,
        args.n_eval_boxes
    );

    JitKey key;
    key.name = "EvalTargetsByBoxKernel";
    key.real = real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"DIM", DIM},
        {"EVAL_LEVEL", eval_level},
        {"N_CHARGE_DIM", n_charge_dim},
        {"N_ORDER", args.n_order},
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes =
        eval_targets_shared_bytes(DIM, args.n_order, sizeof(Real));

    set_dynamic_smem_if_needed(
        *kernel,
        shared_bytes,
        "launch_eval_targets_jit"
    );

    kernel->launch(
        dim3(args.n_eval_boxes, 1, 1),
        dim3(blocksize, 1, 1),
        shared_bytes,
        stream,
        args
    );
}

template void launch_eval_targets_jit<float, 2>(
    JitCache&,
    int,
    int,
    const dmk::cuda::EvalTargetsArgs<float>&,
    cudaStream_t,
    int
);

template void launch_eval_targets_jit<float, 3>(
    JitCache&,
    int,
    int,
    const dmk::cuda::EvalTargetsArgs<float>&,
    cudaStream_t,
    int
);

template void launch_eval_targets_jit<double, 2>(
    JitCache&,
    int,
    int,
    const dmk::cuda::EvalTargetsArgs<double>&,
    cudaStream_t,
    int
);

template void launch_eval_targets_jit<double, 3>(
    JitCache&,
    int,
    int,
    const dmk::cuda::EvalTargetsArgs<double>&,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit
