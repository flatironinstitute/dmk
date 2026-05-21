#include "eval_targets_launcher.hpp"

#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace dmk::cuda::jit {
namespace {

std::string make_specialization_constants(const JitKey& key) {
    const int dim          = required_int_param(key, "DIM", "EvalTargets");
    const int eval_level   = required_int_param(key, "EVAL_LEVEL", "EvalTargets");
    const int n_charge_dim = required_int_param(key, "N_CHARGE_DIM", "EvalTargets");
    const int n_order      = required_int_param(key, "N_ORDER", "EvalTargets");
    const int blocksize    = required_int_param(key, "BLOCK_SIZE", "EvalTargets");

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

std::string make_self_correction_specialization_constants(const JitKey& key) {
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

void check_eval_targets_shape_or_throw(
    int dim,
    int eval_level,
    int n_charge_dim,
    const char* real,
    int n_order,
    int n_eval_boxes
) {
    const bool supported_eval_shape =
        (n_charge_dim == 1 && (eval_level == 1 || eval_level == 2)) ||
        (dim == 3 && n_charge_dim == 3 && eval_level == 1);

    if (!(dim == 2 || dim == 3) ||
        !supported_eval_shape ||
        n_order <= 0) {
        throw std::runtime_error(
            std::string("EvalTargets JIT: unsupported shape") +
            " real=" + real +
            " dim=" + std::to_string(dim) +
            " eval_level=" + std::to_string(eval_level) +
            " n_charge_dim=" + std::to_string(n_charge_dim) +
            " n_order=" + std::to_string(n_order) +
            " n_eval_boxes=" + std::to_string(n_eval_boxes)
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
    const SplitSource split = load_split_jit_source("eval_targets.cu", "EvalTargets");

    std::ostringstream generated;

    generated << make_specialization_constants(key) << "\n";
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

std::string make_self_correction_source(const JitKey& key) {
    const SplitSource split = load_split_jit_source("self_correction.cu", "SelfCorrection");

    std::ostringstream generated;

    generated << make_self_correction_specialization_constants(key) << "\n";
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
        jit_real_name<Real>(),
        args.n_order,
        args.n_eval_boxes
    );

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

template <typename Real>
void launch_self_correction_jit(
    JitCache& cache,
    const dmk::cuda::SelfCorrectionArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
) {
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

    kernel->launch(
        dim3(args.n_direct_work, 1, 1),
        dim3(blocksize, 1, 1),
        0,
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

template void launch_self_correction_jit<float>(
    JitCache&,
    const dmk::cuda::SelfCorrectionArgs<float>&,
    cudaStream_t,
    int
);

template void launch_self_correction_jit<double>(
    JitCache&,
    const dmk::cuda::SelfCorrectionArgs<double>&,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit
