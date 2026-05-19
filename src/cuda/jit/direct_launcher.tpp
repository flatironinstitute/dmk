#pragma once

#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_types.hpp"

#ifdef DMK_CUDA_USE_NVRTC_JIT
#include <dmk_jit_config.hpp>
#endif

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dmk::cuda::jit {
namespace detail {

template <typename Real>
inline const char *direct_real_name();

template <>
inline const char *direct_real_name<float>() {
    return "float";
}

template <>
inline const char *direct_real_name<double>() {
    return "double";
}

inline std::filesystem::path direct_jit_source_root() {
#ifdef DMK_JIT_SOURCE_DIR
    return std::filesystem::path(DMK_JIT_SOURCE_DIR);
#else
    if (const char *env = std::getenv("DMK_JIT_SOURCE_DIR")) {
        return std::filesystem::path(env);
    }

    return std::filesystem::path("src/cuda/jit_sources");
#endif
}

inline std::string direct_read_text_file(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error("DirectByBox JIT: failed to open source file: " + path.string());
    }

    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct DirectSplitSource {
    std::string header;
    std::string kernel;
};

inline DirectSplitSource direct_split_at_kernel_start(const std::string &source) {
    constexpr const char *marker = "// KERNEL_START";

    const std::size_t pos = source.find(marker);

    if (pos == std::string::npos) {
        throw std::runtime_error("DirectByBox JIT source is missing // KERNEL_START marker");
    }

    return DirectSplitSource{source.substr(0, pos), source.substr(pos)};
}

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

    static std::string coeff_prelude() { return emit_coeff_tag<Coeffs>("Coeff0"); }

    static std::string evaluator_expr() { return "LaplacePolyEvaluator2DCuda<Coeff0>"; }
};

template <typename Coeffs>
struct DirectJitTraits<dmk::cuda::LaplacePolyEvaluator3DCuda<Coeffs>> {
    using Real = typename Coeffs::value_type;

    static std::string coeff_prelude() { return emit_coeff_tag<Coeffs>("Coeff0"); }

    static std::string evaluator_expr() { return "LaplacePolyEvaluator3DCuda<Coeff0>"; }
};

template <typename Coeffs>
struct DirectJitTraits<dmk::cuda::SqrtLaplacePolyEvaluator2DCuda<Coeffs>> {
    using Real = typename Coeffs::value_type;

    static std::string coeff_prelude() { return emit_coeff_tag<Coeffs>("Coeff0"); }

    static std::string evaluator_expr() { return "SqrtLaplacePolyEvaluator2DCuda<Coeff0>"; }
};

template <typename Coeffs>
struct DirectJitTraits<dmk::cuda::SqrtLaplacePolyEvaluator3DCuda<Coeffs>> {
    using Real = typename Coeffs::value_type;

    static std::string coeff_prelude() { return emit_coeff_tag<Coeffs>("Coeff0"); }

    static std::string evaluator_expr() { return "SqrtLaplacePolyEvaluator3DCuda<Coeff0>"; }
};

template <typename CoeffsDiag, typename CoeffsOffdiag>
struct DirectJitTraits<dmk::cuda::StokesletPolyEvaluator3DCuda<CoeffsDiag, CoeffsOffdiag>> {
    using Real = typename CoeffsDiag::value_type;

    static std::string coeff_prelude() {
        return emit_coeff_tag<CoeffsDiag>("CoeffDiag") + emit_coeff_tag<CoeffsOffdiag>("CoeffOffdiag");
    }

    static std::string evaluator_expr() { return "StokesletPolyEvaluator3DCuda<CoeffDiag, CoeffOffdiag>"; }
};

template <typename CoeffsDiag, typename CoeffsOffdiag>
struct DirectJitTraits<dmk::cuda::StressletPolyEvaluator3DCuda<CoeffsDiag, CoeffsOffdiag>> {
    using Real = typename CoeffsDiag::value_type;

    static std::string coeff_prelude() {
        return emit_coeff_tag<CoeffsDiag>("CoeffDiag") + emit_coeff_tag<CoeffsOffdiag>("CoeffOffdiag");
    }

    static std::string evaluator_expr() { return "StressletPolyEvaluator3DCuda<CoeffDiag, CoeffOffdiag>"; }
};

template <typename Evaluator>
std::string make_direct_prelude(int src_tile, int blocksize) {
    using Traits = DirectJitTraits<Evaluator>;
    using Real = typename Traits::Real;

    std::ostringstream ss;

    ss << "using Real = " << direct_real_name<Real>() << ";\n\n";

    ss << Traits::coeff_prelude();

    ss << "#define DMK_DIRECT_EVALUATOR " << Traits::evaluator_expr() << "\n\n";

    ss << "constexpr int SRC_TILE = " << src_tile << ";\n";
    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n\n";

    return ss.str();
}

inline std::string make_direct_source_from_prelude(const std::string &prelude) {
    const auto source_path = direct_jit_source_root() / "direct_kernels.cu";

    const std::string file_source = direct_read_text_file(source_path);

    const DirectSplitSource split = direct_split_at_kernel_start(file_source);

    std::ostringstream generated;

    generated << prelude;
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

} // namespace detail

template <typename Evaluator, typename Real>
void launch_direct_by_box_jit(JitCache &cache, const dmk::cuda::DirectByBoxArgs<Real> &args, cudaStream_t stream,
                              int src_tile, int blocksize) {
    if (args.n_work == 0) {
        return;
    }

    static_assert(std::is_same_v<Real, typename Evaluator::scalar_type>,
                  "DirectByBox JIT: Real must match Evaluator::scalar_type");
    
    nvtxRangePush("make_prelude");
    const std::string prelude = detail::make_direct_prelude<Evaluator>(src_tile, blocksize);
    nvtxRangePop();

    nvtxRangePush("make_source_from_prelude");
    const std::string source = detail::make_direct_source_from_prelude(prelude);
    nvtxRangePop();
    
    JitKey key;
    key.name = "DirectByBoxKernel";
    key.real = detail::direct_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"SRC_TILE", src_tile},
        {"BLOCK_SIZE", blocksize},
    };

    const std::string name_expression = "static_cast<void (*)(DirectArgs)>"
                                        "(&DirectByBoxKernel<Evaluator, " +
                                        std::to_string(src_tile) + ">)";

    nvtxRangePush("get_kernel_from_source");
    auto kernel = cache.get_kernel_from_source(key, source, name_expression);
    nvtxRangePop();
    constexpr int values_per_source = Evaluator::SPATIAL_DIM + Evaluator::KERNEL_INPUT_DIM + Evaluator::NORMAL_DIM;

    const std::size_t shared_bytes = std::size_t(src_tile) * std::size_t(values_per_source) * sizeof(Real);

    kernel->launch(dim3(args.n_work, 1, 1), dim3(blocksize, 1, 1), shared_bytes, stream, args);
}

} // namespace dmk::cuda::jit