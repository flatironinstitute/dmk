#pragma once

#include "direct_source_registry.hpp"
#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <cuda_runtime.h>

#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

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

} // namespace detail

template <typename Evaluator, typename Real>
void launch_direct_by_box_jit(JitCache &cache, const dmk::cuda::DirectByBoxArgs<Real> &args, cudaStream_t stream,
                              int src_tile, int blocksize) {
    if (args.n_work == 0) {
        return;
    }

    static_assert(std::is_same_v<Real, typename Evaluator::scalar_type>,
                  "DirectByBox JIT: Real must match Evaluator::scalar_type");

    detail::ensure_direct_descriptor_registered<Evaluator>();

    JitKey key;
    key.name = detail::direct_kernel_name_for_type<Evaluator>();
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"SRC_TILE", src_tile},
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    constexpr int values_per_source = Evaluator::SPATIAL_DIM + Evaluator::KERNEL_INPUT_DIM + Evaluator::NORMAL_DIM;

    const std::size_t shared_bytes = std::size_t(src_tile) * std::size_t(values_per_source) * sizeof(Real);

    kernel->launch(dim3(args.n_work, 1, 1), dim3(blocksize, 1, 1), shared_bytes, stream, args);
}

} // namespace dmk::cuda::jit
