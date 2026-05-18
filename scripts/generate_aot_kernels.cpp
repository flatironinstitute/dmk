// generate_aot_kernels.cpp
//
// Builds and runs as part of the build process. Generates one of:
//   * src/aot_kernels.cpp  (host AOT, default; --target=cpu)
//   * src/cuda/kernels.cu  (CUDA AOT;        --target=cuda)
//
// Both outputs are independent of CMake's DMK_GPU_OFFLOAD: the CUDA file is
// always generatable, the build just doesn't compile it unless the option is
// set.
//
// Usage:
//   ./generate_aot_kernels                    > src/aot_kernels.cpp
//   ./generate_aot_kernels --target=cpu       > src/aot_kernels.cpp
//   ./generate_aot_kernels --target=cuda      > src/cuda/aot_kernels.cu

#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/util.hpp>
#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

enum class Target { CPU, CUDA };

struct KernelDef {
    dmk_ikernel kernel;
    std::string kernel_enum;
    int dim;
    std::string func_name;
    std::string getter_name;
};

static const std::vector<KernelDef> all_kernels = {
    {DMK_LAPLACE, "DMK_LAPLACE", 2, "laplace_2d_poly_all_pairs", "get_laplace_2d_kernel"},
    {DMK_LAPLACE, "DMK_LAPLACE", 3, "laplace_3d_poly_all_pairs", "get_laplace_3d_kernel"},
    {DMK_SQRT_LAPLACE, "DMK_SQRT_LAPLACE", 2, "sqrt_laplace_2d_poly_all_pairs", "get_sqrt_laplace_2d_kernel"},
    {DMK_SQRT_LAPLACE, "DMK_SQRT_LAPLACE", 3, "sqrt_laplace_3d_poly_all_pairs", "get_sqrt_laplace_3d_kernel"},
    // {DMK_STOKES, "DMK_STOKES", 2, "stokeslet_2d_poly_all_pairs", "get_stokeslet_2d_kernel"},
    {DMK_STOKESLET, "DMK_STOKESLET", 3, "stokeslet_3d_poly_all_pairs", "get_stokeslet_3d_kernel"},
    {DMK_STRESSLET, "DMK_STRESSLET", 3, "stresslet_3d_poly_all_pairs", "get_stresslet_3d_kernel"},
};

constexpr int min_digits = 2;
constexpr int max_digits = 12;

struct CoeffsInfo {
    int digits;
    double beta;
    std::vector<size_t> sub_sizes; // size of each sub-array
    size_t total_size;             // sum of sub_sizes
    std::vector<std::vector<double>> values;
};

void emit_coeffs_array(const std::string &name, const std::vector<std::vector<double>> &coeffs, double beta) {
    std::cout << std::format("// beta: {}\n", beta);
    std::cout << std::format("constexpr double {}[] = {{", name);
    int count = 0;
    for (const auto &cvec : coeffs) {
        for (size_t i = 0; i < cvec.size(); ++i) {
            if (count > 0)
                std::cout << ",";
            if (count % 4 == 0)
                std::cout << "\n    ";
            std::cout << std::format(" {:.17e}", cvec[i]);
            count++;
        }
    }
    std::cout << "\n};\n\n";
}

std::string coeff_name(const KernelDef &k, int digits) {
    return std::format(
        "{}_{}d_{}",
        [&] {
            switch (k.kernel) {
            case DMK_LAPLACE:
                return "laplace";
            case DMK_SQRT_LAPLACE:
                return "sqrt_laplace";
            case DMK_STOKESLET:
                return "stokeslet";
            case DMK_STRESSLET:
                return "stresslet";
            default:
                return "unknown";
            }
        }(),
        k.dim, digits);
}

// Build the "constexpr int NC0 = X; constexpr int NC1 = Y;" decl block and the
// "NC0, NC1" arg list that several emitters share.
struct NcStrings {
    std::string decls;
    std::string args;
};
NcStrings build_nc_strings(const CoeffsInfo &info) {
    NcStrings out;
    for (size_t i = 0; i < info.sub_sizes.size(); ++i) {
        if (i > 0) {
            out.decls += "\n";
            out.args += ", ";
        }
        out.decls += std::format("        constexpr int NC{} = {};", i, info.sub_sizes[i]);
        out.args += std::format("NC{}", i);
    }
    return out;
}

// =====================================================================
// CPU (host AOT) emission
// =====================================================================

void emit_getter_cpu(const KernelDef &k, const std::vector<CoeffsInfo> &infos) {
    std::cout << std::format(R"(
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> {}(dmk_eval_type eval_level, int n_digits) {{
    constexpr int UF = unroll_factor;
)",
                             k.getter_name);

    for (const auto &info : infos) {
        const auto cn = coeff_name(k, info.digits);
        const auto nc = build_nc_strings(info);

        std::cout << std::format(
            "    if (n_digits <= {}) {{\n"
            "        constexpr int ND = {}, NC_TOTAL = {};\n"
            "{}\n"
            "        std::array<Real, NC_TOTAL> coeffs;\n"
            "        std::copy_n({}, NC_TOTAL, coeffs.data());\n"
            "        return [=](Real rsc, Real cen, Real d2max, Real thresh2,\n"
            "                   int n_src, const Real *r_src, const Real *charge,\n"
            "                   const Real *normals, int n_trg, const Real *r_trg, Real *pot) {{\n"
            "            {}<Real, MaxVecLen, ND, {}>(\n"
            "                eval_level, ND, rsc, cen, d2max, thresh2, {},\n"
            "                coeffs.data(), n_src, r_src, charge, normals, n_trg, r_trg, pot, UF);\n"
            "        }};\n"
            "    }}\n",
            info.digits, info.digits, info.total_size, nc.decls, cn, k.func_name, nc.args, nc.args);
    }

    std::cout << std::format("    throw std::runtime_error(\"Unsupported n_digits: \" + std::to_string(n_digits));\n"
                             "}}\n");
}

void emit_cpu_file(const std::vector<std::pair<KernelDef, std::vector<CoeffsInfo>>> &kernels) {
    std::cout << R"(// Auto-generated by generate_aot_kernels. Do not edit.
#include <dmk.h>
#include <dmk/types.hpp>
#include <dmk/vector_kernels.hpp>
#include <sctl.hpp>

namespace dmk {
constexpr int unroll_factor = 3;

)";

    for (const auto &[k, infos] : kernels)
        for (const auto &info : infos)
            emit_coeffs_array(coeff_name(k, info.digits), info.values, info.beta);

    for (const auto &[k, infos] : kernels)
        emit_getter_cpu(k, infos);

    std::cout << "\n// Explicit instantiations\n";
    for (const auto &[k, _] : kernels)
        for (auto type : {"float", "double"})
            std::cout << std::format("template residual_evaluator_func<{0}>\n"
                                     "{1}<{0}, sctl::DefaultVecLen<{0}>()>(dmk_eval_type, int);\n",
                                     type, k.getter_name);

    std::cout << "\n} // namespace dmk\n";
}

// =====================================================================
// CUDA AOT emission. Coefficients ride into the kernel as compile-time
// *types* — one tag struct per (kernel, digits, sub-array, precision)
// exposing `value_type`, `size`, and `data[size]`. Each getter selects
// the matching tag by Real via pack_for<> and passes it as a type-template
// argument to the launcher.
// =====================================================================

// Sub-array semantic name (for stokeslet/stresslet which have diag + offdiag).
// For single-array kernels returns empty string.
std::string sub_label(const KernelDef &k, std::size_t sub_idx) {
    if (k.kernel == DMK_STOKESLET || k.kernel == DMK_STRESSLET)
        return sub_idx == 0 ? "diag" : "offdiag";
    return "";
}

std::string tag_name(const KernelDef &k, int digits, std::size_t sub_idx, char prec) {
    auto label = sub_label(k, sub_idx);
    if (label.empty())
        return std::format("{}_{}", coeff_name(k, digits), prec);
    return std::format("{}_{}_{}", coeff_name(k, digits), label, prec);
}

void emit_tag(const std::string &name, const std::vector<double> &vals, char prec) {
    const char *type = (prec == 'f') ? "float" : "double";
    std::cout << std::format("struct {} {{\n", name);
    std::cout << std::format("    using value_type = {};\n", type);
    std::cout << std::format("    static constexpr std::size_t size = {};\n", vals.size());
    std::cout << std::format("    __host__ __device__ static constexpr {} at(std::size_t i) {{\n", type);
    std::cout << std::format("        constexpr {} v[{}] = {{", type, vals.size());
    for (std::size_t i = 0; i < vals.size(); ++i) {
        if (i > 0)
            std::cout << ",";
        if (i % 4 == 0)
            std::cout << "\n            ";
        std::cout << std::format(" {:.17e}", vals[i]);
        if (prec == 'f')
            std::cout << "f";
    }
    std::cout << "};\n";
    std::cout << "        return v[i];\n";
    std::cout << "    }\n};\n";
}

void emit_getter_cuda(const KernelDef &k, const std::vector<CoeffsInfo> &infos) {
    std::cout << std::format(R"(
template <class Real>
residual_evaluator_func<Real> {}_cuda(dmk_eval_type, int n_digits) {{
)",
                             k.getter_name);

    for (const auto &info : infos) {
        // Build "using Coeffs0 = ...; using Coeffs1 = ...;" lines and the
        // comma-separated type list "Coeffs0, Coeffs1".
        std::string using_decls, using_args;
        for (std::size_t i = 0; i < info.sub_sizes.size(); ++i) {
            auto td = tag_name(k, info.digits, i, 'd');
            auto tf = tag_name(k, info.digits, i, 'f');
            using_decls += std::format("            using Coeffs{} = "
                                       "cuda_aot::pack_for<Real, cuda_aot::{}, cuda_aot::{}>;\n",
                                       i, td, tf);
            if (i > 0)
                using_args += ", ";
            using_args += std::format("Coeffs{}", i);
        }

        std::cout << std::format("    if (n_digits <= {}) {{\n"
                                 "        return [](Real rsc, Real cen, Real d2max, Real thresh2,\n"
                                 "                  int n_src, const Real *r_src, const Real *charge,\n"
                                 "                  const Real *normals, int n_trg, const Real *r_trg, Real *pot) {{\n"
                                 "{}"
                                 "            dmk::cuda::{}<{}>(rsc, cen, d2max, thresh2, n_src, r_src, charge,\n"
                                 "                                                  normals, n_trg, r_trg, pot);\n"
                                 "        }};\n"
                                 "    }}\n",
                                 info.digits, using_decls, k.func_name, using_args);
    }

    std::cout << "    throw std::runtime_error(\"Unsupported n_digits: \" + std::to_string(n_digits));\n"
                 "}\n";
}

// Map a kernel/dim back to its CUDA evaluator class name.
std::string cuda_evaluator_class(const KernelDef &k) {
    switch (k.kernel) {
    case DMK_LAPLACE:
        return k.dim == 2 ? "LaplacePolyEvaluator2DCuda" : "LaplacePolyEvaluator3DCuda";
    case DMK_SQRT_LAPLACE:
        return k.dim == 2 ? "SqrtLaplacePolyEvaluator2DCuda" : "SqrtLaplacePolyEvaluator3DCuda";
    case DMK_STOKESLET:
        return "StokesletPolyEvaluator3DCuda";
    case DMK_STRESSLET:
        return "StressletPolyEvaluator3DCuda";
    default:
        return "UNKNOWN";
    }
}

void emit_direct_dispatch_block(const KernelDef &k, const std::vector<CoeffsInfo> &infos) {
    std::cout << std::format("    if (kernel == {} && dim == {}) {{\n", k.kernel_enum, k.dim);
    for (const auto &info : infos) {
        std::string using_decls;
        std::string tparams;
        for (std::size_t i = 0; i < info.sub_sizes.size(); ++i) {
            const auto td = tag_name(k, info.digits, i, 'd');
            const auto tf = tag_name(k, info.digits, i, 'f');
            using_decls += std::format("            using Coeffs{} = "
                                       "cuda_aot::pack_for<Real, cuda_aot::{}, cuda_aot::{}>;\n",
                                       i, td, tf);
            if (i > 0)
                tparams += ", ";
            tparams += std::format("Coeffs{}", i);
        }
        std::cout << std::format("        if (n_digits <= {}) {{\n"
                                 "{}"
                                 "            cuda::launch_direct_by_box<cuda::{}<{}>>(args, stream);\n"
                                 "            return;\n"
                                 "        }}\n",
                                 info.digits, using_decls, cuda_evaluator_class(k), tparams);
    }
    std::cout << std::format("        throw std::runtime_error(\"launch_direct_by_box_dispatch: unsupported "
                             "n_digits=\" + std::to_string(n_digits) + \" for {} dim={}\");\n"
                             "    }}\n",
                             k.kernel_enum, k.dim);
}

void emit_direct_dispatch_cuda(const std::vector<std::pair<KernelDef, std::vector<CoeffsInfo>>> &kernels) {
    std::cout << "\n// Per-box direct-residual dispatch.\n";
    std::cout << "template <typename Real>\n"
                 "void cuda::launch_direct_by_box_dispatch(dmk_ikernel kernel, int dim, int n_digits,\n"
                 "                                          const cuda::DirectByBoxArgs<Real> &args,\n"
                 "                                          cudaStream_t stream) {\n";
    for (const auto &[k, infos] : kernels)
        emit_direct_dispatch_block(k, infos);
    std::cout << "    throw std::runtime_error(\"launch_direct_by_box_dispatch: unsupported (kernel,dim)\");\n"
                 "}\n\n";
    std::cout << "template void cuda::launch_direct_by_box_dispatch<float>(dmk_ikernel, int, int, "
                 "const cuda::DirectByBoxArgs<float> &, cudaStream_t);\n";
    std::cout << "template void cuda::launch_direct_by_box_dispatch<double>(dmk_ikernel, int, int, "
                 "const cuda::DirectByBoxArgs<double> &, cudaStream_t);\n";
}

void emit_cuda_file(const std::vector<std::pair<KernelDef, std::vector<CoeffsInfo>>> &kernels) {
    std::cout << R"(// Auto-generated by generate_aot_kernels. Do not edit.
#include <dmk.h>
#include <dmk/cuda/aot_kernels.hpp>
#include <dmk/cuda/direct_kernels.cuh>
#include <dmk/cuda/cuda_kernels.cuh>
#include <dmk/types.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace dmk {

namespace cuda_aot {

template <typename Real, typename TagD, typename TagF>
using pack_for = std::conditional_t<std::is_same_v<Real, double>, TagD, TagF>;

)";

    // Per-kernel/per-digit tag struct definitions, both precisions.
    for (const auto &[k, infos] : kernels) {
        for (const auto &info : infos) {
            std::cout << std::format("// {} digits={} (beta={})\n", k.getter_name, info.digits, info.beta);
            for (std::size_t i = 0; i < info.values.size(); ++i) {
                emit_tag(tag_name(k, info.digits, i, 'd'), info.values[i], 'd');
                emit_tag(tag_name(k, info.digits, i, 'f'), info.values[i], 'f');
            }
            std::cout << "\n";
        }
    }

    std::cout << "} // namespace cuda_aot\n";

    for (const auto &[k, infos] : kernels)
        emit_getter_cuda(k, infos);

    std::cout << "\n// Explicit instantiations\n";
    for (const auto &[k, _] : kernels)
        for (auto type : {"float", "double"})
            std::cout << std::format("template residual_evaluator_func<{0}>\n"
                                     "{1}_cuda<{0}>(dmk_eval_type, int);\n",
                                     type, k.getter_name);

    emit_direct_dispatch_cuda(kernels);

    std::cout << "\n} // namespace dmk\n";
}

// =====================================================================
// Driver
// =====================================================================

int main(int argc, char **argv) {
    Target target = Target::CPU;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--target=cpu")
            target = Target::CPU;
        else if (arg == "--target=cuda")
            target = Target::CUDA;
        else {
            std::cerr << std::format("Unknown argument: {}\n", arg);
            std::cerr << "Usage: generate_aot_kernels [--target=cpu|cuda]\n";
            return 1;
        }
    }

    // Compute coefficient tables once.
    std::vector<std::pair<KernelDef, std::vector<CoeffsInfo>>> kernels;
    for (auto &k : all_kernels) {
        std::vector<CoeffsInfo> infos;
        for (int digits = min_digits; digits <= max_digits; ++digits) {
            try {
                pdmk_params p;
                p.kernel = k.kernel;
                p.n_dim = k.dim;
                p.eps = std::pow(10, -digits);
                p.debug_flags = 0;
                const double beta = dmk::util::calc_bandlimiting(p);
                auto coeffs = dmk::get_local_correction_coeffs<double>(k.kernel, k.dim, digits, beta);

                CoeffsInfo info;
                info.digits = digits;
                info.beta = beta;
                info.total_size = 0;
                for (const auto &cvec : coeffs) {
                    info.sub_sizes.push_back(cvec.size());
                    info.total_size += cvec.size();
                }
                info.values = std::move(coeffs);
                infos.push_back(std::move(info));
            } catch (std::exception &e) {
                std::cerr << std::format("// Skipped {} digits={}: {}\n", k.getter_name, digits, e.what());
            }
        }
        kernels.emplace_back(k, std::move(infos));
    }

    if (target == Target::CPU)
        emit_cpu_file(kernels);
    else
        emit_cuda_file(kernels);

    return 0;
}
