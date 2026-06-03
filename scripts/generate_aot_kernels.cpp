// generate_aot_kernels.cpp
//
// Builds and runs as part of the build process. Generates aot_kernels.cpp
// with all coefficient tables and kernel dispatch functions.
//
// Usage: ./generate_aot_kernels > src/aot_kernels.cpp

#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/util.hpp>
#include <format>
#include <iostream>
#include <string>
#include <vector>

struct KernelDef {
    dmk_ikernel kernel;
    std::string kernel_enum;
    int dim;
    std::string func_name;
    std::string getter_name;
    std::vector<dmk_eval_type> eval_levels;
};

// clang-format off
static const std::vector<KernelDef> all_kernels = {
    {DMK_LAPLACE, "DMK_LAPLACE", 2, "laplace_2d_poly_all_pairs", "get_laplace_2d_kernel",
     {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}},
    {DMK_LAPLACE, "DMK_LAPLACE", 3, "laplace_3d_poly_all_pairs", "get_laplace_3d_kernel",
     {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}},
    {DMK_SQRT_LAPLACE, "DMK_SQRT_LAPLACE", 2, "sqrt_laplace_2d_poly_all_pairs", "get_sqrt_laplace_2d_kernel",
     {DMK_POTENTIAL}},
    {DMK_SQRT_LAPLACE, "DMK_SQRT_LAPLACE", 3, "sqrt_laplace_3d_poly_all_pairs", "get_sqrt_laplace_3d_kernel",
     {DMK_POTENTIAL}},
    {DMK_STOKESLET, "DMK_STOKESLET", 3, "stokeslet_3d_poly_all_pairs", "get_stokeslet_3d_kernel", {DMK_VELOCITY}},
    {DMK_STRESSLET, "DMK_STRESSLET", 3, "stresslet_3d_poly_all_pairs", "get_stresslet_3d_kernel", {DMK_VELOCITY}},
    {DMK_LAPLACE_DIPOLE, "DMK_LAPLACE_DIPOLE", 3, "laplace_dipole_3d_poly_all_pairs", "get_laplace_dipole_3d_kernel",
     {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}},
};
// clang-format on

constexpr int min_digits = 2;
constexpr int max_digits = 12;

struct CoeffsInfo {
    int digits;
    double beta;
    std::vector<size_t> sub_sizes; // size of each sub-array
    size_t total_size;             // sum of sub_sizes
    dmk_eval_type eval_level;
};

std::string eval_level_suffix(dmk_eval_type el) {
    switch (el) {
    case DMK_POTENTIAL:
        return "pot";
    case DMK_POTENTIAL_GRAD:
        return "potgrad";
    case DMK_POTENTIAL_GRAD_HESSIAN:
        return "pothess";
    case DMK_VELOCITY:
        return "vel";
    case DMK_VELOCITY_PRESSURE:
        return "velprs";
    }
    return "unknown";
}

std::string eval_level_enum_name(dmk_eval_type el) {
    switch (el) {
    case DMK_POTENTIAL:
        return "DMK_POTENTIAL";
    case DMK_POTENTIAL_GRAD:
        return "DMK_POTENTIAL_GRAD";
    case DMK_POTENTIAL_GRAD_HESSIAN:
        return "DMK_POTENTIAL_GRAD_HESSIAN";
    case DMK_VELOCITY:
        return "DMK_VELOCITY";
    case DMK_VELOCITY_PRESSURE:
        return "DMK_VELOCITY_PRESSURE";
    }
    return "DMK_POTENTIAL";
}

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

std::string kernel_base_name(const KernelDef &k) {
    switch (k.kernel) {
    case DMK_LAPLACE:
        return "laplace";
    case DMK_SQRT_LAPLACE:
        return "sqrt_laplace";
    case DMK_STOKESLET:
        return "stokeslet";
    case DMK_STRESSLET:
        return "stresslet";
    case DMK_LAPLACE_DIPOLE:
        return "laplace_dipole";
    default:
        return "unknown";
    }
}

std::string coeff_name(const KernelDef &k, int digits, dmk_eval_type el) {
    return std::format("{}_{}d_{}_{}", kernel_base_name(k), k.dim, eval_level_suffix(el), digits);
}

void emit_getter_branch_for_level(const KernelDef &k, dmk_eval_type el, const std::vector<CoeffsInfo> &infos) {
    for (const auto &info : infos) {
        if (info.eval_level != el)
            continue;
        const auto cn = coeff_name(k, info.digits, el);

        // Build the n_coeffs_rt template args string
        // e.g. for 1 sub-array: "NC0"
        // for 2 sub-arrays: "NC0, NC1"
        std::string nc_decls, nc_args;
        for (size_t i = 0; i < info.sub_sizes.size(); ++i) {
            if (i > 0) {
                nc_decls += "\n";
                nc_args += ", ";
            }
            nc_decls += std::format("            constexpr int NC{} = {};", i, info.sub_sizes[i]);
            nc_args += std::format("NC{}", i);
        }

        std::cout << std::format(
            "        if (n_digits <= {}) {{\n"
            "            constexpr int ND = {}, NC_TOTAL = {};\n"
            "{}\n"
            "            std::array<Real, NC_TOTAL> coeffs;\n"
            "            std::copy_n({}, NC_TOTAL, coeffs.data());\n"
            "            return [=](Real rsc, Real cen, Real d2max, Real thresh2,\n"
            "                       int n_src, const Real *r_src, const Real *charge,\n"
            "                       const Real *normals, int n_trg, const Real *r_trg, Real *pot) {{\n"
            "                {}<Real, MaxVecLen, ND, {}, {}>(\n"
            "                    eval_level, ND, rsc, cen, d2max, thresh2, {},\n"
            "                    coeffs.data(), n_src, r_src, charge, normals, n_trg, r_trg, pot, UF);\n"
            "            }};\n"
            "        }}\n",
            info.digits, info.digits, info.total_size, nc_decls, cn, k.func_name, nc_args, eval_level_enum_name(el),
            nc_args);
    }
}

void emit_getter(const KernelDef &k, const std::vector<CoeffsInfo> &infos) {
    std::cout << std::format(R"(
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> {}(dmk_eval_type eval_level, int n_digits) {{
    constexpr int UF = unroll_factor;
)",
                             k.getter_name);

    bool first = true;
    for (auto el : k.eval_levels) {
        std::cout << std::format("    {}if (eval_level == {}) {{\n", first ? "" : "} else ", eval_level_enum_name(el));
        emit_getter_branch_for_level(k, el, infos);
        first = false;
    }
    if (!k.eval_levels.empty())
        std::cout << "    }\n";

    std::cout << "    throw std::runtime_error(\"Unsupported eval_level/n_digits combination\");\n"
              << "}\n";
}

int main() {
    std::cout << R"(// Auto-generated by generate_aot_kernels. Do not edit.
#include <dmk.h>
#include <dmk/types.hpp>
#include <dmk/vector_kernels.hpp>
#include <sctl.hpp>

namespace dmk {
constexpr int unroll_factor = 3;

)";

    for (auto &k : all_kernels) {
        std::vector<CoeffsInfo> infos;

        for (auto el : k.eval_levels) {
            for (int digits = min_digits; digits <= max_digits; ++digits) {
                try {
                    pdmk_params p;
                    p.kernel = k.kernel;
                    p.n_dim = k.dim;
                    p.eps = std::pow(10, -digits);
                    p.eval_src = el;
                    p.eval_trg = el;
                    p.debug_flags = 0;
                    const double beta = dmk::util::calc_bandlimiting(p);
                    const auto coeffs = dmk::get_local_correction_coeffs<double>(k.kernel, k.dim, digits, beta);

                    CoeffsInfo info;
                    info.digits = digits;
                    info.beta = beta;
                    info.total_size = 0;
                    info.eval_level = el;
                    for (const auto &cvec : coeffs) {
                        info.sub_sizes.push_back(cvec.size());
                        info.total_size += cvec.size();
                    }

                    emit_coeffs_array(coeff_name(k, digits, el), coeffs, beta);
                    infos.push_back(std::move(info));
                } catch (std::exception &e) {
                    std::cerr << std::format("// Skipped {} digits={} eval_level={}: {}\n", k.getter_name, digits,
                                             eval_level_suffix(el), e.what());
                }
            }
        }

        emit_getter(k, infos);
    }

    // Emit explicit instantiations
    std::cout << "\n// Explicit instantiations\n";
    for (auto &k : all_kernels) {
        for (auto type : {"float", "double"}) {
            std::cout << std::format("template residual_evaluator_func<{0}>\n"
                                     "{1}<{0}, sctl::DefaultVecLen<{0}>()>(dmk_eval_type, int);\n",
                                     type, k.getter_name);
        }
    }

    std::cout << "\n} // namespace dmk\n";

    return 0;
}
