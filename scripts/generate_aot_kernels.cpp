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
    int dim;
    std::vector<dmk_eval_type> eval_levels;
    // Overrides for pseudo-kernels (e.g. ESP) that reuse another kernel's
    // poly_all_pairs template but need distinct getter/coeff names. Empty =>
    // derive from the canonical kernel name.
    std::string name_override = ""; // used for getter + coeff names
    std::string func_override = ""; // poly_all_pairs template to call
};

// clang-format off
static const std::vector<KernelDef> all_kernels = {
    {DMK_LAPLACE,        2, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}},
    {DMK_LAPLACE,        3, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}},
    {DMK_SQRT_LAPLACE,   2, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}},
    {DMK_SQRT_LAPLACE,   3, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}},
    {DMK_STOKESLET,      3, {DMK_VELOCITY}},
    {DMK_STRESSLET,      3, {DMK_VELOCITY}},
    {DMK_LAPLACE_DIPOLE, 3, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}},
};
// clang-format on

// All generated names derive from the canonical kernel name (dmk::util::to_string)
// unless name_override is set, so the generator, the poly_all_pairs templates, and
// the getters consumed by aot_evaluator.cpp share one nomenclature.
std::string base_name(const KernelDef &k) {
    return k.name_override.empty() ? std::string(dmk::util::to_string(k.kernel)) : k.name_override;
}

std::string func_name(const KernelDef &k) {
    if (!k.func_override.empty())
        return k.func_override;
    return std::format("{}_{}d_poly_all_pairs", dmk::util::to_string(k.kernel), k.dim);
}

std::string getter_name(const KernelDef &k) { return std::format("get_{}_{}d_kernel", base_name(k), k.dim); }

constexpr int min_digits = 2;
constexpr int max_digits = 12;

struct CoeffsInfo {
    int digits;
    double beta;
    std::vector<size_t> sub_sizes; // size of each sub-array
    size_t total_size;             // sum of sub_sizes
    dmk_eval_type eval_level;
};

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

std::string coeff_name(const KernelDef &k, int digits, dmk_eval_type el) {
    return std::format("{}_{}d_{}_{}", base_name(k), k.dim, dmk::util::to_string(el), digits);
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
            info.digits, info.digits, info.total_size, nc_decls, cn, func_name(k), nc_args, eval_level_enum_name(el),
            nc_args);
    }
}

void emit_getter(const KernelDef &k, const std::vector<CoeffsInfo> &infos) {
    std::cout << std::format(R"(
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> {}(dmk_eval_type eval_level, int n_digits) {{
    constexpr int UF = unroll_factor;
)",
                             getter_name(k));

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

void emit_getter_branch_for_level_ranges(const KernelDef &k, dmk_eval_type el, const std::vector<CoeffsInfo> &infos) {
    for (const auto &info : infos) {
        if (info.eval_level != el)
            continue;
        const auto cn = coeff_name(k, info.digits, el);

        std::string nc_decls, nc_args;
        for (size_t i = 0; i < info.sub_sizes.size(); ++i) {
            if (i > 0) {
                nc_decls += "\n";
                nc_args += ", ";
            }
            nc_decls += std::format("            constexpr int NC{} = {};", i, info.sub_sizes[i]);
            nc_args += std::format("NC{}", i);
        }

        std::cout << std::format("        if (n_digits <= {}) {{\n"
                                 "            constexpr int ND = {}, NC_TOTAL = {};\n"
                                 "{}\n"
                                 "            std::array<Real, NC_TOTAL> coeffs;\n"
                                 "            std::copy_n({}, NC_TOTAL, coeffs.data());\n"
                                 "            return [=](Real rsc, Real cen, Real d2max, Real thresh2,\n"
                                 "                       int n_src, const Real *r_src, const Real *charge,\n"
                                 "                       const Real *normals, int n_ranges,\n"
                                 "                       const int *range_starts, const int *range_lens,\n"
                                 "                       int n_trg, const Real *r_trg, Real *pot,\n"
                                 "                       const Real *q_trg, Real *pot_src) {{\n"
                                 "                {}_ranges<Real, MaxVecLen, ND, {}, {}>(\n"
                                 "                    eval_level, ND, rsc, cen, d2max, thresh2, {},\n"
                                 "                    coeffs.data(), n_ranges, range_starts, range_lens, n_src,\n"
                                 "                    r_src, charge, normals, n_trg, r_trg, pot, q_trg, pot_src, UF);\n"
                                 "            }};\n"
                                 "        }}\n",
                                 info.digits, info.digits, info.total_size, nc_decls, cn, func_name(k), nc_args,
                                 eval_level_enum_name(el), nc_args);
    }
}

void emit_getter_ranges(const KernelDef &k, const std::vector<CoeffsInfo> &infos) {
    std::cout << std::format(R"(
template <class Real, int MaxVecLen>
residual_evaluator_range_func<Real> {}_ranges(dmk_eval_type eval_level, int n_digits) {{
    constexpr int UF = unroll_factor;
)",
                             getter_name(k));

    bool first = true;
    for (auto el : k.eval_levels) {
        std::cout << std::format("    {}if (eval_level == {}) {{\n", first ? "" : "} else ", eval_level_enum_name(el));
        emit_getter_branch_for_level_ranges(k, el, infos);
        first = false;
    }
    if (!k.eval_levels.empty())
        std::cout << "    }\n";

    std::cout << "    throw std::runtime_error(\"Unsupported eval_level/n_digits combination\");\n"
              << "}\n";
}

// Yukawa ESP getters. Yukawa's residual has a free parameter (lambda), so its coeff count and
// values are unknown at generator time. We enumerate n_coeffs over the make_polyfit_abs_error range
// [3,31], keep N_DIGITS runtime (-1; it only gates the cheap transform_poly branch), and copy the
// runtime-computed coeffs into the matching precompiled branch. 3D reuses laplace_3d (single poly);
// 2D uses yukawa_2d (log-split [PA|PB], N_COEFFS_REG baked, N_COEFFS_LOG runtime).
constexpr int min_coeffs = 3;
constexpr int max_coeffs = 31;

void emit_yukawa_branch(const KernelDef &k, dmk_eval_type el, int nc, bool ranges) {
    const std::string ev = eval_level_enum_name(el);
    if (ranges) {
        std::cout << std::format(
            "        if (n_coeffs == {0}) {{\n"
            "            constexpr int NC0 = {0};\n"
            "            std::vector<Real> cf(coeffs, coeffs + NC0);\n"
            "            return [cf = std::move(cf), eval_level, n_digits](\n"
            "                       Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,\n"
            "                       const Real *charge, const Real *normals, int n_ranges, const int *range_starts,\n"
            "                       const int *range_lens, int n_trg, const Real *r_trg, Real *pot,\n"
            "                       const Real *q_trg, Real *pot_src) {{\n"
            "                {1}_ranges<Real, MaxVecLen, -1, NC0, {2}>(\n"
            "                    eval_level, n_digits, rsc, cen, d2max, thresh2, NC0, cf.data(), n_ranges,\n"
            "                    range_starts, range_lens, n_src, r_src, charge, normals, n_trg, r_trg, pot,\n"
            "                    q_trg, pot_src, UF);\n"
            "            }};\n"
            "        }}\n",
            nc, func_name(k), ev);
    } else if (k.dim == 3) {
        std::cout << std::format(
            "        if (n_coeffs == {0}) {{\n"
            "            constexpr int NC0 = {0};\n"
            "            std::vector<Real> cf(coeffs, coeffs + NC0);\n"
            "            return [cf = std::move(cf), eval_level, n_digits](\n"
            "                       Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,\n"
            "                       const Real *charge, const Real *normals, int n_trg, const Real *r_trg, Real *pot) "
            "{{\n"
            "                {1}<Real, MaxVecLen, -1, NC0, {2}>(\n"
            "                    eval_level, n_digits, rsc, cen, d2max, thresh2, NC0, cf.data(), n_src, r_src,\n"
            "                    charge, normals, n_trg, r_trg, pot, UF);\n"
            "            }};\n"
            "        }}\n",
            nc, func_name(k), ev);
    } else { // 2D log-split yukawa_2d: bake N_COEFFS_REG, keep N_COEFFS_LOG runtime
        std::cout << std::format(
            "        if (n_coeffs == {0}) {{\n"
            "            constexpr int NC0 = {0};\n"
            "            std::vector<Real> cf(coeffs, coeffs + n_coeffs_log + NC0);\n"
            "            return [cf = std::move(cf), eval_level, n_digits, n_coeffs_log](\n"
            "                       Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,\n"
            "                       const Real *charge, const Real *normals, int n_trg, const Real *r_trg, Real *pot) "
            "{{\n"
            "                {1}<Real, MaxVecLen, -1, -1, NC0, {2}>(\n"
            "                    eval_level, n_digits, rsc, cen, d2max, thresh2, n_coeffs_log, NC0, cf.data(),\n"
            "                    n_src, r_src, charge, normals, n_trg, r_trg, pot, UF);\n"
            "            }};\n"
            "        }}\n",
            nc, func_name(k), ev);
    }
}

void emit_getter_yukawa(const KernelDef &k, bool ranges) {
    std::cout << std::format(
        "\ntemplate <class Real, int MaxVecLen>\n"
        "{0}<Real> {1}{2}(dmk_eval_type eval_level, int n_digits, const Real *coeffs, int n_coeffs,\n"
        "                 int n_coeffs_log) {{\n"
        "    constexpr int UF = unroll_factor;\n"
        "    (void)n_coeffs_log;\n",
        ranges ? "residual_evaluator_range_func" : "residual_evaluator_func", getter_name(k), ranges ? "_ranges" : "");

    bool first = true;
    for (auto el : k.eval_levels) {
        std::cout << std::format("    {}if (eval_level == {}) {{\n", first ? "" : "} else ", eval_level_enum_name(el));
        for (int nc = min_coeffs; nc <= max_coeffs; ++nc)
            emit_yukawa_branch(k, el, nc, ranges);
        first = false;
    }
    if (!k.eval_levels.empty())
        std::cout << "    }\n";
    std::cout << "    throw std::runtime_error(\"ESP Yukawa: n_coeffs outside AOT range [3,31]\");\n"
              << "}\n";
}

int main() {
    std::cout << R"(// Auto-generated by generate_aot_kernels. Do not edit.
#include <dmk.h>
#include <dmk/types.hpp>
#include <dmk/vector_kernels.hpp>
#include <sctl.hpp>
#include <utility>
#include <vector>

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
                    std::cerr << std::format("// Skipped {} digits={} eval_level={}: {}\n", getter_name(k), digits,
                                             dmk::util::to_string(el), e.what());
                }
            }
        }

        emit_getter(k, infos);
    }

    // ESP short-range residuals. Not dmk_ikernels: they reuse the scalar poly_all_pairs templates but
    // with FINUFFT-derived PSWF coefficients (get_esp_correction_coeffs), so only the coefficient
    // source differs. Overrides give each a distinct getter/coeff name. Laplace/Sqrt-Laplace counts +
    // values are known at generator time (fixed sigma=1.35) -> baked per-digit tables, identical to
    // the DMK-kernel mechanism. 3D also emits the range twin; 2D has none.
    const std::vector<KernelDef> esp_baked = {
        {DMK_LAPLACE, 2, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}, "esp_laplace", "laplace_2d_poly_all_pairs"},
        {DMK_LAPLACE, 3, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}, "esp_laplace", "laplace_3d_poly_all_pairs"},
        {DMK_SQRT_LAPLACE,
         2,
         {DMK_POTENTIAL, DMK_POTENTIAL_GRAD},
         "esp_sqrt_laplace",
         "sqrt_laplace_2d_poly_all_pairs"},
        {DMK_SQRT_LAPLACE,
         3,
         {DMK_POTENTIAL, DMK_POTENTIAL_GRAD},
         "esp_sqrt_laplace",
         "sqrt_laplace_3d_poly_all_pairs"},
    };
    for (const auto &k : esp_baked) {
        std::vector<CoeffsInfo> infos;
        for (auto el : k.eval_levels) {
            for (int digits = min_digits; digits <= max_digits; ++digits) {
                try {
                    const auto coeffs = dmk::get_esp_correction_coeffs<double>(k.kernel, 0.0, 0.0, k.dim, digits, 1.35);

                    CoeffsInfo info;
                    info.digits = digits;
                    info.beta = 0.0;
                    info.total_size = 0;
                    info.eval_level = el;
                    for (const auto &cvec : coeffs) {
                        info.sub_sizes.push_back(cvec.size());
                        info.total_size += cvec.size();
                    }

                    emit_coeffs_array(coeff_name(k, digits, el), coeffs, 0.0);
                    infos.push_back(std::move(info));
                } catch (std::exception &e) {
                    std::cerr << std::format("// Skipped {} digits={} eval_level={}: {}\n", getter_name(k), digits,
                                             dmk::util::to_string(el), e.what());
                }
            }
        }
        emit_getter(k, infos);
        if (k.dim == 3)
            emit_getter_ranges(k, infos);
    }

    // Yukawa ESP: free parameter lambda makes coeff count/values runtime -> enumerate n_coeffs in
    // [3,31] and pipe runtime-computed coeffs into the matching branch. 3D reuses laplace_3d (+range
    // twin); 2D uses yukawa_2d (dense only).
    const std::vector<KernelDef> esp_yukawa = {
        {DMK_YUKAWA, 3, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}, "esp_yukawa", "laplace_3d_poly_all_pairs"},
        {DMK_YUKAWA, 2, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}, "esp_yukawa", "yukawa_2d_poly_all_pairs"},
    };
    for (const auto &k : esp_yukawa) {
        emit_getter_yukawa(k, false);
        if (k.dim == 3)
            emit_getter_yukawa(k, true);
    }

    // Emit explicit instantiations
    std::cout << "\n// Explicit instantiations\n";
    for (auto &k : all_kernels) {
        for (auto type : {"float", "double"}) {
            std::cout << std::format("template residual_evaluator_func<{0}>\n"
                                     "{1}<{0}, sctl::DefaultVecLen<{0}>()>(dmk_eval_type, int);\n",
                                     type, getter_name(k));
        }
    }
    for (const auto &k : esp_baked) {
        for (auto type : {"float", "double"}) {
            std::cout << std::format("template residual_evaluator_func<{0}>\n"
                                     "{1}<{0}, sctl::DefaultVecLen<{0}>()>(dmk_eval_type, int);\n",
                                     type, getter_name(k));
            if (k.dim == 3)
                std::cout << std::format("template residual_evaluator_range_func<{0}>\n"
                                         "{1}_ranges<{0}, sctl::DefaultVecLen<{0}>()>(dmk_eval_type, int);\n",
                                         type, getter_name(k));
        }
    }
    for (const auto &k : esp_yukawa) {
        for (auto type : {"float", "double"}) {
            std::cout << std::format(
                "template residual_evaluator_func<{0}>\n"
                "{1}<{0}, sctl::DefaultVecLen<{0}>()>(dmk_eval_type, int, const {0} *, int, int);\n",
                type, getter_name(k));
            if (k.dim == 3)
                std::cout << std::format(
                    "template residual_evaluator_range_func<{0}>\n"
                    "{1}_ranges<{0}, sctl::DefaultVecLen<{0}>()>(dmk_eval_type, int, const {0} *, int, int);\n",
                    type, getter_name(k));
        }
    }

    std::cout << "\n} // namespace dmk\n";

    return 0;
}
