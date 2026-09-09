// generate_aot_kernels.cpp
//
// Generates the host AOT residual evaluators into src/aot/aot_kernels_*.cpp.
//
// The output is split into one translation unit per getter so the build can
// compile them in parallel; a single unit dominated total build wall time. Each
// unit carries its own coefficient tables and explicit instantiations, so no
// symbol is defined twice. Stale units in the output directory are removed
// first, since the build globs whatever it finds there.
//
// Usage. The default output path is relative, so bare invocation must be from the project
// root; --outdir=DIR names the destination explicitly and works from anywhere:
//   ./build/scripts/generate_aot_kernels
//   ./scripts/generate_aot_kernels --outdir=../src/aot     (from a build directory)

#include <cmath>
#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/esp.hpp>
#include <dmk/util.hpp>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

// Emission sink, repointed at each per-unit file in turn.
static std::ostream *g_os = &std::cout;
static std::ostream &out() { return *g_os; }

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

// ESP short-range residuals. Not dmk_ikernels: they reuse the scalar poly_all_pairs templates but
// with FINUFFT-derived PSWF coefficients (get_esp_correction_coeffs), so only the coefficient
// source differs. Overrides give each a distinct getter/coeff name. Laplace/Sqrt-Laplace counts +
// values are known at generator time (fixed sigma=1.35) -> baked per-digit tables, identical to
// the DMK-kernel mechanism. 3D also emits the range twin; 2D has none.
static const std::vector<KernelDef> esp_baked = {
    {DMK_LAPLACE, 2, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}, "esp_laplace", "laplace_2d_poly_all_pairs"},
    {DMK_LAPLACE, 3, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}, "esp_laplace", "laplace_3d_poly_all_pairs"},
    {DMK_SQRT_LAPLACE, 2, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}, "esp_sqrt_laplace", "sqrt_laplace_2d_poly_all_pairs"},
    {DMK_SQRT_LAPLACE, 3, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}, "esp_sqrt_laplace", "sqrt_laplace_3d_poly_all_pairs"},
    // Laplace-dipole reuses the Laplace residual profile (get_esp_correction_coeffs delegates to
    // get_local_correction_coeffs) via the dipole poly driver, which differentiates it.
    {DMK_LAPLACE_DIPOLE,
     3,
     {DMK_POTENTIAL, DMK_POTENTIAL_GRAD},
     "esp_laplace_dipole",
     "laplace_dipole_3d_poly_all_pairs"},
    // Stokeslet/Stresslet: their biharmonic residual is scale-invariant (get_esp_correction_coeffs
    // delegates to the cached bsize=1 get_local_correction_coeffs), so counts + values are fixed at
    // generator time and bake exactly like the scalars. Two coeff sub-arrays (diag, offd); velocity.
    {DMK_STOKESLET, 3, {DMK_VELOCITY}, "esp_stokeslet", "stokeslet_3d_poly_all_pairs"},
    {DMK_STRESSLET, 3, {DMK_VELOCITY}, "esp_stresslet", "stresslet_3d_poly_all_pairs"},
};

// Yukawa ESP: free parameter lambda makes coeff count/values runtime -> enumerate n_coeffs in
// [3,31] and pipe runtime-computed coeffs into the matching branch. 3D reuses laplace_3d (+range
// twin); 2D uses yukawa_2d (dense only).
static const std::vector<KernelDef> esp_yukawa = {
    {DMK_YUKAWA, 3, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}, "esp_yukawa", "laplace_3d_poly_all_pairs"},
    {DMK_YUKAWA, 2, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}, "esp_yukawa", "yukawa_2d_poly_all_pairs"},
};

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
    std::vector<size_t> sub_sizes;           // size of each sub-array
    size_t total_size;                       // sum of sub_sizes
    std::vector<std::vector<double>> values; // per-sub-array coefficients (used by CUDA tag emission)
    dmk_eval_type eval_level;
};

std::string eval_level_enum_name(dmk_eval_type el) {
    switch (el) {
    case DMK_POTENTIAL:
        return "DMK_POTENTIAL";
    case DMK_POTENTIAL_GRAD:
        return "DMK_POTENTIAL_GRAD";
    case DMK_VELOCITY:
        return "DMK_VELOCITY";
    }
    return "DMK_POTENTIAL";
}

void emit_coeffs_array(const std::string &name, const std::vector<std::vector<double>> &coeffs, double beta) {
    out() << std::format("// beta: {}\n", beta);
    out() << std::format("constexpr double {}[] = {{", name);
    int count = 0;
    for (const auto &cvec : coeffs) {
        for (size_t i = 0; i < cvec.size(); ++i) {
            if (count > 0)
                out() << ",";
            if (count % 4 == 0)
                out() << "\n    ";
            out() << std::format(" {:.17e}", cvec[i]);
            count++;
        }
    }
    out() << "\n};\n\n";
}

std::string coeff_name(const KernelDef &k, int digits, dmk_eval_type el) {
    return std::format("{}_{}d_{}_{}", base_name(k), k.dim, dmk::util::to_string(el), digits);
}

// =====================================================================
// CPU (host AOT) emission
// =====================================================================

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

        out() << std::format(
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
    out() << std::format(R"(
template <class Real, int MaxVecLen>
residual_evaluator_func<Real> {}(dmk_eval_type eval_level, int n_digits) {{
    constexpr int UF = unroll_factor;
)",
                         getter_name(k));

    bool first = true;
    for (auto el : k.eval_levels) {
        out() << std::format("    {}if (eval_level == {}) {{\n", first ? "" : "} else ", eval_level_enum_name(el));
        emit_getter_branch_for_level(k, el, infos);
        first = false;
    }
    if (!k.eval_levels.empty())
        out() << "    }\n";

    out() << "    throw std::runtime_error(\"Unsupported eval_level/n_digits combination\");\n"
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

        out() << std::format("        if (n_digits <= {}) {{\n"
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
    out() << std::format(R"(
template <class Real, int MaxVecLen>
residual_evaluator_range_func<Real> {}_ranges(dmk_eval_type eval_level, int n_digits) {{
    constexpr int UF = unroll_factor;
)",
                         getter_name(k));

    bool first = true;
    for (auto el : k.eval_levels) {
        out() << std::format("    {}if (eval_level == {}) {{\n", first ? "" : "} else ", eval_level_enum_name(el));
        emit_getter_branch_for_level_ranges(k, el, infos);
        first = false;
    }
    if (!k.eval_levels.empty())
        out() << "    }\n";

    out() << "    throw std::runtime_error(\"Unsupported eval_level/n_digits combination\");\n"
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
        out() << std::format(
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
        out() << std::format(
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
        out() << std::format(
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
    out() << std::format("\ntemplate <class Real, int MaxVecLen>\n"
                         "{0}<Real> {1}{2}(dmk_eval_type eval_level, int n_digits, const Real *coeffs, int n_coeffs,\n"
                         "                 int n_coeffs_log) {{\n"
                         "    constexpr int UF = unroll_factor;\n"
                         "    (void)n_coeffs_log;\n",
                         ranges ? "residual_evaluator_range_func" : "residual_evaluator_func", getter_name(k),
                         ranges ? "_ranges" : "");

    bool first = true;
    for (auto el : k.eval_levels) {
        out() << std::format("    {}if (eval_level == {}) {{\n", first ? "" : "} else ", eval_level_enum_name(el));
        for (int nc = min_coeffs; nc <= max_coeffs; ++nc)
            emit_yukawa_branch(k, el, nc, ranges);
        first = false;
    }
    if (!k.eval_levels.empty())
        out() << "    }\n";
    out() << "    throw std::runtime_error(\"ESP Yukawa: n_coeffs outside AOT range [3,31]\");\n"
          << "}\n";
}

// =====================================================================
// Host AOT: one translation unit per getter, so the build compiles them in
// parallel. Each unit re-derives the coefficient tables it references and
// carries its own explicit instantiations.
// =====================================================================

enum class Kind { Dmk, EspBaked, EspYukawa };

struct Unit {
    KernelDef k;
    Kind kind;
    bool ranges;
};

std::string unit_name(const Unit &u) {
    return std::format("{}_{}d{}", base_name(u.k), u.k.dim, u.ranges ? "_ranges" : "");
}

std::vector<Unit> host_units() {
    std::vector<Unit> units;
    for (const auto &k : all_kernels)
        units.push_back({k, Kind::Dmk, false});
    for (const auto &k : esp_baked) {
        units.push_back({k, Kind::EspBaked, false});
        if (k.dim == 3)
            units.push_back({k, Kind::EspBaked, true});
    }
    for (const auto &k : esp_yukawa) {
        units.push_back({k, Kind::EspYukawa, false});
        if (k.dim == 3)
            units.push_back({k, Kind::EspYukawa, true});
    }
    return units;
}

// Coefficient tables for one kernel over every eval_level and digit count. DMK kernels take beta
// from calc_bandlimiting; baked ESP kernels use the fixed sigma=1.35 derivation (esp.hpp) but record
// beta as 0 in the generated comment, since it is not the bandlimit the DMK path reports.
std::vector<CoeffsInfo> collect_coeffs(const KernelDef &k, Kind kind) {
    std::vector<CoeffsInfo> infos;
    for (auto el : k.eval_levels) {
        for (int digits = min_digits; digits <= max_digits; ++digits) {
            try {
                double beta = 0.0;
                std::vector<std::vector<double>> coeffs;
                if (kind == Kind::EspBaked) {
                    beta = dmk::esp_beta_from_P(1.35, dmk::esp_P_from_eps(std::pow(10.0, -digits), 1.35, k.dim));
                    coeffs = dmk::get_esp_correction_coeffs<double>(k.kernel, 0.0, 0.0, k.dim, digits, beta);
                } else {
                    pdmk_params p;
                    p.kernel = k.kernel;
                    p.n_dim = k.dim;
                    p.eps = std::pow(10, -digits);
                    p.eval_src = el;
                    p.eval_trg = el;
                    p.debug_flags = 0;
                    beta = dmk::util::calc_bandlimiting(p);
                    coeffs = dmk::get_local_correction_coeffs<double>(k.kernel, k.dim, digits, beta);
                }

                CoeffsInfo info;
                info.digits = digits;
                info.beta = kind == Kind::EspBaked ? 0.0 : beta;
                info.total_size = 0;
                info.eval_level = el;
                for (const auto &cvec : coeffs) {
                    info.sub_sizes.push_back(cvec.size());
                    info.total_size += cvec.size();
                }
                info.values = std::move(coeffs);
                infos.push_back(std::move(info));
            } catch (std::exception &e) {
                std::cerr << std::format("// Skipped {} digits={} eval_level={}: {}\n", getter_name(k), digits,
                                         dmk::util::to_string(el), e.what());
            }
        }
    }
    return infos;
}

void emit_instantiations(const Unit &u) {
    const std::string ret = u.ranges ? "residual_evaluator_range_func" : "residual_evaluator_func";
    const std::string sfx = u.ranges ? "_ranges" : "";
    out() << "\n// Explicit instantiations\n";
    for (auto type : {"float", "double"}) {
        const std::string args = u.kind == Kind::EspYukawa
                                     ? std::format("dmk_eval_type, int, const {} *, int, int", type)
                                     : "dmk_eval_type, int";
        out() << std::format("template {0}<{1}>\n{2}{3}<{1}, sctl::DefaultVecLen<{1}>()>({4});\n", ret, type,
                             getter_name(u.k), sfx, args);
    }
}

void emit_host_unit(const Unit &u) {
    out() << "// Auto-generated by generate_aot_kernels. Do not edit.\n";
    out() << std::format("// Unit: {}\n", unit_name(u));
    out() << R"(#include <dmk.h>
#include <dmk/types.hpp>
#include <dmk/vector_kernels.hpp>
#include <sctl.hpp>
#include <utility>
#include <vector>

namespace dmk {
constexpr int unroll_factor = 3;

)";

    if (u.kind == Kind::EspYukawa) {
        emit_getter_yukawa(u.k, u.ranges);
    } else {
        const auto infos = collect_coeffs(u.k, u.kind);
        for (const auto &info : infos)
            emit_coeffs_array(coeff_name(u.k, info.digits, info.eval_level), info.values, info.beta);
        if (u.ranges)
            emit_getter_ranges(u.k, infos);
        else
            emit_getter(u.k, infos);
    }

    emit_instantiations(u);
    out() << "\n} // namespace dmk\n";
}

// =====================================================================
// Driver
// =====================================================================

int main(int argc, char **argv) {
    std::filesystem::path outdir = "src/aot";
    bool outdir_given = false;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg.starts_with("--outdir=")) {
            outdir = std::string(arg.substr(9));
            outdir_given = true;
        } else {
            std::cerr << std::format("Unknown argument: {}\n", arg);
            std::cerr << "Usage: generate_aot_kernels [--outdir=DIR]\n";
            return 1;
        }
    }

    // The default output path is relative, so without --outdir a wrong working directory would
    // scatter files somewhere unnoticed. An explicit --outdir already says where to write.
    if (!outdir_given && !std::filesystem::exists("CMakeLists.txt")) {
        std::cerr << "generate_aot_kernels: run from the project root, or pass --outdir=DIR "
                     "(no CMakeLists.txt in the working directory)\n";
        return 1;
    }

    std::filesystem::create_directories(outdir);
    // Drop stale units, so a getter removed from the tables above cannot linger in the
    // directory and get picked up by the build's glob.
    for (const auto &entry : std::filesystem::directory_iterator(outdir)) {
        const auto name = entry.path().filename().string();
        if (name.starts_with("aot_kernels_") && entry.path().extension() == ".cpp")
            std::filesystem::remove(entry.path());
    }

    for (const auto &u : host_units()) {
        const auto path = outdir / std::format("aot_kernels_{}.cpp", unit_name(u));
        std::ofstream f(path);
        if (!f) {
            std::cerr << std::format("generate_aot_kernels: cannot write {}\n", path.string());
            return 1;
        }
        g_os = &f;
        emit_host_unit(u);
        g_os = &std::cout;
        if (!f) {
            std::cerr << std::format("generate_aot_kernels: write failed for {}\n", path.string());
            return 1;
        }
        std::cerr << std::format("wrote {}\n", path.string());
    }

    return 0;
}
