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

// ESP short-range residuals. Not dmk_ikernels: they reuse the scalar poly_all_pairs templates with
// FINUFFT-derived PSWF coefficients, so only the coefficient source differs. Overrides give each a
// distinct getter name. 3D also emits the range twin; 2D has none.
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
    // Stokeslet/Stresslet split into two sub-arrays (diag, offd), which share one compiled length.
    {DMK_STOKESLET, 3, {DMK_VELOCITY}, "esp_stokeslet", "stokeslet_3d_poly_all_pairs"},
    {DMK_STRESSLET, 3, {DMK_VELOCITY}, "esp_stresslet", "stresslet_3d_poly_all_pairs"},
};

// Yukawa ESP. 3D reuses the laplace_3d driver (+ range twin); 2D uses yukawa_2d (dense only).
static const std::vector<KernelDef> esp_yukawa = {
    {DMK_YUKAWA, 3, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}, "esp_yukawa", "laplace_3d_poly_all_pairs"},
    {DMK_YUKAWA, 2, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}, "esp_yukawa", "yukawa_2d_poly_all_pairs"},
};

// Yukawa in the DMK tree. lambda*bsize is a free run-time parameter, so the coefficient count is
// not a function of the digit count -- measured 5 to 18 at three digits over lambda -- and cannot be
// keyed off it. Enumerating n_coeffs the way ESP Yukawa does gives it the same compiled Horner
// length every other kernel gets, instead of a dynamic loop bound.
static const std::vector<KernelDef> dmk_yukawa = {
    {DMK_YUKAWA, 3, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}},
    {DMK_YUKAWA, 2, {DMK_POTENTIAL, DMK_POTENTIAL_GRAD}},
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

constexpr int min_digits = 2;
constexpr int max_digits = 12;

// =====================================================================
// CPU (host AOT) emission
// =====================================================================

// Yukawa ESP getters. Yukawa's residual has a free parameter (lambda), so its coeff count and
// values are unknown at generator time. We enumerate n_coeffs over the make_polyfit_abs_error range
// [3,31], keep N_DIGITS runtime (-1; it only gates the cheap transform_poly branch), and copy the
// runtime-computed coeffs into the matching precompiled branch. 3D reuses laplace_3d (single poly);
// 2D uses yukawa_2d (log-split [PA|PB], N_COEFFS_REG baked, N_COEFFS_LOG runtime).
constexpr int min_coeffs = 3;
constexpr int max_coeffs = 31;

// Every residual takes one of three shapes: a single polynomial, two of equal compiled length with
// the shorter zero-padded, or -- for 2D Yukawa's log split alone -- a dynamic-length log polynomial
// paired with a static one. Coefficients always come from the caller, fit at whatever beta it used,
// and the branch is picked by their run-time length.
enum class Shape { Single, PaddedPair, DynamicLogPair };

// Declarations and copies for one branch; `n` is the compiled length being matched.
std::string branch_prologue(Shape shape, int nc) {
    if (shape == Shape::Single)
        return std::format("            constexpr int NC0 = {};\n"
                           "            std::array<Real, NC0> cf{{}};\n"
                           "            std::copy_n(coeffs[0].data(), coeffs[0].size(), cf.data());\n",
                           nc);
    if (shape == Shape::PaddedPair)
        return std::format("            constexpr int NC = {};\n"
                           "            std::array<Real, 2 * NC> cf{{}};\n"
                           "            std::copy_n(coeffs[0].data(), coeffs[0].size(), cf.data());\n"
                           "            std::copy_n(coeffs[1].data(), coeffs[1].size(), cf.data() + NC);\n",
                           nc);
    return std::format("            constexpr int NC0 = {};\n"
                       "            const int n_log = static_cast<int>(coeffs[0].size());\n"
                       "            std::vector<Real> cf;\n"
                       "            cf.reserve(n_log + NC0);\n"
                       "            cf.insert(cf.end(), coeffs[0].begin(), coeffs[0].end());\n"
                       "            cf.insert(cf.end(), coeffs[1].begin(), coeffs[1].end());\n",
                       nc);
}

// Template length arguments, and the matching run-time ones the driver still takes.
std::pair<std::string, std::string> branch_lengths(Shape shape) {
    if (shape == Shape::Single)
        return {"NC0", "NC0"};
    if (shape == Shape::PaddedPair)
        return {"NC, NC", "NC, NC"};
    return {"-1, NC0", "n_log, NC0"};
}

// Which polynomial's length selects the branch: the single one, the longer of a padded pair, or the
// static half of the log split.
std::string dispatch_key(Shape shape) {
    if (shape == Shape::Single)
        return "static_cast<int>(coeffs.at(0).size())";
    if (shape == Shape::PaddedPair)
        return "static_cast<int>(std::max(coeffs.at(0).size(), coeffs.at(1).size()))";
    return "static_cast<int>(coeffs.at(1).size())";
}

void emit_branch(const KernelDef &k, dmk_eval_type el, Shape shape, int nc, bool ranges) {
    const auto [targs, rtargs] = branch_lengths(shape);
    const std::string body =
        ranges ? std::format("            return [=](Real rsc, Real cen, Real d2max, Real thresh2,\n"
                             "                       int n_src, const Real *r_src, const Real *charge,\n"
                             "                       const Real *normals, int n_ranges,\n"
                             "                       const int *range_starts, const int *range_lens,\n"
                             "                       int n_trg, const Real *r_trg, Real *pot,\n"
                             "                       const Real *q_trg, Real *pot_src) {{\n"
                             "                {0}_ranges<Real, MaxVecLen, -1, {1}, {2}>(\n"
                             "                    eval_level, n_digits, rsc, cen, d2max, thresh2, {3},\n"
                             "                    cf.data(), n_ranges, range_starts, range_lens, n_src,\n"
                             "                    r_src, charge, normals, n_trg, r_trg, pot, q_trg, pot_src, UF);\n"
                             "            }};\n",
                             func_name(k), targs, eval_level_enum_name(el), rtargs)
               : std::format("            return [=](Real rsc, Real cen, Real d2max, Real thresh2,\n"
                             "                       int n_src, const Real *r_src, const Real *charge,\n"
                             "                       const Real *normals, int n_trg, const Real *r_trg, Real *pot) {{\n"
                             "                {0}<Real, MaxVecLen, -1, {1}, {2}>(\n"
                             "                    eval_level, n_digits, rsc, cen, d2max, thresh2, {3},\n"
                             "                    cf.data(), n_src, r_src, charge, normals, n_trg, r_trg, pot, UF);\n"
                             "            }};\n",
                             func_name(k), targs, eval_level_enum_name(el), rtargs);
    out() << std::format("        if (n == {}) {{\n{}{}        }}\n", nc, branch_prologue(shape, nc), body);
}

void emit_getter(const KernelDef &k, Shape shape, bool ranges) {
    out() << std::format(R"(
template <class Real, int MaxVecLen>
{0}<Real> {1}{2}(dmk_eval_type eval_level, int n_digits,
                                  const std::vector<std::vector<Real>> &coeffs) {{
    constexpr int UF = unroll_factor;
    const int n = {3};
)",
                         ranges ? "residual_evaluator_range_func" : "residual_evaluator_func", getter_name(k),
                         ranges ? "_ranges" : "", dispatch_key(shape));
    bool first = true;
    for (auto el : k.eval_levels) {
        out() << std::format("    {}if (eval_level == {}) {{\n", first ? "" : "} else ", eval_level_enum_name(el));
        for (int nc = min_coeffs; nc <= max_coeffs; ++nc)
            emit_branch(k, el, shape, nc, ranges);
        first = false;
    }
    if (!k.eval_levels.empty())
        out() << "    }\n";
    out() << "    throw std::runtime_error(\"Unsupported eval_level, or n_coeffs outside the AOT range\");\n"
          << "}\n";
}

// =====================================================================
// Host AOT: one translation unit per getter, so the build compiles them in parallel. Each unit
// carries its own explicit instantiations.
// =====================================================================

// How many polynomials a kernel's residual splits into, measured once with the values discarded.
// Yukawa is stated rather than measured: its coefficients come from FourierData, not from
// get_local_correction_coeffs, and its count depends on lambda*bsize anyway.
int sub_count(const KernelDef &k) {
    for (int digits = min_digits; digits <= max_digits; ++digits) {
        try {
            pdmk_params p;
            p.kernel = k.kernel;
            p.n_dim = k.dim;
            p.eps = std::pow(10, -digits);
            p.eval_src = k.eval_levels.front();
            p.eval_trg = k.eval_levels.front();
            p.debug_flags = 0;
            const double beta = dmk::util::calc_bandlimiting(p);
            return static_cast<int>(dmk::get_local_correction_coeffs<double>(k.kernel, k.dim, digits, beta).size());
        } catch (const std::exception &) {
        }
    }
    throw std::runtime_error("no coefficient shape for " + getter_name(k));
}

Shape shape_of(const KernelDef &k) {
    if (k.kernel == DMK_YUKAWA)
        return k.dim == 2 ? Shape::DynamicLogPair : Shape::Single; // only 2D K0 splits off a log term
    return sub_count(k) == 2 ? Shape::PaddedPair : Shape::Single;
}

struct Unit {
    KernelDef k;
    bool ranges;
};

std::string unit_name(const Unit &u) {
    return std::format("{}_{}d{}", base_name(u.k), u.k.dim, u.ranges ? "_ranges" : "");
}

std::vector<Unit> host_units() {
    std::vector<Unit> units;
    for (const auto &table : {all_kernels, dmk_yukawa})
        for (const auto &k : table)
            units.push_back({k, false});
    // 3D ESP short-range also drives the range-list evaluator; nothing else does.
    for (const auto &table : {esp_baked, esp_yukawa})
        for (const auto &k : table) {
            units.push_back({k, false});
            if (k.dim == 3)
                units.push_back({k, true});
        }
    return units;
}

void emit_instantiations(const Unit &u) {
    const std::string ret = u.ranges ? "residual_evaluator_range_func" : "residual_evaluator_func";
    const std::string sfx = u.ranges ? "_ranges" : "";
    out() << "\n// Explicit instantiations\n";
    for (auto type : {"float", "double"})
        out() << std::format("template {0}<{1}>\n{2}{3}<{1}, sctl::DefaultVecLen<{1}>()>(dmk_eval_type, int,\n"
                             "    const std::vector<std::vector<{1}>> &);\n",
                             ret, type, getter_name(u.k), sfx);
}

void emit_host_unit(const Unit &u) {
    out() << "// Auto-generated by generate_aot_kernels. Do not edit.\n";
    out() << std::format("// Unit: {}\n", unit_name(u));
    out() << R"(#include <algorithm>
#include <array>
#include <dmk.h>
#include <dmk/types.hpp>
#include <dmk/vector_kernels.hpp>
#include <sctl.hpp>
#include <utility>
#include <vector>

namespace dmk {
constexpr int unroll_factor = 3;

)";

    emit_getter(u.k, shape_of(u.k), u.ranges);

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
