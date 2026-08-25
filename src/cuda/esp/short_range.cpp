#include "short_range.hpp"
#include "sort.hpp"
#include "state.hpp"
#include "support.hpp"

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "../pt/launchers.hpp"

#include <dmk/cuda/esp_sr_kernelargs.hpp>
#include <dmk/direct.hpp>
#include <dmk/error.hpp>
#include <dmk/util.hpp>

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda::esp {
namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

// Hand-tuned launch geometry, baked into the module.
constexpr int kBlockSize = 128; // 4 warps
constexpr int kTileWidth = 32;  // = warpSize; one warp per source/target tile

// The dense path keeps out_dim accumulators per target slot, so register pressure scales with both.
int targets_per_thread(int out_dim) { return out_dim <= 1 ? 3 : (out_dim <= 3 ? 2 : 1); }

std::string fnv1a_hex(const std::string &text) {
    std::uint64_t h = 14695981039346656037ull;
    for (unsigned char c : text) {
        h ^= c;
        h *= 1099511628211ull;
    }
    std::ostringstream ss;
    ss << std::hex << h;
    return ss.str();
}

// The horner_const contract from poly_evaluators_device.hpp. Stokeslet and Stresslet carry two packs.
template <typename Real>
std::string emit_coeff_struct(const char *name, const std::vector<Real> &coeffs) {
    std::ostringstream ss;
    ss << std::setprecision(std::numeric_limits<Real>::max_digits10);
    ss << "struct " << name << " {\n";
    ss << "    static constexpr int size = " << coeffs.size() << ";\n";
    ss << "    __device__ static constexpr Real at(int i) {\n";
    ss << "        constexpr Real data[size] = {\n";
    for (std::size_t i = 0; i < coeffs.size(); ++i)
        ss << "            Real{" << coeffs[i] << "}" << (i + 1 < coeffs.size() ? "," : "") << "\n";
    ss << "        };\n        return data[i];\n    }\n};\n\n";
    return ss.str();
}

// ESP reuses the DMK evaluators unchanged; only the coefficients differ. Yukawa 3D reuses Laplace
// 3D because both are P(r)/r. See the esp_baked table in generate_aot_kernels.cpp.
const char *evaluator_family(dmk_ikernel kernel) {
    switch (kernel) {
    case DMK_LAPLACE:
    case DMK_YUKAWA:
        return "LaplacePolyEvaluator3DCuda";
    case DMK_SQRT_LAPLACE:
        return "SqrtLaplacePolyEvaluator3DCuda";
    case DMK_LAPLACE_DIPOLE:
        return "LaplaceDipolePolyEvaluator3DCuda";
    case DMK_STOKESLET:
        return "StokesletPolyEvaluator3DCuda";
    case DMK_STRESSLET:
        return "StressletPolyEvaluator3DCuda";
    default:
        break;
    }
    throw std::runtime_error("esp::short_range: unsupported kernel");
}

// Stokeslet/Stresslet fix their output dim.
bool takes_eval_level(dmk_ikernel kernel) { return kernel != DMK_STOKESLET && kernel != DMK_STRESSLET; }

int eval_level_for(dmk_eval_type ev) {
    if (ev == DMK_POTENTIAL)
        return 1;
    if (ev == DMK_POTENTIAL_GRAD)
        return 2;
    throw api_error(DMK_ERR_INVALID_ARGUMENT, "esp::short_range: unsupported eval_type");
}

} // namespace

// The residual polynomial comes from get_esp_correction_coeffs, the same function the CPU
// evaluators use, and is baked into the NVRTC module as literals.
template <typename Real>
void short_range_gpu(GpuState &gpu, int n, const Real *d_pos_aos, const Real *d_charges, Real *d_pot, Real *d_gx,
                     Real *d_gy, Real *d_gz) {
    const int nc = gpu.nc;
    const KernelDims &dims = gpu.dims;
    const int out_dim = dims.out_dim;
    const int ncells = nc * nc * nc;

    int *d_cell_start = nullptr;
    int *d_orig = nullptr;
    Real *d_xs = nullptr, *d_ys = nullptr, *d_zs = nullptr, *d_qs = nullptr;
    build_cell_list_gpu<Real>(gpu, n, nc, dims.charge_dim, d_pos_aos, d_charges, &d_cell_start, &d_orig, &d_xs, &d_ys,
                              &d_zs, &d_qs);

    ensure_capacity(gpu.d_scratch_pg, gpu.scratch_pg_cap, std::size_t(out_dim) * n * sizeof(Real));
    Real *d_pg_sorted = reinterpret_cast<Real *>(gpu.d_scratch_pg);

    // Poly variable: R^2 for 3D Sqrt-Laplace, R otherwise. Both map [0, r_c] onto [-1, 1]; matches
    // the CPU's r2_var switch.
    const bool r2_var = (gpu.kernel == DMK_SQRT_LAPLACE);
    const Real rsc = r2_var ? Real(2.0 / (gpu.r_c * gpu.r_c)) : Real(2.0 / gpu.r_c);
    const Real cen = r2_var ? Real(-1.0) : Real(-0.5 * gpu.r_c);
    const Real r_c_sq = Real(gpu.r_c * gpu.r_c);

    const bool pruned = (gpu.strategy != GpuSrStrategy::Dense);

    // Sized from the worst-case cell population, which costs a host-device sync.
    int max_tiles = 0;
    std::size_t shared_bytes = 0;
    if (pruned) {
        if (gpu.pruned_max_tiles_cache_n != n) {
            const int max_pop = max_cell_population(d_cell_start, ncells, gpu.stream);
            const int tiles_per_cell = (max_pop + kTileWidth - 1) / kTileWidth;
            gpu.pruned_max_tiles_cache = (27 * tiles_per_cell * 5) / 4; // +25% margin
            gpu.pruned_max_tiles_cache_n = n;
        }
        max_tiles = gpu.pruned_max_tiles_cache;
        shared_bytes = std::size_t(max_tiles) * (9 * sizeof(Real) + 2 * sizeof(int));
    }
    const int strategy = static_cast<int>(gpu.strategy);

    const bool prune_stats = pruned && util::env_is_set("DMK_ESP_PRUNE_STATS");
    unsigned long long *d_prune_stats = nullptr;
    if (prune_stats) {
        ensure_capacity(gpu.d_scratch_prune_stats, gpu.scratch_prune_stats_cap, 4 * sizeof(unsigned long long));
        d_prune_stats = reinterpret_cast<unsigned long long *>(gpu.d_scratch_prune_stats);
        cudaMemsetAsync(d_prune_stats, 0, 4 * sizeof(unsigned long long), gpu.stream);
    }

    // Yukawa is the only kernel for which fparam and r_c matter.
    const std::vector<std::vector<Real>> coeffs =
        get_esp_correction_coeffs<Real>(gpu.kernel, gpu.fparam, gpu.r_c, /*n_dim=*/3, gpu.n_digits, gpu.beta);
    if (coeffs.empty() || coeffs[0].empty())
        throw std::runtime_error("short_range_gpu: empty coefficient set");

    std::string coeff_struct;
    std::vector<std::string> targs;
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        const std::string name = "Coeff" + std::to_string(i);
        coeff_struct += emit_coeff_struct<Real>(name.c_str(), coeffs[i]);
        targs.push_back(name);
    }
    if (takes_eval_level(gpu.kernel))
        targs.push_back(std::to_string(eval_level_for(gpu.eval_type)));

    std::string evaluator_expr = std::string(evaluator_family(gpu.kernel)) + "<";
    for (std::size_t i = 0; i < targs.size(); ++i)
        evaluator_expr += (i ? ", " : "") + targs[i];
    evaluator_expr += ">";

    // JitKey::to_string does not hash the source text, so baked coefficients reach the cache key
    // through the name; anything else that changes the code goes into key.params.
    static JitCache cache;
    const std::string kernel_name = "EspShortRangeKernel_" + fnv1a_hex(std::string(jit_real_name<Real>()) + "|" +
                                                                       evaluator_expr + "|" + coeff_struct);

    JitKey key;
    key.name = kernel_name;
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params["BLOCK_SIZE"] = kBlockSize;
    key.params["TILE_WIDTH"] = kTileWidth;
    key.params["TARGETS_PER_THREAD"] = targets_per_thread(out_dim);
    key.params["STRATEGY"] = strategy;
    key.params["PRUNE_STATS"] = prune_stats ? 1 : 0;
    // Laplace and Laplace-dipole bake identical coefficients but need different device code, so the
    // kernel identity cannot ride on the name hash alone.
    key.params["KERNEL"] = static_cast<int>(gpu.kernel);

    const std::shared_ptr<jit::JitKernel> kernel = cache.get_kernel_from_source(key, [&] {
        const std::string prelude = "#define DMK_ESP_SR_KERNEL_NAME " + kernel_name + "\n\n" + coeff_struct +
                                    "#define DMK_ESP_SR_EVALUATOR " + evaluator_expr + "\n\n";
        return pt::make_stage_source("esp/short_range.cu", key, prelude, "EspShortRange");
    });

    if (shared_bytes > 0) {
        // The per-tile AABB table grows with particles-per-cell, and shares the device budget with
        // the kernel's static __shared__.
        const std::size_t static_bytes = pt::static_smem(*kernel);
        const std::size_t cap = pt::device_max_shared_bytes();
        if (shared_bytes + static_bytes > cap)
            throw api_error(
                DMK_ERR_INVALID_ARGUMENT,
                "esp::short_range: " +
                    std::string(gpu.strategy == GpuSrStrategy::PruneTile ? "PruneTile" : "PruneSource") + " needs " +
                    std::to_string(shared_bytes) + " B of dynamic + " + std::to_string(static_bytes) +
                    " B of static shared memory (max_tiles=" + std::to_string(max_tiles) + ", n=" + std::to_string(n) +
                    ", nc=" + std::to_string(nc) + ") but this device allows " + std::to_string(cap) +
                    " B per block. Use a smaller r_c (more, smaller cells), fewer sources, or the Dense strategy.");
        pt::set_max_dynamic_smem(*kernel, shared_bytes);
    }

    EspSrArgs<Real> a;
    a.nc = nc;
    a.rsc = rsc;
    a.cen = cen;
    a.r_c_sq = r_c_sq;
    a.cell_start = d_cell_start;
    a.xs = d_xs;
    a.ys = d_ys;
    a.zs = d_zs;
    a.qs = d_qs;
    a.ns = dims.normal_dim > 0 ? d_qs + std::size_t(dims.in_dim) * n : nullptr;
    a.n_sorted = n;
    a.nbc_tab = reinterpret_cast<const int *>(gpu.d_nbc_tab);
    a.off_tab = reinterpret_cast<const Real *>(gpu.d_off_tab);
    a.pg_sorted = d_pg_sorted;
    a.max_tiles = max_tiles;
    a.prune_stats = d_prune_stats;

    kernel->launch(dim3(ncells, 1, 1), dim3(kBlockSize, 1, 1), shared_bytes, gpu.stream, a);

    if (prune_stats)
        report_prune_stats(gpu, d_prune_stats);

    scatter_gpu<Real>(gpu, n, out_dim, d_orig, d_pg_sorted, d_pot, d_gx, d_gy, d_gz);
}

template void short_range_gpu<float>(GpuState &, int, const float *, const float *, float *, float *, float *, float *);
template void short_range_gpu<double>(GpuState &, int, const double *, const double *, double *, double *, double *,
                                      double *);

} // namespace dmk::cuda::esp
