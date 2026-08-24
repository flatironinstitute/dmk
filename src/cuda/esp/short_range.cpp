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
#include <dmk/util.hpp>

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

namespace dmk::cuda::esp {
namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

// Hand-tuned launch geometry, baked into the module -- ready for pt::autotuned_launch later.
constexpr int kBlockSize = 128;      // 4 warps
constexpr int kTargetsPerThread = 3; // dense strategy only
constexpr int kTileWidth = 32;       // = warpSize; one warp per source/target tile

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

// Emits the horner_const contract from esp/short_range.cu: size + __device__ constexpr at(i).
template <typename Real>
std::string emit_coeff_struct(const std::vector<Real> &coeffs) {
    std::ostringstream ss;
    ss << std::setprecision(std::numeric_limits<Real>::max_digits10);
    ss << "struct Coeffs {\n";
    ss << "    static constexpr int size = " << coeffs.size() << ";\n";
    ss << "    __device__ static constexpr Real at(int i) {\n";
    ss << "        constexpr Real data[size] = {\n";
    for (std::size_t i = 0; i < coeffs.size(); ++i)
        ss << "            Real{" << coeffs[i] << "}" << (i + 1 < coeffs.size() ? "," : "") << "\n";
    ss << "        };\n        return data[i];\n    }\n};\n\n";
    return ss.str();
}

} // namespace

// The residual polynomial comes from get_esp_correction_coeffs -- the same function the CPU
// evaluators use -- and is baked into the NVRTC module as literals. One source of truth, and the
// plan's own beta is honoured rather than a generator's fixed sigma.
template <typename Real>
void short_range_gpu(GpuState &gpu, int n, const Real *d_pos_aos, const Real *d_charges, Real *d_pot, Real *d_fx,
                     Real *d_fy, Real *d_fz) {
    const int nc = gpu.nc;
    const bool want_force = (gpu.eval_type >= DMK_POTENTIAL_GRAD);
    const int out_dim = want_force ? 4 : 1;
    const int ncells = nc * nc * nc;

    int *d_cell_start = nullptr;
    int *d_orig = nullptr;
    Real *d_xs = nullptr, *d_ys = nullptr, *d_zs = nullptr, *d_qs = nullptr;
    build_cell_list_gpu<Real>(gpu, n, nc, d_pos_aos, d_charges, &d_cell_start, &d_orig, &d_xs, &d_ys, &d_zs, &d_qs);

    ensure_capacity(gpu.d_scratch_pg, gpu.scratch_pg_cap, std::size_t(out_dim) * n * sizeof(Real));
    Real *d_pg_sorted = reinterpret_cast<Real *>(gpu.d_scratch_pg);

    // Argument is (R + cen)*rsc in [-1,1]; matches the CPU's r2_var == false branch.
    const Real rsc = Real(2.0 / gpu.r_c);
    const Real cen = Real(-0.5 * gpu.r_c);
    const Real r_c_sq = Real(gpu.r_c * gpu.r_c);

    const int strategy = static_cast<int>(gpu.strategy);
    const bool pruned = (gpu.strategy != GpuSrStrategy::Dense);

    // Sized from the worst-case cell population, which costs a host-device sync -- hence the cache.
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

    const bool prune_stats = pruned && util::env_is_set("DMK_ESP_PRUNE_STATS");
    unsigned long long *d_prune_stats = nullptr;
    if (prune_stats) {
        ensure_capacity(gpu.d_scratch_prune_stats, gpu.scratch_prune_stats_cap, 4 * sizeof(unsigned long long));
        d_prune_stats = reinterpret_cast<unsigned long long *>(gpu.d_scratch_prune_stats);
        cudaMemsetAsync(d_prune_stats, 0, 4 * sizeof(unsigned long long), gpu.stream);
    }

    // JitKey::to_string does not hash the source text, so baked coefficients have to reach the
    // cache key through the name; anything else that changes the code goes into key.params.
    const std::vector<std::vector<Real>> coeffs =
        get_esp_correction_coeffs<Real>(DMK_LAPLACE, /*fparam=*/0.0, gpu.r_c, /*n_dim=*/3, gpu.n_digits, gpu.beta);
    if (coeffs.empty() || coeffs[0].empty())
        throw std::runtime_error("short_range_gpu: empty coefficient set");
    const std::string coeff_struct = emit_coeff_struct<Real>(coeffs[0]);

    static JitCache cache;
    const std::string kernel_name =
        "EspShortRangeKernel_" + fnv1a_hex(std::string(jit_real_name<Real>()) + "|" + coeff_struct);

    JitKey key;
    key.name = kernel_name;
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params["BLOCK_SIZE"] = kBlockSize;
    key.params["TILE_WIDTH"] = kTileWidth;
    key.params["TARGETS_PER_THREAD"] = kTargetsPerThread;
    key.params["STRATEGY"] = strategy;
    key.params["WANT_FORCE"] = want_force ? 1 : 0;
    key.params["PRUNE_STATS"] = prune_stats ? 1 : 0;

    const std::shared_ptr<jit::JitKernel> kernel = cache.get_kernel_from_source(key, [&] {
        const std::string prelude = "#define DMK_ESP_SR_KERNEL_NAME " + kernel_name + "\n\n" + coeff_struct;
        return pt::make_stage_source("esp/short_range.cu", key, prelude, "EspShortRange");
    });

    if (shared_bytes > 0)
        pt::set_max_dynamic_smem(*kernel, shared_bytes);

    EspSrArgs<Real> a;
    a.nc = nc;
    a.out_dim = out_dim;
    a.rsc = rsc;
    a.cen = cen;
    a.r_c_sq = r_c_sq;
    a.cell_start = d_cell_start;
    a.xs = d_xs;
    a.ys = d_ys;
    a.zs = d_zs;
    a.qs = d_qs;
    a.nbc_tab = reinterpret_cast<const int *>(gpu.d_nbc_tab);
    a.off_tab = reinterpret_cast<const Real *>(gpu.d_off_tab);
    a.pg_sorted = d_pg_sorted;
    a.max_tiles = max_tiles;
    a.prune_stats = d_prune_stats;

    kernel->launch(dim3(ncells, 1, 1), dim3(kBlockSize, 1, 1), shared_bytes, gpu.stream, a);

    if (prune_stats)
        report_prune_stats(gpu, d_prune_stats);

    scatter_gpu<Real>(gpu, n, out_dim, d_orig, d_qs, d_pg_sorted, d_pot, d_fx, d_fy, d_fz);
}

template void short_range_gpu<float>(GpuState &, int, const float *, const float *, float *, float *, float *, float *);
template void short_range_gpu<double>(GpuState &, int, const double *, const double *, double *, double *, double *,
                                      double *);

} // namespace dmk::cuda::esp
