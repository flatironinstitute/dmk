#include <dmk/cuda/pt/passes.hpp>

#include "../jit/autotune.hpp"
#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"

#include <dmk.h>
#include <dmk/cuda/direct_kernelargs.hpp>
#include <dmk/cuda/helpers.hpp>
#include <dmk/direct.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dmk::cuda::pt {
namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

// Dev knobs for the direct-pass source prefilter (see pt/direct.cu):
//   DMK_DIRECT_PREFILTER  0 = off, 1 = cull compacting indices, 2 = run the cull machinery
//                         but keep every source, which isolates its overhead and leaves
//                         output unchanged, 3 = cull compacting source data (the default;
//                         1 loses to 0 because an index list is a dependent LDS -> LDS chain)
//   DMK_DIRECT_CULL_TILE  lanes per cull group; must be a power-of-two divisor of 32
//   DMK_DIRECT_EVAL_UNROLL  unroll factor for the survivor loop. A knob and not a tuning
//                         axis because the grid is exhaustive and compiles inside the benchmark
int env_int(const char *name, int fallback) {
    const char *value = std::getenv(name);
    if (!value || !*value)
        return fallback;
    return std::atoi(value);
}

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

// Emit a compile-time coefficient struct matching the horner_const<Coeffs>
// contract in pt/direct.cu (size + __device__ constexpr at(i)).
template <typename Real>
std::string emit_coeff_struct(const char *name, const std::vector<Real> &coeffs) {
    std::ostringstream ss;
    ss << std::setprecision(std::numeric_limits<Real>::max_digits10);
    ss << "struct " << name << " {\n";
    ss << "    static constexpr int size = " << coeffs.size() << ";\n";
    ss << "    __device__ static constexpr Real at(int i) {\n";
    ss << "        constexpr Real data[size] = {\n";
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        ss << "            Real{" << coeffs[i] << "}" << (i + 1 < coeffs.size() ? "," : "") << "\n";
    }
    ss << "        };\n        return data[i];\n    }\n};\n\n";
    return ss.str();
}

const char *evaluator_family(dmk_ikernel kernel, int dim) {
    if (kernel == DMK_LAPLACE)
        return dim == 3 ? "LaplacePolyEvaluator3DCuda" : "LaplacePolyEvaluator2DCuda";
    if (kernel == DMK_SQRT_LAPLACE)
        return dim == 3 ? "SqrtLaplacePolyEvaluator3DCuda" : "SqrtLaplacePolyEvaluator2DCuda";
    if (kernel == DMK_YUKAWA && dim == 3)
        return "YukawaLevelsCuda";
    if (kernel == DMK_LAPLACE_DIPOLE && dim == 3)
        return "LaplaceDipolePolyEvaluator3DCuda";
    if (kernel == DMK_STOKESLET && dim == 3)
        return "StokesletPolyEvaluator3DCuda";
    if (kernel == DMK_STRESSLET && dim == 3)
        return "StressletPolyEvaluator3DCuda";
    throw std::runtime_error("pt::direct: unsupported kernel");
}

// Evaluators taking a trailing EVAL_LEVEL template argument. Yukawa is excluded: its
// pack list must come last, so it emits EVAL_LEVEL first instead.
bool takes_eval_level(dmk_ikernel kernel) {
    return kernel == DMK_LAPLACE || kernel == DMK_SQRT_LAPLACE || kernel == DMK_LAPLACE_DIPOLE;
}

int eval_level_for(dmk_eval_type ev) {
    if (ev == DMK_POTENTIAL || ev == DMK_VELOCITY)
        return 1;
    if (ev == DMK_POTENTIAL_GRAD)
        return 2;
    throw std::runtime_error("pt::direct: unsupported eval_type");
}

} // namespace

template <typename Real, int DIM>
void direct(State<Real, DIM> &s, cudaStream_t stream) {
    const int n_work = static_cast<int>(s.topology.d_direct_work.size());
    if (n_work == 0)
        return;

    // Shared-tile stride: SPATIAL_DIM + KERNEL_INPUT_DIM + NORMAL_DIM. Scalar
    // kernels have one charge and no normal; Stokeslet a 3-vector charge; the
    // Stresslet additionally reads a per-source normal.
    const int input_dim = get_kernel_input_dim(DIM, s.kernel);
    const int normal_dim = (s.kernel == DMK_STRESSLET) ? DIM : 0;
    const int values_per_source = DIM + input_dim + normal_dim;

    // The shift table is only built for periodic trees, so its presence is the flag.
    const bool periodic = s.topology.d_list1_shift.size() != 0;

    const int prefilter = env_int("DMK_DIRECT_PREFILTER", 3);
    const int cull_tile = env_int("DMK_DIRECT_CULL_TILE", 32);
    const int eval_unroll = env_int("DMK_DIRECT_EVAL_UNROLL", 4);
    const int prefilter_stats = (prefilter != 0) ? env_int("DMK_DIRECT_PREFILTER_STATS", 0) : 0;
    // Only PREFILTER 3 can skip the staged source tile; the other modes re-read a source once
    // per target, which is what a shared broadcast is for. Off where it is optional: PREFILTER 3
    // compacts source data into the survivor list, so staging holds a second copy and spends
    // shared that occupancy would rather have.
    const int stage_src = (prefilter == 3) ? env_int("DMK_DIRECT_STAGE_SRC", 0) : 1;
    if (prefilter < 0 || prefilter > 3)
        throw std::runtime_error("DMK_DIRECT_PREFILTER must be 0, 1, 2 or 3");
    if (cull_tile < 1 || cull_tile > 32 || 32 % cull_tile != 0)
        throw std::runtime_error("DMK_DIRECT_CULL_TILE must be a power-of-two divisor of 32");
    if (eval_unroll < 1)
        throw std::runtime_error("DMK_DIRECT_EVAL_UNROLL must be positive");
    const int n_cull_groups = 32 / cull_tile;

    // Staged source tile, plus (when culling) the per-cull-group target boxes and one
    // 32-entry survivor list per cull group per warp. The list holds source indices
    // (PREFILTER 1/2) or the compacted source data (PREFILTER 3). Must match the carve-up
    // in DirectByBoxBody exactly.
    const auto shared_bytes_for = [&](int src_tile, int block_size, int targets) {
        std::size_t bytes = stage_src ? std::size_t(src_tile) * values_per_source * sizeof(Real) : 0;
        if (prefilter == 0)
            return bytes;
        const std::size_t warps = std::size_t(block_size / 32);
        bytes += warps * targets * n_cull_groups * 2 * DIM * sizeof(Real);
        const std::size_t slots = warps * n_cull_groups * 32;
        // PREFILTER 3 pads each compacted source to a 16-byte multiple so the inner loop can
        // fetch it with one aligned vector load; must match VPS_PAD in DirectByBoxBody.
        const std::size_t vec_n = 16 / sizeof(Real);
        const std::size_t vps_pad = ((values_per_source + vec_n - 1) / vec_n) * vec_n;
        // The index array is rounded up to a whole Real because the device advances its
        // shared cursor in Real units before carving the regions that follow it.
        bytes += (prefilter == 3) ? slots * vps_pad * sizeof(Real)
                                  : ((slots * sizeof(int) + sizeof(Real) - 1) / sizeof(Real)) * sizeof(Real);
        return bytes;
    };

    // Selectivity counters live in a small static device buffer: this is a dev-only knob,
    // so it is not worth a State member, and it forces a sync to read back.
    constexpr int kCullStats = 6;
    static unsigned long long *d_cull_stats = nullptr;
    if (prefilter_stats && !d_cull_stats)
        DMK_CHECK_CUDA(cudaMalloc(&d_cull_stats, kCullStats * sizeof(unsigned long long)));
    if (prefilter_stats)
        DMK_CHECK_CUDA(cudaMemsetAsync(d_cull_stats, 0, kCullStats * sizeof(unsigned long long), stream));

    // Common args; the src/trg sides differ only in the target/pot fields.
    dmk::cuda::DirectByBoxArgs<Real> base;
    base.n_work = n_work;
    base.nlist1_stride = s.topology.nlist1_stride;
    base.thresh2 = Real{1e-30};
    base.direct_work = s.topology.d_direct_work.data();
    base.list1_flat = s.topology.d_list1_flat.data();
    base.list1_count = s.topology.d_list1_count.data();
    base.list1_shift = s.topology.d_list1_shift.data();
    base.box_levels = s.topology.d_box_levels.data();
    base.ifpwexp = s.topology.d_ifpwexp.data();
    base.direct_rsc = s.fourier.d_direct_rsc.data();
    base.direct_cen = s.fourier.d_direct_cen.data();
    base.direct_d2max = s.fourier.d_direct_d2max.data();
    base.cull_stats = prefilter_stats ? d_cull_stats : nullptr;
    base.r_src_flat = s.particles.d_r_src.data();
    base.r_src_offsets = s.particles.d_r_src_offsets.data();
    base.src_counts = s.particles.d_src_counts.data();
    base.charge_flat = s.particles.d_charge.data();
    base.charge_offsets = s.particles.d_charge_offsets.data();
    if (normal_dim > 0) {
        base.normal_flat = s.particles.d_normal.data();
        base.normal_offsets = s.particles.d_normal_offsets.data();
    }

    dmk::cuda::DirectByBoxArgs<Real> a_src = base;
    a_src.r_target_flat = s.particles.d_r_src_owned.data();
    a_src.r_target_offsets = s.particles.d_r_src_offsets_owned.data();
    a_src.target_counts = s.particles.d_src_counts_owned.data();
    a_src.pot_flat = s.outputs.d_pot_direct_src.data();
    a_src.pot_offsets = s.outputs.d_pot_src_offsets.data();

    dmk::cuda::DirectByBoxArgs<Real> a_trg = base;
    a_trg.r_target_flat = s.particles.d_r_trg.data();
    a_trg.r_target_offsets = s.particles.d_r_trg_offsets.data();
    a_trg.target_counts = s.particles.d_trg_counts.data();
    a_trg.pot_flat = s.outputs.d_pot_direct_trg.data();
    a_trg.pot_offsets = s.outputs.d_pot_trg_offsets.data();

    static JitCache cache;
    static std::mutex plan_mtx;
    static std::map<std::string, std::pair<std::shared_ptr<jit::JitKernel>, TuningParams>> plans;

    auto launch_with = [&](const std::shared_ptr<jit::JitKernel> &kernel, const TuningParams &config,
                           const dmk::cuda::DirectByBoxArgs<Real> &args, cudaStream_t st) {
        if (args.n_work == 0)
            return;
        const std::size_t shared_bytes =
            shared_bytes_for(config.at("SRC_TILE"), config.at("BLOCK_SIZE"), config.at("TARGETS_PER_THREAD"));
        kernel->launch(dim3(args.n_work, 1, 1), dim3(config.at("BLOCK_SIZE"), 1, 1), shared_bytes, st, args);
    };

    // Bake coefficients, tune, and compile for one eval level. `bench_args` supplies
    // the tuning launch, so its pot buffer must match that level's output dim. One
    // coefficient pack serves both levels: beta carries the gradient bandlimiting, so
    // the gradient is the analytic derivative of the same polynomial.
    auto resolve = [&](int eval_level, const dmk::cuda::DirectByBoxArgs<Real> &bench_args)
        -> std::pair<std::shared_ptr<jit::JitKernel>, TuningParams> {
        // Shape-only: the tuning optimum (and the persistent JSON key) depends on
        // launch geometry, not on the coefficient values.
        std::ostringstream tune_key_ss;
        tune_key_ss << "PtDirect|real=" << jit_real_name<Real>() << "|kernel=" << int(s.kernel) << "|dim=" << DIM
                    << "|vps=" << values_per_source << "|nlist1=" << s.topology.nlist1_stride << "|el=" << eval_level;
        // Pack degrees change the unrolled Horner's register profile, so they need their own
        // tuning entry. n_digits and beta determine the degrees; direct_coeffs_by_level is
        // populated for Yukawa only and so cannot carry this alone.
        tune_key_ss << "|nd=" << s.fourier.n_digits << "|beta=" << s.fourier.beta;
        for (const auto &pack : s.fourier.direct_coeffs_by_level)
            tune_key_ss << "|nc=" << pack.size();
        // The prefilter changes the shared-memory budget and the inner loop, so it needs
        // its own tuning entry; without this a config cached for one variant is silently
        // reused for another and the comparison measures nothing.
        tune_key_ss << "|pf=" << prefilter << "|ct=" << cull_tile << "|pfs=" << prefilter_stats << "|eu=" << eval_unroll
                    << "|ss=" << stage_src;
        tune_key_ss << "|src=" << jit::jit_source_hash("pt/direct.cu")
                    << "|ev=" << jit::jit_header_hash("dmk/cuda/poly_evaluators_device.hpp");
        const std::string tune_key = tune_key_ss.str();

        // `plans` is process-wide and hits before coefficients are generated, so its
        // key must also carry everything the baked literals depend on.
        std::ostringstream plan_key_ss;
        plan_key_ss << std::setprecision(std::numeric_limits<double>::max_digits10);
        plan_key_ss << tune_key << "|nd=" << s.fourier.n_digits << "|beta=" << s.fourier.beta
                    << "|fp=" << s.fourier.fparam << "|nlev=" << s.n_levels << "|l0=" << s.fourier.direct_coeffs_level0
                    << "|per=" << int(periodic);
        const std::string plan_key = plan_key_ss.str();
        {
            std::lock_guard<std::mutex> lock(plan_mtx);
            auto it = plans.find(plan_key);
            if (it != plans.end())
                return it->second;
        }

        const std::vector<std::vector<Real>> coeffs =
            (s.kernel == DMK_YUKAWA)
                ? s.fourier.direct_coeffs_by_level
                : get_local_correction_coeffs<Real>(s.kernel, DIM, s.fourier.n_digits, s.fourier.beta);
        if (coeffs.empty())
            throw std::runtime_error("pt::direct: empty coefficient set");

        std::string coeff_struct;
        std::vector<std::string> targs;
        if (s.kernel == DMK_YUKAWA) {
            targs.push_back(std::to_string(eval_level));
            targs.push_back(std::to_string(s.fourier.direct_coeffs_level0));
        }
        for (std::size_t i = 0; i < coeffs.size(); ++i) {
            const std::string name = "Coeff" + std::to_string(i);
            coeff_struct += emit_coeff_struct<Real>(name.c_str(), coeffs[i]);
            targs.push_back(name);
        }
        if (takes_eval_level(s.kernel))
            targs.push_back(std::to_string(eval_level));

        std::string evaluator_expr = std::string(evaluator_family(s.kernel, DIM)) + "<";
        for (std::size_t i = 0; i < targs.size(); ++i)
            evaluator_expr += (i ? ", " : "") + targs[i];
        evaluator_expr += ">";
        const std::string kernel_name = "PtDirectKernel_" + fnv1a_hex(std::string(jit_real_name<Real>()) + "|" +
                                                                      evaluator_expr + "|" + coeff_struct);

        std::ostringstream prelude_ss;
        prelude_ss << "#define DMK_DIRECT_KERNEL_NAME " << kernel_name << "\n\n"; // Real provided by make_stage_source
        prelude_ss << coeff_struct;
        prelude_ss << "#define DMK_DIRECT_EVALUATOR " << evaluator_expr << "\n\n";
        const std::string prelude = prelude_ss.str();

        auto get_kernel = [&](const TuningParams &cfg) {
            JitKey key;
            key.name = kernel_name;
            key.real = jit_real_name<Real>();
            key.sm_major = cache.sm_major();
            key.sm_minor = cache.sm_minor();
            key.params = cfg;
            // Emitted into the source as `constexpr int PERIODIC` and folded into the
            // module cache key, which the coefficient-derived kernel_name cannot
            // distinguish: the two variants bake identical coefficients.
            key.params["PERIODIC"] = periodic ? 1 : 0;
            key.params["PREFILTER"] = prefilter;
            key.params["CULL_TILE"] = cull_tile;
            key.params["EVAL_UNROLL"] = eval_unroll;
            key.params["PREFILTER_STATS"] = prefilter_stats;
            key.params["STAGE_SRC"] = stage_src;
            return cache.get_kernel_from_source(
                key, [&] { return make_stage_source("pt/direct.cu", key, prelude, "PtDirect"); });
        };

        const cudaDeviceProp &prop = device_prop();
        const std::size_t max_shared = device_max_shared_bytes();

        const std::vector<TuningParameter> space{
            {"SRC_TILE", {16, 32, 64, 96, 128, 192, 256}},
            {"BLOCK_SIZE", {64, 128, 256, 512}},
            {"TARGETS_PER_THREAD", {1, 2, 3, 4}},
        };
        const TuningParams defaults{{"SRC_TILE", 32}, {"BLOCK_SIZE", 128}, {"TARGETS_PER_THREAD", 1}};

        const auto constraint = [&](const TuningParams &p) {
            const int st = p.at("SRC_TILE"), bs = p.at("BLOCK_SIZE"), tg = p.at("TARGETS_PER_THREAD");
            if (st <= 0 || bs <= 0 || bs > prop.maxThreadsPerBlock || bs % 32 != 0 || tg < 1 || tg > 4)
                return false;
            // Unstaged, SRC_TILE reaches nothing, so collapse the axis instead of timing the
            // same kernel seven times.
            if (!stage_src && st != defaults.at("SRC_TILE"))
                return false;
            return shared_bytes_for(st, bs, tg) <= max_shared;
        };
        const auto benchmark = [&](const TuningParams &p) {
            return jit::benchmark_cuda_ms(stream, jit::CudaBenchmarkOptions{},
                                          [&](cudaStream_t bs) { launch_with(get_kernel(p), p, bench_args, bs); });
        };

        // The stats kernel recomputes every pair distance to measure what the cull *should*
        // have kept, so it is far slower than the real one and nobody runs it for speed.
        // Tuning it would JIT and benchmark the whole grid of a kernel that does not matter.
        const auto precompile = [&](const TuningParams &p) { get_kernel(p); };
        const TuningParams config = prefilter_stats ? defaults
                                                    : autotune_config(tune_key, "PtDirectKernel", space, defaults,
                                                                      constraint, benchmark, precompile, {});
        std::pair<std::shared_ptr<jit::JitKernel>, TuningParams> plan{get_kernel(config), config};

        std::lock_guard<std::mutex> lock(plan_mtx);
        plans[plan_key] = plan;
        return plan;
    };

    const int el_src = eval_level_for(s.outputs.eval_src);
    const int el_trg = eval_level_for(s.outputs.eval_trg);
    const auto plan_src = resolve(el_src, a_src);
    const auto plan_trg = (el_trg == el_src || s.outputs.pot_trg_size == 0) ? plan_src : resolve(el_trg, a_trg);

    launch_with(plan_src.first, plan_src.second, a_src, stream);
    launch_with(plan_trg.first, plan_trg.second, a_trg, stream);

    if (prefilter_stats) {
        unsigned long long h[kCullStats] = {};
        DMK_CHECK_CUDA(cudaStreamSynchronize(stream));
        DMK_CHECK_CUDA(cudaMemcpy(h, d_cull_stats, sizeof(h), cudaMemcpyDeviceToHost));
        const double scanned = double(h[0]); // per (source, cull group)
        const double sources = scanned / n_cull_groups;
        const auto pct = [](double num, double den) { return den > 0 ? 100.0 * num / den : 0.0; };
        // survived vs needed is the cull's own quality: the gap is pure AABB over-inclusion.
        // warp_needed is what the per-pair branch already skips without any prefilter, so the
        // cull only sells Horner reductions below that line -- above it, only discovery cost.
        // `work` is the warp-iteration count: the warp runs max-over-groups times, not
        // mean-over-groups, so it -- not survived% -- is what is comparable across CULL_TILE.
        // At CULL_TILE=32 work% and survived% coincide by construction.
        // Geometry is echoed because the counters depend on it: BLOCK_SIZE and
        // TARGETS_PER_THREAD decide which targets share a warp (and so how tight a cull
        // group's box is), and SRC_TILE sets the chunking. Stats runs use the defaults, not
        // the tuned config, so these are not necessarily production's numbers.
        const TuningParams &cfg = plan_src.second;
        std::fprintf(stderr,
                     "[dmk direct prefilter] pf=%d cull_tile=%d block=%d targets=%d src_tile=%d\n"
                     "  per (src,group): scanned=%llu survived=%.1f%% needed=%.1f%% (aabb slack %.1f pts)\n"
                     "  work=%.1f%% of src slots (max over groups; the cost metric) | "
                     "warp_needed=%.1f%% | lane-pairs in range=%.2f%% (floor)\n",
                     prefilter, cull_tile, cfg.at("BLOCK_SIZE"), cfg.at("TARGETS_PER_THREAD"), cfg.at("SRC_TILE"), h[0],
                     pct(double(h[1]), scanned), pct(double(h[2]), scanned), pct(double(h[1]) - double(h[2]), scanned),
                     pct(double(h[5]), sources), pct(double(h[4]), sources), pct(double(h[3]), sources * 32.0));
    }
}

template void direct<float, 2>(State<float, 2> &, cudaStream_t);
template void direct<float, 3>(State<float, 3> &, cudaStream_t);
template void direct<double, 2>(State<double, 2> &, cudaStream_t);
template void direct<double, 3>(State<double, 3> &, cudaStream_t);

} // namespace dmk::cuda::pt
