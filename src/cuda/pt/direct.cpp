#include <dmk/cuda/pt/passes.hpp>

#include "../jit/autotune.hpp"
#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"

#include <dmk.h>
#include <dmk/cuda/direct_kernelargs.hpp>
#include <dmk/direct.hpp>

#include <cuda_runtime.h>

#include <cstdint>
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
    if (kernel == DMK_LAPLACE_DIPOLE && dim == 3)
        return "LaplaceDipolePolyEvaluator3DCuda";
    if (kernel == DMK_STOKESLET && dim == 3)
        return "StokesletPolyEvaluator3DCuda";
    if (kernel == DMK_STRESSLET && dim == 3)
        return "StressletPolyEvaluator3DCuda";
    throw std::runtime_error("pt::direct: unsupported kernel");
}

// Evaluators taking a trailing EVAL_LEVEL template argument.
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

    // Common args; the src/trg sides differ only in the target/pot fields.
    dmk::cuda::DirectByBoxArgs<Real> base;
    base.n_work = n_work;
    base.nlist1_stride = s.topology.nlist1_stride;
    base.thresh2 = Real{1e-30};
    base.direct_work = s.topology.d_direct_work.data();
    base.list1_flat = s.topology.d_list1_flat.data();
    base.list1_count = s.topology.d_list1_count.data();
    base.box_levels = s.topology.d_box_levels.data();
    base.ifpwexp = s.topology.d_ifpwexp.data();
    base.direct_rsc = s.fourier.d_direct_rsc.data();
    base.direct_cen = s.fourier.d_direct_cen.data();
    base.direct_d2max = s.fourier.d_direct_d2max.data();
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
    a_src.r_target_flat = s.particles.d_r_src.data();
    a_src.r_target_offsets = s.particles.d_r_src_offsets.data();
    a_src.target_counts = s.particles.d_src_counts.data();
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
        const std::size_t shared_bytes = std::size_t(config.at("SRC_TILE")) * values_per_source * sizeof(Real);
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
        const std::string tune_key = tune_key_ss.str();

        // `plans` is process-wide and hits before coefficients are generated, so its
        // key must also carry everything the baked literals depend on.
        std::ostringstream plan_key_ss;
        plan_key_ss << std::setprecision(std::numeric_limits<double>::max_digits10);
        plan_key_ss << tune_key << "|nd=" << s.fourier.n_digits << "|beta=" << s.fourier.beta
                    << "|fp=" << s.fourier.fparam << "|nlev=" << s.n_levels;
        const std::string plan_key = plan_key_ss.str();
        {
            std::lock_guard<std::mutex> lock(plan_mtx);
            auto it = plans.find(plan_key);
            if (it != plans.end())
                return it->second;
        }

        const auto coeffs = get_local_correction_coeffs<Real>(s.kernel, DIM, s.fourier.n_digits, s.fourier.beta);
        if (coeffs.empty())
            throw std::runtime_error("pt::direct: empty coefficient set");

        std::string coeff_struct;
        std::string coeff_args;
        for (std::size_t i = 0; i < coeffs.size(); ++i) {
            const std::string name = "Coeff" + std::to_string(i);
            coeff_struct += emit_coeff_struct<Real>(name.c_str(), coeffs[i]);
            coeff_args += (i ? ", " : "") + name;
        }
        std::string evaluator_expr = std::string(evaluator_family(s.kernel, DIM)) + "<" + coeff_args;
        if (takes_eval_level(s.kernel))
            evaluator_expr += ", " + std::to_string(eval_level);
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
            return std::size_t(st) * values_per_source * sizeof(Real) <= max_shared;
        };
        const auto benchmark = [&](const TuningParams &p) {
            return jit::benchmark_cuda_ms(stream, jit::CudaBenchmarkOptions{2, 5},
                                          [&](cudaStream_t bs) { launch_with(get_kernel(p), p, bench_args, bs); });
        };

        const TuningParams config = autotune_config(tune_key, "PtDirectKernel", space, defaults, constraint, benchmark);
        std::pair<std::shared_ptr<jit::JitKernel>, TuningParams> plan{get_kernel(config), config};

        std::lock_guard<std::mutex> lock(plan_mtx);
        plans[plan_key] = plan;
        return plan;
    };

    const int el_src = eval_level_for(s.outputs.eval_src);
    const int el_trg = eval_level_for(s.outputs.eval_trg);
    const auto plan_src = resolve(el_src, a_src);
    const auto plan_trg = (el_trg == el_src || s.outputs.pot_trg_size == 0) ? plan_src : resolve(el_trg, a_trg);

    // Boxes without near-field work keep their zeroed pot region.
    s.outputs.d_pot_direct_src.zero_async(stream);
    s.outputs.d_pot_direct_trg.zero_async(stream);
    launch_with(plan_src.first, plan_src.second, a_src, stream);
    launch_with(plan_trg.first, plan_trg.second, a_trg, stream);
}

template void direct<float, 2>(State<float, 2> &, cudaStream_t);
template void direct<float, 3>(State<float, 3> &, cudaStream_t);
template void direct<double, 2>(State<double, 2> &, cudaStream_t);
template void direct<double, 3>(State<double, 3> &, cudaStream_t);

} // namespace dmk::cuda::pt
