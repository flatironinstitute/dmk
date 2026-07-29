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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
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
// contract in pt_direct.cu (size + __device__ constexpr at(i)).
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
    throw std::runtime_error("pt::direct: unsupported kernel (step 2: Laplace / Sqrt-Laplace potential only)");
}

} // namespace

template <typename Real, int DIM>
void direct(State<Real, DIM> &s, cudaStream_t stream) {
    const int n_work = static_cast<int>(s.topology.d_direct_work.size());
    if (n_work == 0)
        return;

    // Scalar potential only for step 2: one input component, no normals.
    const int values_per_source = DIM + 1; // SPATIAL_DIM + KERNEL_INPUT_DIM(1) + NORMAL_DIM(0)

    // Baked coefficient literals from the host generator (scale-free single poly
    // for the scalar kernels).
    const auto coeffs = get_local_correction_coeffs<Real>(s.kernel, DIM, s.fourier.n_digits, s.fourier.beta);
    if (coeffs.empty())
        throw std::runtime_error("pt::direct: empty coefficient set");

    const std::string coeff_struct = emit_coeff_struct<Real>("Coeff0", coeffs[0]);
    const std::string evaluator_expr = std::string(evaluator_family(s.kernel, DIM)) + "<Coeff0>";
    const std::string kernel_name =
        "PtDirectKernel_" + fnv1a_hex(std::string(jit_real_name<Real>()) + "|" + evaluator_expr + "|" + coeff_struct);

    std::ostringstream prelude_ss;
    prelude_ss << "using Real = " << jit_real_name<Real>() << ";\n";
    prelude_ss << "#define DMK_DIRECT_KERNEL_NAME " << kernel_name << "\n\n";
    prelude_ss << coeff_struct;
    prelude_ss << "#define DMK_DIRECT_EVALUATOR " << evaluator_expr << "\n\n";
    const std::string prelude = prelude_ss.str();

    // Common args; the src/trg sides differ only in the target/pot fields.
    dmk::cuda::DirectByBoxArgs<Real> base;
    base.n_work = n_work;
    base.n_levels = s.n_levels;
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

    // Compile (cache-keyed) + launch one config.
    auto launch_cfg = [&](dmk::cuda::DirectByBoxArgs<Real> args, const TuningParams &config, cudaStream_t st) {
        if (args.n_work == 0)
            return;
        JitKey key;
        key.name = kernel_name;
        key.real = jit_real_name<Real>();
        key.sm_major = cache.sm_major();
        key.sm_minor = cache.sm_minor();
        key.params = config;
        const std::string source = make_stage_source("pt_direct.cu", key, prelude, "PtDirect");
        auto kernel = cache.get_kernel_from_source(key, source);
        const std::size_t shared_bytes = std::size_t(config.at("SRC_TILE")) * values_per_source * sizeof(Real);
        kernel->launch(dim3(args.n_work, 1, 1), dim3(config.at("BLOCK_SIZE"), 1, 1), shared_bytes, st, args);
    };

    // Autotune (src side is representative of the launch cost).
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    const std::size_t max_shared = prop.sharedMemPerBlockOptin > 0 ? std::size_t(prop.sharedMemPerBlockOptin)
                                                                   : std::size_t(prop.sharedMemPerBlock);

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
                                      [&](cudaStream_t bs) { launch_cfg(a_src, p, bs); });
    };

    std::ostringstream tune_key;
    tune_key << "PtDirect|real=" << jit_real_name<Real>() << "|kernel=" << int(s.kernel) << "|dim=" << DIM
             << "|vps=" << values_per_source << "|nlist1=" << s.topology.nlist1_stride;

    const TuningParams config =
        autotune_config(tune_key.str(), "PtDirectKernel", space, defaults, constraint, benchmark);

    // Boxes without near-field work keep their zeroed pot region.
    s.outputs.d_pot_direct_src.zero_async(stream);
    s.outputs.d_pot_direct_trg.zero_async(stream);
    launch_cfg(a_src, config, stream);
    launch_cfg(a_trg, config, stream);
}

template void direct<float, 2>(State<float, 2> &, cudaStream_t);
template void direct<float, 3>(State<float, 3> &, cudaStream_t);
template void direct<double, 2>(State<double, 2> &, cudaStream_t);
template void direct<double, 3>(State<double, 3> &, cudaStream_t);

} // namespace dmk::cuda::pt
