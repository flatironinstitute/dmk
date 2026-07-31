#include <dmk/cuda/pt/passes.hpp>

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"

#include <dmk.h>
#include <dmk/cuda/eval_targets_kernelargs.hpp>
#include <dmk/direct.hpp>

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda::pt {
namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

int eval_level_for(dmk_eval_type ev) {
    if (ev == DMK_POTENTIAL || ev == DMK_VELOCITY)
        return 1;
    if (ev == DMK_POTENTIAL_GRAD)
        return 2;
    throw std::runtime_error("pt::eval_targets: unsupported eval_type");
}

std::size_t eval_shared_bytes(int dim, int n_order, std::size_t sizeof_real) {
    const std::size_t n2 = std::size_t(n_order) * n_order;
    return (dim == 2 ? n2 : n2 * n_order) * sizeof_real;
}

// One eval-targets side (src or trg), autotuned. Additive into args.pot_flat, so
// the tuner snapshots that buffer (`pot_size` reals).
template <typename Real, int DIM>
void launch_eval_side(JitCache &cache, dmk::cuda::EvalTargetsArgs<Real> args, int eval_level, int n_charge_dim,
                      Real *pot_flat, std::size_t pot_size, cudaStream_t stream) {
    if (args.n_eval_boxes == 0)
        return;
    const int n_order = args.n_order;

    auto launch_one = [&](const TuningParams &p, cudaStream_t st) {
        JitKey key;
        key.name = "PtEvalTargetsByBoxKernel";
        key.real = jit_real_name<Real>();
        key.sm_major = cache.sm_major();
        key.sm_minor = cache.sm_minor();
        key.params = {{"DIM", DIM},
                      {"EVAL_LEVEL", eval_level},
                      {"N_CHARGE_DIM", n_charge_dim},
                      {"N_ORDER", n_order},
                      {"BLOCK_SIZE", p.at("BLOCK_SIZE")},
                      {"TARGETS_PER_THREAD", p.at("TARGETS_PER_THREAD")}};
        auto kernel = cache.get_kernel_from_source(
            key, [&] { return make_stage_source("pt/eval_targets.cu", key, "", "PtEvalTargets"); });
        const std::size_t shared = eval_shared_bytes(DIM, n_order, sizeof(Real));
        set_max_dynamic_smem(*kernel, shared);
        kernel->launch(dim3(args.n_eval_boxes, 1, 1), dim3(p.at("BLOCK_SIZE"), 1, 1), shared, st, args);
    };

    std::ostringstream tune_key;
    tune_key << "PtEvalTargets|real=" << jit_real_name<Real>() << "|dim=" << DIM << "|eval_level=" << eval_level
             << "|n_charge_dim=" << n_charge_dim << "|n_order=" << n_order;
    const std::string tk = tune_key.str();

    if (auto cfg = autotune_cached(tk)) {
        launch_one(*cfg, stream);
        return;
    }

    const cudaDeviceProp &prop = device_prop();
    const std::size_t max_shared = device_max_shared_bytes();

    const std::vector<TuningParameter> space{{"BLOCK_SIZE", {128, 256, 512}}, {"TARGETS_PER_THREAD", {1, 2, 3, 4}}};
    const TuningParams defaults{{"BLOCK_SIZE", 256}, {"TARGETS_PER_THREAD", 1}};
    const auto constraint = [&](const TuningParams &p) {
        const int bs = p.at("BLOCK_SIZE");
        if (bs <= 0 || bs > prop.maxThreadsPerBlock || bs % 32 != 0 || p.at("TARGETS_PER_THREAD") <= 0)
            return false;
        return eval_shared_bytes(DIM, n_order, sizeof(Real)) <= max_shared;
    };

    autotuned_launch<Real>(tk, "PtEvalTargetsByBoxKernel", space, defaults, constraint, launch_one, pot_flat, pot_size,
                           stream);
}

template <typename Real>
void launch_self_correction(JitCache &cache, const dmk::cuda::SelfCorrectionArgs<Real> &args, cudaStream_t stream) {
    if (args.n_direct_work == 0)
        return;
    constexpr int BLOCK = 128;
    JitKey key;
    key.name = "PtSelfCorrectionKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    auto kernel = cache.get_kernel_from_source(
        key, [&] { return make_stage_source("pt/self_correction.cu", key, "", "PtSelfCorrection"); });
    dmk::cuda::SelfCorrectionArgs<Real> a = args;
    kernel->launch(dim3(a.n_direct_work, 1, 1), dim3(BLOCK, 1, 1), 0, stream, a);
}

} // namespace

template <typename Real, int DIM>
void eval_targets(State<Real, DIM> &s, cudaStream_t stream) {
    auto &o = s.outputs;
    auto &f = s.fourier;
    auto &w = s.worklists;

    o.d_pot_eval_src.zero_async(stream);
    o.d_pot_eval_trg.zero_async(stream);
    if (w.n_eval_boxes == 0)
        return;

    static JitCache eval_cache;
    static JitCache sc_cache;

    dmk::cuda::EvalTargetsArgs<Real> args;
    args.n_eval_boxes = w.n_eval_boxes;
    args.n_order = f.n_order;
    args.eval_targets_box_list = w.d_eval_targets_box_list.data();
    args.box_levels = s.topology.d_box_levels.data();
    args.sc_per_level = f.d_inv_box_scale.data();
    args.proxy_flat = s.scratch.d_proxy_coeffs_downward.data();
    args.proxy_offsets = s.scratch.d_proxy_offsets_downward.data();
    args.centers = f.d_centers.data();

    if (o.pot_src_size) {
        args.r_target_flat = s.particles.d_r_src.data();
        args.r_target_offsets = s.particles.d_r_src_offsets.data();
        args.target_counts = s.particles.d_src_counts.data();
        args.pot_flat = o.d_pot_eval_src.data();
        args.pot_offsets = o.d_pot_src_offsets.data();
        launch_eval_side<Real, DIM>(eval_cache, args, eval_level_for(o.eval_src), f.n_charge_dim,
                                    o.d_pot_eval_src.data(), o.pot_src_size, stream);
    }

    if (o.pot_trg_size) {
        args.r_target_flat = s.particles.d_r_trg.data();
        args.r_target_offsets = s.particles.d_r_trg_offsets.data();
        args.target_counts = s.particles.d_trg_counts.data();
        args.pot_flat = o.d_pot_eval_trg.data();
        args.pot_offsets = o.d_pot_trg_offsets.data();
        launch_eval_side<Real, DIM>(eval_cache, args, eval_level_for(o.eval_trg), f.n_charge_dim,
                                    o.d_pot_eval_trg.data(), o.pot_trg_size, stream);
    }

    // Self-correction modifies the source eval potential in sorted layout.
    if (o.pot_src_size && s.worklists.d_self_correction_work.size()) {
        dmk::cuda::SelfCorrectionArgs<Real> sc;
        sc.direct_work = s.topology.d_direct_work.data();
        sc.correction_factors = w.d_self_correction_work.data();
        sc.src_counts = s.particles.d_src_counts.data();
        sc.charge = s.particles.d_charge.data();
        sc.charge_offsets = s.particles.d_charge_offsets.data();
        sc.pot_src = o.d_pot_eval_src.data();
        sc.pot_src_offsets = o.d_pot_src_offsets.data();
        sc.n_direct_work = static_cast<int>(s.topology.d_direct_work.size());
        sc.n_input_dim = get_kernel_input_dim(DIM, s.kernel);
        sc.pot_stride = o.pot_src_dof;
        launch_self_correction<Real>(sc_cache, sc, stream);
    }
}

template void eval_targets<float, 2>(State<float, 2> &, cudaStream_t);
template void eval_targets<float, 3>(State<float, 3> &, cudaStream_t);
template void eval_targets<double, 2>(State<double, 2> &, cudaStream_t);
template void eval_targets<double, 3>(State<double, 3> &, cudaStream_t);

} // namespace dmk::cuda::pt
