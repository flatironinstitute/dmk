#include <dmk/cuda/pt/passes.hpp>

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"

#include <dmk.h>
#include <dmk/cuda/eval_targets_kernelargs.hpp>

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

template <typename Real, int DIM>
void launch_eval_side(JitCache &cache, dmk::cuda::EvalTargetsArgs<Real> args, int eval_level, int n_charge_dim,
                      cudaStream_t stream) {
    if (args.n_eval_boxes == 0)
        return;
    const int n_order = args.n_order;

    const std::size_t smem_bytes = eval_shared_bytes(DIM, n_order, sizeof(Real));
    const int smem_coeffs = smem_bytes <= device_max_shared_bytes() ? 1 : 0;

    auto launch_one = [&](const TuningParams &p, cudaStream_t st, bool compile_only) {
        JitKey key;
        key.name = "PtEvalTargetsByBoxKernel";
        key.real = jit_real_name<Real>();
        key.sm_major = cache.sm_major();
        key.sm_minor = cache.sm_minor();
        key.params = {{"DIM", DIM},
                      {"EVAL_LEVEL", eval_level},
                      {"N_CHARGE_DIM", n_charge_dim},
                      {"N_ORDER", n_order},
                      {"SMEM_COEFFS", smem_coeffs},
                      {"BLOCK_SIZE", p.at("BLOCK_SIZE")},
                      {"TARGETS_PER_THREAD", p.at("TARGETS_PER_THREAD")}};
        auto kernel = cache.get_kernel_from_source(
            key, [&] { return make_stage_source("pt/eval_targets.cu", key, "", "PtEvalTargets"); });
        const std::size_t shared = smem_coeffs ? smem_bytes : 0;
        set_max_dynamic_smem(*kernel, shared);
        if (compile_only)
            return;
        kernel->launch(dim3(args.n_eval_boxes, 1, 1), dim3(p.at("BLOCK_SIZE"), 1, 1), shared, st, args);
    };

    std::ostringstream tune_key;
    tune_key << "PtEvalTargets|real=" << jit_real_name<Real>() << "|dim=" << DIM << "|eval_level=" << eval_level
             << "|n_charge_dim=" << n_charge_dim << "|n_order=" << n_order << "|smem=" << smem_coeffs
             << "|src=" << jit::jit_source_hash("pt/eval_targets.cu");
    const std::string tk = tune_key.str();

    if (auto cfg = autotune_cached(tk)) {
        launch_one(*cfg, stream, false);
        return;
    }

    const cudaDeviceProp &prop = device_prop();

    const std::vector<TuningParameter> space{{"BLOCK_SIZE", {128, 256, 512}},
                                             {"TARGETS_PER_THREAD", {1, 2, 3, 4, 6, 8}}};
    const TuningParams defaults{{"BLOCK_SIZE", 256}, {"TARGETS_PER_THREAD", 1}};
    const auto constraint = [&](const TuningParams &p) {
        const int bs = p.at("BLOCK_SIZE");
        return bs > 0 && bs <= prop.maxThreadsPerBlock && bs % 32 == 0 && p.at("TARGETS_PER_THREAD") > 0;
    };

    autotuned_launch<Real>(tk, "PtEvalTargetsByBoxKernel", space, defaults, constraint, launch_one, nullptr, 0, stream,
                           {});
}

} // namespace

template <typename Real, int DIM>
void eval_targets(State<Real, DIM> &s, cudaStream_t stream) {
    auto &o = s.outputs;
    auto &f = s.fourier;
    auto &w = s.worklists;

    if (w.n_eval_boxes == 0)
        return;

    static JitCache eval_cache;

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
        launch_eval_side<Real, DIM>(eval_cache, args, eval_level_for(o.eval_src), f.n_charge_dim, stream);
    }

    if (o.pot_trg_size) {
        args.r_target_flat = s.particles.d_r_trg.data();
        args.r_target_offsets = s.particles.d_r_trg_offsets.data();
        args.target_counts = s.particles.d_trg_counts.data();
        args.pot_flat = o.d_pot_eval_trg.data();
        args.pot_offsets = o.d_pot_trg_offsets.data();
        launch_eval_side<Real, DIM>(eval_cache, args, eval_level_for(o.eval_trg), f.n_charge_dim, stream);
    }
}

template void eval_targets<float, 2>(State<float, 2> &, cudaStream_t);
template void eval_targets<float, 3>(State<float, 3> &, cudaStream_t);
template void eval_targets<double, 2>(State<double, 2> &, cudaStream_t);
template void eval_targets<double, 3>(State<double, 3> &, cudaStream_t);

} // namespace dmk::cuda::pt
