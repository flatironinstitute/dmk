#include <dmk/cuda/pt/passes.hpp>

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"
#include "tensorprod.hpp"

#include <dmk.h>
#include <dmk/cuda/charge2proxy_kernelargs.hpp>

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

// Spatial tiles ordered live-first, as a device array baked into the JIT source. The order
// depends on the tile shape, which is a tuning parameter, so it cannot be a single uploaded
// table. Live-first makes the dead-tile test warp-uniform: threads take consecutive tiles, so
// only the warps straddling the boundary diverge, where a per-tile test diverged everywhere.
std::string c2p_tile_order_prelude(int n_order, int it, int jt, int kt, int ball_r2) {
    const int ti = (n_order + it - 1) / it, tj = (n_order + jt - 1) / jt, tk = (n_order + kt - 1) / kt;
    std::vector<int> live, dead;
    for (int k = 0; k < tk; ++k)
        for (int j = 0; j < tj; ++j)
            for (int i = 0; i < ti; ++i) {
                const int i0 = i * it, j0 = j * jt, k0 = k * kt;
                const int flat = i + j * ti + k * ti * tj;
                if (!ball_r2 || i0 * i0 + j0 * j0 + k0 * k0 <= ball_r2)
                    live.push_back(flat);
                else
                    dead.push_back(flat);
            }
    std::ostringstream os;
    os << "__device__ constexpr int kTileOrder[] = {";
    for (int v : live)
        os << v << ",";
    for (int v : dead)
        os << v << ",";
    os << "};\nconstexpr int N_LIVE_SPATIAL_TILES = " << live.size() << ";\n\n";
    return os.str();
}

std::size_t c2p_shared_bytes(int n_order, int n_charge_dim, int chunk, std::size_t sizeof_real) {
    const int ld = chunk + 2;
    return (std::size_t(3) * n_order * ld + std::size_t(n_charge_dim) * ld) * sizeof_real;
}

template <typename Real>
void launch_zero_box_slabs(Real *flat, const long *offsets, const int *boxes, int n_boxes, int slab_reals,
                           cudaStream_t stream) {
    if (n_boxes == 0)
        return;
    constexpr int BLOCK = 256;
    static JitCache cache;
    JitKey key;
    key.name = "PtZeroBoxSlabsKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {{"BLOCK_SIZE", BLOCK}};
    auto kernel = cache.get_kernel_from_source(
        key, [&] { return make_stage_source("pt/shared_state.cu", key, "", "SharedState"); });
    kernel->launch(dim3(n_boxes, 1, 1), dim3(BLOCK, 1, 1), 0, stream, flat, offsets, boxes, n_boxes, slab_reals);
}

} // namespace

template <typename Real, int DIM>
void upward(State<Real, DIM> &s, cudaStream_t stream) {
    if constexpr (DIM != 3) {
        throw std::runtime_error("pt::upward: long-range pipeline is 3D-only");
    } else {
        auto &w = s.worklists;
        auto &f = s.fourier;

        const std::size_t proxy_count = s.scratch.d_proxy_coeffs_upward.size();
        const bool is_stresslet = s.kernel == DMK_STRESSLET;

        // Every box with owned sources is assigned by charge2proxy or by its first child below,
        // so only the source-free boxes proxy2pw still reads need clearing.
        launch_zero_box_slabs<Real>(s.scratch.d_proxy_coeffs_upward.data(), s.scratch.d_proxy_offsets_upward.data(),
                                    w.d_proxy_zero_boxes.data(), w.n_proxy_zero_boxes,
                                    f.n_order * f.n_order * f.n_order * f.n_tables_up, stream);

        // ---- charge2proxy (leaf sources -> center-box proxy coeffs) ----
        if (w.n_c2p_groups && w.n_c2p_active_groups) {
            static JitCache cache;
            dmk::cuda::Charge2ProxyArgs<Real> a;
            a.n_groups = w.n_c2p_groups;
            a.n_order = f.n_order;
            a.n_charge_dim = f.n_tables_up;
            a.center_boxes = w.d_c2p_center_boxes.data();
            a.levels = w.d_c2p_levels.data();
            a.src_box_flat_offsets = w.d_c2p_src_box_flat_offsets.data();
            a.n_src_boxes_per_group = w.d_c2p_n_src_boxes_per_group.data();
            a.src_boxes_flat = w.d_c2p_src_boxes_flat.data();
            a.centers = f.d_centers.data();
            a.inv_box_scale = f.d_inv_box_scale.data();
            a.r_src = s.particles.d_r_src.data();
            a.r_src_offsets = s.particles.d_r_src_offsets.data();
            a.src_counts = s.particles.d_src_counts.data();
            a.charge = is_stresslet ? s.particles.d_charge_outer.data() : s.particles.d_charge.data();
            a.charge_offsets =
                is_stresslet ? s.particles.d_charge_outer_offsets.data() : s.particles.d_charge_offsets.data();
            a.proxy_flat = s.scratch.d_proxy_coeffs_upward.data();
            a.proxy_offsets = s.scratch.d_proxy_offsets_upward.data();
            a.group_perm = w.d_c2p_group_perm.data();
            a.n_active_groups = w.n_c2p_active_groups;
            const int *group_perm = w.d_c2p_group_perm.data();
            const int n_launch = w.n_c2p_active_groups;

            auto launch_one = [&](const TuningParams &p, cudaStream_t st) {
                JitKey key;
                key.name = "PtCharge2ProxyKernel";
                key.real = jit_real_name<Real>();
                key.sm_major = cache.sm_major();
                key.sm_minor = cache.sm_minor();
                key.params = {{"N_ORDER", a.n_order},
                              {"N_CHARGE_DIM", a.n_charge_dim},
                              {"CHUNK", p.at("CHUNK")},
                              {"I_TILE", p.at("I_TILE")},
                              {"J_TILE", p.at("J_TILE")},
                              {"K_TILE", p.at("K_TILE")},
                              {"BLOCK_SIZE", p.at("BLOCK_SIZE")},
                              {"PROXY_BALL_R2", f.proxy_ball_r2}};
                const std::string prelude =
                    c2p_tile_order_prelude(a.n_order, p.at("I_TILE"), p.at("J_TILE"), p.at("K_TILE"), f.proxy_ball_r2);
                auto kernel = cache.get_kernel_from_source(
                    key, [&] { return make_stage_source("pt/charge2proxy.cu", key, prelude, "PtCharge2Proxy"); });
                const std::size_t shared = c2p_shared_bytes(a.n_order, a.n_charge_dim, p.at("CHUNK"), sizeof(Real));
                set_max_dynamic_smem(*kernel, shared);
                kernel->launch(dim3(n_launch, 1, 1), dim3(p.at("BLOCK_SIZE"), 1, 1), shared, st, a, group_perm);
            };

            std::ostringstream tune_key;
            tune_key << "PtCharge2Proxy|real=" << jit_real_name<Real>() << "|n_order=" << a.n_order
                     << "|n_charge_dim=" << a.n_charge_dim << "|ball_r2=" << f.proxy_ball_r2;
            tune_key << "|src=" << jit::jit_source_hash("pt/charge2proxy.cu");
            const std::string tk = tune_key.str();

            if (auto cfg = autotune_cached(tk)) {
                launch_one(*cfg, stream);
            } else {
                const cudaDeviceProp &prop = device_prop();
                const std::size_t max_shared = device_max_shared_bytes();
                const int n_order = f.n_order;
                const int n_charge_dim = a.n_charge_dim;

                const std::vector<TuningParameter> space{{"CHUNK", {128, 256, 512}},
                                                         {"I_TILE", {3, 4, 6, 9, n_order}},
                                                         {"J_TILE", {2, 3, 4}},
                                                         {"K_TILE", {2, 3, 4}},
                                                         {"BLOCK_SIZE", {128, 256}}};
                const TuningParams defaults{
                    {"CHUNK", 128}, {"I_TILE", 3}, {"J_TILE", 3}, {"K_TILE", 4}, {"BLOCK_SIZE", 128}};

                const auto constraint = [&, n_order, n_charge_dim](const TuningParams &p) {
                    const int ch = p.at("CHUNK"), it = p.at("I_TILE"), jt = p.at("J_TILE"), kt = p.at("K_TILE"),
                              bs = p.at("BLOCK_SIZE");
                    if (bs <= 0 || bs > prop.maxThreadsPerBlock || bs % 32 != 0)
                        return false;
                    if (ch <= 0 || it <= 0 || it > n_order || jt <= 0 || jt > n_order || kt <= 0 || kt > n_order)
                        return false;
                    if (it * jt * kt > 128)
                        return false;
                    return c2p_shared_bytes(n_order, n_charge_dim, ch, sizeof(Real)) <= max_shared;
                };

                autotuned_launch<Real>(tk, "PtCharge2ProxyKernel", space, defaults, constraint, launch_one,
                                       s.scratch.d_proxy_coeffs_upward.data(), proxy_count, stream);
            }
        }

        // ---- per-level upward tensorprod (deepest level first, gathered by parent) ----
        int tune_level = -1;
        for (int L = 0; L < s.n_levels; ++L)
            if (w.tp_up_count_h[L] && (tune_level < 0 || w.tp_up_par_count_h[L] > w.tp_up_par_count_h[tune_level]))
                tune_level = L;

        for (int L = s.n_levels - 1; L >= 0; --L) {
            const int n_pairs = w.tp_up_count_h[L];
            if (n_pairs == 0)
                continue;
            const int off = w.tp_up_offset_h[L];
            const int par_off = w.tp_up_par_offset_h[L];
            dmk::cuda::TensorprodArgs<Real> ta;
            ta.n_pairs = n_pairs;
            ta.n_order = f.n_order;
            ta.n_charge_dim = f.n_tables_up;
            ta.src_boxes = w.d_tp_up_src_boxes.data() + off;
            ta.dst_boxes = w.d_tp_up_dst_boxes.data() + off;
            ta.child_octants = w.d_tp_up_octants.data() + off;
            // Marks the first child of each parent charge2proxy does not write, so that child
            // assigns and the slab needs no pre-zeroing.
            ta.assign_dst = w.d_tp_up_assign.data() + off;
            // One block per parent walks that parent's children, so the accumulation is
            // block-local and needs no atomics.
            ta.par_boxes = w.d_tp_up_par.data() + par_off;
            ta.par_child_begin = w.d_tp_up_par_child_begin.data() + par_off;
            ta.par_child_count = w.d_tp_up_par_child_count.data() + par_off;
            ta.n_par = w.tp_up_par_count_h[L];
            ta.proxy_flat = s.scratch.d_proxy_coeffs_upward.data();
            ta.proxy_offsets = s.scratch.d_proxy_offsets_upward.data();
            ta.umat_flat = f.d_c2p.data();
            ta.scratch = s.scratch.d_tensorprod_scratch.data();
            ta.scratch_stride = s.scratch.tensorprod_scratch_stride_reals;
            launch_tensorprod<Real>(ta, proxy_count, stream, L == tune_level);
        }
    }
}

template void upward<float, 2>(State<float, 2> &, cudaStream_t);
template void upward<float, 3>(State<float, 3> &, cudaStream_t);
template void upward<double, 2>(State<double, 2> &, cudaStream_t);
template void upward<double, 3>(State<double, 3> &, cudaStream_t);

} // namespace dmk::cuda::pt
