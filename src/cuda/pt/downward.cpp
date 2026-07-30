#include <dmk/cuda/pt/passes.hpp>

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"
#include "pw2proxy.hpp"
#include "tensorprod.hpp"

#include <dmk.h>
#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/pw2proxy_kernelargs.hpp>
#include <dmk/cuda/shift_pw_kernelargs.hpp>
#include <dmk/cuda/tensorprod_kernelargs.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda::pt {
namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

// Batched neighbor plane-wave translation (one launch, all levels; n=1 unused).
// Assigns into each box's pw_in_pool slab, so it is idempotent (no snapshot).
template <typename Real>
void launch_shift_pw(std::vector<dmk::cuda::ShiftPwArgs<Real>> &args_h, cudaStream_t stream) {
    if (args_h.empty())
        return;
    int max_boxes = 0;
    for (const auto &a : args_h)
        max_boxes = std::max(max_boxes, a.n_boxes_at_level);
    if (max_boxes == 0)
        return;

    static JitCache cache;
    static cuda_helpers::DeviceBuffer<dmk::cuda::ShiftPwArgs<Real>> d_args;
    d_args.upload_async(args_h.data(), args_h.size(), stream);
    const int n_args = static_cast<int>(args_h.size());
    const auto a0 = args_h[0];

    auto launch_one = [&](const TuningParams &p, cudaStream_t st) {
        JitKey key;
        key.name = "PtShiftPwKernel";
        key.real = jit_real_name<Real>();
        key.sm_major = cache.sm_major();
        key.sm_minor = cache.sm_minor();
        key.params = {{"N_PW_MODES", a0.n_pw_modes},
                      {"N_CHARGE_DIM", a0.n_charge_dim},
                      {"N_NEIGHBORS", a0.n_neighbors},
                      {"BLOCK_SIZE", p.at("BLOCK_SIZE")}};
        const std::string source = make_stage_source("pt/shiftpw.cu", key, "", "PtShiftPw");
        auto kernel = cache.get_kernel_from_source(key, source);
        const dmk::cuda::ShiftPwArgs<Real> *dev_args = d_args.data();
        int n = n_args;
        kernel->launch(dim3(max_boxes, n_args, 1), dim3(p.at("BLOCK_SIZE"), 1, 1), 0, st, dev_args, n);
    };

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    const std::vector<TuningParameter> space{{"BLOCK_SIZE", {64, 128, 256, 512, 768}}};
    const TuningParams defaults{{"BLOCK_SIZE", 256}};
    const auto constraint = [&](const TuningParams &p) {
        const int bs = p.at("BLOCK_SIZE");
        return bs > 0 && bs <= prop.maxThreadsPerBlock && bs % 32 == 0;
    };

    std::ostringstream tune_key;
    tune_key << "PtShiftPw|real=" << jit_real_name<Real>() << "|n_pw_modes=" << a0.n_pw_modes
             << "|n_charge_dim=" << a0.n_charge_dim << "|n_neighbors=" << a0.n_neighbors;

    autotuned_launch<Real>(tune_key.str(), "PtShiftPwKernel", space, defaults, constraint, launch_one,
                           /*snapshot_base=*/static_cast<Real *>(nullptr), 0, stream);
}

} // namespace

template <typename Real, int DIM>
void downward(State<Real, DIM> &s, cudaStream_t stream) {
    if constexpr (DIM != 3) {
        throw std::runtime_error("pt::downward: long-range pipeline is 3D-only");
    } else {
        auto &f = s.fourier;
        auto &w = s.worklists;
        auto &sc = s.scratch;

        // ---- per-level shift_pw (-> pw_in_pool) + pw2proxy (-> proxy_downward) ----
        std::vector<dmk::cuda::ShiftPwArgs<Real>> shift_h;
        std::vector<dmk::cuda::PwToProxyArgs<Real>> pw2p_h;
        for (int L = 0; L < s.n_levels; ++L) {
            const int n_box = w.pw_eval_box_count_h[L];
            if (n_box == 0)
                continue;
            const int box_off = w.pw_eval_box_offset_h[L];
            Real *level_pw_in = sc.d_pw_in_pool.data() + w.pw_in_pool_base_h[L] * sc.pw_in_stride_reals;

            dmk::cuda::ShiftPwArgs<Real> sa;
            sa.n_boxes_at_level = n_box;
            sa.n_neighbors = s.topology.n_neighbors;
            sa.n_charge_dim = f.n_charge_dim;
            sa.n_pw_modes = f.n_pw_modes;
            sa.pw_in_stride = sc.pw_in_stride_reals;
            sa.box_ids = w.d_pw_eval_box_flat.data() + box_off;
            sa.neighbors = s.topology.d_neighbors.data();
            sa.pw_out_offsets = sc.d_pw_out_offsets.data();
            sa.is_global_leaf = s.topology.d_is_global_leaf.data();
            sa.pw_out_flat = sc.d_pw_out.data();
            sa.wpwshift = f.slab(L).wpwshift;
            sa.pw_in_pool = level_pw_in;
            shift_h.push_back(sa);

            dmk::cuda::PwToProxyArgs<Real> pa;
            pa.n_boxes_at_level = n_box;
            pa.n_order = f.n_order;
            pa.n_pw = f.n_pw;
            pa.n_pw2 = f.n_pw2;
            pa.n_charge_dim = f.n_charge_dim;
            pa.pw_in_stride = sc.pw_in_stride_reals;
            pa.box_ids = w.d_pw_eval_box_flat.data() + box_off;
            pa.pw_in_pool = level_pw_in;
            pa.pw2poly = f.slab(L).pw2poly;
            pa.proxy_flat = sc.d_proxy_coeffs_downward.data();
            pa.proxy_offsets = sc.d_proxy_offsets_downward.data();
            pw2p_h.push_back(pa);
        }

        launch_shift_pw<Real>(shift_h, stream);
        launch_pw2proxy<Real>(pw2p_h, sc.d_proxy_coeffs_downward.data(), sc.d_proxy_coeffs_downward.size(), stream);

        // ---- per-level downward tensorprod (parent->child, p2c, non-atomic add) ----
        for (int L = 0; L < s.n_levels; ++L) {
            const int n_pairs = w.tp_count_h[L];
            if (n_pairs == 0)
                continue;
            const int off = w.tp_offset_h[L];
            dmk::cuda::TensorprodArgs<Real> ta;
            ta.n_pairs = n_pairs;
            ta.n_order = f.n_order;
            ta.n_charge_dim = f.n_charge_dim;
            ta.src_boxes = w.d_tp_parents.data() + off;
            ta.dst_boxes = w.d_tp_children.data() + off;
            ta.child_octants = w.d_tp_octants.data() + off;
            ta.proxy_flat = sc.d_proxy_coeffs_downward.data();
            ta.proxy_offsets = sc.d_proxy_offsets_downward.data();
            ta.umat_flat = f.d_p2c.data();
            ta.scratch = sc.d_tensorprod_scratch.data();
            ta.scratch_stride = sc.tensorprod_scratch_stride_reals;
            ta.additive_atomic = false;
            launch_tensorprod<Real>(ta, sc.d_proxy_coeffs_downward.size(), stream);
        }
    }
}

template void downward<float, 2>(State<float, 2> &, cudaStream_t);
template void downward<float, 3>(State<float, 3> &, cudaStream_t);
template void downward<double, 2>(State<double, 2> &, cudaStream_t);
template void downward<double, 3>(State<double, 3> &, cudaStream_t);

} // namespace dmk::cuda::pt
