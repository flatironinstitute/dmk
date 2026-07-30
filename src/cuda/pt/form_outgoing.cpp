#include <dmk/cuda/pt/passes.hpp>

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"
#include "proxy2pw.hpp"
#include "pw2proxy.hpp"

#include <dmk.h>
#include <dmk/cuda/multiply_kernelft_kernelargs.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda::pt {
namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

// Per-box in-place kernel-FT multiply (Laplace / Sqrt-Laplace). Fixed block
// size, single launch (no autotune / no snapshot).
template <typename Real>
void launch_multiply_cd2p(JitCache &cache, const dmk::cuda::MultiplyCd2pArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;
    constexpr int BLOCK = 128;
    JitKey key;
    key.name = "PtMultiplyCd2pKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {{"BLOCK_SIZE", BLOCK}};
    const std::string source = make_stage_source("pt/multiply.cu", key, "", "PtMultiply");
    auto kernel = cache.get_kernel_from_source(key, source);
    dmk::cuda::MultiplyCd2pArgs<Real> a = args;
    kernel->launch(dim3(a.n_boxes_at_level, 1, 1), dim3(BLOCK, 1, 1), 0, stream, a);
}

} // namespace

template <typename Real, int DIM>
void form_outgoing(State<Real, DIM> &s, cudaStream_t stream) {
    if constexpr (DIM != 3) {
        throw std::runtime_error("pt::form_outgoing: long-range pipeline is 3D-only");
    } else {
        if (s.kernel != DMK_LAPLACE && s.kernel != DMK_SQRT_LAPLACE)
            throw std::runtime_error("pt::form_outgoing: step 3b supports Laplace / Sqrt-Laplace only");

        auto &f = s.fourier;
        auto &w = s.worklists;
        auto &sc = s.scratch;
        static JitCache multiply_cache;

        sc.d_pw_out.zero_async(stream);
        sc.d_proxy_coeffs_downward.zero_async(stream);

        // ---- per-level proxy2pw -> d_pw_out (one batched launch) ----
        std::vector<dmk::cuda::Proxy2PwArgs<Real>> pa_h;
        for (int L = 0; L < s.n_levels; ++L) {
            const int n_box = w.pw_form_box_count_h[L];
            if (n_box == 0)
                continue;
            const int box_off = w.pw_form_box_offset_h[L];
            dmk::cuda::Proxy2PwArgs<Real> pa;
            pa.n_boxes_at_level = n_box;
            pa.n_order = f.n_order;
            pa.n_pw = f.n_pw;
            pa.n_pw2 = f.n_pw2;
            pa.n_charge_dim = f.n_tables_up;
            pa.box_ids = w.d_pw_form_box_flat.data() + box_off;
            pa.proxy_flat = sc.d_proxy_coeffs_upward.data();
            pa.proxy_offsets = sc.d_proxy_offsets_upward.data();
            pa.poly2pw = f.slab(L).poly2pw;
            pa.dst_flat = sc.d_pw_out.data();
            pa.dst_offsets = sc.d_pw_out_offsets.data();
            pa.dst_stride_complex = 0;
            pa_h.push_back(pa);
        }
        launch_proxy2pw<Real>(pa_h, stream);

        // ---- per-level kernel-FT multiply (in place on d_pw_out) ----
        for (int L = 0; L < s.n_levels; ++L) {
            const int n_box = w.pw_form_box_count_h[L];
            if (n_box == 0)
                continue;
            const int box_off = w.pw_form_box_offset_h[L];
            dmk::cuda::MultiplyCd2pArgs<Real> ma;
            ma.n_boxes_at_level = n_box;
            ma.n_charge_dim = f.n_charge_dim;
            ma.n_pw_modes = f.n_pw_modes;
            ma.box_ids = w.d_pw_form_box_flat.data() + box_off;
            ma.radialft = f.slab(L).radialft;
            ma.pw_flat = sc.d_pw_out.data();
            ma.pw_offsets = sc.d_pw_out_offsets.data();
            ma.pw_stride_complex = 0;
            launch_multiply_cd2p<Real>(multiply_cache, ma, stream);
        }

        // ---- windowed root -> d_proxy_coeffs_downward[0] ----
        const long window_in_stride_complex = static_cast<long>(f.n_tables_up) * f.n_pw_modes_win;

        std::vector<dmk::cuda::Proxy2PwArgs<Real>> root_pa(1);
        {
            auto &pa = root_pa[0];
            pa.n_boxes_at_level = 1;
            pa.n_order = f.n_order;
            pa.n_pw = f.n_pw_win;
            pa.n_pw2 = f.n_pw2_win;
            pa.n_charge_dim = f.n_tables_up;
            pa.box_ids = sc.d_box0_id.data();
            pa.proxy_flat = sc.d_proxy_coeffs_upward.data();
            pa.proxy_offsets = sc.d_proxy_offsets_upward.data();
            pa.poly2pw = f.d_window_poly2pw.data();
            pa.dst_flat = sc.d_window_pw_form_in.data();
            pa.dst_offsets = nullptr;
            pa.dst_stride_complex = window_in_stride_complex;
        }
        launch_proxy2pw<Real>(root_pa, stream);

        dmk::cuda::MultiplyCd2pArgs<Real> ma;
        ma.n_boxes_at_level = 1;
        ma.n_charge_dim = f.n_charge_dim;
        ma.n_pw_modes = f.n_pw_modes_win;
        ma.box_ids = sc.d_box0_id.data();
        ma.radialft = f.d_window_radialft.data();
        ma.pw_flat = sc.d_window_pw_form_in.data();
        ma.pw_offsets = nullptr;
        ma.pw_stride_complex = window_in_stride_complex;
        launch_multiply_cd2p<Real>(multiply_cache, ma, stream);

        std::vector<dmk::cuda::PwToProxyArgs<Real>> root_pp(1);
        {
            auto &pp = root_pp[0];
            pp.n_boxes_at_level = 1;
            pp.n_order = f.n_order;
            pp.n_pw = f.n_pw_win;
            pp.n_pw2 = f.n_pw2_win;
            pp.n_charge_dim = f.n_charge_dim;
            pp.pw_in_stride = 0;
            pp.box_ids = sc.d_box0_id.data();
            pp.pw_in_pool = sc.d_window_pw_form_in.data();
            pp.pw2poly = f.d_window_pw2poly.data();
            pp.proxy_flat = sc.d_proxy_coeffs_downward.data();
            pp.proxy_offsets = sc.d_proxy_offsets_downward.data();
        }
        launch_pw2proxy<Real>(root_pp, sc.d_proxy_coeffs_downward.data(), sc.d_proxy_coeffs_downward.size(), stream);
    }
}

template void form_outgoing<float, 2>(State<float, 2> &, cudaStream_t);
template void form_outgoing<float, 3>(State<float, 3> &, cudaStream_t);
template void form_outgoing<double, 2>(State<double, 2> &, cudaStream_t);
template void form_outgoing<double, 3>(State<double, 3> &, cudaStream_t);

} // namespace dmk::cuda::pt
