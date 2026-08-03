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
    auto kernel =
        cache.get_kernel_from_source(key, [&] { return make_stage_source("pt/multiply.cu", key, "", "PtMultiply"); });
    dmk::cuda::MultiplyCd2pArgs<Real> a = args;
    kernel->launch(dim3(a.n_boxes_at_level, 1, 1), dim3(BLOCK, 1, 1), 0, stream, a);
}

// Stokeslet far-field projector f*(k^2 delta - kk), in place on the 3-table PW
// field. Needs 6 reals of shared for the windowed zero-mode correction.
template <typename Real>
void launch_multiply_stokeslet_3d(JitCache &cache, const dmk::cuda::MultiplyStokeslet3DArgs<Real> &args,
                                  cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;
    constexpr int BLOCK = 128;
    JitKey key;
    key.name = "PtMultiplyStokeslet3DByBoxKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {{"BLOCK_SIZE", BLOCK}};
    auto kernel =
        cache.get_kernel_from_source(key, [&] { return make_stage_source("pt/multiply.cu", key, "", "PtMultiply"); });
    dmk::cuda::MultiplyStokeslet3DArgs<Real> a = args;
    kernel->launch(dim3(a.n_boxes_at_level, 1, 1), dim3(BLOCK, 1, 1), 6 * sizeof(Real), stream, a);
}

// Stresslet far-field: contract a 3x3 PW tensor (9 src tables) into a 3-vector
// (3 dst tables). Distinct src/dst buffers.
template <typename Real>
void launch_multiply_stresslet_3d(JitCache &cache, const dmk::cuda::MultiplyStresslet3DArgs<Real> &args,
                                  cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;
    constexpr int BLOCK = 128;
    JitKey key;
    key.name = "PtMultiplyStresslet3DByBoxKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {{"BLOCK_SIZE", BLOCK}};
    auto kernel =
        cache.get_kernel_from_source(key, [&] { return make_stage_source("pt/multiply.cu", key, "", "PtMultiply"); });
    dmk::cuda::MultiplyStresslet3DArgs<Real> a = args;
    kernel->launch(dim3(a.n_boxes_at_level, 1, 1), dim3(BLOCK, 1, 1), 0, stream, a);
}

template <typename Real>
void launch_multiply_laplace_dipole_3d(JitCache &cache, const dmk::cuda::MultiplyLaplaceDipole3DArgs<Real> &args,
                                       cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;
    constexpr int BLOCK = 128;
    JitKey key;
    key.name = "PtMultiplyLaplaceDipole3DByBoxKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {{"BLOCK_SIZE", BLOCK}};
    auto kernel =
        cache.get_kernel_from_source(key, [&] { return make_stage_source("pt/multiply.cu", key, "", "PtMultiply"); });
    dmk::cuda::MultiplyLaplaceDipole3DArgs<Real> a = args;
    kernel->launch(dim3(a.n_boxes_at_level, 1, 1), dim3(BLOCK, 1, 1), 0, stream, a);
}

} // namespace

template <typename Real, int DIM>
void form_outgoing(State<Real, DIM> &s, cudaStream_t stream) {
    if constexpr (DIM != 3) {
        throw std::runtime_error("pt::form_outgoing: long-range pipeline is 3D-only");
    } else {
        const dmk_ikernel kernel = s.kernel;
        if (kernel != DMK_LAPLACE && kernel != DMK_SQRT_LAPLACE && kernel != DMK_STOKESLET && kernel != DMK_STRESSLET &&
            kernel != DMK_LAPLACE_DIPOLE)
            throw std::runtime_error("pt::form_outgoing: unsupported kernel");
        // Differing up/down table counts cannot multiply in place: the PW field is
        // formed per level into d_pw_form_pool. Stresslet is 9->3, dipole 3->1.
        const bool split_up_down = s.fourier.n_tables_up != s.fourier.n_charge_dim;

        auto &f = s.fourier;
        auto &w = s.worklists;
        auto &sc = s.scratch;
        static JitCache multiply_cache;

        sc.d_pw_out.zero_async(stream);
        sc.d_proxy_coeffs_downward.zero_async(stream);

        // Apply the kernel FT at a given PW size. Scalar/Stokeslet operate in
        // place on `src`; Stresslet reads 9 tables from `src` and writes 3 to
        // `dst`.
        auto multiply_at = [&](int n_box, int n_pw_local, int n_pw_modes_local, Real hpw_local, bool windowed,
                               const int *box_ids, const Real *radialft, Real *src, const long *src_offsets,
                               long src_stride_complex, Real *dst, const long *dst_offsets, long dst_stride_complex) {
            if (kernel == DMK_LAPLACE || kernel == DMK_SQRT_LAPLACE) {
                dmk::cuda::MultiplyCd2pArgs<Real> ma;
                ma.n_boxes_at_level = n_box;
                ma.n_charge_dim = f.n_charge_dim;
                ma.n_pw_modes = n_pw_modes_local;
                ma.box_ids = box_ids;
                ma.radialft = radialft;
                ma.pw_flat = src;
                ma.pw_offsets = src_offsets;
                ma.pw_stride_complex = src_stride_complex;
                launch_multiply_cd2p<Real>(multiply_cache, ma, stream);
            } else if (kernel == DMK_STOKESLET) {
                dmk::cuda::MultiplyStokeslet3DArgs<Real> ma;
                ma.n_boxes_at_level = n_box;
                ma.n_pw = n_pw_local;
                ma.n_pw2 = (n_pw_local + 1) / 2;
                ma.n_pw_modes = n_pw_modes_local;
                ma.hpw = hpw_local;
                ma.is_windowed = windowed;
                ma.box_ids = box_ids;
                ma.radialft = radialft;
                ma.pw_flat = src;
                ma.pw_offsets = src_offsets;
                ma.pw_stride_complex = src_stride_complex;
                launch_multiply_stokeslet_3d<Real>(multiply_cache, ma, stream);
            } else if (kernel == DMK_LAPLACE_DIPOLE) {
                dmk::cuda::MultiplyLaplaceDipole3DArgs<Real> ma;
                ma.n_boxes_at_level = n_box;
                ma.n_pw = n_pw_local;
                ma.n_pw_modes = n_pw_modes_local;
                ma.hpw = hpw_local;
                ma.box_ids = box_ids;
                ma.radialft = radialft;
                ma.src_flat = src;
                ma.src_offsets = src_offsets;
                ma.src_stride_complex = src_stride_complex;
                ma.dst_flat = dst;
                ma.dst_offsets = dst_offsets;
                ma.dst_stride_complex = dst_stride_complex;
                launch_multiply_laplace_dipole_3d<Real>(multiply_cache, ma, stream);
            } else { // Stresslet
                dmk::cuda::MultiplyStresslet3DArgs<Real> ma;
                ma.n_boxes_at_level = n_box;
                ma.n_pw = n_pw_local;
                ma.n_pw2 = (n_pw_local + 1) / 2;
                ma.n_pw_modes = n_pw_modes_local;
                ma.hpw = hpw_local;
                ma.box_ids = box_ids;
                ma.radialft = radialft;
                ma.src_flat = src;
                ma.src_offsets = src_offsets;
                ma.src_stride_complex = src_stride_complex;
                ma.dst_flat = dst;
                ma.dst_offsets = dst_offsets;
                ma.dst_stride_complex = dst_stride_complex;
                launch_multiply_stresslet_3d<Real>(multiply_cache, ma, stream);
            }
        };

        // Matched table counts: proxy2pw for all levels up front into d_pw_out.
        // Otherwise proxy2pw runs per level into d_pw_form_pool alongside the
        // multiply (below).
        if (!split_up_down) {
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
        }

        // Per-level multiply (split_up_down first fills the form pool per level).
        for (int L = 0; L < s.n_levels; ++L) {
            const int n_box = w.pw_form_box_count_h[L];
            if (n_box == 0)
                continue;
            const int box_off = w.pw_form_box_offset_h[L];
            const int *box_ids = w.d_pw_form_box_flat.data() + box_off;

            if (split_up_down) {
                std::vector<dmk::cuda::Proxy2PwArgs<Real>> pa_h(1);
                auto &pa = pa_h[0];
                pa.n_boxes_at_level = n_box;
                pa.n_order = f.n_order;
                pa.n_pw = f.n_pw;
                pa.n_pw2 = f.n_pw2;
                pa.n_charge_dim = f.n_tables_up;
                pa.box_ids = box_ids;
                pa.proxy_flat = sc.d_proxy_coeffs_upward.data();
                pa.proxy_offsets = sc.d_proxy_offsets_upward.data();
                pa.poly2pw = f.slab(L).poly2pw;
                pa.dst_flat = sc.d_pw_form_pool.data();
                pa.dst_offsets = nullptr;
                pa.dst_stride_complex = sc.pw_form_stride_reals / 2;
                launch_proxy2pw<Real>(pa_h, stream);
            }

            Real *src = split_up_down ? sc.d_pw_form_pool.data() : sc.d_pw_out.data();
            const long *src_offsets = split_up_down ? nullptr : sc.d_pw_out_offsets.data();
            const long src_stride = split_up_down ? sc.pw_form_stride_reals / 2 : 0L;
            multiply_at(n_box, f.n_pw, f.n_pw_modes, f.hpw_per_level[L], /*windowed=*/false, box_ids,
                        f.slab(L).radialft, src, src_offsets, src_stride, sc.d_pw_out.data(),
                        sc.d_pw_out_offsets.data(), 0);
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
        launch_proxy2pw<Real>(root_pa, stream, "root");

        const long window_out_stride_complex = static_cast<long>(f.n_charge_dim) * f.n_pw_modes_win;
        multiply_at(1, f.n_pw_win, f.n_pw_modes_win, f.hpw_win, /*windowed=*/true, sc.d_box0_id.data(),
                    f.d_window_radialft.data(), sc.d_window_pw_form_in.data(), nullptr, window_in_stride_complex,
                    sc.d_window_pw_form_out.data(), nullptr, window_out_stride_complex);

        Real *pw_for_pw2proxy = split_up_down ? sc.d_window_pw_form_out.data() : sc.d_window_pw_form_in.data();

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
            pp.pw_in_pool = pw_for_pw2proxy;
            pp.pw2poly = f.d_window_pw2poly.data();
            pp.proxy_flat = sc.d_proxy_coeffs_downward.data();
            pp.proxy_offsets = sc.d_proxy_offsets_downward.data();
        }
        launch_pw2proxy<Real>(root_pp, sc.d_proxy_coeffs_downward.data(), sc.d_proxy_coeffs_downward.size(), stream,
                              "root");
    }
}

template void form_outgoing<float, 2>(State<float, 2> &, cudaStream_t);
template void form_outgoing<float, 3>(State<float, 3> &, cudaStream_t);
template void form_outgoing<double, 2>(State<double, 2> &, cudaStream_t);
template void form_outgoing<double, 3>(State<double, 3> &, cudaStream_t);

} // namespace dmk::cuda::pt
