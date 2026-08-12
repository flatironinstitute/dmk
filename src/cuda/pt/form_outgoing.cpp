#include <dmk/cuda/pt/passes.hpp>

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"
#include "proxy2pw.hpp"
#include "pw2proxy.hpp"

#include <dmk.h>
#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/multiply_kernelft_kernelargs.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda::pt {
namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

// Per-box in-place kernel-FT multiply (Laplace / Sqrt-Laplace / Yukawa), batched over a
// device array of per-level args. Fixed block size (no autotune / no snapshot).
template <typename Real>
void launch_multiply_cd2p(JitCache &cache, std::vector<dmk::cuda::MultiplyCd2pArgs<Real>> &args_h,
                          cudaStream_t stream) {
    if (args_h.empty())
        return;
    int max_boxes = 0;
    for (const auto &a : args_h)
        max_boxes = std::max(max_boxes, a.n_boxes_at_level);
    if (max_boxes == 0)
        return;

    constexpr int BLOCK = 128;
    static cuda_helpers::DeviceBuffer<dmk::cuda::MultiplyCd2pArgs<Real>> d_args;
    d_args.upload_async_grow(args_h.data(), args_h.size(), stream);
    const int n_args = static_cast<int>(args_h.size());

    JitKey key;
    key.name = "PtMultiplyCd2pMultiLevelKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {{"BLOCK_SIZE", BLOCK}};
    auto kernel =
        cache.get_kernel_from_source(key, [&] { return make_stage_source("pt/multiply.cu", key, "", "PtMultiply"); });
    const dmk::cuda::MultiplyCd2pArgs<Real> *dev_args = d_args.data();
    int n = n_args;
    kernel->launch(dim3(max_boxes, n_args, 1), dim3(BLOCK, 1, 1), 0, stream, dev_args, n);
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
        if (kernel != DMK_LAPLACE && kernel != DMK_SQRT_LAPLACE && kernel != DMK_YUKAWA && kernel != DMK_STOKESLET &&
            kernel != DMK_STRESSLET && kernel != DMK_LAPLACE_DIPOLE)
            throw std::runtime_error("pt::form_outgoing: unsupported kernel");
        // Differing up/down table counts cannot multiply in place: the PW field is
        // formed per level into d_pw_form_pool. Stresslet is 9->3, dipole 3->1.
        const bool split_up_down = s.fourier.n_tables_up != s.fourier.n_charge_dim;

        auto &f = s.fourier;
        auto &w = s.worklists;
        auto &sc = s.scratch;
        static JitCache multiply_cache;

        // Applying the kernel FT at proxy2pw's phase-3 store, where the modes are already in
        // registers, removes a whole read+write pass over the PW field. Scalar kernels just scale
        // by radialft; Stokeslet mixes the three charge dims, so it also needs a phase-2 shared
        // buffer per dim, which does not always fit. Stresslet (9->3) and Laplace-dipole (3->1)
        // write a different table count than they read and stay unfused. DMK_FUSE_MULTIPLY=0 disables.
        const int fuse_mode = [&] {
            const char *v = std::getenv("DMK_FUSE_MULTIPLY");
            if (v && std::atoi(v) == 0)
                return 0;
            if (kernel == DMK_LAPLACE || kernel == DMK_SQRT_LAPLACE || kernel == DMK_YUKAWA)
                return 1;
            if (kernel == DMK_STOKESLET)
                return 2;
            return 0;
        }();
        const int ff2_copies = (fuse_mode == 2) ? f.n_tables_up : 1;
        const auto fits = [&](int n_pw) {
            return proxy2pw_min_shared_bytes(f.n_order, n_pw, ff2_copies, sizeof(Real)) <= device_max_shared_bytes();
        };
        const int fuse_levels = (fuse_mode && fits(f.n_pw)) ? fuse_mode : 0;
        const int fuse_root = (fuse_mode && fits(f.n_pw_win)) ? fuse_mode : 0;

        // Scalar multiplies accumulate here and go out in one launch. Safe for every level at
        // once because the Cd2p path implies !split_up_down (its kernels have
        // n_tables_up == n_charge_dim), so no level shares the form pool with another.
        std::vector<dmk::cuda::MultiplyCd2pArgs<Real>> cd2p_batch;
        const auto flush_cd2p = [&] {
            launch_multiply_cd2p<Real>(multiply_cache, cd2p_batch, stream);
            cd2p_batch.clear();
        };

        // Apply the kernel FT at a given PW size. Scalar/Stokeslet operate in
        // place on `src`; Stresslet reads 9 tables from `src` and writes 3 to
        // `dst`.
        auto multiply_at = [&](int n_box, int n_pw_local, int n_pw_modes_local, int n_pw_live_local, const int *cube_of,
                               Real hpw_local, bool windowed, const int *box_ids, const Real *radialft, Real *src,
                               const long *src_offsets, long src_stride_complex, Real *dst, const long *dst_offsets,
                               long dst_stride_complex) {
            if (kernel == DMK_LAPLACE || kernel == DMK_SQRT_LAPLACE || kernel == DMK_YUKAWA) {
                dmk::cuda::MultiplyCd2pArgs<Real> ma;
                ma.n_boxes_at_level = n_box;
                ma.n_charge_dim = f.n_charge_dim;
                ma.n_pw_modes = n_pw_modes_local;
                ma.n_pw_live = n_pw_live_local;
                ma.full_of_compact = cube_of;
                ma.box_ids = box_ids;
                ma.radialft = radialft;
                ma.pw_flat = src;
                ma.pw_offsets = src_offsets;
                ma.pw_stride_complex = src_stride_complex;
                cd2p_batch.push_back(ma);
            } else if (kernel == DMK_STOKESLET) {
                dmk::cuda::MultiplyStokeslet3DArgs<Real> ma;
                ma.n_boxes_at_level = n_box;
                ma.n_pw = n_pw_local;
                ma.n_pw2 = (n_pw_local + 1) / 2;
                ma.n_pw_modes = n_pw_modes_local;
                ma.n_pw_live = n_pw_live_local;
                ma.full_of_compact = cube_of;
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
                ma.n_pw_live = n_pw_live_local;
                ma.full_of_compact = cube_of;
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
                ma.n_pw_live = n_pw_live_local;
                ma.full_of_compact = cube_of;
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
                pa.pencil_slots = f.d_pencil_slots.data();
                pa.multiply_mode = fuse_levels;
                pa.radialft = f.slab(L).radialft;
                pa.hpw = f.hpw_per_level[L];
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
                pa.pencil_slots = f.d_pencil_slots.data();
                launch_proxy2pw<Real>(pa_h, stream);
            }

            if (fuse_levels)
                continue;

            Real *src = split_up_down ? sc.d_pw_form_pool.data() : sc.d_pw_out.data();
            const long *src_offsets = split_up_down ? nullptr : sc.d_pw_out_offsets.data();
            const long src_stride = split_up_down ? sc.pw_form_stride_reals / 2 : 0L;
            multiply_at(n_box, f.n_pw, f.n_pw_modes, f.n_pw_live, f.d_full_of_compact.data(), f.hpw_per_level[L],
                        /*windowed=*/false, box_ids, f.slab(L).radialft, src, src_offsets, src_stride,
                        sc.d_pw_out.data(), sc.d_pw_out_offsets.data(), 0);
        }
        flush_cd2p();

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
            pa.multiply_mode = fuse_root;
            pa.radialft = f.d_window_radialft.data();
            pa.hpw = f.hpw_win;
            pa.is_windowed = 1;
        }
        launch_proxy2pw<Real>(root_pa, stream, "root");

        const long window_out_stride_complex = static_cast<long>(f.n_charge_dim) * f.n_pw_modes_win;
        if (!fuse_root) {
            multiply_at(1, f.n_pw_win, f.n_pw_modes_win, f.n_pw_modes_win, /*cube_of=*/nullptr, f.hpw_win,
                        /*windowed=*/true, sc.d_box0_id.data(), f.d_window_radialft.data(),
                        sc.d_window_pw_form_in.data(), nullptr, window_in_stride_complex,
                        sc.d_window_pw_form_out.data(), nullptr, window_out_stride_complex);
            // The root's modes and radialft differ from any level's, so it is its own launch.
            flush_cd2p();
        }

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
            // First writer of box 0 in the downward buffer.
            pp.assign = 1;
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
