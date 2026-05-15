#include <cuda_runtime.h>
#include <dmk/cuda/form_outgoing.hpp>
#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/multiply_kernelft_kernels.hpp>
#include <dmk/cuda/proxy2pw_kernels.hpp>
#include <dmk/cuda/pw_to_proxy_kernels.hpp>
#include <dmk/cuda/shared_state.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/nvtx_wrapper.h>
#include <dmk/tree.hpp>
#include <stdexcept>
#include <vector>

namespace dmk {

template <typename Real, int DIM>
CudaFormOutgoingContext<Real, DIM>::CudaFormOutgoingContext(DMKPtTree<Real, DIM> &tree,
                                                            CudaSharedDeviceState<Real, DIM> &shared)
    : tree_(tree), shared_(shared) {
    if (DIM != 3)
        throw std::runtime_error("CUDA form_outgoing: only DIM=3 supported");
}

template <typename Real, int DIM>
void CudaFormOutgoingContext<Real, DIM>::run() {
    auto &t = tree_;
    auto &s = shared_;

    s.allocate_pw_out(t);
    if (!s.proxy_upward_resident_on_device)
        s.upload_proxy_upward(t);

    // Reset both d_pw_out (covers boxes with no upward proxy: pw_out = 0)
    // and the root box's slot of d_proxy_coeffs_downward (the windowed
    // contribution overwrites it below).
    s.d_pw_out.zero_async(s.downward_stream);
    s.d_proxy_coeffs_downward.zero_async(s.downward_stream);

    const int n_levels = s.n_levels;
    const dmk_ikernel kernel = s.kernel;
    const bool is_stresslet = (kernel == DMK_STRESSLET);

    // proxy2pw: non-stresslet writes its n_tables_down complex tables straight
    // into d_pw_out via a single multilevel batch. Stresslet writes
    // n_tables_up = 9 tables, which don't fit d_pw_out (sized for n_tables_down
    // = 3), so it lands in d_pw_form_pool and we issue per-level launches.
    nvtxRangePush("proxy2pw");
    std::vector<cuda::Proxy2PwArgs<Real>> pa_h;
    for (int L = 0; L < n_levels; ++L) {
        const int n_box = s.pw_form_box_count_h[L];
        if (n_box == 0)
            continue;
        const int box_offset = s.pw_form_box_offset_h[L];

        cuda::Proxy2PwArgs<Real> pa;
        pa.n_boxes_at_level = n_box;
        pa.n_order = s.n_order;
        pa.n_pw = s.n_pw;
        pa.n_pw2 = s.n_pw2;
        pa.n_charge_dim = s.n_tables_up;
        pa.box_ids = s.d_pw_form_box_flat.data() + box_offset;
        pa.proxy_flat = s.d_proxy_coeffs_upward.data();
        pa.proxy_offsets = s.d_proxy_offsets_upward.data();
        pa.poly2pw = s.d_poly2pw_flat.data() + L * s.poly2pw_per_level_reals;
        if (!is_stresslet) {
            pa.dst_flat = s.d_pw_out.data();
            pa.dst_offsets = s.d_pw_out_offsets.data();
            pa.dst_stride_complex = 0;
            pa_h.push_back(pa);
        } else {
            pa.dst_flat = s.d_pw_form_pool.data();
            pa.dst_offsets = nullptr;
            pa.dst_stride_complex = s.pw_form_stride_reals / 2;
            cuda::launch_proxy2pw<Real, DIM>(pa, s.downward_stream);
        }
    }
    if (!is_stresslet)
        cuda::launch_proxy2pw_multilevel<Real, DIM>(pa_h, s.d_proxy2pw_args.data(), s.downward_stream);
    nvtxRangePop();

    // multiply_kernelFT: per-level, kernel-specific formula applied to the
    // proxy2pw output. cd2p/stokeslet operate in place on d_pw_out; stresslet
    // reads from d_pw_form_pool (9 tables) and writes to d_pw_out (3).
    auto multiply_at = [&](int n_box, int n_pw_local, int n_pw2_local, int n_pw_modes_local, Real hpw_local,
                           bool windowed, const int *box_ids, const Real *radialft, Real *src, const long *src_offsets,
                           long src_stride_complex, Real *stresslet_dst, const long *stresslet_dst_offsets,
                           long stresslet_dst_stride) {
        if (kernel == DMK_LAPLACE || kernel == DMK_SQRT_LAPLACE) {
            cuda::MultiplyCd2pArgs<Real> ma;
            ma.n_boxes_at_level = n_box;
            ma.n_charge_dim = s.n_charge_dim;
            ma.n_pw_modes = n_pw_modes_local;
            ma.box_ids = box_ids;
            ma.radialft = radialft;
            ma.pw_flat = src;
            ma.pw_offsets = src_offsets;
            ma.pw_stride_complex = src_stride_complex;
            cuda::launch_multiply_cd2p<Real, DIM>(ma, s.downward_stream);
        } else if (kernel == DMK_STOKESLET) {
            cuda::MultiplyStokeslet3DArgs<Real> ma;
            ma.n_boxes_at_level = n_box;
            ma.n_pw = n_pw_local;
            ma.n_pw2 = n_pw2_local;
            ma.n_pw_modes = n_pw_modes_local;
            ma.hpw = hpw_local;
            ma.is_windowed = windowed;
            ma.box_ids = box_ids;
            ma.radialft = radialft;
            ma.pw_flat = src;
            ma.pw_offsets = src_offsets;
            ma.pw_stride_complex = src_stride_complex;
            cuda::launch_multiply_stokeslet_3d<Real>(ma, s.downward_stream);
        } else if (kernel == DMK_STRESSLET) {
            cuda::MultiplyStresslet3DArgs<Real> ma;
            ma.n_boxes_at_level = n_box;
            ma.n_pw = n_pw_local;
            ma.n_pw2 = n_pw2_local;
            ma.n_pw_modes = n_pw_modes_local;
            ma.hpw = hpw_local;
            ma.box_ids = box_ids;
            ma.radialft = radialft;
            ma.src_flat = src;
            ma.src_offsets = src_offsets;
            ma.src_stride_complex = src_stride_complex;
            ma.dst_flat = stresslet_dst;
            ma.dst_offsets = stresslet_dst_offsets;
            ma.dst_stride_complex = stresslet_dst_stride;
            cuda::launch_multiply_stresslet_3d<Real>(ma, s.downward_stream);
        } else {
            throw std::runtime_error("CUDA form_outgoing: unsupported kernel");
        }
    };

    for (int L = 0; L < n_levels; ++L) {
        const int n_box = s.pw_form_box_count_h[L];
        if (n_box == 0)
            continue;
        nvtxRangePush(std::string{"multiply_kernelft: level" + std::to_string(L)}.c_str());
        const int box_offset = s.pw_form_box_offset_h[L];
        const Real *radialft_L = s.d_radialft_flat.data() + L * s.radialft_per_level_reals;

        Real *src = is_stresslet ? s.d_pw_form_pool.data() : s.d_pw_out.data();
        const long *src_offsets = is_stresslet ? nullptr : s.d_pw_out_offsets.data();
        const long src_stride = is_stresslet ? s.pw_form_stride_reals / 2 : 0L;
        multiply_at(n_box, s.n_pw, s.n_pw2, s.n_pw_modes, s.hpw_per_level_h[L], /*windowed=*/false,
                    s.d_pw_form_box_flat.data() + box_offset, radialft_L, src, src_offsets, src_stride,
                    s.d_pw_out.data(), s.d_pw_out_offsets.data(), 0);
        nvtxRangePop();
    }

    // Windowed root: writes proxy_coeffs_downward[0].
    {
        const long window_in_stride_complex = s.n_tables_up * s.n_pw_modes_win;
        cuda::Proxy2PwArgs<Real> pa;
        pa.n_boxes_at_level = 1;
        pa.n_order = s.n_order;
        pa.n_pw = s.n_pw_win;
        pa.n_pw2 = s.n_pw2_win;
        pa.n_charge_dim = s.n_tables_up;
        pa.box_ids = s.d_box0_id.data();
        pa.proxy_flat = s.d_proxy_coeffs_upward.data();
        pa.proxy_offsets = s.d_proxy_offsets_upward.data();
        pa.poly2pw = s.d_window_poly2pw.data();
        pa.dst_flat = s.d_window_pw_form_in.data();
        pa.dst_offsets = nullptr;
        pa.dst_stride_complex = window_in_stride_complex;
        cuda::launch_proxy2pw<Real, DIM>(pa, s.downward_stream);

        // Multiply at window size.
        const long window_out_stride_complex = s.n_charge_dim * s.n_pw_modes_win;
        multiply_at(1, s.n_pw_win, s.n_pw2_win, s.n_pw_modes_win, s.hpw_win, /*windowed=*/true, s.d_box0_id.data(),
                    s.d_window_radialft.data(), s.d_window_pw_form_in.data(), nullptr, window_in_stride_complex,
                    s.d_window_pw_form_out.data(), nullptr, window_out_stride_complex);
        Real *pw_for_pw_to_proxy = is_stresslet ? s.d_window_pw_form_out.data() : s.d_window_pw_form_in.data();

        // pw_to_proxy at window size → d_proxy_coeffs_downward[box=0].
        cuda::PwToProxyArgs<Real> pp;
        pp.n_boxes_at_level = 1;
        pp.n_order = s.n_order;
        pp.n_pw = s.n_pw_win;
        pp.n_pw2 = s.n_pw2_win;
        pp.n_charge_dim = s.n_charge_dim;
        pp.pw_in_stride = 0; // single buffer
        pp.box_ids = s.d_box0_id.data();
        pp.pw_in_pool = pw_for_pw_to_proxy;
        pp.pw2poly = s.d_window_pw2poly.data();
        pp.proxy_flat = s.d_proxy_coeffs_downward.data();
        pp.proxy_offsets = s.d_proxy_offsets_downward.data();
        cuda::launch_pw_to_proxy<Real, DIM>(pp, s.downward_stream);
    }
}

template class CudaFormOutgoingContext<float, 2>;
template class CudaFormOutgoingContext<float, 3>;
template class CudaFormOutgoingContext<double, 2>;
template class CudaFormOutgoingContext<double, 3>;

} // namespace dmk
