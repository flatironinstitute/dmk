// Construction / destruction of CudaSharedDeviceState — uploads all
// read-only inputs + topology that GPU offload operations need.
//
// Plain C++; no <<<>>> launch syntax. Compiled into DMKOBJS_CUDA.

#include <dmk/cuda_helpers.hpp>
#include <dmk/cuda_shared_state.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/tree.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace dmk {

using cuda_helpers::device_alloc;
using cuda_helpers::device_free;
using cuda_helpers::device_upload;
using cuda_helpers::sctl_int_vec_to_std;

template <typename Real, int DIM>
CudaSharedDeviceState<Real, DIM>::CudaSharedDeviceState(DMKPtTree<Real, DIM> &tree) {
    if (tree.params.use_periodic)
        throw std::runtime_error("CUDA offload: periodic boundary conditions are not yet supported");
    if (tree.params.kernel == DMK_YUKAWA)
        throw std::runtime_error("CUDA offload: Yukawa kernel is not yet supported on the GPU path");

    const auto &node_mid = tree.GetNodeMID();
    n_boxes = (int)tree.n_boxes();
    // Use tree.n_levels() (= level_indices.Dim()) — boxsize.Dim() is n_levels+1
    // because it carries an extra slot beyond the deepest live level.
    n_levels = tree.n_levels();
    nlist1_stride = (1 << (2 * DIM)) - (1 << DIM) + 1;

    // ---------- host-side topology buffers ----------
    std::vector<int> direct_work_h(tree.direct_work.begin(), tree.direct_work.end());
    n_direct_work = (int)direct_work_h.size();

    std::vector<int> list1_flat_h((std::size_t)n_boxes * nlist1_stride, -1);
    std::vector<int> list1_count_h(n_boxes, 0);
    for (int b = 0; b < n_boxes; ++b) {
        const auto sp = tree.list1(b);
        list1_count_h[b] = (int)sp.size();
        for (std::size_t k = 0; k < sp.size(); ++k)
            list1_flat_h[(std::size_t)b * nlist1_stride + k] = sp[k];
    }

    std::vector<int> box_levels_h(n_boxes);
    std::vector<unsigned char> ifpwexp_h(n_boxes);
    for (int b = 0; b < n_boxes; ++b) {
        box_levels_h[b] = node_mid[b].Depth();
        ifpwexp_h[b] = tree.ifpwexp[b] ? 1 : 0;
    }

    // ---------- uploads ----------
    d_direct_work = device_upload(direct_work_h.data(), direct_work_h.size());
    d_list1_flat = device_upload(list1_flat_h.data(), list1_flat_h.size());
    d_list1_count = device_upload(list1_count_h.data(), list1_count_h.size());
    d_box_levels = device_upload(box_levels_h.data(), box_levels_h.size());
    d_ifpwexp = device_upload(ifpwexp_h.data(), ifpwexp_h.size());

    d_direct_rsc = device_upload(&tree.direct_rsc[0], tree.direct_rsc.Dim());
    d_direct_cen = device_upload(&tree.direct_cen[0], tree.direct_cen.Dim());
    d_direct_d2max = device_upload(&tree.direct_d2max[0], tree.direct_d2max.Dim());

    if (tree.r_src_sorted_with_halo.Dim())
        d_r_src_halo = device_upload(&tree.r_src_sorted_with_halo[0], tree.r_src_sorted_with_halo.Dim());
    d_r_src_halo_offsets =
        device_upload((const long *)&tree.r_src_offsets_with_halo[0], tree.r_src_offsets_with_halo.Dim());
    {
        auto h = sctl_int_vec_to_std(tree.src_counts_with_halo);
        d_src_counts_halo = device_upload(h.data(), h.size());
    }

    const bool is_stresslet = tree.params.kernel == DMK_STRESSLET;
    if (is_stresslet) {
        if (tree.density_sorted_with_halo.Dim())
            d_charge_halo = device_upload(&tree.density_sorted_with_halo[0], tree.density_sorted_with_halo.Dim());
        d_charge_halo_offsets =
            device_upload((const long *)&tree.density_offsets_with_halo[0], tree.density_offsets_with_halo.Dim());
        if (tree.normal_sorted_with_halo.Dim())
            d_normal_halo = device_upload(&tree.normal_sorted_with_halo[0], tree.normal_sorted_with_halo.Dim());
        d_normal_halo_offsets =
            device_upload((const long *)&tree.normal_offsets_with_halo[0], tree.normal_offsets_with_halo.Dim());
    } else {
        if (tree.charge_sorted_with_halo.Dim())
            d_charge_halo = device_upload(&tree.charge_sorted_with_halo[0], tree.charge_sorted_with_halo.Dim());
        d_charge_halo_offsets =
            device_upload((const long *)&tree.charge_offsets_with_halo[0], tree.charge_offsets_with_halo.Dim());
    }

    if (tree.r_src_sorted_owned.Dim())
        d_r_src_owned = device_upload(&tree.r_src_sorted_owned[0], tree.r_src_sorted_owned.Dim());
    d_r_src_owned_offsets = device_upload((const long *)&tree.r_src_offsets_owned[0], tree.r_src_offsets_owned.Dim());
    {
        auto h = sctl_int_vec_to_std(tree.src_counts_owned);
        d_src_counts_owned = device_upload(h.data(), h.size());
    }

    if (tree.r_trg_sorted_owned.Dim())
        d_r_trg_owned = device_upload(&tree.r_trg_sorted_owned[0], tree.r_trg_sorted_owned.Dim());
    d_r_trg_owned_offsets = device_upload((const long *)&tree.r_trg_offsets_owned[0], tree.r_trg_offsets_owned.Dim());
    {
        auto h = sctl_int_vec_to_std(tree.trg_counts_owned);
        d_trg_counts_owned = device_upload(h.data(), h.size());
    }

    pot_src_size = tree.pot_src_sorted.Dim();
    pot_trg_size = tree.pot_trg_sorted.Dim();
    d_pot_src_offsets = device_upload((const long *)&tree.pot_src_offsets[0], tree.pot_src_offsets.Dim());
    d_pot_trg_offsets = device_upload((const long *)&tree.pot_trg_offsets[0], tree.pot_trg_offsets.Dim());

    // Downward proxy buffer: allocated zero-initialized; populated later (by
    // host upload from eval_targets, or by GPU planewave_to_proxy / tensorprod
    // kernels once those are in place).
    proxy_size = tree.proxy_coeffs_downward.Dim();
    if (proxy_size) {
        d_proxy_coeffs_downward = device_alloc<Real>(proxy_size);
        DMK_CHECK_CUDA(cudaMemset(d_proxy_coeffs_downward, 0, proxy_size * sizeof(Real)));
    }
    d_proxy_offsets_downward =
        device_upload((const long *)&tree.proxy_coeffs_offsets_downward[0], tree.proxy_coeffs_offsets_downward.Dim());

    // ============== Downward-pass GPU plumbing ==============
    DMK_CHECK_CUDA(cudaStreamCreateWithFlags(&downward_stream, cudaStreamNonBlocking));

    n_neighbors = 1;
    for (int d = 0; d < DIM; ++d)
        n_neighbors *= 3;
    n_pw = tree.expansion_constants.n_pw_diff;
    n_pw2 = (n_pw + 1) / 2;
    if constexpr (DIM == 3)
        n_pw_modes = n_pw * n_pw * n_pw2;
    else
        n_pw_modes = n_pw * n_pw2; // 2D
    n_charge_dim = tree.n_tables_down;
    n_tables_up = tree.n_tables_up;
    n_order = tree.expansion_constants.n_order;
    n_pw_win = tree.expansion_constants.n_pw_win;
    n_pw2_win = (n_pw_win + 1) / 2;
    if constexpr (DIM == 3)
        n_pw_modes_win = n_pw_win * n_pw_win * n_pw2_win;
    else
        n_pw_modes_win = n_pw_win * n_pw2_win;
    hpw_win = (Real)tree.expansion_constants.hpw_win;
    kernel = tree.params.kernel;

    // Neighbor list: flat n_boxes * n_neighbors.
    {
        const auto &node_lists = tree.GetNodeLists();
        std::vector<int> nbr_h((std::size_t)n_boxes * n_neighbors);
        for (int b = 0; b < n_boxes; ++b)
            for (int k = 0; k < n_neighbors; ++k)
                nbr_h[(std::size_t)b * n_neighbors + k] = node_lists[b].nbr[k];
        d_neighbors = device_upload(nbr_h.data(), nbr_h.size());
    }

    // is_global_leaf as 0/1 bytes.
    {
        std::vector<unsigned char> leaf_h(n_boxes);
        for (int b = 0; b < n_boxes; ++b)
            leaf_h[b] = tree.is_global_leaf[b] ? 1 : 0;
        d_is_global_leaf = device_upload(leaf_h.data(), leaf_h.size());
    }

    // pw_out_offsets: may or may not be sized yet. Pull it in if non-empty;
    // otherwise allocate_pw_out() will do it later.
    if (tree.pw_out_offsets.Dim())
        d_pw_out_offsets = device_upload((const long *)&tree.pw_out_offsets[0], tree.pw_out_offsets.Dim());

    // Per-level pw2poly / poly2pw / radialft. pw2poly and poly2pw are
    // std::complex<Real>[n_pw * n_order] = 2 * n_pw * n_order reals each;
    // radialft is real[n_pw_modes].
    pw2poly_per_level_reals = 2 * n_pw * n_order;
    poly2pw_per_level_reals = 2 * n_pw * n_order;
    radialft_per_level_reals = n_pw_modes;
    {
        const std::size_t total_pw2poly = (std::size_t)n_levels * pw2poly_per_level_reals;
        const std::size_t total_poly2pw = (std::size_t)n_levels * poly2pw_per_level_reals;
        const std::size_t total_radialft = (std::size_t)n_levels * radialft_per_level_reals;
        std::vector<Real> h_pw2poly(total_pw2poly);
        std::vector<Real> h_poly2pw(total_poly2pw);
        std::vector<Real> h_radialft(total_radialft);
        hpw_per_level_h.assign(n_levels, Real{0});
        for (int L = 0; L < n_levels; ++L) {
            const auto &dfd = tree.difference_fourier_data[L];
            std::copy(reinterpret_cast<const Real *>(&dfd.pw2poly[0]),
                      reinterpret_cast<const Real *>(&dfd.pw2poly[0]) + pw2poly_per_level_reals,
                      &h_pw2poly[(std::size_t)L * pw2poly_per_level_reals]);
            std::copy(reinterpret_cast<const Real *>(&dfd.poly2pw[0]),
                      reinterpret_cast<const Real *>(&dfd.poly2pw[0]) + poly2pw_per_level_reals,
                      &h_poly2pw[(std::size_t)L * poly2pw_per_level_reals]);
            std::copy(&dfd.radialft[0], &dfd.radialft[0] + radialft_per_level_reals,
                      &h_radialft[(std::size_t)L * radialft_per_level_reals]);
            hpw_per_level_h[L] = (Real)tree.expansion_constants.hpw_diff / (Real)tree.boxsize[L];
        }
        d_pw2poly_flat = device_upload(h_pw2poly.data(), total_pw2poly);
        d_poly2pw_flat = device_upload(h_poly2pw.data(), total_poly2pw);
        d_radialft_flat = device_upload(h_radialft.data(), total_radialft);
    }

    // Windowed Fourier data (single-instance, used at the root).
    if (n_pw_win) {
        const auto &wfd = tree.window_fourier_data;
        d_window_pw2poly =
            device_upload(reinterpret_cast<const Real *>(&wfd.pw2poly[0]), (std::size_t)2 * n_pw_win * n_order);
        d_window_poly2pw =
            device_upload(reinterpret_cast<const Real *>(&wfd.poly2pw[0]), (std::size_t)2 * n_pw_win * n_order);
        d_window_radialft = device_upload(&wfd.radialft[0], (std::size_t)n_pw_modes_win);
    }

    // Per-level wpwshift: SoA per neighbor (already SoA in calc_planewave_translation_matrix).
    // 2 * n_neighbors * n_pw_modes reals per level.
    wpwshift_per_level_reals = 2 * n_neighbors * n_pw_modes;
    {
        const std::size_t total = (std::size_t)n_levels * wpwshift_per_level_reals;
        std::vector<Real> h(total);
        for (int L = 0; L < n_levels; ++L) {
            const auto &dfd = tree.difference_fourier_data[L];
            // Already laid out SoA per neighbor by calc_planewave_translation_matrix.
            const Real *src = reinterpret_cast<const Real *>(&dfd.wpwshift[0]);
            std::copy(src, src + wpwshift_per_level_reals, &h[(std::size_t)L * wpwshift_per_level_reals]);
        }
        d_wpwshift_flat = device_upload(h.data(), total);
    }

    // p2c matrices.
    if (tree.p2c.Dim())
        d_p2c = device_upload(&tree.p2c[0], tree.p2c.Dim());

    // Per-level pw_eval box lists: for each level, the boxes that do PW work
    // (ifpwexp[b] && (src_counts_owned[b] + trg_counts_owned[b]) > 0).
    {
        pw_eval_box_offset_h.assign(n_levels + 1, 0);
        pw_eval_box_count_h.assign(n_levels, 0);
        std::vector<int> flat;
        flat.reserve(n_boxes);
        for (int L = 0; L < n_levels; ++L) {
            pw_eval_box_offset_h[L] = (int)flat.size();
            for (int idx = 0; idx < tree.level_indices[L].Dim(); ++idx) {
                const int b = tree.level_indices[L][idx];
                const int nboxpts = tree.src_counts_owned[b] + tree.trg_counts_owned[b];
                if (tree.ifpwexp[b] && nboxpts) {
                    flat.push_back(b);
                    pw_eval_box_count_h[L]++;
                }
            }
            max_pw_eval_per_level = std::max(max_pw_eval_per_level, pw_eval_box_count_h[L]);
        }
        pw_eval_box_offset_h[n_levels] = (int)flat.size();
        pw_eval_box_count_total = (int)flat.size();
        if (pw_eval_box_count_total)
            d_pw_eval_box_flat = device_upload(flat.data(), flat.size());
    }

    // Per-level tensorprod pairs.
    {
        tp_offset_h.assign(n_levels + 1, 0);
        tp_count_h.assign(n_levels, 0);
        std::vector<int> parents, children, octants;
        for (int L = 0; L < n_levels; ++L) {
            tp_offset_h[L] = (int)parents.size();
            for (const auto &p : tree.tensorprod_pairs_per_level[L]) {
                parents.push_back(p.parent);
                children.push_back(p.child);
                octants.push_back(p.child_octant);
                tp_count_h[L]++;
            }
            max_tp_per_level = std::max(max_tp_per_level, tp_count_h[L]);
        }
        tp_offset_h[n_levels] = (int)parents.size();
        tp_count_total = (int)parents.size();
        if (tp_count_total) {
            d_tp_parents = device_upload(parents.data(), parents.size());
            d_tp_children = device_upload(children.data(), children.size());
            d_tp_octants = device_upload(octants.data(), octants.size());
        }
    }

    // Tensorprod global scratch: 2 * n_order^3 reals per block, one slab per
    // pair processed concurrently in a level.
    tensorprod_scratch_stride_reals = 2L * n_order * n_order * n_order;
    if (max_tp_per_level && tensorprod_scratch_stride_reals)
        d_tensorprod_scratch = device_alloc<Real>((std::size_t)max_tp_per_level * tensorprod_scratch_stride_reals);

    // pw_in scratch pool.
    pw_in_stride_reals = 2L * n_charge_dim * n_pw_modes;
    if (max_pw_eval_per_level && pw_in_stride_reals)
        d_pw_in_pool = device_alloc<Real>((std::size_t)max_pw_eval_per_level * pw_in_stride_reals);

    // Per-level pw_form (proxy2pw target) box list. Subset of pw_eval_box_flat
    // restricted to boxes that have an upward proxy to project from.
    {
        pw_form_box_offset_h.assign(n_levels + 1, 0);
        pw_form_box_count_h.assign(n_levels, 0);
        std::vector<int> flat;
        flat.reserve(n_boxes);
        for (int L = 0; L < n_levels; ++L) {
            pw_form_box_offset_h[L] = (int)flat.size();
            for (int idx = 0; idx < tree.level_indices[L].Dim(); ++idx) {
                const int b = tree.level_indices[L][idx];
                if (tree.ifpwexp[b] && tree.proxy_coeffs_offsets[b] != -1) {
                    flat.push_back(b);
                    pw_form_box_count_h[L]++;
                }
            }
            max_pw_form_per_level = std::max(max_pw_form_per_level, pw_form_box_count_h[L]);
        }
        pw_form_box_offset_h[n_levels] = (int)flat.size();
        pw_form_box_count_total = (int)flat.size();
        if (pw_form_box_count_total)
            d_pw_form_box_flat = device_upload(flat.data(), flat.size());
    }

    // Stresslet only: per-block pw_form pool sized for n_tables_up tables.
    // Other kernels write proxy2pw output directly into d_pw_out (since
    // n_tables_up == n_tables_down).
    if (kernel == DMK_STRESSLET) {
        pw_form_stride_reals = 2L * n_tables_up * n_pw_modes;
        if (max_pw_form_per_level && pw_form_stride_reals)
            d_pw_form_pool = device_alloc<Real>((std::size_t)max_pw_form_per_level * pw_form_stride_reals);
    }

    // Windowed root scratch buffers (one slot, n_pw_win-sized).
    if (n_pw_modes_win) {
        const std::size_t in_reals = 2 * (std::size_t)n_tables_up * n_pw_modes_win;
        d_window_pw_form_in = device_alloc<Real>(in_reals);
        if (kernel == DMK_STRESSLET) {
            const std::size_t out_reals = 2 * (std::size_t)n_charge_dim * n_pw_modes_win;
            d_window_pw_form_out = device_alloc<Real>(out_reals);
        }
    }

    // Single-element {0} scratch for kernels that take per-block box-id /
    // offset arrays but operate on just box 0 at the root.
    {
        const int zero_int = 0;
        const long zero_long = 0;
        d_box0_id = device_upload(&zero_int, 1);
        d_box0_offset = device_upload(&zero_long, 1);
    }
}

template <typename Real, int DIM>
void CudaSharedDeviceState<Real, DIM>::allocate_pw_out(DMKPtTree<Real, DIM> &tree) {
    const std::size_t needed = tree.pw_out.Dim() * 2; // pw_out is complex; reals are 2x.
    if (!needed)
        return;
    if (!d_pw_out_offsets)
        d_pw_out_offsets = device_upload((const long *)&tree.pw_out_offsets[0], tree.pw_out_offsets.Dim());
    if (!d_pw_out || pw_out_size != needed) {
        if (d_pw_out)
            device_free(d_pw_out);
        d_pw_out = device_alloc<Real>(needed);
        pw_out_size = needed;
    }
}

template <typename Real, int DIM>
void CudaSharedDeviceState<Real, DIM>::upload_proxy_upward(DMKPtTree<Real, DIM> &tree) {
    const std::size_t needed = tree.proxy_coeffs_upward.Dim();
    if (!needed)
        return;
    if (!d_proxy_offsets_upward)
        d_proxy_offsets_upward =
            device_upload((const long *)&tree.proxy_coeffs_offsets[0], tree.proxy_coeffs_offsets.Dim());
    if (!d_proxy_coeffs_upward || proxy_upward_size != needed) {
        if (d_proxy_coeffs_upward)
            device_free(d_proxy_coeffs_upward);
        d_proxy_coeffs_upward = device_alloc<Real>(needed);
        proxy_upward_size = needed;
    }
    DMK_CHECK_CUDA(
        cudaMemcpy(d_proxy_coeffs_upward, &tree.proxy_coeffs_upward[0], needed * sizeof(Real), cudaMemcpyHostToDevice));
}

template <typename Real, int DIM>
CudaSharedDeviceState<Real, DIM>::~CudaSharedDeviceState() {
    device_free(d_direct_work);
    device_free(d_list1_flat);
    device_free(d_list1_count);
    device_free(d_box_levels);
    device_free(d_ifpwexp);
    device_free(d_direct_rsc);
    device_free(d_direct_cen);
    device_free(d_direct_d2max);
    device_free(d_r_src_halo);
    device_free(d_r_src_halo_offsets);
    device_free(d_src_counts_halo);
    device_free(d_charge_halo);
    device_free(d_charge_halo_offsets);
    device_free(d_normal_halo);
    device_free(d_normal_halo_offsets);
    device_free(d_r_src_owned);
    device_free(d_r_src_owned_offsets);
    device_free(d_src_counts_owned);
    device_free(d_r_trg_owned);
    device_free(d_r_trg_owned_offsets);
    device_free(d_trg_counts_owned);
    device_free(d_pot_src_offsets);
    device_free(d_pot_trg_offsets);
    device_free(d_proxy_coeffs_downward);
    device_free(d_proxy_offsets_downward);
    device_free(d_neighbors);
    device_free(d_is_global_leaf);
    device_free(d_pw_out);
    device_free(d_pw_out_offsets);
    device_free(d_pw2poly_flat);
    device_free(d_poly2pw_flat);
    device_free(d_radialft_flat);
    device_free(d_wpwshift_flat);
    device_free(d_p2c);
    device_free(d_proxy_coeffs_upward);
    device_free(d_proxy_offsets_upward);
    device_free(d_pw_eval_box_flat);
    device_free(d_pw_form_box_flat);
    device_free(d_tp_parents);
    device_free(d_tp_children);
    device_free(d_tp_octants);
    device_free(d_tensorprod_scratch);
    device_free(d_pw_in_pool);
    device_free(d_pw_form_pool);
    device_free(d_window_pw_form_in);
    device_free(d_window_pw_form_out);
    device_free(d_window_poly2pw);
    device_free(d_window_pw2poly);
    device_free(d_window_radialft);
    device_free(d_box0_id);
    device_free(d_box0_offset);
    if (downward_stream)
        cudaStreamDestroy(downward_stream);
}

template <typename Real, int DIM>
void CudaSharedDeviceState<Real, DIM>::dump(DMKPtTree<Real, DIM> &tree) {
    const std::string prefix = "gpu/";
    tree.dump(prefix);

    auto write = [&](const std::string &name, const Real *d_ptr, std::size_t n) {
        const std::string path = prefix + name + "." + std::to_string(tree.comm().Size()) + "." +
                                 std::to_string(tree.comm().Rank()) + ".dat";
        cuda_helpers::dump_device_buffer_to_file<Real>(path, d_ptr, n);
    };
    write("dmk_proxy_coeffs_downward", d_proxy_coeffs_downward, proxy_size);
}

template struct CudaSharedDeviceState<float, 2>;
template struct CudaSharedDeviceState<float, 3>;
template struct CudaSharedDeviceState<double, 2>;
template struct CudaSharedDeviceState<double, 3>;

} // namespace dmk
