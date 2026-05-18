#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/shared_state.hpp>
#include <dmk/cuda/shared_state_kernels.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/tree.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dmk {

using cuda_helpers::sctl_int_vec_to_std;

namespace {

// Build a per-level flat box list filtered by `pred`. Writes offset_h
// ([n_levels+1]), count_h ([n_levels]), running max and total, and uploads
// the flat list to d_flat (if non-empty).
template <typename Real, int DIM, typename Pred>
void build_per_level_box_list(DMKPtTree<Real, DIM> &tree, int n_levels, std::vector<int> &offset_h,
                              std::vector<int> &count_h, int &max_per_level, int &count_total,
                              DeviceBuffer<int> &d_flat, Pred pred) {
    offset_h.assign(n_levels + 1, 0);
    count_h.assign(n_levels, 0);
    std::vector<int> flat;
    flat.reserve(tree.n_boxes());
    for (int L = 0; L < n_levels; ++L) {
        offset_h[L] = flat.size();
        for (int idx = 0; idx < tree.level_indices[L].Dim(); ++idx) {
            const int b = tree.level_indices[L][idx];
            if (pred(b)) {
                flat.push_back(b);
                count_h[L]++;
            }
        }
        max_per_level = std::max(max_per_level, count_h[L]);
    }
    offset_h[n_levels] = flat.size();
    count_total = flat.size();
    if (count_total)
        d_flat.upload(flat.data(), flat.size());
}

template <typename Real, int DIM>
void build_charge2proxy_groups(CudaSharedDeviceState<Real, DIM> &s, DMKPtTree<Real, DIM> &tree) {
    s.n_c2p_groups = tree.charge2proxy_groups.size();
    std::vector<int> centers_h, levels_h, off_h, count_h, src_flat_h;
    centers_h.reserve(s.n_c2p_groups);
    levels_h.reserve(s.n_c2p_groups);
    off_h.reserve(s.n_c2p_groups);
    count_h.reserve(s.n_c2p_groups);
    for (const auto &g : tree.charge2proxy_groups) {
        centers_h.push_back(g.center_box);
        levels_h.push_back(g.level);
        off_h.push_back(src_flat_h.size());
        count_h.push_back(g.n_src_boxes);
        for (int k = 0; k < g.n_src_boxes; ++k)
            src_flat_h.push_back(g.src_boxes[k]);
    }
    if (s.n_c2p_groups) {
        s.d_c2p_center_boxes.upload(centers_h.data(), centers_h.size());
        s.d_c2p_levels.upload(levels_h.data(), levels_h.size());
        s.d_c2p_src_box_flat_offsets.upload(off_h.data(), off_h.size());
        s.d_c2p_n_src_boxes_per_group.upload(count_h.data(), count_h.size());
        s.d_c2p_src_boxes_flat.upload(src_flat_h.data(), src_flat_h.size());
    }

    // Group ordering: largest work first so heavy groups grab CTAs early.
    // Work key matches the device kernel's previous formula (CHUNK=32 here
    // is the tiebreaker granularity used in the old work-key kernel; the
    // primary sort term is total_sources).
    if (s.n_c2p_groups) {
        constexpr int CHUNK = 32;
        std::vector<std::pair<long long, int>> work_perm;
        work_perm.reserve(s.n_c2p_groups);
        for (int g = 0; g < s.n_c2p_groups; ++g) {
            long long total_sources = 0;
            long long total_chunks = 0;
            const auto &grp = tree.charge2proxy_groups[g];
            for (int sbi = 0; sbi < grp.n_src_boxes; ++sbi) {
                const int sb = grp.src_boxes[sbi];
                const int n_src = tree.src_counts_owned[sb];
                total_sources += n_src;
                total_chunks += (n_src + CHUNK - 1) / CHUNK;
            }
            work_perm.emplace_back(total_sources * 1024LL + total_chunks, g);
        }
        std::sort(work_perm.begin(), work_perm.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        std::vector<int> perm_h(s.n_c2p_groups);
        s.n_c2p_active_groups = 0;
        for (int i = 0; i < s.n_c2p_groups; ++i) {
            perm_h[i] = work_perm[i].second;
            if (work_perm[i].first > 0)
                ++s.n_c2p_active_groups;
        }
        s.d_c2p_group_perm.upload(perm_h.data(), perm_h.size());
    }
}

// Per-level upward tensorprod pair lists. Gating: parent has
// src_counts_owned[parent] > 0 && ifpwexp[parent], same for child. Mirrors
// the CPU upward sweep (level n_levels-1..0). Pairs at level L = parent's
// level.
template <typename Real, int DIM>
void build_tp_up_pair_lists(CudaSharedDeviceState<Real, DIM> &s, DMKPtTree<Real, DIM> &tree) {
    const auto &node_lists = tree.GetNodeLists();
    constexpr int n_children = 1 << DIM;
    const int n_levels = s.n_levels;
    s.tp_up_offset_h.assign(n_levels + 1, 0);
    s.tp_up_count_h.assign(n_levels, 0);
    std::vector<int> srcs, dsts, octs;
    for (int L = 0; L < n_levels; ++L) {
        s.tp_up_offset_h[L] = srcs.size();
        for (int idx = 0; idx < tree.level_indices[L].Dim(); ++idx) {
            const int parent = tree.level_indices[L][idx];
            if (!(tree.src_counts_owned[parent] > 0 && tree.ifpwexp[parent]))
                continue;
            for (int ic = 0; ic < n_children; ++ic) {
                const int child = node_lists[parent].child[ic];
                if (child < 0)
                    continue;
                if (!(tree.src_counts_owned[child] > 0 && tree.ifpwexp[child]))
                    continue;
                srcs.push_back(child);
                dsts.push_back(parent);
                octs.push_back(ic);
                s.tp_up_count_h[L]++;
            }
        }
        s.max_tp_up_per_level = std::max(s.max_tp_up_per_level, s.tp_up_count_h[L]);
    }
    s.tp_up_offset_h[n_levels] = srcs.size();
    if (!srcs.empty()) {
        s.d_tp_up_src_boxes.upload(srcs.data(), srcs.size());
        s.d_tp_up_dst_boxes.upload(dsts.data(), dsts.size());
        s.d_tp_up_octants.upload(octs.data(), octs.size());
    }
}

} // namespace

template <typename Real, int DIM>
CudaSharedDeviceState<Real, DIM>::CudaSharedDeviceState(DMKPtTree<Real, DIM> &tree) {
    if (tree.params.use_periodic)
        throw std::runtime_error("CUDA offload: periodic boundary conditions are not yet supported");
    if (tree.params.kernel == DMK_YUKAWA)
        throw std::runtime_error("CUDA offload: Yukawa kernel is not yet supported on the GPU path");

    const auto &node_mid = tree.GetNodeMID();
    n_boxes = tree.n_boxes();
    // Use tree.n_levels() (= level_indices.Dim()) — boxsize.Dim() is n_levels+1
    // because it carries an extra slot beyond the deepest live level.
    n_levels = tree.n_levels();
    nlist1_stride = (1 << (2 * DIM)) - (1 << DIM) + 1;

    std::vector<int> direct_work_h(tree.direct_work.begin(), tree.direct_work.end());
    n_direct_work = direct_work_h.size();

    std::vector<int> list1_flat_h((std::size_t)n_boxes * nlist1_stride, -1);
    std::vector<int> list1_count_h(n_boxes, 0);
    for (int b = 0; b < n_boxes; ++b) {
        const auto sp = tree.list1(b);
        list1_count_h[b] = sp.size();
        for (std::size_t k = 0; k < sp.size(); ++k)
            list1_flat_h[(std::size_t)b * nlist1_stride + k] = sp[k];
    }

    std::vector<int> box_levels_h(n_boxes);
    std::vector<unsigned char> ifpwexp_h(n_boxes);
    for (int b = 0; b < n_boxes; ++b) {
        box_levels_h[b] = node_mid[b].Depth();
        ifpwexp_h[b] = tree.ifpwexp[b] ? 1 : 0;
    }

    d_direct_work.upload(direct_work_h.data(), direct_work_h.size());
    d_list1_flat.upload(list1_flat_h.data(), list1_flat_h.size());
    d_list1_count.upload(list1_count_h.data(), list1_count_h.size());
    d_box_levels.upload(box_levels_h.data(), box_levels_h.size());
    d_ifpwexp.upload(ifpwexp_h.data(), ifpwexp_h.size());

    d_direct_rsc.upload(&tree.direct_rsc[0], tree.direct_rsc.Dim());
    d_direct_cen.upload(&tree.direct_cen[0], tree.direct_cen.Dim());
    d_direct_d2max.upload(&tree.direct_d2max[0], tree.direct_d2max.Dim());

    if (tree.r_src_sorted_with_halo.Dim())
        d_r_src_halo.upload(&tree.r_src_sorted_with_halo[0], tree.r_src_sorted_with_halo.Dim());
    d_r_src_halo_offsets.upload((const long *)&tree.r_src_offsets_with_halo[0], tree.r_src_offsets_with_halo.Dim());
    {
        auto h = sctl_int_vec_to_std(tree.src_counts_with_halo);
        d_src_counts_halo.upload(h.data(), h.size());
    }

    const bool is_stresslet = tree.params.kernel == DMK_STRESSLET;
    if (is_stresslet) {
        if (tree.density_sorted_with_halo.Dim())
            d_charge_halo.upload(&tree.density_sorted_with_halo[0], tree.density_sorted_with_halo.Dim());
        d_charge_halo_offsets.upload((const long *)&tree.density_offsets_with_halo[0],
                                     tree.density_offsets_with_halo.Dim());
        if (tree.normal_sorted_with_halo.Dim())
            d_normal_halo.upload(&tree.normal_sorted_with_halo[0], tree.normal_sorted_with_halo.Dim());
        d_normal_halo_offsets.upload((const long *)&tree.normal_offsets_with_halo[0],
                                     tree.normal_offsets_with_halo.Dim());
    } else {
        if (tree.charge_sorted_with_halo.Dim())
            d_charge_halo.upload(&tree.charge_sorted_with_halo[0], tree.charge_sorted_with_halo.Dim());
        d_charge_halo_offsets.upload((const long *)&tree.charge_offsets_with_halo[0],
                                     tree.charge_offsets_with_halo.Dim());
    }

    if (tree.r_src_sorted_owned.Dim())
        d_r_src_owned.upload(&tree.r_src_sorted_owned[0], tree.r_src_sorted_owned.Dim());
    d_r_src_owned_offsets.upload((const long *)&tree.r_src_offsets_owned[0], tree.r_src_offsets_owned.Dim());
    {
        auto h = sctl_int_vec_to_std(tree.src_counts_owned);
        d_src_counts_owned.upload(h.data(), h.size());
    }

    if (tree.r_trg_sorted_owned.Dim())
        d_r_trg_owned.upload(&tree.r_trg_sorted_owned[0], tree.r_trg_sorted_owned.Dim());
    d_r_trg_owned_offsets.upload((const long *)&tree.r_trg_offsets_owned[0], tree.r_trg_offsets_owned.Dim());
    {
        auto h = sctl_int_vec_to_std(tree.trg_counts_owned);
        d_trg_counts_owned.upload(h.data(), h.size());
    }

    pot_src_size = tree.pot_src_sorted.Dim();
    pot_trg_size = tree.pot_trg_sorted.Dim();
    pot_src_dof = tree.kernel_output_dim_src;
    pot_trg_dof = tree.kernel_output_dim_trg;
    d_pot_src_offsets.upload((const long *)&tree.pot_src_offsets[0], tree.pot_src_offsets.Dim());
    d_pot_trg_offsets.upload((const long *)&tree.pot_trg_offsets[0], tree.pot_trg_offsets.Dim());

    const auto &src_idx = tree.GetScatterIdx("pdmk_src");
    const auto &trg_idx = tree.GetScatterIdx("pdmk_trg");
    d_scatter_index_src.upload(&src_idx[0], src_idx.Dim());
    d_scatter_index_trg.upload(&trg_idx[0], trg_idx.Dim());

    if (pot_src_size)
        d_pot_src_final.resize(pot_src_size);
    if (pot_trg_size)
        d_pot_trg_final.resize(pot_trg_size);

    // Downward proxy buffer: allocated zero-initialized; populated later (by
    // host upload from eval_targets, or by GPU planewave_to_proxy / tensorprod
    // kernels once those are in place).
    if (tree.proxy_coeffs_downward.Dim()) {
        d_proxy_coeffs_downward.resize(tree.proxy_coeffs_downward.Dim());
        d_proxy_coeffs_downward.zero_async();
    }
    d_proxy_offsets_downward.upload((const long *)&tree.proxy_coeffs_offsets_downward[0],
                                    tree.proxy_coeffs_offsets_downward.Dim());

    direct_stream = cuda_helpers::DeviceStream::non_blocking();
    downward_stream = cuda_helpers::DeviceStream::non_blocking();

    n_neighbors = sctl::pow<DIM>(3);
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
    hpw_win = tree.expansion_constants.hpw_win;
    kernel = tree.params.kernel;

    // Neighbor list: flat n_boxes * n_neighbors.
    {
        const auto &node_lists = tree.GetNodeLists();
        std::vector<int> nbr_h((std::size_t)n_boxes * n_neighbors);
        for (int b = 0; b < n_boxes; ++b)
            for (int k = 0; k < n_neighbors; ++k)
                nbr_h[(std::size_t)b * n_neighbors + k] = node_lists[b].nbr[k];
        d_neighbors.upload(nbr_h.data(), nbr_h.size());
    }

    // is_global_leaf as 0/1 bytes.
    {
        std::vector<unsigned char> leaf_h(n_boxes);
        for (int b = 0; b < n_boxes; ++b)
            leaf_h[b] = tree.is_global_leaf[b] ? 1 : 0;
        d_is_global_leaf.upload(leaf_h.data(), leaf_h.size());
    }

    // pw_out_offsets: may or may not be sized yet. Pull it in if non-empty;
    // otherwise allocate_pw_out() will do it later.
    if (tree.pw_out_offsets.Dim())
        d_pw_out_offsets.upload((const long *)&tree.pw_out_offsets[0], tree.pw_out_offsets.Dim());

    // Per-level pw2poly / poly2pw / radialft. pw2poly and poly2pw are
    // std::complex<Real>[n_pw * n_order] = 2 * n_pw * n_order reals each;
    // radialft is real[n_pw_modes].
    pw2poly_per_level_reals = 2 * n_pw * n_order;
    poly2pw_per_level_reals = 2 * n_pw * n_order;
    radialft_per_level_reals = n_pw_modes;
    {
        const std::size_t total_pw2poly = n_levels * pw2poly_per_level_reals;
        const std::size_t total_poly2pw = n_levels * poly2pw_per_level_reals;
        const std::size_t total_radialft = n_levels * radialft_per_level_reals;
        std::vector<Real> h_pw2poly(total_pw2poly);
        std::vector<Real> h_poly2pw(total_poly2pw);
        std::vector<Real> h_radialft(total_radialft);
        hpw_per_level_h.assign(n_levels, Real{0});
        for (int L = 0; L < n_levels; ++L) {
            const auto &dfd = tree.difference_fourier_data[L];
            std::copy(reinterpret_cast<const Real *>(&dfd.pw2poly[0]),
                      reinterpret_cast<const Real *>(&dfd.pw2poly[0]) + pw2poly_per_level_reals,
                      &h_pw2poly[L * pw2poly_per_level_reals]);
            std::copy(reinterpret_cast<const Real *>(&dfd.poly2pw[0]),
                      reinterpret_cast<const Real *>(&dfd.poly2pw[0]) + poly2pw_per_level_reals,
                      &h_poly2pw[L * poly2pw_per_level_reals]);
            std::copy(&dfd.radialft[0], &dfd.radialft[0] + radialft_per_level_reals,
                      &h_radialft[L * radialft_per_level_reals]);
            hpw_per_level_h[L] = (Real)tree.expansion_constants.hpw_diff / (Real)tree.boxsize[L];
        }
        d_pw2poly_flat.upload(h_pw2poly.data(), total_pw2poly);
        d_poly2pw_flat.upload(h_poly2pw.data(), total_poly2pw);
        d_radialft_flat.upload(h_radialft.data(), total_radialft);
    }

    // Windowed Fourier data (single-instance, used at the root).
    if (n_pw_win) {
        const auto &wfd = tree.window_fourier_data;
        d_window_pw2poly.upload(reinterpret_cast<const Real *>(&wfd.pw2poly[0]), 2 * n_pw_win * n_order);
        d_window_poly2pw.upload(reinterpret_cast<const Real *>(&wfd.poly2pw[0]), 2 * n_pw_win * n_order);
        d_window_radialft.upload(&wfd.radialft[0], n_pw_modes_win);
    }

    // Per-level wpwshift: SoA per neighbor (already SoA in calc_planewave_translation_matrix).
    // 2 * n_neighbors * n_pw_modes reals per level.
    wpwshift_per_level_reals = 2 * n_neighbors * n_pw_modes;
    {
        const std::size_t total = n_levels * wpwshift_per_level_reals;
        std::vector<Real> h(total);
        for (int L = 0; L < n_levels; ++L) {
            const auto &dfd = tree.difference_fourier_data[L];
            // Already laid out SoA per neighbor by calc_planewave_translation_matrix.
            const Real *src = reinterpret_cast<const Real *>(&dfd.wpwshift[0]);
            std::copy(src, src + wpwshift_per_level_reals, &h[L * wpwshift_per_level_reals]);
        }
        d_wpwshift_flat.upload(h.data(), total);
    }

    // p2c / c2p matrices (per child-octant; identical layout, different
    // direction of propagation).
    if (tree.p2c.Dim())
        d_p2c.upload(&tree.p2c[0], tree.p2c.Dim());
    if (tree.c2p.Dim())
        d_c2p.upload(&tree.c2p[0], tree.c2p.Dim());

    // Box centers + inverse half-boxsize per level.
    if (tree.centers.Dim())
        d_centers.upload(&tree.centers[0], tree.centers.Dim());
    {
        std::vector<Real> sc_h(n_levels);
        for (int L = 0; L < n_levels; ++L)
            sc_h[L] = Real{2} / (Real)tree.boxsize[L];
        d_inv_box_scale.upload(sc_h.data(), sc_h.size());
    }

    // Owned charges (used by GPU charge2proxy). For Stresslet this is the
    // pre-multiplied force×normal product (n_tables_up = 9 components per
    // source), exactly what the CPU upward path reads via charge_owned_view.
    if (tree.charge_sorted_owned.Dim())
        d_charge_owned.upload(&tree.charge_sorted_owned[0], tree.charge_sorted_owned.Dim());
    if (tree.charge_offsets_owned.Dim())
        d_charge_owned_offsets.upload((const long *)&tree.charge_offsets_owned[0], tree.charge_offsets_owned.Dim());

    build_charge2proxy_groups(*this, tree);
    build_tp_up_pair_lists(*this, tree);

    // Allocate the upward proxy buffer up front (zero-initialized). The
    // upward orchestrator zeroes it again at run() to handle re-evals.
    if (tree.proxy_coeffs_upward.Dim()) {
        d_proxy_coeffs_upward.resize(tree.proxy_coeffs_upward.Dim());
        d_proxy_coeffs_upward.zero_async();
    }
    if (tree.proxy_coeffs_offsets.Dim())
        d_proxy_offsets_upward.upload((const long *)&tree.proxy_coeffs_offsets[0], tree.proxy_coeffs_offsets.Dim());

    // Per-level pw_eval box lists: for each level, the boxes that do PW work
    // (ifpwexp[b] && (src_counts_owned[b] + trg_counts_owned[b]) > 0).
    build_per_level_box_list(tree, n_levels, pw_eval_box_offset_h, pw_eval_box_count_h, max_pw_eval_per_level,
                             pw_eval_box_count_total, d_pw_eval_box_flat, [&](int b) {
                                 return tree.ifpwexp[b] && (tree.src_counts_owned[b] + tree.trg_counts_owned[b]) > 0;
                             });

    // Per-level tensorprod pairs.
    {
        tp_offset_h.assign(n_levels + 1, 0);
        tp_count_h.assign(n_levels, 0);
        std::vector<int> parents, children, octants;
        for (int L = 0; L < n_levels; ++L) {
            tp_offset_h[L] = parents.size();
            for (const auto &p : tree.tensorprod_pairs_per_level[L]) {
                parents.push_back(p.parent);
                children.push_back(p.child);
                octants.push_back(p.child_octant);
                tp_count_h[L]++;
            }
            max_tp_per_level = std::max(max_tp_per_level, tp_count_h[L]);
        }
        tp_offset_h[n_levels] = parents.size();
        tp_count_total = parents.size();
        if (tp_count_total) {
            d_tp_parents.upload(parents.data(), parents.size());
            d_tp_children.upload(children.data(), children.size());
            d_tp_octants.upload(octants.data(), octants.size());
        }
    }

    // Tensorprod global scratch: 2 * n_order^3 reals per block, one slab per
    // pair processed concurrently in a level. Sized for the max across both
    // directions (downward tp_pairs and upward tp_up_pairs share the slab).
    tensorprod_scratch_stride_reals = 2L * n_order * n_order * n_order;
    const int max_tp_any = std::max(max_tp_per_level, max_tp_up_per_level);
    if (max_tp_any && tensorprod_scratch_stride_reals)
        d_tensorprod_scratch.resize(max_tp_any * tensorprod_scratch_stride_reals);

    // pw_in scratch pool. The multilevel kernels launch all levels concurrently
    // on the same stream with disjoint slab regions, so the buffer is the SUM
    // of per-level slot counts (not max).
    pw_in_stride_reals = 2L * n_charge_dim * n_pw_modes;
    pw_in_pool_base_h.assign(n_levels, 0);
    {
        long total_slots = 0;
        for (int L = 0; L < n_levels; ++L) {
            pw_in_pool_base_h[L] = total_slots;
            total_slots += pw_eval_box_count_h[L];
        }
        if (total_slots && pw_in_stride_reals)
            d_pw_in_pool.resize(total_slots * pw_in_stride_reals);
    }

    // Persistent device scratch for multilevel kernel-arg arrays. One slot per
    // level even though not every level uses one — sizes are tiny compared to
    // the data buffers.
    if (n_levels) {
        d_shift_pw_args.resize(n_levels);
        d_pw_to_proxy_args.resize(n_levels);
        d_proxy2pw_args.resize(n_levels);
    }

    // Per-level pw_form (proxy2pw target) box list. Subset of pw_eval_box_flat
    // restricted to boxes that have an upward proxy to project from.
    build_per_level_box_list(tree, n_levels, pw_form_box_offset_h, pw_form_box_count_h, max_pw_form_per_level,
                             pw_form_box_count_total, d_pw_form_box_flat,
                             [&](int b) { return tree.ifpwexp[b] && tree.proxy_coeffs_offsets[b] != -1; });

    // Stresslet only: per-block pw_form pool sized for n_tables_up tables.
    // Other kernels write proxy2pw output directly into d_pw_out (since
    // n_tables_up == n_tables_down).
    if (kernel == DMK_STRESSLET) {
        pw_form_stride_reals = 2L * n_tables_up * n_pw_modes;
        if (max_pw_form_per_level && pw_form_stride_reals)
            d_pw_form_pool.resize(max_pw_form_per_level * pw_form_stride_reals);
    }

    // Windowed root scratch buffers (one slot, n_pw_win-sized).
    if (n_pw_modes_win) {
        d_window_pw_form_in.resize(2 * n_tables_up * n_pw_modes_win);
        if (kernel == DMK_STRESSLET)
            d_window_pw_form_out.resize(2 * n_charge_dim * n_pw_modes_win);
    }

    // Single-element {0} scratch for kernels that take a per-block box-id
    // array but only operate on the root box.
    {
        const int zero_int = 0;
        d_box0_id.upload(&zero_int, 1);
    }
}

template <typename Real, int DIM>
void CudaSharedDeviceState<Real, DIM>::allocate_pw_out(DMKPtTree<Real, DIM> &tree) {
    const std::size_t needed = tree.pw_out.Dim() * 2; // pw_out is complex; reals are 2x.
    if (!needed)
        return;
    if (!d_pw_out_offsets)
        d_pw_out_offsets.upload((const long *)&tree.pw_out_offsets[0], tree.pw_out_offsets.Dim());
    d_pw_out.resize(needed);
}

template <typename Real, int DIM>
void CudaSharedDeviceState<Real, DIM>::upload_proxy_upward(DMKPtTree<Real, DIM> &tree) {
    const std::size_t needed = tree.proxy_coeffs_upward.Dim();
    if (!needed)
        return;
    if (!d_proxy_offsets_upward)
        d_proxy_offsets_upward.upload((const long *)&tree.proxy_coeffs_offsets[0], tree.proxy_coeffs_offsets.Dim());
    d_proxy_coeffs_upward.upload(&tree.proxy_coeffs_upward[0], needed);
}

template <typename Real, int DIM>
void CudaSharedDeviceState<Real, DIM>::upload_charges(DMKPtTree<Real, DIM> &tree) {
    const bool is_stresslet = tree.params.kernel == DMK_STRESSLET;
    const auto &halo_src = is_stresslet ? tree.density_sorted_with_halo : tree.charge_sorted_with_halo;
    if (d_charge_halo && halo_src.Dim())
        d_charge_halo.upload(&halo_src[0], halo_src.Dim());
    if (is_stresslet && d_normal_halo && tree.normal_sorted_with_halo.Dim())
        d_normal_halo.upload(&tree.normal_sorted_with_halo[0], tree.normal_sorted_with_halo.Dim());
    if (d_charge_owned && tree.charge_sorted_owned.Dim())
        d_charge_owned.upload(&tree.charge_sorted_owned[0], tree.charge_sorted_owned.Dim());
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
    write("dmk_proxy_coeffs_downward", d_proxy_coeffs_downward.data(), d_proxy_coeffs_downward.size());
    write("dmk_proxy_coeffs", d_proxy_coeffs_upward.data(), d_proxy_coeffs_upward.size());
}

template <typename Real, int DIM>
void CudaSharedDeviceState<Real, DIM>::finalize_pot(cudaStream_t eval_stream, const Real *d_pot_eval_src,
                                                    const Real *d_pot_eval_trg, const Real *d_extra_src,
                                                    const Real *d_extra_trg) {
    // Make direct_stream wait for eval_stream's pending writes to d_pot_eval_*
    // before reading them. (direct_stream is already serial with direct's own
    // writes to d_extra_*.)
    auto eval_done = cuda_helpers::DeviceEvent::disable_timing();
    DMK_CHECK_CUDA(cudaEventRecord(eval_done, eval_stream));
    DMK_CHECK_CUDA(cudaStreamWaitEvent(direct_stream, eval_done, 0));

    if (pot_src_size && d_pot_eval_src && d_extra_src) {
        const long n = (long)(pot_src_size / pot_src_dof);
        cuda::launch_accumulate_and_scatter<Real>(d_pot_src_final.data(), d_pot_eval_src, d_extra_src,
                                                  d_scatter_index_src.data(), pot_src_dof, n, direct_stream);
    }
    if (pot_trg_size && d_pot_eval_trg && d_extra_trg) {
        const long n = (long)(pot_trg_size / pot_trg_dof);
        cuda::launch_accumulate_and_scatter<Real>(d_pot_trg_final.data(), d_pot_eval_trg, d_extra_trg,
                                                  d_scatter_index_trg.data(), pot_trg_dof, n, direct_stream);
    }

    DMK_CHECK_CUDA(cudaStreamSynchronize(direct_stream));
}

template struct CudaSharedDeviceState<float, 2>;
template struct CudaSharedDeviceState<float, 3>;
template struct CudaSharedDeviceState<double, 2>;
template struct CudaSharedDeviceState<double, 3>;

} // namespace dmk
