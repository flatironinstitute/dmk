#include <dmk/cuda/pt/state.hpp>

#include <dmk.h>
#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/shared_state_kernels.hpp>
#include <dmk/direct.hpp>
#include <dmk/tree.hpp>

#include <cuda_runtime.h>
#include <sctl.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace dmk::cuda::pt {

namespace {

// Upload any host container (std::vector or std::span) to a device buffer.
// Empty sources leave the buffer unallocated.
template <typename T, typename Src>
void up(DeviceBuffer<T> &d, const Src &s) {
    if (!s.empty())
        d.upload(s.data(), s.size());
}

// Non-owning span helpers over the tree's flat host arrays. `long` matches
// sctl::Long's ABI (V1 uploads via the same reinterpret cast).
template <typename Real, typename V>
std::span<const Real> real_span(const V &v) {
    return v.Dim() ? std::span<const Real>(&v[0], v.Dim()) : std::span<const Real>();
}
template <typename V>
std::span<const int> int_span(const V &v) {
    return v.Dim() ? std::span<const int>(&v[0], v.Dim()) : std::span<const int>();
}
template <typename V>
std::span<const long> long_span(const V &v) {
    return v.Dim() ? std::span<const long>(reinterpret_cast<const long *>(&v[0]), v.Dim()) : std::span<const long>();
}

// Per-level flat box list filtered by `pred`. Fills offset ([n_levels+1]),
// count ([n_levels]), running max, and the flattened box ids.
template <typename Real, int DIM, typename Pred>
void build_per_level_box_list(DMKPtTree<Real, DIM> &tree, int n_levels, std::vector<int> &offset,
                              std::vector<int> &count, int &max_per_level, std::vector<int> &flat, Pred pred) {
    offset.assign(n_levels + 1, 0);
    count.assign(n_levels, 0);
    flat.clear();
    flat.reserve(tree.n_boxes());
    for (int L = 0; L < n_levels; ++L) {
        offset[L] = flat.size();
        for (int idx = 0; idx < tree.level_indices[L].Dim(); ++idx) {
            const int b = tree.level_indices[L][idx];
            if (pred(b)) {
                flat.push_back(b);
                count[L]++;
            }
        }
        max_per_level = std::max(max_per_level, count[L]);
    }
    offset[n_levels] = flat.size();
}

template <typename Real, int DIM>
void build_charge2proxy_groups(BuildInputs<Real, DIM> &in, DMKPtTree<Real, DIM> &tree) {
    auto &w = in.worklists;
    w.n_c2p_groups = tree.charge2proxy_groups.size();
    w.c2p_center_boxes.reserve(w.n_c2p_groups);
    w.c2p_levels.reserve(w.n_c2p_groups);
    w.c2p_src_box_flat_offsets.reserve(w.n_c2p_groups);
    w.c2p_n_src_boxes_per_group.reserve(w.n_c2p_groups);
    for (const auto &g : tree.charge2proxy_groups) {
        w.c2p_center_boxes.push_back(g.center_box);
        w.c2p_levels.push_back(g.level);
        w.c2p_src_box_flat_offsets.push_back(w.c2p_src_boxes_flat.size());
        w.c2p_n_src_boxes_per_group.push_back(g.n_src_boxes);
        for (int k = 0; k < g.n_src_boxes; ++k)
            w.c2p_src_boxes_flat.push_back(g.src_boxes[k]);
    }

    // Group ordering: heaviest source work first so heavy groups grab CTAs
    // early. Work key matches the device kernel's formula (CHUNK=32 tiebreaker).
    if (w.n_c2p_groups) {
        constexpr int CHUNK = 32;
        std::vector<std::pair<long long, int>> work_perm;
        work_perm.reserve(w.n_c2p_groups);
        for (int g = 0; g < w.n_c2p_groups; ++g) {
            long long total_sources = 0;
            long long total_chunks = 0;
            const auto &grp = tree.charge2proxy_groups[g];
            for (int sbi = 0; sbi < grp.n_src_boxes; ++sbi) {
                const int n_src = tree.src_counts_owned[grp.src_boxes[sbi]];
                total_sources += n_src;
                total_chunks += (n_src + CHUNK - 1) / CHUNK;
            }
            work_perm.emplace_back(total_sources * 1024LL + total_chunks, g);
        }
        std::sort(work_perm.begin(), work_perm.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        w.c2p_group_perm.resize(w.n_c2p_groups);
        w.n_c2p_active_groups = 0;
        for (int i = 0; i < w.n_c2p_groups; ++i) {
            w.c2p_group_perm[i] = work_perm[i].second;
            if (work_perm[i].first > 0)
                ++w.n_c2p_active_groups;
        }
    }
}

template <typename Real, int DIM>
void build_tp_up_pair_lists(BuildInputs<Real, DIM> &in, DMKPtTree<Real, DIM> &tree) {
    auto &w = in.worklists;
    const int n_levels = in.topology.n_levels;
    const auto &node_lists = tree.GetNodeLists();
    constexpr int n_children = 1 << DIM;
    w.tp_up_offset.assign(n_levels + 1, 0);
    w.tp_up_count.assign(n_levels, 0);
    for (int L = 0; L < n_levels; ++L) {
        w.tp_up_offset[L] = w.tp_up_src.size();
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
                w.tp_up_src.push_back(child);
                w.tp_up_dst.push_back(parent);
                w.tp_up_octants.push_back(ic);
                w.tp_up_count[L]++;
            }
        }
        w.max_tp_up_per_level = std::max(w.max_tp_up_per_level, w.tp_up_count[L]);
    }
    w.tp_up_offset[n_levels] = w.tp_up_src.size();
}

} // namespace

template <typename Real, int DIM>
BuildInputs<Real, DIM> to_build_inputs(DMKPtTree<Real, DIM> &tree) {
    BuildInputs<Real, DIM> in;

    // --- Topology ---
    auto &topo = in.topology;
    const int n_boxes = tree.n_boxes();
    const int n_levels = tree.n_levels();
    topo.n_boxes = n_boxes;
    topo.n_levels = n_levels;
    topo.nlist1_stride = (1 << (2 * DIM)) - (1 << DIM) + 1;
    topo.n_neighbors = sctl::pow<DIM>(3);

    topo.direct_work = std::span<const int>(tree.direct_work.data(), tree.direct_work.size());

    topo.list1_flat.assign((std::size_t)n_boxes * topo.nlist1_stride, -1);
    topo.list1_count.assign(n_boxes, 0);
    for (int b = 0; b < n_boxes; ++b) {
        const auto sp = tree.list1(b);
        topo.list1_count[b] = sp.size();
        for (std::size_t k = 0; k < sp.size(); ++k)
            topo.list1_flat[(std::size_t)b * topo.nlist1_stride + k] = sp[k];
    }

    const auto &node_mid = tree.GetNodeMID();
    const auto &node_lists = tree.GetNodeLists();
    topo.box_levels.resize(n_boxes);
    topo.ifpwexp.resize(n_boxes);
    topo.is_global_leaf.resize(n_boxes);
    topo.neighbors.resize((std::size_t)n_boxes * topo.n_neighbors);
    for (int b = 0; b < n_boxes; ++b) {
        topo.box_levels[b] = node_mid[b].Depth();
        topo.ifpwexp[b] = tree.ifpwexp[b] ? 1 : 0;
        topo.is_global_leaf[b] = tree.is_global_leaf[b] ? 1 : 0;
        for (int k = 0; k < topo.n_neighbors; ++k)
            topo.neighbors[(std::size_t)b * topo.n_neighbors + k] = node_lists[b].nbr[k];
    }

    // --- Particles ---
    auto &part = in.particles;
    part.is_stresslet = tree.params.kernel == DMK_STRESSLET;
    part.r_src = real_span<Real>(tree.r_src_sorted_owned);
    part.r_trg = real_span<Real>(tree.r_trg_sorted_owned);
    part.src_counts = int_span(tree.src_counts_owned);
    part.trg_counts = int_span(tree.trg_counts_owned);
    part.r_src_offsets = long_span(tree.r_src_offsets_owned);
    part.r_trg_offsets = long_span(tree.r_trg_offsets_owned);
    part.scatter_index_src = long_span(tree.GetScatterIdx("pdmk_src"));
    part.scatter_index_trg = long_span(tree.GetScatterIdx("pdmk_trg"));
    if (part.is_stresslet) {
        part.charge_offsets = long_span(tree.density_offsets_with_halo);
        part.normal_offsets = long_span(tree.normal_offsets_with_halo);
        part.charge_outer_offsets = long_span(tree.charge_offsets_owned);
    } else {
        part.charge_offsets = long_span(tree.charge_offsets_owned);
    }

    // --- Fourier + per-level geometry ---
    auto &fou = in.fourier;
    fou.n_pw = tree.expansion_constants.n_pw_diff;
    fou.n_pw2 = (fou.n_pw + 1) / 2;
    if constexpr (DIM == 3)
        fou.n_pw_modes = fou.n_pw * fou.n_pw * fou.n_pw2;
    else
        fou.n_pw_modes = fou.n_pw * fou.n_pw2;
    fou.n_charge_dim = tree.n_tables_down;
    fou.n_tables_up = tree.n_tables_up;
    fou.n_order = tree.expansion_constants.n_order;
    fou.n_pw_win = tree.expansion_constants.n_pw_win;
    fou.n_pw2_win = (fou.n_pw_win + 1) / 2;
    if constexpr (DIM == 3)
        fou.n_pw_modes_win = fou.n_pw_win * fou.n_pw_win * fou.n_pw2_win;
    else
        fou.n_pw_modes_win = fou.n_pw_win * fou.n_pw2_win;
    fou.hpw_win = (Real)tree.expansion_constants.hpw_win;
    fou.n_digits = tree.n_digits;
    fou.beta = tree.expansion_constants.beta;

    fou.p2c = real_span<Real>(tree.p2c);
    fou.c2p = real_span<Real>(tree.c2p);
    fou.centers = real_span<Real>(tree.centers);
    fou.direct_rsc = real_span<Real>(tree.direct_rsc);
    fou.direct_cen = real_span<Real>(tree.direct_cen);
    fou.direct_d2max = real_span<Real>(tree.direct_d2max);

    fou.pw2poly_per_level_reals = 2 * fou.n_pw * fou.n_order;
    fou.poly2pw_per_level_reals = 2 * fou.n_pw * fou.n_order;
    fou.radialft_per_level_reals = fou.n_pw_modes;
    fou.wpwshift_per_level_reals = 2 * topo.n_neighbors * fou.n_pw_modes;

    fou.pw2poly_flat.resize((std::size_t)n_levels * fou.pw2poly_per_level_reals);
    fou.poly2pw_flat.resize((std::size_t)n_levels * fou.poly2pw_per_level_reals);
    fou.radialft_flat.resize((std::size_t)n_levels * fou.radialft_per_level_reals);
    fou.wpwshift_flat.resize((std::size_t)n_levels * fou.wpwshift_per_level_reals);
    fou.inv_box_scale.resize(n_levels);
    fou.hpw_per_level.resize(n_levels);
    for (int L = 0; L < n_levels; ++L) {
        const auto &dfd = tree.difference_fourier_data[L];
        const Real *pw2poly = reinterpret_cast<const Real *>(&dfd.pw2poly[0]);
        const Real *poly2pw = reinterpret_cast<const Real *>(&dfd.poly2pw[0]);
        const Real *wpwshift = reinterpret_cast<const Real *>(&dfd.wpwshift[0]);
        std::copy(pw2poly, pw2poly + fou.pw2poly_per_level_reals,
                  &fou.pw2poly_flat[(std::size_t)L * fou.pw2poly_per_level_reals]);
        std::copy(poly2pw, poly2pw + fou.poly2pw_per_level_reals,
                  &fou.poly2pw_flat[(std::size_t)L * fou.poly2pw_per_level_reals]);
        std::copy(&dfd.radialft[0], &dfd.radialft[0] + fou.radialft_per_level_reals,
                  &fou.radialft_flat[(std::size_t)L * fou.radialft_per_level_reals]);
        std::copy(wpwshift, wpwshift + fou.wpwshift_per_level_reals,
                  &fou.wpwshift_flat[(std::size_t)L * fou.wpwshift_per_level_reals]);
        fou.inv_box_scale[L] = Real{2} / (Real)tree.boxsize[L];
        fou.hpw_per_level[L] = (Real)tree.expansion_constants.hpw_diff / (Real)tree.boxsize[L];
    }

    if (fou.n_pw_win) {
        const auto &wfd = tree.window_fourier_data;
        const Real *pw2poly = reinterpret_cast<const Real *>(&wfd.pw2poly[0]);
        const Real *poly2pw = reinterpret_cast<const Real *>(&wfd.poly2pw[0]);
        fou.window_pw2poly.assign(pw2poly, pw2poly + 2 * fou.n_pw_win * fou.n_order);
        fou.window_poly2pw.assign(poly2pw, poly2pw + 2 * fou.n_pw_win * fou.n_order);
        fou.window_radialft.assign(&wfd.radialft[0], &wfd.radialft[0] + fou.n_pw_modes_win);
    }

    // --- Worklists ---
    auto &w = in.worklists;
    build_charge2proxy_groups(in, tree);
    build_tp_up_pair_lists(in, tree);

    w.tp_offset.assign(n_levels + 1, 0);
    w.tp_count.assign(n_levels, 0);
    for (int L = 0; L < n_levels; ++L) {
        w.tp_offset[L] = w.tp_parents.size();
        for (const auto &p : tree.tensorprod_pairs_per_level[L]) {
            w.tp_parents.push_back(p.parent);
            w.tp_children.push_back(p.child);
            w.tp_octants.push_back(p.child_octant);
            w.tp_count[L]++;
        }
        w.max_tp_per_level = std::max(w.max_tp_per_level, w.tp_count[L]);
    }
    w.tp_offset[n_levels] = w.tp_parents.size();

    // pw_eval per-level max is unused (dead in V1); discard it.
    int pw_eval_max_discard = 0;
    build_per_level_box_list(
        tree, n_levels, w.pw_eval_box_offset, w.pw_eval_box_count, pw_eval_max_discard, w.pw_eval_box_flat,
        [&](int b) { return tree.ifpwexp[b] && (tree.src_counts_owned[b] + tree.trg_counts_owned[b]) > 0; });

    build_per_level_box_list(tree, n_levels, w.pw_form_box_offset, w.pw_form_box_count, w.max_pw_form_per_level,
                             w.pw_form_box_flat,
                             [&](int b) { return tree.ifpwexp[b] && tree.proxy_coeffs_offsets[b] != -1; });

    w.pw_in_pool_base.assign(n_levels, 0);
    long total_slots = 0;
    for (int L = 0; L < n_levels; ++L) {
        w.pw_in_pool_base[L] = total_slots;
        total_slots += w.pw_eval_box_count[L];
    }

    // --- Scratch strides / sizes ---
    auto &sc = in.scratch;
    sc.tensorprod_scratch_stride_reals = 2L * fou.n_order * fou.n_order * fou.n_order; // matches V1 (n_order^3)
    sc.pw_in_stride_reals = 2L * fou.n_charge_dim * fou.n_pw_modes;
    sc.pw_form_stride_reals = part.is_stresslet ? 2L * fou.n_tables_up * fou.n_pw_modes : 0;
    sc.proxy_coeffs_upward_dim = tree.proxy_coeffs_upward.Dim();
    sc.proxy_coeffs_downward_dim = tree.proxy_coeffs_downward.Dim();
    sc.proxy_offsets_upward = long_span(tree.proxy_coeffs_offsets);
    sc.proxy_offsets_downward = long_span(tree.proxy_coeffs_offsets_downward);
    sc.pw_out_offsets = long_span(tree.pw_out_offsets);

    // --- Outputs ---
    auto &out = in.outputs;
    out.kernel = tree.params.kernel;
    out.eval_src = tree.params.eval_src;
    out.eval_trg = tree.params.eval_trg;
    out.pot_src_dof = tree.kernel_output_dim_src;
    out.pot_trg_dof = tree.kernel_output_dim_trg;
    out.pot_src_size = (tree.r_src_sorted_owned.Dim() / DIM) * out.pot_src_dof;
    out.pot_trg_size = (tree.r_trg_sorted_owned.Dim() / DIM) * out.pot_trg_dof;
    out.pot_src_offsets = long_span(tree.pot_src_offsets);
    out.pot_trg_offsets = long_span(tree.pot_trg_offsets);

    return in;
}

template <typename Real, int DIM>
State<Real, DIM>::State(const BuildInputs<Real, DIM> &in) {
    kernel = in.outputs.kernel;
    n_boxes = in.topology.n_boxes;
    n_levels = in.topology.n_levels;

    // --- Topology ---
    topology.nlist1_stride = in.topology.nlist1_stride;
    topology.n_neighbors = in.topology.n_neighbors;
    up(topology.d_direct_work, in.topology.direct_work);
    up(topology.d_list1_flat, in.topology.list1_flat);
    up(topology.d_list1_count, in.topology.list1_count);
    up(topology.d_box_levels, in.topology.box_levels);
    up(topology.d_neighbors, in.topology.neighbors);
    up(topology.d_ifpwexp, in.topology.ifpwexp);
    up(topology.d_is_global_leaf, in.topology.is_global_leaf);

    // --- Particles ---
    const auto &pi = in.particles;
    up(particles.d_r_src, pi.r_src);
    up(particles.d_r_trg, pi.r_trg);
    up(particles.d_src_counts, pi.src_counts);
    up(particles.d_trg_counts, pi.trg_counts);
    up(particles.d_r_src_offsets, pi.r_src_offsets);
    up(particles.d_r_trg_offsets, pi.r_trg_offsets);
    up(particles.d_charge_offsets, pi.charge_offsets);
    up(particles.d_scatter_index_src, pi.scatter_index_src);
    up(particles.d_scatter_index_trg, pi.scatter_index_trg);
    if (pi.is_stresslet) {
        up(particles.d_normal_offsets, pi.normal_offsets);
        up(particles.d_charge_outer_offsets, pi.charge_outer_offsets);
    }

    // --- Fourier (scalars mirror BuildInputs; buffers uploaded verbatim) ---
    const auto &fi = in.fourier;
    fourier.n_pw = fi.n_pw;
    fourier.n_pw2 = fi.n_pw2;
    fourier.n_pw_modes = fi.n_pw_modes;
    fourier.n_charge_dim = fi.n_charge_dim;
    fourier.n_tables_up = fi.n_tables_up;
    fourier.n_order = fi.n_order;
    fourier.n_pw_win = fi.n_pw_win;
    fourier.n_pw2_win = fi.n_pw2_win;
    fourier.n_pw_modes_win = fi.n_pw_modes_win;
    fourier.hpw_win = fi.hpw_win;
    fourier.n_digits = fi.n_digits;
    fourier.beta = fi.beta;
    fourier.pw2poly_per_level_reals = fi.pw2poly_per_level_reals;
    fourier.poly2pw_per_level_reals = fi.poly2pw_per_level_reals;
    fourier.radialft_per_level_reals = fi.radialft_per_level_reals;
    fourier.wpwshift_per_level_reals = fi.wpwshift_per_level_reals;
    up(fourier.d_pw2poly_flat, fi.pw2poly_flat);
    up(fourier.d_poly2pw_flat, fi.poly2pw_flat);
    up(fourier.d_radialft_flat, fi.radialft_flat);
    up(fourier.d_wpwshift_flat, fi.wpwshift_flat);
    up(fourier.d_window_pw2poly, fi.window_pw2poly);
    up(fourier.d_window_poly2pw, fi.window_poly2pw);
    up(fourier.d_window_radialft, fi.window_radialft);
    up(fourier.d_p2c, fi.p2c);
    up(fourier.d_c2p, fi.c2p);
    up(fourier.d_centers, fi.centers);
    up(fourier.d_inv_box_scale, fi.inv_box_scale);
    up(fourier.d_direct_rsc, fi.direct_rsc);
    up(fourier.d_direct_cen, fi.direct_cen);
    up(fourier.d_direct_d2max, fi.direct_d2max);

    // --- Worklists ---
    const auto &wi = in.worklists;
    worklists.n_c2p_groups = wi.n_c2p_groups;
    worklists.n_c2p_active_groups = wi.n_c2p_active_groups;
    up(worklists.d_c2p_center_boxes, wi.c2p_center_boxes);
    up(worklists.d_c2p_levels, wi.c2p_levels);
    up(worklists.d_c2p_src_box_flat_offsets, wi.c2p_src_box_flat_offsets);
    up(worklists.d_c2p_n_src_boxes_per_group, wi.c2p_n_src_boxes_per_group);
    up(worklists.d_c2p_src_boxes_flat, wi.c2p_src_boxes_flat);
    up(worklists.d_c2p_group_perm, wi.c2p_group_perm);
    up(worklists.d_tp_parents, wi.tp_parents);
    up(worklists.d_tp_children, wi.tp_children);
    up(worklists.d_tp_octants, wi.tp_octants);
    up(worklists.d_tp_up_src_boxes, wi.tp_up_src);
    up(worklists.d_tp_up_dst_boxes, wi.tp_up_dst);
    up(worklists.d_tp_up_octants, wi.tp_up_octants);
    up(worklists.d_pw_eval_box_flat, wi.pw_eval_box_flat);
    up(worklists.d_pw_form_box_flat, wi.pw_form_box_flat);
    worklists.pw_in_pool_base_h = wi.pw_in_pool_base;
    worklists.tp_offset_h = wi.tp_offset;
    worklists.tp_count_h = wi.tp_count;
    worklists.tp_up_offset_h = wi.tp_up_offset;
    worklists.tp_up_count_h = wi.tp_up_count;
    worklists.pw_eval_box_offset_h = wi.pw_eval_box_offset;
    worklists.pw_eval_box_count_h = wi.pw_eval_box_count;
    worklists.pw_form_box_offset_h = wi.pw_form_box_offset;
    worklists.pw_form_box_count_h = wi.pw_form_box_count;

    // --- Scratch (buffers sized/zeroed here; contents produced by the passes) ---
    const auto &si = in.scratch;
    scratch.tensorprod_scratch_stride_reals = si.tensorprod_scratch_stride_reals;
    scratch.pw_in_stride_reals = si.pw_in_stride_reals;
    scratch.pw_form_stride_reals = si.pw_form_stride_reals;
    if (si.proxy_coeffs_upward_dim) {
        scratch.d_proxy_coeffs_upward.resize(si.proxy_coeffs_upward_dim);
        scratch.d_proxy_coeffs_upward.zero_async();
    }
    if (si.proxy_coeffs_downward_dim) {
        scratch.d_proxy_coeffs_downward.resize(si.proxy_coeffs_downward_dim);
        scratch.d_proxy_coeffs_downward.zero_async();
    }
    up(scratch.d_proxy_offsets_upward, si.proxy_offsets_upward);
    up(scratch.d_proxy_offsets_downward, si.proxy_offsets_downward);
    up(scratch.d_pw_out_offsets, si.pw_out_offsets);

    const int max_tp_any = std::max(wi.max_tp_per_level, wi.max_tp_up_per_level);
    if (max_tp_any && scratch.tensorprod_scratch_stride_reals)
        scratch.d_tensorprod_scratch.resize((std::size_t)max_tp_any * scratch.tensorprod_scratch_stride_reals);

    long total_slots = 0;
    for (int L = 0; L < n_levels; ++L)
        total_slots += wi.pw_eval_box_count[L];
    if (total_slots && scratch.pw_in_stride_reals)
        scratch.d_pw_in_pool.resize((std::size_t)total_slots * scratch.pw_in_stride_reals);

    if (pi.is_stresslet && wi.max_pw_form_per_level && scratch.pw_form_stride_reals)
        scratch.d_pw_form_pool.resize((std::size_t)wi.max_pw_form_per_level * scratch.pw_form_stride_reals);

    if (fourier.n_pw_modes_win) {
        scratch.d_window_pw_form_in.resize(2 * (std::size_t)fourier.n_tables_up * fourier.n_pw_modes_win);
        if (pi.is_stresslet)
            scratch.d_window_pw_form_out.resize(2 * (std::size_t)fourier.n_charge_dim * fourier.n_pw_modes_win);
    }
    const int zero_int = 0;
    scratch.d_box0_id.upload(&zero_int, 1);

    // --- Outputs ---
    outputs.eval_src = in.outputs.eval_src;
    outputs.eval_trg = in.outputs.eval_trg;
    outputs.pot_src_dof = in.outputs.pot_src_dof;
    outputs.pot_trg_dof = in.outputs.pot_trg_dof;
    outputs.pot_src_size = in.outputs.pot_src_size;
    outputs.pot_trg_size = in.outputs.pot_trg_size;
    up(outputs.d_pot_src_offsets, in.outputs.pot_src_offsets);
    up(outputs.d_pot_trg_offsets, in.outputs.pot_trg_offsets);
    outputs.d_pot_direct_src.resize(outputs.pot_src_size);
    outputs.d_pot_direct_trg.resize(outputs.pot_trg_size);
    outputs.d_pot_src_final.resize(outputs.pot_src_size);
    outputs.d_pot_trg_final.resize(outputs.pot_trg_size);

    direct_stream = cuda_helpers::DeviceStream::non_blocking();
    downward_stream = cuda_helpers::DeviceStream::non_blocking();
}

template <typename Real, int DIM>
void State<Real, DIM>::upload_and_sort_charges(const Real *charges, const Real *normals, long n_src) {
    const int charge_dof = get_kernel_input_dim(DIM, kernel);
    DeviceBuffer<Real> d_charge_input;
    d_charge_input.upload_async(charges, n_src * charge_dof, direct_stream.get());
    particles.d_charge.resize(n_src * charge_dof);
    cuda::launch_scatter_forward(d_charge_input.data(), particles.d_charge.data(), particles.d_scatter_index_src.data(),
                                 n_src, charge_dof, direct_stream.get());

    if (kernel == DMK_STRESSLET) {
        DeviceBuffer<Real> d_normal_input;
        d_normal_input.upload_async(normals, n_src * DIM, direct_stream.get());
        particles.d_normal.resize(n_src * DIM);
        cuda::launch_scatter_forward(d_normal_input.data(), particles.d_normal.data(),
                                     particles.d_scatter_index_src.data(), n_src, DIM, direct_stream.get());
        particles.d_charge_outer.resize(n_src * DIM * DIM);
        cuda::launch_scatter_forward_stresslet(d_charge_input.data(), d_normal_input.data(),
                                               particles.d_charge_outer.data(), particles.d_scatter_index_src.data(),
                                               n_src, DIM, direct_stream.get());
        direct_stream.sync();
        return;
    }
    direct_stream.sync();
}

template <typename Real, int DIM>
void State<Real, DIM>::dump(DMKPtTree<Real, DIM> &tree) {
    const std::string prefix = "gpu_v2/";
    tree.dump(prefix);
    auto write = [&](const std::string &name, const Real *d_ptr, std::size_t n) {
        const std::string path = prefix + name + "." + std::to_string(tree.comm().Size()) + "." +
                                 std::to_string(tree.comm().Rank()) + ".dat";
        cuda_helpers::dump_device_buffer_to_file<Real>(path, d_ptr, n);
    };
    write("dmk_proxy_coeffs_downward", scratch.d_proxy_coeffs_downward.data(), scratch.d_proxy_coeffs_downward.size());
    write("dmk_proxy_coeffs", scratch.d_proxy_coeffs_upward.data(), scratch.d_proxy_coeffs_upward.size());
}

template BuildInputs<float, 2> to_build_inputs<float, 2>(DMKPtTree<float, 2> &);
template BuildInputs<float, 3> to_build_inputs<float, 3>(DMKPtTree<float, 3> &);
template BuildInputs<double, 2> to_build_inputs<double, 2>(DMKPtTree<double, 2> &);
template BuildInputs<double, 3> to_build_inputs<double, 3>(DMKPtTree<double, 3> &);

template struct State<float, 2>;
template struct State<float, 3>;
template struct State<double, 2>;
template struct State<double, 3>;

} // namespace dmk::cuda::pt
