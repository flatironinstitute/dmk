#include <dmk/cuda/pt/state.hpp>
#include <dmk/nvtx_wrapper.h>

#include "metadata.hpp"

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"

#include <dmk.h>
#include <dmk/cuda/helpers.hpp>
#include <dmk/direct.hpp>
#include <dmk/tree.hpp>

#include <cuda_runtime.h>
#include <sctl.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace dmk::cuda::pt {

/// Every device table that depends only on the configuration, so it is uploaded on the first build
/// of that configuration and adopted by every State after: the four per-level Fourier tables and
/// their mode pruning, the proxy transfer matrices, the windowed root tables, and the per-level
/// direct-evaluation constants.
template <typename Real>
struct FourierTables {
    DeviceBuffer<Real> pw2poly, poly2pw, radialft, phase;
    DeviceBuffer<int> pencil_slots, full_of_compact, mode_axes;
    int n_pw_live = 0, pw_slab = 0;
    DeviceBuffer<Real> p2c, c2p;
    DeviceBuffer<Real> window_pw2poly, window_poly2pw, window_radialft;
    DeviceBuffer<Real> inv_box_scale, direct_rsc, direct_cen, direct_d2max;
};

namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

// Charge/potential helpers, JIT-compiled from pt/shared_state.cu. Fixed
// block, one launch each.
template <typename Real>
void launch_stresslet_charge(const Real *densities, const Real *normals, Real *out, long n_particles, int dim,
                             cudaStream_t stream) {
    if (n_particles == 0)
        return;
    constexpr int BLOCK = 256;
    static JitCache cache;
    JitKey key;
    key.name = "PtStressletChargeKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {{"BLOCK_SIZE", BLOCK}};
    auto kernel = cache.get_kernel_from_source(
        key, [&] { return make_stage_source("pt/shared_state.cu", key, "", "SharedState"); });
    const long grid = (n_particles + BLOCK - 1) / BLOCK;
    kernel->launch(dim3(grid, 1, 1), dim3(BLOCK, 1, 1), 0, stream, densities, normals, out, n_particles, dim);
}

template <typename Real>
void launch_accumulate(Real *out, const Real *pot_eval, const Real *pot_extra, int dof, long n_particles,
                       cudaStream_t stream) {
    if (n_particles == 0)
        return;
    constexpr int BLOCK = 256;
    static JitCache cache;
    JitKey key;
    key.name = "PtAccumulateKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {{"BLOCK_SIZE", BLOCK}};
    auto kernel = cache.get_kernel_from_source(
        key, [&] { return make_stage_source("pt/shared_state.cu", key, "", "SharedState"); });
    const long grid = (n_particles + BLOCK - 1) / BLOCK;
    kernel->launch(dim3(grid, 1, 1), dim3(BLOCK, 1, 1), 0, stream, out, pot_eval, pot_extra, dof, n_particles);
}

// Upload any host container (std::vector or std::span) to a device buffer.
// Empty sources leave the buffer unallocated.
template <typename T, typename Src>
void up(DeviceBuffer<T> &d, const Src &s) {
    if (!s.empty())
        d.upload(s.data(), s.size());
}

// The four per-level Fourier tables depend only on the kernel and the expansion constants, which
// build_shared_precompute already caches on the host. Their device copies are identical build to
// build, so they are uploaded once and handed out as raw pointers; the mode pruning that permutes
// them is deterministic from the same key and is cached with them.
struct FourierKey {
    int kernel, eval_src, dim, periodic, n_order, n_pw_win, n_pw_diff, n_pw_periodic, n_levels;
    double eps, fparam, beta;
    bool operator<(const FourierKey &o) const {
        return std::tie(kernel, eval_src, dim, periodic, n_order, n_pw_win, n_pw_diff, n_pw_periodic, n_levels, eps,
                        fparam, beta) < std::tie(o.kernel, o.eval_src, o.dim, o.periodic, o.n_order, o.n_pw_win,
                                                 o.n_pw_diff, o.n_pw_periodic, o.n_levels, o.eps, o.fparam, o.beta);
    }
};

template <typename Real, int DIM>
void build_fourier_tables(const BuildInputs<Real, DIM> &in, DMKPtTree<Real, DIM> &tree, FourierTables<Real> &ftab) {
    const auto &fou = in.fourier;
    const auto &topo = in.topology;
    const int n_levels = in.topology.n_levels;

    std::vector<Real> pw2poly_flat((std::size_t)n_levels * fou.pw2poly_per_level_reals);
    std::vector<Real> poly2pw_flat((std::size_t)n_levels * fou.poly2pw_per_level_reals);
    std::vector<Real> radialft_flat((std::size_t)n_levels * fou.radialft_per_level_reals);
    std::vector<Real> phase_flat((std::size_t)n_levels * fou.phase_per_level_reals);
    for (int L = 0; L < n_levels; ++L) {
        const auto &dfd = tree.difference_fourier_data[L];
        const Real *pw2poly = reinterpret_cast<const Real *>(&dfd.pw2poly[0]);
        const Real *poly2pw = reinterpret_cast<const Real *>(&dfd.poly2pw[0]);
        std::copy(pw2poly, pw2poly + fou.pw2poly_per_level_reals,
                  &pw2poly_flat[(std::size_t)L * fou.pw2poly_per_level_reals]);
        std::copy(poly2pw, poly2pw + fou.poly2pw_per_level_reals,
                  &poly2pw_flat[(std::size_t)L * fou.poly2pw_per_level_reals]);
        std::copy(&dfd.radialft[0], &dfd.radialft[0] + fou.radialft_per_level_reals,
                  &radialft_flat[(std::size_t)L * fou.radialft_per_level_reals]);
        // The translation table is separable, w(o, (j_0, ..)) = prod_a e^{i o_a theta_{j_a}}, so its block
        // for the offset (0, .., 0, +1) over the modes whose other indices are all 0 is e^{i theta_j}.
        const Real *unit = reinterpret_cast<const Real *>(&dfd.wpwshift[0]) +
                           (std::size_t)(topo.n_neighbors / 2 + 1) * 2 * fou.n_pw_modes;
        for (int j = 0; j < fou.n_pw; ++j) {
            phase_flat[(std::size_t)L * fou.phase_per_level_reals + 2 * j] = unit[j];
            phase_flat[(std::size_t)L * fou.phase_per_level_reals + 2 * j + 1] = unit[fou.n_pw_modes + j];
        }
    }

    auto [pencil_slots, full_of_compact, n_live, slab] = pw_slab_layout(tree);
    ftab.n_pw_live = n_live;
    ftab.pw_slab = slab;

    if (!full_of_compact.empty()) {
        // Permute the per-mode table into slab order.
        std::vector<Real> rad_new(fou.n_pw_modes);
        for (int L = 0; L < n_levels; ++L) {
            Real *rad = &radialft_flat[(std::size_t)L * fou.radialft_per_level_reals];
            std::fill(rad_new.begin(), rad_new.end(), Real{0});
            for (int c = 0; c < n_live; ++c)
                rad_new[c] = rad[full_of_compact[c]];
            std::copy(rad_new.begin(), rad_new.end(), rad);
        }
    }

    // Each live slot's index along every axis of the mode cube, axis 0 (the outermost) lowest.
    std::vector<int> mode_axes(n_live);
    for (int c = 0; c < n_live; ++c) {
        int full = full_of_compact.empty() ? c : full_of_compact[c];
        for (int a = DIM - 1; a >= 0; --a) {
            mode_axes[c] |= (full % fou.n_pw) << (8 * a);
            full /= fou.n_pw;
        }
    }

    ftab.pencil_slots.upload(pencil_slots.data(), pencil_slots.size());
    ftab.full_of_compact.upload(full_of_compact.data(), full_of_compact.size());
    ftab.pw2poly.upload(pw2poly_flat.data(), pw2poly_flat.size());
    ftab.poly2pw.upload(poly2pw_flat.data(), poly2pw_flat.size());
    ftab.radialft.upload(radialft_flat.data(), radialft_flat.size());
    ftab.phase.upload(phase_flat.data(), phase_flat.size());
    ftab.mode_axes.upload(mode_axes.data(), mode_axes.size());

    const auto up = [](DeviceBuffer<Real> &d, const sctl::Vector<Real> &v) {
        d.upload(v.Dim() ? &v[0] : nullptr, v.Dim());
    };
    up(ftab.p2c, tree.p2c);
    up(ftab.c2p, tree.c2p);
    up(ftab.direct_rsc, tree.direct_rsc);
    up(ftab.direct_cen, tree.direct_cen);
    up(ftab.direct_d2max, tree.direct_d2max);
    std::vector<Real> inv_box_scale(n_levels);
    for (int L = 0; L < n_levels; ++L)
        inv_box_scale[L] = Real{2} / (Real)tree.boxsize[L];
    ftab.inv_box_scale.upload(inv_box_scale.data(), inv_box_scale.size());
    if (fou.n_pw_win) {
        const auto &wfd = tree.window_fourier_data;
        ftab.window_pw2poly.upload(reinterpret_cast<const Real *>(&wfd.pw2poly[0]),
                                   2 * (std::size_t)fou.n_pw_win * fou.n_order);
        ftab.window_poly2pw.upload(reinterpret_cast<const Real *>(&wfd.poly2pw[0]),
                                   2 * (std::size_t)fou.n_pw_win * fou.n_order);
        ftab.window_radialft.upload(&wfd.radialft[0], fou.n_pw_modes_win);
    }
}

} // namespace

template <typename Real, int DIM>
PwSlabLayout pw_slab_layout(DMKPtTree<Real, DIM> &tree) {
    const int n_pw = tree.expansion_constants.n_pw_diff;
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_levels = tree.n_levels();

    // The kernel FT is tabulated against integer radius j1^2+j2^2+j3^2 and gathered onto the mode
    // cube, so modes past the band edge fall outside the prolate window's support and the multiply
    // sets them to exactly zero. Taking the mask from the tabulated values rather than from the
    // ball geometry means kernels that build the FT by quadrature, with no hard gate, keep every
    // mode and simply see no pruning. DMK_PW_MASK=0 disables.
    const bool prune = [] {
        const char *v = std::getenv("DMK_PW_MASK");
        return !v || std::atoi(v) != 0;
    }();
    std::vector<char> live(n_pw_modes, !prune);
    for (int m = 0; m < n_pw_modes && prune; ++m)
        for (int L = 0; L < n_levels && !live[m]; ++L)
            live[m] = tree.difference_fourier_data[L].radialft[m] != Real{0};

    // Each (m2, m3) pencil keeps one run of m1 symmetric about the zero mode: everything from its
    // first live mode to its last, widened to its mirror. A dead mode inside the run is harmless, as
    // its FT is zero. One run lets proxy2pw and pw2proxy get a slot by arithmetic off an n_pw*n_pw2
    // table, and the symmetry lets pw2proxy pair each mode with its mirror in place.
    PwSlabLayout out;
    const int n_pencil = n_pw_modes / n_pw;
    out.pencil_slots.assign(2 * (std::size_t)n_pencil, 0);
    for (int p = 0; p < n_pencil; ++p) {
        const char *row = &live[(std::size_t)p * n_pw];
        int lo = 0, hi = n_pw;
        while (lo < n_pw && !row[lo])
            ++lo;
        while (hi > lo && !row[hi - 1])
            --hi;
        if (lo < hi) {
            lo = std::min(lo, n_pw - hi);
            hi = n_pw - lo;
        }

        out.pencil_slots[2 * p] = (int)out.full_of_compact.size() - lo;
        out.pencil_slots[2 * p + 1] = lo | (hi << 16);
        for (int m1 = lo; m1 < hi; ++m1)
            out.full_of_compact.push_back(p * n_pw + m1);
    }

    out.n_live = (int)out.full_of_compact.size();
    // Padded to 32 modes so every slab starts 256-byte aligned: ShiftPw's streaming loads are
    // L2-bound and a misaligned slab costs them an extra line per warp.
    out.slab = (out.n_live + 31) / 32 * 32;
    if (out.n_live == n_pw_modes) {
        // Nothing to prune, as for the quadrature-built kernels whose FT is nonzero everywhere. Drop
        // the tables so the kernels take their cube-order path.
        out.pencil_slots.clear();
        out.full_of_compact.clear();
    }
    return out;
}

template <typename Real, int DIM>
BuildInputs<Real, DIM> to_build_inputs(DMKPtTree<Real, DIM> &tree, const DeviceMetadataOutputs<Real> &md) {
    BuildInputs<Real, DIM> in;
    in.device_metadata = &md;

    sctl::Profile::Tic("tbi_topology", &tree.comm());
    nvtxRangePush("tbi_topology");
    // --- Topology ---
    auto &topo = in.topology;
    const int n_boxes = tree.n_boxes();
    const int n_levels = tree.n_levels();
    topo.n_boxes = n_boxes;
    topo.n_levels = n_levels;
    topo.nlist1_stride = (1 << (2 * DIM)) - (1 << DIM) + 1;
    topo.n_neighbors = sctl::pow<DIM>(3);
    nvtxRangePop();
    sctl::Profile::Toc();

    // --- Particles ---
    auto &part = in.particles;
    part.is_stresslet = tree.params.kernel == DMK_STRESSLET;
    // d_r_src / d_r_trg are set by pt::Tree
    part.n_src_owned = tree.n_src_owned;
    part.n_src_before_owned = tree.n_src_before_owned;

    sctl::Profile::Tic("tbi_fourier", &tree.comm());
    nvtxRangePush("tbi_fourier");
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
    // Root plane-wave grid: the periodic root kernel lives on the reciprocal lattice and
    // uses n_pw_periodic modes, so the `_win` fields carry whichever the tree built.
    fou.n_pw_win =
        tree.params.use_periodic ? tree.expansion_constants.n_pw_periodic : tree.expansion_constants.n_pw_win;
    fou.n_pw2_win = (fou.n_pw_win + 1) / 2;
    if constexpr (DIM == 3)
        fou.n_pw_modes_win = fou.n_pw_win * fou.n_pw_win * fou.n_pw2_win;
    else
        fou.n_pw_modes_win = fou.n_pw_win * fou.n_pw2_win;
    fou.hpw_win = (Real)tree.expansion_constants.hpw_win;
    fou.n_digits = tree.n_digits;
    fou.beta = tree.expansion_constants.beta;
    fou.fparam = tree.params.fparam;

    // Yukawa's residual fit depends on lambda*boxsize, so it needs one polynomial per
    // level rather than a single scale-invariant pack. src_level spans [min target-box
    // depth, n_levels]; skipping shallower levels also avoids a fit that would throw
    // at large lambda. The device metadata reports the shallowest work-list box directly.
    if (tree.params.kernel == DMK_YUKAWA) {
        fou.direct_coeffs_level0 = md.min_direct_level;
        for (int L = md.min_direct_level; L <= n_levels; ++L) {
            const auto c = tree.fourier_data.local_correction_coeffs(L, tree.n_digits);
            fou.direct_coeffs_by_level.emplace_back(c.reg_poly.begin(), c.reg_poly.end());
        }
    }

    fou.pw2poly_per_level_reals = 2 * fou.n_pw * fou.n_order;
    fou.poly2pw_per_level_reals = 2 * fou.n_pw * fou.n_order;
    fou.radialft_per_level_reals = fou.n_pw_modes;
    fou.phase_per_level_reals = 2 * fou.n_pw;

    fou.hpw_per_level.resize(n_levels);
    for (int L = 0; L < n_levels; ++L)
        fou.hpw_per_level[L] = (Real)tree.expansion_constants.hpw_diff / (Real)tree.boxsize[L];

    static std::mutex fourier_mtx;
    static std::map<FourierKey, FourierTables<Real>> fourier_cache;
    const FourierKey fkey{static_cast<int>(tree.params.kernel),
                          static_cast<int>(tree.params.eval_src),
                          DIM,
                          tree.params.use_periodic ? 1 : 0,
                          tree.expansion_constants.n_order,
                          tree.expansion_constants.n_pw_win,
                          tree.expansion_constants.n_pw_diff,
                          tree.expansion_constants.n_pw_periodic,
                          n_levels,
                          tree.params.eps,
                          tree.params.fparam,
                          tree.expansion_constants.beta};
    std::lock_guard<std::mutex> fourier_lock(fourier_mtx);
    const auto [fit, fourier_fresh] = fourier_cache.try_emplace(fkey);
    auto &ftab = fit->second;
    if (fourier_fresh)
        build_fourier_tables(in, tree, ftab);

    fou.n_pw_live = ftab.n_pw_live;
    fou.pw_slab = ftab.pw_slab;
    fou.tables = &ftab;

    // The proxy basis is radially compact to tolerance, so like the plane-wave grid its tensor
    // cube carries nothing in the corners. 110% of n_order-1 measured free (l2 unchanged to three
    // figures) across laplace/yukawa/stokeslet/stresslet, eps 1e-3..1e-6, both precisions and
    // n_order 9..21, while dropping ~35% of the coefficients at every one of those orders; 100%
    // costs up to 50% more error. Only charge2proxy exploits it: eval_targets' inner block is
    // tight enough that any control flow to skip dead coefficients costs more than it saves.
    // DMK_PROXY_BALL overrides the percentage, 0 disables.
    const int ball_pct = [] {
        const char *v = std::getenv("DMK_PROXY_BALL");
        return v ? std::atoi(v) : 110;
    }();
    fou.proxy_ball_r2 = 0;
    if (ball_pct > 0 && fou.n_order > 0) {
        const double r = (fou.n_order - 1) * ball_pct / 100.0;
        fou.proxy_ball_r2 = (int)(r * r);
    }

    nvtxRangePop();
    sctl::Profile::Toc();
    sctl::Profile::Tic("tbi_worklists", &tree.comm());
    nvtxRangePush("tbi_worklists");
    // --- Worklists: the lists were built on the device; only the level tables come back ---
    auto &w = in.worklists;
    const auto level_table = [&](const long *off, std::vector<int> &o, std::vector<int> &c) {
        o.assign(off, off + n_levels + 1);
        c.assign(n_levels, 0);
        for (int L = 0; L < n_levels; ++L)
            c[L] = o[L + 1] - o[L];
    };
    level_table(md.h_table(0), w.tp_offset, w.tp_count);
    level_table(md.h_table(1), w.tp_up_offset, w.tp_up_count);
    level_table(md.h_table(2), w.tp_up_par_offset, w.tp_up_par_count);
    level_table(md.h_table(3), w.pw_eval_box_offset, w.pw_eval_box_count);
    level_table(md.h_table(4), w.pw_form_box_offset, w.pw_form_box_count);
    w.shift_group_base.assign(md.h_table(5), md.h_table(5) + n_levels + 1);
    w.max_pw_form_per_level = md.max_pw_form_per_level;

    w.pw_in_pool_base.assign(n_levels, 0);
    long total_slots = 0;
    for (int L = 0; L < n_levels; ++L) {
        w.pw_in_pool_base[L] = total_slots;
        total_slots += w.pw_eval_box_count[L];
    }
    nvtxRangePop();
    sctl::Profile::Toc();

    // --- Scratch strides / sizes ---
    auto &sc = in.scratch;
    sc.pw_in_stride_reals = 2L * fou.n_charge_dim * fou.pw_slab;
    sc.pw_form_stride_reals = (fou.n_tables_up != fou.n_charge_dim) ? 2L * fou.n_tables_up * fou.pw_slab : 0;
    sc.proxy_coeffs_downward_dim = tree.proxy_coeffs_downward_dim;
    sc.pw_out_dim = tree.pw_out_size;

    // --- Outputs ---
    auto &out = in.outputs;
    out.kernel = tree.params.kernel;
    out.eval_src = tree.params.eval_src;
    out.eval_trg = tree.params.eval_trg;
    out.pot_src_dof = tree.kernel_output_dim_src;
    out.pot_trg_dof = tree.kernel_output_dim_trg;
    out.pot_src_size = static_cast<std::size_t>(tree.n_src_owned) * out.pot_src_dof;
    out.pot_trg_size = static_cast<std::size_t>(tree.n_trg_owned) * out.pot_trg_dof;

    return in;
}

template <typename Real, int DIM>
State<Real, DIM>::State(const BuildInputs<Real, DIM> &in) {
    kernel = in.outputs.kernel;
    n_boxes = in.topology.n_boxes;
    n_levels = in.topology.n_levels;
    const auto &md = *in.device_metadata;
    const long nb = n_boxes;

    // --- Topology, adopted: derived on the device and owned by the tree, which outlives this ---
    topology.nlist1_stride = in.topology.nlist1_stride;
    topology.n_neighbors = in.topology.n_neighbors;
    topology.direct_occupancy_src = md.direct_occupancy_src;
    topology.direct_occupancy_trg = md.direct_occupancy_trg;
    topology.d_direct_work.adopt(md.d_sorted, md.n_direct_work);
    topology.d_list1_flat.adopt(md.d_list1, nb * topology.nlist1_stride);
    topology.d_list1_count.adopt(md.d_list1_count, nb);
    if (md.d_list1_shift)
        topology.d_list1_shift.adopt(md.d_list1_shift, nb * topology.nlist1_stride * DIM);
    topology.d_box_levels.adopt(md.d_box_levels, nb);
    topology.d_ifpwexp.adopt(md.d_ifpwexp, nb);
    topology.d_is_global_leaf.adopt(md.d_is_global_leaf, nb);

    // --- Particles ---
    const auto &pi = in.particles;
    particles.d_r_src = pi.d_r_src;
    particles.d_r_trg = pi.d_r_trg;
    particles.d_src_counts.adopt(md.d_src_counts_with_halo, nb);
    particles.d_trg_counts.adopt(md.d_trg_counts_owned, nb);
    particles.d_src_counts_owned.adopt(md.d_src_counts_owned, nb);
    const auto offsets = [&](int kind) { return md.d_offsets + (long)kind * nb; };
    particles.d_r_src_offsets.adopt(offsets(off_r_src_halo), nb);
    particles.d_r_trg_offsets.adopt(offsets(off_r_trg), nb);
    particles.d_charge_offsets.adopt(offsets(off_charge_halo), nb);
    const int charge_dof = get_kernel_input_dim(DIM, kernel);
    if (pi.is_stresslet) {
        particles.d_normal_offsets.adopt(offsets(off_normal_halo), nb);
        particles.normal_offset_owned = DIM * pi.n_src_before_owned;
    }
    particles.d_r_src_owned = pi.d_r_src.subspan(DIM * pi.n_src_before_owned, DIM * pi.n_src_owned);
    particles.d_r_src_offsets_owned.adopt(offsets(off_r_src_owned), nb);
    particles.d_charge_offsets_owned.adopt(offsets(off_charge_owned), nb);
    particles.charge_offset_owned = charge_dof * pi.n_src_before_owned;

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
    fourier.hpw_per_level = fi.hpw_per_level;
    fourier.direct_coeffs_by_level = fi.direct_coeffs_by_level;
    fourier.direct_coeffs_level0 = fi.direct_coeffs_level0;
    fourier.n_digits = fi.n_digits;
    fourier.beta = fi.beta;
    fourier.fparam = fi.fparam;
    fourier.pw2poly_per_level_reals = fi.pw2poly_per_level_reals;
    fourier.poly2pw_per_level_reals = fi.poly2pw_per_level_reals;
    fourier.radialft_per_level_reals = fi.radialft_per_level_reals;
    fourier.phase_per_level_reals = fi.phase_per_level_reals;
    fourier.proxy_ball_r2 = fi.proxy_ball_r2;
    fourier.n_pw_live = fi.n_pw_live;
    fourier.pw_slab = fi.pw_slab;
    fourier.d_centers.adopt(md.d_centers, nb * DIM);
    { // cached for the configuration and owned by the cache, which outlives every State
        const auto &t = *fi.tables;
        const auto adopt = [](auto &dst, const auto &src) { dst.adopt(src.data(), src.size()); };
        adopt(fourier.d_pw2poly_flat, t.pw2poly);
        adopt(fourier.d_poly2pw_flat, t.poly2pw);
        adopt(fourier.d_radialft_flat, t.radialft);
        adopt(fourier.d_phase_flat, t.phase);
        adopt(fourier.d_mode_axes, t.mode_axes);
        adopt(fourier.d_pencil_slots, t.pencil_slots);
        adopt(fourier.d_full_of_compact, t.full_of_compact);
        adopt(fourier.d_window_pw2poly, t.window_pw2poly);
        adopt(fourier.d_window_poly2pw, t.window_poly2pw);
        adopt(fourier.d_window_radialft, t.window_radialft);
        adopt(fourier.d_p2c, t.p2c);
        adopt(fourier.d_c2p, t.c2p);
        adopt(fourier.d_inv_box_scale, t.inv_box_scale);
        adopt(fourier.d_direct_rsc, t.direct_rsc);
        adopt(fourier.d_direct_cen, t.direct_cen);
        adopt(fourier.d_direct_d2max, t.direct_d2max);
    }

    // --- Worklists, adopted ---
    const auto &wi = in.worklists;
    worklists.n_c2p_groups = md.n_c2p_groups;
    worklists.d_c2p_center_boxes.adopt(md.d_c2p_center_boxes, md.n_c2p_groups);
    worklists.d_c2p_levels.adopt(md.d_c2p_levels, md.n_c2p_groups);
    worklists.d_c2p_src_box_flat_offsets.adopt(md.d_c2p_src_box_flat_offsets, md.n_c2p_groups);
    worklists.d_c2p_n_src_boxes_per_group.adopt(md.d_c2p_n_src_boxes_per_group, md.n_c2p_groups);
    worklists.d_c2p_src_boxes_flat.adopt(md.d_c2p_src_boxes_flat, nb);
    worklists.d_c2p_group_perm.adopt(md.d_sorted + nb, md.n_c2p_groups);

    const int n_tp = wi.tp_offset.empty() ? 0 : wi.tp_offset[n_levels];
    worklists.d_tp_parents.adopt(md.d_tp_parents, n_tp);
    worklists.d_tp_children.adopt(md.d_tp_children, n_tp);
    worklists.d_tp_octants.adopt(md.d_tp_octants, n_tp);
    worklists.d_tp_assign_dst.adopt(md.d_tp_assign_dst, n_tp);
    worklists.n_eval_boxes = md.n_eval_boxes;
    worklists.d_eval_targets_box_list.adopt(md.d_eval_targets, md.n_eval_boxes);

    const int n_up = wi.tp_up_offset.empty() ? 0 : wi.tp_up_offset[n_levels];
    const int n_up_par = wi.tp_up_par_offset.empty() ? 0 : wi.tp_up_par_offset[n_levels];
    worklists.d_tp_up_src_boxes.adopt(md.d_tp_up_src, n_up);
    worklists.d_tp_up_dst_boxes.adopt(md.d_tp_up_dst, n_up);
    worklists.d_tp_up_octants.adopt(md.d_tp_up_octants, n_up);
    worklists.d_tp_up_assign.adopt(md.d_tp_up_assign, n_up);
    worklists.d_tp_up_par.adopt(md.d_tp_up_par, n_up_par);
    worklists.d_tp_up_par_child_begin.adopt(md.d_tp_up_par_begin, n_up_par);
    worklists.d_tp_up_par_child_count.adopt(md.d_tp_up_par_count, n_up_par);

    worklists.n_proxy_zero_boxes = md.n_proxy_zero_boxes;
    worklists.d_proxy_zero_boxes.adopt(md.d_proxy_zero_boxes, md.n_proxy_zero_boxes);
    const int n_pw_eval = wi.pw_eval_box_offset.empty() ? 0 : wi.pw_eval_box_offset[n_levels];
    const int n_pw_form = wi.pw_form_box_offset.empty() ? 0 : wi.pw_form_box_offset[n_levels];
    worklists.d_pw_eval_box_flat.adopt(md.d_pw_eval_flat, n_pw_eval);
    worklists.d_pw_form_box_flat.adopt(md.d_pw_form_flat, n_pw_form);
    worklists.d_shift_halo.adopt(wi.d_shift_halo, md.n_shift_groups);
    worklists.d_self_correction_work.adopt(md.d_self_correction_work, md.n_direct_work);
    worklists.shift_group_base_h = wi.shift_group_base;
    worklists.pw_in_pool_base_h = wi.pw_in_pool_base;
    worklists.tp_offset_h = wi.tp_offset;
    worklists.tp_count_h = wi.tp_count;
    worklists.tp_up_offset_h = wi.tp_up_offset;
    worklists.tp_up_count_h = wi.tp_up_count;
    worklists.tp_up_par_offset_h = wi.tp_up_par_offset;
    worklists.tp_up_par_count_h = wi.tp_up_par_count;
    worklists.pw_eval_box_offset_h = wi.pw_eval_box_offset;
    worklists.pw_eval_box_count_h = wi.pw_eval_box_count;
    worklists.pw_form_box_offset_h = wi.pw_form_box_offset;
    worklists.pw_form_box_count_h = wi.pw_form_box_count;

    // --- Scratch (buffers sized/zeroed here; contents produced by the passes) ---
    const auto &si = in.scratch;
    scratch.pw_in_stride_reals = si.pw_in_stride_reals;
    scratch.pw_form_stride_reals = si.pw_form_stride_reals;
    // Neither proxy buffer is zeroed: every eval assigns each slab it reads before accumulating into it.
    if (si.d_proxy_coeffs.size())
        scratch.d_proxy_coeffs_upward = si.d_proxy_coeffs;
    scratch.d_proxy_offsets_upward.adopt(md.d_proxy_off_up, nb);
    scratch.d_proxy_offsets_downward.adopt(md.d_proxy_off_down, nb);
    scratch.d_pw_out_offsets.adopt(md.d_pw_out_offsets, nb);
    // One allocation holds the outgoing PW field and then the downward proxies: every write to the
    // proxies comes after shift_pw, the field's last reader, and every read of them comes before
    // the next eval's proxy2pw.
    scratch.d_pw_out_proxy_down.resize(std::max<std::size_t>(2 * si.pw_out_dim, si.proxy_coeffs_downward_dim));
    if (si.proxy_coeffs_downward_dim)
        scratch.d_proxy_coeffs_downward.adopt(scratch.d_pw_out_proxy_down.data(), si.proxy_coeffs_downward_dim);

    long total_slots = 0;
    for (int L = 0; L < n_levels; ++L)
        total_slots += wi.pw_eval_box_count[L];
    if (total_slots && scratch.pw_in_stride_reals)
        scratch.d_pw_in_pool.resize((std::size_t)total_slots * scratch.pw_in_stride_reals);

    const bool split_up_down = fourier.n_tables_up != fourier.n_charge_dim;
    if (split_up_down && wi.max_pw_form_per_level && scratch.pw_form_stride_reals)
        scratch.d_pw_form_pool.resize((std::size_t)wi.max_pw_form_per_level * scratch.pw_form_stride_reals);

    if (fourier.n_pw_modes_win) {
        scratch.d_window_pw_form_in.resize(2 * (std::size_t)fourier.n_tables_up * fourier.n_pw_modes_win);
        if (split_up_down)
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
    outputs.d_pot_src_offsets.adopt(offsets(off_pot_src), nb);
    outputs.d_pot_trg_offsets.adopt(offsets(off_pot_trg), nb);

    outputs.d_pot_direct_src.resize(outputs.pot_src_size);
    outputs.d_pot_direct_trg.resize(outputs.pot_trg_size);
    outputs.d_pot_eval_src.resize(outputs.pot_src_size);
    outputs.d_pot_eval_trg.resize(outputs.pot_trg_size);
    outputs.d_pot_src_final.resize(outputs.pot_src_size);
    outputs.d_pot_trg_final.resize(outputs.pot_trg_size);

    direct_stream = cuda_helpers::direct_stream();
    downward_stream = cuda_helpers::downward_stream();

    outputs.d_pot_eval_src.zero_async(downward_stream);
    outputs.d_pot_eval_trg.zero_async(downward_stream);
    outputs.d_pot_direct_src.zero_async(direct_stream);
    outputs.d_pot_direct_trg.zero_async(direct_stream);
}

template <typename Real, int DIM>
void State<Real, DIM>::set_charges(Real *d_charge_sorted, long n_charge, Real *d_normal_sorted, long n_normal) {
    particles.d_charge = {d_charge_sorted, (std::size_t)n_charge};
    const long n_owned = particles.d_r_src_owned.size() / DIM;
    Real *d_charge_owned = d_charge_sorted + particles.charge_offset_owned;

    if (kernel == DMK_STRESSLET) {
        particles.d_normal = {d_normal_sorted, (std::size_t)n_normal};
        particles.d_charge_outer.resize(n_owned * DIM * DIM);
        launch_stresslet_charge(d_charge_owned, d_normal_sorted + particles.normal_offset_owned,
                                particles.d_charge_outer.data(), n_owned, DIM, direct_stream);
        DMK_CHECK_CUDA(cudaStreamSynchronize(direct_stream));
    } else {
        particles.d_charge_owned = {d_charge_owned, (std::size_t)(fourier.n_tables_up * n_owned)};
    }
}

template <typename Real, int DIM>
void State<Real, DIM>::finalize() {
    // direct_stream must see the eval-side writes (queued on downward_stream)
    // before the accumulate; direct_stream is already serial with the direct
    // pass's own writes to d_pot_direct_*.
    auto eval_done = cuda_helpers::DeviceEvent::disable_timing();
    DMK_CHECK_CUDA(cudaEventRecord(eval_done, downward_stream));
    DMK_CHECK_CUDA(cudaStreamWaitEvent(direct_stream, eval_done, 0));

    if (outputs.pot_src_size) {
        const long n = static_cast<long>(outputs.pot_src_size / outputs.pot_src_dof);
        launch_accumulate<Real>(outputs.d_pot_src_final.data(), outputs.d_pot_eval_src.data(),
                                outputs.d_pot_direct_src.data(), outputs.pot_src_dof, n, direct_stream);
    }
    if (outputs.pot_trg_size) {
        const long n = static_cast<long>(outputs.pot_trg_size / outputs.pot_trg_dof);
        launch_accumulate<Real>(outputs.d_pot_trg_final.data(), outputs.d_pot_eval_trg.data(),
                                outputs.d_pot_direct_trg.data(), outputs.pot_trg_dof, n, direct_stream);
    }
    DMK_CHECK_CUDA(cudaStreamSynchronize(direct_stream));
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

template PwSlabLayout pw_slab_layout<float, 2>(DMKPtTree<float, 2> &);
template PwSlabLayout pw_slab_layout<float, 3>(DMKPtTree<float, 3> &);
template PwSlabLayout pw_slab_layout<double, 2>(DMKPtTree<double, 2> &);
template PwSlabLayout pw_slab_layout<double, 3>(DMKPtTree<double, 3> &);
template BuildInputs<float, 2> to_build_inputs<float, 2>(DMKPtTree<float, 2> &, const DeviceMetadataOutputs<float> &);
template BuildInputs<float, 3> to_build_inputs<float, 3>(DMKPtTree<float, 3> &, const DeviceMetadataOutputs<float> &);
template BuildInputs<double, 2> to_build_inputs<double, 2>(DMKPtTree<double, 2> &,
                                                           const DeviceMetadataOutputs<double> &);
template BuildInputs<double, 3> to_build_inputs<double, 3>(DMKPtTree<double, 3> &,
                                                           const DeviceMetadataOutputs<double> &);

template struct State<float, 2>;
template struct State<float, 3>;
template struct State<double, 2>;
template struct State<double, 3>;

} // namespace dmk::cuda::pt
