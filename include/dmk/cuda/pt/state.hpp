#pragma once

/// @file
/// V2 point-tree GPU state.
///
/// `BuildInputs` is the tree->GPU seam: a bundle of host arrays (derived arrays
/// owned inline, verbatim tree arrays viewed as spans) plus scalars and
/// per-level stride constants, grouped by role. Both `BuildInputs` and `State`
/// share the same six groups (topology / particles / fourier / worklists /
/// scratch / outputs). `to_build_inputs` is the single point that reads
/// DMKPtTree internals; `State`'s ctor consumes only a `BuildInputs` and knows
/// nothing of the tree. The passes that consume each buffer are noted per
/// member: upward (charge2proxy + tensorprod), form_outgoing (proxy2pw +
/// multiply_kernelft), downward (shift_pw + pw2proxy + tensorprod),
/// eval_targets, direct (near-field), and finalize (descatter/merge).

#include <cstddef>
#include <span>
#include <vector>

#include <cuda_runtime.h>
#include <dmk.h>
#include <dmk/cuda/helpers.hpp>

namespace dmk {

template <typename Real, int DIM>
struct DMKPtTree;

namespace cuda::pt {

using cuda_helpers::DeviceBuffer;

/// The tree->GPU seam. Derived arrays (host reshaping of nested tree
/// structures) are owned std::vectors; verbatim tree arrays are non-owning
/// spans (the tree outlives the BuildInputs). `State` reads both uniformly.
template <typename Real, int DIM>
struct BuildInputs {
    /// Box structure and per-box gating flags.
    struct Topology {
        int n_boxes = 0;                           ///< total boxes in the tree
        int n_levels = 0;                          ///< number of live levels
        int nlist1_stride = 0;                     ///< max near-neighbor source boxes per box
        int n_neighbors = 0;                       ///< 3^DIM colocated-neighbor slots per box
        std::span<const int> direct_work;          ///< target boxes with near-field work (direct)
        std::vector<int> list1_flat;               ///< [n_boxes*nlist1_stride] near source boxes, -1 pad (direct)
        std::vector<int> list1_count;              ///< [n_boxes] valid entries per row (direct)
        std::vector<int> box_levels;               ///< [n_boxes] depth per box (all passes)
        std::vector<int> neighbors;                ///< [n_boxes*n_neighbors] neighbor ids, -1 invalid (downward)
        std::vector<unsigned char> ifpwexp;        ///< [n_boxes] has-PW-expansion flag (upward/form_outgoing/downward)
        std::vector<unsigned char> is_global_leaf; ///< [n_boxes] leaf-of-eval flag (direct/eval_targets)
    } topology;

    /// Sorted source/target coordinates, charges, and the sort permutation.
    struct Particles {
        bool is_stresslet = false;                  ///< selects the outer(force,normal) proxy path
        std::span<const Real> r_src;                ///< sorted source coords (direct/upward)
        std::span<const Real> r_trg;                ///< sorted target coords (direct/eval_targets)
        std::span<const int> src_counts;            ///< [n_boxes] owned sources per box
        std::span<const int> trg_counts;            ///< [n_boxes] owned targets per box
        std::span<const long> r_src_offsets;        ///< [n_boxes+1] into r_src
        std::span<const long> r_trg_offsets;        ///< [n_boxes+1] into r_trg
        std::span<const long> charge_offsets;       ///< [n_boxes+1] into d_charge (upward/direct)
        std::span<const long> normal_offsets;       ///< stresslet: [n_boxes+1] into d_normal
        std::span<const long> charge_outer_offsets; ///< stresslet: [n_boxes+1] into d_charge_outer
        std::span<const long> scatter_index_src;    ///< sorted->user source perm (upload/finalize)
        std::span<const long> scatter_index_trg;    ///< sorted->user target perm (finalize)
    } particles;

    /// Fourier transforms plus per-level/per-box geometry constants consumed by
    /// the compute kernels.
    struct Fourier {
        int n_pw = 0;                       ///< linear plane-wave count (difference kernel)
        int n_pw2 = 0;                      ///< (n_pw+1)/2
        int n_pw_modes = 0;                 ///< PW modes per box (difference)
        int n_charge_dim = 0;               ///< = n_tables_down (PW component count)
        int n_tables_up = 0;                ///< upward proxy component count
        int n_order = 0;                    ///< linear proxy order
        int n_pw_win = 0;                   ///< linear plane-wave count (windowed root)
        int n_pw2_win = 0;                  ///< (n_pw_win+1)/2
        int n_pw_modes_win = 0;             ///< PW modes per box (windowed root)
        Real hpw_win = 0;                   ///< windowed plane-wave spacing
        int n_digits = 0;                   ///< requested accuracy digits (direct coeff generation)
        double beta = 0;                    ///< PSWF bandwidth (direct coeff generation)
        int pw2poly_per_level_reals = 0;    ///< = 2*n_pw*n_order
        int poly2pw_per_level_reals = 0;    ///< = 2*n_pw*n_order
        int radialft_per_level_reals = 0;   ///< = n_pw_modes
        int wpwshift_per_level_reals = 0;   ///< = 2*n_neighbors*n_pw_modes
        std::span<const Real> p2c;          ///< parent->child proxy matrices (downward tensorprod)
        std::span<const Real> c2p;          ///< child->parent proxy matrices (upward tensorprod)
        std::span<const Real> centers;      ///< [n_boxes*DIM] box centers (charge2proxy/eval_targets)
        std::span<const Real> direct_rsc;   ///< per-level direct coord rescale
        std::span<const Real> direct_cen;   ///< per-level direct center scale
        std::span<const Real> direct_d2max; ///< per-level direct cutoff radius^2
        std::vector<Real> inv_box_scale;    ///< [n_levels] 2/boxsize (charge2proxy/eval_targets)
        std::vector<Real> hpw_per_level;    ///< [n_levels] hpw_diff/boxsize (form_outgoing/downward)
        std::vector<Real> pw2poly_flat;     ///< [n_levels] flat, PW->proxy (downward pw2proxy)
        std::vector<Real> poly2pw_flat;     ///< [n_levels] flat, proxy->PW (form_outgoing proxy2pw)
        std::vector<Real> radialft_flat;    ///< [n_levels] flat, per-mode kernel FT (form_outgoing)
        std::vector<Real> wpwshift_flat;    ///< [n_levels] flat, PW translation per neighbor (downward shift_pw)
        std::vector<Real> window_pw2poly;   ///< root windowed PW->proxy
        std::vector<Real> window_poly2pw;   ///< root windowed proxy->PW
        std::vector<Real> window_radialft;  ///< root windowed per-mode kernel FT
    } fourier;

    /// Precomputed per-group / per-level work lists driving the pass launches.
    struct Worklists {
        int n_c2p_groups = 0;                       ///< charge2proxy groups (all levels)
        int n_c2p_active_groups = 0;                ///< groups with non-zero source work
        int max_tp_per_level = 0;                   ///< max downward tensorprod pairs on any level
        int max_tp_up_per_level = 0;                ///< max upward tensorprod pairs on any level
        int max_pw_form_per_level = 0;              ///< max proxy2pw boxes on any level
        std::vector<int> c2p_center_boxes;          ///< [n_c2p_groups] target box per group (upward)
        std::vector<int> c2p_levels;                ///< [n_c2p_groups] level per group (upward)
        std::vector<int> c2p_src_box_flat_offsets;  ///< [n_c2p_groups] into c2p_src_boxes_flat
        std::vector<int> c2p_n_src_boxes_per_group; ///< [n_c2p_groups] source boxes per group
        std::vector<int> c2p_src_boxes_flat;        ///< flattened source boxes (upward)
        std::vector<int> c2p_group_perm;            ///< [n_c2p_groups] heaviest-work-first order
        std::vector<int> tp_parents;                ///< downward tensorprod parent boxes
        std::vector<int> tp_children;               ///< downward tensorprod child boxes
        std::vector<int> tp_octants;                ///< downward tensorprod child octant (p2c slab)
        std::vector<int> tp_offset;                 ///< [n_levels+1] into tp_* (downward)
        std::vector<int> tp_count;                  ///< [n_levels] pairs per level (downward)
        std::vector<int> tp_up_src;                 ///< upward tensorprod child boxes
        std::vector<int> tp_up_dst;                 ///< upward tensorprod parent boxes
        std::vector<int> tp_up_octants;             ///< upward tensorprod octant (c2p slab)
        std::vector<int> tp_up_offset;              ///< [n_levels+1] into tp_up_* (upward)
        std::vector<int> tp_up_count;               ///< [n_levels] pairs per level (upward)
        std::vector<int> pw_eval_box_flat;          ///< per-level boxes doing PW work (downward)
        std::vector<int> pw_eval_box_offset;        ///< [n_levels+1] into pw_eval_box_flat
        std::vector<int> pw_eval_box_count;         ///< [n_levels] boxes per level
        std::vector<int> pw_form_box_flat;          ///< per-level boxes doing proxy2pw (form_outgoing)
        std::vector<int> pw_form_box_offset;        ///< [n_levels+1] into pw_form_box_flat
        std::vector<int> pw_form_box_count;         ///< [n_levels] boxes per level
        std::vector<long> pw_in_pool_base;          ///< [n_levels] pw_in slab base (prefix sum of pw_eval_box_count)
    } worklists;

    /// Pipeline scratch/intermediate buffers and their strides.
    struct Scratch {
        long tensorprod_scratch_stride_reals = 0;     ///< = 2*n_order^DIM per pair slab
        long pw_in_stride_reals = 0;                  ///< = 2*n_charge_dim*n_pw_modes per slot
        long pw_form_stride_reals = 0;                ///< stresslet: 2*n_tables_up*n_pw_modes per slot
        std::size_t proxy_coeffs_upward_dim = 0;      ///< upward proxy buffer size (reals)
        std::size_t proxy_coeffs_downward_dim = 0;    ///< downward proxy buffer size (reals)
        std::span<const long> proxy_offsets_upward;   ///< [n_boxes] into d_proxy_coeffs_upward
        std::span<const long> proxy_offsets_downward; ///< [n_boxes] into d_proxy_coeffs_downward
        std::span<const long> pw_out_offsets;         ///< [n_boxes] into d_pw_out, -1 = none
    } scratch;

    /// Potential output layout.
    struct Outputs {
        dmk_ikernel kernel = DMK_LAPLACE;       ///< kernel (single source of truth)
        dmk_eval_type eval_src = DMK_POTENTIAL; ///< eval type at sources (direct output dim + coeffs)
        dmk_eval_type eval_trg = DMK_POTENTIAL; ///< eval type at targets
        int pot_src_dof = 0;                    ///< output components per source
        int pot_trg_dof = 0;                    ///< output components per target
        std::size_t pot_src_size = 0;           ///< total source pot reals
        std::size_t pot_trg_size = 0;           ///< total target pot reals
        std::span<const long> pot_src_offsets;  ///< [n_boxes+1] into source pot
        std::span<const long> pot_trg_offsets;  ///< [n_boxes+1] into target pot
    } outputs;
};

/// Populate a BuildInputs from a host-precomputed tree (build_tree_for_gpu +
/// generate_metadata_for_gpu must have run). The only place that reads tree
/// internals for the V2 path.
template <typename Real, int DIM>
BuildInputs<Real, DIM> to_build_inputs(DMKPtTree<Real, DIM> &tree);

/// Grouped device state for the V2 point-tree pipeline. Uploaded verbatim from
/// a BuildInputs; the only state that must outlive the producer tree is the
/// scatter indices + pot metadata (for update_charges / finalize).
template <typename Real, int DIM>
struct State {
    explicit State(const BuildInputs<Real, DIM> &in);
    State(const State &) = delete;
    State &operator=(const State &) = delete;

    dmk_ikernel kernel = DMK_LAPLACE; ///< kernel family (all passes)
    int n_boxes = 0;                  ///< total boxes
    int n_levels = 0;                 ///< live levels

    /// Box structure and per-box gating flags.
    struct Topology {
        int nlist1_stride = 0;                        ///< max near source boxes per box
        int n_neighbors = 0;                          ///< 3^DIM neighbor slots per box
        DeviceBuffer<int> d_direct_work;              ///< target boxes with near-field work (direct)
        DeviceBuffer<int> d_list1_flat;               ///< near source boxes per box (direct)
        DeviceBuffer<int> d_list1_count;              ///< valid list1 entries per box (direct)
        DeviceBuffer<int> d_box_levels;               ///< depth per box (all passes)
        DeviceBuffer<int> d_neighbors;                ///< neighbor box ids per box (downward)
        DeviceBuffer<unsigned char> d_ifpwexp;        ///< has-PW-expansion flag (upward/form_outgoing/downward)
        DeviceBuffer<unsigned char> d_is_global_leaf; ///< leaf-of-eval flag (direct/eval_targets)
    } topology;

    /// Sorted source/target coordinates, charges, and the sort permutation.
    struct Particles {
        DeviceBuffer<Real> d_r_src;                ///< sorted source coords (direct/upward)
        DeviceBuffer<long> d_r_src_offsets;        ///< per-box offsets into d_r_src
        DeviceBuffer<int> d_src_counts;            ///< owned sources per box
        DeviceBuffer<Real> d_charge;               ///< sorted charges, input dof per source (upward/direct)
        DeviceBuffer<long> d_charge_offsets;       ///< per-box offsets into d_charge
        DeviceBuffer<Real> d_normal;               ///< stresslet: sorted normals (upward)
        DeviceBuffer<long> d_normal_offsets;       ///< stresslet: per-box offsets into d_normal
        DeviceBuffer<Real> d_charge_outer;         ///< stresslet: outer(force,normal) proxy charges (upward)
        DeviceBuffer<long> d_charge_outer_offsets; ///< stresslet: per-box offsets into d_charge_outer
        DeviceBuffer<Real> d_r_trg;                ///< sorted target coords (direct/eval_targets)
        DeviceBuffer<long> d_r_trg_offsets;        ///< per-box offsets into d_r_trg
        DeviceBuffer<int> d_trg_counts;            ///< owned targets per box
        DeviceBuffer<long> d_scatter_index_src;    ///< sorted->user source perm (upload/finalize)
        DeviceBuffer<long> d_scatter_index_trg;    ///< sorted->user target perm (finalize)
    } particles;

    /// Fourier transforms plus per-level/per-box geometry constants.
    struct Fourier {
        int n_pw = 0;                         ///< linear PW count (difference)
        int n_pw2 = 0;                        ///< (n_pw+1)/2
        int n_pw_modes = 0;                   ///< PW modes per box (difference)
        int n_charge_dim = 0;                 ///< PW component count (= n_tables_down)
        int n_tables_up = 0;                  ///< upward proxy component count
        int n_order = 0;                      ///< linear proxy order
        int n_pw_win = 0;                     ///< linear PW count (windowed root)
        int n_pw2_win = 0;                    ///< (n_pw_win+1)/2
        int n_pw_modes_win = 0;               ///< PW modes per box (windowed root)
        Real hpw_win = 0;                     ///< windowed PW spacing
        int n_digits = 0;                     ///< requested accuracy digits (direct coeff generation)
        double beta = 0;                      ///< PSWF bandwidth (direct coeff generation)
        int pw2poly_per_level_reals = 0;      ///< flat stride for d_pw2poly_flat
        int poly2pw_per_level_reals = 0;      ///< flat stride for d_poly2pw_flat
        int radialft_per_level_reals = 0;     ///< flat stride for d_radialft_flat
        int wpwshift_per_level_reals = 0;     ///< flat stride for d_wpwshift_flat
        DeviceBuffer<Real> d_pw2poly_flat;    ///< per-level PW->proxy (downward pw2proxy)
        DeviceBuffer<Real> d_poly2pw_flat;    ///< per-level proxy->PW (form_outgoing proxy2pw)
        DeviceBuffer<Real> d_radialft_flat;   ///< per-level per-mode kernel FT (form_outgoing multiply)
        DeviceBuffer<Real> d_wpwshift_flat;   ///< per-level PW translation per neighbor (downward shift_pw)
        DeviceBuffer<Real> d_window_pw2poly;  ///< root windowed PW->proxy (downward)
        DeviceBuffer<Real> d_window_poly2pw;  ///< root windowed proxy->PW (form_outgoing)
        DeviceBuffer<Real> d_window_radialft; ///< root windowed per-mode kernel FT (form_outgoing)
        DeviceBuffer<Real> d_p2c;             ///< parent->child proxy matrices (downward tensorprod)
        DeviceBuffer<Real> d_c2p;             ///< child->parent proxy matrices (upward tensorprod)
        DeviceBuffer<Real> d_centers;         ///< box centers (charge2proxy/eval_targets)
        DeviceBuffer<Real> d_inv_box_scale;   ///< 2/boxsize per level (charge2proxy/eval_targets)
        DeviceBuffer<Real> d_direct_rsc;      ///< per-level direct coord rescale (direct)
        DeviceBuffer<Real> d_direct_cen;      ///< per-level direct center scale (direct)
        DeviceBuffer<Real> d_direct_d2max;    ///< per-level direct cutoff radius^2 (direct)

        /// Per-level base pointers into the flat transform buffers. Single
        /// source of truth for the level*stride arithmetic.
        struct Slab {
            const Real *pw2poly;
            const Real *poly2pw;
            const Real *radialft;
            const Real *wpwshift;
        };
        Slab slab(int level) const {
            return {d_pw2poly_flat.data() + (long)level * pw2poly_per_level_reals,
                    d_poly2pw_flat.data() + (long)level * poly2pw_per_level_reals,
                    d_radialft_flat.data() + (long)level * radialft_per_level_reals,
                    d_wpwshift_flat.data() + (long)level * wpwshift_per_level_reals};
        }
    } fourier;

    /// Precomputed per-group / per-level work lists.
    struct Worklists {
        int n_c2p_groups = 0;                          ///< charge2proxy groups
        int n_c2p_active_groups = 0;                   ///< groups with non-zero work
        DeviceBuffer<int> d_c2p_center_boxes;          ///< target box per group (upward)
        DeviceBuffer<int> d_c2p_levels;                ///< level per group (upward)
        DeviceBuffer<int> d_c2p_src_box_flat_offsets;  ///< into d_c2p_src_boxes_flat
        DeviceBuffer<int> d_c2p_n_src_boxes_per_group; ///< source boxes per group
        DeviceBuffer<int> d_c2p_src_boxes_flat;        ///< flattened source boxes (upward)
        DeviceBuffer<int> d_c2p_group_perm;            ///< heaviest-work-first group order
        DeviceBuffer<int> d_tp_parents;                ///< downward tensorprod parent boxes
        DeviceBuffer<int> d_tp_children;               ///< downward tensorprod child boxes
        DeviceBuffer<int> d_tp_octants;                ///< downward tensorprod octant (p2c slab)
        DeviceBuffer<int> d_tp_up_src_boxes;           ///< upward tensorprod child boxes
        DeviceBuffer<int> d_tp_up_dst_boxes;           ///< upward tensorprod parent boxes
        DeviceBuffer<int> d_tp_up_octants;             ///< upward tensorprod octant (c2p slab)
        DeviceBuffer<int> d_pw_eval_box_flat;          ///< per-level PW-work boxes (downward)
        DeviceBuffer<int> d_pw_form_box_flat;          ///< per-level proxy2pw boxes (form_outgoing)

        // Per-level prefix sums that drive kernel launches (host-resident).
        std::vector<long> pw_in_pool_base_h;   ///< [n_levels] pw_in slab base per level
        std::vector<int> tp_offset_h;          ///< [n_levels+1] into d_tp_* (downward)
        std::vector<int> tp_count_h;           ///< [n_levels] pairs per level (downward)
        std::vector<int> tp_up_offset_h;       ///< [n_levels+1] into d_tp_up_* (upward)
        std::vector<int> tp_up_count_h;        ///< [n_levels] pairs per level (upward)
        std::vector<int> pw_eval_box_offset_h; ///< [n_levels+1] into d_pw_eval_box_flat
        std::vector<int> pw_eval_box_count_h;  ///< [n_levels] boxes per level
        std::vector<int> pw_form_box_offset_h; ///< [n_levels+1] into d_pw_form_box_flat
        std::vector<int> pw_form_box_count_h;  ///< [n_levels] boxes per level
    } worklists;

    /// Pipeline scratch and intermediate expansion buffers.
    struct Scratch {
        long tensorprod_scratch_stride_reals = 0; ///< ping-pong slab stride (up/down tensorprod)
        long pw_in_stride_reals = 0;              ///< pw_in pool slot stride (downward)
        long pw_form_stride_reals = 0;            ///< stresslet pw_form pool slot stride (form_outgoing)
        DeviceBuffer<Real> d_tensorprod_scratch;  ///< ff/ff2 ping-pong (up/down tensorprod)
        DeviceBuffer<Real> d_pw_in_pool;          ///< shift_pw output consumed by pw2proxy (downward)
        DeviceBuffer<Real> d_pw_form_pool;        ///< stresslet proxy2pw->multiply intermediate (form_outgoing)
        DeviceBuffer<Real> d_window_pw_form_in;   ///< root windowed proxy2pw scratch (form_outgoing)
        DeviceBuffer<Real> d_window_pw_form_out;  ///< stresslet root windowed multiply out (form_outgoing)
        DeviceBuffer<int> d_box0_id;              ///< single {0} scratch for root-only kernels

        DeviceBuffer<Real> d_proxy_coeffs_upward;   ///< upward proxy expansion (produced upward, read form_outgoing)
        DeviceBuffer<long> d_proxy_offsets_upward;  ///< per-box offsets into d_proxy_coeffs_upward
        DeviceBuffer<Real> d_proxy_coeffs_downward; ///< downward proxy expansion (produced downward, read eval_targets)
        DeviceBuffer<long> d_proxy_offsets_downward; ///< per-box offsets into d_proxy_coeffs_downward
        DeviceBuffer<Real> d_pw_out;         ///< outgoing PW field per box (produced form_outgoing, read downward)
        DeviceBuffer<long> d_pw_out_offsets; ///< per-box offsets into d_pw_out, -1 = none
    } scratch;

    /// Potential outputs.
    struct Outputs {
        dmk_eval_type eval_src = DMK_POTENTIAL; ///< eval type at sources (direct output dim + coeffs)
        dmk_eval_type eval_trg = DMK_POTENTIAL; ///< eval type at targets
        int pot_src_dof = 0;                    ///< output components per source
        int pot_trg_dof = 0;                    ///< output components per target
        std::size_t pot_src_size = 0;           ///< total source pot reals
        std::size_t pot_trg_size = 0;           ///< total target pot reals
        DeviceBuffer<long> d_pot_src_offsets;   ///< per-box offsets into source pot
        DeviceBuffer<long> d_pot_trg_offsets;   ///< per-box offsets into target pot
        DeviceBuffer<Real> d_pot_direct_src;    ///< near-field src pot, sorted order (direct pass)
        DeviceBuffer<Real> d_pot_direct_trg;    ///< near-field trg pot, sorted order (direct pass)
        DeviceBuffer<Real> d_pot_src_final;     ///< descattered user-order source pot (finalize->desort)
        DeviceBuffer<Real> d_pot_trg_final;     ///< descattered user-order target pot (finalize->desort)
    } outputs;

    /// Direct runs concurrently with the upward+downward chain; eval waits on
    /// both via events.
    cuda_helpers::DeviceStream direct_stream;
    cuda_helpers::DeviceStream downward_stream;

    /// Upload raw (user-order) charges/normals and sort them onto the tree.
    void upload_and_sort_charges(const Real *charges, const Real *normals, long n_src);

    /// Dump device-resident buffers into "gpu_v2/" for offline diffing.
    void dump(DMKPtTree<Real, DIM> &tree);
};

} // namespace cuda::pt
} // namespace dmk
