#pragma once

/// @file
/// V2 point-tree GPU state.
///
/// `BuildInputs` is the tree->GPU seam: the scalars and per-level tables the passes launch with,
/// the Fourier tables, and a view of the per-box metadata the device derived. Both `BuildInputs`
/// and `State` share the same six groups (topology / particles / fourier / worklists / scratch /
/// outputs). `to_build_inputs` is the single point that reads DMKPtTree internals; `State`'s ctor
/// consumes only a `BuildInputs` and knows nothing of the tree. The passes that consume each buffer
/// are noted per member: upward (charge2proxy + tensorprod), form_outgoing (proxy2pw +
/// multiply_kernelft), downward (shift_pw + pw2proxy + tensorprod), eval_targets, direct
/// (near-field), and finalize (descatter/merge).

#include <cstddef>
#include <span>
#include <vector>

#include <cuda_runtime.h>
#include <dmk.h>
#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/shift_pw_kernelargs.hpp>

namespace dmk {

template <typename Real, int DIM>
struct DMKPtTree;

namespace cuda::pt {

/// Per-box metadata the device derived; defined in src/cuda/pt/metadata.hpp, which this header
/// deliberately does not include (it is an nvcc-facing header).
template <typename Real>
struct DeviceMetadataOutputs;

/// The per-configuration device tables, cached across builds; defined in state.cpp.
template <typename Real>
struct FourierTables;

using cuda_helpers::DeviceBuffer;

/// The tree->GPU seam. Every per-box array is device-resident already and reached through
/// `device_metadata`; what is here is scalars, per-level host tables, and the Fourier tables.
template <typename Real, int DIM>
struct BuildInputs {
    /// Box structure.
    struct Topology {
        int n_boxes = 0;             ///< total boxes in the tree
        int n_levels = 0;            ///< number of live levels
        int nlist1_stride = 0;       ///< max near-neighbor source boxes per box
        int n_neighbors = 0;         ///< 3^DIM colocated-neighbor slots per box
        const long *d_nbr = nullptr; ///< [n_boxes*3^DIM] in the tree's device memory
    } topology;

    /// Sorted source/target coordinates and charges.
    struct Particles {
        bool is_stresslet = false;   ///< selects the outer(force,normal) proxy path
        std::span<Real> d_r_src;     ///< sorted source coords, in the tree's device memory
        std::span<Real> d_r_trg;     ///< sorted target coords, in the tree's device memory
        long n_src_owned = 0;        ///< sources this rank owns; they are one contiguous run
        long n_src_before_owned = 0; ///< halo sources ahead of that run in the sorted arrays
    } particles;

    /// Fourier transforms plus per-level geometry constants consumed by the compute kernels.
    struct Fourier {
        int n_pw = 0;                                ///< linear plane-wave count (difference kernel)
        int n_pw2 = 0;                               ///< (n_pw+1)/2
        int n_pw_modes = 0;                          ///< PW modes per box (difference)
        int n_charge_dim = 0;                        ///< = n_tables_down (PW component count)
        int n_tables_up = 0;                         ///< upward proxy component count
        int n_order = 0;                             ///< linear proxy order
        int n_pw_win = 0;                            ///< linear plane-wave count (windowed root)
        int n_pw2_win = 0;                           ///< (n_pw_win+1)/2
        int n_pw_modes_win = 0;                      ///< PW modes per box (windowed root)
        Real hpw_win = 0;                            ///< windowed plane-wave spacing
        int n_digits = 0;                            ///< requested accuracy digits (direct coeff generation)
        double beta = 0;                             ///< PSWF bandwidth (direct coeff generation)
        double fparam = 0;                           ///< kernel param (Yukawa lambda); direct coeff cache key
        int pw2poly_per_level_reals = 0;             ///< = 2*n_pw*n_order
        int poly2pw_per_level_reals = 0;             ///< = 2*n_pw*n_order
        int radialft_per_level_reals = 0;            ///< = n_pw_modes
        int wpwshift_per_level_reals = 0;            ///< = 2*n_neighbors*n_pw_modes
        std::vector<Real> hpw_per_level;             ///< [n_levels] hpw_diff/boxsize (form_outgoing/downward)
        int n_pw_live = 0;                           ///< live modes per slab; == n_pw_modes when nothing is prunable
        int proxy_ball_r2 = 0;                       ///< squared ball radius in (i,j,k); 0 keeps every coefficient
        const FourierTables<Real> *tables = nullptr; ///< every device table that depends only on the configuration
        std::vector<std::vector<Real>> direct_coeffs_by_level; ///< yukawa: residual poly per level (direct)
        int direct_coeffs_level0 = 0;                          ///< tree level of direct_coeffs_by_level[0]
    } fourier;

    /// Per-level tables driving the pass launches; the lists they index are device-resident.
    struct Worklists {
        int max_tp_per_level = 0;            ///< max downward tensorprod pairs on any level
        int max_tp_up_per_level = 0;         ///< max upward tensorprod pairs on any level
        int max_pw_form_per_level = 0;       ///< max proxy2pw boxes on any level
        std::vector<int> tp_offset;          ///< [n_levels+1] into the downward pairs
        std::vector<int> tp_count;           ///< [n_levels]
        std::vector<int> tp_up_offset;       ///< [n_levels+1] into the upward pairs
        std::vector<int> tp_up_count;        ///< [n_levels]
        std::vector<int> tp_up_par_offset;   ///< [n_levels+1] into the upward parents
        std::vector<int> tp_up_par_count;    ///< [n_levels]
        std::vector<int> pw_eval_box_offset; ///< [n_levels+1] into the PW-work boxes
        std::vector<int> pw_eval_box_count;  ///< [n_levels]
        std::vector<int> shift_group_base;   ///< [n_levels+1] first global group id of each level
        std::vector<int> pw_form_box_offset; ///< [n_levels+1] into the proxy2pw boxes
        std::vector<int> pw_form_box_count;  ///< [n_levels]
        std::vector<long> pw_in_pool_base;   ///< [n_levels] pw_in slab base (prefix sum of pw_eval_box_count)
        const ShiftPwGroupSrc *d_shift_group_src = nullptr; ///< merged shift sources, CSR by group, device
        const int *d_shift_group_offsets = nullptr;         ///< [n_groups+1] CSR offsets, device
    } worklists;

    /// Pipeline scratch/intermediate buffers and their strides.
    struct Scratch {
        long tensorprod_scratch_stride_reals = 0;  ///< = 2*n_order^DIM per pair slab
        long pw_in_stride_reals = 0;               ///< = 2*n_charge_dim*n_pw_modes per slot
        long pw_form_stride_reals = 0;             ///< stresslet: 2*n_tables_up*n_pw_modes per slot
        std::size_t proxy_coeffs_downward_dim = 0; ///< downward proxy buffer size (reals)
        std::size_t pw_out_dim = 0;                ///< pw_out complex count (d_pw_out reals = 2x)
        std::span<Real> d_proxy_coeffs;            ///< the tree's own upward proxy storage
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
    } outputs;

    /// The per-box arrays, all device-resident and owned by the tree, which outlives the State.
    const DeviceMetadataOutputs<Real> *device_metadata = nullptr;
};

/// Populate a BuildInputs from the tree's level tables and the device metadata. The only place
/// that reads tree internals for the V2 path.
template <typename Real, int DIM>
BuildInputs<Real, DIM> to_build_inputs(DMKPtTree<Real, DIM> &tree, const DeviceMetadataOutputs<Real> &md);

/// Grouped device state for the V2 point-tree pipeline. Adopted from the device metadata or
/// uploaded from a BuildInputs; the only state that must outlive the producer tree is the
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
        DeviceBuffer<signed char> d_list1_shift;      ///< PBC image shift per list1 entry, unallocated if aperiodic
        DeviceBuffer<int> d_box_levels;               ///< depth per box (all passes)
        DeviceBuffer<unsigned char> d_ifpwexp;        ///< has-PW-expansion flag (upward/form_outgoing/downward)
        DeviceBuffer<unsigned char> d_is_global_leaf; ///< leaf over all ranks (shift group build)
    } topology;

    /// Sorted source/target coordinates and charges.
    struct Particles {
        std::span<Real> d_r_src;                   ///< sorted source coords, the tree's storage (direct)
        DeviceBuffer<long> d_r_src_offsets;        ///< per-box offsets into d_r_src
        DeviceBuffer<int> d_src_counts;            ///< sources per box, ghosts included
        std::span<Real> d_charge;                  ///< sorted charges, input dof per source (direct)
        DeviceBuffer<long> d_charge_offsets;       ///< per-box offsets into d_charge
        std::span<Real> d_normal;                  ///< stresslet: sorted normals, the tree's storage (direct)
        DeviceBuffer<long> d_normal_offsets;       ///< stresslet: per-box offsets into d_normal
        DeviceBuffer<Real> d_charge_outer;         ///< stresslet: outer(force,normal) proxy charges (upward)
        std::span<Real> d_r_trg;                   ///< sorted target coords, the tree's storage (direct/eval_targets)
        DeviceBuffer<long> d_r_trg_offsets;        ///< per-box offsets into d_r_trg
        DeviceBuffer<int> d_trg_counts;            ///< owned targets per box
        std::span<Real> d_r_src_owned;             ///< owned part of d_r_src (upward/direct/eval_targets)
        DeviceBuffer<long> d_r_src_offsets_owned;  ///< per-box offsets into d_r_src_owned
        DeviceBuffer<int> d_src_counts_owned;      ///< owned sources per box
        std::span<Real> d_charge_owned;            ///< owned part of d_charge (upward)
        DeviceBuffer<long> d_charge_offsets_owned; ///< per-box offsets into d_charge_owned / d_charge_outer
        long charge_offset_owned = 0;              ///< start of the owned part of d_charge
        long normal_offset_owned = 0;              ///< stresslet: the same in d_normal
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
        double fparam = 0;                    ///< kernel param (Yukawa lambda); direct coeff cache key
        int pw2poly_per_level_reals = 0;      ///< flat stride for d_pw2poly_flat
        int poly2pw_per_level_reals = 0;      ///< flat stride for d_poly2pw_flat
        int radialft_per_level_reals = 0;     ///< flat stride for d_radialft_flat
        int wpwshift_per_level_reals = 0;     ///< flat stride for d_wpwshift_flat
        DeviceBuffer<Real> d_pw2poly_flat;    ///< per-level PW->proxy (downward pw2proxy)
        DeviceBuffer<Real> d_poly2pw_flat;    ///< per-level proxy->PW (form_outgoing proxy2pw)
        DeviceBuffer<Real> d_radialft_flat;   ///< per-level per-mode kernel FT (form_outgoing multiply)
        DeviceBuffer<Real> d_wpwshift_flat;   ///< per-level PW translation per neighbor (downward shift_pw)
        DeviceBuffer<int> d_pencil_slots;     ///< per-(m2,m3) slot arithmetic for proxy2pw/pw2proxy
        DeviceBuffer<int> d_full_of_compact;  ///< slab slot -> cube index (multiply, for its k vector)
        int n_pw_live = 0;                    ///< live modes per slab; == n_pw_modes when nothing is prunable
        int proxy_ball_r2 = 0;                ///< squared ball radius in (i,j,k); 0 keeps every coefficient
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
        std::vector<Real> hpw_per_level;      ///< [n_levels] hpw_diff/boxsize, host (Stokeslet/Stresslet multiply)
        std::vector<std::vector<Real>> direct_coeffs_by_level; ///< yukawa: residual poly per level, host (direct)
        int direct_coeffs_level0 = 0;                          ///< tree level of direct_coeffs_by_level[0]

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
        int n_c2p_groups = 0;                          ///< charge2proxy groups; every one has work
        DeviceBuffer<int> d_c2p_center_boxes;          ///< target box per group (upward)
        DeviceBuffer<int> d_c2p_levels;                ///< level per group (upward)
        DeviceBuffer<int> d_c2p_src_box_flat_offsets;  ///< into d_c2p_src_boxes_flat
        DeviceBuffer<int> d_c2p_n_src_boxes_per_group; ///< source boxes per group
        DeviceBuffer<int> d_c2p_src_boxes_flat;        ///< flattened source boxes (upward)
        DeviceBuffer<int> d_c2p_group_perm;            ///< heaviest-work-first group order
        DeviceBuffer<int> d_tp_parents;                ///< downward tensorprod parent boxes
        DeviceBuffer<int> d_tp_children;               ///< downward tensorprod child boxes
        DeviceBuffer<int> d_tp_octants;                ///< downward tensorprod octant (p2c slab)
        DeviceBuffer<int> d_tp_assign_dst;             ///< downward tensorprod: pair is child's first writer
        DeviceBuffer<int> d_tp_up_src_boxes;           ///< upward tensorprod child boxes
        DeviceBuffer<int> d_tp_up_dst_boxes;           ///< upward tensorprod parent boxes
        DeviceBuffer<int> d_tp_up_octants;             ///< upward tensorprod octant (c2p slab)
        DeviceBuffer<int> d_tp_up_assign;              ///< [n_pairs] pair is its parent's first writer
        DeviceBuffer<int> d_proxy_zero_boxes;          ///< upward proxy slabs no writer touches
        int n_proxy_zero_boxes = 0;
        DeviceBuffer<int> d_tp_up_par;                   ///< upward tensorprod parents, one entry each
        DeviceBuffer<int> d_tp_up_par_child_begin;       ///< [n_par] first child in d_tp_up_src_boxes
        DeviceBuffer<int> d_tp_up_par_child_count;       ///< [n_par] children of that parent
        DeviceBuffer<int> d_pw_eval_box_flat;            ///< per-level PW-work boxes (downward)
        DeviceBuffer<ShiftPwGroupSrc> d_shift_group_src; ///< merged shift sources, CSR by group (downward)
        DeviceBuffer<int> d_shift_group_offsets;         ///< [n_groups+1] CSR offsets into d_shift_group_src
        DeviceBuffer<int> d_pw_form_box_flat;            ///< per-level proxy2pw boxes (form_outgoing)
        int n_eval_boxes = 0;                            ///< |eval_targets_box_list| (eval_targets)
        DeviceBuffer<int> d_eval_targets_box_list;       ///< leaf-of-eval boxes (eval_targets)
        DeviceBuffer<Real> d_self_correction_work;       ///< [n_direct_work] self-correction factors (self_correction)

        // Per-level prefix sums that drive kernel launches (host-resident).
        std::vector<long> pw_in_pool_base_h;   ///< [n_levels] pw_in slab base per level
        std::vector<int> tp_offset_h;          ///< [n_levels+1] into d_tp_* (downward)
        std::vector<int> tp_count_h;           ///< [n_levels] pairs per level (downward)
        std::vector<int> tp_up_offset_h;       ///< [n_levels+1] into d_tp_up_* (upward)
        std::vector<int> tp_up_count_h;        ///< [n_levels] pairs per level (upward)
        std::vector<int> tp_up_par_offset_h;   ///< [n_levels+1] into d_tp_up_par_*
        std::vector<int> tp_up_par_count_h;    ///< [n_levels] parents per level
        std::vector<int> pw_eval_box_offset_h; ///< [n_levels+1] into d_pw_eval_box_flat
        std::vector<int> pw_eval_box_count_h;  ///< [n_levels] boxes per level
        std::vector<int> shift_group_base_h;   ///< [n_levels+1] first global group id of each level (downward)
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

        std::span<Real> d_proxy_coeffs_upward;      ///< upward proxy expansion, the tree's storage
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
        DeviceBuffer<Real> d_pot_direct_src;    ///< near-field src pot, sorted order (direct + self_correction)
        DeviceBuffer<Real> d_pot_direct_trg;    ///< near-field trg pot, sorted order (direct pass)
        DeviceBuffer<Real> d_pot_eval_src;      ///< far-field src pot, sorted order (eval_targets)
        DeviceBuffer<Real> d_pot_eval_trg;      ///< far-field trg pot, sorted order (eval_targets)
        DeviceBuffer<Real> d_pot_src_final;     ///< descattered user-order source pot (finalize->desort)
        DeviceBuffer<Real> d_pot_trg_final;     ///< descattered user-order target pot (finalize->desort)
    } outputs;

    /// Direct runs concurrently with the upward+downward chain; eval waits on both via events.
    /// Process-wide streams, see cuda_helpers::direct_stream.
    cudaStream_t direct_stream = nullptr;
    cudaStream_t downward_stream = nullptr;

    /// Upload raw (user-order) charges/normals and sort them onto the tree.
    /// Charges, and for the stresslet normals, as device pointers in the tree's particle order.
    void set_charges(Real *d_charge_sorted, long n_charge, Real *d_normal_sorted, long n_normal);

    /// Merge the direct (near) + eval (far) sorted potentials and descatter into
    /// user order (d_pot_*_final). direct_stream waits on the eval writes queued
    /// on downward_stream, runs the accumulate+scatter, and syncs.
    void finalize();

    /// Dump device-resident buffers into "gpu_v2/" for offline diffing.
    void dump(DMKPtTree<Real, DIM> &tree);
};

} // namespace cuda::pt
} // namespace dmk
