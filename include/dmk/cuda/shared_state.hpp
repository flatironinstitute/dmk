#ifndef DMK_CUDA_SHARED_STATE_HPP
#define DMK_CUDA_SHARED_STATE_HPP

// Device-side state shared across GPU offload operations within one
// upward_pass→downward_pass cycle. Owned by the tree; borrowed by each
// per-operation context (direct, eval_targets, future tensorprod, …).
//
// Holds the read-only inputs (positions, charges, normals), the tree
// topology arrays (direct_work, list1, box_levels, ifpwexp), the per-level
// direct-eval params (rsc/cen/d2max), and the pot offsets used to index
// per-context output buffers. Output buffers themselves are owned by their
// respective contexts.

#include <vector>

#include <cuda_runtime.h>
#include <dmk.h>
#include <dmk/cuda/helpers.hpp>

namespace dmk {
using cuda_helpers::DeviceBuffer;

template <typename Real, int DIM>
struct DMKPtTree;

template <typename Real, int DIM>
struct CudaSharedDeviceState {
    explicit CudaSharedDeviceState(DMKPtTree<Real, DIM> &tree);
    CudaSharedDeviceState(const CudaSharedDeviceState &) = delete;
    CudaSharedDeviceState &operator=(const CudaSharedDeviceState &) = delete;

    // Topology
    int n_levels = 0;
    int n_boxes = 0;
    int nlist1_stride = 0;
    int n_direct_work = 0;

    DeviceBuffer<int> d_direct_work;
    DeviceBuffer<int> d_list1_flat;
    DeviceBuffer<int> d_list1_count;
    DeviceBuffer<int> d_box_levels;
    DeviceBuffer<unsigned char> d_ifpwexp;

    // Per-level direct-eval params
    DeviceBuffer<Real> d_direct_rsc;
    DeviceBuffer<Real> d_direct_cen;
    DeviceBuffer<Real> d_direct_d2max;

    // Source data with halo
    DeviceBuffer<Real> d_r_src_halo;
    DeviceBuffer<long> d_r_src_halo_offsets;
    DeviceBuffer<int> d_src_counts_halo;

    // For non-stresslet kernels this holds charges; for stresslet it holds densities.
    DeviceBuffer<Real> d_charge_halo;
    DeviceBuffer<long> d_charge_halo_offsets;

    // Stresslet only.
    DeviceBuffer<Real> d_normal_halo;
    DeviceBuffer<long> d_normal_halo_offsets;

    // Owned source positions (target points for the pot_src side).
    DeviceBuffer<Real> d_r_src_owned;
    DeviceBuffer<long> d_r_src_owned_offsets;
    DeviceBuffer<int> d_src_counts_owned;

    // Owned target positions (target points for the pot_trg side).
    DeviceBuffer<Real> d_r_trg_owned;
    DeviceBuffer<long> d_r_trg_owned_offsets;
    DeviceBuffer<int> d_trg_counts_owned;

    // Pot offsets (index into per-context output buffers).
    DeviceBuffer<long> d_pot_src_offsets;
    DeviceBuffer<long> d_pot_trg_offsets;

    std::size_t pot_src_size = 0;
    std::size_t pot_trg_size = 0;

    // Downward-pass proxy expansion. Allocated zero-initialized at construction;
    // populated either by host upload (current eval_targets path) or by GPU
    // kernels writing to it directly (planewave_to_proxy / tensorprod). Read
    // by eval_targets.
    DeviceBuffer<Real> d_proxy_coeffs_downward;
    DeviceBuffer<long> d_proxy_offsets_downward;
    // Set true once the GPU downward path has populated d_proxy_coeffs_downward
    // and the host buffer is *not* the source of truth — eval_targets will
    // then skip its H2D upload.
    bool proxy_resident_on_device = false;

    // ============== Downward-pass GPU plumbing ==============

    // Tree topology.
    int n_neighbors = 0;                          // sctl::pow<DIM>(3)
    DeviceBuffer<int> d_neighbors;                // [n_boxes * n_neighbors]; -1 = invalid
    DeviceBuffer<unsigned char> d_is_global_leaf; // [n_boxes]

    // pw_out written by the GPU form_outgoing kernels each downward_pass.
    // Sizes aren't known at shared-state ctor (init_planewave_data hasn't run
    // yet); allocated by allocate_pw_out().
    DeviceBuffer<Real> d_pw_out;         // interleaved complex
    DeviceBuffer<long> d_pw_out_offsets; // [n_boxes]; -1 = no expansion

    // Per-level pw2poly (interleaved complex, n_pw × n_order each).
    DeviceBuffer<Real> d_pw2poly_flat;
    int pw2poly_per_level_reals = 0; // = 2 * n_pw * n_order

    // Per-level poly2pw (interleaved complex, n_pw × n_order each). Used by
    // proxy2pw to project upward proxy coefficients onto plane-wave modes.
    DeviceBuffer<Real> d_poly2pw_flat;
    int poly2pw_per_level_reals = 0; // = 2 * n_pw * n_order

    // Per-level radialft (real, length n_pw_modes). The kernel-Fourier-transform
    // multiplier applied to each plane-wave mode. For Stokeslet/Stresslet the
    // formula is more involved but uses the same radial profile.
    DeviceBuffer<Real> d_radialft_flat;
    int radialft_per_level_reals = 0; // = n_pw_modes

    // Per-level scaling applied to wave numbers in stokeslet/stresslet
    // multiplies. Host-side (kernels take it as a scalar arg).
    std::vector<Real> hpw_per_level_h; // [n_levels]; = expansion_constants.hpw_diff / boxsize[L]

    // Per-level wpwshift (SoA, n_neighbors × n_pw_modes each).
    DeviceBuffer<Real> d_wpwshift_flat;
    int wpwshift_per_level_reals = 0; // = 2 * n_neighbors * n_pw_modes

    // p2c matrices (per child-octant, DIM matrices of n_order × n_order).
    DeviceBuffer<Real> d_p2c; // [n_octants * DIM * n_order * n_order]
    // c2p matrices, same layout. Used by the upward tensorprod sweep.
    DeviceBuffer<Real> d_c2p;

    // Upward proxy coefficients. Allocated in the ctor, zero-initialized.
    // Either populated by GPU upward (charge2proxy + tensorprod) or by
    // upload_proxy_upward() (CPU fallback).
    DeviceBuffer<Real> d_proxy_coeffs_upward;
    DeviceBuffer<long> d_proxy_offsets_upward;
    // Set true once the GPU upward path has populated d_proxy_coeffs_upward;
    // form_outgoing and any other consumer skip the H2D upload when set.
    bool proxy_upward_resident_on_device = false;

    // Box centers and per-level inverse half-boxsize (= 2/boxsize[L]).
    // Used by charge2proxy and could later replace eval_targets's local copies.
    DeviceBuffer<Real> d_centers;       // [n_boxes * DIM]
    DeviceBuffer<Real> d_inv_box_scale; // [n_levels]

    // Owned source charges (analogue of d_charge_halo, but indexed by owned
    // offsets/counts). Used by upward charge2proxy.
    DeviceBuffer<Real> d_charge_owned;
    DeviceBuffer<long> d_charge_owned_offsets;

    // Per-group charge2proxy work lists (flattened across all levels).
    DeviceBuffer<int> d_c2p_center_boxes;          // [n_c2p_groups]
    DeviceBuffer<int> d_c2p_levels;                // [n_c2p_groups]
    DeviceBuffer<int> d_c2p_src_box_flat_offsets;  // [n_c2p_groups]
    DeviceBuffer<int> d_c2p_n_src_boxes_per_group; // [n_c2p_groups]
    DeviceBuffer<int> d_c2p_src_boxes_flat;        // [c2p_src_boxes_total]
    int n_c2p_groups = 0;
    int c2p_src_boxes_total = 0;

    // Per-level upward tensorprod pair lists (gating differs from downward
    // tp_pairs, hence the separate arrays). Pairs at level L link a child at
    // level L+1 (src) to its parent at level L (dst, additive).
    DeviceBuffer<int> d_tp_up_src_boxes; // i.e. child boxes at level L+1
    DeviceBuffer<int> d_tp_up_dst_boxes; // i.e. parent boxes at level L
    DeviceBuffer<int> d_tp_up_octants;
    std::vector<int> tp_up_offset_h; // [n_levels + 1]
    std::vector<int> tp_up_count_h;  // [n_levels]
    int tp_up_count_total = 0;
    int max_tp_up_per_level = 0;

    // Per-level: list of boxes that do PW work at that level
    // (ifpwexp[b] && nboxpts > 0). Flat array, with start/count per level.
    DeviceBuffer<int> d_pw_eval_box_flat;  // total len = pw_eval_box_count_total
    std::vector<int> pw_eval_box_offset_h; // [n_levels + 1] (host-side, used for kernel launches)
    std::vector<int> pw_eval_box_count_h;  // [n_levels]
    int pw_eval_box_count_total = 0;
    int max_pw_eval_per_level = 0;

    // Per-level: list of boxes that need proxy2pw (form_outgoing) work, gated
    // by ifpwexp[b] && proxy_coeffs_offsets[b] != -1 (i.e. ifpwexp &&
    // src_counts_with_halo > 0). Subset of the PW-eval boxes plus possibly
    // ghost boxes that have upward proxy.
    DeviceBuffer<int> d_pw_form_box_flat;
    std::vector<int> pw_form_box_offset_h; // [n_levels + 1]
    std::vector<int> pw_form_box_count_h;  // [n_levels]
    int pw_form_box_count_total = 0;
    int max_pw_form_per_level = 0;

    // Per-level tensorprod pairs (parent, child, octant).
    DeviceBuffer<int> d_tp_parents; // total len = tp_count_total
    DeviceBuffer<int> d_tp_children;
    DeviceBuffer<int> d_tp_octants;
    std::vector<int> tp_offset_h; // [n_levels + 1]
    std::vector<int> tp_count_h;  // [n_levels]
    int tp_count_total = 0;
    int max_tp_per_level = 0;

    // Global scratch for tensorprod's ff/ff2 ping-pong buffers. Used instead
    // of shared memory because 2 * n_order^3 * sizeof(Real) overruns the
    // per-block shared limit at moderate n_order.
    DeviceBuffer<Real> d_tensorprod_scratch;
    long tensorprod_scratch_stride_reals = 0; // = 2 * n_order^3

    // pw_in scratch pool: per-block slot for the shift_pw output, consumed
    // by pw_to_proxy on the same stream. Sized for the worst-case level.
    DeviceBuffer<Real> d_pw_in_pool;
    long pw_in_stride_reals = 0; // 2 * n_charge_dim * n_pw_modes

    // Stresslet only: per-box pw_form pool used as the proxy2pw → multiply
    // intermediate (n_tables_up = 9, larger than n_tables_down = 3). Empty for
    // other kernels. Sized for max_pw_form_per_level slots.
    DeviceBuffer<Real> d_pw_form_pool;
    long pw_form_stride_reals = 0; // 2 * n_tables_up * n_pw_modes

    // Windowed (root) buffers. Single-slot scratch at n_pw_win size used only
    // for box 0. For non-Stresslet kernels just `_in` is used (multiply runs
    // in place); Stresslet additionally needs `_out` (different table count).
    DeviceBuffer<Real> d_window_pw_form_in;  // 2 * n_tables_up * n_pw_modes_win reals
    DeviceBuffer<Real> d_window_pw_form_out; // 2 * n_tables_down * n_pw_modes_win reals (stresslet only)

    // Windowed Fourier data (single-instance — only used at the root).
    DeviceBuffer<Real> d_window_poly2pw;  // 2 * n_pw_win * n_order reals
    DeviceBuffer<Real> d_window_pw2poly;  // 2 * n_pw_win * n_order reals
    DeviceBuffer<Real> d_window_radialft; // n_pw_modes_win reals

    // Single-element scratch for routines that take per-block box-id / offset
    // arrays but operate on just box 0 (root). Both arrays point at GPU memory
    // populated with [0].
    DeviceBuffer<int> d_box0_id;      // {0}
    DeviceBuffer<long> d_box0_offset; // {0}

    // Cached scalar params for the kernels.
    int n_pw = 0;
    int n_pw2 = 0;        // (n_pw + 1) / 2
    int n_pw_modes = 0;   // n_pw * n_pw * n_pw2 for 3D
    int n_charge_dim = 0; // = n_tables_down
    int n_tables_up = 0;
    int n_order = 0;
    int n_pw_win = 0;
    int n_pw2_win = 0; // (n_pw_win + 1) / 2
    int n_pw_modes_win = 0;
    Real hpw_win = 0; // expansion_constants.hpw_win
    dmk_ikernel kernel = DMK_LAPLACE;

    // Streams (eval_targets stream still owned by its context).
    cudaStream_t direct_stream = 0; // default stream
    cuda_helpers::DeviceStream downward_stream;

    /// Allocate d_pw_out + d_pw_out_offsets. Idempotent. Call once per
    /// downward_pass after init_planewave_data has run; the actual contents
    /// are written by the GPU form_outgoing kernels (or pre-uploaded by the
    /// caller for non-GPU paths if anyone needs that).
    void allocate_pw_out(DMKPtTree<Real, DIM> &tree);

    /// Upload tree.proxy_coeffs_upward to d_proxy_coeffs_upward. Allocates
    /// the device buffer on first call, refills its contents on each call so
    /// repeat evals see the latest charges.
    void upload_proxy_upward(DMKPtTree<Real, DIM> &tree);

    /// Re-upload the charge (and, for stresslet, density + normal) buffers
    /// from the tree's host-side sorted arrays. Call after
    /// DMKPtTree::update_charges so the device sees the new values; sizes are
    /// assumed unchanged (the buffers were sized at construction).
    void upload_charges(DMKPtTree<Real, DIM> &tree);

    /// Dump GPU-resident state to a "gpu/" subdirectory. First runs
    /// tree.dump("gpu/") to lay down topology + metadata, then overwrites the
    /// buffers the GPU owns with their device-side values.
    void dump(DMKPtTree<Real, DIM> &tree);
};

} // namespace dmk

#endif // DMK_CUDA_SHARED_STATE_HPP
