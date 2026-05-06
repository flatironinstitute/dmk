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
//
// One streams lives here too, since they're shared by definition: direct
// uses the default stream for now; non-default streams created here will be
// added as more contexts come online.

#include <vector>

#include <cuda_runtime.h>
#include <dmk.h>

namespace dmk {

template <typename Real, int DIM>
struct DMKPtTree;

template <typename Real, int DIM>
struct CudaSharedDeviceState {
    explicit CudaSharedDeviceState(DMKPtTree<Real, DIM> &tree);
    ~CudaSharedDeviceState();
    CudaSharedDeviceState(const CudaSharedDeviceState &) = delete;
    CudaSharedDeviceState &operator=(const CudaSharedDeviceState &) = delete;

    // Topology
    int n_levels = 0;
    int n_boxes = 0;
    int nlist1_stride = 0;
    int n_direct_work = 0;

    int *d_direct_work = nullptr;
    int *d_list1_flat = nullptr;
    int *d_list1_count = nullptr;
    int *d_box_levels = nullptr;
    unsigned char *d_ifpwexp = nullptr;

    // Per-level direct-eval params
    Real *d_direct_rsc = nullptr;
    Real *d_direct_cen = nullptr;
    Real *d_direct_d2max = nullptr;

    // Source data with halo
    Real *d_r_src_halo = nullptr;
    long *d_r_src_halo_offsets = nullptr;
    int *d_src_counts_halo = nullptr;

    // For non-stresslet kernels this holds charges; for stresslet it holds densities.
    Real *d_charge_halo = nullptr;
    long *d_charge_halo_offsets = nullptr;

    // Stresslet only.
    Real *d_normal_halo = nullptr;
    long *d_normal_halo_offsets = nullptr;

    // Owned source positions (target points for the pot_src side).
    Real *d_r_src_owned = nullptr;
    long *d_r_src_owned_offsets = nullptr;
    int *d_src_counts_owned = nullptr;

    // Owned target positions (target points for the pot_trg side).
    Real *d_r_trg_owned = nullptr;
    long *d_r_trg_owned_offsets = nullptr;
    int *d_trg_counts_owned = nullptr;

    // Pot offsets (index into per-context output buffers).
    long *d_pot_src_offsets = nullptr;
    long *d_pot_trg_offsets = nullptr;

    std::size_t pot_src_size = 0;
    std::size_t pot_trg_size = 0;

    // Downward-pass proxy expansion. Allocated zero-initialized at construction;
    // populated either by host upload (current eval_targets path) or by GPU
    // kernels writing to it directly (planewave_to_proxy / tensorprod). Read
    // by eval_targets.
    Real *d_proxy_coeffs_downward = nullptr;
    long *d_proxy_offsets_downward = nullptr;
    std::size_t proxy_size = 0;
    // Set true once the GPU downward path has populated d_proxy_coeffs_downward
    // and the host buffer is *not* the source of truth — eval_targets will
    // then skip its H2D upload.
    bool proxy_resident_on_device = false;

    // ============== Downward-pass GPU plumbing ==============

    // Tree topology.
    int n_neighbors = 0;                       // sctl::pow<DIM>(3)
    int *d_neighbors = nullptr;                // [n_boxes * n_neighbors]; -1 = invalid
    unsigned char *d_is_global_leaf = nullptr; // [n_boxes]

    // pw_out written by the GPU form_outgoing kernels each downward_pass.
    // Sizes aren't known at shared-state ctor (init_planewave_data hasn't run
    // yet); allocated by allocate_pw_out().
    Real *d_pw_out = nullptr;         // interleaved complex
    long *d_pw_out_offsets = nullptr; // [n_boxes]; -1 = no expansion
    std::size_t pw_out_size = 0;      // reals

    // Per-level pw2poly (interleaved complex, n_pw × n_order each).
    Real *d_pw2poly_flat = nullptr;
    int pw2poly_per_level_reals = 0; // = 2 * n_pw * n_order

    // Per-level poly2pw (interleaved complex, n_pw × n_order each). Used by
    // proxy2pw to project upward proxy coefficients onto plane-wave modes.
    Real *d_poly2pw_flat = nullptr;
    int poly2pw_per_level_reals = 0; // = 2 * n_pw * n_order

    // Per-level radialft (real, length n_pw_modes). The kernel-Fourier-transform
    // multiplier applied to each plane-wave mode. For Stokeslet/Stresslet the
    // formula is more involved but uses the same radial profile.
    Real *d_radialft_flat = nullptr;
    int radialft_per_level_reals = 0; // = n_pw_modes

    // Per-level scaling applied to wave numbers in stokeslet/stresslet
    // multiplies. Host-side (kernels take it as a scalar arg).
    std::vector<Real> hpw_per_level_h; // [n_levels]; = expansion_constants.hpw_diff / boxsize[L]

    // Per-level wpwshift (SoA, n_neighbors × n_pw_modes each).
    Real *d_wpwshift_flat = nullptr;
    int wpwshift_per_level_reals = 0; // = 2 * n_neighbors * n_pw_modes

    // p2c matrices (per child-octant, DIM matrices of n_order × n_order).
    Real *d_p2c = nullptr; // [n_octants * DIM * n_order * n_order]
    // c2p matrices, same layout. Used by the upward tensorprod sweep.
    Real *d_c2p = nullptr;

    // Upward proxy coefficients. Allocated in the ctor, zero-initialized.
    // Either populated by GPU upward (charge2proxy + tensorprod) or by
    // upload_proxy_upward() (CPU fallback).
    Real *d_proxy_coeffs_upward = nullptr;
    long *d_proxy_offsets_upward = nullptr;
    std::size_t proxy_upward_size = 0;
    // Set true once the GPU upward path has populated d_proxy_coeffs_upward;
    // form_outgoing and any other consumer skip the H2D upload when set.
    bool proxy_upward_resident_on_device = false;

    // Box centers and per-level inverse half-boxsize (= 2/boxsize[L]).
    // Used by charge2proxy and could later replace eval_targets's local copies.
    Real *d_centers = nullptr;       // [n_boxes * DIM]
    Real *d_inv_box_scale = nullptr; // [n_levels]

    // Owned source charges (analogue of d_charge_halo, but indexed by owned
    // offsets/counts). Used by upward charge2proxy.
    Real *d_charge_owned = nullptr;
    long *d_charge_owned_offsets = nullptr;

    // Per-group charge2proxy work lists (flattened across all levels).
    int *d_c2p_center_boxes = nullptr;          // [n_c2p_groups]
    int *d_c2p_levels = nullptr;                // [n_c2p_groups]
    int *d_c2p_src_box_flat_offsets = nullptr;  // [n_c2p_groups]
    int *d_c2p_n_src_boxes_per_group = nullptr; // [n_c2p_groups]
    int *d_c2p_src_boxes_flat = nullptr;        // [c2p_src_boxes_total]
    int n_c2p_groups = 0;
    int c2p_src_boxes_total = 0;

    // Per-level upward tensorprod pair lists (gating differs from downward
    // tp_pairs, hence the separate arrays). Pairs at level L link a child at
    // level L+1 (src) to its parent at level L (dst, additive).
    int *d_tp_up_src_boxes = nullptr; // i.e. child boxes at level L+1
    int *d_tp_up_dst_boxes = nullptr; // i.e. parent boxes at level L
    int *d_tp_up_octants = nullptr;
    std::vector<int> tp_up_offset_h; // [n_levels + 1]
    std::vector<int> tp_up_count_h;  // [n_levels]
    int tp_up_count_total = 0;
    int max_tp_up_per_level = 0;

    // Per-level: list of boxes that do PW work at that level
    // (ifpwexp[b] && nboxpts > 0). Flat array, with start/count per level.
    int *d_pw_eval_box_flat = nullptr;     // total len = pw_eval_box_count_total
    std::vector<int> pw_eval_box_offset_h; // [n_levels + 1] (host-side, used for kernel launches)
    std::vector<int> pw_eval_box_count_h;  // [n_levels]
    int pw_eval_box_count_total = 0;
    int max_pw_eval_per_level = 0;

    // Per-level: list of boxes that need proxy2pw (form_outgoing) work, gated
    // by ifpwexp[b] && proxy_coeffs_offsets[b] != -1 (i.e. ifpwexp &&
    // src_counts_with_halo > 0). Subset of the PW-eval boxes plus possibly
    // ghost boxes that have upward proxy.
    int *d_pw_form_box_flat = nullptr;
    std::vector<int> pw_form_box_offset_h; // [n_levels + 1]
    std::vector<int> pw_form_box_count_h;  // [n_levels]
    int pw_form_box_count_total = 0;
    int max_pw_form_per_level = 0;

    // Per-level tensorprod pairs (parent, child, octant).
    int *d_tp_parents = nullptr; // total len = tp_count_total
    int *d_tp_children = nullptr;
    int *d_tp_octants = nullptr;
    std::vector<int> tp_offset_h; // [n_levels + 1]
    std::vector<int> tp_count_h;  // [n_levels]
    int tp_count_total = 0;
    int max_tp_per_level = 0;

    // Global scratch for tensorprod's ff/ff2 ping-pong buffers. Used instead
    // of shared memory because 2 * n_order^3 * sizeof(Real) overruns the
    // per-block shared limit at moderate n_order.
    Real *d_tensorprod_scratch = nullptr;
    long tensorprod_scratch_stride_reals = 0; // = 2 * n_order^3

    // pw_in scratch pool: per-block slot for the shift_pw output, consumed
    // by pw_to_proxy on the same stream. Sized for the worst-case level.
    Real *d_pw_in_pool = nullptr;
    long pw_in_stride_reals = 0; // 2 * n_charge_dim * n_pw_modes

    // Stresslet only: per-box pw_form pool used as the proxy2pw → multiply
    // intermediate (n_tables_up = 9, larger than n_tables_down = 3). Empty for
    // other kernels. Sized for max_pw_form_per_level slots.
    Real *d_pw_form_pool = nullptr;
    long pw_form_stride_reals = 0; // 2 * n_tables_up * n_pw_modes

    // Windowed (root) buffers. Single-slot scratch at n_pw_win size used only
    // for box 0. For non-Stresslet kernels just `_in` is used (multiply runs
    // in place); Stresslet additionally needs `_out` (different table count).
    Real *d_window_pw_form_in = nullptr;  // 2 * n_tables_up * n_pw_modes_win reals
    Real *d_window_pw_form_out = nullptr; // 2 * n_tables_down * n_pw_modes_win reals (stresslet only)

    // Windowed Fourier data (single-instance — only used at the root).
    Real *d_window_poly2pw = nullptr;  // 2 * n_pw_win * n_order reals
    Real *d_window_pw2poly = nullptr;  // 2 * n_pw_win * n_order reals
    Real *d_window_radialft = nullptr; // n_pw_modes_win reals

    // Single-element scratch for routines that take per-block box-id / offset
    // arrays but operate on just box 0 (root). Both arrays point at GPU memory
    // populated with [0].
    int *d_box0_id = nullptr;      // {0}
    long *d_box0_offset = nullptr; // {0}

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
    cudaStream_t direct_stream = 0;         // default stream
    cudaStream_t downward_stream = nullptr; // non-blocking

    /// Allocate d_pw_out + d_pw_out_offsets. Idempotent. Call once per
    /// downward_pass after init_planewave_data has run; the actual contents
    /// are written by the GPU form_outgoing kernels (or pre-uploaded by the
    /// caller for non-GPU paths if anyone needs that).
    void allocate_pw_out(DMKPtTree<Real, DIM> &tree);

    /// Upload tree.proxy_coeffs_upward to d_proxy_coeffs_upward. Allocates
    /// the device buffer on first call, refills its contents on each call so
    /// repeat evals see the latest charges.
    void upload_proxy_upward(DMKPtTree<Real, DIM> &tree);

    /// Dump GPU-resident state to a "gpu/" subdirectory. First runs
    /// tree.dump("gpu/") to lay down topology + metadata, then overwrites the
    /// buffers the GPU owns with their device-side values.
    void dump(DMKPtTree<Real, DIM> &tree);
};

} // namespace dmk

#endif // DMK_CUDA_SHARED_STATE_HPP
