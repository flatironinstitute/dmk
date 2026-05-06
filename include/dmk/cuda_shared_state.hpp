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

    // pw_out (filled in upward_pass). Sizes are unknown at shared-state ctor
    // (init_planewave_data hasn't run yet); allocated + uploaded by
    // upload_pw_out() after form_outgoing_expansions completes.
    Real *d_pw_out = nullptr;         // interleaved complex
    long *d_pw_out_offsets = nullptr; // [n_boxes]; -1 = no expansion
    std::size_t pw_out_size = 0;      // reals

    // Per-level pw2poly (interleaved complex, n_pw × n_order each).
    Real *d_pw2poly_flat = nullptr;
    int pw2poly_per_level_reals = 0; // = 2 * n_pw * n_order

    // Per-level wpwshift (SoA, n_neighbors × n_pw_modes each).
    Real *d_wpwshift_flat = nullptr;
    int wpwshift_per_level_reals = 0; // = 2 * n_neighbors * n_pw_modes

    // p2c matrices (per child-octant, DIM matrices of n_order × n_order).
    Real *d_p2c = nullptr; // [n_octants * DIM * n_order * n_order]

    // Per-level: list of boxes that do PW work at that level
    // (ifpwexp[b] && nboxpts > 0). Flat array, with start/count per level.
    int *d_pw_eval_box_flat = nullptr;     // total len = pw_eval_box_count_total
    std::vector<int> pw_eval_box_offset_h; // [n_levels + 1] (host-side, used for kernel launches)
    std::vector<int> pw_eval_box_count_h;  // [n_levels]
    int pw_eval_box_count_total = 0;
    int max_pw_eval_per_level = 0;

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

    // Cached scalar params for the kernels.
    int n_pw = 0;
    int n_pw2 = 0;        // (n_pw + 1) / 2
    int n_pw_modes = 0;   // n_pw * n_pw * n_pw2 for 3D
    int n_charge_dim = 0; // = n_tables_down
    int n_order = 0;

    // Streams (eval_targets stream still owned by its context).
    cudaStream_t direct_stream = 0;         // default stream
    cudaStream_t downward_stream = nullptr; // non-blocking

    /// Allocate + upload pw_out + offsets. Idempotent (no-op if already done).
    /// Call once per downward_pass after form_outgoing_expansions completes.
    void upload_pw_out(DMKPtTree<Real, DIM> &tree);

    /// Dump GPU-resident state to a "gpu/" subdirectory. First runs
    /// tree.dump("gpu/") to lay down topology + metadata, then overwrites the
    /// buffers the GPU owns with their device-side values.
    void dump(DMKPtTree<Real, DIM> &tree);
};

} // namespace dmk

#endif // DMK_CUDA_SHARED_STATE_HPP
