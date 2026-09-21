#pragma once
// Per-box metadata derived on the device from the tree's own node arrays. Kept free of <sctl.hpp>
// and of every dmk header but device_tree.hpp: metadata.cu is compiled by nvcc, which crashes in
// cudafe++ on the SCTL monolith (see src/cuda/esp/sort.cu). device_tree.hpp is safe -- it pulls
// only the narrow sctl headers the device tree itself uses.

#include <dmk/cuda/device_tree.hpp>
#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/shift_pw_kernelargs.hpp>

#include <sctl/morton.hpp>

#include <cuda_runtime.h>

namespace dmk::cuda::pt {

/// Which self-correction constant a box gets. Mirrors the kernel switch in
/// DMKPtTree::build_self_correction_work_list, which is the only place the choice is made.
enum class SelfCorrectionMode { w0, zero, w0_grad };

/// Deepest level a tree can have, plus one: the kernels size their per-level shared tables by it.
constexpr int kMaxLevels = SCTL_MAX_DEPTH + 1;

/// A per-level host table small enough to travel as a kernel argument.
template <typename Real>
struct LevelTable {
    Real v[kMaxLevels + 1];
};

/// The per-box particle offset arrays the eval kernels index, all scalar multiples of the three
/// count scans box_status already runs. Laid out [n_kinds][n_boxes] in one allocation.
enum OffsetKind {
    off_r_src_halo = 0, ///< DIM * sources under the box, halo included
    off_charge_halo,    ///< charge dof * the same
    off_normal_halo,    ///< DIM * the same (stresslet)
    off_r_trg,          ///< DIM * owned targets
    off_pot_trg,        ///< target output dof * owned targets
    off_pot_src,        ///< source output dof * owned sources
    off_r_src_owned,    ///< DIM * owned sources
    off_charge_owned,   ///< n_tables_up * owned sources
    n_offset_kinds
};

/// The tree arrays this pass reads, plus the per-box particle counts the device tree keeps on the
/// host. Reproduces DMKPtTree::accumulate_subtree_counts, compute_proxy_expansion_flags,
/// compute_proxy_evaluation_flags and compute_data_offsets, so neither the neighbour list nor a
/// level sweep is needed here. Owned counts are the halo counts restricted to the owned node range.
template <int DIM>
struct BoxStatusInputs {
    const sctl::Morton<DIM> *d_mid = nullptr;        ///< [n_boxes] device
    const sctl::Long *d_parent = nullptr;            ///< [n_boxes] device
    const sctl::Long *d_child = nullptr;             ///< [n_boxes*2^DIM] device
    const sctl::Long *d_nbr = nullptr;               ///< [n_boxes*3^DIM] device
    const device_tree::NodeAttr *d_attr = nullptr;   ///< [n_boxes] device; used when deriving leafness here
    const unsigned char *h_is_global_leaf = nullptr; ///< [n_boxes] host; only when a reduction produced it
    const sctl::Long *h_src_cnt_with_halo = nullptr; ///< [n_boxes] host, this box only
    const sctl::Long *h_trg_cnt_owned = nullptr;     ///< [n_boxes] host, this box only; null if no targets
    long owned_begin = 0;                            ///< the owned nodes are one contiguous range
    long owned_end = 0;
    long n_boxes = 0;
    int charge_dof = 0; ///< kernel input dim
    int n_tables_up = 0;
    int dof_src = 0; ///< output components per source
    int dof_trg = 0;
};

/// Device arrays are caller-owned and outlive the call. Enqueues only: `ready` is recorded once
/// the depth has landed in `h_max_depth`, and everything else is stream-ordered behind it.
struct BoxStatusOutputs {
    unsigned char *d_is_global_leaf = nullptr;
    int *d_src_counts_with_halo = nullptr;
    int *d_src_counts_owned = nullptr;
    int *d_trg_counts_owned = nullptr;
    unsigned char *d_ifpwexp = nullptr;
    unsigned char *d_iftensprodeval = nullptr;
    long *d_count_scan = nullptr; ///< [3*(n_boxes+1)] exclusive scans of the three per-box counts
    long *d_offsets = nullptr;    ///< [n_offset_kinds*n_boxes]
    long *h_max_depth = nullptr;  ///< page-locked; deepest live level, read once `ready` fires
    cudaEvent_t ready = nullptr;
    int *d_zero = nullptr; ///< [n_zero] cleared by the pass, for the counters the next pass accumulates into
    int n_zero = 0;
};

/// A box's subtree is one contiguous run in Morton order, so the subtree sums are a prefix scan
/// plus a run-end lookup rather than the host's sweep from the deepest level up.
template <int DIM>
void box_status(const BoxStatusInputs<DIM> &in, BoxStatusOutputs &out);

/// The tree's node arrays, as the device already holds them, plus what box_status derived.
/// `is_global_leaf` is deliberately separate from `attr`: leafness as the interaction lists define
/// it is "no rank refined this node", which equals attr.Leaf only at one rank.
template <typename Real, int DIM>
struct DeviceMetadataInputs {
    const sctl::Morton<DIM> *d_mid = nullptr;        ///< [n_boxes] device
    const device_tree::NodeAttr *d_attr = nullptr;   ///< [n_boxes] device; only Ghost is read
    const sctl::Long *d_parent = nullptr;            ///< [n_boxes] device
    const sctl::Long *d_child = nullptr;             ///< [n_boxes*2^DIM] device
    const sctl::Long *d_nbr = nullptr;               ///< [n_boxes*3^DIM] device
    const unsigned char *d_is_global_leaf = nullptr; ///< [n_boxes] device, from box_status
    const unsigned char *d_ifpwexp = nullptr;        ///< [n_boxes] device, from box_status
    const unsigned char *d_iftensprodeval = nullptr; ///< [n_boxes] device, from box_status
    const int *d_src_cnt_owned = nullptr;            ///< [n_boxes] device, from box_status
    const int *d_trg_cnt_owned = nullptr;            ///< [n_boxes] device, from box_status
    const int *d_src_cnt_with_halo = nullptr;        ///< [n_boxes] device, from box_status
    const int *d_src_cnt_global = nullptr;           ///< [n_boxes] device, sources under a box on all ranks
    int n_coeffs_up = 0;
    int n_coeffs_down = 0;
    LevelTable<Real> w0{}; ///< self-interaction constant per level; unused for the zero mode
    long n_boxes = 0;
    int n_levels = 0;
    int nlist1_stride = 0;
    bool periodic = false;
    bool skip_root_form = false; ///< under PBC the root's difference kernel is skipped
    long n_pw_per_box = 0;
    int shift_group = 0; ///< target boxes merged per shift_pw group
    SelfCorrectionMode self_mode = SelfCorrectionMode::w0;
};

/// The one block the host reads back at the join, as longs: eight scalars then six per-level
/// tables of n_levels+1 entries each.
enum JoinSlot {
    join_n_direct_work = 0,
    join_min_direct_level,
    join_n_c2p_groups,
    join_n_eval,
    join_max_pw_form,
    join_n_zero,
    join_pw_out_size,
    join_shift_total,
    join_tables ///< tp_off | tp_up_off | tp_up_par_off | pw_eval_off | pw_form_off | group_base
};
constexpr int join_size(int n_levels) { return join_tables + 6 * (n_levels + 1); }

/// Device-resident outputs. Every pointer is owned by the caller and must be sized before the call.
/// Work-list arrays are bounded by n_boxes and sized to it; only a prefix is live.
template <typename Real>
struct DeviceMetadataOutputs {
    Real *d_centers = nullptr;              ///< [n_boxes*DIM]
    int *d_box_levels = nullptr;            ///< [n_boxes]
    int *d_list1 = nullptr;                 ///< [n_boxes*nlist1_stride]
    int *d_list1_count = nullptr;           ///< [n_boxes]
    signed char *d_list1_shift = nullptr;   ///< [n_boxes*nlist1_stride*DIM], null if aperiodic
    int *d_sorted = nullptr;                ///< [2*n_boxes]: direct work list, then the c2p group permutation
    long *d_sort_keys = nullptr;            ///< [2*n_boxes] scratch
    Real *d_self_correction_work = nullptr; ///< [n_boxes], parallel to the direct work list

    // charge2proxy groups, one per emitting box. Group ids run in box order rather than the
    // host's level-descending order, which the work permutation makes moot.
    int *d_c2p_center_boxes = nullptr;
    int *d_c2p_levels = nullptr;
    int *d_c2p_src_box_flat_offsets = nullptr;
    int *d_c2p_n_src_boxes_per_group = nullptr;
    int *d_c2p_src_boxes_flat = nullptr; ///< a box is one group's source at most
    unsigned char *d_is_c2p_center = nullptr;

    long *d_proxy_off_up = nullptr;    ///< [n_boxes], -1 where the box has no slot
    long *d_proxy_off_down = nullptr;  ///< [n_boxes]
    long *d_proxy_counts_up = nullptr; ///< [n_boxes]
    long *d_proxy_totals = nullptr;    ///< [2] upward, downward slot totals
    long *h_proxy_counts_up = nullptr; ///< [n_boxes] page-locked; the tree's AddData needs it host-side
    long *h_proxy_totals = nullptr;    ///< [2] page-locked
    cudaEvent_t proxy_ready = nullptr; ///< signalled once the two readbacks above have landed

    int *d_pw_eval_flat = nullptr;     ///< [n_boxes] per-level plane-wave boxes, in box order within a level
    int *d_pw_form_flat = nullptr;     ///< [n_boxes]
    int *d_proxy_zero_boxes = nullptr; ///< [n_boxes]
    long *d_pw_out_offsets = nullptr;  ///< [n_boxes], -1 where the box has no slab

    // Tensorprod pairs, level-major. A box is the child of exactly one parent, so every array is
    // bounded by n_boxes.
    int *d_tp_parents = nullptr;
    int *d_tp_children = nullptr;
    int *d_tp_octants = nullptr;
    int *d_tp_assign_dst = nullptr;
    int *d_eval_targets = nullptr; ///< boxes with iftensprodeval, in box order

    // Upward tensorprod pairs, same bound. Parents are ordered heaviest-first within a level;
    // up_par_begin is level-relative, as the launcher indexes it.
    int *d_tp_up_src = nullptr;
    int *d_tp_up_dst = nullptr;
    int *d_tp_up_octants = nullptr;
    int *d_tp_up_assign = nullptr;
    int *d_tp_up_par = nullptr;
    int *d_tp_up_par_begin = nullptr;
    int *d_tp_up_par_count = nullptr;
    unsigned char *d_is_tp_up_par = nullptr;

    // shift_pw groups: counts and CSR offsets over the group bound, the source list once sized.
    int *d_shift_counts = nullptr;  ///< [n_groups_bound+1]
    int *d_shift_offsets = nullptr; ///< [n_groups_bound+1]
    long n_groups_bound = 0;

    // Scratch the pass owns for its duration; the caller frees it after the join.
    void *d_scan = nullptr;         ///< [(n_boxes+1)*sizeof(ScanIn)]
    int *d_chunk_table = nullptr;   ///< [n_chunks*kMaxLevels*2]
    int *d_level_scratch = nullptr; ///< [level_scratch_size(n_levels)]

    long *d_join = nullptr; ///< [join_size(n_levels)] device
    long *h_join = nullptr; ///< page-locked mirror, filled by finish_device_metadata

    // Filled by box_status and kept device-resident so the State adopts them instead of any host
    // mirror making the round trip back across the link.
    const unsigned char *d_is_global_leaf = nullptr;
    const unsigned char *d_ifpwexp = nullptr;
    const int *d_src_counts_with_halo = nullptr;
    const int *d_src_counts_owned = nullptr;
    const int *d_trg_counts_owned = nullptr;
    const long *d_offsets = nullptr; ///< [n_offset_kinds*n_boxes], see OffsetKind

    long n_boxes = 0; ///< set by compute_device_metadata, read by finish_device_metadata
    int n_direct_work = 0;
    int n_c2p_groups = 0;
    int n_eval_boxes = 0;
    int n_proxy_zero_boxes = 0;
    int max_pw_form_per_level = 0;
    int max_tp_per_level = 0;
    int max_tp_up_per_level = 0;
    int n_shift_groups = 0;
    int n_shift_src = 0;
    long pw_out_size = 0;
    /// Shallowest box in the direct work list, or n_levels if it is empty. Yukawa fits one residual
    /// polynomial per level starting here, which is a host decision.
    int min_direct_level = 0;
    const long *h_table(int which) const { return h_join + join_tables + which * (n_levels_ + 1); }
    int n_levels_ = 0;
};

/// Bytes of scan scratch and ints of per-level scratch the pass needs, for the caller to size.
std::size_t scan_scratch_bytes(long n_boxes);
long n_chunks(long n_boxes);
int level_scratch_size(int n_levels);
long shift_groups_bound(long n_boxes, int n_levels, int shift_group);

/// Reproduces DMKPtTree::compute_box_centers, build_direct_interaction_lists,
/// build_direct_work_lists, build_self_correction_work_list, build_upward_pass_work_lists's
/// charge2proxy groups and the per-level lists to_build_inputs used to derive. Enqueues only.
template <typename Real, int DIM>
void compute_device_metadata(const DeviceMetadataInputs<Real, DIM> &in, DeviceMetadataOutputs<Real> &out);

/// Waits only for the proxy readbacks, which the pass enqueues first, rather than for the whole
/// pass; the host needs the counts well before the rest of the metadata is done.
template <typename Real>
void finish_proxy_metadata(const DeviceMetadataOutputs<Real> &out);

/// Enqueued by compute_device_metadata and joined here; the host is free to build its own
/// metadata in between. Fills the scalars in `out` from the join block.
template <typename Real>
void finish_device_metadata(DeviceMetadataOutputs<Real> &out, int n_levels);

/// The shift_pw source lists, once the join has sized them. `d_group_src` must hold n_shift_src.
template <typename Real, int DIM>
void fill_shift_groups(const DeviceMetadataInputs<Real, DIM> &in, const DeviceMetadataOutputs<Real> &out,
                       ShiftPwGroupSrc *d_group_src);

} // namespace dmk::cuda::pt
