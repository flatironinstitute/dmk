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
    // kernels writing to it directly (future tensorprod / planewave_to_proxy
    // offload). Read by eval_targets.
    Real *d_proxy_coeffs_downward = nullptr;
    long *d_proxy_offsets_downward = nullptr;
    std::size_t proxy_size = 0;

    // Streams. Direct stays on the default stream for now; eval_targets etc.
    // will get non-blocking streams here when they come online.
    cudaStream_t direct_stream = 0;
};

} // namespace dmk

#endif // DMK_CUDA_SHARED_STATE_HPP
