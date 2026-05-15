#ifndef DMK_CUDA_DIRECT_KERNELS_HPP
#define DMK_CUDA_DIRECT_KERNELS_HPP

// Host-includable header for the GPU per-box direct evaluator.
//
// DirectByBoxArgs<Real> is the bag of device pointers + small scalars passed
// to the kernel — orchestration code (cuda_direct.cpp) fills it in once after
// uploading tree metadata, and the kernel reads through it.
//
// launch_direct_by_box_dispatch<Real>(...) selects the right
// (kernel, dim, n_digits) instantiation of DirectResidualByBoxKernel and
// queues it on `stream`. Defined in src/cuda_kernels.cu (auto-generated).

#include <dmk.h>

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real>
struct DirectByBoxArgs {
    // Work list & tree topology
    int n_work = 0;        // |direct_work|
    int n_levels = 0;      // tree level count
    int nlist1_stride = 0; // nlist1_max for the (DIM)
    Real thresh2 = Real{1e-30};

    const int *direct_work = nullptr;       // [n_work]
    const int *list1_flat = nullptr;        // [n_boxes * nlist1_stride]
    const int *list1_count = nullptr;       // [n_boxes]
    const int *box_levels = nullptr;        // [n_boxes]
    const unsigned char *ifpwexp = nullptr; // [n_boxes] (0/1)

    // Per-level direct-eval params (precomputed by tree)
    const Real *direct_rsc = nullptr;   // [n_levels]
    const Real *direct_cen = nullptr;   // [n_levels]
    const Real *direct_d2max = nullptr; // [n_levels]

    // Source data with halo
    const Real *r_src_halo_flat = nullptr;
    const long *r_src_halo_offsets = nullptr;
    const int *src_counts_halo = nullptr;

    const Real *charge_halo_flat = nullptr;
    const long *charge_halo_offsets = nullptr;

    const Real *normal_halo_flat = nullptr;    // null when normal_dim == 0
    const long *normal_halo_offsets = nullptr; // null when normal_dim == 0

    // Target points (one of: r_src_owned (side=src), r_trg_owned (side=trg))
    const Real *r_target_flat = nullptr;
    const long *r_target_offsets = nullptr;
    const int *target_counts = nullptr;

    // Output buffer (one of: pot_src_sorted, pot_trg_sorted)
    Real *pot_flat = nullptr;
    const long *pot_offsets = nullptr;
};

template <typename Real>
void launch_direct_by_box_dispatch(dmk_ikernel kernel, int dim, int n_digits, const DirectByBoxArgs<Real> &args,
                                   cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_DIRECT_KERNELS_HPP
