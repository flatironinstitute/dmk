#ifndef DMK_CUDA_DIRECT_KERNELS_CUH
#define DMK_CUDA_DIRECT_KERNELS_CUH

// Per-box direct-residual kernel. One CUDA block per target box; threads
// stride over target points; the source-box loop happens *inside* the kernel
// so each block owns its slice of pot output (no cross-block race).
//
// One launch handles a single output side: either pot_src (target points =
// owned sources of the trg_box) or pot_trg (target points = owned targets).
// cuda_direct.cpp issues two launches per direct evaluation.
//
// Tree-derived runtime parameters (rsc/cen/d2max per level, ifpwexp, list1)
// are uploaded once and consumed via the DirectByBoxArgs<Real> bundle.
//
// Simplifications relative to the CPU loop:
//   * no ContactGeometry filtering
//   * no PBC support (caller must reject use_periodic before getting here)
//   * no Yukawa support (no CUDA Yukawa evaluator yet)
//
// Performance simplifications (deliberate, will revisit):
//   * no shared-memory tiling — each thread re-reads source data per pair.
//   * load is uneven: blocks with bigger trg boxes / more list1 source pairs
//     dominate runtime.

#include <dmk/cuda_direct_kernels.hpp>
#include <dmk/cuda_kernels.cuh>

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Evaluator>
__global__ void DirectResidualByBoxKernel(DirectByBoxArgs<typename Evaluator::scalar_type> a) {
    using Real = typename Evaluator::scalar_type;
    constexpr int SPATIAL_DIM = Evaluator::SPATIAL_DIM;
    constexpr int KERNEL_INPUT_DIM = Evaluator::KERNEL_INPUT_DIM;
    constexpr int KERNEL_OUTPUT_DIM = Evaluator::KERNEL_OUTPUT_DIM;
    constexpr int NORMAL_DIM = Evaluator::NORMAL_DIM;
    constexpr Real scale_factor = Evaluator::scale_factor;

    const int trg_box_idx = blockIdx.x;
    if (trg_box_idx >= a.n_work)
        return;
    const int trg_box = a.direct_work[trg_box_idx];
    const int n_targets = a.target_counts[trg_box];
    if (n_targets == 0)
        return;

    const int trg_level = a.box_levels[trg_box];
    const int n_list1 = a.list1_count[trg_box];

    const Real *r_targets = a.r_target_flat + a.r_target_offsets[trg_box];
    Real *pot_targets = a.pot_flat + a.pot_offsets[trg_box];

    // Stride loop over the target box's points.
    for (int t = threadIdx.x; t < n_targets; t += blockDim.x) {
        Real xt[SPATIAL_DIM];
        for (int k = 0; k < SPATIAL_DIM; ++k)
            xt[k] = r_targets[t * SPATIAL_DIM + k];

        Real vt[KERNEL_OUTPUT_DIM];
        for (int k = 0; k < KERNEL_OUTPUT_DIM; ++k)
            vt[k] = Real{0};

        for (int li = 0; li < n_list1; ++li) {
            const int src_box = a.list1_flat[trg_box * a.nlist1_stride + li];

            // Mirrors the CPU loop's src_level adjustment.
            int src_level = a.box_levels[src_box];
            if (a.ifpwexp[src_box] && src_box == trg_box)
                src_level = src_level + 1;
            else if (src_level < trg_level)
                src_level = trg_level;
            if (src_level >= a.n_levels)
                src_level = a.n_levels - 1;

            const int n_src = a.src_counts_halo[src_box];
            if (n_src == 0)
                continue;

            const Real rsc = a.direct_rsc[src_level];
            const Real cen = a.direct_cen[src_level];
            const Real d2max = a.direct_d2max[src_level];

            const Real *r_src = a.r_src_halo_flat + a.r_src_halo_offsets[src_box];
            const Real *charge = a.charge_halo_flat + a.charge_halo_offsets[src_box];
            const Real *normals =
                (NORMAL_DIM > 0) ? (a.normal_halo_flat + a.normal_halo_offsets[src_box]) : nullptr;

            Evaluator evaluator{a.thresh2, d2max, rsc, cen};

            for (int s = 0; s < n_src; ++s) {
                Real xs[SPATIAL_DIM];
                for (int k = 0; k < SPATIAL_DIM; ++k)
                    xs[k] = r_src[s * SPATIAL_DIM + k];

                Real dX[SPATIAL_DIM];
                for (int k = 0; k < SPATIAL_DIM; ++k)
                    dX[k] = xt[k] - xs[k];

                Real vs[KERNEL_INPUT_DIM];
                for (int k = 0; k < KERNEL_INPUT_DIM; ++k)
                    vs[k] = charge[s * KERNEL_INPUT_DIM + k];

                Real U[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM];
                if constexpr (NORMAL_DIM > 0) {
                    Real ns[NORMAL_DIM];
                    for (int k = 0; k < NORMAL_DIM; ++k)
                        ns[k] = normals[s * NORMAL_DIM + k];
                    evaluator(U, dX, ns);
                } else {
                    evaluator(U, dX);
                }

                for (int k0 = 0; k0 < KERNEL_INPUT_DIM; ++k0)
                    for (int k1 = 0; k1 < KERNEL_OUTPUT_DIM; ++k1)
                        vt[k1] += U[k0][k1] * vs[k0];
            }
        }

        for (int k = 0; k < KERNEL_OUTPUT_DIM; ++k)
            pot_targets[t * KERNEL_OUTPUT_DIM + k] += vt[k] * scale_factor;
    }
}

template <typename Evaluator, typename Real = typename Evaluator::scalar_type>
inline void launch_direct_by_box(const DirectByBoxArgs<Real> &args, cudaStream_t stream = 0) {
    if (args.n_work == 0)
        return;
    constexpr int block_size = 128;
    DirectResidualByBoxKernel<Evaluator><<<args.n_work, block_size, 0, stream>>>(args);
}

} // namespace dmk::cuda

#endif // DMK_CUDA_DIRECT_KERNELS_CUH
