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

#include <dmk/cuda/direct_kernels.hpp>
#include <dmk/cuda/kernels.cuh>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

template <typename Evaluator, int SRC_TILE>
__global__ void DirectResidualByBoxKernelTiled(
    DirectByBoxArgs<typename Evaluator::scalar_type> a
) {
    using Real = typename Evaluator::scalar_type;

    constexpr int SPATIAL_DIM       = Evaluator::SPATIAL_DIM;
    constexpr int KERNEL_INPUT_DIM  = Evaluator::KERNEL_INPUT_DIM;
    constexpr int KERNEL_OUTPUT_DIM = Evaluator::KERNEL_OUTPUT_DIM;
    constexpr int NORMAL_DIM        = Evaluator::NORMAL_DIM;
    constexpr Real scale_factor     = Evaluator::scale_factor;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    Real *smem = reinterpret_cast<Real *>(smem_raw);

    Real *s_r_src = smem;
    smem += SRC_TILE * SPATIAL_DIM;

    Real *s_charge = smem;
    smem += SRC_TILE * KERNEL_INPUT_DIM;

    Real *s_normal = nullptr;
    if constexpr (NORMAL_DIM > 0) {
        s_normal = smem;
        smem += SRC_TILE * NORMAL_DIM;
    }

    const int trg_box_idx = blockIdx.x;
    if (trg_box_idx >= a.n_work)
        return;

    const int trg_box = a.direct_work[trg_box_idx];
    const int n_targets = a.target_counts[trg_box];
    if (n_targets == 0)
        return;

    const int trg_level = a.box_levels[trg_box];
    const int n_list1 = a.list1_count[trg_box];

    const Real *__restrict__ r_targets = a.r_target_flat + a.r_target_offsets[trg_box];

    Real *__restrict__ pot_targets = a.pot_flat + a.pot_offsets[trg_box];

    const int n_target_rounds = (n_targets + blockDim.x - 1) / blockDim.x;

    for (int tr = 0; tr < n_target_rounds; ++tr) {
        const int t = tr * blockDim.x + threadIdx.x;
        const bool active_target = (t < n_targets);

        Real xt[SPATIAL_DIM];

        if (active_target) {
            #pragma unroll
            for (int k = 0; k < SPATIAL_DIM; ++k) {
                xt[k] = r_targets[t * SPATIAL_DIM + k];
            }
        }

        Real vt[KERNEL_OUTPUT_DIM];

        #pragma unroll
        for (int k = 0; k < KERNEL_OUTPUT_DIM; ++k) {
            vt[k] = Real{0};
        }

        for (int li = 0; li < n_list1; ++li) {
            const int src_box =
                a.list1_flat[trg_box * a.nlist1_stride + li];

            int src_level = a.box_levels[src_box];

            if (a.ifpwexp[src_box] && src_box == trg_box) {
                src_level = src_level + 1;
            } else if (src_level < trg_level) {
                src_level = trg_level;
            }

            if (src_level >= a.n_levels) {
                src_level = a.n_levels - 1;
            }

            const int n_src = a.src_counts_halo[src_box];

            const Real *__restrict__ r_src = a.r_src_halo_flat + a.r_src_halo_offsets[src_box];

            const Real *__restrict__ charge = a.charge_halo_flat + a.charge_halo_offsets[src_box];

            const Real *__restrict__ normals = nullptr;
            if constexpr (NORMAL_DIM > 0) {
                normals = a.normal_halo_flat + a.normal_halo_offsets[src_box];
            }

            const Real rsc   = a.direct_rsc[src_level];
            const Real cen   = a.direct_cen[src_level];
            const Real d2max = a.direct_d2max[src_level];

            Evaluator evaluator{a.thresh2, d2max, rsc, cen};

            for (int tile0 = 0; tile0 < n_src; tile0 += SRC_TILE) {
                const int rem = n_src - tile0;
                const int tile_count = (rem < SRC_TILE) ? rem : SRC_TILE;

                for (int idx = threadIdx.x; idx < tile_count * SPATIAL_DIM; idx += blockDim.x) {
                    const int s = idx / SPATIAL_DIM;
                    const int k = idx - s * SPATIAL_DIM;

                    s_r_src[s * SPATIAL_DIM + k] = r_src[(tile0 + s) * SPATIAL_DIM + k];
                }

                for (int idx = threadIdx.x; idx < tile_count * KERNEL_INPUT_DIM; idx += blockDim.x) {
                    const int s = idx / KERNEL_INPUT_DIM;
                    const int k = idx - s * KERNEL_INPUT_DIM;

                    s_charge[s * KERNEL_INPUT_DIM + k] = charge[(tile0 + s) * KERNEL_INPUT_DIM + k];
                }

                if constexpr (NORMAL_DIM > 0) {
                    for (int idx = threadIdx.x; idx < tile_count * NORMAL_DIM; idx += blockDim.x) {
                        const int s = idx / NORMAL_DIM;
                        const int k = idx - s * NORMAL_DIM;

                        s_normal[s * NORMAL_DIM + k] = normals[(tile0 + s) * NORMAL_DIM + k];
                    }
                }

                __syncthreads();

                if (active_target) {
                    #pragma unroll
                    for (int ss = 0; ss < tile_count; ++ss) {
                        Real xs[SPATIAL_DIM];

                        #pragma unroll
                        for (int k = 0; k < SPATIAL_DIM; ++k) {
                            xs[k] = s_r_src[ss * SPATIAL_DIM + k];
                        }

                        Real dX[SPATIAL_DIM];

                        #pragma unroll
                        for (int k = 0; k < SPATIAL_DIM; ++k) {
                            dX[k] = xt[k] - xs[k];
                        }

                        Real vs[KERNEL_INPUT_DIM];

                        #pragma unroll
                        for (int k = 0; k < KERNEL_INPUT_DIM; ++k) {
                            vs[k] = s_charge[ss * KERNEL_INPUT_DIM + k];
                        }

                        Real U[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM];

                        if constexpr (NORMAL_DIM > 0) {
                            Real ns[NORMAL_DIM];

                            #pragma unroll
                            for (int k = 0; k < NORMAL_DIM; ++k) {
                                ns[k] = s_normal[ss * NORMAL_DIM + k];
                            }

                            evaluator(U, dX, ns);
                        } else {
                            evaluator(U, dX);
                        }

                        #pragma unroll
                        for (int k0 = 0; k0 < KERNEL_INPUT_DIM; ++k0) {
                            #pragma unroll
                            for (int k1 = 0; k1 < KERNEL_OUTPUT_DIM; ++k1) {
                                vt[k1] += U[k0][k1] * vs[k0];
                            }
                        }
                    }
                }

                __syncthreads();
            }
        }

        if (active_target) {
            #pragma unroll
            for (int k = 0; k < KERNEL_OUTPUT_DIM; ++k) {
                pot_targets[t * KERNEL_OUTPUT_DIM + k] +=
                    vt[k] * scale_factor;
            }
        }
    }
}

template <typename Evaluator, typename Real = typename Evaluator::scalar_type, int SRC_TILE = 32>
inline void launch_direct_by_box(
    const DirectByBoxArgs<Real> &args,
    cudaStream_t stream = 0
) {
    if (args.n_work == 0)
        return;

    constexpr int block_size = 256;

    constexpr int SPATIAL_DIM = Evaluator::SPATIAL_DIM;

    constexpr int KERNEL_INPUT_DIM = Evaluator::KERNEL_INPUT_DIM;

    constexpr int NORMAL_DIM = Evaluator::NORMAL_DIM;

    constexpr int values_per_source = SPATIAL_DIM + KERNEL_INPUT_DIM + NORMAL_DIM;

    const std::size_t shared_bytes = static_cast<std::size_t>(SRC_TILE) * values_per_source * sizeof(Real);

    DirectResidualByBoxKernelTiled<Evaluator, SRC_TILE> <<<args.n_work, block_size, shared_bytes, stream>>>(args);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("launch_direct_by_box_tiled: ") +
            cudaGetErrorString(err) +
            " shared_bytes=" + std::to_string(shared_bytes)
        );
    }
}

} // namespace dmk::cuda

#endif // DMK_CUDA_DIRECT_KERNELS_CUH
