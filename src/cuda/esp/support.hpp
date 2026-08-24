#pragma once
// Launchers for the ESP GPU support kernels (esp/support.cu).

#include "state.hpp"

#include <dmk/cuda/esp_support_kernelargs.hpp>

namespace dmk::cuda::esp {

// Must match the dispatch in esp/support.cu.
enum : int {
    kEspStageCellIndex = 0,
    kEspStageCellIndexMorton = 1,
    kEspStageGatherSorted = 2,
    kEspStageScatter = 3,
    kEspStageScaling = 4,
    kEspStageNormalize = 5,
    kEspStageExtractReal = 6,
    kEspStageGradScaling = 7,
    kEspStageAccumForce = 8,
    kEspStageSelfInteraction = 9,
    kEspStageScalePack = 10,
};

template <typename Real>
void launch_stage(int stage, int n_elems, EspSupportArgs<Real, ComplexT<Real>> &a, cudaStream_t stream);

// Builds the cell list into GpuState-owned scratch: CSR cell_start, the cell-sorted xs/ys/zs/qs,
// and the orig permutation.
template <typename Real>
void build_cell_list_gpu(GpuState &gpu, int n, int nc, const Real *d_pos_aos, const Real *d_charges, int **d_cell_start,
                         int **d_orig, Real **d_xs, Real **d_ys, Real **d_zs, Real **d_qs);

// Un-permutes the cell-sorted accumulator onto the caller's arrays.
template <typename Real>
void scatter_gpu(GpuState &gpu, int n, int out_dim, const int *d_orig, const Real *d_qs_sorted, const Real *d_pg_sorted,
                 Real *d_pot, Real *d_fx, Real *d_fy, Real *d_fz);

void report_prune_stats(GpuState &gpu, const unsigned long long *d_prune_stats);

} // namespace dmk::cuda::esp
