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
    kEspStageProject = 4,
    kEspStageNormalize = 5,
    kEspStageExtractReal = 6,
    kEspStageAccumForce = 7,
    kEspStageSelfInteraction = 8,
    kEspStageAddConst = 9,
    kEspStageScalePack = 10,
};

// Per-mode far-field operator, baked into the project stage's module (PROJECTOR).
enum : int {
    kEspProjScalar = 0,
    kEspProjDipole = 1,
    kEspProjStokeslet = 2,
    kEspProjStresslet = 3,
};

int esp_projector_for(dmk_ikernel kernel);

template <typename Real>
void launch_stage(int stage, int n_elems, EspSupportArgs<Real, ComplexT<Real>> &a, cudaStream_t stream,
                  int projector = kEspProjScalar);

// Builds the cell list into GpuState-owned scratch: CSR cell_start, the cell-sorted xs/ys/zs/qs,
// and the orig permutation.
template <typename Real>
void build_cell_list_gpu(GpuState &gpu, int n, int nc, int charge_dim, const Real *d_pos_aos, const Real *d_charges,
                         int **d_cell_start, int **d_orig, Real **d_xs, Real **d_ys, Real **d_zs, Real **d_qs);

// Un-permutes the cell-sorted accumulator onto the caller's arrays.
template <typename Real>
void scatter_gpu(GpuState &gpu, int n, int out_dim, bool grad_is_force, const int *d_orig, const Real *d_qs_sorted,
                 const Real *d_pg_sorted, Real *d_pot, Real *d_fx, Real *d_fy, Real *d_fz);

void report_prune_stats(GpuState &gpu, const unsigned long long *d_prune_stats);

} // namespace dmk::cuda::esp
