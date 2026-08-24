#include "support.hpp"
#include "sort.hpp"
#include "state.hpp"

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "../pt/launchers.hpp"

#include <cstdio>
#include <memory>
#include <string>

namespace dmk::cuda::esp {
namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

constexpr int kSupportBlockSize = 256;

// One module per stage. No coefficients are baked, so (stage, real, sm) names the module completely.
template <typename Real>
const jit::JitKernel &support_kernel(int stage) {
    static JitCache cache;
    const std::string kernel_name = "EspSupportKernel_s" + std::to_string(stage);

    JitKey key;
    key.name = kernel_name;
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params["STAGE"] = stage;
    key.params["BLOCK_SIZE"] = kSupportBlockSize;

    // JitCache owns the kernel and outlives every launch, so returning a reference is safe.
    return *cache.get_kernel_from_source(key, [&] {
        const std::string prelude = "#define DMK_ESP_SUPPORT_KERNEL_NAME " + kernel_name + "\n\n";
        return pt::make_stage_source("esp/support.cu", key, prelude, "EspSupport");
    });
}

} // namespace

template <typename Real>
void launch_stage(int stage, int n_elems, EspSupportArgs<Real, ComplexT<Real>> &a, cudaStream_t stream) {
    if (n_elems <= 0)
        return;
    const int blocks = (n_elems + kSupportBlockSize - 1) / kSupportBlockSize;
    support_kernel<Real>(stage).launch(dim3(blocks, 1, 1), dim3(kSupportBlockSize, 1, 1), 0, stream, a);
}

// Sorts particles into cubic cells (CSR cell_start), via sort_by_key + lower_bound rather than the
// CPU's counting sort.
template <typename Real>
void build_cell_list_gpu(GpuState &gpu, int n, int nc, const Real *d_pos_aos, const Real *d_charges,
                         int **d_cell_start_out, int **d_orig_out, Real **d_xs_out, Real **d_ys_out, Real **d_zs_out,
                         Real **d_qs_out) {
    const int ncells = nc * nc * nc;

    EspSupportArgs<Real, ComplexT<Real>> a;
    a.n = n;
    a.nc = nc;
    a.L = Real(gpu.L);
    a.pos_aos = d_pos_aos;
    a.charges = d_charges;

    int *d_orig = nullptr;

    if (gpu.sort_mode == GpuSortMode::Morton) {
        // The 8-byte keys are dead after the sort; d_orig is the real output. Wider keys mean this
        // path sizes the shared scratch differently than Bins.
        ensure_capacity(gpu.d_scratch_idx, gpu.scratch_idx_cap,
                        std::size_t(n) * sizeof(unsigned long long) + std::size_t(n) * sizeof(int));
        unsigned long long *d_cell_idx = reinterpret_cast<unsigned long long *>(gpu.d_scratch_idx);
        d_orig = reinterpret_cast<int *>(d_cell_idx + n);
        a.cell_idx64 = d_cell_idx;

        launch_stage<Real>(kEspStageCellIndexMorton, n, a, gpu.stream);

        sort_cell_keys(d_cell_idx, d_orig, n, ncells, kMortonBuckets, gpu.d_cell_start, gpu.stream);
    } else {
        // Both int, so key and permutation share one 2n-int scratch.
        ensure_capacity(gpu.d_scratch_idx, gpu.scratch_idx_cap, 2 * std::size_t(n) * sizeof(int));
        int *d_cell_idx = reinterpret_cast<int *>(gpu.d_scratch_idx);
        d_orig = d_cell_idx + n;
        a.cell_idx = d_cell_idx;

        launch_stage<Real>(kEspStageCellIndex, n, a, gpu.stream);

        sort_cell_keys(d_cell_idx, d_orig, n, ncells, kEspNbuckets, gpu.d_cell_start, gpu.stream);
    }

    // Always used together downstream, so one 4n-Real scratch.
    ensure_capacity(gpu.d_scratch_sorted, gpu.scratch_sorted_cap, 4 * std::size_t(n) * sizeof(Real));
    Real *d_xs = reinterpret_cast<Real *>(gpu.d_scratch_sorted);
    Real *d_ys = d_xs + n;
    Real *d_zs = d_ys + n;
    Real *d_qs = d_zs + n;

    a.orig = d_orig;
    a.xs = d_xs;
    a.ys = d_ys;
    a.zs = d_zs;
    a.qs = d_qs;
    launch_stage<Real>(kEspStageGatherSorted, n, a, gpu.stream);

    *d_cell_start_out = gpu.d_cell_start;
    *d_orig_out = d_orig;
    *d_xs_out = d_xs;
    *d_ys_out = d_ys;
    *d_zs_out = d_zs;
    *d_qs_out = d_qs;
}

// Un-permutes the cell-sorted accumulator onto the caller's arrays. The gradient-to-force sign
// conversion (-q*grad, using the target's own charge) happens here.
template <typename Real>
void scatter_gpu(GpuState &gpu, int n, int out_dim, const int *d_orig, const Real *d_qs_sorted, const Real *d_pg_sorted,
                 Real *d_pot, Real *d_fx, Real *d_fy, Real *d_fz) {
    EspSupportArgs<Real, ComplexT<Real>> a;
    a.n = n;
    a.out_dim = out_dim;
    a.orig = d_orig;
    a.qs_sorted = d_qs_sorted;
    a.pg_sorted = d_pg_sorted;
    a.pot = d_pot;
    a.fx = d_fx;
    a.fy = d_fy;
    a.fz = d_fz;
    launch_stage<Real>(kEspStageScatter, n, a, gpu.stream);
}

// Opt-in via DMK_ESP_PRUNE_STATS: costs a stream sync per call plus in-loop atomics.
void report_prune_stats(GpuState &gpu, const unsigned long long *d_prune_stats) {
    unsigned long long h[4] = {0, 0, 0, 0};
    cudaMemcpyAsync(h, d_prune_stats, sizeof(h), cudaMemcpyDeviceToHost, gpu.stream);
    cudaStreamSynchronize(gpu.stream);
    if (gpu.strategy == GpuSrStrategy::PruneTile && h[0])
        std::fprintf(stderr, "# esp prune diag: tiles tested=%llu evaluated=%llu (%.1f%% skipped)\n", h[0], h[1],
                     100.0 * double(h[0] - h[1]) / double(h[0]));
    if (gpu.strategy == GpuSrStrategy::PruneSource && h[2])
        std::fprintf(stderr, "# esp prune diag: points tested=%llu evaluated=%llu (%.1f%% skipped)\n", h[2], h[3],
                     100.0 * double(h[2] - h[3]) / double(h[2]));
}

#define DMK_ESP_SUPPORT_INST(Real)                                                                                     \
    template void launch_stage<Real>(int, int, EspSupportArgs<Real, ComplexT<Real>> &, cudaStream_t);                  \
    template void build_cell_list_gpu<Real>(GpuState &, int, int, const Real *, const Real *, int **, int **, Real **, \
                                            Real **, Real **, Real **);                                                \
    template void scatter_gpu<Real>(GpuState &, int, int, const int *, const Real *, const Real *, Real *, Real *,     \
                                    Real *, Real *)

DMK_ESP_SUPPORT_INST(float);
DMK_ESP_SUPPORT_INST(double);
#undef DMK_ESP_SUPPORT_INST

} // namespace dmk::cuda::esp
