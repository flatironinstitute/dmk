// GPU implementation of the ESP solver.

#include <dmk/cuda/esp_gpu.hpp>
#include <dmk/cuda/esp_sr_coeffs.cuh>
#include <dmk/esp.hpp>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufinufft.h>
#include <nvtx3/nvToolsExt.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace dmk {

// ---------------------------------------------------------------------------
// NvtxRange — RAII wrapper so every pushed range pops even if the block throws
// (several stages below throw on CUDA/cuFINUFFT errors) or returns early.
// Purely a profiling aid for nsys/ncu -- push/pop are no-ops without a profiler
// attached, so this has no effect on normal runs.
// ---------------------------------------------------------------------------
struct NvtxRange {
    explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
};

// ---------------------------------------------------------------------------
// ComplexT<Real> — the cuFFT/cuFINUFFT complex type matching Real, so the
// long-range pipeline (spread/FFT/interp) can be genuinely Real-templated
// instead of always running in double. A GpuState is created for exactly one
// Real (see GpuState::use_float / gpu_create_state) -- this alias is what lets
// the same long_range_gpu<Real> body compile against either library's API.
// ---------------------------------------------------------------------------
template <typename Real>
using ComplexT = std::conditional_t<std::is_same_v<Real, double>, cuDoubleComplex, cuFloatComplex>;

// ---------------------------------------------------------------------------
// GpuState — owns all physics params and CUDA objects for one GPU plan.
// ---------------------------------------------------------------------------
struct GpuState {
    // Physics params — set once at esp_create_gpu_plan time, read at every eval.
    int           nf;
    int           n_digits;
    double        L, r_c, gpu_upsampfac, tol;
    double        self_factor_d;
    float         self_factor_f;
    dmk_eval_type eval_type;
    // The one Real this plan was created for (esp_create_gpu_plan's use_float arg).
    // Every eval on this plan must be called with the matching Real -- checked at
    // esp_eval_gpu_impl/short_range_impl/long_range_impl's entry via check_plan_real<Real>.
    bool          use_float = false;
    // Short-range strategy (see GpuSrStrategy in esp.hpp). Fixed at plan-creation
    // time (esp_create_gpu_plan's strategy arg), read in short_range_gpu.
    GpuSrStrategy strategy = GpuSrStrategy::Dense;
    // Within-cell sort (see GpuSortMode in esp.hpp), independent of strategy.
    // Fixed at plan-creation time, read in build_cell_list_gpu.
    GpuSortMode   sort_mode = GpuSortMode::Bins;
    // Cache for short_range_kernel_pruned/short_range_kernel_pruned_source's
    // shared-memory sizing (see compute_max_cell_pop / short_range_gpu):
    // computing the exact max cell population requires a host-device sync
    // every call, so it's only recomputed when n changes -- reused as-is
    // across repeated calls with the same n (the common case: same plan,
    // same particle count, e.g. a benchmark loop or successive timesteps of
    // one simulation).
    int           pruned_max_tiles_cache   = 0;
    int           pruned_max_tiles_cache_n = -1;

    // Host output workspace; grown as needed, never shrunk between calls.
    std::vector<double> h_dbl_buf;
    std::vector<float>  h_flt_buf;

    // Plan-level device data — uploaded once at esp_create_gpu_plan time. void*
    // because the concrete type (Real, or ComplexT<Real>, or cufinufft_plan vs
    // cufinufftf_plan -- both just opaque pointers) depends on use_float; cast at
    // each point of use via reinterpret_cast<...>(this).
    cudaStream_t   stream            = nullptr;
    void          *d_scaling_coeffs  = nullptr;   // nf³ Real
    int            nc                = 0;         // cells per dimension, = floor(L/r_c)
    int           *d_nbc_tab         = nullptr;   // nc*3 ints — neighbor cell index per (cell,delta)
    void          *d_off_tab         = nullptr;   // nc*3 Real — periodic image shift per (cell,delta)
    void          *d_b               = nullptr;   // nf³ ComplexT<Real> — spread output (NU → uniform)
    void          *d_b_hat           = nullptr;   // nf³ ComplexT<Real> — FFT of d_b (k-space)
    cufftHandle    fft_plan{};                     // nf³ 3-D c2c (Z2Z or C2C), created at plan time
    bool           fft_plan_valid    = false;
    void          *cfnufft_plan_1    = nullptr;    // cufinufft_plan or cufinufftf_plan — type-1 (NU → uniform)
    void          *cfnufft_plan_2    = nullptr;    // cufinufft_plan or cufinufftf_plan — type-2 (uniform → NU)
    int             *d_cell_start    = nullptr;    // ncells+1 ints (ncells=nc³) — short-range cell list
    void           *d_fhat_x         = nullptr;    // nf³ ComplexT<Real> — force spectra, force path only
    void           *d_fhat_y         = nullptr;
    void           *d_fhat_z         = nullptr;

    // Per-eval device scratch — grown as needed (byte capacity), never shrunk, reused across
    // calls instead of malloc/free'd every eval. Cast to the needed type at each call site
    // (Real can be float or double); roles always used together within one eval (e.g.
    // pot/fx/fy/fz, or xs/ys/zs/qs) share one buffer via sub-offsets, mirroring how
    // gpu_make_spans partitions the host output buffer.
    void *d_scratch_pos    = nullptr; size_t scratch_pos_cap    = 0; // pos_aos (3n) + charges (n), Real
    void *d_scratch_out    = nullptr; size_t scratch_out_cap    = 0; // pot/fx/fy/fz (4n), Real (outputs)
    void *d_scratch_idx    = nullptr; size_t scratch_idx_cap    = 0; // cell_idx (n) + orig (n), int
    void *d_scratch_sorted = nullptr; size_t scratch_sorted_cap = 0; // xs/ys/zs/qs (4n), Real
    void *d_scratch_pg     = nullptr; size_t scratch_pg_cap     = 0; // pg_sorted (out_dim*n), Real
    void *d_scratch_lr_xyz = nullptr; size_t scratch_lr_xyz_cap = 0; // x/y/z (3n), Real
    void *d_scratch_lr_c   = nullptr; size_t scratch_lr_c_cap   = 0; // packed charges (n), ComplexT<Real>
    void *d_scratch_nu_c   = nullptr; size_t scratch_nu_c_cap   = 0; // pot_c / force_c (n), ComplexT<Real>

    GpuState()  = default;
    ~GpuState() {
        if (cfnufft_plan_1) {
            if (use_float) cufinufftf_destroy(reinterpret_cast<cufinufftf_plan>(cfnufft_plan_1));
            else           cufinufft_destroy(reinterpret_cast<cufinufft_plan>(cfnufft_plan_1));
        }
        if (cfnufft_plan_2) {
            if (use_float) cufinufftf_destroy(reinterpret_cast<cufinufftf_plan>(cfnufft_plan_2));
            else           cufinufft_destroy(reinterpret_cast<cufinufft_plan>(cfnufft_plan_2));
        }
        if (fft_plan_valid)   cufftDestroy(fft_plan);
        if (d_b_hat)          cudaFree(d_b_hat);
        if (d_b)              cudaFree(d_b);
        if (d_scaling_coeffs)  cudaFree(d_scaling_coeffs);
        if (d_nbc_tab)         cudaFree(d_nbc_tab);
        if (d_off_tab)         cudaFree(d_off_tab);
        if (d_cell_start)      cudaFree(d_cell_start);
        if (d_fhat_x)          cudaFree(d_fhat_x);
        if (d_fhat_y)          cudaFree(d_fhat_y);
        if (d_fhat_z)          cudaFree(d_fhat_z);
        if (d_scratch_pos)     cudaFree(d_scratch_pos);
        if (d_scratch_out)     cudaFree(d_scratch_out);
        if (d_scratch_idx)     cudaFree(d_scratch_idx);
        if (d_scratch_sorted)  cudaFree(d_scratch_sorted);
        if (d_scratch_pg)      cudaFree(d_scratch_pg);
        if (d_scratch_lr_xyz)  cudaFree(d_scratch_lr_xyz);
        if (d_scratch_lr_c)    cudaFree(d_scratch_lr_c);
        if (d_scratch_nu_c)    cudaFree(d_scratch_nu_c);
        if (stream)            cudaStreamDestroy(stream);
    }
};

// Grows *ptr (byte capacity *cap) to at least needed_bytes, freeing+reallocating only when it
// actually needs to grow. The core of turning "malloc/free every eval" into "malloc once,
// reuse forever" for a fixed (or non-growing) problem size.
static void ensure_capacity(void *&ptr, size_t &cap, size_t needed_bytes) {
    if (cap >= needed_bytes) return;
    if (ptr) cudaFree(ptr);
    if (cudaMalloc(&ptr, needed_bytes) != cudaSuccess)
        throw std::runtime_error("ensure_capacity: cudaMalloc failed");
    cap = needed_bytes;
}

// Allocate and initialise a GpuState with all physics params and CUDA objects.
// use_float selects the ONE Real this plan is created for -- every eval call
// on the returned GpuState must use the matching Real (esp_eval_gpu<float>
// on a use_float=false plan throws). h_scaling_coeffs is always double (it's
// computed CPU-side in esp.cpp); it's narrowed to float here, once, if needed.
GpuState *gpu_create_state(
    int nf, int n_digits,
    double L, double r_c, double gpu_upsampfac, double tol,
    double self_factor_d, float self_factor_f,
    dmk_eval_type eval_type, bool use_float, GpuSrStrategy strategy, GpuSortMode sort_mode,
    const double *h_scaling_coeffs)
{
    auto *gpu = new GpuState;
    gpu->nf            = nf;
    gpu->n_digits      = n_digits;
    gpu->L             = L;
    gpu->r_c           = r_c;
    gpu->gpu_upsampfac = gpu_upsampfac;
    gpu->tol           = tol;
    gpu->self_factor_d = self_factor_d;
    gpu->self_factor_f = self_factor_f;
    gpu->eval_type     = eval_type;
    gpu->use_float     = use_float;
    gpu->strategy      = strategy;
    gpu->sort_mode     = sort_mode;

    const size_t real_sz    = use_float ? sizeof(float) : sizeof(double);
    const size_t complex_sz = use_float ? sizeof(cuFloatComplex) : sizeof(cuDoubleComplex);

    // Dedicated stream for all GPU work on this plan.
    if (cudaStreamCreate(&gpu->stream) != cudaSuccess)
        throw std::runtime_error("GpuState: cudaStreamCreate failed");

    // Upload precomputed scaling coefficients (plan-level constant, nf³ Real).
    const long long ntot = (long long)nf * nf * nf;
    if (cudaMalloc(&gpu->d_scaling_coeffs, ntot * real_sz) != cudaSuccess)
        throw std::runtime_error("GpuState: cudaMalloc d_scaling_coeffs failed");
    if (use_float) {
        std::vector<float> h_scaling_coeffs_f(ntot);
        for (long long i = 0; i < ntot; ++i) h_scaling_coeffs_f[i] = float(h_scaling_coeffs[i]);
        cudaMemcpy(gpu->d_scaling_coeffs, h_scaling_coeffs_f.data(), ntot * real_sz, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpyAsync(gpu->d_scaling_coeffs, h_scaling_coeffs,
                        ntot * real_sz, cudaMemcpyHostToDevice, gpu->stream);
    }

    // Spread output buffer — nf³ ComplexT<Real>, so allocate once at plan time.
    if (cudaMalloc(&gpu->d_b, ntot * complex_sz) != cudaSuccess)
        throw std::runtime_error("GpuState: cudaMalloc d_b failed");

    // Forward 3-D c2c FFT plan: nf × nf × nf.  Created once; reused every eval.
    if (cufftPlan3d(&gpu->fft_plan, nf, nf, nf, use_float ? CUFFT_C2C : CUFFT_Z2Z) != CUFFT_SUCCESS)
        throw std::runtime_error("GpuState: cufftPlan3d failed");
    cufftSetStream(gpu->fft_plan, gpu->stream);
    gpu->fft_plan_valid = true;

    // FFT output buffer: nf³ ComplexT<Real> (k-space).
    if (cudaMalloc(&gpu->d_b_hat, ntot * complex_sz) != cudaSuccess)
        throw std::runtime_error("GpuState: cudaMalloc d_b_hat failed");

    // cuFINUFFT plans — created once per plan (makeplan does not bind to n).
    // Per eval: call setpts (binds NU points) then execute. cufinufft_plan and
    // cufinufftf_plan are both just opaque pointer typedefs (to unrelated struct
    // tags), so either is stored in GpuState's void* fields and reinterpret_cast
    // back to the concrete type wherever it's used, keyed on gpu->use_float.
    cufinufft_opts co;
    cufinufft_default_opts(&co);
    co.gpu_spreadinterponly = 1;
    // gpu_upsampfac is GPU_SPREADER_UPSAMPFAC (esp.cpp) -- deliberately decoupled
    // from the PSWF splitting kernel's sigma (which stays whatever the CPU plan
    // requested, e.g. 1.35, for grid/PSWF consistency). Fixed at the standard 2.0
    // so gpu_kerevalmeth=1 (Horner, faster than the direct exp/sqrt eval
    // non-standard values would require) is valid; precompute_scaling_coefficients_es
    // derives its (ns,beta) from this same constant, so the two stay consistent.
    co.upsampfac            = gpu_upsampfac;
    co.gpu_kerevalmeth      = 1;
    co.gpu_method = 3;
    //co.gpu_sort = 0;  //relevant only if co.gpu_method = 1;
    // Use the default (null) stream for cuFINUFFT internals.
    // We sync explicitly before setpts/execute so the NU data is ready.
    // co.gpu_stream is left at cudaStreamDefault (the default from cufinufft_default_opts).
    const int64_t nmodes[3] = {nf, nf, nf};

    int ier;
    if (use_float) {
        cufinufftf_plan p1 = nullptr, p2 = nullptr;
        ier = cufinufftf_makeplan(/*type=*/1, /*dim=*/3, nmodes,
                                  /*iflag=*/+1, /*ntransf=*/1, float(tol), &p1, &co);
        if (ier != 0) throw std::runtime_error("GpuState: cufinufftf_makeplan type-1 failed, ier=" + std::to_string(ier));
        co.gpu_method = 1;
        ier = cufinufftf_makeplan(/*type=*/2, /*dim=*/3, nmodes,
                                  /*iflag=*/-1, /*ntransf=*/1, float(tol), &p2, &co);
        if (ier != 0) throw std::runtime_error("GpuState: cufinufftf_makeplan type-2 failed, ier=" + std::to_string(ier));
        gpu->cfnufft_plan_1 = p1;
        gpu->cfnufft_plan_2 = p2;
    } else {
        cufinufft_plan p1 = nullptr, p2 = nullptr;
        ier = cufinufft_makeplan(/*type=*/1, /*dim=*/3, nmodes,
                                 /*iflag=*/+1, /*ntransf=*/1, tol, &p1, &co);
        if (ier != 0) throw std::runtime_error("GpuState: cufinufft_makeplan type-1 failed, ier=" + std::to_string(ier));
        co.gpu_method = 1;
        ier = cufinufft_makeplan(/*type=*/2, /*dim=*/3, nmodes,
                                 /*iflag=*/-1, /*ntransf=*/1, tol, &p2, &co);
        if (ier != 0) throw std::runtime_error("GpuState: cufinufft_makeplan type-2 failed, ier=" + std::to_string(ier));
        gpu->cfnufft_plan_1 = p1;
        gpu->cfnufft_plan_2 = p2;
    }

    // Short-range 27-cell-stencil neighbor tables (plan-level constant: depends only
    // on nc = floor(L/r_c), not on particle data). Mirrors esp.cpp short_range()'s
    // nbc_tab/off_tab construction exactly.
    gpu->nc = static_cast<int>(std::floor(L / r_c));
    if (gpu->nc < 3)
        throw std::runtime_error("GpuState: short_range_gpu requires r_c <= L/3 (nc >= 3)");
    {
        const int ntab = gpu->nc * 3;
        std::vector<int>    h_nbc_tab(ntab);
        std::vector<double> h_off_tab(ntab);
        for (int c = 0; c < gpu->nc; ++c) {
            for (int d = 0; d < 3; ++d) {
                int ci = c + d - 1;
                if (ci < 0)              { h_nbc_tab[c * 3 + d] = ci + gpu->nc; h_off_tab[c * 3 + d] = -L; }
                else if (ci >= gpu->nc)  { h_nbc_tab[c * 3 + d] = ci - gpu->nc; h_off_tab[c * 3 + d] = L; }
                else                     { h_nbc_tab[c * 3 + d] = ci;          h_off_tab[c * 3 + d] = 0.0; }
            }
        }
        if (cudaMalloc(&gpu->d_nbc_tab, ntab * sizeof(int)) != cudaSuccess)
            throw std::runtime_error("GpuState: cudaMalloc d_nbc_tab failed");
        if (cudaMalloc(&gpu->d_off_tab, ntab * real_sz) != cudaSuccess)
            throw std::runtime_error("GpuState: cudaMalloc d_off_tab failed");
        // Blocking copy: h_nbc_tab/h_off_tab are locals going out of scope right after,
        // unlike the plan-owned buffers above that justify cudaMemcpyAsync.
        cudaMemcpy(gpu->d_nbc_tab, h_nbc_tab.data(), ntab * sizeof(int), cudaMemcpyHostToDevice);
        if (use_float) {
            std::vector<float> h_off_tab_f(ntab);
            for (int i = 0; i < ntab; ++i) h_off_tab_f[i] = float(h_off_tab[i]);
            cudaMemcpy(gpu->d_off_tab, h_off_tab_f.data(), ntab * real_sz, cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(gpu->d_off_tab, h_off_tab.data(), ntab * real_sz, cudaMemcpyHostToDevice);
        }
    }

    // Short-range cell-list CSR boundaries: size ncells+1 is fixed by nc, so this is plan-level
    // (allocated once here, reused/overwritten every short_range_gpu call).
    {
        const long long ncells = (long long)gpu->nc * gpu->nc * gpu->nc;
        if (cudaMalloc(&gpu->d_cell_start, (ncells + 1) * sizeof(int)) != cudaSuccess)
            throw std::runtime_error("GpuState: cudaMalloc d_cell_start failed");
    }

    // Force spectra (long_range_gpu steps 6-8): size nf³ is plan-level, only needed when the
    // plan actually computes forces.
    if (eval_type == DMK_POTENTIAL_GRAD) {
        if (cudaMalloc(&gpu->d_fhat_x, ntot * complex_sz) != cudaSuccess ||
            cudaMalloc(&gpu->d_fhat_y, ntot * complex_sz) != cudaSuccess ||
            cudaMalloc(&gpu->d_fhat_z, ntot * complex_sz) != cudaSuccess)
            throw std::runtime_error("GpuState: cudaMalloc d_fhat_{x,y,z} failed");
    }

    return gpu;
}

void gpu_destroy_state(GpuState *gpu) { delete gpu; }

// ---------------------------------------------------------------------------
// cell_index_kernel — flat (cell, spatial-bin) composite key per particle.
// Mirrors esp.cpp's particle_cell/cell_linear_index for the cell part, and
// sort_cell_bins for the bin part: within its cell, each particle is further
// classified into one of kEspNbuckets spatial sub-boxes (octants, for the
// default kEspBins=2), so that once sorted by this composite key, particles
// within a cell end up spatially clustered, not just cell-clustered.
//
// This is purely a reordering -- short_range_kernel/short_range_kernel_old
// need no changes at all, since cell_start still demarcates exactly the same
// cell boundaries as before (see build_cell_list_gpu's scaled lower_bound
// search); they just see tighter within-cell tile locality "for free". On
// its own this doesn't reduce any arithmetic (nothing prunes using it yet)
// -- it's the same preparatory-infrastructure role sort_cell_bins plays on
// the CPU side, there specifically enabling short_range_prune_tile /
// short_range_prune_source. kEspBins is a fixed default here, not yet wired
// to a runtime/plan parameter the way esp.cpp's params.esp_bins is.
// ---------------------------------------------------------------------------
constexpr int kEspBins     = 2; // sub-cell bins per axis (octants for DIM=3), matches esp.cpp's default
constexpr int kEspNbuckets = kEspBins * kEspBins * kEspBins;

// Functor (not a lambda) for thrust::make_transform_iterator below -- thrust's device-side
// algorithms need a __host__ __device__-callable functor, and a plain struct avoids depending
// on CUDA extended-lambda support being enabled for this build.
struct MulByEspNbuckets {
    __host__ __device__ int operator()(int c) const { return c * kEspNbuckets; }
};

template <typename Real>
__global__ void cell_index_kernel(const Real *d_pos_aos, int n, Real L, int nc, int *d_cell_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real cell_size = L / Real(nc);

    // Returns the wrapped cell coordinate for axis value x, and writes that axis's
    // sub-cell bin index (0..kEspBins-1) into bin_out.
    auto cell_coord_and_bin = [&](Real x, int &bin_out) {
        const Real u = (x + L / Real(2)) / cell_size; // continuous cell coordinate
        int c = static_cast<int>(floor(u));
        // frac = u mod 1 -- the fraction of the way through this axis's cell. Unaffected
        // by the +-nc wrap below, since that only relabels which physical cell c refers
        // to (by exactly nc), not the particle's position within it.
        Real frac = u - floor(u);
        int b = static_cast<int>(frac * Real(kEspBins));
        b = (b < 0) ? 0 : (b >= kEspBins ? kEspBins - 1 : b);
        bin_out = b;
        c = (c >= nc) ? c - nc : c;
        c = (c < 0)   ? c + nc : c;
        return c;
    };

    int bx, by, bz;
    const int cx = cell_coord_and_bin(d_pos_aos[3 * i + 0], bx);
    const int cy = cell_coord_and_bin(d_pos_aos[3 * i + 1], by);
    const int cz = cell_coord_and_bin(d_pos_aos[3 * i + 2], bz);

    // Composite key: cell_lin*nbuckets + bin_lin. Sorting by this groups by cell first
    // (bin_lin < nbuckets for every cell, so it never crosses a cell boundary), then by
    // spatial bin within each cell -- the same final ordering sort_cell_bins produces via
    // its own separate per-cell counting sort, reached here with one composite-key global
    // sort instead (many small independent per-cell sorts don't map onto GPU parallelism
    // the way a plain OpenMP-over-cells loop does on CPU).
    const int cell_lin = (cx * nc + cy) * nc + cz;
    const int bin_lin  = (bz * kEspBins + by) * kEspBins + bx; // matches sort_cell_bins' key = key*bins + bidx[d], d=DIM-1..0
    d_cell_idx[i] = cell_lin * kEspNbuckets + bin_lin;
}

// ---------------------------------------------------------------------------
// cell_index_kernel_morton — flat (cell, Morton-code) composite key per
// particle. Mirrors CPU's sort_cell_morton (read from the ewald-esp branch),
// but as ONE global composite-key sort instead of many small per-cell radix
// sorts, for the same reason cell_index_kernel above already does the bins
// classification as one global sort: many small independent per-cell sorts
// don't map onto GPU parallelism the way a plain OpenMP-over-cells loop does
// on CPU. Sorting by (cell_index, morton_code) gives the identical relative
// ordering within each cell's range that sorting each cell's particles by
// Morton code independently would.
//
// part1by2_64 is a direct port of the CPU function of the same name (plain
// shift/mask chain) -- CPU also has a byte-chunk lookup-table variant
// (kSpread3) to avoid a dependent shift chain in a tight scalar loop, but
// that's a CPU-scalar micro-optimization with nothing to offer one GPU thread
// per particle, so it's not ported.
//
// kMortonBits = 16/DIM = 5 for DIM=3 (this branch is always 3D), matching
// CPU exactly: each axis is quantized to a 32-level (2^5) grid within the
// particle's own cell, and the 3 axes interleave into a 15-bit code.
// kMortonBuckets = 2^15 is therefore the per-cell key space the Morton code
// ranges over -- the composite key is cell_lin*kMortonBuckets + morton_code,
// analogous to cell_index_kernel's cell_lin*kEspNbuckets + bin_lin, just at
// finer resolution. Unlike that key (safely int-sized, kEspNbuckets=8), this
// one needs 64 bits: cell_lin can be large enough that cell_lin*32768
// overflows int32 for finer r_c / larger domains, so build_cell_list_gpu
// gives this path its own unsigned long long key buffer and 64-bit sort.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t part1by2_64(uint64_t x) {
    x &= 0x1fffffull;
    x = (x | x << 32) & 0x1f00000000ffffull;
    x = (x | x << 16) & 0x1f0000ff0000ffull;
    x = (x | x << 8)  & 0x100f00f00f00f00full;
    x = (x | x << 4)  & 0x10c30c30c30c30c3ull;
    x = (x | x << 2)  & 0x1249249249249249ull;
    return x;
}
__device__ __forceinline__ uint64_t morton3(uint64_t cx, uint64_t cy, uint64_t cz) {
    return (part1by2_64(cx) << 2) | (part1by2_64(cy) << 1) | part1by2_64(cz);
}

constexpr int kMortonBits = 5; // 16/DIM for DIM=3, matches CPU's sort_cell_morton
constexpr unsigned long long kMortonBuckets = 1ull << (3 * kMortonBits); // 32768

// Mirrors MulByEspNbuckets, widened to 64-bit for the Morton composite key.
struct MulByMortonBuckets {
    __host__ __device__ unsigned long long operator()(int c) const {
        return (unsigned long long)c * kMortonBuckets;
    }
};

template <typename Real>
__global__ void cell_index_kernel_morton(const Real *d_pos_aos, int n, Real L, int nc,
                                         unsigned long long *d_cell_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real cell_size = L / Real(nc);

    // Same wrapped-cell-coordinate logic as cell_index_kernel, but returns the
    // continuous within-cell fraction (0..1) instead of a coarse bin index --
    // Morton needs kMortonBits levels of resolution, not just 1.
    auto cell_coord_and_frac = [&](Real x, Real &frac_out) {
        const Real u = (x + L / Real(2)) / cell_size;
        int c = static_cast<int>(floor(u));
        frac_out = u - floor(u);
        c = (c >= nc) ? c - nc : c;
        c = (c < 0)   ? c + nc : c;
        return c;
    };

    Real fx, fy, fz;
    const int cx = cell_coord_and_frac(d_pos_aos[3 * i + 0], fx);
    const int cy = cell_coord_and_frac(d_pos_aos[3 * i + 1], fy);
    const int cz = cell_coord_and_frac(d_pos_aos[3 * i + 2], fz);

    constexpr int kMax = (1 << kMortonBits) - 1;
    auto quantize = [&](Real frac) {
        int q = static_cast<int>(frac * Real(1 << kMortonBits));
        return (uint64_t)((q < 0) ? 0 : (q > kMax ? kMax : q));
    };
    const uint64_t code = morton3(quantize(fx), quantize(fy), quantize(fz));

    const int cell_lin = (cx * nc + cy) * nc + cz;
    d_cell_idx[i] = (unsigned long long)cell_lin * kMortonBuckets + code;
}

// ---------------------------------------------------------------------------
// gather_sorted_kernel — apply the cell-sort permutation to positions/charges.
// ---------------------------------------------------------------------------
template <typename Real>
__global__ void gather_sorted_kernel(
    const Real *d_pos_aos, const Real *d_charges, const int *d_orig, int n,
    Real *d_xs, Real *d_ys, Real *d_zs, Real *d_qs)
{
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n) return;
    const int orig = d_orig[slot];
    d_xs[slot] = d_pos_aos[3 * orig + 0];
    d_ys[slot] = d_pos_aos[3 * orig + 1];
    d_zs[slot] = d_pos_aos[3 * orig + 2];
    d_qs[slot] = d_charges[orig];
}

// ---------------------------------------------------------------------------
// build_cell_list_gpu
// Sorts particles into cubic cells (CSR-style cell_start), matching CPU's
// build_cell_list (esp.cpp) but via a sort_by_key + lower_bound bucketing
// idiom instead of a counting sort.
// ---------------------------------------------------------------------------
template <typename Real>
static void build_cell_list_gpu(
    const Real *d_pos_aos,
    const Real *d_charges,
    int n, int nc, Real L,
    Real **d_xs_out, Real **d_ys_out, Real **d_zs_out,
    Real **d_qs_out,
    int  **d_cell_start_out,
    int  **d_orig_out,
    GpuState &gpu)
{
    const int ncells = nc * nc * nc;
    const auto policy = thrust::cuda::par.on(gpu.stream);

    int *d_orig = nullptr;

    if (gpu.sort_mode == GpuSortMode::Morton) {
        // d_cell_idx (unsigned long long, transient) is dead after the sort below; d_orig
        // is this function's real output, needed by the caller through scatter_kernel. The
        // 8-byte keys give this path a different layout than the Bins path below, so it
        // sizes the same persistent scratch buffer differently rather than sharing offsets.
        ensure_capacity(gpu.d_scratch_idx, gpu.scratch_idx_cap,
                       (size_t)n * sizeof(unsigned long long) + (size_t)n * sizeof(int));
        unsigned long long *d_cell_idx = reinterpret_cast<unsigned long long *>(gpu.d_scratch_idx);
        d_orig = reinterpret_cast<int *>(d_cell_idx + n);

        {
            NvtxRange range("short_range/cell_index_morton");
            const int threads = 256, blocks = (n + threads - 1) / threads;
            cell_index_kernel_morton<Real><<<blocks, threads, 0, gpu.stream>>>(d_pos_aos, n, L, nc, d_cell_idx); //computes each particle's (cell,morton) composite key
        }

        {
            NvtxRange range("short_range/cell_sort_morton");
            thrust::device_ptr<unsigned long long> cell_idx_ptr(d_cell_idx); //keys for sort_by_key (composite cell*kMortonBuckets+morton_code)
            thrust::device_ptr<int> orig_ptr(d_orig);
            thrust::sequence(policy, orig_ptr, orig_ptr + n);
            thrust::sort_by_key(policy, cell_idx_ptr, cell_idx_ptr + n, orig_ptr); //sort particles by composite key; also sort their original indices

            // Same sort+lower_bound bucketing idiom as the Bins path, just with the
            // Morton composite key's own bucket width (MulByMortonBuckets).
            thrust::device_ptr<int> cell_start_ptr(gpu.d_cell_start);
            auto search_begin = thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                                MulByMortonBuckets{});
            thrust::lower_bound(policy, cell_idx_ptr, cell_idx_ptr + n, search_begin, search_begin + ncells + 1,
                                cell_start_ptr);
        }
    } else {
        // d_cell_idx is transient (dead after the sort below); d_orig is this function's real
        // output, needed by the caller through scatter_kernel. Both int, share one persistent
        // 2n-int scratch buffer via sub-offsets.
        ensure_capacity(gpu.d_scratch_idx, gpu.scratch_idx_cap, 2 * (size_t)n * sizeof(int));
        int *d_cell_idx = reinterpret_cast<int *>(gpu.d_scratch_idx);
        d_orig          = d_cell_idx + n;

        {
            NvtxRange range("short_range/cell_index");
            const int threads = 256, blocks = (n + threads - 1) / threads;
            cell_index_kernel<Real><<<blocks, threads, 0, gpu.stream>>>(d_pos_aos, n, L, nc, d_cell_idx); //computes each particle's (cell,bin) composite key
        }

        {
            NvtxRange range("short_range/cell_sort");
            thrust::device_ptr<int> cell_idx_ptr(d_cell_idx); //keys for sort_by_key (composite cell*kEspNbuckets+bin)
            thrust::device_ptr<int> orig_ptr(d_orig);
            thrust::sequence(policy, orig_ptr, orig_ptr + n);
            thrust::sort_by_key(policy, cell_idx_ptr, cell_idx_ptr + n, orig_ptr); //sort particles by composite key; also sort their original indices

            // d_cell_start is plan-level (gpu_create_state): size ncells+1 is fixed by nc. Search for
            // c*kEspNbuckets (not c) since the sort key is now the composite cell*kEspNbuckets+bin --
            // cell_start[c] is still exactly "first sorted position belonging to cell c" either way,
            // the bin part is transparent to it (bin_lin is always < kEspNbuckets, so it never pushes
            // a particle across a cell boundary).
            thrust::device_ptr<int> cell_start_ptr(gpu.d_cell_start);
            auto search_begin = thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                                MulByEspNbuckets{});
            // cell_start[c] = first sorted position with composite key >= c*kEspNbuckets; standard
            // sort+lower_bound bucketing idiom, giving the same CSR boundaries as an explicit counting sort.
            thrust::lower_bound(policy, cell_idx_ptr, cell_idx_ptr + n, search_begin, search_begin + ncells + 1,
                                cell_start_ptr);
        }
    }

    // xs/ys/zs/qs are this function's other output, always used together downstream --
    // one persistent 4n-Real scratch buffer via sub-offsets.
    ensure_capacity(gpu.d_scratch_sorted, gpu.scratch_sorted_cap, 4 * (size_t)n * sizeof(Real));
    Real *d_xs = reinterpret_cast<Real *>(gpu.d_scratch_sorted);
    Real *d_ys = d_xs + n;
    Real *d_zs = d_ys + n;
    Real *d_qs = d_zs + n;
    {
        NvtxRange range("short_range/gather_sorted");
        const int threads = 256, blocks = (n + threads - 1) / threads;
        //apply the permutation computed by sort_by_key, physically rearranging the particle data into cell-sorted order
        gather_sorted_kernel<Real><<<blocks, threads, 0, gpu.stream>>>(
            d_pos_aos, d_charges, d_orig, n, d_xs, d_ys, d_zs, d_qs);
    }

    *d_xs_out = d_xs; *d_ys_out = d_ys; *d_zs_out = d_zs; *d_qs_out = d_qs;
    *d_cell_start_out = gpu.d_cell_start;
    *d_orig_out = d_orig;
}

// ---------------------------------------------------------------------------
// Compile-time Horner evaluation for the short-range polynomial S(R).
//
// Coefficients ride into the kernel as a compile-time *type* (CoeffTag), not
// a runtime (pointer, count) pair -- this lets the compiler unroll the Horner
// chain into a fixed sequence of FMAs against literal immediates instead of a
// runtime loop reading global memory. CoeffTag::at() is a function (not a
// `static constexpr double data[N]` member) because nvcc treats class-scope
// constexpr arrays as host-only; the function form constant-folds on both
// host and device when called with a compile-time-constant index.
//
// clang-format off
template <typename C>
concept CoeffTag = requires {
    typename C::value_type;
    { int(C::size) };
    { typename C::value_type(C::at(std::size_t(0))) };
};
// clang-format on

template <CoeffTag Coeffs, std::size_t I, typename Real>
__device__ constexpr Real horner_recurse(Real x, Real acc) {
    if constexpr (I == 0)
        return acc;
    else
        return horner_recurse<Coeffs, I - 1>(x, acc * x + Real(Coeffs::at(I - 1)));
}

// P(x) only -- no gradient needed.
template <CoeffTag Coeffs, typename Real>
__device__ constexpr Real horner_const(Real x) {
    static_assert(Coeffs::size > 0, "empty coefficient pack");
    return horner_recurse<Coeffs, Coeffs::size - 1>(x, Real(Coeffs::at(Coeffs::size - 1)));
}

// P(x) and dP/dx(x) together, via synthetic division -- eval_esp_pair needs
// both simultaneously (force requires the derivative). Runtime equivalent,
// for reference (n = Coeffs::size, coefficients ascending-order):
//   P = c[n-1]; dP = 0;
//   for (i = n-2; i >= 0; --i) { dP = dP*x + P; P = P*x + c[i]; }
// Note dP's update at each step uses the OLD P (the value *before* that
// step's update to P) -- the same dependency the runtime loop above has.
template <CoeffTag Coeffs, std::size_t I, typename Real>
__device__ constexpr void horner_recurse_deriv(Real x, Real &P, Real &dP) {
    if constexpr (I == 0) {
        return;
    } else {
        Real old_P = P;
        P = P * x + Real(Coeffs::at(I - 1));
        dP = dP * x + old_P;
        return horner_recurse_deriv<Coeffs, I - 1>(x, P, dP);
    }
}

template <CoeffTag Coeffs, typename Real>
__device__ constexpr void horner_const_deriv(Real x, Real &P, Real &dP) {
    static_assert(Coeffs::size > 0, "empty coefficient pack");
    P = Real(Coeffs::at(Coeffs::size - 1));
    dP = Real{0};
    horner_recurse_deriv<Coeffs, Coeffs::size - 1>(x, P, dP);
}

// ---------------------------------------------------------------------------
// eval_esp_pair — the short-range kernel's actual math, for one source-target pair.
template <CoeffTag Coeffs, bool WantForce, typename Real>
__device__ __forceinline__ void eval_esp_pair(
    Real dx, Real dy, Real dz, Real q,
    Real rsc, Real cen, Real r_c_sq,
    Real &pot_acc, Real &gx_acc, Real &gy_acc, Real &gz_acc)
{
    const Real R2 = dx * dx + dy * dy + dz * dz;
    if (R2 <= Real(0) || R2 >= r_c_sq) return; // also masks the R2=0 self-pair

    const Real Rinv = rsqrt(R2); // CUDA's rsqrt()/__drsqrt_rn(), full IEEE precision
    const Real x = (R2 * Rinv + cen) * rsc; // = (R + cen)*rsc, mapped into [-1,1]

    if constexpr (WantForce) {
        Real P, dP;
        horner_const_deriv<Coeffs>(x, P, dP);
        pot_acc += q * P * Rinv;
        const Real df_dR2 = Rinv * Rinv * (dP * rsc - P * Rinv);
        gx_acc += q * dx * df_dR2;
        gy_acc += q * dy * df_dR2;
        gz_acc += q * dz * df_dR2;
    } else {
        const Real P = horner_const<Coeffs>(x);
        pot_acc += q * P * Rinv;
    }
}

// ---------------------------------------------------------------------------
// short_range_kernel_old — pre-Phase-2 version, kept side by side with the
// shared-memory/register-blocked short_range_kernel below so the two can be
// A/B'd directly. One thread per target (grid-stride over the home cell's
// targets), direct global-memory reads for sources -- no shared-memory
// tiling, no register-blocking of multiple targets per thread. Still calls
// the current eval_esp_pair<Coeffs, WantForce, Real> (compile-time
// coefficients), not the original runtime (pointer, count) version it was
// written against, so this isolates just the Phase 2 (tiling/register-
// blocking) change's effect rather than also reverting Phase 1.
// ---------------------------------------------------------------------------
template <CoeffTag Coeffs, bool WantForce, typename Real>
__global__ void short_range_kernel_old(
    int nc, int n, int out_dim,
    Real rsc, Real cen, Real r_c_sq,
    const int    *cell_start,
    const Real   *d_xs, const Real *d_ys, const Real *d_zs,
    const Real   *d_qs,
    const int    *nbc_tab,
    const Real   *off_tab,
    Real *pg_sorted)
{
    const int home = blockIdx.x; // 0 .. nc^3-1, row-major (x*nc+y)*nc+z, one block per cell
    const int cx = home / (nc * nc);
    const int cy = (home / nc) % nc;
    const int cz = home % nc;

    const int hbeg = cell_start[home];
    const int n_trg = cell_start[home + 1] - hbeg;

    for (int t = threadIdx.x; t < n_trg; t += blockDim.x) {
        const int trg = hbeg + t;
        const Real xt = d_xs[trg], yt = d_ys[trg], zt = d_zs[trg];

        Real pot_acc = Real(0), gx_acc = Real(0), gy_acc = Real(0), gz_acc = Real(0);

        for (int dxi = 0; dxi < 3; ++dxi) {
            const int nbx = nbc_tab[cx * 3 + dxi];
            const Real ox = off_tab[cx * 3 + dxi];
            for (int dyi = 0; dyi < 3; ++dyi) {
                const int nby = nbc_tab[cy * 3 + dyi];
                const Real oy = off_tab[cy * 3 + dyi];
                for (int dzi = 0; dzi < 3; ++dzi) {
                    const int nbz = nbc_tab[cz * 3 + dzi];
                    const Real oz = off_tab[cz * 3 + dzi];
                    const int nb = (nbx * nc + nby) * nc + nbz;
                    const int sbeg = cell_start[nb], send = cell_start[nb + 1];
                    for (int s = sbeg; s < send; ++s) {
                        const Real dx = xt - (d_xs[s] + ox);
                        const Real dy = yt - (d_ys[s] + oy);
                        const Real dz = zt - (d_zs[s] + oz);
                        eval_esp_pair<Coeffs, WantForce, Real>(dx, dy, dz, d_qs[s], rsc, cen, r_c_sq,
                                                               pot_acc, gx_acc, gy_acc, gz_acc);
                    }
                }
            }
        }

        pg_sorted[out_dim * trg + 0] = pot_acc;
        if constexpr (WantForce) {
            pg_sorted[out_dim * trg + 1] = gx_acc;
            pg_sorted[out_dim * trg + 2] = gy_acc;
            pg_sorted[out_dim * trg + 3] = gz_acc;
        }
    }
}

// ---------------------------------------------------------------------------
// short_range_kernel
// One CUDA block per home cell (nc³ blocks). Each thread owns up to TARGETS
// target particles at once (register-blocked: their positions/accumulators
// live in per-thread register arrays for the whole source-processing pass),
// processing successive *rounds* of target_stride = blockDim.x*TARGETS
// targets. Within a round, thread i's slot q is target index
// t_base + q*blockDim.x (strided, not blocked), so consecutive threads own
// consecutive target indices -- coalesced position loads and result writes.
// On the source side, each of the 27 neighbor cells' source lists are staged
// into shared memory in blockDim.x-sized tiles -- loaded once per tile by the
// whole block, then read by every thread for every one of its owned targets,
// instead of every thread re-reading the same source data from global memory
// independently.
// ---------------------------------------------------------------------------
constexpr int kShortRangeBlockSize = 128; // must match the launch config in short_range_gpu

template <CoeffTag Coeffs, bool WantForce, int TARGETS, typename Real>
__global__ void short_range_kernel(
    int nc, int n, int out_dim,
    Real rsc, Real cen, Real r_c_sq,
    const int * __restrict__ cell_start,
    const Real * __restrict__ d_xs, const Real * __restrict__ d_ys, const Real * __restrict__ d_zs,
    const Real * __restrict__ d_qs,
    const int * __restrict__ nbc_tab,
    const Real * __restrict__ off_tab,
    Real * __restrict__ pg_sorted)
{
    __shared__ Real s_xs[kShortRangeBlockSize];
    __shared__ Real s_ys[kShortRangeBlockSize];
    __shared__ Real s_zs[kShortRangeBlockSize];
    __shared__ Real s_qs[kShortRangeBlockSize];

    const int home = blockIdx.x; // 0 .. nc^3-1, row-major (x*nc+y)*nc+z, one block per cell
    const int cx = home / (nc * nc);
    const int cy = (home / nc) % nc;
    const int cz = home % nc;

    const int hbeg = cell_start[home];
    const int n_trg = cell_start[home + 1] - hbeg;
    // Strided target-to-thread mapping: for a fixed slot q, thread i owns target index
    // t_base + q*blockDim.x, so consecutive threads own consecutive target indices 
    const int target_stride = blockDim.x * TARGETS;
    const int n_rounds = (n_trg + target_stride - 1) / target_stride;

    for (int round = 0; round < n_rounds; ++round) {
        const int t_base = round * target_stride + threadIdx.x;

        bool active[TARGETS]; //which of this thread's TARGETS target slots actually correspond to a real particle in this round
        int trg_idx[TARGETS];
        bool any_active = false;
        Real xt[TARGETS], yt[TARGETS], zt[TARGETS];
        Real pot_acc[TARGETS] = {}, gx_acc[TARGETS] = {}, gy_acc[TARGETS] = {}, gz_acc[TARGETS] = {};
#pragma unroll
        for (int q = 0; q < TARGETS; ++q) {
            const int t = t_base + q * blockDim.x;
            active[q] = t < n_trg;
            trg_idx[q] = t;
            any_active = any_active || active[q];
            if (active[q]) {
                const int trg = hbeg + t;
                xt[q] = d_xs[trg]; yt[q] = d_ys[trg]; zt[q] = d_zs[trg];
            }
        }

        for (int dxi = 0; dxi < 3; ++dxi) {
            const int nbx = nbc_tab[cx * 3 + dxi];
            const Real ox = off_tab[cx * 3 + dxi];
            for (int dyi = 0; dyi < 3; ++dyi) {
                const int nby = nbc_tab[cy * 3 + dyi];
                const Real oy = off_tab[cy * 3 + dyi];
                for (int dzi = 0; dzi < 3; ++dzi) {
                    const int nbz = nbc_tab[cz * 3 + dzi];
                    const Real oz = off_tab[cz * 3 + dzi];
                    const int nb = (nbx * nc + nby) * nc + nbz;
                    const int sbeg = cell_start[nb], send = cell_start[nb + 1];
                    const int n_tiles = (send - sbeg + kShortRangeBlockSize - 1) / kShortRangeBlockSize;

                    for (int tile = 0; tile < n_tiles; ++tile) {
                        const int s_idx = sbeg + tile * kShortRangeBlockSize + threadIdx.x;
                        if (s_idx < send) {
                            s_xs[threadIdx.x] = d_xs[s_idx] + ox;
                            s_ys[threadIdx.x] = d_ys[s_idx] + oy;
                            s_zs[threadIdx.x] = d_zs[s_idx] + oz;
                            s_qs[threadIdx.x] = d_qs[s_idx];
                        }
                        __syncthreads();

                        const int n_local = min(kShortRangeBlockSize, send - sbeg - tile * kShortRangeBlockSize);

                        if (any_active) {
#pragma unroll
                            for (int k = 0; k < TARGETS; ++k) {
                                if (!active[k]) continue;
                                for (int s = 0; s < n_local; ++s) {
                                    eval_esp_pair<Coeffs, WantForce, Real>(xt[k] - s_xs[s], yt[k] - s_ys[s], zt[k] - s_zs[s], s_qs[s], rsc, cen, r_c_sq, pot_acc[k], gx_acc[k], gy_acc[k], gz_acc[k]);
                                }
                            }
                        }
                        __syncthreads(); // all threads done reading this tile before it's overwritten
                    }
                }
            }
        }

#pragma unroll
        for (int k = 0; k < TARGETS; ++k) {
            if (!active[k]) continue;
            const int trg = hbeg + trg_idx[k];
            pg_sorted[out_dim * trg + 0] = pot_acc[k];
            if constexpr (WantForce) {
                pg_sorted[out_dim * trg + 1] = gx_acc[k];
                pg_sorted[out_dim * trg + 2] = gy_acc[k];
                pg_sorted[out_dim * trg + 3] = gz_acc[k];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// short_range_kernel_pruned
// Sorted + geometrically-pruned strategy, mirrors CPU's short_range_prune_tile
// (one AABB test per target-tile x source-tile pair, skipping whole tiles
// farther than r_c). Kept fully side by side with short_range_kernel above --
// neither is modified by the other; short_range_gpu picks one via
// gpu.strategy (see GpuSrStrategy in esp.hpp).
//
// Unlike short_range_kernel's block-wide register-blocked target grouping,
// pruning only pays off when the AABB under test is tight, so the grouping
// here is per-WARP (32 targets = one warp), not per-block: each warp computes
// its own tight target-tile AABB independently, with no need for
// block-wide __syncthreads() during the per-warp evaluation phase.
//
// Phase 1 (whole block, once per home cell): every one of the 27 neighbor
// cells' source ranges is split into 32-wide tiles that never cross a cell
// boundary (so one periodic-image shift applies to the whole tile, exactly
// like the CPU reference); one warp computes each tile's AABB via a
// shuffle-reduction and stores (lo, hi, shift, base, len) into shared memory.
// Phase 2 (per warp, independent): each warp grid-strides over its own
// 32-target tiles, computes their AABB the same way, tests it against every
// precomputed source-tile AABB (branchless squared-box-distance vs r_c^2,
// same formula as CPU), and for survivors, loads+__shfl_sync-broadcasts each
// source across the warp before calling the unchanged eval_esp_pair.
// ---------------------------------------------------------------------------
template <typename Real>
__device__ __forceinline__ Real warp_reduce_min(Real v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = min(v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}
template <typename Real>
__device__ __forceinline__ Real warp_reduce_max(Real v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = max(v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}

// Functor (not a lambda) for thrust::transform_reduce below, matching
// MulByEspNbuckets's rationale: avoids depending on extended-lambda support.
struct CellPopulation {
    const int *cell_start;
    __host__ __device__ int operator()(int i) const { return cell_start[i + 1] - cell_start[i]; }
};

// Max particle count over all ncells cells -- used to size short_range_kernel_pruned's
// per-tile shared-memory table safely (worst case: all 27 neighbors of some
// home cell are this populated). O(ncells) reduction, negligible next to the
// short-range kernel itself.
static int compute_max_cell_pop(const int *d_cell_start, int ncells, cudaStream_t stream) {
    auto begin = thrust::counting_iterator<int>(0);
    return thrust::transform_reduce(thrust::cuda::par.on(stream), begin, begin + ncells,
                                    CellPopulation{d_cell_start}, 0, thrust::maximum<int>());
}

constexpr int kPrunedBlockSize = 128; // must match the launch config in short_range_gpu (4 warps)
constexpr int kPrunedTileWidth = 32;  // = warpSize; one warp per source/target tile

// Diagnostic-only: how many (target-tile, source-tile) pairs actually get
// pruned vs evaluated. Read back and printed once per short_range_gpu call
// in the pruned path -- not meant to stay long-term, just to quantify whether
// this config gives the AABB test any real headroom to skip tiles.
__device__ unsigned long long g_prune_tiles_tested    = 0;
__device__ unsigned long long g_prune_tiles_evaluated = 0;

// Same idea, but at per-source-POINT granularity for short_range_kernel_pruned_source
// (see below): tested counts every candidate point as if evaluated densely
// (i.e. what the box-vs-box pre-filter would have let through), evaluated
// counts only points that also survive the finer per-point test.
__device__ unsigned long long g_prune_points_tested    = 0;
__device__ unsigned long long g_prune_points_evaluated = 0;

template <CoeffTag Coeffs, bool WantForce, typename Real>
__global__ void short_range_kernel_pruned(
    int nc, int n, int out_dim,
    Real rsc, Real cen, Real r_c_sq,
    const int * __restrict__ cell_start,
    const Real * __restrict__ d_xs, const Real * __restrict__ d_ys, const Real * __restrict__ d_zs,
    const Real * __restrict__ d_qs,
    const int * __restrict__ nbc_tab,
    const Real * __restrict__ off_tab,
    Real * __restrict__ pg_sorted,
    int max_tiles)
{
    // Dynamic shared memory: per-tile AABB/shift/base/len table, sized by the
    // caller (max_tiles) from compute_max_cell_pop's cached-with-margin bound
    // (see short_range_gpu) -- large enough in practice, not an absolute guarantee.
    extern __shared__ unsigned char s_raw[];
    Real *s_lo_x = reinterpret_cast<Real *>(s_raw);
    Real *s_lo_y = s_lo_x + max_tiles;
    Real *s_lo_z = s_lo_y + max_tiles;
    Real *s_hi_x = s_lo_z + max_tiles;
    Real *s_hi_y = s_hi_x + max_tiles;
    Real *s_hi_z = s_hi_y + max_tiles;
    Real *s_shift_x = s_hi_z + max_tiles;
    Real *s_shift_y = s_shift_x + max_tiles;
    Real *s_shift_z = s_shift_y + max_tiles;
    int  *s_base = reinterpret_cast<int *>(s_shift_z + max_tiles);
    int  *s_len  = s_base + max_tiles;

    // Small fixed-size metadata for the 27 neighbor cells themselves (not the
    // tiles within them) -- static shared memory, coexists with the dynamic
    // allocation above.
    __shared__ int  s_nb_lin[27];
    __shared__ int  s_nb_tile0[28]; // exclusive prefix sum of per-neighbor tile counts; [27] = total
    __shared__ Real s_nb_shift_x[27], s_nb_shift_y[27], s_nb_shift_z[27];

    const int home = blockIdx.x; // 0 .. nc^3-1, row-major (x*nc+y)*nc+z, one block per cell
    const int cx = home / (nc * nc);
    const int cy = (home / nc) % nc;
    const int cz = home % nc;

    const int hbeg = cell_start[home];
    const int n_trg = cell_start[home + 1] - hbeg;
    if (n_trg == 0) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane = tid % warpSize;
    const int n_warps = blockDim.x / warpSize;

    // Enumerate the 27 neighbors and each one's tile count (one thread per
    // neighbor -- only 27 needed, cheap and simple vs. spreading over warps).
    if (tid < 27) {
        const int dzi = tid % 3, dyi = (tid / 3) % 3, dxi = tid / 9;
        const int nbx = nbc_tab[cx * 3 + dxi];
        const int nby = nbc_tab[cy * 3 + dyi];
        const int nbz = nbc_tab[cz * 3 + dzi];
        const Real ox = off_tab[cx * 3 + dxi];
        const Real oy = off_tab[cy * 3 + dyi];
        const Real oz = off_tab[cz * 3 + dzi];
        const int nb = (nbx * nc + nby) * nc + nbz;
        s_nb_lin[tid] = nb;
        s_nb_shift_x[tid] = ox;
        s_nb_shift_y[tid] = oy;
        s_nb_shift_z[tid] = oz;
        const int len = cell_start[nb + 1] - cell_start[nb];
        s_nb_tile0[tid] = (len + kPrunedTileWidth - 1) / kPrunedTileWidth;
    }
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < 27; ++i) {
            const int c = s_nb_tile0[i];
            s_nb_tile0[i] = acc;
            acc += c;
        }
        s_nb_tile0[27] = acc;
    }
    __syncthreads();
    // max_tiles is a cached-with-margin bound (see short_range_gpu), not a hard
    // guarantee -- clamp defensively so a rare margin-exceeded cell degrades to a
    // silently truncated (slightly wrong) tile list instead of a shared-memory
    // overflow.
    const int n_stiles = min(s_nb_tile0[27], max_tiles);

    // Phase 1: one warp per source tile (grid-strided over all n_stiles),
    // gather + AABB-reduce + store to shared memory.
    for (int gt = warp_id; gt < n_stiles; gt += n_warps) {
        int nbi = 0;
        while (nbi < 26 && s_nb_tile0[nbi + 1] <= gt)
            ++nbi;
        const int local_tile = gt - s_nb_tile0[nbi];
        const int nb = s_nb_lin[nbi];
        const int nb_beg = cell_start[nb], nb_end = cell_start[nb + 1];
        const int t_base = nb_beg + local_tile * kPrunedTileWidth;
        const int t_len = min(kPrunedTileWidth, nb_end - t_base);
        const Real ox = s_nb_shift_x[nbi], oy = s_nb_shift_y[nbi], oz = s_nb_shift_z[nbi];

        const bool valid = lane < t_len;
        const Real x = valid ? d_xs[t_base + lane] + ox : Real(0);
        const Real y = valid ? d_ys[t_base + lane] + oy : Real(0);
        const Real z = valid ? d_zs[t_base + lane] + oz : Real(0);
        Real lo_x = warp_reduce_min(valid ? x : Real(INFINITY));
        Real hi_x = warp_reduce_max(valid ? x : Real(-INFINITY));
        Real lo_y = warp_reduce_min(valid ? y : Real(INFINITY));
        Real hi_y = warp_reduce_max(valid ? y : Real(-INFINITY));
        Real lo_z = warp_reduce_min(valid ? z : Real(INFINITY));
        Real hi_z = warp_reduce_max(valid ? z : Real(-INFINITY));

        if (lane == 0) {
            s_lo_x[gt] = lo_x; s_lo_y[gt] = lo_y; s_lo_z[gt] = lo_z;
            s_hi_x[gt] = hi_x; s_hi_y[gt] = hi_y; s_hi_z[gt] = hi_z;
            s_shift_x[gt] = ox; s_shift_y[gt] = oy; s_shift_z[gt] = oz;
            s_base[gt] = t_base; s_len[gt] = t_len;
        }
    }
    __syncthreads(); // every warp's Phase 2 reads the full shared tile table

    // Phase 2: each warp grid-strides over its own 32-target tiles.
    for (int t0 = warp_id * kPrunedTileWidth; t0 < n_trg; t0 += n_warps * kPrunedTileWidth) {
        const int tlen = min(kPrunedTileWidth, n_trg - t0);
        const bool my_active = lane < tlen;
        const int trg = hbeg + t0 + lane;
        const Real xt = my_active ? d_xs[trg] : Real(0);
        const Real yt = my_active ? d_ys[trg] : Real(0);
        const Real zt = my_active ? d_zs[trg] : Real(0);

        const Real tlo_x = warp_reduce_min(my_active ? xt : Real(INFINITY));
        const Real thi_x = warp_reduce_max(my_active ? xt : Real(-INFINITY));
        const Real tlo_y = warp_reduce_min(my_active ? yt : Real(INFINITY));
        const Real thi_y = warp_reduce_max(my_active ? yt : Real(-INFINITY));
        const Real tlo_z = warp_reduce_min(my_active ? zt : Real(INFINITY));
        const Real thi_z = warp_reduce_max(my_active ? zt : Real(-INFINITY));

        Real pot_acc = Real(0), gx_acc = Real(0), gy_acc = Real(0), gz_acc = Real(0);

        for (int st = 0; st < n_stiles; ++st) {
            // Branchless squared box-distance, same formula as CPU's short_range_prune_tile.
            const Real ddx = max(Real(0), max(s_lo_x[st] - thi_x, tlo_x - s_hi_x[st]));
            const Real ddy = max(Real(0), max(s_lo_y[st] - thi_y, tlo_y - s_hi_y[st]));
            const Real ddz = max(Real(0), max(s_lo_z[st] - thi_z, tlo_z - s_hi_z[st]));
            const bool pruned = ddx * ddx + ddy * ddy + ddz * ddz > r_c_sq;
            if (lane == 0) {
                atomicAdd(&g_prune_tiles_tested, 1ull);
                if (!pruned)
                    atomicAdd(&g_prune_tiles_evaluated, 1ull);
            }
            if (pruned)
                continue; // whole tile pruned: no source load, no eval_esp_pair calls

            const int s_base_i = s_base[st], s_len_i = s_len[st];
            const Real shx = s_shift_x[st], shy = s_shift_y[st], shz = s_shift_z[st];
            const bool have_src = lane < s_len_i;
            const Real sx = have_src ? d_xs[s_base_i + lane] + shx : Real(0);
            const Real sy = have_src ? d_ys[s_base_i + lane] + shy : Real(0);
            const Real sz = have_src ? d_zs[s_base_i + lane] + shz : Real(0);
            const Real sq = have_src ? d_qs[s_base_i + lane] : Real(0);

            // Every lane loads one source and broadcasts it to the whole warp in
            // turn -- avoids both shared-memory tile-staging and redundant
            // per-lane global reads for the same tile.
            for (int l = 0; l < s_len_i; ++l) {
                const Real bx = __shfl_sync(0xffffffffu, sx, l);
                const Real by = __shfl_sync(0xffffffffu, sy, l);
                const Real bz = __shfl_sync(0xffffffffu, sz, l);
                const Real bq = __shfl_sync(0xffffffffu, sq, l);
                if (my_active)
                    eval_esp_pair<Coeffs, WantForce, Real>(xt - bx, yt - by, zt - bz, bq, rsc, cen, r_c_sq,
                                                           pot_acc, gx_acc, gy_acc, gz_acc);
            }
        }

        if (my_active) {
            pg_sorted[out_dim * trg + 0] = pot_acc;
            if constexpr (WantForce) {
                pg_sorted[out_dim * trg + 1] = gx_acc;
                pg_sorted[out_dim * trg + 2] = gy_acc;
                pg_sorted[out_dim * trg + 3] = gz_acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// short_range_kernel_pruned_source
// Box-vs-POINT pruning, mirrors CPU's short_range_prune_source: instead of
// testing a whole source tile's AABB against the target tile's AABB (as
// short_range_kernel_pruned above does), each individual source point is
// tested against the target box. This is strictly finer -- a source chunk
// whose AABB straddles the r_c boundary can still have most of its individual
// points genuinely out of range, which box-vs-box can't detect but box-vs-
// point does. Measured to matter here: box-vs-box only skips ~10% of tile
// pairs when cell width ~= r_c (see g_prune_tiles_* above), since r_c is
// deliberately close to the cell size by construction (nc = floor(L/r_c)).
//
// Phase 1 is IDENTICAL to short_range_kernel_pruned's (same per-home-cell
// source-chunk table, same shared-memory layout/sizing) -- reused verbatim.
// Phase 2 replaces the per-chunk AABB-vs-AABB test with: (a) the same cheap
// AABB-vs-AABB pre-filter as a free first pass (skips a whole chunk with no
// per-point work when possible), then (b) for chunks that don't fully clear
// it, a per-lane box-vs-point test, warp-ballot to find survivors, and a
// shared-memory scatter/gather compaction (not a register-shuffle sorting
// network -- much simpler to get right, and the extra static shared memory
// is a few KB, negligible next to the existing dynamic per-tile table) that
// packs survivors into a dense sub-list before evaluation.
// ---------------------------------------------------------------------------
template <CoeffTag Coeffs, bool WantForce, typename Real>
__global__ void short_range_kernel_pruned_source(
    int nc, int n, int out_dim,
    Real rsc, Real cen, Real r_c_sq,
    const int * __restrict__ cell_start,
    const Real * __restrict__ d_xs, const Real * __restrict__ d_ys, const Real * __restrict__ d_zs,
    const Real * __restrict__ d_qs,
    const int * __restrict__ nbc_tab,
    const Real * __restrict__ off_tab,
    Real * __restrict__ pg_sorted,
    int max_tiles)
{
    // Dynamic shared memory: identical per-tile AABB/shift/base/len table to
    // short_range_kernel_pruned -- see that kernel's comment for the layout.
    extern __shared__ unsigned char s_raw[];
    Real *s_lo_x = reinterpret_cast<Real *>(s_raw);
    Real *s_lo_y = s_lo_x + max_tiles;
    Real *s_lo_z = s_lo_y + max_tiles;
    Real *s_hi_x = s_lo_z + max_tiles;
    Real *s_hi_y = s_hi_x + max_tiles;
    Real *s_hi_z = s_hi_y + max_tiles;
    Real *s_shift_x = s_hi_z + max_tiles;
    Real *s_shift_y = s_shift_x + max_tiles;
    Real *s_shift_z = s_shift_y + max_tiles;
    int  *s_base = reinterpret_cast<int *>(s_shift_z + max_tiles);
    int  *s_len  = s_base + max_tiles;

    __shared__ int  s_nb_lin[27];
    __shared__ int  s_nb_tile0[28];
    __shared__ Real s_nb_shift_x[27], s_nb_shift_y[27], s_nb_shift_z[27];

    // Per-warp compaction staging buffer: kPrunedBlockSize/kPrunedTileWidth
    // warps, kPrunedTileWidth (=32, one per lane) slots each. Static (not part
    // of the dynamic per-tile allocation above) since its size is fixed at
    // compile time regardless of particle density.
    constexpr int kMaxWarps = kPrunedBlockSize / kPrunedTileWidth;
    __shared__ Real s_stage_x[kMaxWarps][kPrunedTileWidth];
    __shared__ Real s_stage_y[kMaxWarps][kPrunedTileWidth];
    __shared__ Real s_stage_z[kMaxWarps][kPrunedTileWidth];
    __shared__ Real s_stage_q[kMaxWarps][kPrunedTileWidth];

    const int home = blockIdx.x;
    const int cx = home / (nc * nc);
    const int cy = (home / nc) % nc;
    const int cz = home % nc;

    const int hbeg = cell_start[home];
    const int n_trg = cell_start[home + 1] - hbeg;
    if (n_trg == 0) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane = tid % warpSize;
    const int n_warps = blockDim.x / warpSize;

    // --- Phase 1 (identical to short_range_kernel_pruned) ---
    if (tid < 27) {
        const int dzi = tid % 3, dyi = (tid / 3) % 3, dxi = tid / 9;
        const int nbx = nbc_tab[cx * 3 + dxi];
        const int nby = nbc_tab[cy * 3 + dyi];
        const int nbz = nbc_tab[cz * 3 + dzi];
        const Real ox = off_tab[cx * 3 + dxi];
        const Real oy = off_tab[cy * 3 + dyi];
        const Real oz = off_tab[cz * 3 + dzi];
        const int nb = (nbx * nc + nby) * nc + nbz;
        s_nb_lin[tid] = nb;
        s_nb_shift_x[tid] = ox;
        s_nb_shift_y[tid] = oy;
        s_nb_shift_z[tid] = oz;
        const int len = cell_start[nb + 1] - cell_start[nb];
        s_nb_tile0[tid] = (len + kPrunedTileWidth - 1) / kPrunedTileWidth;
    }
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < 27; ++i) {
            const int c = s_nb_tile0[i];
            s_nb_tile0[i] = acc;
            acc += c;
        }
        s_nb_tile0[27] = acc;
    }
    __syncthreads();
    const int n_stiles = min(s_nb_tile0[27], max_tiles);

    for (int gt = warp_id; gt < n_stiles; gt += n_warps) {
        int nbi = 0;
        while (nbi < 26 && s_nb_tile0[nbi + 1] <= gt)
            ++nbi;
        const int local_tile = gt - s_nb_tile0[nbi];
        const int nb = s_nb_lin[nbi];
        const int nb_beg = cell_start[nb], nb_end = cell_start[nb + 1];
        const int t_base = nb_beg + local_tile * kPrunedTileWidth;
        const int t_len = min(kPrunedTileWidth, nb_end - t_base);
        const Real ox = s_nb_shift_x[nbi], oy = s_nb_shift_y[nbi], oz = s_nb_shift_z[nbi];

        const bool valid = lane < t_len;
        const Real x = valid ? d_xs[t_base + lane] + ox : Real(0);
        const Real y = valid ? d_ys[t_base + lane] + oy : Real(0);
        const Real z = valid ? d_zs[t_base + lane] + oz : Real(0);
        Real lo_x = warp_reduce_min(valid ? x : Real(INFINITY));
        Real hi_x = warp_reduce_max(valid ? x : Real(-INFINITY));
        Real lo_y = warp_reduce_min(valid ? y : Real(INFINITY));
        Real hi_y = warp_reduce_max(valid ? y : Real(-INFINITY));
        Real lo_z = warp_reduce_min(valid ? z : Real(INFINITY));
        Real hi_z = warp_reduce_max(valid ? z : Real(-INFINITY));

        if (lane == 0) {
            s_lo_x[gt] = lo_x; s_lo_y[gt] = lo_y; s_lo_z[gt] = lo_z;
            s_hi_x[gt] = hi_x; s_hi_y[gt] = hi_y; s_hi_z[gt] = hi_z;
            s_shift_x[gt] = ox; s_shift_y[gt] = oy; s_shift_z[gt] = oz;
            s_base[gt] = t_base; s_len[gt] = t_len;
        }
    }
    __syncthreads();

    // --- Phase 2: per warp, box-vs-point pruning + compaction ---
    for (int t0 = warp_id * kPrunedTileWidth; t0 < n_trg; t0 += n_warps * kPrunedTileWidth) {
        const int tlen = min(kPrunedTileWidth, n_trg - t0);
        const bool my_active = lane < tlen;
        const int trg = hbeg + t0 + lane;
        const Real xt = my_active ? d_xs[trg] : Real(0);
        const Real yt = my_active ? d_ys[trg] : Real(0);
        const Real zt = my_active ? d_zs[trg] : Real(0);

        const Real tlo_x = warp_reduce_min(my_active ? xt : Real(INFINITY));
        const Real thi_x = warp_reduce_max(my_active ? xt : Real(-INFINITY));
        const Real tlo_y = warp_reduce_min(my_active ? yt : Real(INFINITY));
        const Real thi_y = warp_reduce_max(my_active ? yt : Real(-INFINITY));
        const Real tlo_z = warp_reduce_min(my_active ? zt : Real(INFINITY));
        const Real thi_z = warp_reduce_max(my_active ? zt : Real(-INFINITY));

        Real pot_acc = Real(0), gx_acc = Real(0), gy_acc = Real(0), gz_acc = Real(0);

        for (int st = 0; st < n_stiles; ++st) {
            // Cheap box-vs-box pre-filter, reusing the existing Phase-1 AABBs:
            // a whole chunk out of range needs no per-point work at all. Free
            // (same AABBs short_range_kernel_pruned already computes), and
            // preserves that kernel's ~10% skip as a base case here too.
            const Real ddx = max(Real(0), max(s_lo_x[st] - thi_x, tlo_x - s_hi_x[st]));
            const Real ddy = max(Real(0), max(s_lo_y[st] - thi_y, tlo_y - s_hi_y[st]));
            const Real ddz = max(Real(0), max(s_lo_z[st] - thi_z, tlo_z - s_hi_z[st]));
            const bool box_pruned = ddx * ddx + ddy * ddy + ddz * ddz > r_c_sq;

            const int s_len_i = s_len[st];
            if (lane == 0)
                atomicAdd(&g_prune_points_tested, (unsigned long long)s_len_i);
            if (box_pruned)
                continue;

            const int s_base_i = s_base[st];
            const Real shx = s_shift_x[st], shy = s_shift_y[st], shz = s_shift_z[st];
            const bool have_src = lane < s_len_i;
            const Real sx = have_src ? d_xs[s_base_i + lane] + shx : Real(0);
            const Real sy = have_src ? d_ys[s_base_i + lane] + shy : Real(0);
            const Real sz = have_src ? d_zs[s_base_i + lane] + shz : Real(0);
            const Real sq = have_src ? d_qs[s_base_i + lane] : Real(0);

            // Per-point box-distance test: same formula as the box-vs-box test
            // above, but with the source side degenerated to a single point
            // (lo=hi=s) instead of a chunk AABB -- strictly finer.
            const Real pdx = max(Real(0), max(tlo_x - sx, sx - thi_x));
            const Real pdy = max(Real(0), max(tlo_y - sy, sy - thi_y));
            const Real pdz = max(Real(0), max(tlo_z - sz, sz - thi_z));
            const bool in_range = have_src && (pdx * pdx + pdy * pdy + pdz * pdz <= r_c_sq);

            const unsigned mask = __ballot_sync(0xffffffffu, in_range);
            if (lane == 0)
                atomicAdd(&g_prune_points_evaluated, (unsigned long long)__popc(mask));
            if (mask == 0)
                continue; // chunk's box test didn't clear it, but no individual survivors either

            if (in_range) {
                const int slot = __popc(mask & ((1u << lane) - 1u));
                s_stage_x[warp_id][slot] = sx;
                s_stage_y[warp_id][slot] = sy;
                s_stage_z[warp_id][slot] = sz;
                s_stage_q[warp_id][slot] = sq;
            }
            __syncwarp(); // scatters visible before any lane reads the staging buffer

            const int m = __popc(mask);
            for (int i = 0; i < m; ++i) {
                const Real bx = s_stage_x[warp_id][i];
                const Real by = s_stage_y[warp_id][i];
                const Real bz = s_stage_z[warp_id][i];
                const Real bq = s_stage_q[warp_id][i];
                if (my_active)
                    eval_esp_pair<Coeffs, WantForce, Real>(xt - bx, yt - by, zt - bz, bq, rsc, cen, r_c_sq,
                                                           pot_acc, gx_acc, gy_acc, gz_acc);
            }
            __syncwarp(); // all reads of this chunk's staging buffer done before the next chunk scatters
        }

        if (my_active) {
            pg_sorted[out_dim * trg + 0] = pot_acc;
            if constexpr (WantForce) {
                pg_sorted[out_dim * trg + 1] = gx_acc;
                pg_sorted[out_dim * trg + 2] = gy_acc;
                pg_sorted[out_dim * trg + 3] = gz_acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// scatter_kernel
// ---------------------------------------------------------------------------
template <typename Real>
__global__ void scatter_kernel(
    int n, int out_dim,
    const Real *pg_sorted,
    const int  *d_orig,
    const Real *d_qs_sorted,
    Real *d_pot, Real *d_fx, Real *d_fy, Real *d_fz)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n) return;
    const int o = d_orig[a];
    d_pot[o] += pg_sorted[out_dim * a + 0];
    if (out_dim > 1) {
        const Real q = d_qs_sorted[a];
        d_fx[o] += -q * pg_sorted[out_dim * a + 1];
        d_fy[o] += -q * pg_sorted[out_dim * a + 2];
        d_fz[o] += -q * pg_sorted[out_dim * a + 3];
    }
}

// ---------------------------------------------------------------------------
// scaling_kernel — multiply each k-space grid point by its scaling coefficient.
// Operates in-place: d_b_hat[i] *= scaling_coeffs[i], producing pot_hat.
// Mirrors CPU: pot_hat[i] = b_hat[i] * scaling_coeffs[i].
// ---------------------------------------------------------------------------
template <typename Real>
__global__ void scaling_kernel(
    int ntot,
    const Real       *scaling_coeffs,
    ComplexT<Real>   *b_hat)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntot) return;
    Real s = scaling_coeffs[i];
    b_hat[i] = {b_hat[i].x * s, b_hat[i].y * s}; //.x = real part; .y = imaginary part
}

// ---------------------------------------------------------------------------
// normalize_kernel — divide every element by ntot after cuFFT IFFT.
// cuFFT's inverse transform is unnormalized (output = ntot * true IFFT).
// Mirrors CPU: ifftn_3d divides by ntot internally.
// ---------------------------------------------------------------------------
template <typename Real>
__global__ void normalize_kernel(int ntot, Real inv_ntot, ComplexT<Real> *data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntot) return;
    data[i] = {data[i].x * inv_ntot, data[i].y * inv_ntot};
}

// ---------------------------------------------------------------------------
// extract_real_kernel — write the real part of each NU complex value to d_out.
// Mirrors CPU: pot[j] += real(c[j]) after finufft3d2.
// ---------------------------------------------------------------------------
template <typename Real>
__global__ void extract_real_kernel(int n, const ComplexT<Real> *d_c, Real *d_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_out[i] += Real(d_c[i].x);
}

// ---------------------------------------------------------------------------
// grad_scaling_kernel
// F = -q*grad(u); grad(u)_hat_k = i*k*u_hat_k, so each force spectrum is
// pot_hat (already b_hat*scaling_coeffs, from the in-place scaling_kernel above)
// times i*k_component*coeff_grad. pot_hat is passed in directly (gpu.d_b_hat
// still holds it after step 4's IFFT, since that reads d_b_hat without
// mutating it), so no separate scaling_coeffs/k_idx buffers are needed here —
// k_idx is cheap to recompute per-thread from the flat grid index.
// f_hat_x uses k_idx[iz] (axis swap: see esp.cpp long_range()'s force block).
// ---------------------------------------------------------------------------
__device__ __forceinline__ int grad_kidx(int i, int nf) { return (i <= nf / 2) ? i : i - nf; }

template <typename Real>
__global__ void grad_scaling_kernel(
    int nf, Real coeff_grad,
    const ComplexT<Real> *pot_hat,
    ComplexT<Real> *f_hat_x,
    ComplexT<Real> *f_hat_y,
    ComplexT<Real> *f_hat_z)
{
    const long long ntot = (long long)nf * nf * nf;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntot) return;

    const int iz = int(i % nf);
    const int iy = int((i / nf) % nf);
    const int ix = int(i / ((long long)nf * nf));

    const ComplexT<Real> s = pot_hat[i];
    auto mul_ik = [=](int k) {
        const Real factor = coeff_grad * Real(k);
        // s * (i * factor) = (-s.y*factor, s.x*factor)
        return ComplexT<Real>{-s.y * factor, s.x * factor};
    };
    f_hat_x[i] = mul_ik(grad_kidx(iz, nf));
    f_hat_y[i] = mul_ik(grad_kidx(iy, nf));
    f_hat_z[i] = mul_ik(grad_kidx(ix, nf));
}

// ---------------------------------------------------------------------------
// accumulate_force_kernel — d_force_out[j] += -charge[j]*real(d_force_c[j]).
// Mirrors CPU: fx[j] += -charges[j]*Real(force_x_c[j].real()).
// Charges are read from d_c (already packed as {charge, 0} for spreading), so
// no separate real-charges buffer is needed here.
// ---------------------------------------------------------------------------
template <typename Real>
__global__ void accumulate_force_kernel(int n, const ComplexT<Real> *d_c, const ComplexT<Real> *d_force_c,
                                        Real *d_force_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_force_out[i] += Real(-d_c[i].x * d_force_c[i].x);
}

// ---------------------------------------------------------------------------
// self_interaction_kernel
// ---------------------------------------------------------------------------
template <typename Real>
__global__ void self_interaction_kernel(int n, Real factor, const Real *d_charges, Real *d_pot)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_pot[i] -= d_charges[i] * factor;
}

// ---------------------------------------------------------------------------
// scale_pack_kernel — AoS Real positions/charges (already on device) -> scaled
// [-pi,pi) SoA Real coords + packed complex charges for long_range_gpu.
// Entirely on-device: no host-side loop, no extra host<->device round trip
// when d_pos_aos/d_charges are already resident (esp_eval_gpu_impl).
// ---------------------------------------------------------------------------
template <typename Real>
__global__ void scale_pack_kernel(
    const Real *d_pos_aos, const Real *d_charges, int n, Real scale,
    Real *d_x, Real *d_y, Real *d_z, ComplexT<Real> *d_c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_x[i] = d_pos_aos[3 * i + 0] * scale;
    d_y[i] = d_pos_aos[3 * i + 1] * scale;
    d_z[i] = d_pos_aos[3 * i + 2] * scale;
    d_c[i] = {d_charges[i], Real(0)};
}

// ---------------------------------------------------------------------------
// cufinufft_{setpts,execute}_t / cufft_exec_c2c_t — Real-dispatched wrappers.
// cufinufft_plan/cufinufftf_plan are different C types (not overloads of one
// name), so GpuState stores whichever one this plan was created for as a
// void*; these wrappers reinterpret_cast it back and call the matching
// double/float library entry point, chosen at compile time via if constexpr.
// ---------------------------------------------------------------------------
template <typename Real>
static int cufinufft_setpts_t(void *plan, int n, Real *x, Real *y, Real *z) {
    if constexpr (std::is_same_v<Real, double>)
        return cufinufft_setpts(reinterpret_cast<cufinufft_plan>(plan), n, x, y, z, 0, nullptr, nullptr, nullptr);
    else
        return cufinufftf_setpts(reinterpret_cast<cufinufftf_plan>(plan), n, x, y, z, 0, nullptr, nullptr, nullptr);
}

template <typename Real>
static int cufinufft_execute_t(void *plan, ComplexT<Real> *c, ComplexT<Real> *f) {
    if constexpr (std::is_same_v<Real, double>)
        return cufinufft_execute(reinterpret_cast<cufinufft_plan>(plan), c, f);
    else
        return cufinufftf_execute(reinterpret_cast<cufinufftf_plan>(plan), c, f);
}

template <typename Real>
static cufftResult cufft_exec_c2c_t(cufftHandle plan, ComplexT<Real> *in, ComplexT<Real> *out, int direction) {
    if constexpr (std::is_same_v<Real, double>)
        return cufftExecZ2Z(plan, in, out, direction);
    else
        return cufftExecC2C(plan, in, out, direction);
}

// ---------------------------------------------------------------------------
// long_range_gpu
// Mirrors the CPU long_range(): spread → FFT → scale → IFFT → interp (+ forces).
// Genuinely Real-templated -- gpu must have been created with use_float
// matching Real (see gpu_create_state / check_plan_real).
// ---------------------------------------------------------------------------
template <typename Real>
static void long_range_gpu(
    GpuState &gpu,
    int n,
    const Real *d_x, const Real *d_y, const Real *d_z,
    const ComplexT<Real> *d_c,
    Real coeff_grad,
    bool want_force,
    Real *d_pot, Real *d_fx, Real *d_fy, Real *d_fz)
{
    const long long ntot = (long long)gpu.nf * gpu.nf * gpu.nf;
    auto *d_b      = reinterpret_cast<ComplexT<Real> *>(gpu.d_b);
    auto *d_b_hat  = reinterpret_cast<ComplexT<Real> *>(gpu.d_b_hat);
    auto *d_scaling_coeffs = reinterpret_cast<Real *>(gpu.d_scaling_coeffs);

    // -----------------------------------------------------------------------
    // Step 1: Spread — NU points → uniform grid  (d_b, nf³ complex)
    // Mirrors CPU: finufft3d1 with opts.spreadinterponly=1.
    // Uses the pre-created gpu.cfnufft_plan_1 (type-1, makeplan done at plan time).
    // -----------------------------------------------------------------------
    {
        NvtxRange range("long_range/spread");
        // cuFINUFFT plans use the default stream; our H2D copies run on gpu.stream.
        // Sync the device so both streams see consistent data before setpts.
        cudaError_t cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess)
            throw std::runtime_error(
                std::string("long_range_gpu: pre-setpts sync failed: ") + cudaGetErrorString(cerr));

        int ier = cufinufft_setpts_t<Real>(gpu.cfnufft_plan_1, n,
                                           const_cast<Real *>(d_x),
                                           const_cast<Real *>(d_y),
                                           const_cast<Real *>(d_z));
        if (ier != 0) {
            cudaError_t last = cudaGetLastError();
            throw std::runtime_error(
                "long_range_gpu: cufinufft_setpts spread failed, ier=" + std::to_string(ier) +
                ", last CUDA error: " + cudaGetErrorString(last));
        }

        // Zero before spreading — cuFINUFFT accumulates into the output buffer.
        cudaMemsetAsync(d_b, 0, ntot * sizeof(ComplexT<Real>), gpu.stream);

        ier = cufinufft_execute_t<Real>(gpu.cfnufft_plan_1, const_cast<ComplexT<Real> *>(d_c), d_b);
        if (ier != 0)
            throw std::runtime_error("long_range_gpu: cufinufft_execute spread failed, ier=" + std::to_string(ier));
    }

    // -----------------------------------------------------------------------
    // Step 2: Forward FFT — d_b → d_b_hat  (real-space grid → k-space)
    // Mirrors CPU: fftn_3d(b, b_hat, nf).
    // -----------------------------------------------------------------------
    {
        NvtxRange range("long_range/fft_forward");
        cufftResult r = cufft_exec_c2c_t<Real>(gpu.fft_plan, d_b, d_b_hat, CUFFT_FORWARD);
        if (r != CUFFT_SUCCESS)
            throw std::runtime_error("long_range_gpu: cufft forward failed, err=" + std::to_string(r));
    }

    // -----------------------------------------------------------------------
    // Step 3: Scale — d_b_hat[i] *= scaling_coeffs[i]  (d_b_hat is now pot_hat)
    // Mirrors CPU: pot_hat[i] = b_hat[i] * scaling_coeffs[i].
    // In-place: no extra buffer needed; d_b_hat is reused as pot_hat for IFFT.
    // -----------------------------------------------------------------------
    {
        NvtxRange range("long_range/scale");
        const int threads = 256;
        const int blocks  = static_cast<int>((ntot + threads - 1) / threads);
        scaling_kernel<Real><<<blocks, threads, 0, gpu.stream>>>(
            static_cast<int>(ntot), d_scaling_coeffs, d_b_hat);
    }
    // -----------------------------------------------------------------------
    // Step 4: Inverse FFT — pot_hat (d_b_hat) → d_grid_pot.
    // Reuses d_b as d_grid_pot (spread output no longer needed after step 1).
    // cuFFT IFFT is unnormalized, so follow with normalize_kernel (÷ ntot).
    // Mirrors CPU: ifftn_3d(pot_hat, grid_pot, nf).
    // -----------------------------------------------------------------------
    {
        NvtxRange range("long_range/fft_inverse_normalize");
        cufftResult r = cufft_exec_c2c_t<Real>(gpu.fft_plan, d_b_hat, d_b, CUFFT_INVERSE);
        if (r != CUFFT_SUCCESS)
            throw std::runtime_error("long_range_gpu: cufft inverse failed, err=" + std::to_string(r));

        const Real inv_ntot = Real(1) / Real(ntot);
        const int threads = 256;
        const int blocks  = static_cast<int>((ntot + threads - 1) / threads);
        normalize_kernel<Real><<<blocks, threads, 0, gpu.stream>>>(
            static_cast<int>(ntot), inv_ntot, d_b);
    }
    // gpu.cfnufft_plan_2's NU points don't change within a single long_range_gpu
    // call (same d_x/d_y/d_z throughout), so setpts is bound once here and every
    // type-2 execute below (potential and, if requested, the 3 force components)
    // reuses that binding.
    const bool want_pot_interp   = (d_pot != nullptr);
    const bool want_force_interp = want_force && (d_fx || d_fy || d_fz);
    if (want_pot_interp || want_force_interp) {
        NvtxRange range("long_range/interp_setpts");
        int ier = cufinufft_setpts_t<Real>(gpu.cfnufft_plan_2, n,
                                           const_cast<Real *>(d_x),
                                           const_cast<Real *>(d_y),
                                           const_cast<Real *>(d_z));
        if (ier != 0)
            throw std::runtime_error("long_range_gpu: cufinufft_setpts interp failed, ier=" + std::to_string(ier));
    }

    // -----------------------------------------------------------------------
    // Step 5: Interp (cuFINUFFT type-2, spreadinterponly=1)
    // d_grid_pot (d_b) → d_pot_c (n complex values at NU points) → d_pot.
    // Mirrors CPU: finufft3d2 with spreadinterponly=1, then pot[j] += real(c[j]).
    // Uses the pre-created gpu.cfnufft_plan_2 (type-2, makeplan done at plan time).
    // -----------------------------------------------------------------------
    if (want_pot_interp) {
        NvtxRange range("long_range/interp_potential");
        // Reused for the force components below too (d_scratch_nu_c) -- pot_c is fully
        // consumed here before steps 6-8 even start, so there's no lifetime overlap.
        ensure_capacity(gpu.d_scratch_nu_c, gpu.scratch_nu_c_cap, (size_t)n * sizeof(ComplexT<Real>));
        auto *d_pot_c = reinterpret_cast<ComplexT<Real> *>(gpu.d_scratch_nu_c);

        // d_b is d_grid_pot (after step 4). Execute: uniform grid → NU values.
        int ier = cufinufft_execute_t<Real>(gpu.cfnufft_plan_2, d_pot_c, d_b);
        if (ier != 0) throw std::runtime_error("long_range_gpu: cufinufft_execute interp failed, ier=" + std::to_string(ier));

        const int threads = 256;
        const int blocks  = (n + threads - 1) / threads;
        extract_real_kernel<Real><<<blocks, threads, 0, gpu.stream>>>(n, d_pot_c, d_pot);
    }

    // -----------------------------------------------------------------------
    // Steps 6-8: force path (ik method).
    // d_b_hat still holds pot_hat (= b_hat*scaling_coeffs from step 3 — step
    // 4's IFFT read it without mutating it), so it's used directly here.
    // Mirrors CPU long_range()'s force block (esp.cpp).
    // -----------------------------------------------------------------------
    if (want_force_interp) {
        NvtxRange force_path_range("long_range/force_path");
        // Plan-level (gpu_create_state): size nf³ is fixed, allocated once whenever the plan
        // computes forces at all.
        auto *f_hat_x = reinterpret_cast<ComplexT<Real> *>(gpu.d_fhat_x);
        auto *f_hat_y = reinterpret_cast<ComplexT<Real> *>(gpu.d_fhat_y);
        auto *f_hat_z = reinterpret_cast<ComplexT<Real> *>(gpu.d_fhat_z);

        // Step 6: build the three force spectra from pot_hat.
        {
            NvtxRange range("long_range/force_grad_scaling");
            const int threads = 256;
            const int blocks  = static_cast<int>((ntot + threads - 1) / threads);
            grad_scaling_kernel<Real><<<blocks, threads, 0, gpu.stream>>>(
                gpu.nf, coeff_grad, d_b_hat, f_hat_x, f_hat_y, f_hat_z);
        }

        // Step 7: inverse FFT each component in-place, then normalize (cuFFT's
        // IFFT is unnormalized, mirrors normalize_kernel usage in step 4).
        auto ifft_and_normalize = [&](ComplexT<Real> *buf) {
            NvtxRange range("long_range/force_ifft_normalize");
            cufftResult r = cufft_exec_c2c_t<Real>(gpu.fft_plan, buf, buf, CUFFT_INVERSE);
            if (r != CUFFT_SUCCESS)
                throw std::runtime_error("long_range_gpu: cufft inverse (force) failed, err=" +
                                         std::to_string(r));
            const Real inv_ntot = Real(1) / Real(ntot);
            const int threads = 256;
            const int blocks  = static_cast<int>((ntot + threads - 1) / threads);
            normalize_kernel<Real><<<blocks, threads, 0, gpu.stream>>>(static_cast<int>(ntot), inv_ntot, buf);
        };
        ifft_and_normalize(f_hat_x);
        ifft_and_normalize(f_hat_y);
        ifft_and_normalize(f_hat_z);

        // Step 8: interp (cuFINUFFT type-2, reusing the setpts binding above)
        // each grid_force component → NU points, then accumulate
        // d_f{x,y,z}[j] += -charge[j]*real(force_c[j]).
        // Same scratch buffer as d_pot_c above, reused sequentially across all three
        // components (each is fully consumed by accumulate_force_kernel before the next
        // interp_and_accumulate call touches it).
        ensure_capacity(gpu.d_scratch_nu_c, gpu.scratch_nu_c_cap, (size_t)n * sizeof(ComplexT<Real>));
        auto *d_force_c = reinterpret_cast<ComplexT<Real> *>(gpu.d_scratch_nu_c);

        auto interp_and_accumulate = [&](ComplexT<Real> *grid_force, Real *d_force_out) {
            if (!d_force_out) return;
            NvtxRange range("long_range/force_interp_accumulate");
            int ier = cufinufft_execute_t<Real>(gpu.cfnufft_plan_2, d_force_c, grid_force);
            if (ier != 0)
                throw std::runtime_error("long_range_gpu: cufinufft_execute force-interp failed, ier=" +
                                         std::to_string(ier));

            const int threads = 256;
            const int blocks  = (n + threads - 1) / threads;
            accumulate_force_kernel<Real><<<blocks, threads, 0, gpu.stream>>>(n, d_c, d_force_c, d_force_out);
        };
        interp_and_accumulate(f_hat_x, d_fx);
        interp_and_accumulate(f_hat_y, d_fy);
        interp_and_accumulate(f_hat_z, d_fz);
    }
}

// ---------------------------------------------------------------------------
// short_range_gpu
// ---------------------------------------------------------------------------
template <typename Real>
static void short_range_gpu(
    GpuState &gpu,
    const Real *d_pos_aos, const Real *d_charges,
    int n, int nc, Real L, Real r_c,
    int n_digits, bool want_force,
    Real *d_pot, Real *d_fx, Real *d_fy, Real *d_fz)
{
    Real *d_xs, *d_ys, *d_zs, *d_qs;
    int  *d_cell_start, *d_orig;
    {
        NvtxRange range("short_range/build_cell_list");
        build_cell_list_gpu<Real>(d_pos_aos, d_charges, n, nc, L, &d_xs, &d_ys, &d_zs, &d_qs, &d_cell_start, &d_orig,
                                  gpu);
    }

    const int out_dim = want_force ? 4 : 1;
    ensure_capacity(gpu.d_scratch_pg, gpu.scratch_pg_cap, (size_t)out_dim * n * sizeof(Real));
    Real *d_pg_sorted = reinterpret_cast<Real *>(gpu.d_scratch_pg);

    const Real rsc    = Real(2) / r_c;
    const Real cen    = Real(-0.5) * r_c;
    const Real r_c_sq = r_c * r_c;

    {
        NvtxRange range("short_range/pair_kernel");
        const int threads = kShortRangeBlockSize; // must match short_range_kernel's shared-memory tile size
        constexpr int kTargetsPerThread = 3; // register-blocking width

        // Pruned-path-only (PruneTile and PruneSource share this table): size the
        // per-tile shared-memory table from the max cell population. Computing this
        // exactly requires a host-device sync (thrust::transform_reduce returns a
        // host scalar), which would otherwise happen on EVERY call -- so it's cached
        // in GpuState and only recomputed when n changes (the common case is repeated
        // calls with the same n, e.g. a benchmark loop or successive simulation
        // timesteps, where the sync is pure overhead unrelated to any actual change in
        // cell populations). A 25% margin absorbs modest population drift between
        // calls at the same n (e.g. particles moving slightly between timesteps)
        // without needing to resync; this is a pragmatic bound, not a rigorous one --
        // a dataset whose max cell population grows by >25% while n stays fixed could
        // still overflow it.
        int pruned_max_tiles = 0;
        size_t pruned_shmem_bytes = 0;
        if (gpu.strategy != GpuSrStrategy::Dense) {
            if (n != gpu.pruned_max_tiles_cache_n) {
                NvtxRange pop_range("short_range/max_cell_pop");
                const int max_pop = compute_max_cell_pop(d_cell_start, nc * nc * nc, gpu.stream);
                const int tiles = 27 * ((max_pop + kPrunedTileWidth - 1) / kPrunedTileWidth);
                gpu.pruned_max_tiles_cache = tiles + tiles / 4; // +25% margin
                gpu.pruned_max_tiles_cache_n = n;
            }
            pruned_max_tiles = gpu.pruned_max_tiles_cache;
            pruned_shmem_bytes = (size_t)pruned_max_tiles * (9 * sizeof(Real) + 2 * sizeof(int));
        }

#define DMK_SR_LAUNCH(TAG, WANTFORCE)                                                                                  \
    do {                                                                                                              \
        if (gpu.strategy == GpuSrStrategy::PruneSource) {                                                             \
            auto *_k = short_range_kernel_pruned_source<TAG, WANTFORCE, Real>;                                        \
            cudaError_t _attr_err =                                                                                   \
                cudaFuncSetAttribute(_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pruned_shmem_bytes);        \
            if (_attr_err != cudaSuccess)                                                                             \
                throw std::runtime_error(std::string("short_range_gpu: cudaFuncSetAttribute failed: ") +              \
                                         cudaGetErrorString(_attr_err));                                              \
            _k<<<nc * nc * nc, kPrunedBlockSize, pruned_shmem_bytes, gpu.stream>>>(                                    \
                nc, n, out_dim, rsc, cen, r_c_sq, d_cell_start, d_xs, d_ys, d_zs, d_qs,                                \
                gpu.d_nbc_tab, reinterpret_cast<const Real *>(gpu.d_off_tab), d_pg_sorted, pruned_max_tiles);          \
        } else if (gpu.strategy == GpuSrStrategy::PruneTile) {                                                        \
            auto *_k = short_range_kernel_pruned<TAG, WANTFORCE, Real>;                                              \
            cudaError_t _attr_err =                                                                                   \
                cudaFuncSetAttribute(_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pruned_shmem_bytes);        \
            if (_attr_err != cudaSuccess)                                                                             \
                throw std::runtime_error(std::string("short_range_gpu: cudaFuncSetAttribute failed: ") +              \
                                         cudaGetErrorString(_attr_err));                                              \
            _k<<<nc * nc * nc, kPrunedBlockSize, pruned_shmem_bytes, gpu.stream>>>(                                    \
                nc, n, out_dim, rsc, cen, r_c_sq, d_cell_start, d_xs, d_ys, d_zs, d_qs,                                \
                gpu.d_nbc_tab, reinterpret_cast<const Real *>(gpu.d_off_tab), d_pg_sorted, pruned_max_tiles);          \
        } else {                                                                                                       \
            short_range_kernel<TAG, WANTFORCE, kTargetsPerThread, Real><<<nc * nc * nc, threads, 0, gpu.stream>>>(     \
                nc, n, out_dim, rsc, cen, r_c_sq, d_cell_start, d_xs, d_ys, d_zs, d_qs,                                \
                gpu.d_nbc_tab, reinterpret_cast<const Real *>(gpu.d_off_tab), d_pg_sorted);                           \
        }                                                                                                              \
    } while (0)
// #define DMK_SR_LAUNCH(TAG, WANTFORCE)                                                                              \
//     short_range_kernel_old<TAG, WANTFORCE, Real><<<nc * nc * nc, threads, 0, gpu.stream>>>(                        \
//         nc, n, out_dim, rsc, cen, r_c_sq, d_cell_start, d_xs, d_ys, d_zs, d_qs,                                    \
//         gpu.d_nbc_tab, reinterpret_cast<const Real *>(gpu.d_off_tab), d_pg_sorted)
#define DMK_SR_CASCADE(PREFIX, WANTFORCE)                                                                              \
    do {                                                                                                              \
        if      (n_digits <= 2)  DMK_SR_LAUNCH(EspSrCoeffs_##PREFIX##_2, WANTFORCE);                                   \
        else if (n_digits <= 3)  DMK_SR_LAUNCH(EspSrCoeffs_##PREFIX##_3, WANTFORCE);                                   \
        else if (n_digits <= 4)  DMK_SR_LAUNCH(EspSrCoeffs_##PREFIX##_4, WANTFORCE);                                   \
        else if (n_digits <= 5)  DMK_SR_LAUNCH(EspSrCoeffs_##PREFIX##_5, WANTFORCE);                                   \
        else if (n_digits <= 6)  DMK_SR_LAUNCH(EspSrCoeffs_##PREFIX##_6, WANTFORCE);                                   \
        else if (n_digits <= 7)  DMK_SR_LAUNCH(EspSrCoeffs_##PREFIX##_7, WANTFORCE);                                   \
        else if (n_digits <= 8)  DMK_SR_LAUNCH(EspSrCoeffs_##PREFIX##_8, WANTFORCE);                                   \
        else if (n_digits <= 9)  DMK_SR_LAUNCH(EspSrCoeffs_##PREFIX##_9, WANTFORCE);                                   \
        else if (n_digits <= 10) DMK_SR_LAUNCH(EspSrCoeffs_##PREFIX##_10, WANTFORCE);                                  \
        else if (n_digits <= 11) DMK_SR_LAUNCH(EspSrCoeffs_##PREFIX##_11, WANTFORCE);                                  \
        else if (n_digits <= 12) DMK_SR_LAUNCH(EspSrCoeffs_##PREFIX##_12, WANTFORCE);                                  \
        else throw std::runtime_error("short_range_gpu: unsupported n_digits (must be <= 12)");                       \
    } while (0)

        if (gpu.eval_type == DMK_POTENTIAL_GRAD) DMK_SR_CASCADE(PotGrad, true);
        else                                     DMK_SR_CASCADE(Pot, false);

#undef DMK_SR_CASCADE
#undef DMK_SR_LAUNCH

        if (gpu.strategy == GpuSrStrategy::PruneTile) {
            unsigned long long tested = 0, evaluated = 0;
            cudaMemcpyFromSymbolAsync(&tested, g_prune_tiles_tested, sizeof(tested), 0,
                                       cudaMemcpyDeviceToHost, gpu.stream);
            cudaMemcpyFromSymbolAsync(&evaluated, g_prune_tiles_evaluated, sizeof(evaluated), 0,
                                       cudaMemcpyDeviceToHost, gpu.stream);
            cudaStreamSynchronize(gpu.stream);
            fprintf(stderr, "# prune diag: tiles_tested=%llu tiles_evaluated=%llu skip_frac=%.4f\n",
                    tested, evaluated, tested ? 1.0 - double(evaluated) / double(tested) : 0.0);
            const unsigned long long zero = 0;
            cudaMemcpyToSymbolAsync(g_prune_tiles_tested, &zero, sizeof(zero), 0,
                                     cudaMemcpyHostToDevice, gpu.stream);
            cudaMemcpyToSymbolAsync(g_prune_tiles_evaluated, &zero, sizeof(zero), 0,
                                     cudaMemcpyHostToDevice, gpu.stream);
        } else if (gpu.strategy == GpuSrStrategy::PruneSource) {
            unsigned long long tested = 0, evaluated = 0;
            cudaMemcpyFromSymbolAsync(&tested, g_prune_points_tested, sizeof(tested), 0,
                                       cudaMemcpyDeviceToHost, gpu.stream);
            cudaMemcpyFromSymbolAsync(&evaluated, g_prune_points_evaluated, sizeof(evaluated), 0,
                                       cudaMemcpyDeviceToHost, gpu.stream);
            cudaStreamSynchronize(gpu.stream);
            fprintf(stderr, "# prune diag: points_tested=%llu points_evaluated=%llu skip_frac=%.4f\n",
                    tested, evaluated, tested ? 1.0 - double(evaluated) / double(tested) : 0.0);
            const unsigned long long zero = 0;
            cudaMemcpyToSymbolAsync(g_prune_points_tested, &zero, sizeof(zero), 0,
                                     cudaMemcpyHostToDevice, gpu.stream);
            cudaMemcpyToSymbolAsync(g_prune_points_evaluated, &zero, sizeof(zero), 0,
                                     cudaMemcpyHostToDevice, gpu.stream);
        }
    }

    {
        NvtxRange range("short_range/scatter");
        const int t2 = 256, b2 = (n + t2 - 1) / t2;
        scatter_kernel<Real><<<b2, t2, 0, gpu.stream>>>(n, out_dim, d_pg_sorted, d_orig, d_qs, d_pot, d_fx, d_fy,
                                                        d_fz);
    }
    // d_xs/ys/zs/qs, d_cell_start, d_orig, d_pg_sorted are all persistent GpuState scratch --
    // nothing to free here; they're reused as-is by the next call.
}

// ---------------------------------------------------------------------------
// gpu_host_buf — returns a reference to the typed host output buffer.
// ---------------------------------------------------------------------------
template <typename Real>
static auto &host_buf(GpuState *gpu) {
    if constexpr (std::is_same_v<Real, double>) return gpu->h_dbl_buf;
    else                                         return gpu->h_flt_buf;
}

// ---------------------------------------------------------------------------
// self_factor — GpuState stores both self_factor_d/self_factor_f (computed once
// at plan-creation time in esp_create_gpu_plan); pick the one matching Real.
// ---------------------------------------------------------------------------
template <typename Real>
static Real self_factor(GpuState *gpu) {
    if constexpr (std::is_same_v<Real, double>) return Real(gpu->self_factor_d);
    else                                         return Real(gpu->self_factor_f);
}

// ---------------------------------------------------------------------------
// check_plan_real — a GpuState is created for exactly one Real (use_float,
// set in gpu_create_state); calling esp_eval_gpu<Real> with the other Real on
// the same plan would reinterpret_cast every long-range buffer/plan handle to
// the wrong concrete type. Fail loudly instead.
// ---------------------------------------------------------------------------
template <typename Real>
static void check_plan_real(const GpuState *gpu) {
    const bool wants_float = std::is_same_v<Real, float>;
    if (wants_float != gpu->use_float)
        throw std::runtime_error(
            std::string("esp_eval_gpu: called with Real=") + (wants_float ? "float" : "double") +
            " but this plan was created for " + (gpu->use_float ? "float" : "double"));
}

// ---------------------------------------------------------------------------
// gpu_make_spans — resize + zero the host buffer, return four output spans.
// ---------------------------------------------------------------------------
template <typename Real>
static auto gpu_make_spans(GpuState *gpu, int n) {
    [[maybe_unused]] const bool want_force = (gpu->eval_type == DMK_POTENTIAL_GRAD);
    const int  slots      = want_force ? 4 : 1;
    auto &buf = host_buf<Real>(gpu);
    buf.assign(slots * n, Real(0));
    Real *p = buf.data();
    return std::tuple{
        std::span<Real>(p,         n),
        want_force ? std::span<Real>(p +   n, n) : std::span<Real>{},
        want_force ? std::span<Real>(p + 2*n, n) : std::span<Real>{},
        want_force ? std::span<Real>(p + 3*n, n) : std::span<Real>{}
    };
}

// ---------------------------------------------------------------------------
// esp_eval_gpu_impl — full pipeline (short + long + self-correction).
// ---------------------------------------------------------------------------
template <typename Real>
static PotForce<Real> esp_eval_gpu_impl(
    GpuState *gpu,
    const std::vector<Vec3T<Real>> &r_src,
    const std::vector<Real>        &charges)
{
    check_plan_real<Real>(gpu);
    const int  n  = static_cast<int>(r_src.size());
    const int  nc = static_cast<int>(std::floor(gpu->L / gpu->r_c));
    const bool want_force = (gpu->eval_type == DMK_POTENTIAL_GRAD);
    auto [pot, fx, fy, fz] = gpu_make_spans<Real>(gpu, n);
    const Real *h_pos_aos = reinterpret_cast<const Real *>(r_src.data());

    // Positions/charges in the AoS Real layout short_range_gpu wants -- persistent scratch,
    // reused across calls instead of malloc/free'd every eval (see GpuState).
    ensure_capacity(gpu->d_scratch_pos, gpu->scratch_pos_cap, 4 * (size_t)n * sizeof(Real));
    Real *d_pos_aos = reinterpret_cast<Real *>(gpu->d_scratch_pos);
    Real *d_charges = d_pos_aos + 3 * (size_t)n;
    ensure_capacity(gpu->d_scratch_out, gpu->scratch_out_cap, 4 * (size_t)n * sizeof(Real));
    Real *d_pot = reinterpret_cast<Real *>(gpu->d_scratch_out);
    Real *d_fx = want_force ? d_pot + n : nullptr;
    Real *d_fy = want_force ? d_pot + 2 * (size_t)n : nullptr;
    Real *d_fz = want_force ? d_pot + 3 * (size_t)n : nullptr;
    {
        NvtxRange range("eval/upload_input");
        cudaMemcpyAsync(d_pos_aos, h_pos_aos, 3 * (size_t)n * sizeof(Real), cudaMemcpyHostToDevice, gpu->stream);
        cudaMemcpyAsync(d_charges, charges.data(), n * sizeof(Real), cudaMemcpyHostToDevice, gpu->stream);
        cudaMemsetAsync(d_pot, 0, n * sizeof(Real), gpu->stream);
        if (want_force) {
            cudaMemsetAsync(d_fx, 0, n * sizeof(Real), gpu->stream);
            cudaMemsetAsync(d_fy, 0, n * sizeof(Real), gpu->stream);
            cudaMemsetAsync(d_fz, 0, n * sizeof(Real), gpu->stream);
        }
    }

    // Short-range: accumulates (+=) directly into d_pot/d_fx/d_fy/d_fz.
    {
        NvtxRange range("eval/short_range");
        short_range_gpu<Real>(*gpu, d_pos_aos, d_charges, n, nc, Real(gpu->L), Real(gpu->r_c), gpu->n_digits,
                             want_force, d_pot, d_fx, d_fy, d_fz);
    }

    // Long-range: needs its own scaled [-pi,pi) SoA coords + packed complex charges.
    // d_pos_aos/d_charges are already resident on device (uploaded above for
    // short-range), so this is a pure device-to-device reshape+scale -- no host
    // loop, no extra host<->device traffic.
    const Real scale = Real(2.0 * M_PI) / Real(gpu->L);
    Real *d_x, *d_y, *d_z; ComplexT<Real> *d_c;
    {
        NvtxRange range("eval/long_range_setup");
        ensure_capacity(gpu->d_scratch_lr_xyz, gpu->scratch_lr_xyz_cap, 3 * (size_t)n * sizeof(Real));
        d_x = reinterpret_cast<Real *>(gpu->d_scratch_lr_xyz);
        d_y = d_x + n;
        d_z = d_y + n;
        ensure_capacity(gpu->d_scratch_lr_c, gpu->scratch_lr_c_cap, (size_t)n * sizeof(ComplexT<Real>));
        d_c = reinterpret_cast<ComplexT<Real> *>(gpu->d_scratch_lr_c);

        const int threads = 256, blocks = (n + threads - 1) / threads;
        scale_pack_kernel<Real><<<blocks, threads, 0, gpu->stream>>>(
            d_pos_aos, d_charges, n, scale, d_x, d_y, d_z, d_c);
    }

    {
        NvtxRange range("eval/long_range");
        long_range_gpu<Real>(*gpu, n, d_x, d_y, d_z, d_c, scale, want_force, d_pot, d_fx, d_fy, d_fz);
    }

    // Self-interaction correction (potential only, matches CPU self_interaction).
    {
        NvtxRange range("eval/self_interaction");
        const int threads = 256, blocks = (n + threads - 1) / threads;
        self_interaction_kernel<Real><<<blocks, threads, 0, gpu->stream>>>(n, self_factor<Real>(gpu), d_charges,
                                                                           d_pot);
    }

    {
        NvtxRange range("eval/download_output");
        cudaStreamSynchronize(gpu->stream);
        cudaMemcpy(pot.data(), d_pot, n * sizeof(Real), cudaMemcpyDeviceToHost);
        if (want_force) {
            cudaMemcpy(fx.data(), d_fx, n * sizeof(Real), cudaMemcpyDeviceToHost);
            cudaMemcpy(fy.data(), d_fy, n * sizeof(Real), cudaMemcpyDeviceToHost);
            cudaMemcpy(fz.data(), d_fz, n * sizeof(Real), cudaMemcpyDeviceToHost);
        }
    }
    // d_pos_aos/d_charges, d_pot/d_fx/d_fy/d_fz, d_x/d_y/d_z, d_c are all persistent GpuState
    // scratch -- nothing to free here.

    return {pot, fx, fy, fz};
}

// ---------------------------------------------------------------------------
// esp_eval_gpu_short_range_impl — only the short-range direct sum.
// ---------------------------------------------------------------------------
template <typename Real>
static PotForce<Real> esp_eval_gpu_short_range_impl(
    GpuState *gpu,
    const std::vector<Vec3T<Real>> &r_src,
    const std::vector<Real>        &charges)
{
    check_plan_real<Real>(gpu);
    const int  n  = static_cast<int>(r_src.size());
    const int  nc = static_cast<int>(std::floor(gpu->L / gpu->r_c));
    const bool want_force = (gpu->eval_type == DMK_POTENTIAL_GRAD);
    auto [pot, fx, fy, fz] = gpu_make_spans<Real>(gpu, n);
    const Real *h_pos_aos = reinterpret_cast<const Real *>(r_src.data());

    ensure_capacity(gpu->d_scratch_pos, gpu->scratch_pos_cap, 4 * (size_t)n * sizeof(Real));
    Real *d_pos_aos = reinterpret_cast<Real *>(gpu->d_scratch_pos);
    Real *d_charges = d_pos_aos + 3 * (size_t)n;
    ensure_capacity(gpu->d_scratch_out, gpu->scratch_out_cap, 4 * (size_t)n * sizeof(Real));
    Real *d_pot = reinterpret_cast<Real *>(gpu->d_scratch_out);
    Real *d_fx = want_force ? d_pot + n : nullptr;
    Real *d_fy = want_force ? d_pot + 2 * (size_t)n : nullptr;
    Real *d_fz = want_force ? d_pot + 3 * (size_t)n : nullptr;
    {
        NvtxRange range("eval/upload_input");
        cudaMemcpyAsync(d_pos_aos, h_pos_aos, 3 * (size_t)n * sizeof(Real), cudaMemcpyHostToDevice, gpu->stream);
        cudaMemcpyAsync(d_charges, charges.data(), n * sizeof(Real), cudaMemcpyHostToDevice, gpu->stream);
        cudaMemsetAsync(d_pot, 0, n * sizeof(Real), gpu->stream);
        if (want_force) {
            cudaMemsetAsync(d_fx, 0, n * sizeof(Real), gpu->stream);
            cudaMemsetAsync(d_fy, 0, n * sizeof(Real), gpu->stream);
            cudaMemsetAsync(d_fz, 0, n * sizeof(Real), gpu->stream);
        }
    }

    {
        NvtxRange range("eval/short_range");
        short_range_gpu<Real>(*gpu, d_pos_aos, d_charges, n, nc, Real(gpu->L), Real(gpu->r_c), gpu->n_digits,
                             want_force, d_pot, d_fx, d_fy, d_fz);
    }

    {
        NvtxRange range("eval/download_output");
        cudaStreamSynchronize(gpu->stream);
        cudaMemcpy(pot.data(), d_pot, n * sizeof(Real), cudaMemcpyDeviceToHost);
        if (want_force) {
            cudaMemcpy(fx.data(), d_fx, n * sizeof(Real), cudaMemcpyDeviceToHost);
            cudaMemcpy(fy.data(), d_fy, n * sizeof(Real), cudaMemcpyDeviceToHost);
            cudaMemcpy(fz.data(), d_fz, n * sizeof(Real), cudaMemcpyDeviceToHost);
        }
    }

    return {pot, fx, fy, fz};
}

// ---------------------------------------------------------------------------
// esp_eval_gpu_long_range_impl — only the long-range NUFFT pipeline.
// ---------------------------------------------------------------------------
template <typename Real>
static PotForce<Real> esp_eval_gpu_long_range_impl(
    GpuState *gpu,
    const std::vector<Vec3T<Real>> &r_src,
    const std::vector<Real>        &charges)
{
    check_plan_real<Real>(gpu);
    const int  n  = static_cast<int>(r_src.size());
    [[maybe_unused]] const bool want_force = (gpu->eval_type == DMK_POTENTIAL_GRAD);
    auto [pot, fx, fy, fz] = gpu_make_spans<Real>(gpu, n);

    // Upload raw AoS positions/charges once, then scale+pack into SoA [-pi,pi)
    // coords + complex charges entirely on-device (no host-side loop).
    const Real scale = Real(2.0 * M_PI) / Real(gpu->L);
    const Real *h_pos_aos = reinterpret_cast<const Real *>(r_src.data());
    Real *d_x, *d_y, *d_z; ComplexT<Real> *d_c;
    Real *d_pot, *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
    {
        NvtxRange range("eval/upload_input");
        ensure_capacity(gpu->d_scratch_pos, gpu->scratch_pos_cap, 4 * (size_t)n * sizeof(Real));
        Real *d_pos_aos = reinterpret_cast<Real *>(gpu->d_scratch_pos);
        Real *d_charges = d_pos_aos + 3 * (size_t)n;
        cudaMemcpyAsync(d_pos_aos, h_pos_aos, 3 * (size_t)n * sizeof(Real), cudaMemcpyHostToDevice, gpu->stream);
        cudaMemcpyAsync(d_charges, charges.data(), n * sizeof(Real), cudaMemcpyHostToDevice, gpu->stream);

        ensure_capacity(gpu->d_scratch_lr_xyz, gpu->scratch_lr_xyz_cap, 3 * (size_t)n * sizeof(Real));
        d_x = reinterpret_cast<Real *>(gpu->d_scratch_lr_xyz);
        d_y = d_x + n;
        d_z = d_y + n;
        ensure_capacity(gpu->d_scratch_lr_c, gpu->scratch_lr_c_cap, (size_t)n * sizeof(ComplexT<Real>));
        d_c = reinterpret_cast<ComplexT<Real> *>(gpu->d_scratch_lr_c);
        const int threads = 256, blocks = (n + threads - 1) / threads;
        scale_pack_kernel<Real><<<blocks, threads, 0, gpu->stream>>>(
            d_pos_aos, d_charges, n, scale, d_x, d_y, d_z, d_c);

        // Device output buffers for potential and (if requested) forces -- persistent scratch.
        ensure_capacity(gpu->d_scratch_out, gpu->scratch_out_cap, 4 * (size_t)n * sizeof(Real));
        d_pot = reinterpret_cast<Real *>(gpu->d_scratch_out);
        d_fx = want_force ? d_pot + n : nullptr;
        d_fy = want_force ? d_pot + 2 * (size_t)n : nullptr;
        d_fz = want_force ? d_pot + 3 * (size_t)n : nullptr;
        cudaMemsetAsync(d_pot, 0, n * sizeof(Real), gpu->stream);
        if (want_force) {
            cudaMemsetAsync(d_fx, 0, n * sizeof(Real), gpu->stream);
            cudaMemsetAsync(d_fy, 0, n * sizeof(Real), gpu->stream);
            cudaMemsetAsync(d_fz, 0, n * sizeof(Real), gpu->stream);
        }
    }

    {
        NvtxRange range("eval/long_range");
        long_range_gpu<Real>(*gpu, n,
                             d_x, d_y, d_z, d_c,
                             scale, want_force,
                             d_pot, d_fx, d_fy, d_fz);
    }

    {
        NvtxRange range("eval/download_output");
        cudaStreamSynchronize(gpu->stream); //barrier - force the CPU (host) thread to wait until all previously submitted tasks in a specific GPU stream have completed
        cudaMemcpy(pot.data(), d_pot, n * sizeof(Real), cudaMemcpyDeviceToHost);
        if (want_force) {
            cudaMemcpy(fx.data(), d_fx, n * sizeof(Real), cudaMemcpyDeviceToHost);
            cudaMemcpy(fy.data(), d_fy, n * sizeof(Real), cudaMemcpyDeviceToHost);
            cudaMemcpy(fz.data(), d_fz, n * sizeof(Real), cudaMemcpyDeviceToHost);
        }
    }

    return {pot, fx, fy, fz};
}

// ---------------------------------------------------------------------------
// Public overloads matching esp.hpp declarations.
// ---------------------------------------------------------------------------
PotForce<float>  esp_eval_gpu(GpuState *gpu, const std::vector<Vec3T<float>>  &r_src, const std::vector<float>  &charges) { return esp_eval_gpu_impl<float>(gpu,  r_src, charges); }
PotForce<double> esp_eval_gpu(GpuState *gpu, const std::vector<Vec3T<double>> &r_src, const std::vector<double> &charges) { return esp_eval_gpu_impl<double>(gpu, r_src, charges); }

PotForce<float>  esp_eval_gpu_short_range(GpuState *gpu, const std::vector<Vec3T<float>>  &r_src, const std::vector<float>  &charges) { return esp_eval_gpu_short_range_impl<float>(gpu,  r_src, charges); }
PotForce<double> esp_eval_gpu_short_range(GpuState *gpu, const std::vector<Vec3T<double>> &r_src, const std::vector<double> &charges) { return esp_eval_gpu_short_range_impl<double>(gpu, r_src, charges); }

PotForce<float>  esp_eval_gpu_long_range(GpuState *gpu, const std::vector<Vec3T<float>>  &r_src, const std::vector<float>  &charges) { return esp_eval_gpu_long_range_impl<float>(gpu,  r_src, charges); }
PotForce<double> esp_eval_gpu_long_range(GpuState *gpu, const std::vector<Vec3T<double>> &r_src, const std::vector<double> &charges) { return esp_eval_gpu_long_range_impl<double>(gpu, r_src, charges); }

} // namespace dmk
