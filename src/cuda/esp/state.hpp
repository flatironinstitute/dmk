#pragma once
// GpuState and the shared helpers the ESP GPU launchers need. Private to src/cuda/esp/.

#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/esp.hpp>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufinufft.h>

#include <stdexcept>
#include <type_traits>
#include <vector>

namespace dmk {

template <typename Real>
using ComplexT = std::conditional_t<std::is_same_v<Real, double>, cuDoubleComplex, cuFloatComplex>;

// Per-kernel component counts, from the same tables the CPU plan uses.
struct KernelDims {
    int in_dim;     // charge components per source
    int normal_dim; // normal components per source
    int charge_dim; // packed payload width, [charge | normal]
    int out_dim;    // output components per target
    int n_channels; // long-range spread/forward-FFT passes
};

inline KernelDims kernel_dims(dmk_ikernel kernel, dmk_eval_type eval_type) {
    KernelDims d{};
    d.in_dim = get_kernel_input_dim(3, kernel);
    d.normal_dim = (kernel == DMK_STRESSLET) ? 3 : 0;
    d.charge_dim = d.in_dim + d.normal_dim;
    d.out_dim = get_kernel_output_dim(3, kernel, eval_type);
    d.n_channels = (kernel == DMK_STRESSLET) ? 9 : d.in_dim;
    return d;
}

// Owns all physics params and CUDA objects for one GPU plan.
struct GpuState {
    // Set once at plan creation, read at every eval.
    int nf;
    int n_digits;
    // The particle box is the unit box; L_grid drives the FFT and NU scaling (== 1 when periodic).
    double L_grid, r_c;
    // PSWF bandwidth; the short-range coefficients are generated per launch from (n_digits, beta).
    double beta;
    double self_factor;
    // Yukawa's residual coefficients depend on lambda and r_c.
    double fparam = 0.0;
    double dipole_grad_self = 0.0;
    dmk_ikernel kernel = DMK_LAPLACE;
    dmk_eval_type eval_type;
    // Free space marks out-of-range neighbour cells with -1 instead of wrapping.
    bool use_periodic = true;
    double trunc_rl = 0.0;
    // The one Real this plan exists for; every eval must match it (check_plan_real).
    bool use_float = false;

    GpuSrStrategy strategy = GpuSrStrategy::Dense;

    GpuSortMode sort_mode = GpuSortMode::Bins;
    // Pruned-strategy shared sizing; the max cell population costs a sync, so cache it on n.
    int pruned_max_tiles_cache = 0;
    int pruned_max_tiles_cache_n = -1;

    // Host output workspace; grown as needed, never shrunk between calls.
    std::vector<double> h_dbl_buf;
    std::vector<float> h_flt_buf;

    // Uploaded once at plan creation. void* because the concrete type depends on use_float.
    cudaStream_t stream = nullptr;
    void *d_scaling_coeffs = nullptr; // nf³ Real
    int nc = 0;                       // cells per dimension, = floor(1/r_c)
    int *d_nbc_tab = nullptr;         // nc*3 ints — neighbor cell index per (cell,delta)
    void *d_off_tab = nullptr;        // nc*3 Real — periodic image shift per (cell,delta)
    // n_channels * nf³ ComplexT<Real> — spread output (NU → uniform), forward-FFT'd in place
    void *d_b = nullptr;
    // out_dim * nf³ ComplexT<Real> — the projector's output spectra, inverse-FFT'd in place
    void *d_u_hat = nullptr;
    cufftHandle fft_plan{}; // nf³ 3-D c2c (Z2Z or C2C), created at plan time
    bool fft_plan_valid = false;
    void *cfnufft_plan_1 = nullptr; // cufinufft_plan or cufinufftf_plan — type-1 (NU → uniform)
    void *cfnufft_plan_2 = nullptr; // cufinufft_plan or cufinufftf_plan — type-2 (uniform → NU)
    int *d_cell_start = nullptr;    // ncells+1 ints (ncells=nc³) — short-range cell list

    // Per-eval scratch: grown by byte capacity, never shrunk. Roles used together share a buffer.
    void *d_scratch_pos = nullptr;
    size_t scratch_pos_cap = 0; // pos_aos (3n) + charges (n), Real
    void *d_scratch_out = nullptr;
    size_t scratch_out_cap = 0; // pot/gx/gy/gz (4n), Real (outputs)
    void *d_scratch_idx = nullptr;
    size_t scratch_idx_cap = 0; // cell_idx (n) + orig (n), int
    void *d_scratch_sorted = nullptr;
    size_t scratch_sorted_cap = 0; // xs/ys/zs/qs (4n), Real
    void *d_scratch_pg = nullptr;
    size_t scratch_pg_cap = 0; // pg_sorted (out_dim*n), Real
    void *d_scratch_lr_xyz = nullptr;
    size_t scratch_lr_xyz_cap = 0; // x/y/z (3n), Real
    void *d_scratch_lr_c = nullptr;
    size_t scratch_lr_c_cap = 0; // packed charges (n), ComplexT<Real>
    void *d_scratch_nu_c = nullptr;
    size_t scratch_nu_c_cap = 0; // NU point values (n), ComplexT<Real>
    // Pruned-strategy diagnostics; allocated only when PRUNE_STATS is on.
    void *d_scratch_prune_stats = nullptr;
    size_t scratch_prune_stats_cap = 0;

    KernelDims dims{};

    GpuState() = default;
    GpuState(const GpuState &) = delete;
    GpuState &operator=(const GpuState &) = delete;
    ~GpuState() {
        if (cfnufft_plan_1) {
            if (use_float)
                cufinufftf_destroy(reinterpret_cast<cufinufftf_plan>(cfnufft_plan_1));
            else
                cufinufft_destroy(reinterpret_cast<cufinufft_plan>(cfnufft_plan_1));
        }
        if (cfnufft_plan_2) {
            if (use_float)
                cufinufftf_destroy(reinterpret_cast<cufinufftf_plan>(cfnufft_plan_2));
            else
                cufinufft_destroy(reinterpret_cast<cufinufft_plan>(cfnufft_plan_2));
        }
        if (fft_plan_valid)
            cufftDestroy(fft_plan);
        if (d_u_hat)
            cudaFree(d_u_hat);
        if (d_b)
            cudaFree(d_b);
        if (d_scaling_coeffs)
            cudaFree(d_scaling_coeffs);
        if (d_nbc_tab)
            cudaFree(d_nbc_tab);
        if (d_off_tab)
            cudaFree(d_off_tab);
        if (d_cell_start)
            cudaFree(d_cell_start);
        if (d_scratch_pos)
            cudaFree(d_scratch_pos);
        if (d_scratch_out)
            cudaFree(d_scratch_out);
        if (d_scratch_idx)
            cudaFree(d_scratch_idx);
        if (d_scratch_sorted)
            cudaFree(d_scratch_sorted);
        if (d_scratch_pg)
            cudaFree(d_scratch_pg);
        if (d_scratch_lr_xyz)
            cudaFree(d_scratch_lr_xyz);
        if (d_scratch_lr_c)
            cudaFree(d_scratch_lr_c);
        if (d_scratch_nu_c)
            cudaFree(d_scratch_nu_c);
        if (d_scratch_prune_stats)
            cudaFree(d_scratch_prune_stats);
        if (stream)
            cudaStreamDestroy(stream);
    }
};

inline void ensure_capacity(void *&ptr, size_t &cap, size_t needed_bytes) {
    if (cap >= needed_bytes)
        return;
    if (ptr)
        cudaFree(ptr);
    if (cudaMalloc(&ptr, needed_bytes) != cudaSuccess)
        throw std::runtime_error("ensure_capacity: cudaMalloc failed");
    cap = needed_bytes;
}

} // namespace dmk
