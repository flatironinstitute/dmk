// GPU implementation of the ESP solver.

#include <dmk/cuda/esp_gpu.hpp>
#include <dmk/esp.hpp>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufinufft.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <complex>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace dmk {

// ---------------------------------------------------------------------------
// GpuState — owns all physics params and CUDA objects for one GPU plan.
// ---------------------------------------------------------------------------
struct GpuState {
    // Physics params — set once at esp_create_gpu_plan time, read at every eval.
    int           nf;
    int           n_digits;
    double        L, r_c, sigma, tol;
    double        self_factor_d;
    float         self_factor_f;
    dmk_eval_type eval_type;

    // Host output workspace; grown as needed, never shrunk between calls.
    std::vector<double> h_dbl_buf;
    std::vector<float>  h_flt_buf;

    // Plan-level device data — uploaded once at esp_create_gpu_plan time.
    cudaStream_t   stream            = nullptr;
    double        *d_scaling_coeffs  = nullptr;   // nf³ doubles
    cufftHandle    fft_plan{};                     // nf³ 3-D c2c, created at plan time
    cufinufft_plan cfnufft_plan_1    = nullptr;    // type-1 spread (NU → uniform)
    cufinufft_plan cfnufft_plan_2    = nullptr;    // type-2 interp (uniform → NU)

    // Per-eval device scratch; grown as needed, never shrunk.
    double *d_dbl_buf = nullptr;   size_t dbl_cap = 0;
    float  *d_flt_buf = nullptr;   size_t flt_cap = 0;
    void   *d_tmp     = nullptr;   size_t tmp_cap = 0;

    GpuState()  = default;
    ~GpuState() {
        // TODO: cudaFree all device pointers; cufftDestroy; cufinufft_destroy;
        //       cudaStreamDestroy.
    }
};

// Allocate and initialise a GpuState with all physics params and CUDA objects.
GpuState *gpu_create_state(
    int nf, int n_digits,
    double L, double r_c, double sigma, double tol,
    double self_factor_d, float self_factor_f,
    dmk_eval_type eval_type,
    const double *h_scaling_coeffs)
{
    auto *gpu = new GpuState;
    gpu->nf            = nf;
    gpu->n_digits      = n_digits;
    gpu->L             = L;
    gpu->r_c           = r_c;
    gpu->sigma         = sigma;
    gpu->tol           = tol;
    gpu->self_factor_d = self_factor_d;
    gpu->self_factor_f = self_factor_f;
    gpu->eval_type     = eval_type;
    // TODO: cudaStreamCreate; cudaMalloc + cudaMemcpy for d_scaling_coeffs;
    //       cufftPlan3d for nf³ c2c; cufinufft_makeplan for type-1 and type-2.
    return gpu;
}

void gpu_destroy_state(GpuState *gpu) { delete gpu; }

// ---------------------------------------------------------------------------
// build_cell_list_gpu
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
    // TODO:
    // 1. kernel: compute flat cell index for each particle → d_cell_idx[n]
    // 2. Thrust sort_by_key on d_cell_idx, co-sort positions & charges
    // 3. transpose sorted AoS positions → SoA (d_xs, d_ys, d_zs)
    // 4. Thrust exclusive_scan on per-cell histogram → d_cell_start[nc³+1]
    // 5. record d_orig[n] (permutation before sort)
}

// ---------------------------------------------------------------------------
// short_range_kernel
// One CUDA block per home cell, one thread per target particle in that cell.
// ---------------------------------------------------------------------------
template <typename Real>
__global__ void short_range_kernel(
    int nc, int n, int out_dim, int n_digits,
    Real rsc, Real cen, Real r_c_sq,
    const int  *cell_start,
    const Real *d_xs, const Real *d_ys, const Real *d_zs,
    const Real *d_qs,
    const int  *nbc_tab,
    const Real *off_tab,
    Real       *pg_sorted)
{
    // TODO:
    // int home   = blockIdx.x;
    // int t      = threadIdx.x;
    // int hbeg   = cell_start[home], n_trg = cell_start[home+1] - hbeg;
    // if (t >= n_trg) return;
    // loop over 27 stencil cells → inner loop over sources
    // eval scalar polynomial kernel, accumulate pot and gradient
    // write pg_sorted[out_dim*(hbeg+t) + k]
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
    // TODO:
    // int a = blockIdx.x * blockDim.x + threadIdx.x;
    // if (a >= n) return;
    // int o = d_orig[a];
    // d_pot[o] += pg_sorted[out_dim*a + 0];
    // if (out_dim > 1) {
    //   d_fx[o] += -d_qs_sorted[a] * pg_sorted[out_dim*a + 1]; ...
    // }
}

// ---------------------------------------------------------------------------
// scaling_kernel
// ---------------------------------------------------------------------------
__global__ void scaling_kernel(
    int ntot,
    const cuDoubleComplex *b_hat,
    const double          *scaling_coeffs,
    cuDoubleComplex       *pot_hat)
{
    // TODO:
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i >= ntot) return;
    // pot_hat[i] = {b_hat[i].x * scaling_coeffs[i], b_hat[i].y * scaling_coeffs[i]};
}

// ---------------------------------------------------------------------------
// grad_scaling_kernel
// f_hat_x uses k_idx[iz] (axis swap: see esp.cpp:421–423)
// ---------------------------------------------------------------------------
__global__ void grad_scaling_kernel(
    int nf, double coeff_grad,
    const cuDoubleComplex *b_hat,
    const double          *scaling_coeffs,
    const int             *k_idx,
    cuDoubleComplex *f_hat_x,
    cuDoubleComplex *f_hat_y,
    cuDoubleComplex *f_hat_z)
{
    // TODO
}

// ---------------------------------------------------------------------------
// self_interaction_kernel
// ---------------------------------------------------------------------------
template <typename Real>
__global__ void self_interaction_kernel(int n, Real factor, const Real *d_charges, Real *d_pot)
{
    // TODO:
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i < n) d_pot[i] -= d_charges[i] * factor;
}

// ---------------------------------------------------------------------------
// long_range_gpu
// ---------------------------------------------------------------------------
template <typename Real>
static void long_range_gpu(
    GpuState &gpu,
    int n, double tol,
    const double *d_x, const double *d_y, const double *d_z,
    const cuDoubleComplex *d_c,
    double coeff_grad,
    bool want_force,
    Real *d_pot, Real *d_fx, Real *d_fy, Real *d_fz)
{
    // TODO:
    // 1. cufinufft_setpts + cufinufft_execute (type-1) → d_b [nf³ complex]
    // 2. cufftExecZ2Z(gpu.fft_plan, d_b, d_b_hat, CUFFT_FORWARD)
    // 3. scaling_kernel: d_b_hat * d_scaling_coeffs → d_pot_hat
    // 4. cufftExecZ2Z inverse + normalisation → d_grid_pot
    // 5. cufinufft_execute (type-2): d_grid_pot → d_pot_c; real part += into d_pot
    // if (want_force):
    // 6. grad_scaling_kernel: d_b_hat → d_f_hat_{x,y,z}
    // 7. 3× cufftExecZ2Z inverse + normalisation
    // 8. 3× cufinufft_execute (type-2) → forces; accumulate into d_fx/d_fy/d_fz
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
    // TODO:
    // compute nbc_tab[nc*3] and off_tab[nc*3] on the host (same logic as esp.cpp short_range),
    //   then cudaMemcpy them to device
    // build_cell_list_gpu(...)
    // launch short_range_kernel<<<nc³, blockDim, 0, gpu.stream>>>(...)
    // launch scatter_kernel<<<ceil(n/256), 256, 0, gpu.stream>>>(...)
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
// gpu_make_spans — resize + zero the host buffer, return four output spans.
// ---------------------------------------------------------------------------
template <typename Real>
static auto gpu_make_spans(GpuState *gpu, int n) {
    const bool want_force = (gpu->eval_type == DMK_POTENTIAL_GRAD);
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
    const int  n  = static_cast<int>(r_src.size());
    const int  nc = static_cast<int>(std::floor(gpu->L / gpu->r_c));
    const bool want_force = (gpu->eval_type == DMK_POTENTIAL_GRAD);
    auto [pot, fx, fy, fz] = gpu_make_spans<Real>(gpu, n);
    const Real *h_pos_aos = reinterpret_cast<const Real *>(r_src.data());

    // TODO:
    // 1. Ensure/resize gpu->d_[dbl|flt]_buf for slots*n; cudaMemset to 0 on stream.
    //    d_pot = buf + 0, d_fx = buf + n, d_fy = buf + 2n, d_fz = buf + 3n
    //
    // 2. cudaMemcpy h_pos_aos → d_pos_aos (n*3 Reals)
    //    cudaMemcpy charges.data() → d_charges (n Reals)
    //
    // 3. short_range_gpu(*gpu, d_pos_aos, d_charges, n, nc, gpu->L, Real(gpu->r_c),
    //                    gpu->n_digits, want_force, d_pot, d_fx, d_fy, d_fz)
    //
    // 4. [kernel] AoS → SoA scaled coords (2π/L); pack charges → cuDoubleComplex
    //    long_range_gpu(*gpu, n, gpu->tol, d_x, d_y, d_z, d_c,
    //                   /*coeff_grad*/ 2*M_PI/gpu->L, want_force,
    //                   d_pot, d_fx, d_fy, d_fz)
    //
    // 5. self_interaction_kernel<<<..., gpu->stream>>>(n, Real(gpu->self_factor_d), d_charges, d_pot)
    //
    // 6. cudaStreamSynchronize(gpu->stream)
    //
    // 7. cudaMemcpy d_pot → host_buf<Real>(gpu) (n Reals)
    //    if (want_force): cudaMemcpy d_fx/d_fy/d_fz similarly

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
    const int  n  = static_cast<int>(r_src.size());
    const int  nc = static_cast<int>(std::floor(gpu->L / gpu->r_c));
    const bool want_force = (gpu->eval_type == DMK_POTENTIAL_GRAD);
    auto [pot, fx, fy, fz] = gpu_make_spans<Real>(gpu, n);
    const Real *h_pos_aos = reinterpret_cast<const Real *>(r_src.data());

    // TODO:
    // 1. Ensure/resize device buffer; cudaMemset to 0.
    // 2. cudaMemcpy h_pos_aos → d_pos_aos; charges.data() → d_charges
    // 3. short_range_gpu(*gpu, d_pos_aos, d_charges, n, nc, gpu->L, Real(gpu->r_c),
    //                    gpu->n_digits, want_force, d_pot, d_fx, d_fy, d_fz)
    // 4. cudaStreamSynchronize; cudaMemcpy device → host_buf<Real>(gpu)

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
    const int  n  = static_cast<int>(r_src.size());
    const bool want_force = (gpu->eval_type == DMK_POTENTIAL_GRAD);
    auto [pot, fx, fy, fz] = gpu_make_spans<Real>(gpu, n);
    const Real *h_pos_aos = reinterpret_cast<const Real *>(r_src.data());

    // TODO:
    // 1. Ensure/resize device buffer; cudaMemset to 0.
    // 2. cudaMemcpy h_pos_aos → d_pos_aos; charges.data() → d_charges
    // 3. [kernel] AoS → SoA scaled coords (2π/L); pack charges → cuDoubleComplex
    //    long_range_gpu(*gpu, n, gpu->tol, d_x, d_y, d_z, d_c,
    //                   2*M_PI/gpu->L, want_force, d_pot, d_fx, d_fy, d_fz)
    // 4. cudaStreamSynchronize; cudaMemcpy device → host_buf<Real>(gpu)

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
