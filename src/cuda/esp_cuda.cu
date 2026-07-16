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
    cuDoubleComplex *d_b             = nullptr;    // nf³ complex — spread output (NU → uniform)
    cuDoubleComplex *d_b_hat         = nullptr;    // nf³ complex — FFT of d_b (k-space)
    cufftHandle    fft_plan{};                     // nf³ 3-D c2c, created at plan time
    bool           fft_plan_valid    = false;
    cufinufft_plan cfnufft_plan_1    = nullptr;    // type-1 spread-only (NU → uniform)
    cufinufft_plan cfnufft_plan_2    = nullptr;    // type-2 interp-only (uniform → NU)

    // Per-eval device scratch; grown as needed, never shrunk.
    double *d_dbl_buf = nullptr;   size_t dbl_cap = 0;
    float  *d_flt_buf = nullptr;   size_t flt_cap = 0;
    void   *d_tmp     = nullptr;   size_t tmp_cap = 0;

    GpuState()  = default;
    ~GpuState() {
        if (cfnufft_plan_1)   cufinufft_destroy(cfnufft_plan_1);
        if (cfnufft_plan_2)   cufinufft_destroy(cfnufft_plan_2);
        if (fft_plan_valid)   cufftDestroy(fft_plan);
        if (d_b_hat)          cudaFree(d_b_hat);
        if (d_b)              cudaFree(d_b);
        if (d_scaling_coeffs)  cudaFree(d_scaling_coeffs);
        if (d_dbl_buf)         cudaFree(d_dbl_buf);
        if (d_flt_buf)         cudaFree(d_flt_buf);
        if (d_tmp)             cudaFree(d_tmp);
        if (stream)            cudaStreamDestroy(stream);
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

    // Dedicated stream for all GPU work on this plan.
    if (cudaStreamCreate(&gpu->stream) != cudaSuccess)
        throw std::runtime_error("GpuState: cudaStreamCreate failed");

    // Upload precomputed scaling coefficients (plan-level constant, nf³ doubles).
    const long long ntot = (long long)nf * nf * nf;
    if (cudaMalloc(&gpu->d_scaling_coeffs, ntot * sizeof(double)) != cudaSuccess)
        throw std::runtime_error("GpuState: cudaMalloc d_scaling_coeffs failed");
    cudaMemcpyAsync(gpu->d_scaling_coeffs, h_scaling_coeffs,
                    ntot * sizeof(double), cudaMemcpyHostToDevice, gpu->stream);

    // Spread output buffer — always nf³ complex doubles, so allocate once at plan time.
    if (cudaMalloc(&gpu->d_b, ntot * sizeof(cuDoubleComplex)) != cudaSuccess)
        throw std::runtime_error("GpuState: cudaMalloc d_b failed");

    // Forward 3-D c2c FFT plan: nf × nf × nf.  Created once; reused every eval.
    if (cufftPlan3d(&gpu->fft_plan, nf, nf, nf, CUFFT_Z2Z) != CUFFT_SUCCESS)
        throw std::runtime_error("GpuState: cufftPlan3d failed");
    cufftSetStream(gpu->fft_plan, gpu->stream);
    gpu->fft_plan_valid = true;

    // FFT output buffer: nf³ complex doubles (k-space).
    if (cudaMalloc(&gpu->d_b_hat, ntot * sizeof(cuDoubleComplex)) != cudaSuccess)
        throw std::runtime_error("GpuState: cudaMalloc d_b_hat failed");

    // cuFINUFFT plans — created once per plan (makeplan does not bind to n).
    // Per eval: call setpts (binds NU points) then execute.
    cufinufft_opts co;
    cufinufft_default_opts(&co);
    co.gpu_spreadinterponly = 1;
    co.upsampfac            = sigma;
    co.gpu_kerevalmeth      = 0;  // direct exp(sqrt()) — supports non-standard upsampfac (e.g. 1.35)
    // Use the default (null) stream for cuFINUFFT internals.
    // We sync explicitly before setpts/execute so the NU data is ready.
    // co.gpu_stream is left at cudaStreamDefault (the default from cufinufft_default_opts).
    const int64_t nmodes[3] = {nf, nf, nf};

    int ier = cufinufft_makeplan(/*type=*/1, /*dim=*/3, nmodes,
                                 /*iflag=*/+1, /*ntransf=*/1, tol,
                                 &gpu->cfnufft_plan_1, &co);
    if (ier != 0) throw std::runtime_error("GpuState: cufinufft_makeplan type-1 failed, ier=" + std::to_string(ier));

    ier = cufinufft_makeplan(/*type=*/2, /*dim=*/3, nmodes,
                             /*iflag=*/-1, /*ntransf=*/1, tol,
                             &gpu->cfnufft_plan_2, &co);
    if (ier != 0) throw std::runtime_error("GpuState: cufinufft_makeplan type-2 failed, ier=" + std::to_string(ier));

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
// scaling_kernel — multiply each k-space grid point by its scaling coefficient.
// Operates in-place: d_b_hat[i] *= scaling_coeffs[i], producing pot_hat.
// Mirrors CPU: pot_hat[i] = b_hat[i] * scaling_coeffs[i].
// ---------------------------------------------------------------------------
__global__ void scaling_kernel(
    int ntot,
    const double    *scaling_coeffs,
    cuDoubleComplex *b_hat)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntot) return;
    double s = scaling_coeffs[i];
    b_hat[i] = {b_hat[i].x * s, b_hat[i].y * s}; //.x = real part; .y = imaginary part
}

// ---------------------------------------------------------------------------
// normalize_kernel — divide every element by ntot after cuFFT IFFT.
// cuFFT's inverse transform is unnormalized (output = ntot * true IFFT).
// Mirrors CPU: ifftn_3d divides by ntot internally.
// ---------------------------------------------------------------------------
__global__ void normalize_kernel(int ntot, double inv_ntot, cuDoubleComplex *data)
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
__global__ void extract_real_kernel(int n, const cuDoubleComplex *d_c, Real *d_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_out[i] = Real(d_c[i].x);
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

__global__ void grad_scaling_kernel(
    int nf, double coeff_grad,
    const cuDoubleComplex *pot_hat,
    cuDoubleComplex *f_hat_x,
    cuDoubleComplex *f_hat_y,
    cuDoubleComplex *f_hat_z)
{
    const long long ntot = (long long)nf * nf * nf;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntot) return;

    const int iz = int(i % nf);
    const int iy = int((i / nf) % nf);
    const int ix = int(i / ((long long)nf * nf));

    const cuDoubleComplex s = pot_hat[i];
    auto mul_ik = [=](int k) {
        const double factor = coeff_grad * double(k);
        // s * (i * factor) = (-s.y*factor, s.x*factor)
        return cuDoubleComplex{-s.y * factor, s.x * factor};
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
__global__ void accumulate_force_kernel(int n, const cuDoubleComplex *d_c, const cuDoubleComplex *d_force_c,
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
// long_range_gpu
// Mirrors the CPU long_range(): spread → FFT → scale → IFFT → interp (+ forces).
// Always operates in double precision regardless of Real.
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
    const long long ntot = (long long)gpu.nf * gpu.nf * gpu.nf;

    // -----------------------------------------------------------------------
    // Step 1: Spread — NU points → uniform grid  (gpu.d_b, nf³ complex)
    // Mirrors CPU: finufft3d1 with opts.spreadinterponly=1.
    // Uses the pre-created gpu.cfnufft_plan_1 (type-1, makeplan done at plan time).
    // -----------------------------------------------------------------------
    {
        // cuFINUFFT plans use the default stream; our H2D copies run on gpu.stream.
        // Sync the device so both streams see consistent data before setpts.
        cudaError_t cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess)
            throw std::runtime_error(
                std::string("long_range_gpu: pre-setpts sync failed: ") + cudaGetErrorString(cerr));

        int ier = cufinufft_setpts(gpu.cfnufft_plan_1, n,
                                   const_cast<double *>(d_x),
                                   const_cast<double *>(d_y),
                                   const_cast<double *>(d_z),
                                   0, nullptr, nullptr, nullptr);
        if (ier != 0) {
            cudaError_t last = cudaGetLastError();
            throw std::runtime_error(
                "long_range_gpu: cufinufft_setpts spread failed, ier=" + std::to_string(ier) +
                ", last CUDA error: " + cudaGetErrorString(last));
        }

        // Zero before spreading — cuFINUFFT accumulates into the output buffer.
        cudaMemsetAsync(gpu.d_b, 0, ntot * sizeof(cuDoubleComplex), gpu.stream);

        ier = cufinufft_execute(gpu.cfnufft_plan_1, const_cast<cuDoubleComplex *>(d_c), gpu.d_b);
        if (ier != 0)
            throw std::runtime_error("long_range_gpu: cufinufft_execute spread failed, ier=" + std::to_string(ier));
    }

    // -----------------------------------------------------------------------
    // Step 2: Forward FFT — gpu.d_b → gpu.d_b_hat  (real-space grid → k-space)
    // Mirrors CPU: fftn_3d(b, b_hat, nf).
    // -----------------------------------------------------------------------
    {
        cufftResult r = cufftExecZ2Z(gpu.fft_plan, gpu.d_b, gpu.d_b_hat, CUFFT_FORWARD);
        if (r != CUFFT_SUCCESS)
            throw std::runtime_error("long_range_gpu: cufftExecZ2Z forward failed, err=" + std::to_string(r));
    }

    // -----------------------------------------------------------------------
    // Step 3: Scale — d_b_hat[i] *= scaling_coeffs[i]  (d_b_hat is now pot_hat)
    // Mirrors CPU: pot_hat[i] = b_hat[i] * scaling_coeffs[i].
    // In-place: no extra buffer needed; d_b_hat is reused as pot_hat for IFFT.
    // -----------------------------------------------------------------------
    {
        const int threads = 256;
        const int blocks  = static_cast<int>((ntot + threads - 1) / threads);
        scaling_kernel<<<blocks, threads, 0, gpu.stream>>>(
            static_cast<int>(ntot), gpu.d_scaling_coeffs, gpu.d_b_hat);
    }
    // -----------------------------------------------------------------------
    // Step 4: Inverse FFT — pot_hat (d_b_hat) → d_grid_pot.
    // Reuses d_b as d_grid_pot (spread output no longer needed after step 1).
    // cuFFT IFFT is unnormalized, so follow with normalize_kernel (÷ ntot).
    // Mirrors CPU: ifftn_3d(pot_hat, grid_pot, nf).
    // -----------------------------------------------------------------------
    {
        cufftResult r = cufftExecZ2Z(gpu.fft_plan, gpu.d_b_hat, gpu.d_b, CUFFT_INVERSE);
        if (r != CUFFT_SUCCESS)
            throw std::runtime_error("long_range_gpu: cufftExecZ2Z inverse failed, err=" + std::to_string(r));

        const double inv_ntot = 1.0 / double(ntot);
        const int threads = 256;
        const int blocks  = static_cast<int>((ntot + threads - 1) / threads);
        normalize_kernel<<<blocks, threads, 0, gpu.stream>>>(
            static_cast<int>(ntot), inv_ntot, gpu.d_b);
    }
    // gpu.cfnufft_plan_2's NU points don't change within a single long_range_gpu
    // call (same d_x/d_y/d_z throughout), so setpts is bound once here and every
    // type-2 execute below (potential and, if requested, the 3 force components)
    // reuses that binding.
    const bool want_pot_interp   = (d_pot != nullptr);
    const bool want_force_interp = want_force && (d_fx || d_fy || d_fz);
    if (want_pot_interp || want_force_interp) {
        int ier = cufinufft_setpts(gpu.cfnufft_plan_2, n,
                                   const_cast<double *>(d_x),
                                   const_cast<double *>(d_y),
                                   const_cast<double *>(d_z),
                                   0, nullptr, nullptr, nullptr);
        if (ier != 0)
            throw std::runtime_error("long_range_gpu: cufinufft_setpts interp failed, ier=" + std::to_string(ier));
    }

    // -----------------------------------------------------------------------
    // Step 5: Interp (cuFINUFFT type-2, spreadinterponly=1)
    // d_grid_pot (gpu.d_b) → d_pot_c (n complex values at NU points) → d_pot.
    // Mirrors CPU: finufft3d2 with spreadinterponly=1, then pot[j] += real(c[j]).
    // Uses the pre-created gpu.cfnufft_plan_2 (type-2, makeplan done at plan time).
    // -----------------------------------------------------------------------
    if (want_pot_interp) {
        cuDoubleComplex *d_pot_c;
        if (cudaMalloc(&d_pot_c, n * sizeof(cuDoubleComplex)) != cudaSuccess)
            throw std::runtime_error("long_range_gpu: cudaMalloc d_pot_c failed");

        // gpu.d_b is d_grid_pot (after step 4). Execute: uniform grid → NU values.
        int ier = cufinufft_execute(gpu.cfnufft_plan_2, d_pot_c, gpu.d_b);
        if (ier != 0) { cudaFree(d_pot_c); throw std::runtime_error("long_range_gpu: cufinufft_execute interp failed, ier=" + std::to_string(ier)); }

        const int threads = 256;
        const int blocks  = (n + threads - 1) / threads;
        extract_real_kernel<<<blocks, threads, 0, gpu.stream>>>(n, d_pot_c, d_pot);
        cudaFree(d_pot_c);
    }

    // -----------------------------------------------------------------------
    // Steps 6-8: force path (ik method).
    // gpu.d_b_hat still holds pot_hat (= b_hat*scaling_coeffs from step 3 — step
    // 4's IFFT read it without mutating it), so it's used directly here.
    // Mirrors CPU long_range()'s force block (esp.cpp).
    // -----------------------------------------------------------------------
    if (want_force_interp) {
        cuDoubleComplex *f_hat_x, *f_hat_y, *f_hat_z;
        if (cudaMalloc(&f_hat_x, ntot * sizeof(cuDoubleComplex)) != cudaSuccess ||
            cudaMalloc(&f_hat_y, ntot * sizeof(cuDoubleComplex)) != cudaSuccess ||
            cudaMalloc(&f_hat_z, ntot * sizeof(cuDoubleComplex)) != cudaSuccess)
            throw std::runtime_error("long_range_gpu: cudaMalloc f_hat_{x,y,z} failed");

        // Step 6: build the three force spectra from pot_hat.
        {
            const int threads = 256;
            const int blocks  = static_cast<int>((ntot + threads - 1) / threads);
            grad_scaling_kernel<<<blocks, threads, 0, gpu.stream>>>(
                gpu.nf, coeff_grad, gpu.d_b_hat, f_hat_x, f_hat_y, f_hat_z);
        }

        // Step 7: inverse FFT each component in-place, then normalize (cuFFT's
        // IFFT is unnormalized, mirrors normalize_kernel usage in step 4).
        auto ifft_and_normalize = [&](cuDoubleComplex *buf) {
            cufftResult r = cufftExecZ2Z(gpu.fft_plan, buf, buf, CUFFT_INVERSE);
            if (r != CUFFT_SUCCESS)
                throw std::runtime_error("long_range_gpu: cufftExecZ2Z inverse (force) failed, err=" +
                                         std::to_string(r));
            const double inv_ntot = 1.0 / double(ntot);
            const int threads = 256;
            const int blocks  = static_cast<int>((ntot + threads - 1) / threads);
            normalize_kernel<<<blocks, threads, 0, gpu.stream>>>(static_cast<int>(ntot), inv_ntot, buf);
        };
        ifft_and_normalize(f_hat_x);
        ifft_and_normalize(f_hat_y);
        ifft_and_normalize(f_hat_z);

        // Step 8: interp (cuFINUFFT type-2, reusing the setpts binding above)
        // each grid_force component → NU points, then accumulate
        // d_f{x,y,z}[j] += -charge[j]*real(force_c[j]).
        auto interp_and_accumulate = [&](cuDoubleComplex *grid_force, Real *d_force_out) {
            if (!d_force_out) return;
            cuDoubleComplex *d_force_c;
            if (cudaMalloc(&d_force_c, n * sizeof(cuDoubleComplex)) != cudaSuccess)
                throw std::runtime_error("long_range_gpu: cudaMalloc d_force_c failed");

            int ier = cufinufft_execute(gpu.cfnufft_plan_2, d_force_c, grid_force);
            if (ier != 0) {
                cudaFree(d_force_c);
                throw std::runtime_error("long_range_gpu: cufinufft_execute force-interp failed, ier=" +
                                         std::to_string(ier));
            }

            const int threads = 256;
            const int blocks  = (n + threads - 1) / threads;
            accumulate_force_kernel<Real><<<blocks, threads, 0, gpu.stream>>>(n, d_c, d_force_c, d_force_out);
            cudaFree(d_force_c);
        };
        interp_and_accumulate(f_hat_x, d_fx);
        interp_and_accumulate(f_hat_y, d_fy);
        interp_and_accumulate(f_hat_z, d_fz);

        cudaFree(f_hat_x);
        cudaFree(f_hat_y);
        cudaFree(f_hat_z);
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
    const int  n  = static_cast<int>(r_src.size());
    const int  nc = static_cast<int>(std::floor(gpu->L / gpu->r_c));
    [[maybe_unused]] const bool want_force = (gpu->eval_type == DMK_POTENTIAL_GRAD);
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
    [[maybe_unused]] const bool want_force = (gpu->eval_type == DMK_POTENTIAL_GRAD);
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
    [[maybe_unused]] const bool want_force = (gpu->eval_type == DMK_POTENTIAL_GRAD);
    auto [pot, fx, fy, fz] = gpu_make_spans<Real>(gpu, n);

    // Host-side AoS → SoA; scale coords to [-π, π); pack charges as complex.
    const double scale = 2.0 * M_PI / gpu->L;
    std::vector<double> h_x(n), h_y(n), h_z(n);
    std::vector<cuDoubleComplex> h_c(n);
    for (int j = 0; j < n; ++j) {
        h_x[j] = double(r_src[j][0]) * scale;
        h_y[j] = double(r_src[j][1]) * scale;
        h_z[j] = double(r_src[j][2]) * scale;
        h_c[j] = {double(charges[j]), 0.0};
    }

    // Upload coords and charges to device.
    double *d_x, *d_y, *d_z; cuDoubleComplex *d_c;
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_y, n * sizeof(double));
    cudaMalloc(&d_z, n * sizeof(double));
    cudaMalloc(&d_c, n * sizeof(cuDoubleComplex));
    cudaMemcpyAsync(d_x, h_x.data(), n * sizeof(double), cudaMemcpyHostToDevice, gpu->stream);
    cudaMemcpyAsync(d_y, h_y.data(), n * sizeof(double), cudaMemcpyHostToDevice, gpu->stream);
    cudaMemcpyAsync(d_z, h_z.data(), n * sizeof(double), cudaMemcpyHostToDevice, gpu->stream);
    cudaMemcpyAsync(d_c, h_c.data(), n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, gpu->stream);

    // Device output buffers for potential and (if requested) forces.
    Real *d_pot;
    cudaMalloc(&d_pot, n * sizeof(Real));
    cudaMemsetAsync(d_pot, 0, n * sizeof(Real), gpu->stream);

    Real *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
    if (want_force) {
        cudaMalloc(&d_fx, n * sizeof(Real));
        cudaMalloc(&d_fy, n * sizeof(Real));
        cudaMalloc(&d_fz, n * sizeof(Real));
        cudaMemsetAsync(d_fx, 0, n * sizeof(Real), gpu->stream);
        cudaMemsetAsync(d_fy, 0, n * sizeof(Real), gpu->stream);
        cudaMemsetAsync(d_fz, 0, n * sizeof(Real), gpu->stream);
    }

    long_range_gpu<Real>(*gpu, n, gpu->tol,
                         d_x, d_y, d_z, d_c,
                         scale, want_force,
                         d_pot, d_fx, d_fy, d_fz);

    cudaStreamSynchronize(gpu->stream);
    cudaMemcpy(pot.data(), d_pot, n * sizeof(Real), cudaMemcpyDeviceToHost);
    if (want_force) {
        cudaMemcpy(fx.data(), d_fx, n * sizeof(Real), cudaMemcpyDeviceToHost);
        cudaMemcpy(fy.data(), d_fy, n * sizeof(Real), cudaMemcpyDeviceToHost);
        cudaMemcpy(fz.data(), d_fz, n * sizeof(Real), cudaMemcpyDeviceToHost);
        cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
    }

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_c); cudaFree(d_pot);
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
