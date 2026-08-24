#include "short_range.hpp"
#include "state.hpp"
#include "support.hpp"

#include <dmk.h>
#include <dmk/cuda/esp_gpu.hpp>
#include <dmk/esp.hpp>

#include <cufinufft.h>
#include <dmk/nvtx_wrapper.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace dmk {

struct NvtxRange {
    explicit NvtxRange(const char *name) { nvtxRangePush(name); }
    ~NvtxRange() { nvtxRangePop(); }
};

GpuState *gpu_create_state(int nf, int n_digits, double L, double r_c, double gpu_upsampfac, double tol, double beta,
                           double self_factor, dmk_eval_type eval_type, bool use_float, GpuSrStrategy strategy,
                           GpuSortMode sort_mode, const double *h_scaling_coeffs) {
    auto *gpu = new GpuState;
    gpu->nf = nf;
    gpu->n_digits = n_digits;
    gpu->L = L;
    gpu->r_c = r_c;
    gpu->beta = beta;
    gpu->self_factor = self_factor;
    gpu->eval_type = eval_type;
    gpu->use_float = use_float;
    gpu->strategy = strategy;
    gpu->sort_mode = sort_mode;

    const size_t real_sz = use_float ? sizeof(float) : sizeof(double);
    const size_t complex_sz = use_float ? sizeof(cuFloatComplex) : sizeof(cuDoubleComplex);

    if (cudaStreamCreate(&gpu->stream) != cudaSuccess)
        throw std::runtime_error("GpuState: cudaStreamCreate failed");

    const long long ntot = (long long)nf * nf * nf;
    if (cudaMalloc(&gpu->d_scaling_coeffs, ntot * real_sz) != cudaSuccess)
        throw std::runtime_error("GpuState: cudaMalloc d_scaling_coeffs failed");
    if (use_float) {
        std::vector<float> h_scaling_coeffs_f(ntot);
        for (long long i = 0; i < ntot; ++i)
            h_scaling_coeffs_f[i] = float(h_scaling_coeffs[i]);
        cudaMemcpy(gpu->d_scaling_coeffs, h_scaling_coeffs_f.data(), ntot * real_sz, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpyAsync(gpu->d_scaling_coeffs, h_scaling_coeffs, ntot * real_sz, cudaMemcpyHostToDevice, gpu->stream);
    }

    if (cudaMalloc(&gpu->d_b, ntot * complex_sz) != cudaSuccess)
        throw std::runtime_error("GpuState: cudaMalloc d_b failed");

    if (cufftPlan3d(&gpu->fft_plan, nf, nf, nf, use_float ? CUFFT_C2C : CUFFT_Z2Z) != CUFFT_SUCCESS)
        throw std::runtime_error("GpuState: cufftPlan3d failed");
    cufftSetStream(gpu->fft_plan, gpu->stream);
    gpu->fft_plan_valid = true;

    if (cudaMalloc(&gpu->d_b_hat, ntot * complex_sz) != cudaSuccess)
        throw std::runtime_error("GpuState: cudaMalloc d_b_hat failed");

    // makeplan does not bind to n, so these are created once; each eval does setpts then execute.
    cufinufft_opts co;
    cufinufft_default_opts(&co);
    co.gpu_spreadinterponly = 1;
    co.upsampfac = gpu_upsampfac;
    co.gpu_kerevalmeth = 1;
    co.gpu_method = 3;

    // cuFINUFFT stays on the default stream; we sync explicitly before setpts/execute.
    const int64_t nmodes[3] = {nf, nf, nf};

    int ier;
    if (use_float) {
        cufinufftf_plan p1 = nullptr, p2 = nullptr;
        ier = cufinufftf_makeplan(/*type=*/1, /*dim=*/3, nmodes,
                                  /*iflag=*/+1, /*ntransf=*/1, float(tol), &p1, &co);
        if (ier != 0)
            throw std::runtime_error("GpuState: cufinufftf_makeplan type-1 failed, ier=" + std::to_string(ier));
        co.gpu_method = 1;
        ier = cufinufftf_makeplan(/*type=*/2, /*dim=*/3, nmodes,
                                  /*iflag=*/-1, /*ntransf=*/1, float(tol), &p2, &co);
        if (ier != 0)
            throw std::runtime_error("GpuState: cufinufftf_makeplan type-2 failed, ier=" + std::to_string(ier));
        gpu->cfnufft_plan_1 = p1;
        gpu->cfnufft_plan_2 = p2;
    } else {
        cufinufft_plan p1 = nullptr, p2 = nullptr;
        ier = cufinufft_makeplan(/*type=*/1, /*dim=*/3, nmodes,
                                 /*iflag=*/+1, /*ntransf=*/1, tol, &p1, &co);
        if (ier != 0)
            throw std::runtime_error("GpuState: cufinufft_makeplan type-1 failed, ier=" + std::to_string(ier));
        co.gpu_method = 1;
        ier = cufinufft_makeplan(/*type=*/2, /*dim=*/3, nmodes,
                                 /*iflag=*/-1, /*ntransf=*/1, tol, &p2, &co);
        if (ier != 0)
            throw std::runtime_error("GpuState: cufinufft_makeplan type-2 failed, ier=" + std::to_string(ier));
        gpu->cfnufft_plan_1 = p1;
        gpu->cfnufft_plan_2 = p2;
    }

    // 27-cell-stencil neighbour tables; depend only on nc, not on particle data.
    gpu->nc = static_cast<int>(std::floor(L / r_c));
    if (gpu->nc < 3)
        throw std::runtime_error("GpuState: short_range_gpu requires r_c <= L/3 (nc >= 3)");
    {
        const int ntab = gpu->nc * 3;
        std::vector<int> h_nbc_tab(ntab);
        std::vector<double> h_off_tab(ntab);
        for (int c = 0; c < gpu->nc; ++c) {
            for (int d = 0; d < 3; ++d) {
                int ci = c + d - 1;
                if (ci < 0) {
                    h_nbc_tab[c * 3 + d] = ci + gpu->nc;
                    h_off_tab[c * 3 + d] = -L;
                } else if (ci >= gpu->nc) {
                    h_nbc_tab[c * 3 + d] = ci - gpu->nc;
                    h_off_tab[c * 3 + d] = L;
                } else {
                    h_nbc_tab[c * 3 + d] = ci;
                    h_off_tab[c * 3 + d] = 0.0;
                }
            }
        }
        if (cudaMalloc(&gpu->d_nbc_tab, ntab * sizeof(int)) != cudaSuccess)
            throw std::runtime_error("GpuState: cudaMalloc d_nbc_tab failed");
        if (cudaMalloc(&gpu->d_off_tab, ntab * real_sz) != cudaSuccess)
            throw std::runtime_error("GpuState: cudaMalloc d_off_tab failed");
        // Blocking: the host tables are locals about to go out of scope.
        cudaMemcpy(gpu->d_nbc_tab, h_nbc_tab.data(), ntab * sizeof(int), cudaMemcpyHostToDevice);
        if (use_float) {
            std::vector<float> h_off_tab_f(ntab);
            for (int i = 0; i < ntab; ++i)
                h_off_tab_f[i] = float(h_off_tab[i]);
            cudaMemcpy(gpu->d_off_tab, h_off_tab_f.data(), ntab * real_sz, cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(gpu->d_off_tab, h_off_tab.data(), ntab * real_sz, cudaMemcpyHostToDevice);
        }
    }

    // Cell-list CSR boundaries: size is fixed by nc, so allocate once and overwrite each call.
    {
        const long long ncells = (long long)gpu->nc * gpu->nc * gpu->nc;
        if (cudaMalloc(&gpu->d_cell_start, (ncells + 1) * sizeof(int)) != cudaSuccess)
            throw std::runtime_error("GpuState: cudaMalloc d_cell_start failed");
    }

    // Force spectra, only when this plan computes forces.
    if (eval_type >= DMK_POTENTIAL_GRAD) {
        if (cudaMalloc(&gpu->d_fhat_x, ntot * complex_sz) != cudaSuccess ||
            cudaMalloc(&gpu->d_fhat_y, ntot * complex_sz) != cudaSuccess ||
            cudaMalloc(&gpu->d_fhat_z, ntot * complex_sz) != cudaSuccess)
            throw std::runtime_error("GpuState: cudaMalloc d_fhat_{x,y,z} failed");
    }

    return gpu;
}

void gpu_destroy_state(GpuState *gpu) { delete gpu; }

// Real-dispatched wrappers: cufinufft_plan and cufinufftf_plan are distinct C types, not overloads,
// so GpuState holds whichever as void* and these cast it back.
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

// spread -> FFT -> scale -> IFFT -> interp (+ forces), mirroring the CPU long_range().
template <typename Real>
static void long_range_gpu(GpuState &gpu, int n, const Real *d_x, const Real *d_y, const Real *d_z,
                           const ComplexT<Real> *d_c, Real coeff_grad, bool want_force, Real *d_pot, Real *d_fx,
                           Real *d_fy, Real *d_fz) {
    const long long ntot = (long long)gpu.nf * gpu.nf * gpu.nf;
    auto *d_b = reinterpret_cast<ComplexT<Real> *>(gpu.d_b);
    auto *d_b_hat = reinterpret_cast<ComplexT<Real> *>(gpu.d_b_hat);
    auto *d_scaling_coeffs = reinterpret_cast<Real *>(gpu.d_scaling_coeffs);

    // Step 1: spread NU points -> uniform grid
    {
        NvtxRange range("long_range/spread");
        // cuFINUFFT is on the default stream, our copies on gpu.stream: sync before setpts.
        cudaError_t cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess)
            throw std::runtime_error(std::string("long_range_gpu: pre-setpts sync failed: ") +
                                     cudaGetErrorString(cerr));

        int ier = cufinufft_setpts_t<Real>(gpu.cfnufft_plan_1, n, const_cast<Real *>(d_x), const_cast<Real *>(d_y),
                                           const_cast<Real *>(d_z));
        if (ier != 0) {
            cudaError_t last = cudaGetLastError();
            throw std::runtime_error("long_range_gpu: cufinufft_setpts spread failed, ier=" + std::to_string(ier) +
                                     ", last CUDA error: " + cudaGetErrorString(last));
        }

        // Zero before spreading — cuFINUFFT accumulates into the output buffer.
        cudaMemsetAsync(d_b, 0, ntot * sizeof(ComplexT<Real>), gpu.stream);

        ier = cufinufft_execute_t<Real>(gpu.cfnufft_plan_1, const_cast<ComplexT<Real> *>(d_c), d_b);
        if (ier != 0)
            throw std::runtime_error("long_range_gpu: cufinufft_execute spread failed, ier=" + std::to_string(ier));
    }

    // Step 2: forward FFT
    {
        NvtxRange range("long_range/fft_forward");
        cufftResult r = cufft_exec_c2c_t<Real>(gpu.fft_plan, d_b, d_b_hat, CUFFT_FORWARD);
        if (r != CUFFT_SUCCESS)
            throw std::runtime_error("long_range_gpu: cufft forward failed, err=" + std::to_string(r));
    }

    // Step 3: scale in place -- d_b_hat becomes pot_hat
    {
        NvtxRange range("long_range/scale");
        cuda::EspSupportArgs<Real, ComplexT<Real>> a;
        a.ntot = static_cast<int>(ntot);
        a.scaling_coeffs = d_scaling_coeffs;
        a.grid = d_b_hat;
        cuda::esp::launch_stage<Real>(cuda::esp::kEspStageScaling, static_cast<int>(ntot), a, gpu.stream);
    }
    // Step 4: inverse FFT into d_b (the spread output is dead), then normalize
    {
        NvtxRange range("long_range/fft_inverse_normalize");
        cufftResult r = cufft_exec_c2c_t<Real>(gpu.fft_plan, d_b_hat, d_b, CUFFT_INVERSE);
        if (r != CUFFT_SUCCESS)
            throw std::runtime_error("long_range_gpu: cufft inverse failed, err=" + std::to_string(r));

        cuda::EspSupportArgs<Real, ComplexT<Real>> a;
        a.ntot = static_cast<int>(ntot);
        a.inv_ntot = Real(1) / Real(ntot);
        a.grid = d_b;
        cuda::esp::launch_stage<Real>(cuda::esp::kEspStageNormalize, static_cast<int>(ntot), a, gpu.stream);
    }
    // The NU points do not change within one call, so bind setpts once for every type-2 execute.
    const bool want_pot_interp = (d_pot != nullptr);
    const bool want_force_interp = want_force && (d_fx || d_fy || d_fz);
    if (want_pot_interp || want_force_interp) {
        NvtxRange range("long_range/interp_setpts");
        int ier = cufinufft_setpts_t<Real>(gpu.cfnufft_plan_2, n, const_cast<Real *>(d_x), const_cast<Real *>(d_y),
                                           const_cast<Real *>(d_z));
        if (ier != 0)
            throw std::runtime_error("long_range_gpu: cufinufft_setpts interp failed, ier=" + std::to_string(ier));
    }

    // Step 5: interp back to NU points
    if (want_pot_interp) {
        NvtxRange range("long_range/interp_potential");
        // Shared with the force components below: pot_c is consumed before they start.
        ensure_capacity(gpu.d_scratch_nu_c, gpu.scratch_nu_c_cap, std::size_t(n) * sizeof(ComplexT<Real>));
        auto *d_pot_c = reinterpret_cast<ComplexT<Real> *>(gpu.d_scratch_nu_c);

        int ier = cufinufft_execute_t<Real>(gpu.cfnufft_plan_2, d_pot_c, d_b);
        if (ier != 0)
            throw std::runtime_error("long_range_gpu: cufinufft_execute interp failed, ier=" + std::to_string(ier));

        cuda::EspSupportArgs<Real, ComplexT<Real>> a;
        a.n = n;
        a.c = d_pot_c;
        a.out = d_pot;
        cuda::esp::launch_stage<Real>(cuda::esp::kEspStageExtractReal, n, a, gpu.stream);
    }

    // Steps 6-8: force path (ik method). d_b_hat still holds pot_hat -- step 4's IFFT read it
    // without mutating it.
    if (want_force_interp) {
        NvtxRange force_path_range("long_range/force_path");
        auto *f_hat_x = reinterpret_cast<ComplexT<Real> *>(gpu.d_fhat_x);
        auto *f_hat_y = reinterpret_cast<ComplexT<Real> *>(gpu.d_fhat_y);
        auto *f_hat_z = reinterpret_cast<ComplexT<Real> *>(gpu.d_fhat_z);

        // Step 6: build the three force spectra from pot_hat.
        {
            NvtxRange range("long_range/force_grad_scaling");
            cuda::EspSupportArgs<Real, ComplexT<Real>> a;
            a.nf = gpu.nf;
            a.coeff_grad = coeff_grad;
            a.pot_hat = d_b_hat;
            a.f_hat_x = f_hat_x;
            a.f_hat_y = f_hat_y;
            a.f_hat_z = f_hat_z;
            cuda::esp::launch_stage<Real>(cuda::esp::kEspStageGradScaling, static_cast<int>(ntot), a, gpu.stream);
        }

        // Step 7: inverse FFT each component in place, then normalize.
        auto ifft_and_normalize = [&](ComplexT<Real> *buf) {
            NvtxRange range("long_range/force_ifft_normalize");
            cufftResult r = cufft_exec_c2c_t<Real>(gpu.fft_plan, buf, buf, CUFFT_INVERSE);
            if (r != CUFFT_SUCCESS)
                throw std::runtime_error("long_range_gpu: cufft inverse (force) failed, err=" + std::to_string(r));
            cuda::EspSupportArgs<Real, ComplexT<Real>> a;
            a.ntot = static_cast<int>(ntot);
            a.inv_ntot = Real(1) / Real(ntot);
            a.grid = buf;
            cuda::esp::launch_stage<Real>(cuda::esp::kEspStageNormalize, static_cast<int>(ntot), a, gpu.stream);
        };
        ifft_and_normalize(f_hat_x);
        ifft_and_normalize(f_hat_y);
        ifft_and_normalize(f_hat_z);

        // Step 8: interp each component and accumulate. One scratch buffer serves all three,
        // each fully consumed before the next.
        ensure_capacity(gpu.d_scratch_nu_c, gpu.scratch_nu_c_cap, std::size_t(n) * sizeof(ComplexT<Real>));
        auto *d_force_c = reinterpret_cast<ComplexT<Real> *>(gpu.d_scratch_nu_c);

        auto interp_and_accumulate = [&](ComplexT<Real> *grid_force, Real *d_force_out) {
            if (!d_force_out)
                return;
            NvtxRange range("long_range/force_interp_accumulate");
            int ier = cufinufft_execute_t<Real>(gpu.cfnufft_plan_2, d_force_c, grid_force);
            if (ier != 0)
                throw std::runtime_error("long_range_gpu: cufinufft_execute force-interp failed, ier=" +
                                         std::to_string(ier));

            const int threads = 256;
            const int blocks = (n + threads - 1) / threads;
            cuda::EspSupportArgs<Real, ComplexT<Real>> a;
            a.n = n;
            a.c = d_c;
            a.force_c = d_force_c;
            a.out = d_force_out;
            cuda::esp::launch_stage<Real>(cuda::esp::kEspStageAccumForce, n, a, gpu.stream);
        };
        interp_and_accumulate(f_hat_x, d_fx);
        interp_and_accumulate(f_hat_y, d_fy);
        interp_and_accumulate(f_hat_z, d_fz);
    }
}

template <typename Real>
static auto &host_buf(GpuState *gpu) {
    if constexpr (std::is_same_v<Real, double>)
        return gpu->h_dbl_buf;
    else
        return gpu->h_flt_buf;
}

template <typename Real>
static Real self_factor(GpuState *gpu) {
    return Real(gpu->self_factor);
}

// Calling with the wrong Real would reinterpret_cast every buffer and plan handle to the wrong
// concrete type, so fail loudly instead.
template <typename Real>
static void check_plan_real(const GpuState *gpu) {
    const bool wants_float = std::is_same_v<Real, float>;
    if (wants_float != gpu->use_float)
        throw std::runtime_error(std::string("esp_eval_gpu: called with Real=") + (wants_float ? "float" : "double") +
                                 " but this plan was created for " + (gpu->use_float ? "float" : "double"));
}

// Resize + zero the host buffer, returning the four output spans.
template <typename Real>
static auto gpu_make_spans(GpuState *gpu, int n) {
    [[maybe_unused]] const bool want_force = (gpu->eval_type >= DMK_POTENTIAL_GRAD);
    const int slots = want_force ? 4 : 1;
    auto &buf = host_buf<Real>(gpu);
    buf.assign(slots * n, Real(0));
    Real *p = buf.data();
    return std::tuple{std::span<Real>(p, n), want_force ? std::span<Real>(p + n, n) : std::span<Real>{},
                      want_force ? std::span<Real>(p + 2 * n, n) : std::span<Real>{},
                      want_force ? std::span<Real>(p + 3 * n, n) : std::span<Real>{}};
}

// Device-side inputs and output accumulators, all views into the plan's persistent scratch. The
// three entry points differ only in which passes they run over these, so the setup and teardown live
// here rather than being spelled out (and kept in sync) three times.
template <typename Real>
struct EvalBuffers {
    Real *pos_aos = nullptr;
    Real *charges = nullptr;
    Real *pot = nullptr;
    Real *fx = nullptr;
    Real *fy = nullptr;
    Real *fz = nullptr;
};

template <typename Real>
static EvalBuffers<Real> upload_inputs(GpuState *gpu, int n, bool want_force, const std::vector<Vec3T<Real>> &r_src,
                                       const std::vector<Real> &charges) {
    NvtxRange range("eval/upload_input");
    EvalBuffers<Real> b;

    ensure_capacity(gpu->d_scratch_pos, gpu->scratch_pos_cap, 4 * std::size_t(n) * sizeof(Real));
    b.pos_aos = reinterpret_cast<Real *>(gpu->d_scratch_pos);
    b.charges = b.pos_aos + 3 * n;

    ensure_capacity(gpu->d_scratch_out, gpu->scratch_out_cap, 4 * std::size_t(n) * sizeof(Real));
    b.pot = reinterpret_cast<Real *>(gpu->d_scratch_out);
    b.fx = want_force ? b.pot + n : nullptr;
    b.fy = want_force ? b.pot + 2 * n : nullptr;
    b.fz = want_force ? b.pot + 3 * n : nullptr;

    const Real *h_pos_aos = reinterpret_cast<const Real *>(r_src.data());
    cudaMemcpyAsync(b.pos_aos, h_pos_aos, 3 * std::size_t(n) * sizeof(Real), cudaMemcpyHostToDevice, gpu->stream);
    cudaMemcpyAsync(b.charges, charges.data(), n * sizeof(Real), cudaMemcpyHostToDevice, gpu->stream);

    // The passes accumulate with +=, so the accumulators start at zero.
    cudaMemsetAsync(b.pot, 0, n * sizeof(Real), gpu->stream);
    if (want_force) {
        cudaMemsetAsync(b.fx, 0, n * sizeof(Real), gpu->stream);
        cudaMemsetAsync(b.fy, 0, n * sizeof(Real), gpu->stream);
        cudaMemsetAsync(b.fz, 0, n * sizeof(Real), gpu->stream);
    }
    return b;
}

// Long-range wants scaled [-pi,pi) SoA coords + packed complex charges. The AoS inputs are already
// resident, so this is a device-to-device reshape.
template <typename Real>
static void pack_long_range_inputs(GpuState *gpu, int n, Real scale, const EvalBuffers<Real> &b, Real *&d_x, Real *&d_y,
                                   Real *&d_z, ComplexT<Real> *&d_c) {
    NvtxRange range("eval/long_range_setup");
    ensure_capacity(gpu->d_scratch_lr_xyz, gpu->scratch_lr_xyz_cap, 3 * std::size_t(n) * sizeof(Real));
    d_x = reinterpret_cast<Real *>(gpu->d_scratch_lr_xyz);
    d_y = d_x + n;
    d_z = d_y + n;
    ensure_capacity(gpu->d_scratch_lr_c, gpu->scratch_lr_c_cap, std::size_t(n) * sizeof(ComplexT<Real>));
    d_c = reinterpret_cast<ComplexT<Real> *>(gpu->d_scratch_lr_c);

    cuda::EspSupportArgs<Real, ComplexT<Real>> a;
    a.n = n;
    a.scale = scale;
    a.pos_aos = b.pos_aos;
    a.charges = b.charges;
    a.xs = d_x;
    a.ys = d_y;
    a.zs = d_z;
    a.c_out = d_c;
    cuda::esp::launch_stage<Real>(cuda::esp::kEspStageScalePack, n, a, gpu->stream);
}

template <typename Real>
static void download_outputs(GpuState *gpu, int n, bool want_force, const EvalBuffers<Real> &b, std::span<Real> pot,
                             std::span<Real> fx, std::span<Real> fy, std::span<Real> fz) {
    NvtxRange range("eval/download_output");
    cudaStreamSynchronize(gpu->stream);
    cudaMemcpy(pot.data(), b.pot, n * sizeof(Real), cudaMemcpyDeviceToHost);
    if (want_force) {
        cudaMemcpy(fx.data(), b.fx, n * sizeof(Real), cudaMemcpyDeviceToHost);
        cudaMemcpy(fy.data(), b.fy, n * sizeof(Real), cudaMemcpyDeviceToHost);
        cudaMemcpy(fz.data(), b.fz, n * sizeof(Real), cudaMemcpyDeviceToHost);
    }
}

template <typename Real>
static PotForce<Real> esp_eval_gpu_impl(GpuState *gpu, const std::vector<Vec3T<Real>> &r_src,
                                        const std::vector<Real> &charges) {
    check_plan_real<Real>(gpu);
    const int n = static_cast<int>(r_src.size());
    const bool want_force = (gpu->eval_type >= DMK_POTENTIAL_GRAD);
    auto [pot, fx, fy, fz] = gpu_make_spans<Real>(gpu, n);
    const EvalBuffers<Real> b = upload_inputs<Real>(gpu, n, want_force, r_src, charges);

    {
        NvtxRange range("eval/short_range");
        cuda::esp::short_range_gpu<Real>(*gpu, n, b.pos_aos, b.charges, b.pot, b.fx, b.fy, b.fz);
    }

    const Real scale = Real(2.0 * M_PI) / Real(gpu->L);
    Real *d_x, *d_y, *d_z;
    ComplexT<Real> *d_c;
    pack_long_range_inputs<Real>(gpu, n, scale, b, d_x, d_y, d_z, d_c);

    {
        NvtxRange range("eval/long_range");
        long_range_gpu<Real>(*gpu, n, d_x, d_y, d_z, d_c, scale, want_force, b.pot, b.fx, b.fy, b.fz);
    }

    // Potential only, matching the CPU self_interaction.
    {
        NvtxRange range("eval/self_interaction");
        cuda::EspSupportArgs<Real, ComplexT<Real>> a;
        a.n = n;
        a.factor = self_factor<Real>(gpu);
        a.charges = b.charges;
        a.pot = b.pot;
        cuda::esp::launch_stage<Real>(cuda::esp::kEspStageSelfInteraction, n, a, gpu->stream);
    }

    download_outputs<Real>(gpu, n, want_force, b, pot, fx, fy, fz);
    return {pot, fx, fy, fz};
}
template <typename Real>
static PotForce<Real> esp_eval_gpu_short_range_impl(GpuState *gpu, const std::vector<Vec3T<Real>> &r_src,
                                                    const std::vector<Real> &charges) {
    check_plan_real<Real>(gpu);
    const int n = static_cast<int>(r_src.size());
    const bool want_force = (gpu->eval_type >= DMK_POTENTIAL_GRAD);
    auto [pot, fx, fy, fz] = gpu_make_spans<Real>(gpu, n);
    const EvalBuffers<Real> b = upload_inputs<Real>(gpu, n, want_force, r_src, charges);

    {
        NvtxRange range("eval/short_range");
        cuda::esp::short_range_gpu<Real>(*gpu, n, b.pos_aos, b.charges, b.pot, b.fx, b.fy, b.fz);
    }

    download_outputs<Real>(gpu, n, want_force, b, pot, fx, fy, fz);
    return {pot, fx, fy, fz};
}
template <typename Real>
static PotForce<Real> esp_eval_gpu_long_range_impl(GpuState *gpu, const std::vector<Vec3T<Real>> &r_src,
                                                   const std::vector<Real> &charges) {
    check_plan_real<Real>(gpu);
    const int n = static_cast<int>(r_src.size());
    const bool want_force = (gpu->eval_type >= DMK_POTENTIAL_GRAD);
    auto [pot, fx, fy, fz] = gpu_make_spans<Real>(gpu, n);
    const EvalBuffers<Real> b = upload_inputs<Real>(gpu, n, want_force, r_src, charges);

    const Real scale = Real(2.0 * M_PI) / Real(gpu->L);
    Real *d_x, *d_y, *d_z;
    ComplexT<Real> *d_c;
    pack_long_range_inputs<Real>(gpu, n, scale, b, d_x, d_y, d_z, d_c);

    {
        NvtxRange range("eval/long_range");
        long_range_gpu<Real>(*gpu, n, d_x, d_y, d_z, d_c, scale, want_force, b.pot, b.fx, b.fy, b.fz);
    }

    download_outputs<Real>(gpu, n, want_force, b, pot, fx, fy, fz);
    return {pot, fx, fy, fz};
}

PotForce<float> esp_eval_gpu(GpuState *gpu, const std::vector<Vec3T<float>> &r_src, const std::vector<float> &charges) {
    return esp_eval_gpu_impl<float>(gpu, r_src, charges);
}
PotForce<double> esp_eval_gpu(GpuState *gpu, const std::vector<Vec3T<double>> &r_src,
                              const std::vector<double> &charges) {
    return esp_eval_gpu_impl<double>(gpu, r_src, charges);
}

PotForce<float> esp_eval_gpu_short_range(GpuState *gpu, const std::vector<Vec3T<float>> &r_src,
                                         const std::vector<float> &charges) {
    return esp_eval_gpu_short_range_impl<float>(gpu, r_src, charges);
}
PotForce<double> esp_eval_gpu_short_range(GpuState *gpu, const std::vector<Vec3T<double>> &r_src,
                                          const std::vector<double> &charges) {
    return esp_eval_gpu_short_range_impl<double>(gpu, r_src, charges);
}

PotForce<float> esp_eval_gpu_long_range(GpuState *gpu, const std::vector<Vec3T<float>> &r_src,
                                        const std::vector<float> &charges) {
    return esp_eval_gpu_long_range_impl<float>(gpu, r_src, charges);
}
PotForce<double> esp_eval_gpu_long_range(GpuState *gpu, const std::vector<Vec3T<double>> &r_src,
                                         const std::vector<double> &charges) {
    return esp_eval_gpu_long_range_impl<double>(gpu, r_src, charges);
}

} // namespace dmk
