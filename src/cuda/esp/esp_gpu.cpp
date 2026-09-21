#include "short_range.hpp"
#include "state.hpp"
#include "support.hpp"

#include <dmk.h>
#include <dmk/cuda/esp_gpu.hpp>
#include <dmk/error.hpp>
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

GpuState *gpu_create_state(const GpuPlanConfig &cfg) {
    const int nf = cfg.nf;
    const double L_grid = cfg.L_grid;
    const double r_c = cfg.r_c;
    const double tol = cfg.tol;
    const double gpu_upsampfac = cfg.gpu_upsampfac;
    const dmk_eval_type eval_type = cfg.eval_type;
    const bool use_float = cfg.use_float;
    const double *h_scaling_coeffs = cfg.h_scaling_coeffs;

    auto *gpu = new GpuState;
    gpu->nf = nf;
    gpu->n_digits = cfg.n_digits;
    gpu->L_grid = L_grid;
    gpu->r_c = r_c;
    gpu->use_periodic = cfg.use_periodic;
    gpu->trunc_rl = cfg.trunc_rl;
    gpu->beta = cfg.beta;
    gpu->self_factor = cfg.self_factor;
    gpu->fparam = cfg.fparam;
    gpu->dipole_grad_self = cfg.dipole_grad_self;
    gpu->kernel = cfg.kernel;
    gpu->eval_type = eval_type;
    gpu->use_float = use_float;
    gpu->strategy = cfg.strategy;
    gpu->sort_mode = cfg.sort_mode;

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

    // One grid per spread channel and one per output component; the Stresslet needs 9 + 3.
    const KernelDims dims = kernel_dims(cfg.kernel, eval_type);
    gpu->dims = dims;
    const auto grid_alloc = [&](void **p, int count, const char *what) {
        const std::size_t bytes = std::size_t(count) * std::size_t(ntot) * complex_sz;
        if (cudaMalloc(p, bytes) != cudaSuccess)
            throw std::runtime_error("GpuState: cudaMalloc " + std::string(what) + " failed (" + std::to_string(count) +
                                     " grids of nf^3=" + std::to_string(ntot) + ", " + std::to_string(bytes >> 20) +
                                     " MiB)");
    };
    grid_alloc(&gpu->d_b, dims.n_channels, "d_b");
    grid_alloc(&gpu->d_u_hat, dims.out_dim, "d_u_hat");

    if (cufftPlan3d(&gpu->fft_plan, nf, nf, nf, use_float ? CUFFT_C2C : CUFFT_Z2Z) != CUFFT_SUCCESS)
        throw std::runtime_error("GpuState: cufftPlan3d failed");
    cufftSetStream(gpu->fft_plan, gpu->stream);
    gpu->fft_plan_valid = true;

    // makeplan does not bind to n, so these are created once; each eval does setpts then execute.
    cufinufft_opts co;
    cufinufft_default_opts(&co);
    co.gpu_spreadinterponly = 1;
    co.upsampfac = gpu_upsampfac;
    // The Horner spreader only carries coefficients for the standard upsampfacs.
    co.gpu_kerevalmeth = (gpu_upsampfac == 2.0 || gpu_upsampfac == 1.25) ? 1 : 0;
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

    // 27-cell-stencil neighbour tables. Unit particle box, so 1 and not L_grid.
    gpu->nc = static_cast<int>(std::floor(1.0 / r_c));
    if (gpu->nc < 3)
        throw api_error(DMK_ERR_INVALID_ARGUMENT, "esp: the GPU short-range path requires r_c <= 1/3 (nc >= 3)");
    {
        const int ntab = gpu->nc * 3;
        std::vector<int> h_nbc_tab(ntab);
        std::vector<double> h_off_tab(ntab);
        // Free space: out-of-range neighbours get the -1 sentinel and the device cell walk skips them.
        const bool periodic = cfg.use_periodic;
        for (int c = 0; c < gpu->nc; ++c) {
            for (int d = 0; d < 3; ++d) {
                int ci = c + d - 1;
                if (ci < 0) {
                    h_nbc_tab[c * 3 + d] = periodic ? ci + gpu->nc : -1;
                    h_off_tab[c * 3 + d] = periodic ? -1.0 : 0.0;
                } else if (ci >= gpu->nc) {
                    h_nbc_tab[c * 3 + d] = periodic ? ci - gpu->nc : -1;
                    h_off_tab[c * 3 + d] = periodic ? 1.0 : 0.0;
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

// spread -> FFT -> project -> IFFT -> interp. The kernel decides how many spectra go in
// (n_channels) and come out (out_dim); scalar potential+gradient is 1 / 4, the Stresslet 9 / 3.
template <typename Real>
static void long_range_gpu(GpuState &gpu, int n, const KernelDims &dims, const Real *d_x, const Real *d_y,
                           const Real *d_z, const ComplexT<Real> *d_c, Real coeff_grad, Real *const *d_out) {
    const long long ntot = (long long)gpu.nf * gpu.nf * gpu.nf;
    const int nch = dims.n_channels;
    const int odim = dims.out_dim;
    auto *d_b = reinterpret_cast<ComplexT<Real> *>(gpu.d_b);
    auto *d_u_hat = reinterpret_cast<ComplexT<Real> *>(gpu.d_u_hat);
    auto *d_scaling_coeffs = reinterpret_cast<Real *>(gpu.d_scaling_coeffs);

    const auto fft = [&](ComplexT<Real> *buf, int direction, const char *what) {
        cufftResult r = cufft_exec_c2c_t<Real>(gpu.fft_plan, buf, buf, direction);
        if (r != CUFFT_SUCCESS)
            throw std::runtime_error(std::string("long_range_gpu: cufft ") + what +
                                     " failed, err=" + std::to_string(r));
    };
    const auto normalize = [&](ComplexT<Real> *buf) {
        cuda::EspSupportArgs<Real, ComplexT<Real>> a;
        a.ntot = static_cast<int>(ntot);
        a.inv_ntot = Real(1) / Real(ntot);
        a.grid = buf;
        cuda::esp::launch_stage<Real>(cuda::esp::kEspStageNormalize, static_cast<int>(ntot), a, gpu.stream);
    };

    // Step 1: spread each channel onto its own grid. setpts binds once for all of them.
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

        for (int ch = 0; ch < nch; ++ch) {
            // Zero before spreading -- cuFINUFFT accumulates into the output buffer.
            cudaMemsetAsync(d_b + ch * ntot, 0, ntot * sizeof(ComplexT<Real>), gpu.stream);
            ier = cufinufft_execute_t<Real>(gpu.cfnufft_plan_1, const_cast<ComplexT<Real> *>(d_c) + ch * n,
                                            d_b + ch * ntot);
            if (ier != 0)
                throw std::runtime_error("long_range_gpu: cufinufft_execute spread failed, ier=" + std::to_string(ier));
        }
    }

    // Step 2: forward FFT each channel, in place.
    {
        NvtxRange range("long_range/fft_forward");
        for (int ch = 0; ch < nch; ++ch)
            fft(d_b + ch * ntot, CUFFT_FORWARD, "forward");
    }

    // Step 3: the per-mode projector, n_channels spectra -> out_dim spectra.
    {
        NvtxRange range("long_range/project");
        cuda::EspSupportArgs<Real, ComplexT<Real>> a;
        a.ntot = static_cast<int>(ntot);
        a.nf = gpu.nf;
        a.out_dim = odim;
        a.n_channels = nch;
        a.coeff_grad = coeff_grad;
        a.scaling_coeffs = d_scaling_coeffs;
        a.chan_in = d_b;
        a.chan_out = d_u_hat;
        cuda::esp::launch_stage<Real>(cuda::esp::kEspStageProject, static_cast<int>(ntot), a, gpu.stream,
                                      cuda::esp::esp_projector_for(gpu.kernel));
    }

    // Step 4: inverse FFT + normalize each output component, in place.
    {
        NvtxRange range("long_range/fft_inverse_normalize");
        for (int k = 0; k < odim; ++k) {
            fft(d_u_hat + k * ntot, CUFFT_INVERSE, "inverse");
            normalize(d_u_hat + k * ntot);
        }
    }

    // Step 5: interpolate each component back to the NU points.
    {
        NvtxRange range("long_range/interp_setpts");
        int ier = cufinufft_setpts_t<Real>(gpu.cfnufft_plan_2, n, const_cast<Real *>(d_x), const_cast<Real *>(d_y),
                                           const_cast<Real *>(d_z));
        if (ier != 0)
            throw std::runtime_error("long_range_gpu: cufinufft_setpts interp failed, ier=" + std::to_string(ier));
    }

    ensure_capacity(gpu.d_scratch_nu_c, gpu.scratch_nu_c_cap, std::size_t(n) * sizeof(ComplexT<Real>), gpu.stream);
    auto *d_nu_c = reinterpret_cast<ComplexT<Real> *>(gpu.d_scratch_nu_c);

    for (int k = 0; k < odim; ++k) {
        if (!d_out[k])
            continue;
        NvtxRange range("long_range/interp_accumulate");
        int ier = cufinufft_execute_t<Real>(gpu.cfnufft_plan_2, d_nu_c, d_u_hat + k * ntot);
        if (ier != 0)
            throw std::runtime_error("long_range_gpu: cufinufft_execute interp failed, ier=" + std::to_string(ier));

        cuda::EspSupportArgs<Real, ComplexT<Real>> a;
        a.n = n;
        a.c = d_nu_c;
        a.out = d_out[k];
        cuda::esp::launch_stage<Real>(cuda::esp::kEspStageExtractReal, n, a, gpu.stream);
    }
}

template <typename Real>
static auto &host_buf(GpuState *gpu) {
    if constexpr (std::is_same_v<Real, double>)
        return gpu->h_dbl_buf;
    else
        return gpu->h_flt_buf;
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

// Resize + zero the host buffer, returning one span per output component (unused slots stay empty).
template <typename Real>
static std::array<std::span<Real>, 4> gpu_make_spans(GpuState *gpu, int n, const KernelDims &dims) {
    auto &buf = host_buf<Real>(gpu);
    buf.assign(std::size_t(dims.out_dim) * n, Real(0));
    Real *p = buf.data();
    std::array<std::span<Real>, 4> sp{};
    for (int k = 0; k < dims.out_dim; ++k)
        sp[k] = std::span<Real>(p + std::size_t(k) * n, n);
    return sp;
}

// Device-side inputs and output accumulators, all views into the plan's persistent scratch.
template <typename Real>
struct EvalBuffers {
    Real *pos_aos = nullptr;
    Real *charges = nullptr;                             // packed [charge | normal] payload, charge_dim * n, AoS
    Real *out[4] = {nullptr, nullptr, nullptr, nullptr}; // out_dim output accumulators
};

// `charges` is in_dim components per source, `normals` normal_dim more (Stresslet only); they are
// interleaved into one charge_dim-wide payload, as EspPlan::eval does.
template <typename Real>
static EvalBuffers<Real> upload_inputs(GpuState *gpu, int n, const KernelDims &dims, const Real *r_src,
                                       const Real *charges, const Real *normals) {
    NvtxRange range("eval/upload_input");
    EvalBuffers<Real> b;

    ensure_capacity(gpu->d_scratch_pos, gpu->scratch_pos_cap,
                    std::size_t(3 + dims.charge_dim) * std::size_t(n) * sizeof(Real), gpu->stream);
    b.pos_aos = reinterpret_cast<Real *>(gpu->d_scratch_pos);
    b.charges = b.pos_aos + 3 * n;

    ensure_capacity(gpu->d_scratch_out, gpu->scratch_out_cap, std::size_t(dims.out_dim) * std::size_t(n) * sizeof(Real),
                    gpu->stream);
    Real *out0 = reinterpret_cast<Real *>(gpu->d_scratch_out);
    for (int k = 0; k < dims.out_dim; ++k)
        b.out[k] = out0 + std::size_t(k) * n;

    cudaMemcpyAsync(b.pos_aos, r_src, 3 * std::size_t(n) * sizeof(Real), cudaMemcpyHostToDevice, gpu->stream);

    if (dims.normal_dim > 0) {
        if (!normals)
            throw api_error(DMK_ERR_INVALID_ARGUMENT, "esp_eval_gpu: this kernel requires per-source normals");
        std::vector<Real> packed(std::size_t(dims.charge_dim) * n);
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < dims.in_dim; ++k)
                packed[dims.charge_dim * i + k] = charges[dims.in_dim * i + k];
            for (int k = 0; k < dims.normal_dim; ++k)
                packed[dims.charge_dim * i + dims.in_dim + k] = normals[dims.normal_dim * i + k];
        }
        // Blocking: `packed` is a local about to go out of scope.
        cudaMemcpy(b.charges, packed.data(), packed.size() * sizeof(Real), cudaMemcpyHostToDevice);
    } else {
        cudaMemcpyAsync(b.charges, charges, std::size_t(dims.charge_dim) * n * sizeof(Real), cudaMemcpyHostToDevice,
                        gpu->stream);
    }

    // The passes accumulate with +=, so the accumulators start at zero.
    cudaMemsetAsync(out0, 0, std::size_t(dims.out_dim) * n * sizeof(Real), gpu->stream);
    return b;
}

// Scaled [-pi,pi) SoA coords + one complex plane per spread channel; a device-to-device reshape.
template <typename Real>
static void pack_long_range_inputs(GpuState *gpu, int n, const KernelDims &dims, Real scale, const EvalBuffers<Real> &b,
                                   Real *&d_x, Real *&d_y, Real *&d_z, ComplexT<Real> *&d_c) {
    NvtxRange range("eval/long_range_setup");
    ensure_capacity(gpu->d_scratch_lr_xyz, gpu->scratch_lr_xyz_cap, 3 * std::size_t(n) * sizeof(Real), gpu->stream);
    d_x = reinterpret_cast<Real *>(gpu->d_scratch_lr_xyz);
    d_y = d_x + n;
    d_z = d_y + n;
    ensure_capacity(gpu->d_scratch_lr_c, gpu->scratch_lr_c_cap,
                    std::size_t(dims.n_channels) * std::size_t(n) * sizeof(ComplexT<Real>), gpu->stream);
    d_c = reinterpret_cast<ComplexT<Real> *>(gpu->d_scratch_lr_c);

    cuda::EspSupportArgs<Real, ComplexT<Real>> a;
    a.n = n;
    a.scale = scale;
    a.charge_dim = dims.charge_dim;
    a.n_channels = dims.n_channels;
    a.pack_outer = (gpu->kernel == DMK_STRESSLET) ? 1 : 0;
    a.pos_aos = b.pos_aos;
    a.charges = b.charges;
    a.xs = d_x;
    a.ys = d_y;
    a.zs = d_z;
    a.c_out = d_c;
    cuda::esp::launch_stage<Real>(cuda::esp::kEspStageScalePack, n, a, gpu->stream);
}

template <typename Real>
static void download_outputs(GpuState *gpu, int n, const KernelDims &dims, const EvalBuffers<Real> &b,
                             const std::array<std::span<Real>, 4> &sp) {
    NvtxRange range("eval/download_output");
    cudaStreamSynchronize(gpu->stream);
    for (int k = 0; k < dims.out_dim; ++k)
        cudaMemcpy(sp[k].data(), b.out[k], n * sizeof(Real), cudaMemcpyDeviceToHost);
}

// Vector-field kernels report velocity, potential-family kernels pot + gradient, as EspPlan::eval.
template <typename Real>
static PotGrad<Real> as_pot_grad(const std::array<std::span<Real>, 4> &sp, dmk_ikernel kernel) {
    if (kernel == DMK_STOKESLET || kernel == DMK_STRESSLET)
        return {{}, {}, {}, {}, sp[0], sp[1], sp[2]};
    return {sp[0], sp[1], sp[2], sp[3], {}, {}, {}};
}

template <typename Real>
static PotGrad<Real> esp_eval_gpu_impl(GpuState *gpu, int n, const Real *r_src, const Real *charges,
                                       const Real *normals) {
    check_plan_real<Real>(gpu);
    const KernelDims &dims = gpu->dims;
    const auto sp = gpu_make_spans<Real>(gpu, n, dims);
    const EvalBuffers<Real> b = upload_inputs<Real>(gpu, n, dims, r_src, charges, normals);

    {
        NvtxRange range("eval/short_range");
        cuda::esp::short_range_gpu<Real>(*gpu, n, b.pos_aos, b.charges, b.out[0], b.out[1], b.out[2], b.out[3]);
    }

    const Real scale = Real(2.0 * M_PI) / Real(gpu->L_grid);
    Real *d_x, *d_y, *d_z;
    ComplexT<Real> *d_c;
    pack_long_range_inputs<Real>(gpu, n, dims, scale, b, d_x, d_y, d_z, d_c);

    {
        NvtxRange range("eval/long_range");
        long_range_gpu<Real>(*gpu, n, dims, d_x, d_y, d_z, d_c, scale, b.out);
    }

    // The scalar kernels remove a potential self-energy; the dipole's odd potential self is zero but
    // its gradient carries a constant; the Stresslet has neither.
    {
        int first = 0, count = 0;
        Real factor = Real(0);
        if (dims.in_dim == 1) {
            count = 1;
            factor = Real(gpu->self_factor);
        } else if (gpu->kernel == DMK_LAPLACE_DIPOLE && dims.out_dim > 1) {
            first = 1;
            count = 3;
            factor = Real(gpu->dipole_grad_self);
        } else if (gpu->kernel == DMK_STOKESLET) {
            count = 3;
            factor = Real(gpu->self_factor);
        }
        if (count > 0) {
            NvtxRange range("eval/self_interaction");
            cuda::EspSupportArgs<Real, ComplexT<Real>> a;
            a.n = n;
            a.charge_dim = dims.charge_dim;
            a.self_first = first;
            a.self_count = count;
            a.factor = factor;
            a.charges = b.charges;
            a.pot = b.out[0];
            a.gx = b.out[1];
            a.gy = b.out[2];
            a.gz = b.out[3];
            cuda::esp::launch_stage<Real>(cuda::esp::kEspStageSelfInteraction, n, a, gpu->stream);
        }
    }

    // Free-space zero-mode gauge: the truncated Stokeslet symbol drops k=0, so a non-neutral net
    // force leaves a constant offset (Bagge & Tornberg).
    if (gpu->kernel == DMK_STOKESLET && !gpu->use_periodic) {
        NvtxRange range("eval/zero_mode_gauge");
        Real netf[3] = {Real(0), Real(0), Real(0)};
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k)
                netf[k] += charges[3 * i + k];
        for (int k = 0; k < 3; ++k) {
            cuda::EspSupportArgs<Real, ComplexT<Real>> a;
            a.n = n;
            a.self_first = k;
            a.factor = netf[k] / Real(gpu->trunc_rl);
            a.pot = b.out[0];
            a.gx = b.out[1];
            a.gy = b.out[2];
            a.gz = b.out[3];
            cuda::esp::launch_stage<Real>(cuda::esp::kEspStageAddConst, n, a, gpu->stream);
        }
    }

    download_outputs<Real>(gpu, n, dims, b, sp);
    return as_pot_grad<Real>(sp, gpu->kernel);
}

// Raw component spans, without the velocity relabelling, matching EspPlan's esp_eval_one_step.
template <typename Real>
static PotGrad<Real> esp_eval_gpu_short_range_impl(GpuState *gpu, int n, const Real *r_src, const Real *charges,
                                                   const Real *normals) {
    check_plan_real<Real>(gpu);
    const KernelDims &dims = gpu->dims;
    const auto sp = gpu_make_spans<Real>(gpu, n, dims);
    const EvalBuffers<Real> b = upload_inputs<Real>(gpu, n, dims, r_src, charges, normals);

    {
        NvtxRange range("eval/short_range");
        cuda::esp::short_range_gpu<Real>(*gpu, n, b.pos_aos, b.charges, b.out[0], b.out[1], b.out[2], b.out[3]);
    }

    download_outputs<Real>(gpu, n, dims, b, sp);
    return {sp[0], sp[1], sp[2], sp[3], {}, {}, {}};
}

#define DMK_ESP_GPU_ENTRY(Real)                                                                                        \
    PotGrad<Real> esp_eval_gpu(GpuState *gpu, int n, const Real *r_src, const Real *charges, const Real *normals) {    \
        return esp_eval_gpu_impl<Real>(gpu, n, r_src, charges, normals);                                               \
    }                                                                                                                  \
    PotGrad<Real> esp_eval_gpu_short_range(GpuState *gpu, int n, const Real *r_src, const Real *charges,               \
                                           const Real *normals) {                                                      \
        return esp_eval_gpu_short_range_impl<Real>(gpu, n, r_src, charges, normals);                                   \
    }

DMK_ESP_GPU_ENTRY(float)
DMK_ESP_GPU_ENTRY(double)
#undef DMK_ESP_GPU_ENTRY

} // namespace dmk
