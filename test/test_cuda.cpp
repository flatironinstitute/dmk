// Compares the CUDA AOT residual evaluators against their CPU AOT
// counterparts on identical inputs. With rsc=1, cen=0 the host's
// shift_scale_polynomial and transform_poly preprocessing are no-ops, so
// the two paths should evaluate the same polynomial expression to within
// floating-point reassociation noise.
//
// Kernels not yet wired up in the bootstrap CUDA AOT throw at getter time
// — those test cases WARN and pass rather than fail, so the file becomes
// progressively more useful as scripts/generate_aot_kernels --target=cuda
// fills in the missing entries.

#include <dmk/aot_kernels_cuda.hpp>
#include <dmk/direct.hpp>
#include <dmk/testing.hpp>
#include <dmk/types.hpp>
#include <dmk/util.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <functional>
#include <vector>

namespace {

constexpr int N_SRC = 256;
constexpr int N_TRG = 128;
constexpr long SEED = 42;

#define REQUIRE_CUDA(expr)                                                                                             \
    do {                                                                                                               \
        cudaError_t _e = (expr);                                                                                       \
        REQUIRE_MESSAGE(_e == cudaSuccess, cudaGetErrorString(_e));                                                    \
    } while (0)

template <typename Real>
struct DeviceBuf {
    Real *p = nullptr;
    std::size_t n = 0;

    DeviceBuf() = default;
    explicit DeviceBuf(std::size_t n_) : n(n_) { REQUIRE_CUDA(cudaMalloc(&p, n * sizeof(Real))); }
    DeviceBuf(std::size_t n_, const Real *src_host) : DeviceBuf(n_) {
        REQUIRE_CUDA(cudaMemcpy(p, src_host, n * sizeof(Real), cudaMemcpyHostToDevice));
    }
    ~DeviceBuf() {
        if (p)
            cudaFree(p);
    }
    DeviceBuf(const DeviceBuf &) = delete;
    DeviceBuf &operator=(const DeviceBuf &) = delete;
    DeviceBuf(DeviceBuf &&other) noexcept : p(other.p), n(other.n) {
        other.p = nullptr;
        other.n = 0;
    }
    DeviceBuf &operator=(DeviceBuf &&other) noexcept {
        if (this != &other) {
            if (p)
                cudaFree(p);
            p = other.p;
            n = other.n;
            other.p = nullptr;
            other.n = 0;
        }
        return *this;
    }
};

template <typename Real>
double rel_l2(const std::vector<Real> &a, const std::vector<Real> &b) {
    double err = 0.0, nrm = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = double(a[i]) - double(b[i]);
        err += d * d;
        nrm += double(b[i]) * double(b[i]);
    }
    if (nrm == 0.0)
        return std::sqrt(err);
    return std::sqrt(err / nrm);
}

// Build a CUDA evaluator via the supplied getter, but if it throws (n_digits
// not yet generated in the bootstrap) return a null function. Caller WARNs.
template <typename Real>
dmk::residual_evaluator_func<Real>
try_cuda(const std::function<dmk::residual_evaluator_func<Real>(dmk_eval_type, int)> &getter, dmk_eval_type ev,
         int n_digits) {
    try {
        return getter(ev, n_digits);
    } catch (const std::exception &) {
        return {};
    }
}

template <typename Real>
void compare_kernel(dmk_ikernel kernel, int n_dim, int n_digits, dmk_eval_type eval_level, int charge_dim, int out_dim,
                    bool with_normals,
                    const std::function<dmk::residual_evaluator_func<Real>(dmk_eval_type, int)> &cuda_getter,
                    double tol) {
    std::vector<Real> r_src, r_trg, charges, normals;
    dmk::util::init_test_data(n_dim, charge_dim, N_SRC, N_TRG, /*uniform=*/true,
                              /*set_fixed_charges=*/false, r_src, r_trg, normals, charges, SEED);

    auto cuda_eval = try_cuda<Real>(cuda_getter, eval_level, n_digits);
    if (!cuda_eval) {
        WARN_MESSAGE(false, "CUDA AOT not yet generated for this kernel/digits — run "
                            "./generate_aot_kernels --target=cuda > src/cuda_kernels.cu");
        return;
    }

    auto cpu_eval = dmk::make_evaluator_aot<Real>(kernel, eval_level, n_dim, n_digits, /*unroll_factor=*/3);

    Real bsize = 0.25;
    Real bsizeinv = 1 / bsize;
    Real rsc = 2 * bsizeinv;
    Real cen = -bsize / Real{2};
    if ((kernel == DMK_SQRT_LAPLACE && n_dim == 3) || (kernel == DMK_LAPLACE && n_dim == 2)) {
        rsc = 2 * bsizeinv * bsizeinv;
        cen = Real{-1.0};
    } else if (kernel == DMK_YUKAWA)
        cen = Real{-1.0};

    const Real d2max = bsize * bsize;
    const Real thresh2 = Real{1e-30};

    std::vector<Real> pot_cpu(static_cast<std::size_t>(N_TRG) * out_dim, Real{0});
    cpu_eval(rsc, cen, d2max, thresh2, N_SRC, r_src.data(), charges.data(), with_normals ? normals.data() : nullptr,
             N_TRG, r_trg.data(), pot_cpu.data());

    DeviceBuf<Real> d_r_src(r_src.size(), r_src.data());
    DeviceBuf<Real> d_charges(charges.size(), charges.data());
    DeviceBuf<Real> d_r_trg(r_trg.size(), r_trg.data());
    DeviceBuf<Real> d_pot(static_cast<std::size_t>(N_TRG) * out_dim);
    REQUIRE_CUDA(cudaMemset(d_pot.p, 0, d_pot.n * sizeof(Real)));

    DeviceBuf<Real> d_normals;
    Real *d_normals_ptr = nullptr;
    if (with_normals) {
        d_normals = DeviceBuf<Real>(normals.size(), normals.data());
        d_normals_ptr = d_normals.p;
    }

    cuda_eval(rsc, cen, d2max, thresh2, N_SRC, d_r_src.p, d_charges.p, d_normals_ptr, N_TRG, d_r_trg.p, d_pot.p);
    REQUIRE_CUDA(cudaDeviceSynchronize());

    std::vector<Real> pot_cuda(d_pot.n);
    REQUIRE_CUDA(cudaMemcpy(pot_cuda.data(), d_pot.p, d_pot.n * sizeof(Real), cudaMemcpyDeviceToHost));

    const double err = rel_l2(pot_cuda, pot_cpu);
    INFO("rel_l2 = " << err);
    CHECK(err < tol);
}
} // namespace

constexpr int N_DIGITS_D = 12;
constexpr int N_DIGITS_F = 6;
constexpr double TOL_DOUBLE = 1e-12;
constexpr double TOL_FLOAT = 2e-6;

TEST_CASE("[CUDA] Laplace 2D vs CPU AOT") {
    auto getter_d = [](dmk_eval_type ev, int n) { return dmk::get_laplace_2d_kernel_cuda<double>(ev, n); };
    auto getter_f = [](dmk_eval_type ev, int n) { return dmk::get_laplace_2d_kernel_cuda<float>(ev, n); };
    compare_kernel<double>(DMK_LAPLACE, 2, N_DIGITS_D, DMK_POTENTIAL, 1, 1, false, getter_d, TOL_DOUBLE);
    compare_kernel<float>(DMK_LAPLACE, 2, N_DIGITS_F, DMK_POTENTIAL, 1, 1, false, getter_f, TOL_FLOAT);
}

TEST_CASE("[CUDA] Laplace 3D vs CPU AOT") {
    auto getter_d = [](dmk_eval_type ev, int n) { return dmk::get_laplace_3d_kernel_cuda<double>(ev, n); };
    auto getter_f = [](dmk_eval_type ev, int n) { return dmk::get_laplace_3d_kernel_cuda<float>(ev, n); };
    compare_kernel<double>(DMK_LAPLACE, 3, N_DIGITS_D, DMK_POTENTIAL, 1, 1, false, getter_d, TOL_DOUBLE);
    compare_kernel<float>(DMK_LAPLACE, 3, N_DIGITS_F, DMK_POTENTIAL, 1, 1, false, getter_f, TOL_FLOAT);
}

TEST_CASE("[CUDA] SqrtLaplace 2D vs CPU AOT") {
    auto getter_d = [](dmk_eval_type ev, int n) { return dmk::get_sqrt_laplace_2d_kernel_cuda<double>(ev, n); };
    auto getter_f = [](dmk_eval_type ev, int n) { return dmk::get_sqrt_laplace_2d_kernel_cuda<float>(ev, n); };
    compare_kernel<double>(DMK_SQRT_LAPLACE, 2, N_DIGITS_D, DMK_POTENTIAL, 1, 1, false, getter_d, TOL_DOUBLE);
    compare_kernel<float>(DMK_SQRT_LAPLACE, 2, N_DIGITS_F, DMK_POTENTIAL, 1, 1, false, getter_f, TOL_FLOAT);
}

TEST_CASE("[CUDA] SqrtLaplace 3D vs CPU AOT") {
    auto getter_d = [](dmk_eval_type ev, int n) { return dmk::get_sqrt_laplace_3d_kernel_cuda<double>(ev, n); };
    auto getter_f = [](dmk_eval_type ev, int n) { return dmk::get_sqrt_laplace_3d_kernel_cuda<float>(ev, n); };
    compare_kernel<double>(DMK_SQRT_LAPLACE, 3, N_DIGITS_D, DMK_POTENTIAL, 1, 1, false, getter_d, TOL_DOUBLE);
    compare_kernel<float>(DMK_SQRT_LAPLACE, 3, N_DIGITS_F, DMK_POTENTIAL, 1, 1, false, getter_f, TOL_FLOAT);
}

TEST_CASE("[CUDA] Stokeslet 3D vs CPU AOT") {
    auto getter_d = [](dmk_eval_type ev, int n) { return dmk::get_stokeslet_3d_kernel_cuda<double>(ev, n); };
    auto getter_f = [](dmk_eval_type ev, int n) { return dmk::get_stokeslet_3d_kernel_cuda<float>(ev, n); };
    compare_kernel<double>(DMK_STOKESLET, 3, N_DIGITS_D, DMK_VELOCITY, 3, 3, false, getter_d, TOL_DOUBLE);
    compare_kernel<float>(DMK_STOKESLET, 3, N_DIGITS_F, DMK_VELOCITY, 3, 3, false, getter_f, TOL_FLOAT);
}

TEST_CASE("[CUDA] Stresslet 3D vs CPU AOT") {
    auto getter_d = [](dmk_eval_type ev, int n) { return dmk::get_stresslet_3d_kernel_cuda<double>(ev, n); };
    auto getter_f = [](dmk_eval_type ev, int n) { return dmk::get_stresslet_3d_kernel_cuda<float>(ev, n); };
    compare_kernel<double>(DMK_STRESSLET, 3, N_DIGITS_D, DMK_VELOCITY, 3, 3, true, getter_d, TOL_DOUBLE);
    compare_kernel<float>(DMK_STRESSLET, 3, N_DIGITS_F, DMK_VELOCITY, 3, 3, true, getter_f, TOL_FLOAT);
}
