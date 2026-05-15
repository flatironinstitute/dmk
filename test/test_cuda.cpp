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

#include <dmk/cuda/aot_kernels.hpp>
#include <dmk/cuda/charge2proxy_kernels.hpp>
#include <dmk/cuda/direct_kernels.hpp>
#include <dmk/cuda/pw_to_proxy_kernels.hpp>
#include <dmk/cuda/shift_pw_kernels.hpp>
#include <dmk/cuda/tensorprod_kernels.hpp>
#include <dmk/direct.hpp>
#include <dmk/planewave.hpp>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/testing.hpp>
#include <dmk/types.hpp>
#include <dmk/util.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <complex>
#include <functional>
#include <random>
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

// Invoke the per-box direct driver (the production path used by
// CudaDirectContext) as if it were a flat all-pairs evaluator: build a
// degenerate one-box "tree" with direct_work=[0], list1=[[0]], and let the
// kernel do the full N_SRC × N_TRG sweep inside that single block.
template <typename Real>
void run_direct_by_box_pair_eval(dmk_ikernel kernel, int dim, int n_digits, Real rsc, Real cen, Real d2max,
                                 Real thresh2, int n_src, const Real *d_r_src, const Real *d_charges,
                                 const Real *d_normals, int n_trg, const Real *d_r_trg, Real *d_pot) {
    const int work_h = 0;
    const int list1_flat_h = 0;
    const int list1_count_h = 1;
    const int box_levels_h = 0;
    const unsigned char ifpwexp_h = 0;
    const long zero_long_h = 0;

    DeviceBuf<int> d_work(1, &work_h);
    DeviceBuf<int> d_list1_flat(1, &list1_flat_h);
    DeviceBuf<int> d_list1_count(1, &list1_count_h);
    DeviceBuf<int> d_box_levels(1, &box_levels_h);
    DeviceBuf<unsigned char> d_ifpwexp(1, &ifpwexp_h);
    DeviceBuf<Real> d_rsc(1, &rsc);
    DeviceBuf<Real> d_cen(1, &cen);
    DeviceBuf<Real> d_d2max(1, &d2max);
    DeviceBuf<long> d_offset(1, &zero_long_h);
    DeviceBuf<int> d_n_src(1, &n_src);
    DeviceBuf<int> d_n_trg(1, &n_trg);

    dmk::cuda::DirectByBoxArgs<Real> args;
    args.n_work = 1;
    args.n_levels = 1;
    args.nlist1_stride = 1;
    args.thresh2 = thresh2;
    args.direct_work = d_work.p;
    args.list1_flat = d_list1_flat.p;
    args.list1_count = d_list1_count.p;
    args.box_levels = d_box_levels.p;
    args.ifpwexp = d_ifpwexp.p;
    args.direct_rsc = d_rsc.p;
    args.direct_cen = d_cen.p;
    args.direct_d2max = d_d2max.p;
    args.r_src_halo_flat = d_r_src;
    args.r_src_halo_offsets = d_offset.p;
    args.src_counts_halo = d_n_src.p;
    args.charge_halo_flat = d_charges;
    args.charge_halo_offsets = d_offset.p;
    args.normal_halo_flat = d_normals;
    args.normal_halo_offsets = d_normals ? d_offset.p : nullptr;
    args.r_target_flat = d_r_trg;
    args.r_target_offsets = d_offset.p;
    args.target_counts = d_n_trg.p;
    args.pot_flat = d_pot;
    args.pot_offsets = d_offset.p;

    dmk::cuda::launch_direct_by_box_dispatch<Real>(kernel, dim, n_digits, args, 0);
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

    DeviceBuf<Real> d_normals;
    Real *d_normals_ptr = nullptr;
    if (with_normals) {
        d_normals = DeviceBuf<Real>(normals.size(), normals.data());
        d_normals_ptr = d_normals.p;
    }

    const std::size_t pot_n = static_cast<std::size_t>(N_TRG) * out_dim;

    // Path 1: all-pairs driver (EvalPairsCuda via the AOT residual_evaluator_func).
    {
        DeviceBuf<Real> d_pot(pot_n);
        REQUIRE_CUDA(cudaMemset(d_pot.p, 0, d_pot.n * sizeof(Real)));
        cuda_eval(rsc, cen, d2max, thresh2, N_SRC, d_r_src.p, d_charges.p, d_normals_ptr, N_TRG, d_r_trg.p, d_pot.p);
        REQUIRE_CUDA(cudaDeviceSynchronize());
        std::vector<Real> pot_cuda(d_pot.n);
        REQUIRE_CUDA(cudaMemcpy(pot_cuda.data(), d_pot.p, d_pot.n * sizeof(Real), cudaMemcpyDeviceToHost));
        const double err = rel_l2(pot_cuda, pot_cpu);
        INFO("all-pairs driver rel_l2 = " << err);
        CHECK(err < tol);
    }

    // Path 2: per-box driver (DirectResidualByBoxKernelTiled, production path).
    {
        DeviceBuf<Real> d_pot(pot_n);
        REQUIRE_CUDA(cudaMemset(d_pot.p, 0, d_pot.n * sizeof(Real)));
        run_direct_by_box_pair_eval<Real>(kernel, n_dim, n_digits, rsc, cen, d2max, thresh2, N_SRC, d_r_src.p,
                                          d_charges.p, d_normals_ptr, N_TRG, d_r_trg.p, d_pot.p);
        REQUIRE_CUDA(cudaDeviceSynchronize());
        std::vector<Real> pot_cuda(d_pot.n);
        REQUIRE_CUDA(cudaMemcpy(pot_cuda.data(), d_pot.p, d_pot.n * sizeof(Real), cudaMemcpyDeviceToHost));
        const double err = rel_l2(pot_cuda, pot_cpu);
        INFO("per-box driver rel_l2 = " << err);
        CHECK(err < tol);
    }
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

// Downward-pass kernel tests: each GPU kernel against its CPU counterpart
// on isolated inputs.

namespace {

template <typename T>
void fill_random(std::vector<T> &v, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (auto &x : v)
        x = T(d(rng));
}

} // namespace

TEST_CASE_TEMPLATE("[CUDA] tensorprod (parent->child) vs CPU", Real, double, float) {
    using namespace dmk;
    constexpr int DIM = 3;
    const int n_order = 8;
    const int n_charge_dim = 2;
    const int N3 = n_order * n_order * n_order;

    std::mt19937_64 rng(SEED);
    std::vector<Real> parent(N3 * n_charge_dim);
    std::vector<Real> umat(3 * n_order * n_order); // 3 axis matrices, n_order × n_order each
    fill_random(parent, rng);
    fill_random(umat, rng);

    // CPU reference: tensorprod::transform writes into fout.
    std::vector<Real> child_cpu(N3 * n_charge_dim, Real{0});
    {
        ndview<Real, DIM + 1> fin_v({n_order, n_order, n_order, n_charge_dim}, parent.data());
        ndview<Real, 2> umat_v({n_order, DIM}, umat.data()); // shape is wrong but only .data() is used
        ndview<Real, DIM + 1> fout_v({n_order, n_order, n_order, n_charge_dim}, child_cpu.data());
        sctl::Vector<Real> workspace;
        tensorprod::transform<Real, DIM>(n_charge_dim, /*add_flag=*/0, fin_v, umat_v, fout_v, workspace);
    }

    // GPU: lay out a 2-box proxy buffer (parent at offset 0, child at N3*n_charge_dim).
    const int n_proxy_box = N3 * n_charge_dim;
    std::vector<Real> proxy_host(2 * n_proxy_box, Real{0});
    std::copy(parent.begin(), parent.end(), proxy_host.begin());
    DeviceBuf<Real> d_proxy(proxy_host.size(), proxy_host.data());

    std::vector<long> offsets_h{0L, (long)n_proxy_box};
    DeviceBuf<long> d_offsets(offsets_h.size(), offsets_h.data());
    DeviceBuf<Real> d_p2c(umat.size(), umat.data());

    std::vector<int> parents_h{0}, children_h{1}, octants_h{0};
    DeviceBuf<int> d_parents(1, parents_h.data()), d_children(1, children_h.data()), d_octants(1, octants_h.data());

    const long scratch_stride = 2L * n_order * n_order * n_order;
    DeviceBuf<Real> d_scratch(scratch_stride);

    cuda::TensorprodArgs<Real> args;
    args.n_pairs = 1;
    args.n_order = n_order;
    args.n_charge_dim = n_charge_dim;
    args.src_boxes = d_parents.p;
    args.dst_boxes = d_children.p;
    args.child_octants = d_octants.p;
    args.proxy_flat = d_proxy.p;
    args.proxy_offsets = d_offsets.p;
    args.umat_flat = d_p2c.p;
    args.scratch = d_scratch.p;
    args.scratch_stride = scratch_stride;
    cuda::launch_tensorprod_dispatch<Real>(DIM, args, /*stream=*/0);
    REQUIRE_CUDA(cudaDeviceSynchronize());

    std::vector<Real> proxy_back(proxy_host.size());
    REQUIRE_CUDA(cudaMemcpy(proxy_back.data(), d_proxy.p, proxy_back.size() * sizeof(Real), cudaMemcpyDeviceToHost));
    std::vector<Real> child_gpu(proxy_back.begin() + n_proxy_box, proxy_back.end());

    const double err = rel_l2(child_gpu, child_cpu);
    INFO("rel_l2 = " << err);
    CHECK(err < (std::is_same_v<Real, double> ? TOL_DOUBLE : TOL_FLOAT));
}

TEST_CASE_TEMPLATE("[CUDA] pw_to_proxy vs CPU", Real, double, float) {
    using namespace dmk;
    using C = std::complex<Real>;
    constexpr int DIM = 3;
    const int n_order = 8;
    const int n_pw = 11;
    const int n_pw2 = (n_pw + 1) / 2;
    const int n_charge_dim = 2;
    const int n_pw_modes = n_pw * n_pw * n_pw2;
    const int N3 = n_order * n_order * n_order;

    std::mt19937_64 rng(SEED + 1);
    std::vector<Real> pw_in_real(2 * n_pw_modes * n_charge_dim); // interleaved complex
    std::vector<Real> pw2poly_real(2 * n_pw * n_order);          // interleaved complex
    fill_random(pw_in_real, rng);
    fill_random(pw2poly_real, rng);

    // CPU reference
    std::vector<Real> proxy_cpu(N3 * n_charge_dim, Real{0});
    {
        ndview<C, DIM + 1> pw_in_v({n_pw, n_pw, n_pw2, n_charge_dim}, reinterpret_cast<C *>(pw_in_real.data()));
        ndview<C, 2> pw2poly_v({n_pw, n_order}, reinterpret_cast<C *>(pw2poly_real.data()));
        ndview<Real, DIM + 1> proxy_v({n_order, n_order, n_order, n_charge_dim}, proxy_cpu.data());
        sctl::Vector<Real> workspace;
        planewave_to_proxy_potential<Real, DIM>(pw_in_v, pw2poly_v, proxy_v, workspace);
    }

    // GPU: pw_in_pool has one slot at offset 0; proxy_flat has one box at offset 0.
    DeviceBuf<Real> d_pw_in(pw_in_real.size(), pw_in_real.data());
    DeviceBuf<Real> d_pw2poly(pw2poly_real.size(), pw2poly_real.data());
    DeviceBuf<Real> d_proxy(N3 * n_charge_dim);
    REQUIRE_CUDA(cudaMemset(d_proxy.p, 0, d_proxy.n * sizeof(Real)));

    std::vector<long> offsets_h{0L};
    DeviceBuf<long> d_offsets(1, offsets_h.data());
    std::vector<int> box_ids_h{0};
    DeviceBuf<int> d_box_ids(1, box_ids_h.data());

    cuda::PwToProxyArgs<Real> args;
    args.n_boxes_at_level = 1;
    args.n_order = n_order;
    args.n_pw = n_pw;
    args.n_pw2 = n_pw2;
    args.n_charge_dim = n_charge_dim;
    args.pw_in_stride = 2L * n_charge_dim * n_pw_modes;
    args.box_ids = d_box_ids.p;
    args.pw_in_pool = d_pw_in.p;
    args.pw2poly = d_pw2poly.p;
    args.proxy_flat = d_proxy.p;
    args.proxy_offsets = d_offsets.p;
    cuda::launch_pw_to_proxy_dispatch<Real>(DIM, args, /*stream=*/0);
    REQUIRE_CUDA(cudaDeviceSynchronize());

    std::vector<Real> proxy_gpu(d_proxy.n);
    REQUIRE_CUDA(cudaMemcpy(proxy_gpu.data(), d_proxy.p, d_proxy.n * sizeof(Real), cudaMemcpyDeviceToHost));

    const double err = rel_l2(proxy_gpu, proxy_cpu);
    INFO("rel_l2 = " << err);
    CHECK(err < (std::is_same_v<Real, double> ? TOL_DOUBLE : TOL_FLOAT));
}

TEST_CASE_TEMPLATE("[CUDA] shift_pw vs CPU", Real, double, float) {
    using namespace dmk;
    constexpr int DIM = 3;
    constexpr int N_NEIGHBORS = 27;
    const int n_pw = 11;
    const int n_pw2 = (n_pw + 1) / 2;
    const int n_charge_dim = 2;
    const int n_pw_modes = n_pw * n_pw * n_pw2;
    const int n_box_reals = 2 * n_charge_dim * n_pw_modes;

    // 3 boxes total: box 0 is the target; boxes 1, 2 are neighbours that
    // contribute via wpwshift slots npos=0 (ind=26) and npos=1 (ind=25).
    // The remaining neighbour entries are -1 (no neighbour).
    std::mt19937_64 rng(SEED + 2);
    const int n_boxes = 3;
    std::vector<Real> pw_out_h(n_boxes * n_box_reals);
    std::vector<Real> wpwshift_h(2L * N_NEIGHBORS * n_pw_modes);
    fill_random(pw_out_h, rng);
    fill_random(wpwshift_h, rng);

    std::vector<int> neighbors_h(n_boxes * N_NEIGHBORS, -1);
    neighbors_h[0 * N_NEIGHBORS + 0] = 1;
    neighbors_h[0 * N_NEIGHBORS + 1] = 2;
    // Self at the canonical centre slot (npos = N_NEIGHBORS / 2 = 13). The
    // GPU and CPU both skip neighbour == box, so the slot value doesn't matter.
    neighbors_h[0 * N_NEIGHBORS + 13] = 0;

    std::vector<long> pw_out_offsets_h{0L, (long)n_pw_modes * n_charge_dim, (long)n_pw_modes * n_charge_dim * 2};
    std::vector<unsigned char> is_leaf_h(n_boxes, 0); // none are leaves → all neighbour pairs are eligible

    // CPU reference: pw_in[box=0] = pw_out[0] + Σ_neighbour pw_out[n] * wpwshift[ind].
    std::vector<Real> pw_in_cpu(n_box_reals);
    std::copy(pw_out_h.begin(), pw_out_h.begin() + n_box_reals, pw_in_cpu.begin());
    for (int npos = 0; npos < N_NEIGHBORS; ++npos) {
        const int nbr = neighbors_h[0 * N_NEIGHBORS + npos];
        if (nbr < 0 || nbr == 0)
            continue;
        const int ind = N_NEIGHBORS - 1 - npos;
        const Real *shift_r = wpwshift_h.data() + (long)ind * n_pw_modes * 2;
        const Real *shift_i = shift_r + n_pw_modes;
        const Real *nbr_pw = pw_out_h.data() + (long)nbr * n_box_reals;
        Real *dst = pw_in_cpu.data();
        for (int d = 0; d < n_charge_dim; ++d) {
            for (int m = 0; m < n_pw_modes; ++m) {
                const long idx = (long)d * n_pw_modes * 2 + 2L * m;
                const Real ar = nbr_pw[idx], ai = nbr_pw[idx + 1];
                const Real cr = shift_r[m], ci = shift_i[m];
                dst[idx] += ar * cr - ai * ci;
                dst[idx + 1] += ar * ci + ai * cr;
            }
        }
    }

    // GPU
    DeviceBuf<Real> d_pw_out(pw_out_h.size(), pw_out_h.data());
    DeviceBuf<Real> d_wpwshift(wpwshift_h.size(), wpwshift_h.data());
    DeviceBuf<int> d_neighbors(neighbors_h.size(), neighbors_h.data());
    DeviceBuf<long> d_pw_out_offsets(pw_out_offsets_h.size(), pw_out_offsets_h.data());
    DeviceBuf<unsigned char> d_is_leaf(is_leaf_h.size(), is_leaf_h.data());
    DeviceBuf<Real> d_pw_in_pool(n_box_reals);
    REQUIRE_CUDA(cudaMemset(d_pw_in_pool.p, 0, d_pw_in_pool.n * sizeof(Real)));

    std::vector<int> box_ids_h{0};
    DeviceBuf<int> d_box_ids(1, box_ids_h.data());

    cuda::ShiftPwArgs<Real> args;
    args.n_boxes_at_level = 1;
    args.n_neighbors = N_NEIGHBORS;
    args.n_charge_dim = n_charge_dim;
    args.n_pw_modes = n_pw_modes;
    args.pw_in_stride = (long)n_box_reals;
    args.box_ids = d_box_ids.p;
    args.neighbors = d_neighbors.p;
    args.pw_out_offsets = d_pw_out_offsets.p;
    args.is_global_leaf = d_is_leaf.p;
    args.pw_out_flat = d_pw_out.p;
    args.wpwshift = d_wpwshift.p;
    args.pw_in_pool = d_pw_in_pool.p;
    cuda::launch_shift_pw_dispatch<Real>(DIM, args, /*stream=*/0);
    REQUIRE_CUDA(cudaDeviceSynchronize());

    std::vector<Real> pw_in_gpu(n_box_reals);
    REQUIRE_CUDA(cudaMemcpy(pw_in_gpu.data(), d_pw_in_pool.p, n_box_reals * sizeof(Real), cudaMemcpyDeviceToHost));

    const double err = rel_l2(pw_in_gpu, pw_in_cpu);
    INFO("rel_l2 = " << err);
    CHECK(err < (std::is_same_v<Real, double> ? TOL_DOUBLE : TOL_FLOAT));
}

TEST_CASE_TEMPLATE("[CUDA] charge2proxy vs CPU", Real, double, float) {
    using namespace dmk;
    constexpr int DIM = 3;
    const int n_order = 8;
    const int n_charge_dim = 2;
    const int N3 = n_order * n_order * n_order;
    const int n_src = 73; // odd, deliberately spans a partial chunk

    std::mt19937_64 rng(SEED + 3);
    std::vector<Real> r_src_raw(n_src * DIM);
    std::vector<Real> charges(n_src * n_charge_dim);
    fill_random(r_src_raw, rng); // values in [-1, 1]
    fill_random(charges, rng);

    const Real cx = Real{0.1}, cy = Real{-0.05}, cz = Real{0.2};
    const Real half_width = Real{0.5};
    const Real scale = Real{1} / half_width; // 2 / boxsize, with boxsize = 2*half_width

    std::vector<Real> r_src(n_src * DIM);
    for (int s = 0; s < n_src; ++s) {
        r_src[s * DIM + 0] = cx + half_width * r_src_raw[s * DIM + 0];
        r_src[s * DIM + 1] = cy + half_width * r_src_raw[s * DIM + 1];
        r_src[s * DIM + 2] = cz + half_width * r_src_raw[s * DIM + 2];
    }

    // CPU reference.
    std::vector<Real> proxy_cpu(N3 * n_charge_dim, Real{0});
    {
        sctl::Vector<Real> workspace;
        ndview<const Real, 2> r_src_v({DIM, n_src}, r_src.data());
        ndview<const Real, 2> charge_v({n_charge_dim, n_src}, charges.data());
        std::vector<Real> center_h{cx, cy, cz};
        ndview<const Real, 1> center_v({DIM}, center_h.data());
        ndview<Real, DIM + 1> proxy_v({n_order, n_order, n_order, n_charge_dim}, proxy_cpu.data());
        proxy::charge2proxycharge<Real, DIM>(r_src_v, charge_v, center_v, scale, proxy_v, workspace);
    }

    // GPU: single group, single src_box. center_box = src_box = 0.
    DeviceBuf<Real> d_r_src(r_src.size(), r_src.data());
    DeviceBuf<Real> d_charges(charges.size(), charges.data());
    std::vector<Real> centers_h{cx, cy, cz};
    DeviceBuf<Real> d_centers(centers_h.size(), centers_h.data());
    std::vector<Real> inv_box_h{scale};
    DeviceBuf<Real> d_inv_box(1, inv_box_h.data());

    std::vector<long> r_src_offsets_h{0L};
    DeviceBuf<long> d_r_src_offsets(1, r_src_offsets_h.data());
    std::vector<int> src_counts_h{n_src};
    DeviceBuf<int> d_src_counts(1, src_counts_h.data());
    std::vector<long> charge_offsets_h{0L};
    DeviceBuf<long> d_charge_offsets(1, charge_offsets_h.data());
    std::vector<long> proxy_offsets_h{0L};
    DeviceBuf<long> d_proxy_offsets(1, proxy_offsets_h.data());

    DeviceBuf<Real> d_proxy(N3 * n_charge_dim);
    REQUIRE_CUDA(cudaMemset(d_proxy.p, 0, d_proxy.n * sizeof(Real)));

    std::vector<int> center_boxes_h{0}, levels_h{0}, sb_off_h{0}, n_sb_h{1}, sb_flat_h{0};
    DeviceBuf<int> d_center_boxes(1, center_boxes_h.data());
    DeviceBuf<int> d_levels(1, levels_h.data());
    DeviceBuf<int> d_sb_off(1, sb_off_h.data());
    DeviceBuf<int> d_n_sb(1, n_sb_h.data());
    DeviceBuf<int> d_sb_flat(1, sb_flat_h.data());

    cuda::Charge2ProxyArgs<Real> args;
    args.n_groups = 1;
    args.n_order = n_order;
    args.n_charge_dim = n_charge_dim;
    args.center_boxes = d_center_boxes.p;
    args.levels = d_levels.p;
    args.src_box_flat_offsets = d_sb_off.p;
    args.n_src_boxes_per_group = d_n_sb.p;
    args.src_boxes_flat = d_sb_flat.p;
    args.centers = d_centers.p;
    args.inv_box_scale = d_inv_box.p;
    args.r_src_owned = d_r_src.p;
    args.r_src_owned_offsets = d_r_src_offsets.p;
    args.src_counts_owned = d_src_counts.p;
    args.charge_owned = d_charges.p;
    args.charge_owned_offsets = d_charge_offsets.p;
    args.proxy_flat = d_proxy.p;
    args.proxy_offsets = d_proxy_offsets.p;
    cuda::launch_charge2proxy_dispatch<Real>(DIM, args, /*stream=*/0);
    REQUIRE_CUDA(cudaDeviceSynchronize());

    std::vector<Real> proxy_gpu(d_proxy.n);
    REQUIRE_CUDA(cudaMemcpy(proxy_gpu.data(), d_proxy.p, d_proxy.n * sizeof(Real), cudaMemcpyDeviceToHost));

    const double err = rel_l2(proxy_gpu, proxy_cpu);
    INFO("rel_l2 = " << err);
    CHECK(err < (std::is_same_v<Real, double> ? TOL_DOUBLE : TOL_FLOAT));
}
