#ifdef DMK_GPU_OFFLOAD

#include <array>
#include <cmath>
#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/esp.hpp>
#include <dmk/periodic_reference.hpp>
#include <doctest/doctest.h>
#include <map>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double L = 1.0;
// nc = floor(L/r_c) = 6, so the pruned strategies' shared table fits at N_DENSE.
constexpr double R_C = 0.15;
// Below ~10k sources the measured error has not converged and reports ~1 digit worse than the truth
// (dipole gradient at eps=1e-6: 5.8 digits at 2k, 6.9 at 20k), which reads as a solver failure.
constexpr int N_ACC = 20000;
constexpr int N_DENSE = 100000;
constexpr int N_TEST = 1000;

struct Config {
    dmk_ikernel kernel;
    double fparam;
    dmk_eval_type eval_type;
    bool periodic;
    int n_img; // Yukawa image-sum shells; 0 selects the Ewald (periodic) or direct (free) reference
    const char *name;
};

const Config kConfigs[] = {
    {DMK_LAPLACE, 0.0, DMK_POTENTIAL_GRAD, true, 0, "laplace grad pbc"},
    {DMK_LAPLACE, 0.0, DMK_POTENTIAL, true, 0, "laplace pot pbc"},
    {DMK_SQRT_LAPLACE, 0.0, DMK_POTENTIAL_GRAD, true, 0, "sqrt-laplace grad pbc"},
    {DMK_SQRT_LAPLACE, 0.0, DMK_POTENTIAL, true, 0, "sqrt-laplace pot pbc"},
    {DMK_YUKAWA, 6.0, DMK_POTENTIAL_GRAD, true, 6, "yukawa grad pbc"},
    {DMK_YUKAWA, 6.0, DMK_POTENTIAL, true, 6, "yukawa pot pbc"},
    {DMK_LAPLACE, 0.0, DMK_POTENTIAL_GRAD, false, 0, "laplace grad free"},
    {DMK_SQRT_LAPLACE, 0.0, DMK_POTENTIAL_GRAD, false, 0, "sqrt-laplace grad free"},
    {DMK_YUKAWA, 6.0, DMK_POTENTIAL_GRAD, false, 0, "yukawa grad free"},
    // Free-space only on the CPU too.
    {DMK_LAPLACE_DIPOLE, 0.0, DMK_POTENTIAL, false, 0, "dipole pot free"},
    {DMK_LAPLACE_DIPOLE, 0.0, DMK_POTENTIAL_GRAD, false, 0, "dipole grad free"},
    {DMK_STOKESLET, 0.0, DMK_VELOCITY, false, 0, "stokeslet free"},
    {DMK_STRESSLET, 0.0, DMK_VELOCITY, false, 0, "stresslet free"},
};

// Scalar kernels report force = -q*grad; the dipole reports the raw field gradient.
bool force_convention(dmk_ikernel k) { return k == DMK_LAPLACE || k == DMK_SQRT_LAPLACE || k == DMK_YUKAWA; }

// Sources in [0.01, 0.99)^3 for the periodic references, and the same points shifted into
// [-L/2, L/2) for the solver. A shift is invisible to the free-space kernels.
struct Points {
    int n, in_dim, nrm_dim, out_dim;
    std::vector<double> r_ref, r_esp, q, nrm;
    std::vector<dmk::Vec3T<double>> r_vec;

    const double *nrm_ptr() const { return nrm.empty() ? nullptr : nrm.data(); }

    Points(const Config &c, int n_, unsigned seed) : n(n_) {
        in_dim = dmk::get_kernel_input_dim(3, c.kernel);
        nrm_dim = (c.kernel == DMK_STRESSLET) ? 3 : 0;
        out_dim = dmk::get_kernel_output_dim(3, c.kernel, c.eval_type);

        std::default_random_engine eng(seed);
        std::uniform_real_distribution<double> rng(0.01, 0.99);
        r_ref.resize(3ul * n);
        q.resize(std::size_t(in_dim) * n);
        nrm.resize(std::size_t(nrm_dim) * n);
        for (double &x : r_ref)
            x = rng(eng);
        for (double &v : q)
            v = rng(eng) - 0.5;
        for (double &v : nrm)
            v = rng(eng) - 0.5;
        // The periodic Laplace/Sqrt-Laplace lattice sums only converge for a neutral cell.
        if (c.periodic && c.kernel != DMK_YUKAWA)
            for (int k = 0; k < in_dim; ++k) {
                double s = 0;
                for (int i = 0; i < n; ++i)
                    s += q[in_dim * i + k];
                for (int i = 0; i < n; ++i)
                    q[in_dim * i + k] -= s / n;
            }

        r_esp.resize(r_ref.size());
        for (std::size_t i = 0; i < r_ref.size(); ++i)
            r_esp[i] = r_ref[i] - 0.5 * L;
        r_vec.resize(n);
        for (int i = 0; i < n; ++i)
            r_vec[i] = {r_esp[3 * i], r_esp[3 * i + 1], r_esp[3 * i + 2]};
    }
};

pdmk_esp_params esp_params(const Config &c, double eps) {
    pdmk_esp_params p{};
    p.L = L;
    p.r_c = R_C;
    p.eps = eps;
    p.log_level = 6;
    p.kernel = c.kernel;
    p.fparam = c.fparam;
    p.n_dim = 3;
    p.eval_type = c.eval_type;
    p.sigma = 1.35;
    p.use_periodic = c.periodic ? 1 : 0;
    return p;
}

// Every component the kernel can report, in ESP's output convention, so configs differing only in
// eval_type share one reference. Velocity kernels have 3; the rest have pot + 3 gradient.
std::vector<std::vector<double>> exact_reference(const Config &c, const Points &p) {
    const bool velocity = c.eval_type == DMK_VELOCITY;
    const int nc = velocity ? 3 : 4;
    std::vector<std::vector<double>> ref(nc, std::vector<double>(N_TEST, 0.0));

    if (c.periodic && c.n_img == 0) {
        dmk::pbc_ref::EwaldRef ewald(c.kernel, 3, p.n, p.r_ref.data(), p.q.data(), L);
        for (int i = 0; i < N_TEST; ++i) {
            double g[3] = {0, 0, 0};
            ewald.eval(&p.r_ref[3 * i], i, ref[0][i], g);
            for (int k = 1; k < nc; ++k)
                ref[k][i] = -p.q[i] * g[k - 1];
        }
        return ref;
    }

    const dmk_eval_type et = velocity ? DMK_VELOCITY : DMK_POTENTIAL_GRAD;
    std::vector<double> flat;
    if (c.periodic) {
        dmk::pbc_ref::image_sum(3, c.fparam, c.n_img, et, p.n, p.r_ref.data(), p.q.data(), L, N_TEST, p.r_ref.data(),
                                flat);
    } else {
        flat.assign(std::size_t(N_TEST) * nc, 0.0);
        dmk::get_direct_evaluator<double>(c.kernel, et, 3, c.fparam)(p.n, p.r_esp.data(), p.q.data(),
                                                                     p.nrm.empty() ? nullptr : p.nrm.data(), N_TEST,
                                                                     p.r_esp.data(), flat.data());
    }
    for (int i = 0; i < N_TEST; ++i)
        for (int k = 0; k < nc; ++k) {
            const double v = flat[std::size_t(i) * nc + k];
            ref[k][i] = (k > 0 && force_convention(c.kernel)) ? -p.q[i] * v : v;
        }
    return ref;
}

// ESP reports vector-field kernels as velocity and everything else as pot + gradient.
std::array<std::span<double>, 4> components(dmk::PotForce<double> &pf) {
    if (!pf.vel_x.empty())
        return {pf.vel_x, pf.vel_y, pf.vel_z, std::span<double>{}};
    return {pf.pot, pf.force_x, pf.force_y, pf.force_z};
}

void check_vs_reference(dmk::PotForce<double> pf, const std::vector<std::vector<double>> &ref, double tol,
                        const Config &c, double eps, const char *who) {
    const auto comp = components(pf);
    const int out_dim = dmk::get_kernel_output_dim(3, c.kernel, c.eval_type);
    for (int k = 0; k < out_dim; ++k) {
        double e2 = 0, r2 = 0;
        for (int i = 0; i < N_TEST; ++i) {
            const double d = comp[k][i] - ref[k][i];
            e2 += d * d;
            r2 += ref[k][i] * ref[k][i];
        }
        const double l2 = dmk::pbc_ref::safe_l2(e2, r2);
        CHECK_MESSAGE(l2 < tol, std::string(c.name) << " eps=" << eps << " " << who << " comp" << k << " l2=" << l2);
    }
}

// Relative L2 between two evaluations of the same quantity.
double l2_between(const std::array<std::span<double>, 4> &a, const std::array<std::span<double>, 4> &b, int out_dim,
                  int n) {
    double e2 = 0, r2 = 0;
    for (int k = 0; k < out_dim; ++k)
        for (int i = 0; i < n; ++i) {
            const double d = a[k][i] - b[k][i];
            e2 += d * d;
            r2 += b[k][i] * b[k][i];
        }
    return dmk::pbc_ref::safe_l2(e2, r2);
}

struct Fixture {
    Points p;
    dmk::EspPlan<double> *plan;
    dmk::GpuState *gpu;

    Fixture(const Config &c, double eps, int n_src, dmk::GpuSrStrategy strategy = dmk::GpuSrStrategy::Dense,
            dmk::GpuSortMode sort_mode = dmk::GpuSortMode::Bins)
        : p(c, n_src, 1234u) {
        auto params = esp_params(c, eps);
        plan = new dmk::EspPlan<double>(params);
        gpu = dmk::esp_create_gpu_plan(plan, strategy, sort_mode);
    }
    ~Fixture() {
        dmk::esp_destroy_gpu_plan(gpu);
        delete plan;
    }
    dmk::PotForce<double> cpu_eval() {
        return plan->eval(p.n, p.r_esp.data(), p.q.data(), p.nrm.empty() ? nullptr : p.nrm.data());
    }
};

} // namespace

// The only accuracy check: every kernel against an independent lattice sum (periodic) or all-pairs
// sum (free space), at the requested tolerance. The CPU is checked alongside so a failure is
// attributable to the GPU rather than to the shared plan.
TEST_CASE("[ESP GPU] accuracy vs an exact reference") {
    const double epses[] = {1e-3, 1e-6};
    // The periodic lattice sums dominate the runtime, and configs differing only in eval_type share
    // their points, charges and reference. Key on what the reference actually depends on.
    std::map<std::pair<int, bool>, std::vector<std::vector<double>>> refs;
    for (const Config &c : kConfigs) {
        const auto key = std::make_pair(int(c.kernel), c.periodic);
        auto it = refs.find(key);
        if (it == refs.end())
            it = refs.emplace(key, exact_reference(c, Points(c, N_ACC, 1234u))).first;
        for (const double eps : epses) {
            Fixture f(c, eps, N_ACC);
            check_vs_reference(f.cpu_eval(), it->second, eps, c, eps, "cpu");
            check_vs_reference(dmk::esp_eval_gpu(f.gpu, f.p.n, f.p.r_esp.data(), f.p.q.data(), f.p.nrm_ptr()),
                               it->second, eps, c, eps, "gpu");
        }
    }
}

// Two evaluations of the same finite sum, so this is implementation agreement and not eps-governed.
// Pins the per-kernel residual coefficients and the polynomial variable mapping.
TEST_CASE("[ESP GPU] short-range: GPU vs CPU") {
    constexpr double TOL = 1e-6;
    for (const Config &c : kConfigs) {
        Fixture f(c, 1e-5, N_DENSE);
        auto cpu = dmk::esp_eval_short_range(f.plan, f.p.r_vec, f.p.q, f.p.nrm);
        auto gpu = dmk::esp_eval_gpu_short_range(f.gpu, f.p.n, f.p.r_esp.data(), f.p.q.data(), f.p.nrm_ptr());
        const double l2 = l2_between(components(gpu), components(cpu), f.p.out_dim, f.p.n);
        CHECK_MESSAGE(l2 < TOL, std::string(c.name) << " short-range l2=" << l2);
    }
}

// Shared tiles, warp shuffle and ballot compaction are three separate readers of the source payload.
TEST_CASE("[ESP GPU] short-range: strategies and sort modes agree") {
    constexpr double TOL = 1e-6;
    const dmk::GpuSrStrategy strategies[] = {dmk::GpuSrStrategy::Dense, dmk::GpuSrStrategy::PruneTile,
                                             dmk::GpuSrStrategy::PruneSource};
    const dmk::GpuSortMode sorts[] = {dmk::GpuSortMode::Bins, dmk::GpuSortMode::Morton};
    for (const Config &c : kConfigs) {
        Fixture ref(c, 1e-5, N_DENSE);
        auto cpu = dmk::esp_eval_short_range(ref.plan, ref.p.r_vec, ref.p.q, ref.p.nrm);
        std::vector<std::vector<double>> cpu_copy;
        for (auto sp : components(cpu))
            cpu_copy.emplace_back(sp.begin(), sp.end());
        for (auto st : strategies)
            for (auto sm : sorts) {
                Fixture f(c, 1e-5, N_DENSE, st, sm);
                auto gpu = dmk::esp_eval_gpu_short_range(f.gpu, f.p.n, f.p.r_esp.data(), f.p.q.data(), f.p.nrm_ptr());
                const auto g = components(gpu);
                double e2 = 0, r2 = 0;
                for (int k = 0; k < f.p.out_dim; ++k)
                    for (int i = 0; i < f.p.n; ++i) {
                        const double d = g[k][i] - cpu_copy[k][i];
                        e2 += d * d;
                        r2 += cpu_copy[k][i] * cpu_copy[k][i];
                    }
                const double l2 = dmk::pbc_ref::safe_l2(e2, r2);
                CHECK_MESSAGE(l2 < TOL, std::string(c.name)
                                            << " strategy=" << int(st) << " sort=" << int(sm) << " l2=" << l2);
            }
    }
}

// Free space is the only case where the particle box and the FFT grid differ.
TEST_CASE("[ESP GPU] free-space uses a padded grid") {
    auto params = esp_params({DMK_LAPLACE, 0.0, DMK_POTENTIAL, false, 0, "laplace pot free"}, 1e-5);
    dmk::EspPlan<double> plan(params);
    CHECK(plan.pad > 1.0);
    CHECK(plan.L_grid > L);
    CHECK(plan.params.L == L);
}

// 2D has no GPU path.
TEST_CASE("[ESP GPU] 2D plans are rejected") {
    auto params = esp_params({DMK_LAPLACE, 0.0, DMK_POTENTIAL, true, 0, "laplace pot pbc"}, 1e-5);
    params.n_dim = 2;
    dmk::EspPlan<double> plan(params);
    CHECK_THROWS(dmk::esp_create_gpu_plan(&plan));
}

#endif // DMK_GPU_OFFLOAD
