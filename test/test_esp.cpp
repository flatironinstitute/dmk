#ifdef DMK_BUILD_ESP

#include <cmath>
#include <dmk.h>
#include <dmk/esp.hpp>
#include <doctest/doctest.h>
#include <random>
#include <span>
#include <vector>

#include "periodic_reference.hpp"

// 10-particle fixture
namespace {

constexpr int N = 10;
constexpr double L = 1.0, R_C = 0.05;

const double R_SRC[30] = {0.131538 - 0.5, 0.686773 - 0.5, 0.98255 - 0.5,   0.45865 - 0.5,   0.930436 - 0.5,
                          0.753356 - 0.5, 0.218959 - 0.5, 0.526929 - 0.5,  0.0726859 - 0.5, 0.678865 - 0.5,
                          0.653919 - 0.5, 0.884707 - 0.5, 0.934693 - 0.5,  0.701191 - 0.5,  0.436411 - 0.5,
                          0.519416 - 0.5, 0.762198 - 0.5, 0.477732 - 0.5,  0.0345721 - 0.5, 0.0474645 - 0.5,
                          0.274907 - 0.5, 0.5297 - 0.5,   0.328234 - 0.5,  0.166507 - 0.5,  0.00769819 - 0.5,
                          0.75641 - 0.5,  0.897656 - 0.5, 0.0668422 - 0.5, 0.365339 - 0.5,  0.0605643 - 0.5};

const double CHARGES[10] = {0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.1, -0.1};

} // namespace

// sigma defaults to pdmk_esp_params's own default (1.35), matching every test below.
static pdmk_esp_params make_esp_params(double L, double r_c, double eps, dmk_eval_type eval_type) {
    pdmk_esp_params params{};
    params.L = L;
    params.r_c = r_c;
    params.eps = eps;
    params.eval_type = eval_type;
    return params;
}

// Zero-mean triply-periodic 3D-Laplace (1/r) reference potential at each source, from the shared
// Ewald reference. Zero-meaning makes the comparison gauge-invariant (the fixtures are neutral).
static void laplace_reference(int n, const double *r_src, const double *charges, double L, double *ref_out) {
    dmk::pbc_ref::EwaldRef ewald(DMK_LAPLACE, 3, n, r_src, charges, L);
    double mean = 0;
    for (int i = 0; i < n; ++i) {
        double pot;
        ewald.eval(&r_src[i * 3], i, pot, nullptr);
        ref_out[i] = pot;
        mean += pot;
    }
    mean /= n;
    for (int i = 0; i < n; ++i)
        ref_out[i] -= mean;
}

// Finite-difference force reference: F_{i,a} = -q_i * d(pot_i)/dr_{i,a}, via central differences.
// `plan` must already be created with DMK_POTENTIAL_GRAD and is not destroyed here.
// force_ref must have room for 3*n doubles.
static void fd_force_reference(dmk::EspPlan<double> *plan, int n, const double *r_src, const double *charges,
                               double step_size, double *force_ref) {
    std::vector<double> r_pert(3 * n);

    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < 3; ++a) {
            std::copy(r_src, r_src + 3 * n, r_pert.begin());

            r_pert[3 * i + a] += step_size;
            double pot_plus = plan->eval(n, r_pert.data(), charges).pot[i];

            r_pert[3 * i + a] -= 2.0 * step_size;
            double pot_minus = plan->eval(n, r_pert.data(), charges).pot[i];

            force_ref[3 * i + a] = -charges[i] * (pot_plus - pot_minus) / (2.0 * step_size);
        }
    }
}

// L2-relative error between ESP's analytic forces and a flat [3*n] (x,y,z per particle) reference.
static double force_l2_rel_err(int n, std::span<double> force_x, std::span<double> force_y, std::span<double> force_z,
                               const double *force_ref) {
    double err2 = 0, ref2 = 0;
    for (int i = 0; i < n; ++i) {
        const double f_esp[3] = {force_x[i], force_y[i], force_z[i]};
        for (int a = 0; a < 3; ++a) {
            const double diff = f_esp[a] - force_ref[3 * i + a];
            err2 += diff * diff;
            ref2 += force_ref[3 * i + a] * force_ref[3 * i + a];
        }
    }
    return std::sqrt(err2 / ref2);
}

TEST_CASE("[ESP] 10-particle double vs Ewald") {
    constexpr double eps = 1e-5;

    dmk::EspPlan<double> *plan = new dmk::EspPlan<double>(make_esp_params(L, R_C, eps, DMK_POTENTIAL));
    auto esp = plan->eval(N, R_SRC, CHARGES);

    double ref[N];
    laplace_reference(N, R_SRC, CHARGES, L, ref);

    double esp_mean = 0;
    for (int i = 0; i < N; ++i)
        esp_mean += esp.pot[i];
    esp_mean /= N;

    double err2 = 0, ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double diff = (esp.pot[i] - esp_mean) - ref[i];
        err2 += diff * diff;
        ref2 += ref[i] * ref[i];
    }
    const double l2_rel_err = std::sqrt(err2 / ref2);
    CHECK_MESSAGE(l2_rel_err < eps, "10-particle l2_rel_err=" << l2_rel_err << " >= " << eps);
    delete plan;
}

// Long-range isolation test.
// Particles sit on a regular n_grid^3 cubic lattice with spacing h = L/n_grid.
// r_c < h guarantees every inter-particle distance exceeds r_c, so the
// short-range direct sum contributes exactly zero.
TEST_CASE("[ESP] long-range only: regular grid, no short-range pairs") {
    constexpr int n_grid = 4;
    constexpr int N = n_grid * n_grid * n_grid;
    constexpr double L = 1.0;
    constexpr double h = L / n_grid; // nearest-neighbour distance
    constexpr double r_c = 0.05;     // < h: guaranteed no short-range pairs
    constexpr double eps = 1e-6;
    static_assert(r_c < h, "r_c must be < grid spacing for the short-range sum to be zero");

    // Regular grid positions in [-L/2, L/2)^3; 3D checkerboard ±1 charges (charge-neutral).
    double r_src[N * 3];
    double charges[N];
    for (int ix = 0; ix < n_grid; ++ix)
        for (int iy = 0; iy < n_grid; ++iy)
            for (int iz = 0; iz < n_grid; ++iz) {
                const int i = (ix * n_grid + iy) * n_grid + iz;
                r_src[3 * i + 0] = (ix + 0.5) * h - 0.5 * L;
                r_src[3 * i + 1] = (iy + 0.5) * h - 0.5 * L;
                r_src[3 * i + 2] = (iz + 0.5) * h - 0.5 * L;
                charges[i] = ((ix + iy + iz) % 2 == 0) ? 1.0 : -1.0;
            }

    double ref[N];
    laplace_reference(N, r_src, charges, L, ref);

    // --- Run ESP ---
    dmk::EspPlan<double> *plan = new dmk::EspPlan<double>(make_esp_params(L, r_c, eps, DMK_POTENTIAL_GRAD));
    auto esp = plan->eval(N, r_src, charges);

    // Gauge-correct ESP output and compare to the (already zero-mean) Ewald reference.
    double esp_mean = 0;
    for (int i = 0; i < N; ++i)
        esp_mean += esp.pot[i];
    esp_mean /= N;

    double err2 = 0, ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double diff = (esp.pot[i] - esp_mean) - ref[i];
        err2 += diff * diff;
        ref2 += ref[i] * ref[i];
    }
    const double l2_rel_err = std::sqrt(err2 / ref2);

    CHECK_MESSAGE(l2_rel_err < 5 * eps, "long-range-only l2_rel_err=" << l2_rel_err << " >= 5 * eps=" << 5 * eps);

    // Long-range forces on a symmetric lattice should be ~0. Relative error is
    // undefined (ref ≈ 0), so we check absolute magnitude instead.
    double max_abs = 0;
    for (int i = 0; i < N; ++i) {
        max_abs = std::max({max_abs, std::abs(esp.force_x[i]), std::abs(esp.force_y[i]), std::abs(esp.force_z[i])});
    }
    double force_tol = 10 * std::pow(eps, 2.0 / 3.0);
    CHECK_MESSAGE(max_abs < force_tol, "long-range forces should vanish on symmetric lattice, max=" << max_abs);
    delete plan;
}

// Madelung constant test.
TEST_CASE("[ESP] Madelung constant: NaCl lattice, 1/r kernel") {
    constexpr int n_grid = 8;
    constexpr int N = n_grid * n_grid * n_grid; // 512
    constexpr double L = 1.0;
    constexpr double h = L / n_grid; // 0.125
    constexpr double r_c = 0.05;     // < h: no short-range pairs
    constexpr double eps = 1e-4;
    constexpr double M_NaCl = 1.7475645946331821906;
    static_assert(r_c < h, "r_c must be < h for the NaCl short-range sum to be zero");

    double r_src[N * 3];
    double charges[N];
    for (int ix = 0; ix < n_grid; ++ix)
        for (int iy = 0; iy < n_grid; ++iy)
            for (int iz = 0; iz < n_grid; ++iz) {
                const int i = (ix * n_grid + iy) * n_grid + iz;
                r_src[3 * i + 0] = (ix + 0.5) * h - 0.5 * L;
                r_src[3 * i + 1] = (iy + 0.5) * h - 0.5 * L;
                r_src[3 * i + 2] = (iz + 0.5) * h - 0.5 * L;
                charges[i] = ((ix + iy + iz) % 2 == 0) ? 1.0 : -1.0;
            }

    dmk::EspPlan<double> *plan = new dmk::EspPlan<double>(make_esp_params(L, r_c, eps, DMK_POTENTIAL));
    auto esp = plan->eval(N, r_src, charges);

    // NaCl is charge-neutral so the mean potential is zero; gauge-correct anyway.
    double esp_mean = 0;
    for (int i = 0; i < N; ++i)
        esp_mean += esp.pot[i];
    esp_mean /= N;

    // Analytical reference: phi_i = -q_i * M_NaCl / h.
    double err2 = 0, ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double expected = -charges[i] * M_NaCl / h;
        const double diff = (esp.pot[i] - esp_mean) - expected;
        err2 += diff * diff;
        ref2 += expected * expected;
    }
    const double l2_rel_err = std::sqrt(err2 / ref2);

    CHECK_MESSAGE(l2_rel_err < eps, "Madelung l2_rel_err=" << l2_rel_err << " >= eps=" << eps << " (M_NaCl=" << M_NaCl
                                                           << ", h=" << h << ")");
    delete plan;
}

// Short-range stress test.
// N particles are packed inside a sphere of radius r_c/4, so every pairwise
// distance is at most 2*(r_c/4) = r_c/2 < r_c.

TEST_CASE("[ESP] short-range stress: all pairs within r_c") {
    constexpr int N = 50;
    constexpr double L = 1.0;
    constexpr double r_c = 0.05;
    constexpr double eps = 1e-4;
    constexpr double sphere_r = r_c / 4.0; // max pairwise dist = 2*sphere_r = r_c/2 < r_c

    // Deterministic positions inside the sphere via rejection sampling.
    double r_src[N * 3];
    double charges[N];
    {
        std::mt19937 rng(12345u);
        std::uniform_real_distribution<double> uni(-sphere_r, sphere_r);
        int placed = 0;
        while (placed < N) {
            double x = uni(rng), y = uni(rng), z = uni(rng);
            if (x * x + y * y + z * z <= sphere_r * sphere_r) {
                r_src[3 * placed + 0] = x;
                r_src[3 * placed + 1] = y;
                r_src[3 * placed + 2] = z;
                charges[placed] = (placed % 2 == 0) ? 1.0 : -1.0;
                ++placed;
            }
        }
    }

    double ref[N];
    laplace_reference(N, r_src, charges, L, ref);

    // --- Run ESP ---
    dmk::EspPlan<double> *plan = new dmk::EspPlan<double>(make_esp_params(L, r_c, eps, DMK_POTENTIAL_GRAD));

    // Compute FD reference before the final eval so the plan's buffer is fresh when we hold spans.
    std::vector<double> force_ref(3 * N);
    fd_force_reference(plan, N, r_src, charges, 1e-6, force_ref.data());

    auto esp = plan->eval(N, r_src, charges);

    double esp_mean = 0;
    for (int i = 0; i < N; ++i)
        esp_mean += esp.pot[i];
    esp_mean /= N;

    double err2 = 0, ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double diff = (esp.pot[i] - esp_mean) - ref[i];
        err2 += diff * diff;
        ref2 += ref[i] * ref[i];
    }
    const double l2_rel_err = std::sqrt(err2 / ref2);

    CHECK_MESSAGE(l2_rel_err < 5 * eps, "short-range-stress l2_rel_err=" << l2_rel_err << " >= eps=" << eps);

    const double force_l2_err = force_l2_rel_err(N, esp.force_x, esp.force_y, esp.force_z, force_ref.data());
    double force_tol = 10 * std::pow(eps, 2.0 / 3.0);
    CHECK_MESSAGE(force_l2_err < force_tol,
                  "short-range-stress forces l2_rel_err=" << force_l2_err << " >= force_tol=" << force_tol);
    delete plan;
}

TEST_CASE("[ESP] forces - 10 particles") {
    constexpr double eps = 1e-6;

    dmk::EspPlan<double> *plan = new dmk::EspPlan<double>(make_esp_params(L, R_C, eps, DMK_POTENTIAL_GRAD));

    // Compute FD reference before the final eval so the plan's buffer is fresh when we hold spans.
    std::vector<double> force_ref(3 * N);
    fd_force_reference(plan, N, R_SRC, CHARGES, std::pow(eps, 1.0 / 3.0), force_ref.data());

    auto esp = plan->eval(N, R_SRC, CHARGES);

    // Compare ESP's analytic forces against the finite-difference reference.
    const double l2_rel_err = force_l2_rel_err(N, esp.force_x, esp.force_y, esp.force_z, force_ref.data());
    double force_tol = 10 * std::pow(eps, 2.0 / 3.0);
    CHECK_MESSAGE(l2_rel_err < force_tol,
                  "10-particle forces l2_rel_err=" << l2_rel_err << " >= force_tol=" << force_tol);
    delete plan;
}

// 2D now runs under both JIT and AOT; accuracy is validated by the PBC cases below. Here we just
// confirm the 2D plan constructs and evaluates without throwing on whichever path is active.
TEST_CASE("[ESP] DIM=2 plan runs (JIT and AOT)") {
    constexpr double eps = 1e-5;
    auto params = make_esp_params(L, R_C, eps, DMK_POTENTIAL);
    params.n_dim = 2;
    dmk::EspPlan<double> plan(params);
    CHECK_NOTHROW(plan.eval(N, R_SRC, CHARGES));
}

// esp_eval<float> is instantiated but was never exercised by any test before this; confirms the
// float path actually runs in float (finufftf*/complex<float> FFTs) rather than silently upcasting.
TEST_CASE("[ESP] float precision matches double") {
    constexpr double eps = 1e-5;

    dmk::EspPlan<double> *plan_d = new dmk::EspPlan<double>(make_esp_params(L, R_C, eps, DMK_POTENTIAL_GRAD));
    auto esp_d = plan_d->eval(N, R_SRC, CHARGES);

    float r_src_f[3 * N], charges_f[N];
    for (int i = 0; i < 3 * N; ++i)
        r_src_f[i] = float(R_SRC[i]);
    for (int i = 0; i < N; ++i)
        charges_f[i] = float(CHARGES[i]);

    dmk::EspPlan<float> *plan_f = new dmk::EspPlan<float>(make_esp_params(L, R_C, eps, DMK_POTENTIAL_GRAD));
    auto esp_f = plan_f->eval(N, r_src_f, charges_f);

    double pot_err2 = 0, pot_ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double diff = double(esp_f.pot[i]) - esp_d.pot[i];
        pot_err2 += diff * diff;
        pot_ref2 += esp_d.pot[i] * esp_d.pot[i];
    }
    const double pot_l2_rel_err = std::sqrt(pot_err2 / pot_ref2);
    CHECK_MESSAGE(pot_l2_rel_err < 1e-4, "float vs double potential l2_rel_err=" << pot_l2_rel_err);

    double f_err2 = 0, f_ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double diffs[3] = {double(esp_f.force_x[i]) - esp_d.force_x[i],
                                 double(esp_f.force_y[i]) - esp_d.force_y[i],
                                 double(esp_f.force_z[i]) - esp_d.force_z[i]};
        const double refs[3] = {esp_d.force_x[i], esp_d.force_y[i], esp_d.force_z[i]};
        for (int a = 0; a < 3; ++a) {
            f_err2 += diffs[a] * diffs[a];
            f_ref2 += refs[a] * refs[a];
        }
    }
    const double f_l2_rel_err = std::sqrt(f_err2 / f_ref2);
    CHECK_MESSAGE(f_l2_rel_err < 1e-3, "float vs double force l2_rel_err=" << f_l2_rel_err);

    delete plan_d;
    delete plan_f;
}

// pdmk_esp_eval (the public C API) is not exercised by any other test; confirms it interleaves
// [pot, fx, fy, fz] per particle rather than dropping forces (as it did before this fix).
TEST_CASE("[ESP] C API pdmk_esp_eval interleaves forces") {
    constexpr double eps = 1e-6;
    pdmk_esp_params params{};
    params.L = L;
    params.r_c = R_C;
    params.eps = eps;
    params.eval_type = DMK_POTENTIAL_GRAD;

    pdmk_esp_plan plan = pdmk_esp_plan_create(nullptr, params);
    REQUIRE(plan != nullptr);

    std::vector<double> pot_src(N * 4);
    pdmk_esp_eval(nullptr, plan, N, R_SRC, CHARGES, pot_src.data());

    dmk::EspPlan<double> *plan_cxx = new dmk::EspPlan<double>(params);
    auto esp = plan_cxx->eval(N, R_SRC, CHARGES);

    for (int i = 0; i < N; ++i) {
        CHECK(pot_src[i * 4 + 0] == doctest::Approx(esp.pot[i]));
        CHECK(pot_src[i * 4 + 1] == doctest::Approx(esp.force_x[i]));
        CHECK(pot_src[i * 4 + 2] == doctest::Approx(esp.force_y[i]));
        CHECK(pot_src[i * 4 + 3] == doctest::Approx(esp.force_z[i]));
    }

    pdmk_esp_plan_destroy(plan);
    delete plan_cxx;
}

// ---------------------------------------------------------------------------
// Full periodic-pipeline ESP validation.cpp). Each case draws 2000
// random sources in [0,L)^n_dim, builds an independent periodic reference (Ewald split, or an image
// sum for the screened Yukawa kernel), then runs the ESP solver on the same points shifted into
// [-L/2, L/2) and compares pot (+forces) across a precision sweep. sigma=1.35 cannot reach
// eps=1e-12, so the sweep stops at 9 digits.
namespace {

struct EspPbcParams {
    dmk_ikernel kernel;
    int n_dim;
    double fparam; // lambda (Yukawa); ignored otherwise
    bool neutral;  // neutralize charges (Laplace / Sqrt-Laplace)
    int n_img;     // image shells for the Yukawa image-sum reference (0 => Ewald reference)
    unsigned seed;
    int n_test;
};

static void run_esp_pbc(const EspPbcParams &c) {
    constexpr int n_src = 2000;
    const int n_dim = c.n_dim;
    const double L = 1.0;

    std::default_random_engine eng(c.seed);
    std::uniform_real_distribution<double> rng(0.01, 0.99);
    std::vector<double> r_src(size_t(n_dim) * n_src), charges(n_src);
    for (auto &x : r_src)
        x = rng(eng);
    for (auto &q : charges)
        q = rng(eng) - 0.5;
    if (c.neutral) {
        double s = 0;
        for (double q : charges)
            s += q;
        for (double &q : charges)
            q -= s / n_src;
    }

    // Reference pot (+grad) at the first n_test sources.
    const int n_test = c.n_test;
    std::vector<double> ref_pot(n_test), ref_grad(size_t(n_test) * n_dim);
    if (c.n_img > 0) {
        std::vector<double> ref;
        dmk::pbc_ref::image_sum(n_dim, c.fparam, c.n_img, DMK_POTENTIAL_GRAD, n_src, r_src.data(), charges.data(), L,
                                n_test, r_src.data(), ref);
        const int odim = 1 + n_dim;
        for (int i = 0; i < n_test; ++i) {
            ref_pot[i] = ref[i * odim];
            for (int d = 0; d < n_dim; ++d)
                ref_grad[i * n_dim + d] = ref[i * odim + 1 + d];
        }
    } else {
        dmk::pbc_ref::EwaldRef ewald(c.kernel, n_dim, n_src, r_src.data(), charges.data(), L);
        for (int i = 0; i < n_test; ++i) {
            double g[3];
            ewald.eval(&r_src[i * n_dim], i, ref_pot[i], g);
            for (int d = 0; d < n_dim; ++d)
                ref_grad[i * n_dim + d] = g[d];
        }
    }

    // ESP requires particles in [-L/2, L/2); the shift is periodic-invariant.
    std::vector<double> r_esp(size_t(n_dim) * n_src);
    for (size_t i = 0; i < r_esp.size(); ++i)
        r_esp[i] = r_src[i] - 0.5 * L;

    struct PrecisionCase {
        int n_digits;
        double eps, tol_pot, tol_grad;
    };
    const PrecisionCase cases[] = {{3, 1e-3, 1e-2, 1e-1}, {6, 1e-6, 1e-4, 1e-3}, {9, 1e-9, 1e-7, 1e-6}};
    for (const auto &pc : cases)
        for (int with_grad = 0; with_grad <= 1; ++with_grad) {
            const auto eval = with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
            pdmk_esp_params ep{};
            ep.L = L;
            ep.r_c = L / 4;
            ep.eps = pc.eps;
            ep.n_dim = n_dim;
            ep.kernel = c.kernel;
            if (c.kernel == DMK_YUKAWA)
                ep.fparam = c.fparam;
            ep.eval_type = eval;
            ep.log_level = 6;
            dmk::EspPlan<double> plan(ep);
            auto esp = plan.eval(n_src, r_esp.data(), charges.data());

            double e2p = 0, r2p = 0, e2g = 0, r2g = 0;
            for (int i = 0; i < n_test; ++i) {
                const double dp = esp.pot[i] - ref_pot[i];
                e2p += dp * dp;
                r2p += ref_pot[i] * ref_pot[i];
                if (with_grad) {
                    const double f[3] = {esp.force_x[i], esp.force_y[i], n_dim == 3 ? esp.force_z[i] : 0.0};
                    for (int d = 0; d < n_dim; ++d) {
                        const double ref_force = -charges[i] * ref_grad[i * n_dim + d];
                        const double dg = f[d] - ref_force;
                        e2g += dg * dg;
                        r2g += ref_force * ref_force;
                    }
                }
            }
            const double l2p = dmk::pbc_ref::safe_l2(e2p, r2p);
            CHECK_MESSAGE(l2p < pc.tol_pot,
                          "n_digits=" << pc.n_digits << (with_grad ? " pot+grad" : " pot") << " pot l2=" << l2p);
            if (with_grad) {
                const double l2g = dmk::pbc_ref::safe_l2(e2g, r2g);
                CHECK_MESSAGE(l2g < pc.tol_grad, "n_digits=" << pc.n_digits << " force l2=" << l2g);
            }
        }
}

} // namespace

TEST_CASE("[ESP] 3d Laplace PBC vs Ewald") { run_esp_pbc({DMK_LAPLACE, 3, 0.0, true, 0, 99, 100}); }
TEST_CASE("[ESP] 3d Yukawa PBC vs lattice sum") { run_esp_pbc({DMK_YUKAWA, 3, 6.0, false, 6, 123, 50}); }
TEST_CASE("[ESP] 3d Sqrt-Laplace PBC vs Ewald") { run_esp_pbc({DMK_SQRT_LAPLACE, 3, 0.0, true, 0, 7, 100}); }
TEST_CASE("[ESP] 2d Yukawa PBC vs lattice sum") { run_esp_pbc({DMK_YUKAWA, 2, 4.0, false, 8, 321, 50}); }
TEST_CASE("[ESP] 2d Sqrt-Laplace PBC vs Ewald") { run_esp_pbc({DMK_SQRT_LAPLACE, 2, 0.0, true, 0, 54, 50}); }

// 2D Laplace (log): the ESP source potential carries a self term and a global k=0 gauge, so it is
// checked against the DMK periodic pipeline (which shares the same self convention) with the gauge
// removed. Forces are gauge-free.
TEST_CASE("[ESP] 2d Laplace PBC vs DMK pipeline (gauge-removed)") {
    constexpr int n_dim = 2;
    constexpr int n_src = 2000;
    constexpr int n_trg = 500;
    const double L = 1.0;

    std::default_random_engine eng(88);
    std::uniform_real_distribution<double> rng(0.01, 0.99);
    std::vector<double> r_src(n_dim * n_src), r_trg(n_dim * n_trg), charges(n_src), rnormal(n_dim * n_src, 0.0);
    for (auto &x : r_src)
        x = rng(eng);
    for (auto &x : r_trg)
        x = rng(eng);
    for (auto &q : charges)
        q = rng(eng) - 0.5;
    {
        double s = 0;
        for (double q : charges)
            s += q;
        for (double &q : charges)
            q -= s / n_src;
    }

    std::vector<double> r_esp(n_dim * n_src);
    for (int i = 0; i < n_dim * n_src; ++i)
        r_esp[i] = r_src[i] - 0.5 * L;

    const int n_cmp = std::min(n_src, 50);
    struct PrecisionCase {
        int n_digits;
        double eps, tol_pot, tol_grad;
    };
    const PrecisionCase cases[] = {{3, 1e-3, 1e-2, 1e-1}, {6, 1e-6, 1e-4, 1e-3}, {9, 1e-9, 1e-7, 1e-6}};
    for (const auto &pc : cases)
        for (int with_grad = 0; with_grad <= 1; ++with_grad) {
            const auto eval = with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
            const int odim = with_grad ? 1 + n_dim : 1;

            // DMK periodic pipeline reference at the sources (shares ESP's log self convention).
            pdmk_params params;
            params.eps = pc.eps;
            params.n_dim = n_dim;
            params.n_per_leaf = 50;
            params.eval_src = eval;
            params.eval_trg = eval;
            params.kernel = DMK_LAPLACE;
            params.use_periodic = true;
            params.log_level = 6;
            std::vector<double> pot_src(n_src * odim), pot_trg(n_trg * odim);
            pdmk_tree tree = pdmk_tree_create(nullptr, params, n_src, r_src.data(), charges.data(), rnormal.data(),
                                              n_trg, r_trg.data());
            pdmk_tree_eval(tree, pot_src.data(), pot_trg.data());
            pdmk_tree_destroy(tree);

            pdmk_esp_params ep{};
            ep.L = L;
            ep.r_c = L / 4;
            ep.eps = pc.eps;
            ep.n_dim = 2;
            ep.kernel = DMK_LAPLACE;
            ep.eval_type = eval;
            ep.log_level = 6;
            dmk::EspPlan<double> plan(ep);
            auto esp = plan.eval(n_src, r_esp.data(), charges.data());

            double gauge = 0, pbar = 0;
            for (int i = 0; i < n_cmp; ++i) {
                gauge += esp.pot[i] - pot_src[i * odim];
                pbar += pot_src[i * odim];
            }
            gauge /= n_cmp;
            pbar /= n_cmp;
            double e2p = 0, r2p = 0, e2g = 0, r2g = 0;
            for (int i = 0; i < n_cmp; ++i) {
                const double dp = (esp.pot[i] - pot_src[i * odim]) - gauge;
                e2p += dp * dp;
                r2p += (pot_src[i * odim] - pbar) * (pot_src[i * odim] - pbar);
                if (with_grad) {
                    const double f[2] = {esp.force_x[i], esp.force_y[i]};
                    for (int d = 0; d < n_dim; ++d) {
                        const double ref_force = -charges[i] * pot_src[i * odim + 1 + d];
                        const double dg = f[d] - ref_force;
                        e2g += dg * dg;
                        r2g += ref_force * ref_force;
                    }
                }
            }
            CHECK(dmk::pbc_ref::safe_l2(e2p, r2p) < pc.tol_pot);
            if (with_grad)
                CHECK(dmk::pbc_ref::safe_l2(e2g, r2g) < pc.tol_grad);
        }
}
#endif // DMK_BUILD_ESP
