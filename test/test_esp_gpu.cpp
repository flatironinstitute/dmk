#if defined(DMK_BUILD_ESP) && defined(DMK_GPU_OFFLOAD)

#include <dmk.h>
#include <dmk/esp.hpp>
#include <doctest/doctest.h>
#include <cmath>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Shared fixture — 100 random charge-neutral particles in [-0.5, 0.5)^3.
// ---------------------------------------------------------------------------
namespace {

constexpr int    N   = 100;
constexpr double L   = 1.0;
constexpr double R_C = 0.12;
constexpr double EPS = 1e-5;

struct Fixture {
    std::vector<dmk::Vec3T<double>> r;
    std::vector<double> q;
    dmk::EspPlan  *plan;
    dmk::GpuState *gpu;

    Fixture() {
        r.resize(N); q.resize(N);
        std::mt19937 rng(42u);
        std::uniform_real_distribution<double> uni(-0.5, 0.5);
        for (int i = 0; i < N; ++i) {
            r[i] = {uni(rng), uni(rng), uni(rng)};
            q[i] = (i % 2 == 0) ? 1.0 : -1.0;
        }
        plan = dmk::esp_create_plan(L, R_C, EPS, 1.35, DMK_POTENTIAL_GRAD);
        gpu  = dmk::esp_create_gpu_plan(plan);
    }
    ~Fixture() {
        dmk::esp_destroy_gpu_plan(gpu);
        dmk::esp_destroy_plan(plan);
    }
};

// L2-relative error; falls back to RMS absolute when reference is near zero.
static double l2_rel(const std::vector<double> &a, const std::vector<double> &b) {
    double err2 = 0, ref2 = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        err2 += d * d;
        ref2 += b[i] * b[i];
    }
    return ref2 > 0 ? std::sqrt(err2 / ref2) : std::sqrt(err2 / a.size());
}

static void gauge(std::vector<double> &v) {
    double m = 0;
    for (double x : v) m += x;
    m /= v.size();
    for (double &x : v) x -= m;
}

struct Result { std::vector<double> pot, fx, fy, fz; };

static Result snapshot(dmk::PotForce<double> pf) {
    return {
        std::vector<double>(pf.pot.begin(),     pf.pot.end()),
        std::vector<double>(pf.force_x.begin(), pf.force_x.end()),
        std::vector<double>(pf.force_y.begin(), pf.force_y.end()),
        std::vector<double>(pf.force_z.begin(), pf.force_z.end()),
    };
}

static void check(const Result &got, const Result &ref,
                  double pot_tol, double force_tol, const char *label) {
    auto gp = got.pot, rp = ref.pot;
    gauge(gp); gauge(rp);
    double ep = l2_rel(gp, rp);
    CHECK_MESSAGE(ep < pot_tol,   label << " pot   l2_rel=" << ep   << " tol=" << pot_tol);
    double ex = l2_rel(got.fx, ref.fx);
    double ey = l2_rel(got.fy, ref.fy);
    double ez = l2_rel(got.fz, ref.fz);
    CHECK_MESSAGE(ex < force_tol, label << " force_x l2_rel=" << ex << " tol=" << force_tol);
    CHECK_MESSAGE(ey < force_tol, label << " force_y l2_rel=" << ey << " tol=" << force_tol);
    CHECK_MESSAGE(ez < force_tol, label << " force_z l2_rel=" << ez << " tol=" << force_tol);
}

} // namespace

// ---------------------------------------------------------------------------
// Test 1: Short-range only.
// Calls esp_eval_short_range (CPU) and esp_eval_gpu_short_range (GPU) with
// the same particles.  Both evaluate the same polynomial kernel per pair;
// results should agree near machine epsilon (~1e-12 relative for double).
// ---------------------------------------------------------------------------
TEST_CASE("[ESP GPU] short-range: GPU vs CPU direct comparison") {
    Fixture f;
    auto cpu = snapshot(dmk::esp_eval_short_range(f.plan, f.r, f.q));
    auto gpu = snapshot(dmk::esp_eval_gpu_short_range(f.gpu, f.r, f.q));

    // Short-range polynomial is deterministic and identical on CPU and GPU;
    // expect near-machine-epsilon agreement.
    check(gpu, cpu, /*pot_tol=*/1e-10, /*force_tol=*/1e-10, "short-range");
}

// ---------------------------------------------------------------------------
// Test 2: Long-range only.
// Calls esp_eval_long_range (CPU) and esp_eval_gpu_long_range (GPU).
// CPU uses FINUFFT+DUCC0, GPU uses cuFINUFFT+cuFFT — different implementations
// of the same mathematical operation, so tolerance is algorithm eps.
// ---------------------------------------------------------------------------
TEST_CASE("[ESP GPU] long-range: GPU vs CPU direct comparison") {
    Fixture f;
    auto cpu = snapshot(dmk::esp_eval_long_range(f.plan, f.r, f.q));
    auto gpu = snapshot(dmk::esp_eval_gpu_long_range(f.gpu, f.r, f.q));

    check(gpu, cpu, /*pot_tol=*/5 * EPS, /*force_tol=*/5 * EPS, "long-range");
}

// ---------------------------------------------------------------------------
// Test 3: Full pipeline (short + long + self-correction).
// Calls esp_eval (CPU) and esp_eval_gpu (GPU).
// ---------------------------------------------------------------------------
TEST_CASE("[ESP GPU] full pipeline: GPU vs CPU direct comparison") {
    Fixture f;
    auto cpu = snapshot(dmk::esp_eval(f.plan, f.r, f.q));
    auto gpu = snapshot(dmk::esp_eval_gpu(f.gpu, f.r, f.q));

    check(gpu, cpu, /*pot_tol=*/5 * EPS, /*force_tol=*/5 * EPS, "full");
}

#endif // DMK_BUILD_ESP && DMK_GPU_OFFLOAD
