#ifdef DMK_BUILD_ESP

#include <dmk.h>
#include <dmk/esp.hpp>
#include <doctest/doctest.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <unistd.h>

// 10-particle fixture
namespace {

constexpr int N = 10;
constexpr double L = 1.0, R_C = 0.05;

const double R_SRC[30] = {
    0.131538-0.5, 0.686773-0.5, 0.98255 -0.5,
    0.45865 -0.5, 0.930436-0.5, 0.753356-0.5,
    0.218959-0.5, 0.526929-0.5, 0.0726859-0.5,
    0.678865-0.5, 0.653919-0.5, 0.884707-0.5,
    0.934693-0.5, 0.701191-0.5, 0.436411-0.5,
    0.519416-0.5, 0.762198-0.5, 0.477732-0.5,
    0.0345721-0.5,0.0474645-0.5,0.274907-0.5,
    0.5297  -0.5, 0.328234-0.5, 0.166507-0.5,
    0.00769819-0.5,0.75641-0.5, 0.897656-0.5,
    0.0668422-0.5, 0.365339-0.5,0.0605643-0.5
};

const double CHARGES[10] = { 0.2,-0.2, 0.3,-0.3, 0.4,-0.4, 0.5,-0.5, 0.1,-0.1 };

} // namespace

// Helper: call verify_esp.py with given positions (flat N*3) and charges (N),
// fill ref_out[N] with zero-mean perilap3d potentials. Returns true on success.
static bool run_perilap3d(int n, const double *r_src_flat, const double *charges, double *ref_out) {
    char tmpfile[] = "/tmp/esp_test_data_XXXXXX";
    int fd = mkstemp(tmpfile);
    if (fd < 0) return false;
    int32_t nn = static_cast<int32_t>(n);
    write(fd, &nn, sizeof(nn));
    write(fd, r_src_flat, n * 3 * sizeof(double));
    write(fd, charges, n * sizeof(double));
    close(fd);

    char errfile[] = "/tmp/esp_test_err_XXXXXX";
    int efd = mkstemp(errfile);
    close(efd);

    std::string cmd = std::string("python3 ") + VERIFY_SCRIPT_PATH
                    + " " + tmpfile + " " + PERILAP3D_DIR
                    + " 2>" + errfile;
    FILE *pipe = popen(cmd.c_str(), "r");
    bool ok = pipe != nullptr;
    if (ok) {
        for (int i = 0; i < n; ++i)
            if (fscanf(pipe, "%lf", &ref_out[i]) != 1) { ok = false; break; }
        int rc = pclose(pipe);
        if (rc != 0) ok = false;
    }
    if (!ok) {
        if (FILE *ef = fopen(errfile, "r")) {
            char buf[256];
            while (fgets(buf, sizeof(buf), ef)) fputs(buf, stderr);
            fclose(ef);
        }
    }
    unlink(tmpfile);
    unlink(errfile);
    return ok;
}

// Flat [n*3] positions -> dmk::Vec3T<double> array, for calling dmk::esp_eval directly.
static std::vector<dmk::Vec3T<double>> to_vec3(int n, const double *r_flat) {
    std::vector<dmk::Vec3T<double>> r(n);
    for (int i = 0; i < n; ++i)
        r[i] = {r_flat[3 * i], r_flat[3 * i + 1], r_flat[3 * i + 2]};
    return r;
}

// Finite-difference force reference: F_{i,a} = -q_i * d(pot_i)/dr_{i,a}, via central differences.
// `plan` must already be created with DMK_POTENTIAL_GRAD and is not destroyed here.
// force_ref must have room for 3*n doubles.
static void fd_force_reference(dmk::EspPlan *plan, int n, const double *r_src,
                               const double *charges, double step_size, double *force_ref) {
    std::vector<double> r_pert(3 * n);
    std::vector<double> charges_vec(charges, charges + n);

    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < 3; ++a) {
            std::copy(r_src, r_src + 3 * n, r_pert.begin());

            r_pert[3 * i + a] += step_size;
            double pot_plus = dmk::esp_eval<double>(plan, to_vec3(n, r_pert.data()), charges_vec).pot[i];

            r_pert[3 * i + a] -= 2.0 * step_size;
            double pot_minus = dmk::esp_eval<double>(plan, to_vec3(n, r_pert.data()), charges_vec).pot[i];

            force_ref[3 * i + a] = -charges[i] * (pot_plus - pot_minus) / (2.0 * step_size);
        }
    }
}

// L2-relative error between ESP's analytic forces and a flat [3*n] (x,y,z per particle) reference.
static double force_l2_rel_err(int n, const std::vector<double> &force_x, const std::vector<double> &force_y,
                               const std::vector<double> &force_z, const double *force_ref) {
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

TEST_CASE("[ESP] 10-particle double vs perilap3d") {
    constexpr double eps = 1e-5;
    auto r_vec = to_vec3(N, R_SRC);
    std::vector<double> q_vec(CHARGES, CHARGES + N);

    dmk::EspPlan *plan = dmk::esp_create_plan(L, R_C, eps, 1.35, DMK_POTENTIAL);
    auto esp = dmk::esp_eval<double>(plan, r_vec, q_vec);
    dmk::esp_destroy_plan(plan);

    double ref[N];
    bool ok = run_perilap3d(N, R_SRC, CHARGES, ref);
    REQUIRE_MESSAGE(ok, "perilap3d subprocess failed — check Python env and PERILAP3D_DIR");

    double esp_mean = 0;
    for (int i = 0; i < N; ++i) esp_mean += esp.pot[i];
    esp_mean /= N;

    double err2 = 0, ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double diff = (esp.pot[i] - esp_mean) - ref[i];
        err2 += diff * diff;
        ref2 += ref[i] * ref[i];
    }
    const double l2_rel_err = std::sqrt(err2 / ref2);
    CHECK_MESSAGE(l2_rel_err < eps,
        "10-particle l2_rel_err=" << l2_rel_err << " >= " << eps);
}


// Long-range isolation test.
// Particles sit on a regular n_grid^3 cubic lattice with spacing h = L/n_grid.
// r_c < h guarantees every inter-particle distance exceeds r_c, so the
// short-range direct sum contributes exactly zero.
TEST_CASE("[ESP] long-range only: regular grid, no short-range pairs") {
    constexpr int n_grid = 4;
    constexpr int N     = n_grid * n_grid * n_grid; 
    constexpr double L   = 1.0;
    constexpr double h   = L / n_grid;               // nearest-neighbour distance
    constexpr double r_c = 0.05;                     // < h: guaranteed no short-range pairs
    constexpr double eps = 1e-6;
    static_assert(r_c < h, "r_c must be < grid spacing for the short-range sum to be zero");

    // Regular grid positions in [-L/2, L/2)^3; 3D checkerboard ±1 charges (charge-neutral).
    double r_src[N * 3];
    double charges[N];
    for (int ix = 0; ix < n_grid; ++ix)
        for (int iy = 0; iy < n_grid; ++iy)
            for (int iz = 0; iz < n_grid; ++iz) {
                const int i = (ix * n_grid + iy) * n_grid + iz;
                r_src[3*i+0] = (ix + 0.5) * h - 0.5 * L;
                r_src[3*i+1] = (iy + 0.5) * h - 0.5 * L;
                r_src[3*i+2] = (iz + 0.5) * h - 0.5 * L;
                charges[i]   = ((ix + iy + iz) % 2 == 0) ? 1.0 : -1.0;
            }

    // --- Load or compute perilap3d reference (cached in VERIFY_CACHE_DIR) ---
    double ref[N];
    bool have_ref = false;
    char cache_path[4096];
    snprintf(cache_path, sizeof(cache_path), "%s/perilap_N%d.bin", VERIFY_CACHE_DIR, N);

    if (FILE *cf = fopen(cache_path, "rb")) {
        int32_t cached_n = 0;
        double cached_pos[N * 3], cached_q[N];
        bool ok = fread(&cached_n, sizeof(cached_n), 1, cf) == 1
               && cached_n == static_cast<int32_t>(N)
               && (int)fread(cached_pos, sizeof(double), N * 3, cf) == N * 3
               && memcmp(cached_pos, r_src, N * 3 * sizeof(double)) == 0
               && (int)fread(cached_q,  sizeof(double), N,     cf) == N
               && memcmp(cached_q, charges, N * sizeof(double)) == 0
               && (int)fread(ref,        sizeof(double), N,     cf) == N;
        fclose(cf);
        have_ref = ok;
    }

    if (!have_ref) {
        have_ref = run_perilap3d(N, r_src, charges, ref);
        if (have_ref) {
            if (FILE *cf = fopen(cache_path, "wb")) {
                const int32_t nn = N;
                fwrite(&nn,    sizeof(nn),     1,     cf);
                fwrite(r_src,  sizeof(double), N * 3, cf);
                fwrite(charges,sizeof(double), N,     cf);
                fwrite(ref,    sizeof(double), N,     cf);
                fclose(cf);
            }
        }
    }

    REQUIRE_MESSAGE(have_ref,
        "perilap3d reference unavailable — check Python env and PERILAP3D_DIR");

    // --- Run ESP ---
    dmk::EspPlan *plan = dmk::esp_create_plan(L, r_c, eps, 1.35, DMK_POTENTIAL_GRAD);
    auto r_src_vec = to_vec3(N, r_src);
    std::vector<double> charges_vec(charges, charges + N);
    auto esp = dmk::esp_eval<double>(plan, r_src_vec, charges_vec);
    dmk::esp_destroy_plan(plan);

    // Gauge-correct ESP output and compare to the (already zero-mean) perilap3d reference.
    double esp_mean = 0;
    for (int i = 0; i < N; ++i) esp_mean += esp.pot[i];
    esp_mean /= N;

    double err2 = 0, ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double diff = (esp.pot[i] - esp_mean) - ref[i];
        err2 += diff * diff;
        ref2 += ref[i] * ref[i];
    }
    const double l2_rel_err = std::sqrt(err2 / ref2);

    CHECK_MESSAGE(l2_rel_err < 5 * eps,
        "long-range-only l2_rel_err=" << l2_rel_err << " >= 5 * eps=" << 5 * eps);

    // Long-range forces on a symmetric lattice should be ~0. Relative error is
    // undefined (ref ≈ 0), so we check absolute magnitude instead.
    double max_abs = 0;
    for (int i = 0; i < N; ++i) {
        max_abs = std::max({max_abs,
            std::abs(esp.force_x[i]), std::abs(esp.force_y[i]), std::abs(esp.force_z[i])});
    }
    double force_tol = 10 * std::pow(eps, 2.0 / 3.0);
    CHECK_MESSAGE(max_abs < force_tol,
        "long-range forces should vanish on symmetric lattice, max=" << max_abs);
}

// Madelung constant test.
TEST_CASE("[ESP] Madelung constant: NaCl lattice, 1/(4pi*r) kernel") {
    constexpr int n_grid = 8;
    constexpr int N      = n_grid * n_grid * n_grid;  // 512
    constexpr double L   = 1.0;
    constexpr double h   = L / n_grid;                // 0.125
    constexpr double r_c = 0.05;                      // < h: no short-range pairs
    constexpr double eps = 1e-4;
    constexpr double M_NaCl = 1.7475645946331821906;
    static_assert(r_c < h, "r_c must be < h for the NaCl short-range sum to be zero");

    double r_src[N * 3];
    double charges[N];
    for (int ix = 0; ix < n_grid; ++ix)
        for (int iy = 0; iy < n_grid; ++iy)
            for (int iz = 0; iz < n_grid; ++iz) {
                const int i = (ix * n_grid + iy) * n_grid + iz;
                r_src[3*i+0] = (ix + 0.5) * h - 0.5 * L;
                r_src[3*i+1] = (iy + 0.5) * h - 0.5 * L;
                r_src[3*i+2] = (iz + 0.5) * h - 0.5 * L;
                charges[i] = ((ix + iy + iz) % 2 == 0) ? 1.0 : -1.0;
            }

    dmk::EspPlan *plan = dmk::esp_create_plan(L, r_c, eps, 1.35, DMK_POTENTIAL);
    auto r_vec = to_vec3(N, r_src);
    std::vector<double> q_vec(charges, charges + N);
    auto esp = dmk::esp_eval<double>(plan, r_vec, q_vec);
    dmk::esp_destroy_plan(plan);

    // NaCl is charge-neutral so the mean potential is zero; gauge-correct anyway.
    double esp_mean = 0;
    for (int i = 0; i < N; ++i) esp_mean += esp.pot[i];
    esp_mean /= N;

    // Analytical reference: phi_i = -q_i * M_NaCl / (4π h).
    double err2 = 0, ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double expected = -charges[i] * M_NaCl / (4.0 * M_PI * h);
        const double diff     = (esp.pot[i] - esp_mean) - expected;
        err2 += diff * diff;
        ref2 += expected * expected;
    }
    const double l2_rel_err = std::sqrt(err2 / ref2);

    CHECK_MESSAGE(l2_rel_err < eps,
        "Madelung l2_rel_err=" << l2_rel_err << " >= eps=" << eps
        << " (M_NaCl=" << M_NaCl << ", h=" << h << ")");
}

// Short-range stress test.
// N particles are packed inside a sphere of radius r_c/4, so every pairwise
// distance is at most 2*(r_c/4) = r_c/2 < r_c.

TEST_CASE("[ESP] short-range stress: all pairs within r_c") {
    constexpr int    N        = 50;
    constexpr double L        = 1.0;
    constexpr double r_c      = 0.05;
    constexpr double eps      = 1e-4;
    constexpr double sphere_r = r_c / 4.0;  // max pairwise dist = 2*sphere_r = r_c/2 < r_c

    // Deterministic positions inside the sphere via rejection sampling.
    double r_src[N * 3];
    double charges[N];
    {
        std::mt19937 rng(12345u);
        std::uniform_real_distribution<double> uni(-sphere_r, sphere_r);
        int placed = 0;
        while (placed < N) {
            double x = uni(rng), y = uni(rng), z = uni(rng);
            if (x*x + y*y + z*z <= sphere_r * sphere_r) {
                r_src[3*placed+0] = x;
                r_src[3*placed+1] = y;
                r_src[3*placed+2] = z;
                charges[placed]   = (placed % 2 == 0) ? 1.0 : -1.0;
                ++placed;
            }
        }
    }

    // --- Load or compute perilap3d reference (cached in VERIFY_CACHE_DIR) ---
    double ref[N];
    bool have_ref = false;
    char cache_path[4096];
    snprintf(cache_path, sizeof(cache_path), "%s/perilap_short_N%d.bin", VERIFY_CACHE_DIR, N);

    if (FILE *cf = fopen(cache_path, "rb")) {
        int32_t cached_n = 0;
        double cached_pos[N * 3], cached_q[N];
        bool ok = fread(&cached_n, sizeof(cached_n), 1, cf) == 1
               && cached_n == static_cast<int32_t>(N)
               && (int)fread(cached_pos, sizeof(double), N * 3, cf) == N * 3
               && memcmp(cached_pos, r_src, N * 3 * sizeof(double)) == 0
               && (int)fread(cached_q,  sizeof(double), N,     cf) == N
               && memcmp(cached_q, charges, N * sizeof(double)) == 0
               && (int)fread(ref,        sizeof(double), N,     cf) == N;
        fclose(cf);
        have_ref = ok;
    }

    if (!have_ref) {
        have_ref = run_perilap3d(N, r_src, charges, ref);
        if (have_ref) {
            if (FILE *cf = fopen(cache_path, "wb")) {
                const int32_t nn = N;
                fwrite(&nn,     sizeof(nn),     1,     cf);
                fwrite(r_src,   sizeof(double), N * 3, cf);
                fwrite(charges, sizeof(double), N,     cf);
                fwrite(ref,     sizeof(double), N,     cf);
                fclose(cf);
            }
        }
    }

    REQUIRE_MESSAGE(have_ref,
        "perilap3d reference unavailable — check Python env and PERILAP3D_DIR");

    // --- Run ESP ---
    dmk::EspPlan *plan = dmk::esp_create_plan(L, r_c, eps, 1.35, DMK_POTENTIAL_GRAD);
    auto r_src_vec = to_vec3(N, r_src);
    std::vector<double> charges_vec(charges, charges + N);
    auto esp = dmk::esp_eval<double>(plan, r_src_vec, charges_vec);

    std::vector<double> force_ref(3 * N);
    fd_force_reference(plan, N, r_src, charges, 1e-6, force_ref.data());

    dmk::esp_destroy_plan(plan);

    double esp_mean = 0;
    for (int i = 0; i < N; ++i) esp_mean += esp.pot[i];
    esp_mean /= N;

    double err2 = 0, ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double diff = (esp.pot[i] - esp_mean) - ref[i];
        err2 += diff * diff;
        ref2 += ref[i] * ref[i];
    }
    const double l2_rel_err = std::sqrt(err2 / ref2);

    CHECK_MESSAGE(l2_rel_err < 5 * eps,
        "short-range-stress l2_rel_err=" << l2_rel_err << " >= eps=" << eps);

    const double force_l2_err = force_l2_rel_err(N, esp.force_x, esp.force_y, esp.force_z, force_ref.data());
    double force_tol = 5 * std::pow(eps, 2.0 / 3.0);
    CHECK_MESSAGE(force_l2_err <  force_tol,
        "short-range-stress forces l2_rel_err=" << force_l2_err << " >= force_tol=" << force_tol);
}


TEST_CASE("[ESP] forces - 10 particles") {
    constexpr double eps = 1e-6;
    auto r_src_vec = to_vec3(N, R_SRC);
    std::vector<double> charges_vec(CHARGES, CHARGES + N);

    dmk::EspPlan *plan = dmk::esp_create_plan(L, R_C, eps, 1.35, DMK_POTENTIAL_GRAD);
    auto esp = dmk::esp_eval<double>(plan, r_src_vec, charges_vec);

    std::vector<double> force_ref(3 * N);
    fd_force_reference(plan, N, R_SRC, CHARGES, std::pow(eps, 1.0 / 3.0), force_ref.data());

    dmk::esp_destroy_plan(plan);

    // Compare ESP's analytic forces against the finite-difference reference.
    const double l2_rel_err = force_l2_rel_err(N, esp.force_x, esp.force_y, esp.force_z, force_ref.data());
    double force_tol = 10 * std::pow(eps, 2.0 / 3.0);
    CHECK_MESSAGE(l2_rel_err <  force_tol,
        "10-particle forces l2_rel_err=" << l2_rel_err << " >= force_tol=" << force_tol);
}
#endif // DMK_BUILD_ESP
