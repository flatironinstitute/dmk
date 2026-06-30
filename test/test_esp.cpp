#ifdef DMK_BUILD_ESP

#include <dmk.h>
#include <doctest/extensions/doctest_mpi.h>
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

MPI_TEST_CASE("[ESP] 10-particle double vs perilap3d", 1) {
    pdmk_esp_params params{};
    params.L   = L;
    params.r_c = R_C;
    params.eps = 1e-5;

    auto plan = pdmk_esp_plan_create(test_comm, params);
    double pot[N] = {};
    pdmk_esp_eval(test_comm, plan, N, R_SRC, CHARGES, pot);
    pdmk_esp_plan_destroy(plan);

    double ref[N];
    bool ok = run_perilap3d(N, R_SRC, CHARGES, ref);
    REQUIRE_MESSAGE(ok, "perilap3d subprocess failed — check Python env and PERILAP3D_DIR");

    double esp_mean = 0;
    for (int i = 0; i < N; ++i) esp_mean += pot[i];
    esp_mean /= N;

    double err2 = 0, ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double diff = (pot[i] - esp_mean) - ref[i];
        err2 += diff * diff;
        ref2 += ref[i] * ref[i];
    }
    const double l2_rel_err = std::sqrt(err2 / ref2);
    CHECK_MESSAGE(l2_rel_err < params.eps,
        "10-particle l2_rel_err=" << l2_rel_err << " >= 1e-5");
}

// ---------------------------------------------------------------------------
// Long-range isolation test.
//
// Particles sit on a regular n_grid^3 cubic lattice with spacing h = L/n_grid.
// r_c < h guarantees every inter-particle distance exceeds r_c, so the
// short-range direct sum contributes exactly zero.  The full ESP output must
// therefore equal the long-range FINUFFT path alone.  This test isolates that
// path from the short-range polynomial accumulation that degrades for large N.
//
// perilap3d reference is cached in VERIFY_CACHE_DIR (shared with benchmark_esp).
// ---------------------------------------------------------------------------
MPI_TEST_CASE("[ESP] long-range only: regular grid, no short-range pairs", 1) {
    constexpr int n_grid = 8;
    constexpr int N     = n_grid * n_grid * n_grid;  // 512
    constexpr double L   = 1.0;
    constexpr double h   = L / n_grid;               // 0.125 — nearest-neighbour distance
    constexpr double r_c = 0.05;                     // < h: guaranteed no short-range pairs
    constexpr double eps = 1e-6;
    static_assert(r_c < h, "r_c must be < grid spacing for the short-range sum to be zero");

    // Regular grid positions in [-L/2, L/2)^3; alternating ±1 charges (charge-neutral).
    double r_src[N * 3];
    double charges[N];
    for (int ix = 0; ix < n_grid; ++ix)
        for (int iy = 0; iy < n_grid; ++iy)
            for (int iz = 0; iz < n_grid; ++iz) {
                const int i = (ix * n_grid + iy) * n_grid + iz;
                r_src[3*i+0] = (ix + 0.5) * h - 0.5 * L;
                r_src[3*i+1] = (iy + 0.5) * h - 0.5 * L;
                r_src[3*i+2] = (iz + 0.5) * h - 0.5 * L;
                charges[i]   = (i % 2 == 0) ? 1.0 : -1.0;
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
    pdmk_esp_params params{};
    params.L   = L;
    params.r_c = r_c;
    params.eps = eps;

    auto plan = pdmk_esp_plan_create(test_comm, params);
    double pot[N] = {};
    pdmk_esp_eval(test_comm, plan, N, r_src, charges, pot);
    pdmk_esp_plan_destroy(plan);

    // Gauge-correct ESP output and compare to the (already zero-mean) perilap3d reference.
    double esp_mean = 0;
    for (int i = 0; i < N; ++i) esp_mean += pot[i];
    esp_mean /= N;

    double err2 = 0, ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double diff = (pot[i] - esp_mean) - ref[i];
        err2 += diff * diff;
        ref2 += ref[i] * ref[i];
    }
    const double l2_rel_err = std::sqrt(err2 / ref2);

    CHECK_MESSAGE(l2_rel_err < 5 * eps,
        "long-range-only l2_rel_err=" << l2_rel_err << " >= eps=" << eps);
}

// ---------------------------------------------------------------------------
// Madelung constant test.
//
// NaCl 3D checkerboard: charges[i] = ((ix+iy+iz)%2==0) ? +1 : -1.
// ESP uses the 1/(4πr) kernel, so the periodic Madelung potential at each
// site is  phi_i = -q_i * M_NaCl / (4π h),  where h is the nearest-neighbour
// distance and M_NaCl = 1.7475645946331821906.
//
// r_c < h so the short-range sum is identically zero; the entire result comes
// from the FINUFFT long-range path.  No perilap3d reference is needed — the
// expected values are analytical.
// ---------------------------------------------------------------------------
MPI_TEST_CASE("[ESP] Madelung constant: NaCl lattice, 1/(4pi*r) kernel", 1) {
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

    pdmk_esp_params params{};
    params.L   = L;
    params.r_c = r_c;
    params.eps = eps;

    auto plan = pdmk_esp_plan_create(test_comm, params);
    double pot[N] = {};
    pdmk_esp_eval(test_comm, plan, N, r_src, charges, pot);
    pdmk_esp_plan_destroy(plan);

    // NaCl is charge-neutral so the mean potential is zero; gauge-correct anyway.
    double esp_mean = 0;
    for (int i = 0; i < N; ++i) esp_mean += pot[i];
    esp_mean /= N;

    // Analytical reference: phi_i = -q_i * M_NaCl / (4π h).
    // The expected values are also zero-mean (sum q_i = 0), so gauge correction
    // cancels and the comparison is clean.
    double err2 = 0, ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double expected = -charges[i] * M_NaCl / (4.0 * M_PI * h);
        const double diff     = (pot[i] - esp_mean) - expected;
        err2 += diff * diff;
        ref2 += expected * expected;
    }
    const double l2_rel_err = std::sqrt(err2 / ref2);

    CHECK_MESSAGE(l2_rel_err < eps,
        "Madelung l2_rel_err=" << l2_rel_err << " >= eps=" << eps
        << " (M_NaCl=" << M_NaCl << ", h=" << h << ")");
}

// ---------------------------------------------------------------------------
// Short-range stress test.
//
// N particles are packed inside a sphere of radius r_c/4, so every pairwise
// distance is at most 2*(r_c/4) = r_c/2 < r_c.  All N(N-1)/2 pairs go
// through the short-range polynomial path; none are skipped by the cell-list.
// The long-range (FINUFFT) path still runs and contributes, so the total ESP
// is compared against a perilap3d reference.  Errors in the short-range
// polynomial accumulate as O(N²) pairs, making this a sensitive stress test.
//
// Positions are generated deterministically (mt19937, seed 12345) via
// rejection sampling inside the sphere.  Reference is cached in
// VERIFY_CACHE_DIR under perilap_short_N{N}.bin.
// ---------------------------------------------------------------------------
MPI_TEST_CASE("[ESP] short-range stress: all pairs within r_c", 1) {
    constexpr int    N        = 500;
    constexpr double L        = 1.0;
    constexpr double r_c      = 0.05;
    constexpr double eps      = 1e-3;
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
    pdmk_esp_params params{};
    params.L   = L;
    params.r_c = r_c;
    params.eps = eps;

    auto plan = pdmk_esp_plan_create(test_comm, params);
    double pot[N] = {};
    pdmk_esp_eval(test_comm, plan, N, r_src, charges, pot);
    pdmk_esp_plan_destroy(plan);

    double esp_mean = 0;
    for (int i = 0; i < N; ++i) esp_mean += pot[i];
    esp_mean /= N;

    double err2 = 0, ref2 = 0;
    for (int i = 0; i < N; ++i) {
        const double diff = (pot[i] - esp_mean) - ref[i];
        err2 += diff * diff;
        ref2 += ref[i] * ref[i];
    }
    const double l2_rel_err = std::sqrt(err2 / ref2);

    CHECK_MESSAGE(l2_rel_err < eps,
        "short-range-stress l2_rel_err=" << l2_rel_err << " >= eps=" << eps);
}

#endif // DMK_BUILD_ESP
