#include <array>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

using Vec3 = std::array<double, 3>;

// Declared in esp.cpp
struct ESPResult {
    std::vector<double> total;
    std::vector<double> short_range_pot;
    std::vector<double> long_range_pot;
    std::vector<double> self_pot;
};
ESPResult esp_potential(const std::vector<Vec3> &r_src,
                        const std::vector<double> &charges,
                        double L, double r_c, int P, double eps);

void debug_pswf(double eps, double L, double r_c, int P, int n);

// ---------------------------------------------------------------------------
// Test case scaffolding
// ---------------------------------------------------------------------------
struct TestCase {
    const char*         name;
    std::vector<Vec3>   r_src;
    std::vector<double> charges;
    std::vector<double> reference;   // empty => no reference to compare against
};

// Test 1 — 10 fixed particles, with Python reference values
static TestCase make_test_10() {
    TestCase tc;
    tc.name = "10 particles (fixed, with reference)";
    tc.r_src = {
        { 0.131538  -0.5, 0.686773 -0.5, 0.98255   -0.5 },
        { 0.45865   -0.5, 0.930436 -0.5, 0.753356  -0.5 },
        { 0.218959  -0.5, 0.526929 -0.5, 0.0726859 -0.5 },
        { 0.678865  -0.5, 0.653919 -0.5, 0.884707  -0.5 },
        { 0.934693  -0.5, 0.701191 -0.5, 0.436411  -0.5 },
        { 0.519416  -0.5, 0.762198 -0.5, 0.477732  -0.5 },
        { 0.0345721 -0.5, 0.0474645-0.5, 0.274907  -0.5 },
        { 0.5297    -0.5, 0.328234 -0.5, 0.166507  -0.5 },
        { 0.00769819-0.5, 0.75641  -0.5, 0.897656  -0.5 },
        { 0.0668422 -0.5, 0.365339 -0.5, 0.0605643 -0.5 }
    };
    tc.charges = { 0.2,-0.2, 0.3,-0.3, 0.4,-0.4, 0.5,-0.5, 0.1,-0.1 };
    // from python (long-range-slow, treated as ground truth)
    tc.reference = {
         0.055690493646334494, -0.006852575660153015,
        -0.044049084122909810,  0.040206609732000410,
        -0.057055432216358340,  0.063424832953350830,
        -0.069352639532568540,  0.092150894505931390,
         0.060859529062051890,  0.109678060779027990
    };
    return tc;
}

// Test 2 — N random particles, alternating charges (no reference)
static TestCase make_test_random(int N) {
    TestCase tc;
    tc.name = "random particles";
    tc.r_src.resize(N);
    tc.charges.resize(N);
    srand(42);
    for (int i = 0; i < N; ++i) {
        tc.r_src[i] = { (double)rand()/RAND_MAX - 0.5,
                        (double)rand()/RAND_MAX - 0.5,
                        (double)rand()/RAND_MAX - 0.5 };
        tc.charges[i] = (i % 2 == 0) ? 1.0/100 : -1.0/100;  // total charge 0
    }
    return tc;
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------
static void run_test(const TestCase &tc,
                     double L, double r_c, int P, double eps) {
    printf("=== Test: %s (N=%zu) ===\n", tc.name, tc.charges.size());

    auto res = esp_potential(tc.r_src, tc.charges, L, r_c, P, eps);

    const bool has_ref = !tc.reference.empty();
    const int  n       = static_cast<int>(tc.charges.size());

    // Only dump per-particle detail for small cases; just summarize big ones.
    if (n <= 32) {
        for (int i = 0; i < n; ++i) {
            printf("Point %d\n", i);
            printf("  short-range : %+.8f\n", res.short_range_pot[i]);
            printf("  long-range  : %+.8f\n", res.long_range_pot[i]);
            printf("  self        : %+.8f\n", res.self_pot[i]);
            printf("  total       : %+.8f\n", res.total[i]);
            if (has_ref)
                printf("  error       : %+.2e\n", res.total[i] - tc.reference[i]);
            printf("\n");
        }
    }

    if (has_ref) {
        double max_abs = 0.0;
        for (int i = 0; i < n; ++i)
            max_abs = std::max(max_abs, std::abs(res.total[i] - tc.reference[i]));
        printf("max |error| vs reference: %.3e\n", max_abs);
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Pick the test from argv[1], default to the random benchmark.
    // Usage:  ./esp [test10 | random | pswf]
    const char* which = "test10";

    const double L = 1.0, r_c = 0.2, eps = 1e-6;
    const int    P = 7;

    if (std::strcmp(which, "pswf") == 0) {
        debug_pswf(eps, L, r_c, P, 10);
    } else if (std::strcmp(which, "test10") == 0) {
        run_test(make_test_10(), L, r_c, P, eps);
    } else if (std::strcmp(which, "random") == 0) {
        run_test(make_test_random(10000), L, r_c, P, eps);
    } else {
        fprintf(stderr, "unknown test '%s' (use: test10 | random | pswf)\n", which);
        return 1;
    }

    return 0;
}