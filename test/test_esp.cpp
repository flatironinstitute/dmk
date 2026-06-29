#ifdef DMK_BUILD_ESP

#include <dmk.h>
#include <doctest/extensions/doctest_mpi.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unistd.h>
#include <iostream>

// 10-particle fixture shared by both test cases
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

// Reference from the Python long-range-slow implementation
const double REFERENCE[10] = {
     0.055690493646334494, -0.006852575660153015,
    -0.044049084122909810,  0.040206609732000410,
    -0.057055432216358340,  0.063424832953350830,
    -0.069352639532568540,  0.092150894505931390,
     0.060859529062051890,  0.109678060779027990
};

} // namespace

MPI_TEST_CASE("[ESP] pdmk_esp double 10-particle reference", 1) {
    pdmk_esp_params params{};
    params.L   = L;
    params.r_c = R_C;
    params.eps = 1e-6;

    auto plan = pdmk_esp_plan_create(test_comm, params);
    double pot[N] = {};
    pdmk_esp_eval(test_comm, plan, N, R_SRC, CHARGES, pot);
    pdmk_esp_plan_destroy(plan);

    double esp_mean = 0, ref_mean = 0;
    for (int i = 0; i < N; ++i) { esp_mean += pot[i]; ref_mean += REFERENCE[i]; }
    esp_mean /= N; ref_mean /= N;

    double max_err = 0.0;
    for (int i = 0; i < N; ++i)
        max_err = std::max(max_err, std::abs((pot[i] - esp_mean) - (REFERENCE[i] - ref_mean)));
    CHECK(max_err < 5e-4);
}

MPI_TEST_CASE("[ESP] pdmk_esp float 10-particle reference", 1) {
    float r_src_f[30], charges_f[N];
    for (int i = 0; i < 30; ++i) r_src_f[i] = float(R_SRC[i]);
    for (int i = 0; i < N;  ++i) charges_f[i] = float(CHARGES[i]);

    pdmk_esp_params params{};
    params.L   = L;
    params.r_c = R_C;
    params.eps = 1e-4;  // float precision doesn't benefit from tighter eps

    auto plan = pdmk_esp_plan_createf(test_comm, params);
    float pot[N] = {};
    pdmk_esp_evalf(test_comm, plan, N, r_src_f, charges_f, pot);
    pdmk_esp_plan_destroyf(plan);

    double esp_mean = 0, ref_mean = 0;
    for (int i = 0; i < N; ++i) { esp_mean += double(pot[i]); ref_mean += REFERENCE[i]; }
    esp_mean /= N; ref_mean /= N;

    double max_err = 0.0;
    for (int i = 0; i < N; ++i)
        max_err = std::max(max_err, std::abs((double(pot[i]) - esp_mean) - (REFERENCE[i] - ref_mean)));
    CHECK(max_err < 5e-3);  // float accumulation gives ~1e-3 vs double reference
}

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

MPI_TEST_CASE("[ESP] REFERENCE values match perilap3d", 1) {
    double perilap[N];
    bool ok = run_perilap3d(N, R_SRC, CHARGES, perilap);
    REQUIRE_MESSAGE(ok, "perilap3d subprocess failed — check Python env and PERILAP3D_DIR");

    // REFERENCE was computed by Python long-range-slow; gauge-correct by subtracting its mean
    double ref_mean = 0;
    for (int i = 0; i < N; ++i) ref_mean += REFERENCE[i];
    ref_mean /= N;

    double max_err = 0;
    for (int i = 0; i < N; ++i)
        max_err = std::max(max_err, std::abs((REFERENCE[i] - ref_mean) - perilap[i]));
    CHECK_MESSAGE(max_err < 5e-4,
        "REFERENCE values disagree with perilap3d by " << max_err << " > 5e-4");
}

MPI_TEST_CASE("[ESP] pdmk_esp double matches perilap3d", 1) {
    pdmk_esp_params params{};
    params.L   = L;
    params.r_c = R_C;
    params.eps = 1e-6;

    auto plan = pdmk_esp_plan_create(test_comm, params);
    double pot[N] = {};
    pdmk_esp_eval(test_comm, plan, N, R_SRC, CHARGES, pot);
    pdmk_esp_plan_destroy(plan);

    double perilap[N];
    bool ok = run_perilap3d(N, R_SRC, CHARGES, perilap);
    REQUIRE_MESSAGE(ok, "perilap3d subprocess failed — check Python env and PERILAP3D_DIR");

    // gauge-correct ESP output
    double esp_mean = 0;
    for (int i = 0; i < N; ++i) esp_mean += pot[i];
    esp_mean /= N;

    double max_err = 0;
    for (int i = 0; i < N; ++i){
        max_err = std::max(max_err, std::abs((pot[i] - esp_mean) - perilap[i]));
        std::cout << "Point " << i << " - Perilap reference: " << perilap[i] << " ESP: " << pot[i] << " Difference: " << perilap[i] - pot[i] << " My reference: " << REFERENCE[i] << "\n";
    }
    CHECK_MESSAGE(max_err < 5e-4,
        "ESP vs perilap3d max_err = " << max_err << " > 5e-4 (eps=1e-6)");
}

#endif // DMK_BUILD_ESP
