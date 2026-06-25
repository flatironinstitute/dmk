#ifdef DMK_BUILD_ESP

#include <dmk.h>
#include <doctest/extensions/doctest_mpi.h>
#include <cmath>

// 10-particle fixture shared by both test cases
namespace {

constexpr int N = 10;
constexpr double L = 1.0, R_C = 0.2;

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

    double max_err = 0.0;
    for (int i = 0; i < N; ++i)
        max_err = std::max(max_err, std::abs(pot[i] - REFERENCE[i]));
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

    double max_err = 0.0;
    for (int i = 0; i < N; ++i)
        max_err = std::max(max_err, std::abs(double(pot[i]) - REFERENCE[i]));
    CHECK(max_err < 5e-3);  // float accumulation gives ~1e-3 vs double reference
}

#endif // DMK_BUILD_ESP
