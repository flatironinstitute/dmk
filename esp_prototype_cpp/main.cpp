#include <array>
#include <cstdio>
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

int main() {
    debug_pswf(1e-6, 1.0, 0.2, 7, 10);
    // Test 1 - 10 particles
    std::vector<Vec3> r_src = {{
        { 0.131538-0.5, 0.686773-0.5, 0.98255 -0.5 },
        { 0.45865 -0.5, 0.930436-0.5, 0.753356-0.5 },
        { 0.218959-0.5, 0.526929-0.5, 0.0726859-0.5 },
        { 0.678865-0.5, 0.653919-0.5, 0.884707-0.5 },
        { 0.934693-0.5, 0.701191-0.5, 0.436411-0.5 },
        { 0.519416-0.5, 0.762198-0.5, 0.477732-0.5 },
        { 0.0345721-0.5, 0.0474645-0.5, 0.274907-0.5 },
        { 0.5297  -0.5, 0.328234-0.5, 0.166507-0.5 },
        { 0.00769819-0.5, 0.75641-0.5, 0.897656-0.5 },
        { 0.0668422-0.5, 0.365339-0.5, 0.0605643-0.5 }
    }};
    std::vector<double> charges = { 0.2,-0.2, 0.3,-0.3, 0.4,-0.4, 0.5,-0.5, 0.1,-0.1 };

    const double reference[] = { //from python calculation (based on long range slow, which is surely accurate)
        0.055690493646334494,
        -0.006852575660153015,
        -0.044049084122909810,
        0.040206609732000410,
        -0.057055432216358340,
        0.063424832953350830,
        -0.069352639532568540,
        0.092150894505931390,
        0.060859529062051890,
        0.109678060779027990
    };

    //Test 2 - 10000 particles (random positions)
    // const int N = 10000;
    // srand(42);
    // std::vector<Vec3>   r_src(N);
    // std::vector<double> charges(N);
    // for (int i = 0; i < N; ++i) {
    //     r_src[i] = { (double)rand()/RAND_MAX - 0.5,
    //                  (double)rand()/RAND_MAX - 0.5,
    //                  (double)rand()/RAND_MAX - 0.5 };
    //     // alternate signs so total charge is 0
    //     charges[i] = (i % 2 == 0) ? 1.0/100 : -1.0/100;
    //}

    auto res = esp_potential(r_src, charges,
                             /*L=*/1.0, /*r_c=*/0.2, /*P=*/7, /*eps=*/1e-6);

    int n = static_cast<int>(charges.size());
    for (int i = 0; i < n; ++i) {
        printf("Point %d\n", i);
        printf("  short-range : %+.8f\n", res.short_range_pot[i]);
        printf("  long-range  : %+.8f\n", res.long_range_pot[i]);
        printf("  self        : %+.8f\n", res.self_pot[i]);
        printf("  total       : %+.8f\n", res.total[i]);
        printf("  error       : %+.2e\n", res.total[i] - reference[i]);
        printf("\n");
    }

    return 0;
}