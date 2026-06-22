// #include <cstdio>
// #include <vector>
// #include <omp.h>
// #include "bin_sort.hpp"

// int main() {
//     static const double PI = 3.141592653589793238462643383279502884;

//     // ============================================================
//     // Test points from get_test2_input() in esp.py
//     // Positions shifted from [0,1] to [-0.5, 0.5] then scaled to [-pi, pi)
//     // ============================================================
//     int M  = 10;
//     int nf = 32; // grid size in each dimension

//     std::vector<double> kx = {
//         (0.131538-0.5)*2*PI, (0.45865-0.5)*2*PI,   (0.218959-0.5)*2*PI,
//         (0.678865-0.5)*2*PI, (0.934693-0.5)*2*PI,   (0.519416-0.5)*2*PI,
//         (0.0345721-0.5)*2*PI,(0.5297-0.5)*2*PI,     (0.00769819-0.5)*2*PI,
//         (0.0668422-0.5)*2*PI
//     };
//     std::vector<double> ky = {
//         (0.686773-0.5)*2*PI, (0.930436-0.5)*2*PI,   (0.526929-0.5)*2*PI,
//         (0.653919-0.5)*2*PI, (0.701191-0.5)*2*PI,   (0.762198-0.5)*2*PI,
//         (0.0474645-0.5)*2*PI,(0.328234-0.5)*2*PI,   (0.75641-0.5)*2*PI,
//         (0.365339-0.5)*2*PI
//     };
//     std::vector<double> kz = {
//         (0.98255-0.5)*2*PI,  (0.753356-0.5)*2*PI,   (0.0726859-0.5)*2*PI,
//         (0.884707-0.5)*2*PI, (0.436411-0.5)*2*PI,   (0.477732-0.5)*2*PI,
//         (0.274907-0.5)*2*PI, (0.166507-0.5)*2*PI,   (0.897656-0.5)*2*PI,
//         (0.0605643-0.5)*2*PI
//     };
//     std::vector<double> charges = {0.2,-0.2,0.3,-0.3,0.4,-0.4,0.5,-0.5,0.1,-0.1};

//     // FINUFFT default bin sizes
//     //double bin_size_x = 16.0, bin_size_y = 4.0, bin_size_z = 4.0;
//     double bin_size_x = 8.0, bin_size_y = 4.0, bin_size_z = 4.0;
//     int nthr = omp_get_max_threads();
//     printf("Using %d threads\n", nthr);

//     // ============================================================
//     // Step 1: sort points
//     // ============================================================
//     std::vector<BIGINT> sort_indices(M);
//     bin_sort_multithread_impl<double, 3>(
//         sort_indices, (UBIGINT)M,
//         kx.data(), ky.data(), kz.data(),
//         (UBIGINT)nf, (UBIGINT)nf, (UBIGINT)nf,
//         bin_size_x, bin_size_y, bin_size_z,
//         nthr);

//     printf("\nSorted order:\n");
//     for (int i = 0; i < M; i++) {
//         int j = (int)sort_indices[i];
//         printf("  sorted[%2d] = original pt %d  "
//                "(x=%6.3f y=%6.3f z=%6.3f charge=%+.1f)\n",
//                i, j, kx[j], ky[j], kz[j], charges[j]);
//     }

//     return 0;
// }
#include <cstdio>
#include <vector>
#include <omp.h>
#include "bin_sort.hpp"

int main() {
    static const double PI = 3.141592653589793238462643383279502884;
    const double lo = -PI / 2.0;
    const double hi = +PI / 2.0;

    int nf = 32;

    // 8 corners of a cube in [-pi, pi), shuffled
    // Original spatial order (x fastest, z slowest) would be:
    // z=lo: (x=lo,y=lo)=pt1, (x=hi,y=lo)=pt6, (x=lo,y=hi)=pt3, (x=hi,y=hi)=pt5
    // z=hi: (x=lo,y=lo)=pt4, (x=hi,y=lo)=pt2, (x=lo,y=hi)=pt7, (x=hi,y=hi)=pt0
    //
    // Shuffled input:
    // 0: (hi, hi, hi)
    // 1: (lo, lo, lo)
    // 2: (hi, lo, hi)
    // 3: (lo, hi, lo)
    // 4: (lo, lo, hi)
    // 5: (hi, hi, lo)
    // 6: (hi, lo, lo)
    // 7: (lo, hi, hi)
    std::vector<double> kx = {hi, lo, hi, lo, lo, hi, hi, lo};
    std::vector<double> ky = {hi, lo, lo, hi, lo, hi, lo, hi};
    std::vector<double> kz = {hi, lo, hi, lo, hi, lo, lo, hi};
    std::vector<double> charges = {1, 2, 3, 4, 5, 6, 7, 8}; // label each point

    int M    = (int)kx.size();
    int nthr = omp_get_max_threads();

    double bin_size_x = 4.0, bin_size_y = 4.0, bin_size_z = 4.0;

    std::vector<BIGINT> sort_indices(M);
    bin_sort_multithread_impl<double, 3>(
        sort_indices, (UBIGINT)M,
        kx.data(), ky.data(), kz.data(),
        (UBIGINT)nf, (UBIGINT)nf, (UBIGINT)nf,
        bin_size_x, bin_size_y, bin_size_z,
        nthr);

    printf("Sorted order (expect z slowest, y medium, x fastest):\n");
    printf("  %-10s %-5s %-8s %-8s %-8s %-8s\n",
           "sorted pos", "orig", "x", "y", "z", "charge");
    for (int i = 0; i < M; i++) {
        int j = (int)sort_indices[i];
        printf("  %-10d %-5d %-8.3f %-8.3f %-8.3f %-8.0f\n",
               i, j, kx[j], ky[j], kz[j], charges[j]);
    }

    // expected: z=lo first, z=hi second
    // z=lo: pt1(lo,lo), pt6(hi,lo), pt3(lo,hi), pt5(hi,hi)
    // z=hi: pt4(lo,lo), pt2(hi,lo), pt7(lo,hi), pt0(hi,hi)
    std::vector<int> expected = {1, 6, 3, 5, 4, 2, 7, 0};
    printf("\nExpected original indices: ");
    for (int e : expected) printf("%d ", e);
    printf("\nActual   original indices: ");
    for (int i = 0; i < M; i++) printf("%lld ", sort_indices[i]);
    printf("\n");

    bool correct = true;
    for (int i = 0; i < M; i++)
        if (sort_indices[i] != expected[i]) { correct = false; break; }
    printf("\nTest %s\n", correct ? "PASSED" : "FAILED");

    return 0;
}