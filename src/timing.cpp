#include <dmk/omp_wrapper.hpp>

#ifndef _OPENMP
#include <chrono>

double MY_OMP_GET_WTIME() {
    using namespace std::chrono;
    static auto start = high_resolution_clock::now();
    auto now = high_resolution_clock::now();
    return duration_cast<duration<double>>(now - start).count();
}

#endif
