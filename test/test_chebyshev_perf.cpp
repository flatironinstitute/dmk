#include <cstdlib>
#include <dmk/chebychev.hpp>
#include <iostream>
#include <omp.h>
#include <string>
#include <type_traits>

template <typename T>
T testfunc(const T x) {
    return std::sin(x * x * x) + 0.5;
}

template <typename T>
void print_results(int order, std::string strat, double dt, const Eigen::Ref<Eigen::VectorX<T>> &results) {
    printf("%5d %c %6s %06.5f %7.2f %7.3f %.15f\n", order, std::is_same_v<T, float> ? 'f' : 'd', strat.c_str(), dt,
           results.size() / dt / 1e6, (dt * 1e9) / results.size(), results.mean());
}

template <typename T, int order>
void test_eigen(int N) {
    T lb{0}, ub{1.0};
    Eigen::VectorX<T> X = Eigen::VectorX<T>::LinSpaced(N, lb, ub);
    Eigen::VectorX<T> results = Eigen::VectorX<T>::Zero(N);

    Eigen::VectorX<T> coeffs = dmk::chebyshev::fit(order, testfunc<T>, lb, ub);
    std::string strat = "dyn";

    double st = omp_get_wtime();
    for (int i = 0; i < X.size(); ++i)
        results[i] = dmk::chebyshev::cheb_eval(X[i], order, lb, ub, coeffs.data());

    print_results<T>(order, strat, omp_get_wtime() - st, results);
}

template <typename T, int order, int VecLen>
void test_simd(int N) {
    T lb{0}, ub{1.0};

    Eigen::VectorX<T> X = Eigen::VectorX<T>::LinSpaced(N, 0.0, 1.0);
    Eigen::VectorX<T> results = Eigen::VectorX<T>::Zero(N);
    Eigen::VectorX<T> coeffs = dmk::chebyshev::fit(order, testfunc<T>, lb, ub);
    double st = omp_get_wtime();
    dmk::chebyshev::cheb_eval<T, VecLen>(order, X.size(), lb, ub, X.data(), coeffs.data(), results.data());
    print_results<T>(order, "simd" + std::to_string(VecLen), omp_get_wtime() - st, results);
}

template <typename T, int order>
void test_all(int N) {
    test_eigen<T, order>(N);
    test_simd<T, order, 4>(N);
    test_simd<T, order, 8>(N);
    if (std::is_same_v<float, T>)
        test_simd<T, order, 16>(N);
}

int main(int argc, char *argv[]) {
    int N = 1e8;
    printf("order t method elapsed Meval/s ns/eval result_mean\n");

    test_all<float, 8>(N);
    test_all<float, 16>(N);
    test_all<float, 24>(N);
    test_all<float, 32>(N);

    test_all<double, 8>(N);
    test_all<double, 16>(N);
    test_all<double, 24>(N);
    test_all<double, 32>(N);

    return 0;
}
