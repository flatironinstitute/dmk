#include <cstdlib>
#include <cstring>
#include <dmk/chebychev.hpp>
#include <iostream>
#include <omp.h>
#include <string>
#include <type_traits>

template <typename T>
void print_results(int order, std::string strat, double dt, const Eigen::Ref<Eigen::VectorX<T>> &results) {
    printf("%5d %c %6s %06.5f %7.2f %7.3f %.15f\n", order, std::is_same_v<T, float> ? 'f' : 'd', strat.c_str(), dt,
           results.size() / dt / 1e6, (dt * 1e9) / results.size(), results.mean());
}

template <typename T, int order>
void test_poly(int N) {
    T lb{-1.0}, ub{1.0};
    Eigen::VectorX<T> X = Eigen::VectorX<T>::LinSpaced(N, lb, ub);
    Eigen::MatrixX<T> results = Eigen::MatrixX<T>::Zero(order, N);
    std::string strat = "dyn";

    double t = -omp_get_wtime();
    dmk::chebyshev::calc_polynomial<T>(order, X, results);
    t += omp_get_wtime();
    X = results.row(order - 1);
    print_results<T>(order, strat, t, X);
}

int main(int argc, char *argv[]) {
    int N = 1e6;
    printf("# Evaluation benchmarks\n");
    printf("order t method elapsed Meval/s ns/eval result_mean\n");

    test_poly<float, 8>(N);
    test_poly<float, 16>(N);
    test_poly<float, 24>(N);
    test_poly<float, 32>(N);

    test_poly<double, 8>(N);
    test_poly<double, 16>(N);
    test_poly<double, 24>(N);
    test_poly<double, 32>(N);

    return 0;
}
