#include <cassert>
#include <limits>

#include <dmk/chebychev.hpp>

template <typename T>
T testfunc(const T *x) {
    return std::sin(*x * *x * *x) + 0.5;
}

template <typename T>
void translation_test(int order) {
    T lb{-1.0}, ub{1.0};

    Eigen::MatrixX<T> tm, tp;
    std::tie(tm, tp) = dmk::chebyshev::parent_to_child_matrices<T>(order);
    Eigen::VectorX<T> coeffs = dmk::chebyshev::fit(order, testfunc<T>, lb, ub);
    Eigen::VectorX<T> coeffs_m = tm * coeffs;
    Eigen::VectorX<T> coeffs_p = tp * coeffs;

    for (T x = -1.0; x <= 0.0; x += 0.01) {
        T res = dmk::chebyshev::cheb_eval(order, x, coeffs.data());
        T res_alt = dmk::chebyshev::cheb_eval(order, 2 * x + T{1.0}, coeffs_m.data());
        assert(std::fabs(res - res_alt) < 3 * std::numeric_limits<T>::epsilon());
    }

    for (T x = 0.0; x <= 1.0; x += 0.01) {
        T res = dmk::chebyshev::cheb_eval(order, x, coeffs.data());
        T res_alt = dmk::chebyshev::cheb_eval(order, 2 * x - T{1.0}, coeffs_p.data());
        assert(std::fabs(res - res_alt) < 3 * std::numeric_limits<T>::epsilon());
    }
}

int main(int argc, char *argv[]) {

    for (int order = 8; order < 32; order += 2) {
        translation_test<float>(order);
        translation_test<double>(order);
    }

    return 0;
}
