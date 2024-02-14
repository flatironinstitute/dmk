#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <limits>

#include <dmk/chebychev.hpp>
using namespace dmk::chebyshev;

template <typename T>
T testfunc(const T x) {
    return std::sin(x * x * x) + 0.5;
}

template <typename T>
void fit_test(int order) {
    T lb{-1.0}, ub{1.0};
    // return by value
    Eigen::VectorX<T> coeffs_rbv = fit(order, testfunc<T>, lb, ub);
    // pass by reference
    Eigen::VectorX<T> coeffs_pbr(order);
    fit(order, testfunc<T>, lb, ub, coeffs_pbr.data());

    assert(coeffs_rbv == coeffs_pbr);
}

template <typename T>
void interp_test(int order) {
    // Check that automatic interpolation by passing bounds works the same as the implicit [-1.0, 1.0]
    T lb{-1.0}, ub{1.0};
    Eigen::VectorX<T> coeffs = fit(order, testfunc<T>, lb, ub);

    for (T x = lb; x <= ub; x += 0.01) {
        T res = evaluate(x, order, coeffs.data());
        T res_alt = evaluate(x, order, lb, ub, coeffs.data());
        assert(std::fabs(res - res_alt) <= std::numeric_limits<T>::epsilon());
    }
}

template <typename T>
void translation_test(int order) {
    // Check that parent->child translation matrices work to reasonable precision
    // Larger bounds shifts -> larger errors.
    T lb{-1.3}, ub{1.2};
    T mid = lb + 0.5 * (ub - lb);

    Eigen::MatrixX<T> tm, tp;
    std::tie(tm, tp) = parent_to_child_matrices<T>(order);
    Eigen::VectorX<T> coeffs = fit(order, testfunc<T>, lb, ub);
    Eigen::VectorX<T> coeffs_m = tm * coeffs;
    Eigen::VectorX<T> coeffs_p = tp * coeffs;

    for (T x = lb; x <= mid; x += 0.01) {
        T res = evaluate<T>(x, order, lb, ub, coeffs.data());
        T res_alt = evaluate<T>(x, order, lb, mid, coeffs_m.data());
        assert(std::fabs(res - res_alt) < 10 * std::numeric_limits<T>::epsilon());
    }

    for (T x = mid; x <= ub; x += 0.01) {
        T res = evaluate<T>(x, order, lb, ub, coeffs.data());
        T res_alt = evaluate<T>(x, order, mid, ub, coeffs_p.data());
        assert(std::fabs(res - res_alt) < 10 * std::numeric_limits<T>::epsilon());
    }
}

int main(int argc, char *argv[]) {

    for (int order = 8; order < 32; order += 2) {
        translation_test<float>(order);
        translation_test<double>(order);
        interp_test<float>(order);
        interp_test<double>(order);
        fit_test<float>(order);
        fit_test<double>(order);
    }

    return 0;
}
