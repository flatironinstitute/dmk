#ifndef DMK_POLYFIT_HPP
#define DMK_POLYFIT_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <polyfit/fast_eval.hpp>

namespace dmk {

template <class Vec>
auto to_vector(const auto &arr) {
    Vec vec(arr.size());
    std::copy_n(arr.data(), arr.size(), &vec[0]);
    return vec;
}

// Fit f on [a, b] to ~10^-digits absolute error, returning monomial coefficients
// in Horner order (ascending: coeffs[0] is the constant term, suitable for
// dmk::horner). Returns empty if no fit with fewer than 32 coefficients meets
// the tolerance.
template <class Real, class Func>
std::vector<Real> make_polyfit_abs_error(int digits, Func &&f, Real a, Real b) {
    const Real tol = std::pow(10.0, -digits);

    for (int n_coeffs = 3; n_coeffs < 32; ++n_coeffs) {
        try {
            auto prolate_int_fun = poly_eval::make_func_eval(f, n_coeffs, a, b);

            bool passed = true;
            for (double x = a; x <= b; x += 0.01 * (b - a)) {
                const Real fit = prolate_int_fun(x);
                const Real act = f(x);
                const Real abs_err = std::abs(act - fit);

                if (abs_err > tol) {
                    passed = false;
                    break;
                }
            }
            if (passed) {
                auto coeffs = to_vector<std::vector<Real>>(prolate_int_fun.coeffs());
                std::reverse(coeffs.begin(), coeffs.end());
                return coeffs;
            }
        } catch (std::exception &e) {
            std::cout << "Failed to fit with n_coeffs = " << n_coeffs << "\n";
            std::cout << e.what() << std::endl;
        }
    }
    return {};
}

} // namespace dmk

#endif
