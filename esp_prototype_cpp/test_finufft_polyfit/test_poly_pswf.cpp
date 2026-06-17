#include <dmk/prolate0_fun.hpp>
#include <finufft_common/kernel.h>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // 1. initialize PSWF
    double c;
    dmk::prolc180(1e-6, c);
    dmk::Prolate0Fun pswf(c, 10000);
    double norm = pswf.eval_val(0.0);  // normalize so pswf(0) = 1

    // 2. fit polynomial to normalized pswf on [-1, 1]
    int nc = 24;
    auto pswf_lambda = [&](double x) {
        return pswf.eval_val(std::abs(x)) / norm;
    };
    auto coeffs = finufft::kernel::poly_fit<double>(pswf_lambda, nc);

    // 3. verify at test points
    std::cout << "x | pswf(x) | poly(x) | error\n";
    for (int i = 0; i <= 10; ++i) {
        double x = -1.0 + 2.0 * i / 10.0;

        // evaluate polynomial via Horner
        double poly_val = coeffs[0];
        for (int j = 1; j < nc; ++j)
            poly_val = poly_val * x + coeffs[j];

        double exact = pswf.eval_val(std::abs(x)) / norm;
        std::cout << x << " | " << exact << " | "
                  << poly_val << " | " << std::abs(exact - poly_val) << "\n";
    }

    return 0;
}