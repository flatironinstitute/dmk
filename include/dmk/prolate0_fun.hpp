#ifndef PROLATE0_FUN_HPP
#define PROLATE0_FUN_HPP

#include <array>
#include <stdexcept>
#include <vector>

#include <dmk/prolate.hpp>

namespace dmk {
struct Prolate0Fun {
    Prolate0Fun() = default;

    inline Prolate0Fun(double c_, int lenw_) : c(c_), lenw(lenw_) {
        int ier;
        workarray.resize(lenw);
        prol0ini(ier, c, workarray.data(), rlam20, rkhi, lenw, keep, ltot);
        if (ier)
            throw std::runtime_error("Unable to init Prolate0Fun");
    }

    // evaluate prolate0 function val and derivative
    inline std::pair<double, double> eval_val_derivative(double x) const {
        double psi0, derpsi0;
        prol0eva(x, workarray.data(), psi0, derpsi0);
        return std::make_pair(psi0, derpsi0);
    }

    // evaluate prolate0 function value
    inline double eval_val(double x) const {
        auto [val, dum] = eval_val_derivative(x);
        return val;
    }

    // evaluate prolate0 function derivative
    inline double eval_derivative(double x) const {
        auto [dum, der] = eval_val_derivative(x);
        return der;
    }

    // int_0^r prolate0(x) dx
    inline double int_eval(double r) const {
        double val;
        prol0int0r(workarray.data(), r, val);
        return val;
    }

    inline std::array<double, 4> intvals(double beta) {
        std::array<double, 4> c;
        prolate_intvals(beta, workarray.data(), c[0], c[1], c[2], c[3]);
        return c;
    }

    double c;
    int lenw, keep, ltot;
    std::vector<double> workarray;
    double rlam20, rkhi;
};

} // namespace dmk
#endif
