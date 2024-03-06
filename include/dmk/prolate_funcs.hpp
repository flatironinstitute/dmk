#ifndef PROLATE_FUNCS_HPP
#define PROLATE_FUNCS_HPP

#include "fortran.h"

#include <stdexcept>
#include <tuple>
#include <vector>

namespace dmk {
struct ProlateFuncs {
    inline ProlateFuncs(double beta_, int lenw_) : beta(beta_), lenw(lenw_) {
        int ier;
        workarray.resize(lenw);
        prol0ini_(&ier, &beta, workarray.data(), &rlam20, &rkhi, &lenw, &keep, &ltot);
        if (ier)
            throw std::runtime_error("Unable to init ProlateFuncs");
    }

    inline std::pair<double, double> eval_val_derivative(double x) const {
        // wrapper for prol0eva routine - evaluates the function \psi^c_0 and its
        // derivative at the user-specified point x \in R^1.
        double psi0, derpsi0;
        prol0eva_(&x, workarray.data(), &psi0, &derpsi0);
        return std::make_pair(psi0, derpsi0);
    }

    inline double eval_val(double x) const {
        auto [val, dum] = eval_val_derivative(x);
        return val;
    }

    inline double eval_derivative(double x) const {
        auto [dum, der] = eval_val_derivative(x);
        return der;
    }

    inline std::array<double, 4> intvals(double beta) const {
        std::array<double, 4> c;
        prolate_intvals_(&beta, workarray.data(), &c[0], &c[1], &c[2], &c[3]);
        return c;
    }

    double beta;
    int lenw, keep, ltot;
    std::vector<double> workarray;
    double rlam20, rkhi;
};
} // namespace dmk
#endif
