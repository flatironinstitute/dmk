#ifndef PROLATE0_EVAL_HPP
#define PROLATE0_EVAL_HPP

#include <dmk/prolate0_fun.hpp>
#include <iostream>
#include <unordered_map>

namespace dmk {
/*
evaluate prolate0c at x, i.e., \psi_0^c(x)
*/
template <typename T>
T prolate0_eval(T c, T x) {
    static std::unordered_map<T, Prolate0Fun> prolate0_funcs_cache;
    if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
#pragma omp critical(PROLATE0_EVAL)
        if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
            std::cout << "Creating new Prolate0Fun for c = " << c << std::endl;
            prolate0_funcs_cache[c] = Prolate0Fun(c, 10000);
        }
    }
    return prolate0_funcs_cache[c].eval_val(x);
}

/*
evaluate prolate0c function integral of \int_0^r \psi_0^c(x) dx
*/
template <typename T>
T prolate0_int_eval(T c, T r) {
    static std::unordered_map<T, Prolate0Fun> prolate0_funcs_cache;
    if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
#pragma omp critical(PROLATE0_INT_EVAL)
        if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
            std::cout << "Creating new Prolate0Fun for c = " << c << std::endl;
            prolate0_funcs_cache[c] = Prolate0Fun(c, 10000);
        }
    }
    return prolate0_funcs_cache[c].int_eval(r);
}
} // namespace dmk

#endif
