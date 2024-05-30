#ifndef PLANEWAVE_HPP
#define PLANEWAVE_HPP

#include <complex>
#include <sctl.hpp>

namespace dmk {
template <int DIM, typename T>
void calc_planewave_translation_matrix(int nmax, T xmin, int npw, const sctl::Vector<T> &ts,
                                       sctl::Vector<std::complex<T>> &shift_vec);
}

#endif
