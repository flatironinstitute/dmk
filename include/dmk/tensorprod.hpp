#ifndef TENSORPROD_HPP
#define TENSORPROD_HPP

#include <dmk/types.hpp>
#include <sctl.hpp>

namespace dmk::tensorprod {
template <typename T, int DIM>
void transform(int nvec, int add_flag, const ndview<T, DIM + 1> &fin, const ndview<T, 2> &umat, ndview<T, DIM + 1> fout,
               sctl::Vector<T> &workspace);
}

#endif
