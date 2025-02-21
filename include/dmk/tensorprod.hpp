#ifndef TENSORPROD_HPP
#define TENSORPROD_HPP

#include <dmk/types.hpp>

namespace dmk::tensorprod {
template <typename T, int DIM>
void transform(int nvec, int add_flag, const ndview<const T, DIM + 1> &fin, const ndview<const T, 2> &umat,
               const ndview<T, DIM + 1> &fout);
}

#endif
