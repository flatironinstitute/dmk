#ifndef TENSORPROD_HPP
#define TENSORPROD_HPP

namespace dmk::tensorprod {
template <typename T>
void transform(int n_dim, int nin, int nout, int add_flag, const T *fin, const T *umat, T *fout);
}

#endif
