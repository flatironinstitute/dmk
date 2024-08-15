#include <Eigen/Core>

#include <dmk/gemm.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/types.hpp>
#include <sctl.hpp>
#include <type_traits>

namespace dmk::tensorprod {

template <typename T>
void transform_1d(int nin, int nout, int add_flag, const T *fin, const T *umat, T *fout) {
    Eigen::Map<const Eigen::MatrixX<T>> u(umat, nout, nin);

    for (int j = 0; j < nout; ++j) {
        double res = add_flag ? fout[j] : 0.0;
        for (int i = 0; i < nin; ++i)
            res += u(j, i) * fin[j];

        fout[j] = res;
    }
}

template <typename T>
void transform_2d(int nin, int nout, int add_flag, const T *fin_, const T *umat, T *fout) {
    Eigen::Map<const Eigen::MatrixX<T>> u1(umat, nout, nin);
    Eigen::Map<const Eigen::MatrixX<T>> u2(umat + nout * nin, nout, nin);
    Eigen::Map<const Eigen::MatrixX<T>> fin(fin_, nin, nin);

    Eigen::Map<Eigen::MatrixX<T>> res(fout, nout, nout);
    if (add_flag)
        res += u1 * (fin * u2.transpose());
    else
        res = u1 * (fin * u2.transpose());
}

template <typename T>
void transform_3d(int nin, int nout, int add_flag, const T *fin, const T *umat_, T *fout) {
    sctl::Vector<T> ff_(nin * nin * nout);
    sctl::Vector<T> fft_(nin * nout * nin);
    sctl::Vector<T> ff2(nout * nout * nin);
    dmk::ndview<const T, 2> umat(umat_, nout * nin, 3);
    dmk::ndview<T, 3> ff(&ff_[0], nout, nout, nout);
    dmk::ndview<T, 3> fft(&fft_[0], nout, nout, nin);

    const int nin2 = nin * nin;
    const int noutnin = nout * nin;
    const int nout2 = nout * nout;

    // transform in z
    dmk::gemm::gemm('n', 't', nin2, nout, nin, T{1.0}, fin, nin2, &umat(0, 2), nout, T{0.0}, &ff(0, 0, 0), nin2);

    for (int k = 0; k < nin; ++k)
        for (int j = 0; j < nout; ++j)
            for (int i = 0; i < nin; ++i)
                fft(i, j, k) = ff(k, i, j);

    // transform in y
    dmk::gemm::gemm('n', 'n', nout, noutnin, nin, T{1.0}, &umat(0, 1), nout, &fft(0, 0, 0), nin, T{0.0}, &ff2[0], nout);

    // transform in x
    dmk::gemm::gemm('n', 't', nout, nout2, nin, T{1.0}, &umat(0, 0), nout, &ff2[0], nout2, T(add_flag), fout, nout);
}

template <typename T>
void transform(int n_dim, int nvec, int nin, int nout, int add_flag, const T *fin, const T *umat, T *fout) {
    if (n_dim == 1) {
        const int block_in = nin, block_out = nout;
        for (int i = 0; i < nvec; ++i)
            transform_1d(nin, nout, add_flag, fin + i * block_in, umat, fout + i * block_out);
        return;
    }
    if (n_dim == 2) {
        const int block_in = nin * nin, block_out = nout * nout;
        for (int i = 0; i < nvec; ++i)
            transform_2d(nin, nout, add_flag, fin + i * block_in, umat, fout + i * block_out);
        return;
    }
    if (n_dim == 3) {
        const int block_in = nin * nin * nin, block_out = nout * nout * nout;
        for (int i = 0; i < nvec; ++i)
            transform_3d(nin, nout, add_flag, fin + i * block_in, umat, fout + i * block_out);
        return;
    }
}

template void transform(int n_dim, int nvec, int nin, int nout, int add_flag, const float *fin, const float *umat,
                        float *fout);
template void transform(int n_dim, int nvec, int nin, int nout, int add_flag, const double *fin, const double *umat,
                        double *fout);
} // namespace dmk::tensorprod
