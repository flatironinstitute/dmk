#include <Eigen/Core>

#include <dmk/tensorprod.hpp>
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
void transform_3d(int nin, int nout, int add_flag, const T *fin, const T *umat, T *fout) {
    T alpha{1.0};
    T beta{0.0};
    T ff[nout * nout * nout];
    T ff2[nin * nout * nin];
    T fft[nout * nout * nin];

    const char n = 'n', t = 't';
    const int nin2 = nin * nin;
    const int noutnin = nout * nin;
    const int nout2 = nout * nout;
    if constexpr (std::is_same_v<float, T>)
        sgemm_(&n, &t, &nin2, &nout, &nin, &alpha, fin, &nin2, &umat[2 * nout * nin], &nout, &beta, ff, &nin2);
    else
        dgemm_(&n, &t, &nin2, &nout, &nin, &alpha, fin, &nin2, &umat[2 * nout * nin], &nout, &beta, ff, &nin2);

    // oof
    for (int j1 = 0; j1 < nin; ++j1)
        for (int k3 = 0; k3 < nout; ++k3)
            for (int j2 = 0; j2 < nin; ++j2)
                fft[j2 + k3 * nin + j1 * nout * nin] = ff[j1 + j2 * nin + k3 * nout * nin];

    // transform in y
    if constexpr (std::is_same_v<float, T>)
        sgemm_(&n, &n, &nout, &noutnin, &nin, &alpha, &umat[1 * nout * nin], &nout, fft, &nin, &beta, ff2, &nout);
    else
        dgemm_(&n, &n, &nout, &noutnin, &nin, &alpha, &umat[1 * nout * nin], &nout, fft, &nin, &beta, ff2, &nout);

    // transform in x
    beta = add_flag ? T{1.0} : 0.0;
    if constexpr (std::is_same_v<float, T>)
        sgemm_(&n, &t, &nout, &nout2, &nin, &alpha, umat, &nout, ff2, &nout2, &beta, fout, &nout);
    else
        dgemm_(&n, &t, &nout, &nout2, &nin, &alpha, umat, &nout, ff2, &nout2, &beta, fout, &nout);
}

template <typename T>
void transform(int n_dim, int nin, int nout, int add_flag, const T *fin, const T *umat, T *fout) {
    if (n_dim == 1)
        return transform_1d(nin, nout, add_flag, fin, umat, fout);
    if (n_dim == 2)
        return transform_2d(nin, nout, add_flag, fin, umat, fout);
    if (n_dim == 3)
        return transform_3d(nin, nout, add_flag, fin, umat, fout);
}

template void transform(int n_dim, int nin, int nout, int add_flag, const float *fin, const float *umat, float *fout);
template void transform(int n_dim, int nin, int nout, int add_flag, const double *fin, const double *umat,
                        double *fout);
} // namespace dmk::tensorprod
