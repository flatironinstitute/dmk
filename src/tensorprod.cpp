#include <Eigen/Core>

#include <dmk/gemm.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/types.hpp>
#include <sctl.hpp>

namespace dmk::tensorprod {

template <typename T>
void transform_1d(int add_flag, const ndview<const T, 1> &fin, const ndview<const T, 2> &umat,
                  const ndview<T, 1> &fout) {
    const int nin = fin.extent(0);
    const int nout = fout.extent(0);
    Eigen::Map<const Eigen::MatrixX<T>> u(umat.data_handle(), nout, nin);

    for (int j = 0; j < nout; ++j) {
        double res = add_flag ? fout(j) : 0.0;
        for (int i = 0; i < nin; ++i)
            res += u(j, i) * fin(j);

        fout(j) = res;
    }
}

template <typename T>
void transform_2d(int add_flag, const ndview<const T, 2> &fin_, const ndview<const T, 2> &umat,
                  const ndview<T, 2> &fout) {
    const int nin = fin_.extent(0);
    const int nout = fout.extent(0);
    Eigen::Map<const Eigen::MatrixX<T>> u1(umat.data_handle(), nout, nin);
    Eigen::Map<const Eigen::MatrixX<T>> u2(umat.data_handle() + nout * nin, nout, nin);
    Eigen::Map<const Eigen::MatrixX<T>> fin(fin_.data_handle(), nin, nin);
    Eigen::Map<Eigen::MatrixX<T>> res(fout.data_handle(), nout, nout);

    if (add_flag)
        res += u1 * (fin * u2.transpose());
    else
        res = u1 * (fin * u2.transpose());
}

template <typename T>
void transform_3d(int add_flag, const ndview<const T, 3> &fin, const ndview<const T, 2> &umat_,
                  const ndview<T, 3> &fout) {
    const int nin = fin.extent(0);
    const int nout = fout.extent(0);
    const int nin2 = nin * nin;
    const int noutnin = nout * nin;
    const int nout2 = nout * nout;

    sctl::Vector<T> ff_(nin * nin * nout);
    sctl::Vector<T> fft_(nin * nout * nin);
    sctl::Vector<T> ff2(nout * nout * nin);
    dmk::ndview<const T, 2> umat(umat_.data_handle(), nout * nin, 3);
    dmk::ndview<T, 3> ff(&ff_[0], nout, nout, nout);
    dmk::ndview<T, 3> fft(&fft_[0], nout, nout, nin);

    // transform in z
    dmk::gemm::gemm('n', 't', nin2, nout, nin, T{1.0}, fin.data_handle(), nin2, umat.data_handle() + 2 * nout * nin,
                    nout, T{0.0}, ff.data_handle(), nin2);

    for (int k = 0; k < nin; ++k)
        for (int j = 0; j < nout; ++j)
            for (int i = 0; i < nin; ++i)
                fft(i, j, k) = ff(k, i, j);

    // transform in y
    dmk::gemm::gemm('n', 'n', nout, noutnin, nin, T{1.0}, umat.data_handle() + nout * nin, nout, fft.data_handle(), nin,
                    T{0.0}, &ff2[0], nout);

    // transform in x
    dmk::gemm::gemm('n', 't', nout, nout2, nin, T{1.0}, umat.data_handle(), nout, &ff2[0], nout2, T(add_flag),
                    fout.data_handle(), nout);
}

template <typename T, int DIM>
void transform(int nvec, int add_flag, const ndview<const T, DIM + 1> &fin, const ndview<const T, 2> &umat,
               const ndview<T, DIM + 1> &fout) {
    const int nin = fin.extent(0);
    const int nout = fout.extent(0);
    if (DIM == 1) {
        const int block_in = nin, block_out = nout;
        for (int i = 0; i < nvec; ++i) {
            dmk::ndview<const T, 1> fin_view(fin.data_handle() + i * block_in, block_in);
            dmk::ndview<T, 1> fout_view(fout.data_handle() + i * block_out, block_out);
            transform_1d(add_flag, fin_view, umat, fout_view);
        }
    }
    if (DIM == 2) {
        const int block_in = nin * nin, block_out = nout * nout;
        for (int i = 0; i < nvec; ++i) {
            dmk::ndview<const T, 2> fin_view(fin.data_handle() + i * block_in, nin, nin);
            dmk::ndview<T, 2> fout_view(fout.data_handle() + i * block_out, nout, nout);
            transform_2d(add_flag, fin_view, umat, fout_view);
        }
        return;
    }
    if (DIM == 3) {
        const int block_in = nin * nin * nin, block_out = nout * nout * nout;
        for (int i = 0; i < nvec; ++i) {
            dmk::ndview<const T, 3> fin_view(fin.data_handle() + i * block_in, nin, nin, nin);
            dmk::ndview<T, 3> fout_view(fout.data_handle() + i * block_out, nout, nout, nout);

            transform_3d(add_flag, fin_view, umat, fout_view);
        }
        return;
    }
}

template void transform<float, 1>(int nvec, int add_flag, const dmk::ndview<const float, 2> &fin,
                                  const dmk::ndview<const float, 2> &umat, const ndview<float, 2> &fout);
template void transform<float, 2>(int nvec, int add_flag, const dmk::ndview<const float, 3> &fin,
                                  const dmk::ndview<const float, 2> &umat, const ndview<float, 3> &fout);
template void transform<float, 3>(int nvec, int add_flag, const dmk::ndview<const float, 4> &fin,
                                  const dmk::ndview<const float, 2> &umat, const ndview<float, 4> &fout);
template void transform<double, 1>(int nvec, int add_flag, const dmk::ndview<const double, 2> &fin,
                                   const dmk::ndview<const double, 2> &umat, const ndview<double, 2> &fout);
template void transform<double, 2>(int nvec, int add_flag, const dmk::ndview<const double, 3> &fin,
                                   const dmk::ndview<const double, 2> &umat, const ndview<double, 3> &fout);
template void transform<double, 3>(int nvec, int add_flag, const dmk::ndview<const double, 4> &fin,
                                   const dmk::ndview<const double, 2> &umat, const ndview<double, 4> &fout);
} // namespace dmk::tensorprod
