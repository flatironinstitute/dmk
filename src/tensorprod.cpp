#include <Eigen/Core>

#include <dmk/gemm.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/types.hpp>
#include <dmk/util.hpp>
#include <sctl.hpp>

namespace dmk::tensorprod {

template <typename T>
void transform_1d(int add_flag, const ndview<T, 1> &fin, const ndview<T, 2> &umat, ndview<T, 1> &fout) {
    const int nin = fin.extent(0);
    const int nout = fout.extent(0);
    Eigen::Map<const Eigen::MatrixX<T>> u(umat.data(), nout, nin);

    for (int j = 0; j < nout; ++j) {
        double res = add_flag ? fout(j) : 0.0;
        for (int i = 0; i < nin; ++i)
            res += u(j, i) * fin(j);

        fout(j) = res;
    }
}

template <typename T>
void transform_2d(int add_flag, const ndview<T, 2> &fin_, const ndview<T, 2> &umat, ndview<T, 2> &fout) {
    const int nin = fin_.extent(0);
    const int nout = fout.extent(0);
    Eigen::Map<const Eigen::MatrixX<T>> u1(umat.data(), nout, nin);
    Eigen::Map<const Eigen::MatrixX<T>> u2(umat.data() + nout * nin, nout, nin);
    Eigen::Map<const Eigen::MatrixX<T>> fin(fin_.data(), nin, nin);
    Eigen::Map<Eigen::MatrixX<T>> res(fout.data(), nout, nout);

    if (add_flag)
        res += u1 * (fin * u2.transpose());
    else
        res = u1 * (fin * u2.transpose());
}

template <typename T>
void transform_3d(int add_flag, const ndview<T, 3> &fin, const ndview<T, 2> &umat_, ndview<T, 3> &fout,
                  sctl::Vector<T> &workspace) {
    const int nin = fin.extent(0);
    const int nout = fout.extent(0);
    const int nin2 = nin * nin;
    const int noutnin = nout * nin;
    const int nout2 = nout * nout;

    dmk::ndview<const T, 2> umat({nout * nin, 3}, umat_.data());
    workspace.ReInit(2 * nin * nin * nout + nout * nout * nin);
    dmk::ndview<T, 3> ff({nin, nin, nout}, &workspace[0]);
    dmk::ndview<T, 3> fft({nout, nout, nin}, ff.data() + ff.size());
    dmk::ndview<T, 3> ff2({nout, nout, nin}, fft.data() + fft.size());

    // transform in z
    dmk::gemm::gemm('n', 't', nin2, nout, nin, T{1.0}, fin.data(), nin2, umat.data() + 2 * nout * nin, nout, T{0.0},
                    ff.data(), nin2);

    for (int k = 0; k < nin; ++k)
        for (int j = 0; j < nout; ++j)
            for (int i = 0; i < nin; ++i)
                fft(i, j, k) = ff(k, i, j);

    // transform in y
    dmk::gemm::gemm('n', 'n', nout, noutnin, nin, T{1.0}, umat.data() + nout * nin, nout, fft.data(), nin, T{0.0},
                    ff2.data(), nout);

    // transform in x
    dmk::gemm::gemm('n', 't', nout, nout2, nin, T{1.0}, umat.data(), nout, ff2.data(), nout2, T(add_flag), fout.data(),
                    nout);

    auto flops_per_mm = [](int m, int n, int k) { return 2 * m * n * k; };
    const unsigned long flops =
        flops_per_mm(nin2, nout, nin) + flops_per_mm(nout, noutnin, nin) + flops_per_mm(nout, nout2, nin);
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, flops);
}

template <typename T, int DIM>
void transform(int nvec, int add_flag, const ndview<T, DIM + 1> &fin, const ndview<T, 2> &umat, ndview<T, DIM + 1> fout,
               sctl::Vector<T> &workspace) {
    // dmk::util::PAPICounter papi_counter;
    const int nin = fin.extent(0);
    const int nout = fout.extent(0);
    if (DIM == 1) {
        const int block_in = nin, block_out = nout;
        for (int i = 0; i < nvec; ++i) {
            const dmk::ndview<T, 1> fin_view({block_in}, const_cast<T *>(fin.data()) + i * block_in);
            dmk::ndview<T, 1> fout_view({block_out}, fout.data() + i * block_out);
            transform_1d(add_flag, fin_view, umat, fout_view);
        }
    }
    if (DIM == 2) {
        const int block_in = nin * nin, block_out = nout * nout;
        for (int i = 0; i < nvec; ++i) {
            const dmk::ndview<T, 2> fin_view({nin, nin}, const_cast<T *>(fin.data()) + i * block_in);
            dmk::ndview<T, 2> fout_view({nout, nout}, fout.data() + i * block_out);
            transform_2d(add_flag, fin_view, umat, fout_view);
        }
        return;
    }
    if (DIM == 3) {
        const int block_in = nin * nin * nin, block_out = nout * nout * nout;
        for (int i = 0; i < nvec; ++i) {
            const dmk::ndview<T, 3> fin_view({nin, nin, nin}, const_cast<T *>(fin.data()) + i * block_in);
            dmk::ndview<T, 3> fout_view({nout, nout, nout}, fout.data() + i * block_out);

            transform_3d(add_flag, fin_view, umat, fout_view, workspace);
        }
        return;
    }
}

template void transform<float, 1>(int nvec, int add_flag, const dmk::ndview<float, 2> &fin,
                                  const dmk::ndview<float, 2> &umat, ndview<float, 2> fout,
                                  sctl::Vector<float> &workspace);
template void transform<float, 2>(int nvec, int add_flag, const dmk::ndview<float, 3> &fin,
                                  const dmk::ndview<float, 2> &umat, ndview<float, 3> fout,
                                  sctl::Vector<float> &workspace);
template void transform<float, 3>(int nvec, int add_flag, const dmk::ndview<float, 4> &fin,
                                  const dmk::ndview<float, 2> &umat, ndview<float, 4> fout,
                                  sctl::Vector<float> &workspace);
template void transform<double, 1>(int nvec, int add_flag, const dmk::ndview<double, 2> &fin,
                                   const dmk::ndview<double, 2> &umat, ndview<double, 2> fout,
                                   sctl::Vector<double> &workspace);
template void transform<double, 2>(int nvec, int add_flag, const dmk::ndview<double, 3> &fin,
                                   const dmk::ndview<double, 2> &umat, ndview<double, 3> fout,
                                   sctl::Vector<double> &workspace);
template void transform<double, 3>(int nvec, int add_flag, const dmk::ndview<double, 4> &fin,
                                   const dmk::ndview<double, 2> &umat, ndview<double, 4> fout,
                                   sctl::Vector<double> &workspace);
} // namespace dmk::tensorprod
