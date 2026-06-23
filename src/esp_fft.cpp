#ifdef DMK_WITH_FINUFFT

#include <fftw3.h>
#include <complex>
#include <vector>

using CGrid = std::vector<std::complex<double>>;

void fftn_3d(const CGrid &in, CGrid &out, int n) {
    out.resize(in.size());
    fftw_plan plan = fftw_plan_dft_3d(
        n, n, n,
        reinterpret_cast<fftw_complex *>(const_cast<std::complex<double> *>(in.data())),
        reinterpret_cast<fftw_complex *>(out.data()),
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

void ifftn_3d(const CGrid &in, CGrid &out, int n) {
    out.resize(in.size());
    fftw_plan plan = fftw_plan_dft_3d(
        n, n, n,
        reinterpret_cast<fftw_complex *>(const_cast<std::complex<double> *>(in.data())),
        reinterpret_cast<fftw_complex *>(out.data()),
        FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    double norm = 1.0 / (static_cast<double>(n) * n * n);
    for (auto &v : out) v *= norm;
}

#endif // DMK_WITH_FINUFFT
