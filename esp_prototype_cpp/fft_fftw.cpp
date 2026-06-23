#include <fftw3.h>
#include <complex>
#include <vector>

using CGrid = std::vector<std::complex<double>>;

// Forward 3-D FFT (no normalisation, matches numpy.fft.fftn)
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

// Inverse 3-D FFT, normalised by 1/N^3 
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