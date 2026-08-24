#pragma once

namespace dmk::cuda {

// Shared between the device cell-index kernels and the host code that keys the sort on them.
constexpr int kEspBins = 2; // sub-cell bins per axis (octants for DIM=3), matches esp.cpp's default
constexpr int kEspNbuckets = kEspBins * kEspBins * kEspBins;
constexpr int kMortonBits = 5; // 16/DIM for DIM=3, matches the CPU's sort_cell_morton
constexpr unsigned long long kMortonBuckets = 1ull << (3 * kMortonBits);

// One NVRTC module per stage is compiled from one source (selected by STAGE), so a single args
// struct covers every support kernel and most fields are unused in any given stage. Shared verbatim
// with the device source, so no includes and no std:: here. Complex is ComplexT<Real> on the device
// and cuComplex/cuDoubleComplex on the host -- the same two-Real aggregate either way.
template <typename Real, typename Complex>
struct EspSupportArgs {
    int n = 0;
    int nc = 0;
    int nf = 0;
    int ntot = 0;
    int out_dim = 0;

    Real L = Real{0};
    Real scale = Real{0};
    Real factor = Real{0};
    Real inv_ntot = Real{0};
    Real coeff_grad = Real{0};

    // Particle data
    const Real *pos_aos = nullptr;
    const Real *charges = nullptr;
    const int *orig = nullptr;
    const Real *qs_sorted = nullptr;
    const Real *pg_sorted = nullptr;

    int *cell_idx = nullptr;
    unsigned long long *cell_idx64 = nullptr;

    Real *xs = nullptr;
    Real *ys = nullptr;
    Real *zs = nullptr;
    Real *qs = nullptr;

    // Outputs, in original (un-permuted) particle order
    Real *pot = nullptr;
    Real *fx = nullptr;
    Real *fy = nullptr;
    Real *fz = nullptr;

    // Spectral grids
    const Real *scaling_coeffs = nullptr;
    Complex *grid = nullptr; // b_hat, or the grid normalized in place
    const Complex *pot_hat = nullptr;
    Complex *f_hat_x = nullptr;
    Complex *f_hat_y = nullptr;
    Complex *f_hat_z = nullptr;

    // Non-uniform point values
    const Complex *c = nullptr;
    const Complex *force_c = nullptr;
    Complex *c_out = nullptr;
    Real *out = nullptr;
};

} // namespace dmk::cuda
