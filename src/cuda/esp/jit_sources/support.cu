// ESP GPU support kernels: cell-list construction and the long-range spectral steps. The launcher
// prepends a prelude defining Real, the DMK_ESP_SUPPORT_KERNEL_NAME symbol and the STAGE /
// BLOCK_SIZE constants; one module per stage is compiled from this one source.

#include <dmk/cuda/esp_support_kernelargs.hpp>

// NVRTC has no <type_traits>, hence the explicit specialization instead of std::conditional_t.
struct DmkFloat2 { float x, y; };
struct DmkDouble2 { double x, y; };
template <class T> struct ComplexFor;
template <> struct ComplexFor<float> { using type = DmkFloat2; };
template <> struct ComplexFor<double> { using type = DmkDouble2; };
template <class T> using ComplexT = typename ComplexFor<T>::type;

using Complex = ComplexT<Real>;
using EspSupportArgs = dmk::cuda::EspSupportArgs<Real, Complex>;

using dmk::cuda::kEspBins;
using dmk::cuda::kEspNbuckets;
using dmk::cuda::kMortonBits;
using dmk::cuda::kMortonBuckets;

// Flat (cell, spatial-bin) composite key per particle. Mirrors the CPU's particle_cell /
// cell_linear_index plus sort_cell_bins, so sorting by this key clusters particles spatially within
// each cell, not merely by cell.


__device__ void cell_index_kernel(const EspSupportArgs &args)
{
    const Real *d_pos_aos = args.pos_aos;
    const int n = args.n;
    const Real L = args.L;
    const int nc = args.nc;
    int *d_cell_idx = args.cell_idx;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real cell_size = L / Real(nc);

    // Wrapped cell coordinate for axis value x; writes that axis's sub-cell bin into bin_out.
    auto cell_coord_and_bin = [&](Real x, int &bin_out) {
        const Real u = (x + L / Real(2)) / cell_size; // continuous cell coordinate
        int c = static_cast<int>(floor(u));
        Real frac = u - floor(u);
        int b = static_cast<int>(frac * Real(kEspBins));
        b = (b < 0) ? 0 : (b >= kEspBins ? kEspBins - 1 : b);
        bin_out = b;
        c = (c >= nc) ? c - nc : c;
        c = (c < 0)   ? c + nc : c;
        return c;
    };

    int bx, by, bz;
    const int cx = cell_coord_and_bin(d_pos_aos[3 * i + 0], bx);
    const int cy = cell_coord_and_bin(d_pos_aos[3 * i + 1], by);
    const int cz = cell_coord_and_bin(d_pos_aos[3 * i + 2], bz);

    // bin_lin < nbuckets always, so the composite key never crosses a cell boundary. One global
    // sort replaces the CPU's many small per-cell sorts, which do not map onto GPU parallelism.
    const int cell_lin = (cx * nc + cy) * nc + cz;
    const int bin_lin  = (bz * kEspBins + by) * kEspBins + bx; // matches sort_cell_bins' key = key*bins + bidx[d], d=DIM-1..0
    d_cell_idx[i] = cell_lin * kEspNbuckets + bin_lin;
}

// (cell, Morton-code) composite key. Sorting by it yields the same within-cell ordering as the
// CPU's per-cell Morton sorts.
__device__ __forceinline__ unsigned long long part1by2_64(unsigned long long x) {
    x &= 0x1fffffull;
    x = (x | x << 32) & 0x1f00000000ffffull;
    x = (x | x << 16) & 0x1f0000ff0000ffull;
    x = (x | x << 8)  & 0x100f00f00f00f00full;
    x = (x | x << 4)  & 0x10c30c30c30c30c3ull;
    x = (x | x << 2)  & 0x1249249249249249ull;
    return x;
}
__device__ __forceinline__ unsigned long long morton3(unsigned long long cx, unsigned long long cy, unsigned long long cz) {
    return (part1by2_64(cx) << 2) | (part1by2_64(cy) << 1) | part1by2_64(cz);
}



__device__ void cell_index_kernel_morton(const EspSupportArgs &args)
{
    const Real *d_pos_aos = args.pos_aos;
    const int n = args.n;
    const Real L = args.L;
    const int nc = args.nc;
    unsigned long long *d_cell_idx = args.cell_idx64;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real cell_size = L / Real(nc);

    // As cell_index_kernel, but returning the continuous within-cell fraction: Morton needs
    // kMortonBits of resolution, not one bin.
    auto cell_coord_and_frac = [&](Real x, Real &frac_out) {
        const Real u = (x + L / Real(2)) / cell_size;
        int c = static_cast<int>(floor(u));
        frac_out = u - floor(u);
        c = (c >= nc) ? c - nc : c;
        c = (c < 0)   ? c + nc : c;
        return c;
    };

    Real fx, fy, fz;
    const int cx = cell_coord_and_frac(d_pos_aos[3 * i + 0], fx);
    const int cy = cell_coord_and_frac(d_pos_aos[3 * i + 1], fy);
    const int cz = cell_coord_and_frac(d_pos_aos[3 * i + 2], fz);

    constexpr int kMax = (1 << kMortonBits) - 1;
    auto quantize = [&](Real frac) {
        int q = static_cast<int>(frac * Real(1 << kMortonBits));
        return (unsigned long long)((q < 0) ? 0 : (q > kMax ? kMax : q));
    };
    const unsigned long long code = morton3(quantize(fx), quantize(fy), quantize(fz));

    const int cell_lin = (cx * nc + cy) * nc + cz;
    d_cell_idx[i] = (unsigned long long)cell_lin * kMortonBuckets + code;
}

// Apply the cell-sort permutation to positions/charges.
__device__ void gather_sorted_kernel(const EspSupportArgs &args)
{
    const Real *d_pos_aos = args.pos_aos;
    const Real *d_charges = args.charges;
    const int *d_orig = args.orig;
    const int n = args.n;
    Real *d_xs = args.xs;
    Real *d_ys = args.ys;
    Real *d_zs = args.zs;
    Real *d_qs = args.qs;

    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n) return;
    const int orig = d_orig[slot];
    d_xs[slot] = d_pos_aos[3 * orig + 0];
    d_ys[slot] = d_pos_aos[3 * orig + 1];
    d_zs[slot] = d_pos_aos[3 * orig + 2];
    d_qs[slot] = d_charges[orig];
}

__device__ void scatter_kernel(const EspSupportArgs &args)
{
    const int n = args.n;
    const int out_dim = args.out_dim;
    const Real *pg_sorted = args.pg_sorted;
    const int *d_orig = args.orig;
    const Real *d_qs_sorted = args.qs_sorted;
    Real *d_pot = args.pot;
    Real *d_fx = args.fx;
    Real *d_fy = args.fy;
    Real *d_fz = args.fz;

    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n) return;
    const int o = d_orig[a];
    d_pot[o] += pg_sorted[out_dim * a + 0];
    if (out_dim > 1) {
        const Real q = d_qs_sorted[a];
        d_fx[o] += -q * pg_sorted[out_dim * a + 1];
        d_fy[o] += -q * pg_sorted[out_dim * a + 2];
        d_fz[o] += -q * pg_sorted[out_dim * a + 3];
    }
}

// In place: b_hat[i] *= scaling_coeffs[i], producing pot_hat.
__device__ void scaling_kernel(const EspSupportArgs &args)
{
    const int ntot = args.ntot;
    const Real *scaling_coeffs = args.scaling_coeffs;
    Complex *b_hat = args.grid;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntot) return;
    Real s = scaling_coeffs[i];
    b_hat[i] = {b_hat[i].x * s, b_hat[i].y * s}; //.x = real part; .y = imaginary part
}

// cuFFT's inverse transform is unnormalized, so scale by 1/ntot.
__device__ void normalize_kernel(const EspSupportArgs &args)
{
    const int ntot = args.ntot;
    const Real inv_ntot = args.inv_ntot;
    Complex *data = args.grid;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntot) return;
    data[i] = {data[i].x * inv_ntot, data[i].y * inv_ntot};
}

// extract_real_kernel — write the real part of each NU complex value to d_out.

__device__ void extract_real_kernel(const EspSupportArgs &args)
{
    const int n = args.n;
    const Complex *d_c = args.c;
    Real *d_out = args.out;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_out[i] += Real(d_c[i].x);
}

// grad(u)_hat_k = i*k*u_hat_k, so each force spectrum is pot_hat * i*k_component*coeff_grad. k_idx
// is recomputed per thread rather than buffered. Note the axis swap: f_hat_x uses k_idx[iz], matching
// long_range()'s force block on the CPU.
__device__ __forceinline__ int grad_kidx(int i, int nf) { return (i <= nf / 2) ? i : i - nf; }

__device__ void grad_scaling_kernel(const EspSupportArgs &args)
{
    const int nf = args.nf;
    const Real coeff_grad = args.coeff_grad;
    const Complex *pot_hat = args.pot_hat;
    Complex *f_hat_x = args.f_hat_x;
    Complex *f_hat_y = args.f_hat_y;
    Complex *f_hat_z = args.f_hat_z;

    const long long ntot = (long long)nf * nf * nf;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntot) return;

    const int iz = int(i % nf);
    const int iy = int((i / nf) % nf);
    const int ix = int(i / ((long long)nf * nf));

    const Complex s = pot_hat[i];
    auto mul_ik = [=](int k) {
        const Real factor = coeff_grad * Real(k);
        // s * (i * factor) = (-s.y*factor, s.x*factor)
        return Complex{-s.y * factor, s.x * factor};
    };
    f_hat_x[i] = mul_ik(grad_kidx(iz, nf));
    f_hat_y[i] = mul_ik(grad_kidx(iy, nf));
    f_hat_z[i] = mul_ik(grad_kidx(ix, nf));
}

// force_out[j] += -charge[j]*real(force_c[j]). Charges come from d_c, already packed as {charge, 0}
// for spreading.
__device__ void accumulate_force_kernel(const EspSupportArgs &args)
{
    const int n = args.n;
    const Complex *d_c = args.c;
    const Complex *d_force_c = args.force_c;
    Real *d_force_out = args.out;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_force_out[i] += Real(-d_c[i].x * d_force_c[i].x);
}

__device__ void self_interaction_kernel(const EspSupportArgs &args)
{
    const int n = args.n;
    const Real factor = args.factor;
    const Real *d_charges = args.charges;
    Real *d_pot = args.pot;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_pot[i] -= d_charges[i] * factor;
}

// AoS positions/charges -> scaled [-pi,pi) SoA coords + packed complex charges, on device.
__device__ void scale_pack_kernel(const EspSupportArgs &args)
{
    const Real *d_pos_aos = args.pos_aos;
    const Real *d_charges = args.charges;
    const int n = args.n;
    const Real scale = args.scale;
    Real *d_x = args.xs;
    Real *d_y = args.ys;
    Real *d_z = args.zs;
    Complex *d_c = args.c_out;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_x[i] = d_pos_aos[3 * i + 0] * scale;
    d_y[i] = d_pos_aos[3 * i + 1] * scale;
    d_z[i] = d_pos_aos[3 * i + 2] * scale;
    d_c[i] = {d_charges[i], Real(0)};
}
// KERNEL_START

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE) DMK_ESP_SUPPORT_KERNEL_NAME(EspSupportArgs args) {
    if constexpr (STAGE == 0)
        cell_index_kernel(args);
    else if constexpr (STAGE == 1)
        cell_index_kernel_morton(args);
    else if constexpr (STAGE == 2)
        gather_sorted_kernel(args);
    else if constexpr (STAGE == 3)
        scatter_kernel(args);
    else if constexpr (STAGE == 4)
        scaling_kernel(args);
    else if constexpr (STAGE == 5)
        normalize_kernel(args);
    else if constexpr (STAGE == 6)
        extract_real_kernel(args);
    else if constexpr (STAGE == 7)
        grad_scaling_kernel(args);
    else if constexpr (STAGE == 8)
        accumulate_force_kernel(args);
    else if constexpr (STAGE == 9)
        self_interaction_kernel(args);
    else
        scale_pack_kernel(args);
}
