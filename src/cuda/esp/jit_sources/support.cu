// ESP GPU support kernels: cell-list construction and the long-range spectral steps. The launcher
// prepends Real, DMK_ESP_SUPPORT_KERNEL_NAME and the STAGE / BLOCK_SIZE / PROJECTOR constants.

#include <dmk/cuda/esp_support_kernelargs.hpp>

// NVRTC has no <type_traits>.
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

// Flat (cell, spatial-bin) composite key: sorting by it clusters particles within each cell.


__device__ void cell_index_kernel(const EspSupportArgs &args)
{
    const Real *d_pos_aos = args.pos_aos;
    const int n = args.n;
    const int nc = args.nc;
    int *d_cell_idx = args.cell_idx;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real cell_size = Real(1) / Real(nc);

    // Wrapped cell coordinate for axis value x; writes that axis's sub-cell bin into bin_out.
    auto cell_coord_and_bin = [&](Real x, int &bin_out) {
        const Real u = x / cell_size; // continuous cell coordinate
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

    // bin_lin < nbuckets always, so the composite key never crosses a cell boundary.
    const int cell_lin = (cx * nc + cy) * nc + cz;
    const int bin_lin  = (bz * kEspBins + by) * kEspBins + bx; // matches sort_cell_bins' key = key*bins + bidx[d], d=DIM-1..0
    d_cell_idx[i] = cell_lin * kEspNbuckets + bin_lin;
}

// (cell, Morton-code) composite key.
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
    const int nc = args.nc;
    unsigned long long *d_cell_idx = args.cell_idx64;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real cell_size = Real(1) / Real(nc);

    // Morton needs kMortonBits of resolution, not one bin.
    auto cell_coord_and_frac = [&](Real x, Real &frac_out) {
        const Real u = x / cell_size;
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
    // AoS [charge | normal] in, charge_dim SoA planes of stride n out.
    for (int c = 0; c < args.charge_dim; ++c)
        d_qs[c * n + slot] = d_charges[args.charge_dim * orig + c];
}

__device__ void scatter_kernel(const EspSupportArgs &args)
{
    const int n = args.n;
    const int out_dim = args.out_dim;
    const Real *pg_sorted = args.pg_sorted;
    const int *d_orig = args.orig;
    Real *out[4] = {args.pot, args.gx, args.gy, args.gz};

    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n) return;
    const int o = d_orig[a];

    for (int k = 0; k < out_dim; ++k)
        out[k][o] += pg_sorted[out_dim * a + k];
}

__device__ __forceinline__ Complex cadd(Complex a, Complex b) { return {a.x + b.x, a.y + b.y}; }
__device__ __forceinline__ Complex csub(Complex a, Complex b) { return {a.x - b.x, a.y - b.y}; }
__device__ __forceinline__ Complex cscale(Complex a, Real s) { return {a.x * s, a.y * s}; }
// (i*s)*a
__device__ __forceinline__ Complex cmul_is(Complex a, Real s) { return {-a.y * s, a.x * s}; }

__device__ __forceinline__ int grad_kidx(int i, int nf) { return (i <= nf / 2) ? i : i - nf; }

// DMK's grid_idx is row-major but FINUFFT stores column-major, so axis 0 is the fastest-varying
// index: kx from i%nf, kz from the slowest. Transposing this is invisible for an isotropic symbol
// and wrong for every projector, so all of them go through here.
__device__ __forceinline__ void grid_kvec(int i, int nf, Real coeff, Real &kx, Real &ky, Real &kz) {
    const int i2 = i % nf;
    const int i1 = (i / nf) % nf;
    const int i0 = i / (nf * nf);
    kx = coeff * Real(grad_kidx(i2, nf));
    ky = coeff * Real(grad_kidx(i1, nf));
    kz = coeff * Real(grad_kidx(i0, nf));
}

// Far-field spectrum: n_channels input spectra -> out_dim output spectra. PROJECTOR selects the
// per-mode operator (0 scalar, 1 dipole, 2 Oseen, 3 stresslet); scaling_coeffs carries the scalar
// radial symbol f in every case.
template <int P>
__device__ void project_kernel(const EspSupportArgs &args) {
    const int ntot = args.ntot;
    const int nf = args.nf;
    const int out_dim = args.out_dim;
    const Real *scaling_coeffs = args.scaling_coeffs;
    const Complex *in = args.chan_in;
    Complex *out = args.chan_out;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ntot) return;

    const Real f = scaling_coeffs[i];

    if constexpr (P == 0) {
        // k is only needed for the gradient rows.
        const Complex ph = cscale(in[i], f);
        out[i] = ph;
        if (out_dim > 1) {
            Real kx, ky, kz;
            grid_kvec(i, nf, args.coeff_grad, kx, ky, kz);
            out[ntot + i] = cmul_is(ph, kx);
            out[2 * ntot + i] = cmul_is(ph, ky);
            out[3 * ntot + i] = cmul_is(ph, kz);
        }
        return;
    }

    Real kx, ky, kz;
    grid_kvec(i, nf, args.coeff_grad, kx, ky, kz);

    if constexpr (P == 1) {
        // -i f (k.d), d being the three dipole-component spectra.
        const Complex dot = cadd(cadd(cscale(in[i], kx), cscale(in[ntot + i], ky)), cscale(in[2 * ntot + i], kz));
        const Complex ph = cmul_is(dot, -f);
        out[i] = ph;
        if (out_dim > 1) {
            out[ntot + i] = cmul_is(ph, kx);
            out[2 * ntot + i] = cmul_is(ph, ky);
            out[3 * ntot + i] = cmul_is(ph, kz);
        }
    } else if constexpr (P == 2) {
        // Oseen: u_i = f(k_i (k.F) - |k|^2 F_i).
        const Complex p0 = in[i], p1 = in[ntot + i], p2 = in[2 * ntot + i];
        const Complex dot = cadd(cadd(cscale(p0, kx), cscale(p1, ky)), cscale(p2, kz));
        const Real dd = (kx * kx + ky * ky + kz * kz) * f;
        out[i] = csub(cscale(dot, kx * f), cscale(p0, dd));
        out[ntot + i] = csub(cscale(dot, ky * f), cscale(p1, dd));
        out[2 * ntot + i] = csub(cscale(dot, kz * f), cscale(p2, dd));
    } else {
        // zz = |k|^2 tr(P) - 2 k^T P k;  u_i = -i f (k_i zz + |k|^2 ((P+P^T)k)_i).
        // P[a][b] is the spectrum of channel a*3+b.
        const Real kvec[3] = {kx, ky, kz};
        const Real ksq = kx * kx + ky * ky + kz * kz;
        Complex P3[3][3];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                P3[a][b] = in[(a * 3 + b) * ntot + i];
        Complex trace = cadd(cadd(P3[0][0], P3[1][1]), P3[2][2]);
        Complex kPk{Real(0), Real(0)};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                kPk = cadd(kPk, cscale(P3[a][b], kvec[a] * kvec[b]));
        const Complex zz = csub(cscale(trace, ksq), cscale(kPk, Real(2)));
        for (int a = 0; a < 3; ++a) {
            Complex prod{Real(0), Real(0)};
            for (int b = 0; b < 3; ++b)
                prod = cadd(prod, cscale(cadd(P3[a][b], P3[b][a]), kvec[b]));
            out[a * ntot + i] = cmul_is(cadd(cscale(zz, kvec[a]), cscale(prod, ksq)), -f);
        }
    }
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

// Removes the long-range field's self-interaction. One shape covers the scalar potential self,
// the Laplace-dipole gradient self (components 1..3) and the Stokeslet per-component self.
__device__ void self_interaction_kernel(const EspSupportArgs &args)
{
    const int n = args.n;
    const Real factor = args.factor;
    const Real *d_charges = args.charges;
    Real *out[4] = {args.pot, args.gx, args.gy, args.gz};

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int k = 0; k < args.self_count; ++k)
        out[args.self_first + k][i] -= factor * d_charges[args.charge_dim * i + k];
}

// out[self_first][i] += factor, for the Stokeslet's free-space zero-mode gauge.
__device__ void add_const_kernel(const EspSupportArgs &args)
{
    const int n = args.n;
    Real *out[4] = {args.pot, args.gx, args.gy, args.gz};
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[args.self_first][i] += args.factor;
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

    // One complex plane per spread channel; the Stresslet's are the outer product force[a]*normal[b].
    const int cd = args.charge_dim;
    if (args.pack_outer) {
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                d_c[(a * 3 + b) * n + i] = {d_charges[cd * i + a] * d_charges[cd * i + 3 + b], Real(0)};
    } else {
        for (int c = 0; c < args.n_channels; ++c)
            d_c[c * n + i] = {d_charges[cd * i + c], Real(0)};
    }
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
        project_kernel<PROJECTOR>(args);
    else if constexpr (STAGE == 5)
        normalize_kernel(args);
    else if constexpr (STAGE == 6)
        extract_real_kernel(args);
    else if constexpr (STAGE == 7)
        self_interaction_kernel(args);
    else if constexpr (STAGE == 8)
        add_const_kernel(args);
    else
        scale_pack_kernel(args);
}
