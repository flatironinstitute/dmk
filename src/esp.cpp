#ifdef DMK_WITH_FINUFFT

#include <dmk/esp.hpp>
#include <dmk/prolate0_fun.hpp>
#include <dmk/prolate.hpp>
#include <finufft_common/kernel.h>
#include <finufft.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <omp.h>
#include <ducc0/fft/fft.h>

using CGrid = std::vector<std::complex<double>>;
using DGrid = std::vector<double>;

static inline int grid_idx(int ix, int iy, int iz, int n_f) {
    return ix * n_f * n_f + iy * n_f + iz;
}

static void fftn_3d(const CGrid &in, CGrid &out, int n) {
    out = in;
    ducc0::vfmav<std::complex<double>> v(out.data(), {(size_t)n, (size_t)n, (size_t)n});
    ducc0::c2c(v, v, {0, 1, 2}, true, 1.0);
}

static void ifftn_3d(const CGrid &in, CGrid &out, int n) {
    out = in;
    ducc0::vfmav<std::complex<double>> v(out.data(), {(size_t)n, (size_t)n, (size_t)n});
    ducc0::c2c(v, v, {0, 1, 2}, false, 1.0 / ((double)n * n * n));
}

namespace dmk {

// ---------------------------------------------------------------------------
// Auto-select P (PSWF stencil width) from eps.
// Uses FINUFFT v2.5.0's formula for kerformula=8 (PSWF), upsampfac=2, dim=3.
// ---------------------------------------------------------------------------
static int esp_ns_from_eps(double eps) {
    const double tolfac = 0.18 * 1.96;  // 0.18 * 1.4^(dim-1) for dim=3
    int ns = static_cast<int>(std::ceil(
        std::log(tolfac / eps) / (M_PI * std::sqrt(0.5)) + 1.0));
    return std::max(2, ns);
}

// ---------------------------------------------------------------------------
// PSWFKernel – thin wrapper around dmk::Prolate0Fun
// ---------------------------------------------------------------------------
struct PSWFKernel {
    dmk::Prolate0Fun pswf;
    int    P;    // stencil width, auto-derived from eps
    double eps;  // tolerance used to select P
    double c;
    double lambda0;
    double c0;
    double scale;
    std::vector<double> pswf_poly_coeffs;

    explicit PSWFKernel(double eps_, int lenw = 8000) : eps(eps_) {
        P = esp_ns_from_eps(eps_);
        int sigma = 2;
        c = M_PI * P * (1.0 - 1.0 / (2 * sigma)) - 0.05;
        pswf = dmk::Prolate0Fun(c, lenw);

        scale = 1.0 / pswf.eval_val(0.0);

        double mu = pswf.rlam20 / M_PI;
        lambda0 = std::sqrt(2.0 * M_PI * mu / c);

        c0 = pswf.int_eval(1.0) * scale;

        int nc = 32;
        auto pswf_lambda = [&](double x) {
            return pswf.eval_val(std::abs(x)) * scale;
        };
        pswf_poly_coeffs = finufft::kernel::poly_fit<double>(pswf_lambda, nc);
    }

    double operator()(double x) const {
        double result = pswf_poly_coeffs[0];
        int nc = pswf_poly_coeffs.size();
        for (int j = 1; j < nc; ++j)
            result = result * x + pswf_poly_coeffs[j];
        return result;
    }

    double integral(double a, double b) const {
        double va = (a == 0.0) ? 0.0 : pswf.int_eval(a);
        double vb = pswf.int_eval(b);
        return (vb - va) * scale;
    }

    double pswf_hat(double k) const {
        return lambda0 * pswf.eval_val(k / c) * scale;
    }
};

// ---------------------------------------------------------------------------
// ESPParams
// ---------------------------------------------------------------------------
struct ESPParams {
    double L;
    double r_c;
    int    P;
    int    n_f;
    double h;
    double lambda0;
    double c;
    double c0;
    int    n;

    ESPParams(double L_, double r_c_, int P_, const PSWFKernel &pswf, int n_)
        : L(L_), r_c(r_c_), P(P_), n(n_) {
        n_f     = static_cast<int>(std::ceil(pswf.c * L / (M_PI * r_c)));
        h       = L / n_f;
        lambda0 = pswf.lambda0;
        c       = pswf.c;
        c0      = pswf.c0;
    }
};

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
static inline Vec3 min_image_vector(const Vec3 &ri, const Vec3 &rj, double L) {
    Vec3 d;
    for (int k = 0; k < 3; ++k) {
        d[k] = ri[k] - rj[k];
        d[k] -= L * std::round(d[k] / L);
    }
    return d;
}

static inline double min_image_distance(const Vec3 &ri, const Vec3 &rj, double L) {
    auto d = min_image_vector(ri, rj, L);
    return std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
}

static inline double min_image_distance_sq(const Vec3 &ri, const Vec3 &rj, double L) {
    auto d = min_image_vector(ri, rj, L);
    return d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
}

// ---------------------------------------------------------------------------
// Cell / neighbour list
// ---------------------------------------------------------------------------
struct CellIndex { int x, y, z; };

static inline CellIndex particle_cell(const Vec3 &r, double L, int n_cells) {
    double cell_size = L / n_cells;
    CellIndex ci;
    ci.x = static_cast<int>(std::floor((r[0] + L/2) / cell_size)) % n_cells;
    ci.y = static_cast<int>(std::floor((r[1] + L/2) / cell_size)) % n_cells;
    ci.z = static_cast<int>(std::floor((r[2] + L/2) / cell_size)) % n_cells;
    if (ci.x < 0) ci.x += n_cells;
    if (ci.y < 0) ci.y += n_cells;
    if (ci.z < 0) ci.z += n_cells;
    return ci;
}

static std::vector<std::vector<int>>
build_neighbor_list(const std::vector<Vec3> &r_src, const ESPParams &params) {
    int n_cells = static_cast<int>(std::floor(params.L / params.r_c));
    if (n_cells < 1) n_cells = 1;

    auto cell_key = [&](int cx, int cy, int cz) {
        return cx * n_cells * n_cells + cy * n_cells + cz;
    };

    std::vector<std::vector<int>> cells(n_cells * n_cells * n_cells);
    for (int j = 0; j < params.n; ++j) {
        auto ci = particle_cell(r_src[j], params.L, n_cells);
        cells[cell_key(ci.x, ci.y, ci.z)].push_back(j);
    }

    std::vector<std::vector<int>> neighbors(params.n);
    auto r_c_squared = params.r_c * params.r_c;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < params.n; ++i) {
        auto ci = particle_cell(r_src[i], params.L, n_cells);
        for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
            int nx = ((ci.x + dx) % n_cells + n_cells) % n_cells;
            int ny = ((ci.y + dy) % n_cells + n_cells) % n_cells;
            int nz = ((ci.z + dz) % n_cells + n_cells) % n_cells;
            for (int j : cells[cell_key(nx, ny, nz)]) {
                if (j == i) continue;
                if (min_image_distance_sq(r_src[i], r_src[j], params.L) <= r_c_squared)
                    neighbors[i].push_back(j);
            }
        }
    }
    return neighbors;
}

// ---------------------------------------------------------------------------
// Short-range sum
// ---------------------------------------------------------------------------
static std::vector<double>
short_range(const std::vector<Vec3> &r_src,
            const std::vector<double> &charges,
            const PSWFKernel &pswf,
            const ESPParams &params,
            const std::vector<std::vector<int>> &neighbors) {
    const int TABLE_SIZE = 10000;
    std::vector<double> table(TABLE_SIZE);
    for (int i = 0; i < TABLE_SIZE; ++i)
        table[i] = pswf.integral(0.0, (double)i / (TABLE_SIZE - 1));

    std::vector<double> pot(params.n, 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < params.n; ++i) {
        for (int j : neighbors[i]) {
            double dist = min_image_distance(r_src[i], r_src[j], params.L);
            double t = dist / params.r_c;

            double pos = t * (TABLE_SIZE - 1);
            int idx = (int)pos;
            double frac = pos - idx;
            double intval = (table[idx] * (1.0 - frac) + table[idx + 1] * frac) / params.c0;

            double x = (1.0 - intval) / (4.0 * M_PI * dist);
            pot[i] += charges[j] * x;
        }
    }
    return pot;
}

// ---------------------------------------------------------------------------
// S_hat(k_vec)
// ---------------------------------------------------------------------------
static inline double S_hat(const PSWFKernel &pswf, const ESPParams &params,
                            const Vec3 &k_vec) {
    double k_mag = std::sqrt(k_vec[0]*k_vec[0] + k_vec[1]*k_vec[1] + k_vec[2]*k_vec[2]);
    return pswf.pswf_hat(k_mag * params.r_c) / (2.0 * k_mag * k_mag) / params.c0;
}

// ---------------------------------------------------------------------------
// Precompute scaling coefficients
// ---------------------------------------------------------------------------
static DGrid precompute_scaling_coefficients(const PSWFKernel &pswf,
                                              const ESPParams &params) {
    int nf = params.n_f;
    std::vector<int> k_idx(nf);
    for (int i = 0; i < nf; ++i)
        k_idx[i] = (i <= nf/2) ? i : i - nf;

    // 1-D phi_hat values
    std::vector<double> phi_hat_1d(nf);
    for (int i = 0; i < nf; ++i) {
        double k_vec = 2.0 * M_PI * k_idx[i] / params.L;
        double arg   = k_vec * (params.P * params.h) / 2.0;
        phi_hat_1d[i] = (params.P * params.h / 2.0) * pswf.pswf_hat(arg);
    }

    DGrid p(nf * nf * nf, 0.0);
    for (int ix = 0; ix < nf; ++ix)
    for (int iy = 0; iy < nf; ++iy)
    for (int iz = 0; iz < nf; ++iz) {
        Vec3 k_vec = { 2.0*M_PI*k_idx[ix]/params.L,
                       2.0*M_PI*k_idx[iy]/params.L,
                       2.0*M_PI*k_idx[iz]/params.L };
        if (k_vec[0] == 0.0 && k_vec[1] == 0.0 && k_vec[2] == 0.0) continue;

        double s  = S_hat(pswf, params, k_vec);
        double ph = phi_hat_1d[ix] * phi_hat_1d[iy] * phi_hat_1d[iz];
        p[grid_idx(ix, iy, iz, nf)] = s / (params.L*params.L*params.L
                                            * ph * ph
                                            * static_cast<double>(nf*nf*nf));
    }
    return p;
}

// ---------------------------------------------------------------------------
// Long-range contribution via FINUFFT spreading/interpolation
// ---------------------------------------------------------------------------
static std::vector<double>
long_range(const std::vector<Vec3> &r_src,
           const std::vector<double> &charges,
           const PSWFKernel &pswf,
           const ESPParams &params) {
    int n    = params.n;
    int nf   = params.n_f;
    int ntot = nf * nf * nf;

    double scale = 2.0 * M_PI / params.L;
    std::vector<double> x(n), y(n), z(n);
    for (int j = 0; j < n; j++) {
        x[j] = r_src[j][0] * scale;
        y[j] = r_src[j][1] * scale;
        z[j] = r_src[j][2] * scale;
    }

    std::vector<std::complex<double>> c(n);
    for (int j = 0; j < n; j++)
        c[j] = {charges[j], 0.0};

    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.spreadinterponly = 1;
    opts.upsampfac = 2;
    double tol = pswf.eps;

    // 1. Spread: NU points -> uniform grid (type 1)
    std::vector<std::complex<double>> b(ntot, 0.0);
    int ier = finufft3d1(n, x.data(), y.data(), z.data(),
                         c.data(), +1, tol,
                         nf, nf, nf, b.data(), &opts);
    if (ier > 1)
        throw std::runtime_error("finufft3d1 spread failed, ier=" + std::to_string(ier));

    // 2. Forward FFT
    CGrid b_hat(ntot);
    fftn_3d(b, b_hat, nf);

    // 3. Diagonal scaling
    DGrid p = precompute_scaling_coefficients(pswf, params);
    #pragma omp parallel for
    for (int idx = 0; idx < ntot; ++idx)
        b_hat[idx] *= p[idx];

    // 4. Inverse FFT
    CGrid grid(ntot);
    ifftn_3d(b_hat, grid, nf);

    // 5. Interpolate: uniform grid -> NU points (type 2)
    std::vector<std::complex<double>> pot_c(n);
    ier = finufft3d2(n, x.data(), y.data(), z.data(),
                     pot_c.data(), +1, tol,
                     nf, nf, nf, grid.data(), &opts);
    if (ier > 1)
        throw std::runtime_error("finufft3d2 interp failed, ier=" + std::to_string(ier));

    std::vector<double> pot(n);
    for (int j = 0; j < n; j++)
        pot[j] = pot_c[j].real();
    return pot;
}

// ---------------------------------------------------------------------------
// Self-interaction correction
// ---------------------------------------------------------------------------
static std::vector<double>
self_interaction(const std::vector<double> &charges,
                 const PSWFKernel &pswf,
                 const ESPParams &params) {
    std::vector<double> self(params.n);
    double factor = pswf(0.0) / (params.r_c * 4.0 * M_PI * params.c0);
    for (int i = 0; i < params.n; ++i)
        self[i] = charges[i] * factor;
    return self;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
ESPResult esp_potential(const std::vector<Vec3> &r_src,
                        const std::vector<double> &charges,
                        double L, double r_c, double eps) {
    PSWFKernel pswf(eps);  // P auto-derived from eps via esp_ns_from_eps
    int n = static_cast<int>(charges.size());
    ESPParams params(L, r_c, pswf.P, pswf, n);

    auto neighbors  = build_neighbor_list(r_src, params);
    auto pot_sr     = short_range(r_src, charges, pswf, params, neighbors);
    auto pot_lr     = long_range(r_src, charges, pswf, params);
    auto pot_self   = self_interaction(charges, pswf, params);

    std::vector<double> total(n);
    for (int i = 0; i < n; ++i)
        total[i] = pot_sr[i] + pot_lr[i] - pot_self[i];

    return { total, pot_sr, pot_lr, pot_self };
}

} // namespace dmk

// ---------------------------------------------------------------------------
// Test: 10 fixed particles with Python reference values
// (reference from esp_prototype_cpp/main.cpp::make_test_10, computed with
//  the long-range-slow Python implementation)
// ---------------------------------------------------------------------------
#include <dmk.h>
#include <doctest/extensions/doctest_mpi.h>
#include <cmath>

MPI_TEST_CASE("[ESP] pdmk_esp 10-particle reference", 1) {
    const double L = 1.0, r_c = 0.2, eps = 1e-6;
    const int n = 10;

    const double r_src[30] = {
        0.131538-0.5, 0.686773-0.5, 0.98255 -0.5,
        0.45865 -0.5, 0.930436-0.5, 0.753356-0.5,
        0.218959-0.5, 0.526929-0.5, 0.0726859-0.5,
        0.678865-0.5, 0.653919-0.5, 0.884707-0.5,
        0.934693-0.5, 0.701191-0.5, 0.436411-0.5,
        0.519416-0.5, 0.762198-0.5, 0.477732-0.5,
        0.0345721-0.5,0.0474645-0.5,0.274907-0.5,
        0.5297  -0.5, 0.328234-0.5, 0.166507-0.5,
        0.00769819-0.5,0.75641-0.5, 0.897656-0.5,
        0.0668422-0.5, 0.365339-0.5,0.0605643-0.5
    };
    const double charges[10] = { 0.2,-0.2, 0.3,-0.3, 0.4,-0.4, 0.5,-0.5, 0.1,-0.1 };
    const double reference[10] = {
         0.055690493646334494, -0.006852575660153015,
        -0.044049084122909810,  0.040206609732000410,
        -0.057055432216358340,  0.063424832953350830,
        -0.069352639532568540,  0.092150894505931390,
         0.060859529062051890,  0.109678060779027990
    };

    pdmk_esp_params params{};
    params.L   = L;
    params.r_c = r_c;
    params.eps = eps;

    double pot[10] = {};
    pdmk_esp(test_comm, params, n, r_src, charges, pot);

    double max_err = 0.0;
    for (int i = 0; i < n; ++i)
        max_err = std::max(max_err, std::abs(pot[i] - reference[i]));

    CHECK(max_err < 5e-4);  // ~1e-6 precision expected; 5e-4 is a generous bound
}

#endif // DMK_WITH_FINUFFT
