#ifdef DMK_BUILD_ESP

#include <dmk/esp.hpp>
#include <dmk/prolate.hpp>
#include <dmk/prolate0_fun.hpp>
#include <finufft.h>
#include <finufft_common/kernel.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <ducc0/fft/fft.h>
#include <omp.h>
#include <stdexcept>
#include <vector>

using Vec3 = std::array<double, 3>;
using CGrid = std::vector<std::complex<double>>;
using DGrid = std::vector<double>;

static inline int grid_idx(int ix, int iy, int iz, int n_f) { return ix * n_f * n_f + iy * n_f + iz; }

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
    const double tolfac = 0.18 * 1.96; // 0.18 * 1.4^(dim-1) for dim=3
    int ns = static_cast<int>(std::ceil(std::log(tolfac / eps) / (M_PI * std::sqrt(0.5)) + 1.0));
    return std::max(2, ns);
}

// ---------------------------------------------------------------------------
// PSWFKernel – thin wrapper around dmk::Prolate0Fun
// ---------------------------------------------------------------------------
struct PSWFKernel {
    dmk::Prolate0Fun pswf;
    int P;      // stencil width, auto-derived from eps
    double eps; // tolerance used to select P
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
        auto pswf_lambda = [&](double x) { return pswf.eval_val(std::abs(x)) * scale; };
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

    double pswf_hat(double k) const { return lambda0 * pswf.eval_val(k / c) * scale; }
};

// ---------------------------------------------------------------------------
// ESPParams
// ---------------------------------------------------------------------------
struct ESPParams {
    double L;
    double r_c;
    int P;
    int n_f;
    double h;
    double lambda0;
    double c;
    double c0;
    int n;

    ESPParams(double L_, double r_c_, int P_, const PSWFKernel &pswf, int n_) : L(L_), r_c(r_c_), P(P_), n(n_) {
        n_f = static_cast<int>(std::ceil(pswf.c * L / (M_PI * r_c)));
        h = L / n_f;
        lambda0 = pswf.lambda0;
        c = pswf.c;
        c0 = pswf.c0;
    }
};

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
template <typename Real>
static inline Vec3T<Real> min_image_vector(const Vec3T<Real> &ri, const Vec3T<Real> &rj, Real L) {
    Vec3T<Real> d;
    for (int k = 0; k < 3; ++k) {
        d[k] = ri[k] - rj[k];
        d[k] -= L * std::round(d[k] / L);
    }
    return d;
}

template <typename Real>
static inline Real min_image_distance(const Vec3T<Real> &ri, const Vec3T<Real> &rj, Real L) {
    auto d = min_image_vector<Real>(ri, rj, L);
    return std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
}

template <typename Real>
static inline Real min_image_distance_sq(const Vec3T<Real> &ri, const Vec3T<Real> &rj, Real L) {
    auto d = min_image_vector<Real>(ri, rj, L);
    return d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
}

// ---------------------------------------------------------------------------
// Cell / neighbour list
// ---------------------------------------------------------------------------
struct CellIndex {
    int x, y, z;
};

template <typename Real>
static inline CellIndex particle_cell(const Vec3T<Real> &r, Real L, int n_cells) {
    Real cell_size = L / n_cells;
    CellIndex ci;
    ci.x = static_cast<int>(std::floor((r[0] + L / 2) / cell_size)) % n_cells;
    ci.y = static_cast<int>(std::floor((r[1] + L / 2) / cell_size)) % n_cells;
    ci.z = static_cast<int>(std::floor((r[2] + L / 2) / cell_size)) % n_cells;
    if (ci.x < 0)
        ci.x += n_cells;
    if (ci.y < 0)
        ci.y += n_cells;
    if (ci.z < 0)
        ci.z += n_cells;
    return ci;
}

template <typename Real>
static std::vector<std::vector<int>>
build_neighbor_list(const std::vector<Vec3T<Real>> &r_src, const ESPParams &params) {
    int n_cells = static_cast<int>(std::floor(params.L / params.r_c));
    if (n_cells < 1) n_cells = 1;
    const Real L   = Real(params.L);
    const Real r_c = Real(params.r_c);

    auto cell_key = [&](int cx, int cy, int cz) { return cx * n_cells * n_cells + cy * n_cells + cz; };

    std::vector<std::vector<int>> cells(n_cells * n_cells * n_cells);
    for (int j = 0; j < params.n; ++j) {
        auto ci = particle_cell<Real>(r_src[j], L, n_cells);
        cells[cell_key(ci.x, ci.y, ci.z)].push_back(j);
    }

    std::vector<std::vector<int>> neighbors(params.n);
    const Real r_c_sq = r_c * r_c;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < params.n; ++i) {
        auto ci = particle_cell<Real>(r_src[i], L, n_cells);
        for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
            int nx = ((ci.x + dx) % n_cells + n_cells) % n_cells;
            int ny = ((ci.y + dy) % n_cells + n_cells) % n_cells;
            int nz = ((ci.z + dz) % n_cells + n_cells) % n_cells;
            for (int j : cells[cell_key(nx, ny, nz)]) {
                if (j == i) continue;
                if (min_image_distance_sq<Real>(r_src[i], r_src[j], L) <= r_c_sq)
                    neighbors[i].push_back(j);
            }
        }
    }
    return neighbors;
}

// ---------------------------------------------------------------------------
// Short-range sum
// ---------------------------------------------------------------------------
template <typename Real>
static std::vector<Real>
short_range(const std::vector<Vec3T<Real>> &r_src,
            const std::vector<Real> &charges,
            const PSWFKernel &pswf,
            const ESPParams &params,
            const std::vector<std::vector<int>> &neighbors) {
    // PSWF integral table is always double (kernel precision requirement)
    const int TABLE_SIZE = 10000;
    std::vector<double> table(TABLE_SIZE);
    for (int i = 0; i < TABLE_SIZE; ++i)
        table[i] = pswf.integral(0.0, (double)i / (TABLE_SIZE - 1));

    const Real L   = Real(params.L);
    const Real r_c = Real(params.r_c);
    std::vector<Real> pot(params.n, Real(0));
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < params.n; ++i) {
        for (int j : neighbors[i]) {
            Real dist = min_image_distance<Real>(r_src[i], r_src[j], L);
            double t = double(dist) / params.r_c;

            double pos = t * (TABLE_SIZE - 1);
            int idx = (int)pos;
            double frac = pos - idx;
            double intval = (table[idx] * (1.0 - frac) + table[idx + 1] * frac) / params.c0;

            pot[i] += charges[j] * Real((1.0 - intval) / (4.0 * M_PI * double(dist)));
        }
    }
    return pot;
}

// ---------------------------------------------------------------------------
// S_hat(k_vec)
// ---------------------------------------------------------------------------
static inline double S_hat(const PSWFKernel &pswf, const ESPParams &params, const Vec3 &k_vec) {
    double k_mag = std::sqrt(k_vec[0] * k_vec[0] + k_vec[1] * k_vec[1] + k_vec[2] * k_vec[2]);
    return pswf.pswf_hat(k_mag * params.r_c) / (2.0 * k_mag * k_mag) / params.c0;
}

// ---------------------------------------------------------------------------
// Precompute scaling coefficients
// ---------------------------------------------------------------------------
static DGrid precompute_scaling_coefficients(const PSWFKernel &pswf, const ESPParams &params) {
    int nf = params.n_f;
    std::vector<int> k_idx(nf);
    for (int i = 0; i < nf; ++i)
        k_idx[i] = (i <= nf / 2) ? i : i - nf;

    // 1-D phi_hat values
    std::vector<double> phi_hat_1d(nf);
    for (int i = 0; i < nf; ++i) {
        double k_vec = 2.0 * M_PI * k_idx[i] / params.L;
        double arg = k_vec * (params.P * params.h) / 2.0;
        phi_hat_1d[i] = (params.P * params.h / 2.0) * pswf.pswf_hat(arg);
    }

    DGrid p(nf * nf * nf, 0.0);
    for (int ix = 0; ix < nf; ++ix)
        for (int iy = 0; iy < nf; ++iy)
            for (int iz = 0; iz < nf; ++iz) {
                Vec3 k_vec = {2.0 * M_PI * k_idx[ix] / params.L, 2.0 * M_PI * k_idx[iy] / params.L,
                              2.0 * M_PI * k_idx[iz] / params.L};
                if (k_vec[0] == 0.0 && k_vec[1] == 0.0 && k_vec[2] == 0.0)
                    continue;

                double s = S_hat(pswf, params, k_vec);
                double ph = phi_hat_1d[ix] * phi_hat_1d[iy] * phi_hat_1d[iz];
                p[grid_idx(ix, iy, iz, nf)] =
                    s / (params.L * params.L * params.L * ph * ph * static_cast<double>(nf * nf * nf));
            }
    return p;
}

// ---------------------------------------------------------------------------
// Long-range contribution via FINUFFT spreading/interpolation
// ---------------------------------------------------------------------------
template <typename Real>
static std::vector<Real>
long_range(const std::vector<Vec3T<Real>> &r_src,
           const std::vector<Real> &charges,
           const PSWFKernel &pswf,
           const ESPParams &params,
           const DGrid &scaling_coeffs) {
    int n    = params.n;
    int nf   = params.n_f;
    int ntot = nf * nf * nf;

    // FINUFFT and FFT require double; convert Real inputs here
    double scale = 2.0 * M_PI / params.L;
    std::vector<double> x(n), y(n), z(n);
    for (int j = 0; j < n; j++) {
        x[j] = double(r_src[j][0]) * scale;
        y[j] = double(r_src[j][1]) * scale;
        z[j] = double(r_src[j][2]) * scale;
    }

    std::vector<std::complex<double>> c(n);
    for (int j = 0; j < n; j++)
        c[j] = {double(charges[j]), 0.0};

    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.spreadinterponly = 1;
    opts.upsampfac = 2;
    double tol = pswf.eps;

    // 1. Spread: NU points -> uniform grid (type 1)
    std::vector<std::complex<double>> b(ntot, 0.0);
    int ier = finufft3d1(n, x.data(), y.data(), z.data(), c.data(), +1, tol, nf, nf, nf, b.data(), &opts);
    if (ier > 1)
        throw std::runtime_error("finufft3d1 spread failed, ier=" + std::to_string(ier));

    // 2. Forward FFT
    CGrid b_hat(ntot);
    fftn_3d(b, b_hat, nf);

// 3. Diagonal scaling (precomputed in plan)
#pragma omp parallel for
    for (int idx = 0; idx < ntot; ++idx)
        b_hat[idx] *= scaling_coeffs[idx];

    // 4. Inverse FFT
    CGrid grid(ntot);
    ifftn_3d(b_hat, grid, nf);

    // 5. Interpolate: uniform grid -> NU points (type 2)
    std::vector<std::complex<double>> pot_c(n);
    ier = finufft3d2(n, x.data(), y.data(), z.data(), pot_c.data(), +1, tol, nf, nf, nf, grid.data(), &opts);
    if (ier > 1)
        throw std::runtime_error("finufft3d2 interp failed, ier=" + std::to_string(ier));

    std::vector<Real> pot(n);
    for (int j = 0; j < n; j++)
        pot[j] = Real(pot_c[j].real());
    return pot;
}

// ---------------------------------------------------------------------------
// Self-interaction correction
// ---------------------------------------------------------------------------
template <typename Real>
static std::vector<Real>
self_interaction(const std::vector<Real> &charges,
                 const PSWFKernel &pswf,
                 const ESPParams &params) {
    std::vector<Real> self(params.n);
    Real factor = Real(pswf(0.0) / (params.r_c * 4.0 * M_PI * params.c0));
    for (int i = 0; i < params.n; ++i)
        self[i] = charges[i] * factor;
    return self;
}

// ---------------------------------------------------------------------------
// EspPlan — caches everything independent of particle positions/charges.
// ---------------------------------------------------------------------------
struct EspPlan {
    PSWFKernel pswf;
    ESPParams params_base; // n=0 placeholder; n_f/h/kernel params pre-filled
    DGrid scaling_coeffs;

    EspPlan(double L, double r_c, double eps) : pswf(eps), params_base(L, r_c, pswf.P, pswf, 0) {
        scaling_coeffs = precompute_scaling_coefficients(pswf, params_base);
    }
};

EspPlan *esp_create_plan(double L, double r_c, double eps) { return new EspPlan(L, r_c, eps); }

void esp_destroy_plan(EspPlan *plan) { delete plan; }

template <typename Real>
std::vector<Real> esp_eval(EspPlan *plan,
                            const std::vector<Vec3T<Real>> &r_src,
                            const std::vector<Real> &charges) {
    int n = static_cast<int>(charges.size());
    ESPParams params = plan->params_base;
    params.n = n;

    auto neighbors = build_neighbor_list<Real>(r_src, params);
    auto pot_sr    = short_range<Real>(r_src, charges, plan->pswf, params, neighbors);
    auto pot_lr    = long_range<Real>(r_src, charges, plan->pswf, params, plan->scaling_coeffs);
    auto pot_self  = self_interaction<Real>(charges, plan->pswf, params);

    std::vector<Real> total(n);
    for (int i = 0; i < n; ++i)
        total[i] = pot_sr[i] + pot_lr[i] - pot_self[i];
    return total;
}

// ---------------------------------------------------------------------------
// Convenience one-shot entry point (create + eval + destroy).
// ---------------------------------------------------------------------------
template <typename Real>
std::vector<Real> esp_potential(const std::vector<Vec3T<Real>> &r_src,
                                 const std::vector<Real> &charges,
                                 double L, double r_c, double eps) {
    auto *plan = esp_create_plan(L, r_c, eps);
    auto result = esp_eval<Real>(plan, r_src, charges);
    esp_destroy_plan(plan);
    return result;
}

template std::vector<float>  esp_eval<float> (EspPlan *, const std::vector<Vec3T<float>>  &, const std::vector<float>  &);
template std::vector<double> esp_eval<double>(EspPlan *, const std::vector<Vec3T<double>> &, const std::vector<double> &);
template std::vector<float>  esp_potential<float> (const std::vector<Vec3T<float>>  &, const std::vector<float>  &, double, double, double);
template std::vector<double> esp_potential<double>(const std::vector<Vec3T<double>> &, const std::vector<double> &, double, double, double);

} // namespace dmk

#endif // DMK_BUILD_ESP
