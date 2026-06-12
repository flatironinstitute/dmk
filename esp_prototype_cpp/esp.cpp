#include <dmk/prolate0_fun.hpp>
#include <dmk/prolate.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// PSWFKernel – thin wrapper around dmk::Prolate0Fun
// ---------------------------------------------------------------------------
struct PSWFKernel {
    dmk::Prolate0Fun pswf;
    double c;        // bandwidth parameter
    double lambda0;  // F_c eigenvalue: sqrt(2*pi*mu/c), adjusted for normalisation
    double c0;       // integral of normalised pswf over [0,1]
    double scale;    // 1 / pswf(0), so that (*this)(0) == 1

    explicit PSWFKernel(double eps, int lenw = 8000) {
        double c_val;
        dmk::prolc180(eps, c_val);
        c = c_val;
        pswf = dmk::Prolate0Fun(c_val, lenw);

        // Normalise so that pswf(0) = 1, matching Python's pswf.normalize()
        scale = 1.0 / pswf.eval_val(0.0);

        // sinc-kernel eigenvalue mu is stored as pswf.rlam20
        // lambda0 is a property of the operator, independent of normalisation
        // rlam20 from prol0ini is pi * mu (sinc kernel uses sin(c*dt)/(c*dt)
        // convention without the 1/pi prefactor), so divide out pi to get
        // the same mu as Python's Prolate0.eigenvalue ≈ 1.
        double mu = pswf.rlam20 / M_PI;
        lambda0 = std::sqrt(2.0 * M_PI * mu / c);

        // c0 = integral of normalised pswf over [0, 1]
        c0 = pswf.int_eval(1.0) * scale;
    }

    double operator()(double x) const { return pswf.eval_val(x) * scale; }

    double integral(double a, double b) const {
        double va = (a == 0.0) ? 0.0 : pswf.int_eval(a);
        double vb = pswf.int_eval(b);
        return (vb - va) * scale;
    }

    // Fourier transform of normalised pswf: lambda0 * pswf_normalised(k / c)
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
using Vec3 = std::array<double, 3>;

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

// Returns neighbour lists: neighbours[i] = list of j != i within r_c
static std::vector<std::vector<int>>
build_neighbor_list(const std::vector<Vec3> &r_src, const ESPParams &params) {
    int n_cells = static_cast<int>(std::floor(params.L / params.r_c));
    if (n_cells < 1) n_cells = 1;

    // Map cell -> list of particle indices
    std::unordered_map<int, std::vector<int>> cells;
    auto cell_key = [&](int cx, int cy, int cz) {
        return cx * n_cells * n_cells + cy * n_cells + cz;
    };

    for (int j = 0; j < params.n; ++j) {
        auto ci = particle_cell(r_src[j], params.L, n_cells);
        cells[cell_key(ci.x, ci.y, ci.z)].push_back(j);
    }

    std::vector<std::vector<int>> neighbors(params.n);
    for (int i = 0; i < params.n; ++i) {
        auto ci = particle_cell(r_src[i], params.L, n_cells);
        for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
            int nx = ((ci.x + dx) % n_cells + n_cells) % n_cells;
            int ny = ((ci.y + dy) % n_cells + n_cells) % n_cells;
            int nz = ((ci.z + dz) % n_cells + n_cells) % n_cells;
            auto it = cells.find(cell_key(nx, ny, nz));
            if (it == cells.end()) continue;
            for (int j : it->second) {
                if (j == i) continue;
                if (min_image_distance(r_src[i], r_src[j], params.L) <= params.r_c)
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
    std::vector<double> pot(params.n, 0.0);
    for (int i = 0; i < params.n; ++i) {
        for (int j : neighbors[i]) {
            double dist = min_image_distance(r_src[i], r_src[j], params.L);
            double intval = pswf.integral(0.0, dist / params.r_c) / params.c0;
            double x = (1.0 - intval) / (4.0 * M_PI * dist);
            pot[i] += charges[j] * x;
        }
    }
    return pot;
}

// ---------------------------------------------------------------------------
// S_hat(k_vec)  – spectral Green's function factor
// ---------------------------------------------------------------------------
static inline double S_hat(const PSWFKernel &pswf, const ESPParams &params,
                            const Vec3 &k_vec) {
    double k_mag = std::sqrt(k_vec[0]*k_vec[0] + k_vec[1]*k_vec[1] + k_vec[2]*k_vec[2]);
    return pswf.pswf_hat(k_mag * params.r_c) / (2.0 * k_mag * k_mag) / params.c0;
}

// ---------------------------------------------------------------------------
// Stencil offsets for the ESP spreading/interpolation
// ---------------------------------------------------------------------------
static std::vector<int> stencil_offsets(int P) {
    std::vector<int> off;
    if (P % 2 == 1) {
        int m = P / 2;
        for (int i = -m; i <= m; ++i) off.push_back(i);
    } else {
        for (int i = -P/2; i < P/2; ++i) off.push_back(i);
    }
    return off;
}

// phi(displacement) = pswf(2x/Ph) * pswf(2y/Ph) * pswf(2z/Ph)
static inline double phi_val(const Vec3 &disp, const PSWFKernel &pswf,
                              const ESPParams &params) {
    double Ph = params.P * params.h;
    return pswf(2*disp[0]/Ph) * pswf(2*disp[1]/Ph) * pswf(2*disp[2]/Ph);
}

// ---------------------------------------------------------------------------
// 3-D complex grid helpers (row-major: ix * n_f*n_f + iy * n_f + iz)
// ---------------------------------------------------------------------------
using CGrid = std::vector<std::complex<double>>;
using DGrid = std::vector<double>;

static inline int grid_idx(int ix, int iy, int iz, int n_f) {
    return ix * n_f * n_f + iy * n_f + iz;
}

// ---------------------------------------------------------------------------
// Spreading  (particle charges -> uniform grid)
// ---------------------------------------------------------------------------
static CGrid spreading(const std::vector<Vec3> &r_src,
                        const std::vector<double> &charges,
                        const PSWFKernel &pswf,
                        const ESPParams &params) {
    int nf = params.n_f;
    CGrid b(nf * nf * nf, 0.0);
    auto offsets = stencil_offsets(params.P);

    for (int j = 0; j < params.n; ++j) {
        // nearest grid point
        std::array<int,3> l_center;
        for (int d = 0; d < 3; ++d)
            l_center[d] = static_cast<int>(std::round(r_src[j][d] / params.h));

        // P neighbours in each dimension
        std::array<std::vector<int>,3> lx;
        for (int d = 0; d < 3; ++d) {
            lx[d].resize(offsets.size());
            for (size_t k = 0; k < offsets.size(); ++k)
                lx[d][k] = ((l_center[d] + offsets[k]) % nf + nf) % nf;
        }

        for (size_t ix = 0; ix < offsets.size(); ++ix)
        for (size_t iy = 0; iy < offsets.size(); ++iy)
        for (size_t iz = 0; iz < offsets.size(); ++iz) {
            Vec3 grid_pt = { params.h * lx[0][ix],
                             params.h * lx[1][iy],
                             params.h * lx[2][iz] };
            Vec3 disp = min_image_vector(r_src[j], grid_pt, params.L);
            double pv = phi_val(disp, pswf, params);
            b[grid_idx(lx[0][ix], lx[1][iy], lx[2][iz], nf)] += charges[j] * pv;
        }
    }
    return b;
}

// ---------------------------------------------------------------------------
// Interpolation (uniform grid -> particle potentials)
// ---------------------------------------------------------------------------
static std::vector<double>
interpolation(const CGrid &c_grid,
              const std::vector<Vec3> &r_src,
              const PSWFKernel &pswf,
              const ESPParams &params) {
    int nf = params.n_f;
    std::vector<double> pot(params.n, 0.0);
    auto offsets = stencil_offsets(params.P);

    for (int i = 0; i < params.n; ++i) {
        std::array<int,3> l_center;
        for (int d = 0; d < 3; ++d)
            l_center[d] = static_cast<int>(std::round(r_src[i][d] / params.h));

        std::array<std::vector<int>,3> lx;
        for (int d = 0; d < 3; ++d) {
            lx[d].resize(offsets.size());
            for (size_t k = 0; k < offsets.size(); ++k)
                lx[d][k] = ((l_center[d] + offsets[k]) % nf + nf) % nf;
        }

        for (size_t ix = 0; ix < offsets.size(); ++ix)
        for (size_t iy = 0; iy < offsets.size(); ++iy)
        for (size_t iz = 0; iz < offsets.size(); ++iz) {
            Vec3 grid_pt = { params.h * lx[0][ix],
                             params.h * lx[1][iy],
                             params.h * lx[2][iz] };
            Vec3 disp = min_image_vector(r_src[i], grid_pt, params.L);
            double pv = phi_val(disp, pswf, params);
            pot[i] += c_grid[grid_idx(lx[0][ix], lx[1][iy], lx[2][iz], nf)].real() * pv;
        }
    }
    return pot;
}

// ---------------------------------------------------------------------------
// Precompute 1-D phi_hat values (FFT-order k indices)
// ---------------------------------------------------------------------------
static std::vector<double>
precompute_phi_hat_1d(const std::vector<int> &k_idx,
                      const PSWFKernel &pswf,
                      const ESPParams &params) {
    int nf = params.n_f;
    std::vector<double> phi_hat(nf);
    for (int i = 0; i < nf; ++i) {
        double k_vec = 2.0 * M_PI * k_idx[i] / params.L;
        double arg   = k_vec * (params.P * params.h) / 2.0;
        phi_hat[i]   = (params.P * params.h / 2.0) * pswf.pswf_hat(arg);
    }
    return phi_hat;
}

// ---------------------------------------------------------------------------
// Precompute scaling coefficients p[kx,ky,kz]
// ---------------------------------------------------------------------------
// Returns a real grid in FFT order (same layout as b_hat after fftn)
static DGrid precompute_scaling_coefficients(const PSWFKernel &pswf,
                                              const ESPParams &params) {
    int nf = params.n_f;
    // FFT-order k indices: 0, 1, ..., nf/2-1, -nf/2, ..., -1
    std::vector<int> k_idx(nf);
    for (int i = 0; i < nf; ++i)
        k_idx[i] = (i <= nf/2) ? i : i - nf;

    auto phi_hat_1d = precompute_phi_hat_1d(k_idx, pswf, params);

    DGrid p(nf * nf * nf, 0.0);
    for (int ix = 0; ix < nf; ++ix)
    for (int iy = 0; iy < nf; ++iy)
    for (int iz = 0; iz < nf; ++iz) {
        Vec3 k_vec = { 2.0*M_PI*k_idx[ix]/params.L,
                       2.0*M_PI*k_idx[iy]/params.L,
                       2.0*M_PI*k_idx[iz]/params.L };
        if (k_vec[0] == 0.0 && k_vec[1] == 0.0 && k_vec[2] == 0.0) continue;

        double s = S_hat(pswf, params, k_vec);
        double ph = phi_hat_1d[ix] * phi_hat_1d[iy] * phi_hat_1d[iz];
        // divide by n_f^3 to match numpy's unnormalised ifft convention
        p[grid_idx(ix, iy, iz, nf)] = s / (params.L*params.L*params.L
                                            * ph * ph
                                            * static_cast<double>(nf*nf*nf));
    }
    return p;
}

// ---------------------------------------------------------------------------
// FFT interface – replace with your FFT library (e.g. FFTW)
// Signatures follow numpy convention:
//   fftn:  forward, no normalisation
//   ifftn: inverse, divide by N (already done inside or caller must do it)
// ---------------------------------------------------------------------------
// These are declared extern; provide an implementation file that links
// against FFTW or another library.
extern void fftn_3d (const CGrid &in, CGrid &out, int n);
extern void ifftn_3d(const CGrid &in, CGrid &out, int n);

// ---------------------------------------------------------------------------
// Long-range (fast ESP) contribution
// ---------------------------------------------------------------------------
static std::vector<double>
long_range_fast(const std::vector<Vec3> &r_src,
                const std::vector<double> &charges,
                const PSWFKernel &pswf,
                const ESPParams &params) {
    // 1. Spread charges onto grid
    CGrid b = spreading(r_src, charges, pswf, params);

    // 2. Forward FFT
    int nf = params.n_f;
    CGrid b_hat(nf * nf * nf);
    fftn_3d(b, b_hat, nf);

    // 3. Diagonal scaling
    DGrid p = precompute_scaling_coefficients(pswf, params);
    for (int idx = 0; idx < nf*nf*nf; ++idx)
        b_hat[idx] *= p[idx];

    // 4. Inverse FFT
    CGrid grid(nf * nf * nf);
    ifftn_3d(b_hat, grid, nf);

    // 5. Interpolate back to particle positions
    return interpolation(grid, r_src, pswf, params);
}

// ---------------------------------------------------------------------------
// Self-interaction correction
// ---------------------------------------------------------------------------
static std::vector<double>
self_interaction(const std::vector<Vec3> &/*r_src*/,
                 const std::vector<double> &charges,
                 const PSWFKernel &pswf,
                 const ESPParams &params) {
    std::vector<double> self(params.n);
    double factor = pswf(0.0) / (params.r_c * 4.0 * M_PI * params.c0);
    for (int i = 0; i < params.n; ++i)
        self[i] = charges[i] * factor;
    return self;
}

// ---------------------------------------------------------------------------
// Debug: print internal kernel/params values for cross-checking with Python
// ---------------------------------------------------------------------------
void debug_pswf(double eps, double L, double r_c, int P, int n) {
    PSWFKernel pswf(eps);
    ESPParams params(L, r_c, P, pswf, n);

    printf("=== PSWFKernel / ESPParams debug ===\n");
    printf("c           = %.6f\n", pswf.c);
    printf("mu (rlam20) = %.6f\n", pswf.pswf.rlam20);
    printf("lambda0     = %.6f\n", pswf.lambda0);
    printf("c0          = %.6f\n", pswf.c0);
    printf("pswf(0)     = %.6f\n", pswf(0.0));
    printf("pswf(1)     = %.6e\n", pswf(1.0));
    printf("pswf_hat(0) = %.6f\n", pswf.pswf_hat(0.0));
    printf("n_f         = %d\n",   params.n_f);
    printf("h           = %.6f\n", params.h);
    printf("====================================\n\n");
}

// ---------------------------------------------------------------------------
// Total potential: short_range + long_range_fast - self_interaction
// ---------------------------------------------------------------------------
struct ESPResult {
    std::vector<double> total;
    std::vector<double> short_range_pot;
    std::vector<double> long_range_pot;
    std::vector<double> self_pot;
};

ESPResult esp_potential(const std::vector<Vec3> &r_src,
                        const std::vector<double> &charges,
                        double L, double r_c, int P, double eps) {
    PSWFKernel pswf(eps);
    int n = static_cast<int>(charges.size());
    ESPParams params(L, r_c, P, pswf, n);

    auto neighbors  = build_neighbor_list(r_src, params);
    auto pot_sr     = short_range(r_src, charges, pswf, params, neighbors);
    auto pot_lr     = long_range_fast(r_src, charges, pswf, params);
    auto pot_self   = self_interaction(r_src, charges, pswf, params);

    std::vector<double> total(n);
    for (int i = 0; i < n; ++i)
        total[i] = pot_sr[i] + pot_lr[i] - pot_self[i];

    return { total, pot_sr, pot_lr, pot_self };
}