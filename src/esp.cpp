#include <dmk.h>
#include <dmk/aot_kernels.hpp>
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
#include <sctl.hpp>
#include <stdexcept>
#include <vector>

// After sctl.hpp to avoid type-trait conflicts with <format>
#include <dmk/direct.hpp>
#include <dmk/types.hpp>

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
// PSWFKernel – thin wrapper around dmk::Prolate0Fun.
// P and c are derived from eps (see esp.hpp), matching FINUFFT's PSWF spreader.
// ---------------------------------------------------------------------------
struct PSWFKernel {
    dmk::Prolate0Fun pswf;
    int P;      // stencil width, auto-derived from eps
    double eps; // tolerance used to select P
    double sigma;
    double c;
    double lambda0;
    double c0;
    double scale;
    std::vector<double> pswf_poly_coeffs;
    std::vector<double> pswf_int_poly_coeffs;

    // sigma_ < 0 means "use esp_sigma_from_eps(eps_)" — pass a positive value to override for experiments.
    explicit PSWFKernel(double eps_, double sigma_ = -1.0, int lenw = 8000) : eps(eps_) {
        sigma = (sigma_ > 0) ? sigma_ : esp_sigma_from_eps(eps_);
        P = esp_ns_from_eps(eps_, sigma);
        c = esp_pswf_c_from_eps(eps_, sigma);
        pswf = dmk::Prolate0Fun(c, lenw);

        scale = 1.0 / pswf.eval_val(0.0);

        double mu = pswf.rlam20 / M_PI;
        lambda0 = std::sqrt(2.0 * M_PI * mu / c);

        c0 = pswf.int_eval(1.0) * scale;

        int nc = 24;
        auto pswf_lambda = [&](double x) { return pswf.eval_val(std::abs(x)) * scale; };
        pswf_poly_coeffs = finufft::kernel::poly_fit<double>(pswf_lambda, nc);

        nc = 20;
        auto pswf_int_lambda = [&](double t) { return pswf.int_eval(t) * scale; };
        pswf_int_poly_coeffs = finufft::kernel::poly_fit<double>(pswf_int_lambda, nc);
    }

    double operator()(double x) const {
        double result = pswf_poly_coeffs[0];
        int nc = pswf_poly_coeffs.size();
        for (int j = 1; j < nc; ++j)
            result = result * x + pswf_poly_coeffs[j];
        return result;
    }

    double integral_eval(double t) const {
        double result = pswf_int_poly_coeffs[0];
        int nc = pswf_int_poly_coeffs.size();
        for (int j = 1; j < nc; ++j)
            result = result * t + pswf_int_poly_coeffs[j];
        return result;
    }

    double integral(double a, double b) const {
        double va = (a == 0.0) ? 0.0 : integral_eval(a);
        double vb = integral_eval(b);
        return vb - va;
    }

    double pswf_hat(double k) const {
        const double x = k / c;
        return std::fabs(x) > 1 ? 0.0 : lambda0 * (*this)(x);
    }
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
// Cell list
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

// ---------------------------------------------------------------------------
// Cell list: particles sorted into cubic cells for cache-friendly traversal.
// ---------------------------------------------------------------------------
template <typename Real>
struct CellList {
    int n_cells;
    std::vector<int> cell_start; // size n_cells^3 + 1 (exclusive prefix sum)
    std::vector<Real> rs, qs;    // particle data reordered by cell
    std::vector<int> orig;       // orig[slot] = original particle index
};

template <typename Real>
static CellList<Real> build_cell_list(const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges,
                                      const ESPParams &params, int nc) {
    CellList<Real> cl;
    cl.n_cells = nc;
    const int ncells = nc * nc * nc;
    const int n = params.n;
    const Real L_r = Real(params.L);

    auto cell_of = [&](const Vec3T<Real> &r) {
        CellIndex ci = particle_cell<Real>(r, L_r, nc);
        return (ci.x * nc + ci.y) * nc + ci.z;
    };

    // pass 1: count per cell
    std::vector<int> count(ncells, 0), cidx(n);
    for (int j = 0; j < n; ++j) {
        int c = cell_of(r_src[j]);
        cidx[j] = c;
        ++count[c];
    }

    // pass 2: exclusive prefix sum -> cell start offsets
    cl.cell_start.assign(ncells + 1, 0);
    for (int c = 0; c < ncells; ++c)
        cl.cell_start[c + 1] = cl.cell_start[c] + count[c];

    // pass 3: scatter into sorted arrays
    cl.rs.resize(3 * n);
    cl.qs.resize(n);
    cl.orig.resize(n);
    std::vector<int> cursor(cl.cell_start.begin(), cl.cell_start.end() - 1);
    for (int j = 0; j < n; ++j) {
        int slot = cursor[cidx[j]]++;
        cl.rs[3*slot + 0] = r_src[j][0];
        cl.rs[3*slot + 1] = r_src[j][1];
        cl.rs[3*slot + 2] = r_src[j][2];
        cl.qs[slot] = charges[j];
        cl.orig[slot] = j;
    }
    return cl;
}

// ---------------------------------------------------------------------------
// Scalar (non-SIMD) ESP short-range evaluator built from runtime coefficients.
// Used when a non-default sigma is supplied so the precompiled tables can't be
// used. Computes charge_j * P(t) / d for every pair within the cutoff, where
// t = d/r_c and P is the polynomial approximation of (1-I(t)/c0)/(4*pi).
// ---------------------------------------------------------------------------
template <typename Real>
static residual_evaluator_func<Real> make_esp_scalar_evaluator(std::vector<Real> coeffs) {
    return [coeffs = std::move(coeffs)](Real rsc, Real /*cen*/, Real d2max, Real /*thresh2*/,
                                        int n_src, const Real *r_src, const Real *charge,
                                        const Real */*normals*/, int n_trg, const Real *r_trg, Real *pot) {
        const int nc = static_cast<int>(coeffs.size());
        for (int i = 0; i < n_trg; ++i) {
            const Real xi = r_trg[3*i], yi = r_trg[3*i+1], zi = r_trg[3*i+2];
            for (int j = 0; j < n_src; ++j) {
                const Real dx = r_src[3*j] - xi, dy = r_src[3*j+1] - yi, dz = r_src[3*j+2] - zi;
                const Real d2 = dx*dx + dy*dy + dz*dz;
                if (d2 == Real(0) || d2 > d2max) continue;
                const Real d = std::sqrt(d2);
                const Real t = d * rsc;
                Real pt = coeffs[nc - 1];
                for (int k = nc - 2; k >= 0; --k) pt = pt * t + coeffs[k];
                pot[i] += charge[j] * pt / d;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Short-range sum (fast path: cell-list / box-reordering, no neighbor list).
// Parallelizes over home cells — each cell's output slots are disjoint across
// threads, so pot_sorted[a] is written by exactly one thread (no atomics).
// Falls back to the neighbor-list path when r_c > L/3 (nc < 3).
// ---------------------------------------------------------------------------
template <typename Real>
static std::vector<Real> short_range_fast(const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges,
                                          const ESPParams &params,
                                          const residual_evaluator_func<Real> &evaluator) {
    // 27-cell stencil requires nc >= 3 so periodic images aren't double-counted
    const int nc = static_cast<int>(std::floor(params.L / params.r_c));
    if (nc < 3)
        throw std::runtime_error("short_range_fast requires r_c <= L/3 (nc >= 3)");

    CellList<Real> cl = build_cell_list<Real>(r_src, charges, params, nc);

    const Real L = Real(params.L);
    const Real r_c_sq = Real(params.r_c) * Real(params.r_c);
    const int n = params.n;
    const Real rsc = Real(1.0 / params.r_c);

    std::vector<Real> pot_sorted(n, Real(0));

    // For each cell coordinate c in [0, nc) and each delta d in {-1,0,+1} (stored as d=0,1,2):
    // nbc_tab[c*3+d] = neighbor cell index, off_tab[c*3+d] = image shift to subtract.
    // Precomputed once; all three dimensions share the same table since they share nc.
    std::vector<int> nbc_tab(nc * 3);
    std::vector<Real> off_tab(nc * 3);
    for (int c = 0; c < nc; ++c) {
        for (int d = 0; d < 3; ++d) {
            int ci = c + d - 1;
            if (ci < 0) {
                nbc_tab[c * 3 + d] = ci + nc;
                off_tab[c * 3 + d] = -L;
            } else if (ci >= nc) {
                nbc_tab[c * 3 + d] = ci - nc;
                off_tab[c * 3 + d] = L;
            } else {
                nbc_tab[c * 3 + d] = ci;
                off_tab[c * 3 + d] = Real(0);
            }
        }
    }

#pragma omp parallel
    {
        // Per-thread scratch: the gathered, image-shifted neighbor sources
        std::vector<Real> r_src_g, charge_g;
#pragma omp for schedule(dynamic) collapse(3)
        for (int cx = 0; cx < nc; ++cx)
            for (int cy = 0; cy < nc; ++cy)
                for (int cz = 0; cz < nc; ++cz) {
                    const int home = (cx * nc + cy) * nc + cz;
                    const int hbeg = cl.cell_start[home], hend = cl.cell_start[home + 1];
                    const int n_trg = hend - hbeg;
                    if (n_trg == 0)
                        continue;

                    const Real *__restrict__ r_trg_ptr = cl.rs.data() + 3 * hbeg;

                    const int *__restrict__ nbc_cx = &nbc_tab[cx * 3];
                    const int *__restrict__ nbc_cy = &nbc_tab[cy * 3];
                    const int *__restrict__ nbc_cz = &nbc_tab[cz * 3];
                    const Real *__restrict__ off_cx = &off_tab[cx * 3];
                    const Real *__restrict__ off_cy = &off_tab[cy * 3];
                    const Real *__restrict__ off_cz = &off_tab[cz * 3];

                    int n_src = 0;
                    for (int dx = 0; dx < 3; ++dx)
                        for (int dy = 0; dy < 3; ++dy)
                            for (int dz = 0; dz < 3; ++dz) {
                                const int nb = (nbc_cx[dx] * nc + nbc_cy[dy]) * nc + nbc_cz[dz];
                                const int nbeg = cl.cell_start[nb], nend = cl.cell_start[nb + 1];
                                n_src += nend - nbeg;
                            }

                    r_src_g.resize(3 * n_src);
                    charge_g.resize(n_src);
                    Real *__restrict__ r_src_ptr = r_src_g.data();
                    Real *__restrict__ charge_ptr = charge_g.data();
                    int r_i = 0, c_i = 0;
                    for (int dx = 0; dx < 3; ++dx) {
                        const Real off_x = off_cx[dx];
                        const int nbc_x = nbc_cx[dx];
                        for (int dy = 0; dy < 3; ++dy) {
                            const Real off_y = off_cy[dy];
                            const int nbc_y = nbc_cy[dy];
                            for (int dz = 0; dz < 3; ++dz) {
                                const Real off_z = off_cz[dz];
                                const int nbc_z = nbc_cz[dz];
                                const int nb = (nbc_x * nc + nbc_y) * nc + nbc_z;
                                const int nbeg = cl.cell_start[nb], nend = cl.cell_start[nb + 1];
                                for (int b = nbeg; b < nend; ++b) {
                                    r_src_ptr[r_i++] = cl.rs[b * 3 + 0] + off_x;
                                    r_src_ptr[r_i++] = cl.rs[b * 3 + 1] + off_y;
                                    r_src_ptr[r_i++] = cl.rs[b * 3 + 2] + off_z;
                                    charge_ptr[c_i++] = cl.qs[b];
                                }
                            }
                        }
                    }

                    evaluator(rsc, Real(0), r_c_sq, Real(0), n_src, r_src_ptr, charge_ptr, nullptr, n_trg, r_trg_ptr,
                              pot_sorted.data() + hbeg);
                }
    }

    // restore original particle ordering
    std::vector<Real> pot(n, Real(0));
    for (int a = 0; a < n; ++a)
        pot[cl.orig[a]] = pot_sorted[a];
    return pot;
}

// ---------------------------------------------------------------------------
// Short-range sum (fast path: cell-list / box-reordering, no neighbor list).
// Parallelizes over home cells — each cell's output slots are disjoint across
// threads, so pot_sorted[a] is written by exactly one thread (no atomics).
// Falls back to the neighbor-list path when r_c > L/3 (nc < 3).
// ---------------------------------------------------------------------------
template <typename Real>
static std::vector<Real> short_range_slow(const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges,
                                          const PSWFKernel &pswf, const ESPParams &params) {
    // 27-cell stencil requires nc >= 3 so periodic images aren't double-counted
    const int nc = static_cast<int>(std::floor(params.L / params.r_c));
    if (nc < 3)
        throw std::runtime_error("short_range_fast requires r_c <= L/3 (nc >= 3)");

    CellList<Real> cl = build_cell_list<Real>(r_src, charges, params, nc);

    const Real L = Real(params.L);
    const Real r_c_sq = Real(params.r_c) * Real(params.r_c);
    const double inv_rc = 1.0 / params.r_c;
    const double inv_c0 = 1.0 / params.c0;
    const int n = params.n;

    std::vector<Real> pot_sorted(n, Real(0));

    // For each cell coordinate c in [0, nc) and each delta d in {-1,0,+1} (stored as d=0,1,2):
    // nbc_tab[c*3+d] = neighbor cell index, off_tab[c*3+d] = image shift to subtract.
    // Precomputed once; all three dimensions share the same table since they share nc.
    std::vector<int>  nbc_tab(nc * 3);
    std::vector<Real> off_tab(nc * 3);
    for (int c = 0; c < nc; ++c) {
        for (int d = 0; d < 3; ++d) {
            int ci = c + d - 1;
            if      (ci < 0)   { nbc_tab[c*3+d] = ci + nc; off_tab[c*3+d] = -L; }
            else if (ci >= nc) { nbc_tab[c*3+d] = ci - nc; off_tab[c*3+d] =  L; }
            else               { nbc_tab[c*3+d] = ci;      off_tab[c*3+d] =  Real(0); }
        }
    }

#pragma omp parallel for schedule(dynamic) collapse(3)
    for (int cx = 0; cx < nc; ++cx)
        for (int cy = 0; cy < nc; ++cy)
            for (int cz = 0; cz < nc; ++cz) {
                const int home = (cx * nc + cy) * nc + cz;
                const int hbeg = cl.cell_start[home], hend = cl.cell_start[home + 1];

                const int  *nbc_cx = &nbc_tab[cx * 3];
                const int  *nbc_cy = &nbc_tab[cy * 3];
                const int  *nbc_cz = &nbc_tab[cz * 3];
                const Real *off_cx = &off_tab[cx * 3];
                const Real *off_cy = &off_tab[cy * 3];
                const Real *off_cz = &off_tab[cz * 3];

                for (int a = hbeg; a < hend; ++a) {
                    const Real xi = cl.rs[3*a+0], yi = cl.rs[3*a+1], zi = cl.rs[3*a+2];
                    Real acc = Real(0);

                    for (int dx = 0; dx < 3; ++dx)
                        for (int dy = 0; dy < 3; ++dy)
                            for (int dz = 0; dz < 3; ++dz) {
                                const int nb = (nbc_cx[dx] * nc + nbc_cy[dy]) * nc + nbc_cz[dz];
                                const int nbeg = cl.cell_start[nb], nend = cl.cell_start[nb + 1];

                                for (int b = nbeg; b < nend; ++b) {
                                    if (nb == home && a == b)
                                        continue;
                                    const Real ddx = xi - cl.rs[3*b+0] - off_cx[dx];
                                    const Real ddy = yi - cl.rs[3*b+1] - off_cy[dy];
                                    const Real ddz = zi - cl.rs[3*b+2] - off_cz[dz];
                                    const Real d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                                    if (d2 > r_c_sq)
                                        continue;

                                    double dist = std::sqrt(double(d2));
                                    double t = dist * inv_rc;
                                    double intval = pswf.integral(0.0, t) * inv_c0;

                                    acc += cl.qs[b] * Real((1.0 - intval) / (4.0 * M_PI * dist));
                                }
                            }
                    pot_sorted[a] = acc;
                }
            }

    // restore original particle ordering
    std::vector<Real> pot(n, Real(0));
    for (int a = 0; a < n; ++a)
        pot[cl.orig[a]] = pot_sorted[a];
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
static std::vector<Real> long_range(const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges,
                                    const PSWFKernel &pswf, const ESPParams &params, const DGrid &scaling_coeffs) {
    int n = params.n;
    int nf = params.n_f;
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
    opts.upsampfac = pswf.sigma;
    double tol = pswf.eps;
    //tol = 1e-9;

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
static std::vector<Real> self_interaction(const std::vector<Real> &charges, const PSWFKernel &pswf,
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
    int n_digits; // tolerance bucket selecting the precompiled short-range evaluator
    PSWFKernel pswf;
    ESPParams params_base; // n=0 placeholder; n_f/h/kernel params pre-filled
    DGrid scaling_coeffs;
    // Always built from the actual pswf.sigma so the short-range polynomial
    // matches the long-range FINUFFT window exactly.
    residual_evaluator_func<float>  custom_eval_f;
    residual_evaluator_func<double> custom_eval_d;

    // eps is snapped to 10^-n_digits so the short-range PSWF window (baked into the
    // precompiled evaluator at that bucket) matches the long-range window exactly.
    // sigma < 0 → use esp_sigma_from_eps; sigma > 0 → override (builds on-the-fly evaluator).
    EspPlan(double L, double r_c, double eps, double sigma = -1.0)
        : n_digits(esp_digits_from_eps(eps)), pswf(std::pow(10.0, -double(n_digits)), sigma),
          params_base(L, r_c, pswf.P, pswf, 0) {
        scaling_coeffs = precompute_scaling_coefficients(pswf, params_base);
        auto cd = get_esp_correction_coeffs<double>(n_digits, pswf.sigma)[0];
        auto cf = get_esp_correction_coeffs<float> (n_digits, pswf.sigma)[0];
        custom_eval_d = make_esp_scalar_evaluator<double>(std::move(cd));
        custom_eval_f = make_esp_scalar_evaluator<float> (std::move(cf));
    }
};

EspPlan *esp_create_plan(double L, double r_c, double eps, double sigma) { return new EspPlan(L, r_c, eps, sigma); }

void esp_destroy_plan(EspPlan *plan) { delete plan; }

template <typename Real>
std::vector<Real> esp_eval(EspPlan *plan, const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges,
                           EspTimings *timings) {
    int n = static_cast<int>(charges.size());
    ESPParams params = plan->params_base;
    params.n = n;

    residual_evaluator_func<Real> sr_eval;
    if constexpr (std::is_same_v<Real, double>)
        sr_eval = plan->custom_eval_d;
    else
        sr_eval = plan->custom_eval_f;

    double t0 = omp_get_wtime();
    auto pot_sr = short_range_slow<Real>(r_src, charges, plan->pswf, params); //before: params, sr_eval
    //auto pot_sr = short_range_fast<Real>(r_src, charges, params, sr_eval);
    double t1 = omp_get_wtime();
    auto pot_lr = long_range<Real>(r_src, charges, plan->pswf, params, plan->scaling_coeffs);
    double t2 = omp_get_wtime();
    auto pot_self = self_interaction<Real>(charges, plan->pswf, params);
    double t3 = omp_get_wtime();

    if (timings) {
        timings->t_short = t1 - t0;
        timings->t_long = t2 - t1;
        timings->t_self = t3 - t2;
    }

    std::vector<Real> total(n);
    for (int i = 0; i < n; ++i)
        total[i] = pot_sr[i] + pot_lr[i] - pot_self[i];
    return total;
}

// ---------------------------------------------------------------------------
// Convenience one-shot entry point (create + eval + destroy).
// ---------------------------------------------------------------------------
template <typename Real>
std::vector<Real> esp_potential(const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges, double L,
                                double r_c, double eps) {
    auto *plan = esp_create_plan(L, r_c, eps);
    auto result = esp_eval<Real>(plan, r_src, charges);
    esp_destroy_plan(plan);
    return result;
}

template std::vector<float> esp_eval<float>(EspPlan *, const std::vector<Vec3T<float>> &, const std::vector<float> &,
                                            EspTimings *);
template std::vector<double> esp_eval<double>(EspPlan *, const std::vector<Vec3T<double>> &,
                                              const std::vector<double> &, EspTimings *);
template std::vector<float> esp_potential<float>(const std::vector<Vec3T<float>> &, const std::vector<float> &, double,
                                                 double, double);
template std::vector<double> esp_potential<double>(const std::vector<Vec3T<double>> &, const std::vector<double> &,
                                                   double, double, double);

} // namespace dmk
