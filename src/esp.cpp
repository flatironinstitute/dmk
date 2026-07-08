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
#include <iostream>
#include <omp.h>
#include <sctl.hpp>
#include <stdexcept>
#include <vector>
#include <dmk/direct.hpp>
#include <dmk/types.hpp>
#include <dmk/util.hpp>

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
struct PSWFKernel {
    dmk::Prolate0Fun pswf;
    double eps;
    double c;
    double lambda0;
    double c0;
    double scale;
    std::vector<double> pswf_poly_coeffs;
    std::vector<double> pswf_int_poly_coeffs;

    explicit PSWFKernel(double eps_, double c_, int lenw = 8000) : eps(eps_), c(c_) {
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

    double operator()(double x) const { return pswf.eval_val(x) * scale; }

    double integral_eval(double t) const { return pswf.int_eval(t) * scale; }

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

struct ESPParams {
    double L;
    double r_c;
    double sigma;
    int P;
    int n_f;
    double h;
    double lambda0;
    double c;
    double c0;
    int n;

    ESPParams(double L_, double r_c_, double sigma_, const PSWFKernel &pswf, int n_)
        : L(L_), r_c(r_c_), sigma(sigma_), n(n_) {
        P = esp_ns_from_eps(pswf.eps, sigma);
        n_f = static_cast<int>(std::ceil(pswf.c * L / (M_PI * r_c)));
        h = L / n_f;
        lambda0 = pswf.lambda0;
        c = pswf.c;
        c0 = pswf.c0;
    }
};

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

// Cell list: particles sorted into cubic cells for cache-friendly traversal.
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
        cl.rs[3 * slot + 0] = r_src[j][0];
        cl.rs[3 * slot + 1] = r_src[j][1];
        cl.rs[3 * slot + 2] = r_src[j][2];
        cl.qs[slot] = charges[j];
        cl.orig[slot] = j;
    }
    return cl;
}

// Short-range sum 
template <typename Real>
static PotForce<Real> short_range_fast(const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges,
                                       const ESPParams &params, int n_digits, const PSWFKernel &pswf) {
    // 27-cell stencil requires nc >= 3 so periodic images aren't double-counted
    const int nc = static_cast<int>(std::floor(params.L / params.r_c));
    if (nc < 3)
        throw std::runtime_error("short_range_fast requires r_c <= L/3 (nc >= 3)");

    CellList<Real> cl = build_cell_list<Real>(r_src, charges, params, nc);

    const Real L = Real(params.L);
    const Real r_c_sq = Real(params.r_c) * Real(params.r_c);
    const int n = params.n;

    constexpr int MaxVecLen = sctl::DefaultVecLen<Real>();
    auto evaluator = get_esp_3d_kernel<Real, MaxVecLen>(DMK_POTENTIAL_GRAD, n_digits);
#ifdef DMK_USE_JIT
    if (!util::env_is_set("DMK_DEBUG_FORCE_AOT"))
        evaluator = make_esp_evaluator_jit<Real>(DMK_POTENTIAL_GRAD, n_digits, params.sigma, 3);
#endif
    const Real rsc = Real(2.0 / params.r_c);
    const Real cen = Real(-params.r_c / 2.0);

    // Interleaved [pot, d/dx, d/dy, d/dz] per particle, in cell-sorted order.
    std::vector<Real> pg_sorted(4 * n, Real(0));

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
                    evaluator(rsc, cen, r_c_sq, Real(0), n_src, r_src_ptr, charge_ptr, nullptr, n_trg, r_trg_ptr,
                              pg_sorted.data() + 4 * hbeg);
                }
    }
    PotForce<Real> out;
    out.pot.assign(n, Real(0));
    out.force_x.assign(n, Real(0));
    out.force_y.assign(n, Real(0));
    out.force_z.assign(n, Real(0));
    for (int a = 0; a < n; ++a) {
        const int orig = cl.orig[a];
        const Real q = cl.qs[a];
        out.pot[orig] = pg_sorted[4 * a + 0];
        out.force_x[orig] = -q * pg_sorted[4 * a + 1];
        out.force_y[orig] = -q * pg_sorted[4 * a + 2];
        out.force_z[orig] = -q * pg_sorted[4 * a + 3];
    }
    return out;
}

static inline double S_hat(const PSWFKernel &pswf, const ESPParams &params, const Vec3 &k_vec) {
    double k_mag = std::sqrt(k_vec[0] * k_vec[0] + k_vec[1] * k_vec[1] + k_vec[2] * k_vec[2]);
    return pswf.pswf_hat(k_mag * params.r_c) / (2.0 * k_mag * k_mag) / params.c0;
}

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

// Long-range contribution via FINUFFT spreading/interpolation
template <typename Real>
static PotForce<Real> long_range(const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges,
                                 const PSWFKernel &pswf, const ESPParams &params, const DGrid &scaling_coeffs) {
    int n = params.n;
    int nf = params.n_f;
    int ntot = nf * nf * nf;

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
    opts.upsampfac = params.sigma;
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
    CGrid pot_hat(ntot);
#pragma omp parallel for
    for (int idx = 0; idx < ntot; ++idx)
        pot_hat[idx] = b_hat[idx] * scaling_coeffs[idx];

    std::vector<int> k_idx(nf);
    for (int i = 0; i < nf; ++i)
        k_idx[i] = (i <= nf / 2) ? i : i - nf;

    // ik method: F = -q*grad(u), and grad(u)_hat_k = i*k*u_hat_k, so the force spectrum is
    // obtained from the potential spectrum (b_hat * scaling_coeffs, i.e. pot_hat) by multiplying
    // by -i*k component-wise. 
    std::complex<double> coeff_grad = std::complex<double>(0, 1) * (2.0 * M_PI / params.L);
    CGrid f_hat_x(ntot), f_hat_y(ntot), f_hat_z(ntot);
#pragma omp parallel for collapse(3)
    for (int ix = 0; ix < nf; ++ix)
        for (int iy = 0; iy < nf; ++iy)
            for (int iz = 0; iz < nf; ++iz) {
                const int idx = grid_idx(ix, iy, iz, nf);
                const std::complex<double> scaled = b_hat[idx] * scaling_coeffs[idx];
                // DMK's grid_idx (row-major, z fastest) and FINUFFT's internal grid storage
                // (column-major, x fastest) disagree on which loop variable is which physical axis; 
                // ix here lines up with FINUFFT's z slot and iz lines up with its x slot 
                f_hat_x[idx] = scaled * coeff_grad * double(k_idx[iz]);
                f_hat_y[idx] = scaled * coeff_grad * double(k_idx[iy]);
                f_hat_z[idx] = scaled * coeff_grad * double(k_idx[ix]);
            }

    // 4. Inverse FFT
    CGrid grid_pot(ntot);
    ifftn_3d(pot_hat, grid_pot, nf);

    CGrid grid_force_x(ntot), grid_force_y(ntot), grid_force_z(ntot);
    ifftn_3d(f_hat_x, grid_force_x, nf);
    ifftn_3d(f_hat_y, grid_force_y, nf);
    ifftn_3d(f_hat_z, grid_force_z, nf);

    // 5. Interpolate: uniform grid -> NU points (type 2)
    std::vector<std::complex<double>> pot_c(n);
    ier = finufft3d2(n, x.data(), y.data(), z.data(), pot_c.data(), +1, tol, nf, nf, nf, grid_pot.data(), &opts);
    if (ier > 1)
        throw std::runtime_error("finufft3d2 interp failed, ier=" + std::to_string(ier));

    std::vector<std::complex<double>> force_x_c(n), force_y_c(n), force_z_c(n);
    ier =
        finufft3d2(n, x.data(), y.data(), z.data(), force_x_c.data(), +1, tol, nf, nf, nf, grid_force_x.data(), &opts);
    if (ier > 1)
        throw std::runtime_error("finufft3d2 interp failed, ier=" + std::to_string(ier));
    ier =
        finufft3d2(n, x.data(), y.data(), z.data(), force_y_c.data(), +1, tol, nf, nf, nf, grid_force_y.data(), &opts);
    if (ier > 1)
        throw std::runtime_error("finufft3d2 interp failed, ier=" + std::to_string(ier));
    ier =
        finufft3d2(n, x.data(), y.data(), z.data(), force_z_c.data(), +1, tol, nf, nf, nf, grid_force_z.data(), &opts);
    if (ier > 1)
        throw std::runtime_error("finufft3d2 interp failed, ier=" + std::to_string(ier));

    PotForce<Real> out;
    out.pot.resize(n);
    out.force_x.resize(n);
    out.force_y.resize(n);
    out.force_z.resize(n);
    for (int j = 0; j < n; j++) {
        out.pot[j] = Real(pot_c[j].real());
        out.force_x[j] = -charges[j] * Real(force_x_c[j].real());
        out.force_y[j] = -charges[j] * Real(force_y_c[j].real());
        out.force_z[j] = -charges[j] * Real(force_z_c[j].real());
    }
    return out;
}

// Self-interaction correction
template <typename Real>
static std::vector<Real> self_interaction(const std::vector<Real> &charges, const PSWFKernel &pswf,
                                          const ESPParams &params) {
    std::vector<Real> self(params.n);
    Real factor = Real(pswf(0.0) / (params.r_c * 4.0 * M_PI * params.c0));
    for (int i = 0; i < params.n; ++i)
        self[i] = charges[i] * factor;
    return self;
}

struct EspPlan {
    int n_digits; // tolerance bucket selecting the precompiled short-range evaluator
    PSWFKernel pswf;
    ESPParams params_base; 
    DGrid scaling_coeffs;

    EspPlan(double L, double r_c, double eps, double sigma)
        : n_digits(esp_digits_from_eps(eps)),
          pswf(std::pow(10.0, -double(n_digits)), esp_pswf_c_from_eps(std::pow(10.0, -double(n_digits)), sigma)),
          params_base(L, r_c, sigma, pswf, 0) {
        scaling_coeffs = precompute_scaling_coefficients(pswf, params_base);
    }
};

EspPlan *esp_create_plan(double L, double r_c, double eps, double sigma) { return new EspPlan(L, r_c, eps, sigma); }

void esp_destroy_plan(EspPlan *plan) { delete plan; }

template <typename Real>
PotForce<Real> esp_eval(EspPlan *plan, const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges,
                        EspTimings *timings) {
    int n = static_cast<int>(charges.size());
    ESPParams params = plan->params_base;
    params.n = n;

    PotForce<Real> total;
    total.pot.resize(n);
    total.force_x.resize(n);
    total.force_y.resize(n);
    total.force_z.resize(n);

    double t0_start = omp_get_wtime();
    auto aux = short_range_fast<Real>(r_src, charges, params, plan->n_digits, plan->pswf);
    double t0_end = omp_get_wtime();

    for (int i = 0; i < n; ++i) {
        total.pot[i] = aux.pot[i];
        total.force_x[i] = aux.force_x[i];
        total.force_y[i] = aux.force_y[i];
        total.force_z[i] = aux.force_z[i];
    }

    double t1_start = omp_get_wtime();
    aux = long_range<Real>(r_src, charges, plan->pswf, params, plan->scaling_coeffs);
    double t1_end = omp_get_wtime();

    for(int i = 0; i < n; ++i) {
        total.pot[i] += aux.pot[i];
        total.force_x[i] += aux.force_x[i];
        total.force_y[i] += aux.force_y[i];
        total.force_z[i] += aux.force_z[i];
    }

    double t2_start = omp_get_wtime();
    auto pot_self = self_interaction<Real>(charges, plan->pswf, params);
    double t2_end = omp_get_wtime();

    for(int i = 0; i < n; ++i) {
        total.pot[i] -= pot_self[i];
    }

    if (timings) {
        timings->t_short = t0_end - t0_start;
        timings->t_long = t1_end - t1_start;
        timings->t_self = t2_end - t2_start;
    }
    
    return total;
}

template <typename Real>
PotForce<Real> esp_potential(const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges, double L,
                             double r_c, double eps) {
    // FIXME: Magic number for sigma
    auto *plan = esp_create_plan(L, r_c, eps, 1.35);
    auto result = esp_eval<Real>(plan, r_src, charges);
    esp_destroy_plan(plan);
    return result;
}

template PotForce<float> esp_eval<float>(EspPlan *, const std::vector<Vec3T<float>> &, const std::vector<float> &,
                                         EspTimings *);
template PotForce<double> esp_eval<double>(EspPlan *, const std::vector<Vec3T<double>> &, const std::vector<double> &,
                                           EspTimings *);
template PotForce<float> esp_potential<float>(const std::vector<Vec3T<float>> &, const std::vector<float> &, double,
                                              double, double);
template PotForce<double> esp_potential<double>(const std::vector<Vec3T<double>> &, const std::vector<double> &, double,
                                                double, double);

} // namespace dmk
