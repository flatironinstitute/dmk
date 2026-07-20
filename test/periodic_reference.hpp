#ifndef DMK_PERIODIC_REFERENCE_HPP
#define DMK_PERIODIC_REFERENCE_HPP

// Ground-truth periodic references for validating the DMK/ESP solvers. Shared by
// test_pbc_periodic.cpp and the benchmarks. Header-only; the only DMK dependency is
// get_direct_evaluator (Yukawa image sum), so consumers must link the dmk library.

#include <cmath>
#include <complex>
#include <vector>

#include <dmk.h>
#include <dmk/direct.hpp>

namespace dmk::pbc_ref {

// Relative L2 error.
inline double safe_l2(double e2, double r2) { return r2 > 0 ? std::sqrt(e2 / r2) : std::sqrt(e2); }

// Exponential integral E1(x) = \int_x^inf exp(-t)/t dt, x > 0 (Numerical Recipes expint(1, .)).
inline double expint_E1(double x) {
    const double EULER = 0.5772156649015328606;
    if (x < 1.0) {
        double ans = -std::log(x) - EULER, fact = 1.0;
        for (int i = 1; i <= 100; ++i) {
            fact *= -x / i;
            const double del = -fact / i;
            ans += del;
            if (std::abs(del) < std::abs(ans) * 1e-16)
                break;
        }
        return ans;
    }
    double b = x + 1.0, c = 1e300, dd = 1.0 / b, h = dd;
    for (int i = 1; i <= 100; ++i) {
        const double an = -1.0 * i * i;
        b += 2.0;
        dd = 1.0 / (an * dd + b);
        c = b + an / c;
        const double del = c * dd;
        h *= del;
        if (std::abs(del - 1.0) < 1e-16)
            break;
    }
    return h * std::exp(-x);
}

// Absolutely-convergent kernels (Yukawa 2D/3D): DMK's free-space evaluator summed over
// (2*n_img+1)^n_dim image shifts m*L. The evaluator masks the r==0 pair, so a source
// evaluated against the unshifted sources drops only its self term while nonzero shifts
// add its periodic images -- matching the periodic solver. Writes n_eval*odim into ref
// (odim = 1 for DMK_POTENTIAL, 1+n_dim for DMK_POTENTIAL_GRAD).
inline void image_sum(int n_dim, double lambda, int n_img, dmk_eval_type eval, int n_src, const double *r_src,
                      const double *charges, double L, int n_eval, const double *r_eval, std::vector<double> &ref) {
    const int odim = (eval == DMK_POTENTIAL_GRAD) ? 1 + n_dim : 1;
    ref.assign(size_t(n_eval) * odim, 0.0);
    auto eval_fn = dmk::get_direct_evaluator<double>(DMK_YUKAWA, eval, n_dim, lambda);
    std::vector<double> src_shift(size_t(n_dim) * n_src);
    for (int mx = -n_img; mx <= n_img; ++mx)
        for (int my = -n_img; my <= n_img; ++my)
            for (int mz = (n_dim == 3 ? -n_img : 0); mz <= (n_dim == 3 ? n_img : 0); ++mz) {
                for (int is = 0; is < n_src; ++is) {
                    src_shift[is * n_dim + 0] = r_src[is * n_dim + 0] + mx * L;
                    src_shift[is * n_dim + 1] = r_src[is * n_dim + 1] + my * L;
                    if (n_dim == 3)
                        src_shift[is * n_dim + 2] = r_src[is * n_dim + 2] + mz * L;
                }
                eval_fn(n_src, src_shift.data(), charges, nullptr, n_eval, r_eval, ref.data());
            }
}

// Ewald-split reference for the conditionally/slowly convergent scalar kernels, selected
// by (kernel, n_dim):
//   Laplace 3D (1/r):        S(r)=erfc(a r)/r,        G(k)=exp(-k^2/4a^2)/k^2 * 4 pi/V,  self=2a/sqrt(pi)
//   Sqrt-Laplace 3D (1/r^2): S(r)=exp(-a^2 r^2)/r^2,  G(k)=erfc(k/2a)/k     * 2 pi^2/V,  self=a^2
//   Sqrt-Laplace 2D (1/r):   S(r)=erfc(a r)/r,        G(k)=erfc(k/2a)/k     * 2 pi/V,    self=2a/sqrt(pi)
//   Laplace 2D (log):        S(r)=-1/2 E1(a^2 r^2),   G(k)=-exp(-k^2/4a^2)/k^2 * 2 pi/V, self=0 (gauge-fit)
// (a is the split parameter, alpha/eta in the literature.) The real-space sum uses a
// periodic cell list; the reciprocal sum uses a precomputed structure factor rho.
struct EwaldRef {
    dmk_ikernel kernel;
    int n_dim, n_src, n_ewald, n_cells;
    double L, split, r_cut, cell_size, prefactor, self_const;
    const double *r_src;
    const double *charges;
    std::vector<std::complex<double>> rho; // (2*n_ewald+1)^n_dim structure factor
    std::vector<std::vector<int>> cells;   // source indices per cell (row-major over n_cells^n_dim)

    // split == 0 selects the per-kernel default that reproduces the test references.
    EwaldRef(dmk_ikernel kernel_, int n_dim_, int n_src_, const double *r_src_, const double *charges_, double L_,
             double split_ = 0.0)
        : kernel(kernel_), n_dim(n_dim_), n_src(n_src_), L(L_), r_src(r_src_), charges(charges_) {
        const bool is_laplace3d = kernel == DMK_LAPLACE && n_dim == 3;
        split = split_ > 0 ? split_ : (is_laplace3d ? 10.0 / L : 6.0 / L);

        // Short-range and reciprocal both decay like exp(-(split*r or k/2split)^2); ~1e-12 accuracy
        // needs split*r_cut ~ 5.25 and k_max/(2 split) ~ 5.25.
        r_cut = 5.25 / split;
        n_ewald = int(std::ceil(1.7 * split * L)) + 2;
        n_cells = std::max(1, int(L / r_cut));
        cell_size = L / n_cells;

        const double V = std::pow(L, n_dim);
        if (kernel == DMK_LAPLACE && n_dim == 3) {
            prefactor = 4.0 * M_PI / V;
            self_const = 2.0 * split / std::sqrt(M_PI);
        } else if (kernel == DMK_SQRT_LAPLACE && n_dim == 3) {
            prefactor = 2.0 * M_PI * M_PI / V;
            self_const = split * split;
        } else if (kernel == DMK_SQRT_LAPLACE && n_dim == 2) {
            prefactor = 2.0 * M_PI / V;
            self_const = 2.0 * split / std::sqrt(M_PI);
        } else { // DMK_LAPLACE && n_dim == 2 (log)
            prefactor = 2.0 * M_PI / V;
            self_const = 0.0;
        }

        build_rho();
        build_cells();
    }

    double self_factor() const { return self_const; }

    void eval(const double *r_eval, int self_idx, double &pot, double *grad) const {
        pot = 0.0;
        if (grad)
            for (int d = 0; d < n_dim; ++d)
                grad[d] = 0.0;
        real_space(r_eval, pot, grad);
        reciprocal(r_eval, pot, grad);
        if (self_idx >= 0)
            pot -= charges[self_idx] * self_const;
    }

  private:
    static double wrap01(double x, double box) { return x - box * std::floor(x / box); }

    int cell_index(const double *r) const {
        int idx = 0;
        for (int d = 0; d < n_dim; ++d) {
            int c = int(wrap01(r[d], L) / cell_size);
            if (c >= n_cells)
                c = n_cells - 1;
            idx = idx * n_cells + c;
        }
        return idx;
    }

    void build_cells() {
        cells.assign(size_t(std::pow(n_cells, n_dim) + 0.5), {});
        for (int is = 0; is < n_src; ++is)
            cells[cell_index(&r_src[is * n_dim])].push_back(is);
    }

    void build_rho() {
        const int d = 2 * n_ewald + 1;
        const size_t M = size_t(std::pow(d, n_dim) + 0.5);
        rho.assign(M, {0.0, 0.0});
        const double dk = 2.0 * M_PI / L;
#pragma omp parallel
        {
            std::vector<std::complex<double>> local(M, {0.0, 0.0});
            std::vector<std::complex<double>> e[3];
            for (int a = 0; a < n_dim; ++a)
                e[a].resize(d);
#pragma omp for
            for (int is = 0; is < n_src; ++is) {
                for (int a = 0; a < n_dim; ++a) {
                    const std::complex<double> e0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * n_dim + a]));
                    for (int n = -n_ewald; n <= n_ewald; ++n)
                        e[a][n + n_ewald] = std::pow(e0, n);
                }
                if (n_dim == 3) {
                    for (int ix = 0; ix < d; ++ix)
                        for (int iy = 0; iy < d; ++iy) {
                            const auto t2 = charges[is] * e[0][ix] * e[1][iy];
                            for (int iz = 0; iz < d; ++iz)
                                local[(ix * d + iy) * d + iz] += t2 * e[2][iz];
                        }
                } else {
                    for (int ix = 0; ix < d; ++ix)
                        for (int iy = 0; iy < d; ++iy)
                            local[ix * d + iy] += charges[is] * e[0][ix] * e[1][iy];
                }
            }
#pragma omp critical
            for (size_t m = 0; m < M; ++m)
                rho[m] += local[m];
        }
    }

    // Short-range S(r2): potential term (per unit charge) and gradient coefficient g such that
    // grad += charge * g * r_vec.
    void short_range(double r2, double &s, double &g) const {
        const double a2 = split * split;
        if (kernel == DMK_LAPLACE && n_dim == 3) {
            const double r = std::sqrt(r2);
            s = std::erfc(split * r) / r;
            g = -(std::erfc(split * r) / (r * r2) + 2.0 * split * std::exp(-a2 * r2) / (std::sqrt(M_PI) * r2));
        } else if (kernel == DMK_SQRT_LAPLACE && n_dim == 3) {
            const double ex = std::exp(-a2 * r2);
            s = ex / r2;
            g = -2.0 * ex * (a2 * r2 + 1.0) / (r2 * r2);
        } else if (kernel == DMK_SQRT_LAPLACE && n_dim == 2) {
            const double r = std::sqrt(r2);
            s = std::erfc(split * r) / r;
            g = -(std::erfc(split * r) / (r * r2) + 2.0 * split * std::exp(-a2 * r2) / (std::sqrt(M_PI) * r2));
        } else { // DMK_LAPLACE && n_dim == 2 (log)
            s = -0.5 * expint_E1(a2 * r2);
            g = std::exp(-a2 * r2) / r2;
        }
    }

    // Reciprocal Green's function G(k) (without prefactor).
    double recip_G(double k2, double kmag) const {
        if (kernel == DMK_LAPLACE && n_dim == 3)
            return std::exp(-k2 / (4.0 * split * split)) / k2;
        if (kernel == DMK_SQRT_LAPLACE) // 2D and 3D
            return std::erfc(kmag / (2.0 * split)) / kmag;
        return -std::exp(-k2 / (4.0 * split * split)) / k2; // log
    }

    void real_space(const double *r_eval, double &pot, double *grad) const {
        constexpr double thresh2 = 1e-28;
        const double r_cut2 = r_cut * r_cut;
        int ec[3] = {0, 0, 0};
        double ew[3] = {0, 0, 0};
        for (int d = 0; d < n_dim; ++d) {
            ew[d] = wrap01(r_eval[d], L);
            ec[d] = std::min(int(ew[d] / cell_size), n_cells - 1);
        }
        const int oz_lo = n_dim == 3 ? -1 : 0, oz_hi = n_dim == 3 ? 1 : 0;
        for (int ox = -1; ox <= 1; ++ox)
            for (int oy = -1; oy <= 1; ++oy)
                for (int oz = oz_lo; oz <= oz_hi; ++oz) {
                    const int o[3] = {ox, oy, oz};
                    int cidx = 0;
                    double shift[3] = {0, 0, 0};
                    for (int d = 0; d < n_dim; ++d) {
                        const int raw = ec[d] + o[d];
                        const int wrapped = ((raw % n_cells) + n_cells) % n_cells;
                        shift[d] = raw < 0 ? -L : (raw >= n_cells ? L : 0.0);
                        cidx = cidx * n_cells + wrapped;
                    }
                    for (int is : cells[cidx]) {
                        double disp[3];
                        double r2 = 0.0;
                        for (int d = 0; d < n_dim; ++d) {
                            disp[d] = ew[d] - (wrap01(r_src[is * n_dim + d], L) + shift[d]);
                            r2 += disp[d] * disp[d];
                        }
                        if (r2 < thresh2 || r2 > r_cut2)
                            continue;
                        double s, g;
                        short_range(r2, s, g);
                        pot += charges[is] * s;
                        if (grad)
                            for (int d = 0; d < n_dim; ++d)
                                grad[d] += charges[is] * g * disp[d];
                    }
                }
    }

    void reciprocal(const double *r_eval, double &pot, double *grad) const {
        const int d = 2 * n_ewald + 1;
        const double dk = 2.0 * M_PI / L;
        std::vector<std::complex<double>> et[3];
        for (int a = 0; a < n_dim; ++a) {
            et[a].resize(d);
            const std::complex<double> e0 = std::exp(std::complex<double>(0.0, dk * r_eval[a]));
            for (int n = -n_ewald; n <= n_ewald; ++n)
                et[a][n + n_ewald] = std::pow(e0, n);
        }
        double pot_long = 0.0, grad_long[3] = {0, 0, 0};
        for (int nx = -n_ewald; nx <= n_ewald; ++nx)
            for (int ny = -n_ewald; ny <= n_ewald; ++ny) {
                const int nz_lo = n_dim == 3 ? -n_ewald : 0, nz_hi = n_dim == 3 ? n_ewald : 0;
                for (int nz = nz_lo; nz <= nz_hi; ++nz) {
                    if (nx == 0 && ny == 0 && nz == 0)
                        continue;
                    const double kx = dk * nx, ky = dk * ny, kz = dk * nz;
                    const double k2 = kx * kx + ky * ky + kz * kz;
                    const double G = recip_G(k2, std::sqrt(k2));
                    std::complex<double> eikr = et[0][nx + n_ewald] * et[1][ny + n_ewald];
                    size_t ridx = size_t(nx + n_ewald) * d + (ny + n_ewald);
                    if (n_dim == 3) {
                        eikr *= et[2][nz + n_ewald];
                        ridx = ridx * d + (nz + n_ewald);
                    }
                    const auto rho_eikr = rho[ridx] * eikr;
                    pot_long += G * std::real(rho_eikr);
                    if (grad) {
                        const double im = -std::imag(rho_eikr);
                        grad_long[0] += G * kx * im;
                        grad_long[1] += G * ky * im;
                        grad_long[2] += G * kz * im;
                    }
                }
            }
        pot += prefactor * pot_long;
        if (grad)
            for (int d2 = 0; d2 < n_dim; ++d2)
                grad[d2] += prefactor * grad_long[d2];
    }
};

} // namespace dmk::pbc_ref

#endif // DMK_PERIODIC_REFERENCE_HPP
