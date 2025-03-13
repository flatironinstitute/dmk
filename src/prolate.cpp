#include <dmk/legeexps.hpp>
#include <dmk/prolate.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>


namespace dmk {
void prolc180(double eps, double &c) {
    static const std::array<double, 180> cs = {
        0.43368E-16, 0.10048E+01, 0.17298E+01, 0.22271E+01, 0.26382E+01, 0.30035E+01, 0.33409E+01, 0.36598E+01,
        0.39658E+01, 0.42621E+01, 0.45513E+01, 0.48347E+01, 0.51136E+01, 0.53887E+01, 0.56606E+01, 0.59299E+01,
        0.61968E+01, 0.64616E+01, 0.67247E+01, 0.69862E+01, 0.72462E+01, 0.75049E+01, 0.77625E+01, 0.80189E+01,
        0.82744E+01, 0.85289E+01, 0.87826E+01, 0.90355E+01, 0.92877E+01, 0.95392E+01, 0.97900E+01, 0.10040E+02,
        0.10290E+02, 0.10539E+02, 0.10788E+02, 0.11036E+02, 0.11284E+02, 0.11531E+02, 0.11778E+02, 0.12024E+02,
        0.12270E+02, 0.12516E+02, 0.12762E+02, 0.13007E+02, 0.13251E+02, 0.13496E+02, 0.13740E+02, 0.13984E+02,
        0.14228E+02, 0.14471E+02, 0.14714E+02, 0.14957E+02, 0.15200E+02, 0.15443E+02, 0.15685E+02, 0.15927E+02,
        0.16169E+02, 0.16411E+02, 0.16652E+02, 0.16894E+02, 0.17135E+02, 0.17376E+02, 0.17617E+02, 0.17858E+02,
        0.18098E+02, 0.18339E+02, 0.18579E+02, 0.18819E+02, 0.19059E+02, 0.19299E+02, 0.19539E+02, 0.19778E+02,
        0.20018E+02, 0.20257E+02, 0.20496E+02, 0.20736E+02, 0.20975E+02, 0.21214E+02, 0.21452E+02, 0.21691E+02,
        0.21930E+02, 0.22168E+02, 0.22407E+02, 0.22645E+02, 0.22884E+02, 0.23122E+02, 0.23360E+02, 0.23598E+02,
        0.23836E+02, 0.24074E+02, 0.24311E+02, 0.24549E+02, 0.24787E+02, 0.25024E+02, 0.25262E+02, 0.25499E+02,
        0.25737E+02, 0.25974E+02, 0.26211E+02, 0.26448E+02, 0.26685E+02, 0.26922E+02, 0.27159E+02, 0.27396E+02,
        0.27633E+02, 0.27870E+02, 0.28106E+02, 0.28343E+02, 0.28580E+02, 0.28816E+02, 0.29053E+02, 0.29289E+02,
        0.29526E+02, 0.29762E+02, 0.29998E+02, 0.30234E+02, 0.30471E+02, 0.30707E+02, 0.30943E+02, 0.31179E+02,
        0.31415E+02, 0.31651E+02, 0.31887E+02, 0.32123E+02, 0.32358E+02, 0.32594E+02, 0.32830E+02, 0.33066E+02,
        0.33301E+02, 0.33537E+02, 0.33773E+02, 0.34008E+02, 0.34244E+02, 0.34479E+02, 0.34714E+02, 0.34950E+02,
        0.35185E+02, 0.35421E+02, 0.35656E+02, 0.35891E+02, 0.36126E+02, 0.36362E+02, 0.36597E+02, 0.36832E+02,
        0.37067E+02, 0.37302E+02, 0.37537E+02, 0.37772E+02, 0.38007E+02, 0.38242E+02, 0.38477E+02, 0.38712E+02,
        0.38947E+02, 0.39181E+02, 0.39416E+02, 0.39651E+02, 0.39886E+02, 0.40120E+02, 0.40355E+02, 0.40590E+02,
        0.40824E+02, 0.41059E+02, 0.41294E+02, 0.41528E+02, 0.41763E+02, 0.41997E+02, 0.42232E+02, 0.42466E+02,
        0.42700E+02, 0.42935E+02, 0.43169E+02, 0.43404E+02, 0.43638E+02, 0.43872E+02, 0.44107E+02, 0.44341E+02,
        0.44575E+02, 0.44809E+02, 0.45044E+02, 0.45278E+02};

    double d = -log10(eps);
    int i = static_cast<int>(d * 10 + 0.1);
    c = cs[i - 1];
}

void prosinin(double c, const double *ts, const double *whts, const double *fs, double x, int n, double &rint,
              double &derrint) {
    rint = 0.0;
    derrint = 0.0;

    for (int i = 0; i < n; ++i) {
        double diff = x - ts[i];
        double sin_term = sin(c * diff);
        double cos_term = cos(c * diff);

        rint += whts[i] * fs[i] * sin_term / diff;

        derrint += whts[i] * fs[i] / (diff * diff) * (c * diff * cos_term - sin_term);
    }
}

void prolps0i(int &ier, double c, double *w, int lenw, int &nterms, int &ltot, double &rkhi) {
    static const std::array<int, 20> ns = {48,  64,  80,  92,  106, 120, 130, 144, 156, 168,
                                           178, 190, 202, 214, 224, 236, 248, 258, 268, 280};

    double eps = 1.0e-16;
    int n = static_cast<int>(c * 3);
    n = n / 2;

    int i = static_cast<int>(c / 10);
    if (i <= 19)
        n = ns[i];

    ier = 0;
    int ixk = 1;
    int lxk = n + 2;

    int ias = ixk + lxk;
    int las = n + 2;

    int ibs = ias + las;
    int lbs = n + 2;

    int ics = ibs + lbs;
    int lcs = n + 2;

    int iu = ics + lcs;
    int lu = n + 2;

    int iv = iu + lu;
    int lv = n + 2;

    int iw = iv + lv;
    int lw = n + 2;

    ltot = iw + lw;

    if (ltot >= lenw) {
        ier = 512;
        return;
    }

    // Call to prolfun0 (to be implemented)
    prolfun0(ier, n, c, w + ias - 1, w + ibs - 1, w + ics - 1, w + ixk - 1, w + iu - 1, w + iv - 1, w + iw - 1, eps,
             nterms, rkhi);

    if (ier != 0)
        return;
}

void prolfun0(int &ier, int n, double c, double *as, double *bs, double *cs, double *xk, double *u, double *v,
              double *w, double eps, int &nterms, double &rkhi) {
    ier = 0;
    double delta = 1.0e-8;
    int ifsymm = 1;
    int numit = 4;
    double rlam = 0;
    int ifodd = -1;

    prolmatr(as, bs, cs, n, c, rlam, ifsymm, ifodd);

    prolql1(n / 2, bs, as, ier);
    if (ier != 0) {
        ier = 2048;
        return;
    }

    rkhi = -bs[n / 2 - 1];
    rlam = -bs[n / 2 - 1] + delta;

    std::fill(xk, xk + n, 1.0);

    prolmatr(as, bs, cs, n, c, rlam, ifsymm, ifodd);

    prolfact(bs, cs, as, n / 2, u, v, w);

    for (int ijk = 0; ijk < numit; ++ijk) {
        prolsolv(u, v, w, n / 2, xk);

        double d = 0;
        for (int j = 0; j < n / 2; ++j) {
            d += xk[j] * xk[j];
        }

        d = std::sqrt(d);
        for (int j = 0; j < n / 2; ++j) {
            xk[j] /= d;
        }

        double err = 0;
        for (int j = 0; j < n / 2; ++j) {
            err += (as[j] - xk[j]) * xk[j];
            as[j] = xk[j];
        }
        err = std::sqrt(err);
    }

    double half = 0.5;
    for (int i = 0; i < n / 2; ++i) {
        if (std::abs(xk[i]) > eps)
            nterms = i + 1;
        xk[i] *= std::sqrt(i * 2 + half);
        cs[i] = xk[i];
    }

    int j = 0;
    for (int i = 0; i <= nterms; ++i) {
        xk[j++] = cs[i];
        xk[j++] = 0;
    }

    nterms *= 2;
}

void prolcoef(double rlam, int k, double c, double &alpha0, double &beta0, double &gamma0, double &alpha, double &beta,
              double &gamma) {
    double d = k * (k - 1);
    d = d / (2 * k + 1) / (2 * k - 1);
    double uk = d;

    d = (k + 1) * (k + 1);
    d = d / (2 * k + 3);
    double d2 = k * k;
    d2 = d2 / (2 * k - 1);
    double vk = (d + d2) / (2 * k + 1);

    d = (k + 1) * (k + 2);
    d = d / (2 * k + 1) / (2 * k + 3);
    double wk = d;

    alpha = -c * c * uk;
    beta = rlam - k * (k + 1) - c * c * vk;
    gamma = -c * c * wk;

    alpha0 = uk;
    beta0 = vk;
    gamma0 = wk;
}

void prolmatr(double *as, double *bs, double *cs, int n, double c, double rlam, int ifsymm, int ifodd) {
    double done = 1.0;
    double half = done / 2.0;
    int k = 0;

    if (ifodd > 0) {
        for (int k0 = 1; k0 <= n + 2; k0 += 2) {
            k++;
            double alpha0, beta0, gamma0, alpha, beta, gamma;
            prolcoef(rlam, k0, c, alpha0, beta0, gamma0, alpha, beta, gamma);

            as[k - 1] = alpha;
            bs[k - 1] = beta;
            cs[k - 1] = gamma;

            if (ifsymm != 0) {
                if (k0 > 1) {
                    as[k - 1] = as[k - 1] / std::sqrt(k0 - 2 + half) * std::sqrt(k0 + half);
                }
                cs[k - 1] = cs[k - 1] * std::sqrt(k0 + half) / std::sqrt(k0 + half + 2);
            }
        }
    } else {
        for (int k0 = 0; k0 <= n + 2; k0 += 2) {
            k++;
            double alpha0, beta0, gamma0, alpha, beta, gamma;
            prolcoef(rlam, k0, c, alpha0, beta0, gamma0, alpha, beta, gamma);

            as[k - 1] = alpha;
            bs[k - 1] = beta;
            cs[k - 1] = gamma;

            if (ifsymm != 0) {
                if (k0 != 0) {
                    as[k - 1] = as[k - 1] / std::sqrt(k0 - 2 + half) * std::sqrt(k0 + half);
                }
                cs[k - 1] = cs[k - 1] * std::sqrt(k0 + half) / std::sqrt(k0 + half + 2);
            }
        }
    }
}

void prolql1(int n, double *d, double *e, int &ierr) {
    ierr = 0;
    if (n == 1)
        return;

    for (int i = 1; i < n; ++i) {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    for (int l = 0; l < n; ++l) {
        int j = 0;
        while (true) {
            int m;
            for (m = l; m < n - 1; ++m) {
                double tst1 = std::abs(d[m]) + std::abs(d[m + 1]);
                double tst2 = tst1 + std::abs(e[m]);
                if (tst2 == tst1)
                    break;
            }

            if (m == l)
                break;
            if (j == 30) {
                ierr = l + 1;
                return;
            }
            ++j;

            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = std::sqrt(g * g + 1.0);
            g = d[m] - d[l] + e[l] / (g + std::copysign(r, g));
            double s = 1.0;
            double c = 1.0;
            double p = 0.0;

            for (int i = m - 1; i >= l; --i) {
                double f = s * e[i];
                double b = c * e[i];
                r = std::sqrt(f * f + g * g);
                e[i + 1] = r;
                if (r == 0.0) {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }

            if (r == 0.0)
                break;
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }

        if (l == 0)
            continue;
        for (int i = l; i > 0; --i) {
            if (d[i] >= d[i - 1])
                break;
            std::swap(d[i], d[i - 1]);
        }
    }
}

void prolfact(double *a, double *b, double *c, int n, double *u, double *v, double *w) {
    // Eliminate down
    for (int i = 0; i < n - 1; ++i) {
        double d = c[i + 1] / a[i];
        a[i + 1] -= b[i] * d;
        u[i] = d;
    }

    // Eliminate up
    for (int i = n - 1; i > 0; --i) {
        double d = b[i - 1] / a[i];
        v[i] = d;
    }

    // Scale the diagonal
    double done = 1.0;
    for (int i = 0; i < n; ++i) {
        w[i] = done / a[i];
    }
}

void prolsolv(const double *u, const double *v, const double *w, int n, double *rhs) {
    // Eliminate down
    for (int i = 0; i < n - 1; ++i) {
        rhs[i + 1] -= u[i] * rhs[i];
    }

    // Eliminate up
    for (int i = n - 1; i > 0; --i) {
        rhs[i - 1] -= rhs[i] * v[i];
    }

    // Scale
    for (int i = 0; i < n; ++i) {
        rhs[i] *= w[i];
    }
}

void prol0ini(int &ier, double c, double *w, double &rlam20, double &rkhi, int lenw, int &keep, int &ltot) {
    ier = 0;
    double thresh = 45;
    int iw = 11;
    w[0] = iw + 0.1;
    w[8] = thresh;

    // Create the data to be used in the evaluation of the function ψ^c_0(x) for x ∈ [-1,1]
    int nterms;
    prolps0i(ier, c, w + iw - 1, lenw, nterms, ltot, rkhi);

    if (ier != 0)
        return;

    // If c > thresh, do not prepare data for the evaluation of ψ^c_0 outside the interval [-1,1]
    if (c >= thresh) {
        w[7] = c;
        w[4] = nterms + 0.1;
        keep = nterms + 3;
        return;
    }

    // Create the data to be used in the evaluation of the function ψ^c_0(x) for x outside the
    // interval [-1,1]
    int ngauss = nterms * 2;
    int lw = nterms + 2;
    int its = iw + lw;
    int lts = ngauss + 2;
    int iwhts = its + lts;
    int lwhts = ngauss + 2;
    int ifs = iwhts + lwhts;
    int lfs = ngauss + 2;

    keep = ifs + lfs;
    if (keep > ltot)
        ltot = keep;
    if (keep >= lenw) {
        ier = 1024;
        return;
    }

    w[1] = its + 0.1;
    w[2] = iwhts + 0.1;
    w[3] = ifs + 0.1;

    int itype = 1;
    std::vector<std::vector<double>> u;
    std::vector<std::vector<double>> v;
    legeexps(itype, ngauss, w + its - 1, u, v, w + iwhts - 1);

    // Evaluate the prolate function at the Gaussian nodes
    for (int i = 0; i < ngauss; ++i) {
        legeexev(w[its + i - 2], w[ifs + i - 2], w + iw - 1, nterms - 1);
    }

    // Calculate the eigenvalue corresponding to ψ^c_0
    double rlam = 0;
    double x0 = 0;
    double f0;
    legeexev(x0, f0, w + iw - 1, nterms - 1);
    double der;
    prosinin(c, w + its - 1, w + iwhts - 1, w + ifs - 1, x0, ngauss, rlam, der);

    rlam = rlam / f0;
    rlam20 = rlam;

    w[4] = nterms + 0.1;
    w[5] = ngauss + 0.1;
    w[6] = rlam;
    w[7] = c;
}

void prol0eva(double x, const double *w, double &psi0, double &derpsi0) {
    int iw = static_cast<int>(w[0]);
    int its = static_cast<int>(w[1]);
    int iwhts = static_cast<int>(w[2]);
    int ifs = static_cast<int>(w[3]);

    int nterms = static_cast<int>(w[4]);
    int ngauss = static_cast<int>(w[5]);
    double rlam = w[6];
    double c = w[7];
    double thresh = w[8];

    if (std::abs(x) > 1) {
        if (c < thresh - 1.0e-10) {
            psi0 = 0;
            derpsi0 = 0;
            return;
        }

        prosinin(c, &w[its - 1], &w[iwhts - 1], &w[ifs - 1], x, ngauss, psi0, derpsi0);
        psi0 /= rlam;
        derpsi0 /= rlam;
        return;
    }

    legeFDER(x, psi0, derpsi0, &w[iw - 1], nterms - 2);
    // to match chebfun psi0, needs a factor of sqrt(2)
    // psi0 = sqrt(2.0) * psi0;
    // derpsi0 = sqrt(2.0) * derpsi0;
}

void prol0int0r(const double *w, double r, double &val) {
    static int npts = 200;
    static int itype = 1;
    double derpsi0;
    static std::vector<double> xs(npts, 0), ws(npts, 0), fvals(npts, 0);
    static int need_init = 1;
    std::vector<std::vector<double>> u;
    std::vector<std::vector<double>> v;

    // since xs, ws, fval of size 200 are static
    // only need to get nodes and weights once
    if (need_init) {
#pragma omp critical(PROL0INT0R)
        if (need_init) {
            legeexps(itype, npts, xs.data(), u, v, ws.data());
            need_init = 0;
        }
    }

    // Scale the nodes and weights to [0, r]
    double xs_r;
    for (int i = 0; i < npts; ++i) {
        xs_r = (xs[i] + 1) * r / 2;
        prol0eva(xs_r, w, fvals[i], derpsi0);
    }

    val = 0;
    for (int i = 0; i < npts; ++i) {
        val += ws[i] * r / 2 * fvals[i];
    }
}

void prolate_intvals(double beta, double *wprolate, double &c0, double &c1, double &c2, double &c4) {
    const int n_pts = 200;
    std::array<double, n_pts> xs, ws, fvals;

    const int itype = 1;
    legerts(itype, n_pts, xs.data(), ws.data());

    // scale the nodes and weights to [0,1]
    for (int i = 0; i < n_pts; ++i) {
        xs[i] = 0.5 * (xs[i] + 1);
        ws[i] = 0.5 * ws[i];
    }

    for (int i = 0; i < n_pts; ++i) {
        double derpsi0;
        prol0eva(xs[i], wprolate, fvals[i], derpsi0);
    }

    c0 = c1 = c2 = c4 = 0;
    for (int i = 0; i < n_pts; ++i) {
        const double xs2 = xs[i] * xs[i];
        const double xs4 = xs2 * xs2;
        c0 += ws[i] * fvals[i];
        c2 += ws[i] * fvals[i] * xs2;
        c4 += ws[i] * fvals[i] * xs4;
        c1 += ws[i] * fvals[i] * xs[i];
    }
    c2 /= c0;
}

} // namespace dmk
