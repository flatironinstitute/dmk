// Original direct port to C++ from the Fortran code by Libin Lu and github copilot.
// Templates and further features/redesigns added by Robert Blackwell.

#include <dmk/legeexps.hpp>

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

namespace dmk {
template <typename Real>
void legepol(Real x, int n, Real &pol, Real &der) {
    Real pkm1 = 1.0;
    Real pk = x;
    Real pkp1;

    if (n == 0) {
        pol = 1.0;
        der = 0.0;
        return;
    }

    if (n == 1) {
        pol = x;
        der = 1.0;
        return;
    }

    pk = 1.0;
    pkp1 = x;

    for (int k = 1; k < n; ++k) {
        pkm1 = pk;
        pk = pkp1;
        pkp1 = ((2 * k + 1) * x * pk - k * pkm1) / (k + 1);
    }

    pol = pkp1;
    der = n * (x * pkp1 - pk) / (x * x - 1);
}

template <typename Real>
void legetayl(Real pol, Real der, Real x, Real h, int n, int k, Real &sum, Real &sumder) {
    Real done = 1.0;
    Real q0 = pol;
    Real q1 = der * h;
    Real q2 = (2 * x * der - n * (n + done) * pol) / (1 - x * x);
    q2 = q2 * h * h / 2;

    sum = q0 + q1 + q2;
    sumder = q1 / h + q2 * 2 / h;

    if (k <= 2)
        return;

    Real qi = q1;
    Real qip1 = q2;

    for (int i = 1; i <= k - 2; ++i) {
        Real d = 2 * x * (i + 1) * (i + 1) / h * qip1 - (n * (n + done) - i * (i + 1)) * qi;
        d = d / (i + 1) / (i + 2) * h * h / (1 - x * x);
        Real qip2 = d;

        sum += qip2;
        sumder += d * (i + 2) / h;

        qi = qip1;
        qip1 = qip2;
    }
}

/* Constructs Gaussian quadrature of order n.
   itype=1 => both roots (ts) and weights (whts) are computed.
   itype=0 => only roots (ts) are computed. */
template <typename Real>
void legerts(int itype, int n, Real *ts, Real *whts) {
    int k = 30;
    Real d = 1.0;
    Real d2 = d + 1.0e-24;
    if (d2 != d) {
        k = 54;
    }

    int half = n / 2;
    int ifodd = n - 2 * half;
    Real pi_val = atan(1.0) * 4.0;
    Real h = pi_val / (2.0 * n);

    /* Initial approximations (for i >= n/2+1) */
    int ii = 0;
    for (int i = 1; i <= n; i++) {
        if (i < (n / 2 + 1)) {
            continue;
        }
        ii++;
        Real t = (2.0 * i - 1.0) * h;
        ts[ii - 1] = -cos(t);
    }

    /* Start from center: find roots one by one via Newton updates */
    Real pol = 1.0, der = 0.0;
    Real x0 = 0.0;
    legepol(x0, n, pol, der);
    Real x1 = ts[0];

    int n2 = (n + 1) / 2;
    Real pol3 = pol, der3 = der;

    for (int kk = 1; kk <= n2; kk++) {
        if ((ifodd == 1) && (kk == 1)) {
            ts[kk - 1] = x0;
            if (itype > 0) {
                whts[kk - 1] = der;
            }
            x0 = x1;
            x1 = ts[kk];
            pol3 = pol;
            der3 = der;
            continue;
        }

        /* Newton iteration */
        int ifstop = 0;
        for (int i = 1; i <= 10; i++) {
            Real hh = x1 - x0;

            legetayl(pol3, der3, x0, hh, n, k, pol, der);
            x1 = x1 - pol / der;

            if (fabs(pol) < 1.0e-12) {
                ifstop++;
            }
            if (ifstop == 3) {
                break;
            }
        }

        ts[kk - 1] = x1;
        if (itype > 0) {
            whts[kk - 1] = der;
        }

        x0 = x1;
        x1 = ts[kk];
        pol3 = pol;
        der3 = der;
    }

    /* Mirror roots around 0: fill second half of ts[] */
    for (int i = n2; i >= 1; i--) {
        ts[i - 1 + half] = ts[i - 1];
    }
    for (int i = 1; i <= half; i++) {
        ts[i - 1] = -ts[n - i];
    }
    if (itype <= 0) {
        return;
    }

    /* Mirror weights similarly */
    for (int i = n2; i >= 1; i--) {
        whts[i - 1 + half] = whts[i - 1];
    }
    for (int i = 1; i <= half; i++) {
        whts[i - 1] = whts[n - i];
    }

    /* Compute final weights = 2 / (1 - ts[i]^2) / (der[i]^2) */
    for (int i = 0; i < n; i++) {
        Real tmp = 1.0 - ts[i] * ts[i];
        whts[i] = 2.0 / tmp / (whts[i] * whts[i]);
    }
}

template <typename Real>
void legepols(Real x, int n, Real *pols) {
    Real pkm1 = 1.0;
    Real pk = x;

    if (n == 0) {
        pols[0] = 1.0;
        return;
    }

    if (n == 1) {
        pols[0] = 1.0;
        pols[1] = x;
        return;
    }

    pols[0] = 1.0;
    pols[1] = x;

    for (int k = 1; k < n; ++k) {
        Real pkp1 = ((2 * k + 1) * x * pk - k * pkm1) / (k + 1);
        pols[k + 1] = pkp1;
        pkm1 = pk;
        pk = pkp1;
    }
}

// TODO: legepols() is not tested yet.
// only itype !=2 is tested.
template <typename Real>
void legeexps(int itype, int n, Real *x, std::vector<std::vector<Real>> &u, std::vector<std::vector<Real>> &v,
              Real *whts) {
    int itype_rts = (itype > 0) ? 1 : 0;

    // Call legerts to construct the nodes and weights of the n-point Gaussian quadrature
    legerts(itype_rts, n, x, whts);
    // gaussian_quadrature(n, x, whts);

    // legerts() is buggy now, below is not tested yet.
    // legerts_(&itype_rts, &n, x, whts);

    // If itype is not 2, return early
    if (itype != 2)
        return;

    // Construct the matrix of values of the Legendre polynomials at these nodes
    for (int i = 0; i < n; ++i) {
        std::vector<Real> pols(n);
        legepols(x[i], n - 1, pols.data());
        for (int j = 0; j < n; ++j) {
            u[j][i] = pols[j];
        }
    }

    // Transpose u to get v
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            v[i][j] = u[j][i];
        }
    }

    // Construct the inverse u, converting the values of a function at Gaussian nodes into the
    // coefficients of a Legendre expansion of that function
    for (int i = 0; i < n; ++i) {
        Real d = 1.0 * (2 * (i + 1) - 1) / 2;
        for (int j = 0; j < n; ++j) {
            u[i][j] = v[j][i] * whts[j] * d;
        }
    }
}

template <typename Real>
void legeexev(Real x, Real &val, const Real *pexp, int n) {
    Real done = 1.0;
    Real pjm2 = 1.0;
    Real pjm1 = x;

    val = pexp[0] * pjm2 + pexp[1] * pjm1;

    for (int j = 2; j <= n; ++j) {
        Real pj = ((2 * j - 1) * x * pjm1 - (j - 1) * pjm2) / j;
        val += pexp[j] * pj;
        pjm2 = pjm1;
        pjm1 = pj;
    }
}

template <typename Real>
void legeFDER(Real x, Real &val, Real &der, const Real *pexp, int n) {
    Real done = 1.0;
    Real pjm2 = 1.0;
    Real pjm1 = x;
    Real derjm2 = 0.0;
    Real derjm1 = 1.0;

    val = pexp[0] * pjm2 + pexp[1] * pjm1;
    der = pexp[1];

    for (int j = 2; j <= n; ++j) {
        Real pj = ((2 * j - 1) * x * pjm1 - (j - 1) * pjm2) / j;
        val += pexp[j] * pj;

        Real derj = (2 * j - 1) * (pjm1 + x * derjm1) - (j - 1) * derjm2;
        derj /= j;
        der += pexp[j] * derj;

        pjm2 = pjm1;
        pjm1 = pj;
        derjm2 = derjm1;
        derjm1 = derj;
    }
}

// Function to compute the Legendre polynomial p_n(x) and its derivative p'_n(x)
template <typename Real>
static inline void legendre(int n, Real x, Real &pn, Real &pn_prime) {
    if (n == 0) {
        pn = 1.0;
        pn_prime = 0.0;
        return;
    }

    if (n == 1) {
        pn = x;
        pn_prime = 1.0;
        return;
    }

    Real pn_minus1 = 1.0; // P_0(x)
    Real pn_minus2 = 0.0; // P_-1(x)
    pn = x;               // P_1(x)

    for (int k = 2; k <= n; ++k) {
        pn_minus2 = pn_minus1;
        pn_minus1 = pn;
        pn = ((2.0 * k - 1.0) * x * pn_minus1 - (k - 1.0) * pn_minus2) / k;
    }

    pn_prime = n * (x * pn - pn_minus1) / (x * x - 1.0);
}

// Function to compute the nodes and weights of the n-point Gaussian quadrature
template <typename Real>
void gaussian_quadrature(int n, Real *nodes, Real *weights) {
    static std::vector<std::vector<Real>> precomp_nodes(1000);
    static std::vector<std::vector<Real>> precomp_weights(1000);

    { // Precompute
        assert(n < precomp_nodes.size() && n < precomp_weights.size());
        if (precomp_nodes[n].size() == 0 && precomp_weights[n].size() == 0) {
#pragma omp critical(GAUSS_QUAD)
            if (precomp_nodes[n].size() == 0 && precomp_weights[n].size() == 0) {
                std::vector<Real> nodes1(n);
                std::vector<Real> weights1(n);

                if (true) { // usr original code
                    legerts(1, n, nodes1.data(), weights1.data());
                } else { // self implementation test code
                    const Real tolerance = 1e-16;
                    const int max_iterations = 100;
                    for (int i = 0; i < (n + 1) / 2; ++i) {
                        Real x = std::cos(M_PI * (i + 0.75) / (n + 0.5));
                        Real pn, pn_prime;

                        for (int iter = 0; iter < max_iterations; ++iter) {
                            legendre(n, x, pn, pn_prime);
                            Real delta_x = -pn / pn_prime;
                            x += delta_x;

                            if (std::abs(delta_x) < tolerance) {
                                break;
                            }
                        }

                        nodes1[i] = x;
                        nodes1[n - 1 - i] = -x;
                        weights1[i] = 2.0 / ((1.0 - x * x) * pn_prime * pn_prime);
                        weights1[n - 1 - i] = weights1[i];
                    }
                }

                precomp_nodes[n].swap(nodes1);
                precomp_weights[n].swap(weights1);
            }
        }
    }

    memcpy(nodes, precomp_nodes[n].data(), n * sizeof(Real));
    memcpy(weights, precomp_weights[n].data(), n * sizeof(Real));
}

template void legeexps<float>(int itype, int n, float *x, std::vector<std::vector<float>> &u,
                              std::vector<std::vector<float>> &v, float *whts);
template void legeexps<double>(int itype, int n, double *x, std::vector<std::vector<double>> &u,
                               std::vector<std::vector<double>> &v, double *whts);
template void legeexev<float>(float x, float &val, const float *pexp, int n);
template void legeexev<double>(double x, double &val, const double *pexp, int n);
template void legeFDER<float>(float x, float &val, float &der, const float *pexp, int n);
template void legeFDER<double>(double x, double &val, double &der, const double *pexp, int n);

} // namespace dmk
