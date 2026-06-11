"""Zeroth prolate spheroidal wave function psi_0^c and utilities.

References
----------
[1] Slepian, D. and Pollak, H. O. (1961). "Prolate spheroidal wave functions,
    Fourier analysis and uncertainty - I." Bell System Technical Journal,
    40(1):43-63.  -- Original PSWF theory; gives |lambda_n|^2 = 2*pi*mu_n/c
    between the Fourier (F_c) and sinc-kernel eigenvalues.

[2] Jiang, S. and Greengard, L. (2025). "A dual-space multilevel kernel-
    splitting framework for discrete and continuous convolution."
    Comm. Pure Appl. Math., 78(5):1086-1143.  -- Defines F_c (eq. A21) and
    states the self-Fourier-transform identity psi_hat_0^c(k) = lambda_0 *
    psi_0^c(k/c) on |k| <= c (eq. A25).

[3] Liang, J., Lu, L., Barnett, A., Greengard, L., Jiang, S. (2025).
    "Accelerating Fast Ewald Summation with Prolates for Molecular Dynamics
    Simulations." arXiv:2505.09727.  -- Uses psi_0^c as the Ewald splitting
    kernel; gives the c <-> epsilon prescription psi_0^c(1) = epsilon and
    the truncation rule n_f = ceil(c*L/(pi*r_c)) (eq. 14).
"""

import copy

import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.integrate import fixed_quad, quad
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq


class Prolate0:
    """Zeroth prolate spheroidal wave function psi_0^c on R.

    Built from the Legendre expansion on [-1, 1]; extended outside via the
    sinc integral operator. Returns 0 for |x| > 1 when c >= OUTSIDE_THRESH
    (where psi_0^c is numerically zero in double precision).
    """

    OUTSIDE_THRESH = 45.0

    @classmethod
    def from_eps(cls, eps, bracket=(1.0, 60.0), n_terms=None):
        """Construct so that psi_0^c(1) = eps. Shape parameter is available as `.c`."""
        c = brentq(lambda c: cls(c, n_terms=n_terms)(1) - eps, *bracket)
        return cls(c, n_terms=n_terms)

    def __init__(self, c, n_terms=None):
        self.c = float(c)
        n_terms = n_terms or max(50, int(2 * self.c) + 30)

        n = 2 * np.arange((n_terms + 1) // 2)
        diag = n*(n+1) + c**2 * (2*n*(n+1) - 1) / ((2*n - 1) * (2*n + 3))
        m = n[:-1]
        off = c**2 * (m+1)*(m+2) / ((2*m+3) * np.sqrt((2*m+1)*(2*m+5)))

        evals, evecs = eigh_tridiagonal(diag, off)
        self.chi = float(evals[0])
        beta = evecs[:, 0]

        coeffs = np.zeros(int(n.max()) + 1)
        coeffs[n] = beta * np.sqrt(n + 0.5)
        if coeffs[0] < 0:
            coeffs = -coeffs
        self._poly = Legendre(coeffs)
        self._dpoly = self._poly.deriv()
        self._ipoly = self._poly.integ()
        self._lam = self._compute_eigenvalue()
        self.normalized = False

    @property
    def eigenvalue(self):
        return self._lam

    def __repr__(self):
        norm = ", normalized" if self.normalized else ""
        return (f"Prolate0(c={self.c:.4f}, lambda={self._lam:.3e}, "
                f"psi(0)={self._poly(0.0):.4g}, psi(1)={self._poly(1.0):.3e}{norm})")

    def __call__(self, x):
        return self._eval(x, deriv=False)

    def derivative(self, x):
        return self._eval(x, deriv=True)

    def normalize(self):
        """Return a copy scaled so psi(0) = 1 (range [0, 1] on [-1, 1])."""
        other = copy.copy(self)
        #s = 1.0 / self.integral(0.0, 1.0)
        s = 1.0 / self._poly(0.0)  # alternative normalization so psi(0) = 1
        other._poly = self._poly * s
        other._dpoly = self._dpoly * s
        other._ipoly = self._ipoly * s
        other.normalized = True
        return other

    def integral(self, a=0.0, b=1.0):
        """Integral of psi_0^c from a to b."""
        if abs(a) <= 1.0 and abs(b) <= 1.0:
            return float(self._ipoly(b) - self._ipoly(a))
        val, _ = quad(self, a, b)
        return val

    def _eval(self, x, deriv):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.empty_like(x)

        inside = np.abs(x) <= 1.0
        local = self._dpoly if deriv else self._poly
        out[inside] = local(x[inside])

        outside = ~inside
        if outside.any():
            if self.c >= self.OUTSIDE_THRESH:
                out[outside] = 0.0
            else:
                for i in np.flatnonzero(outside):
                    out[i] = self._sinc_eval(float(x[i]), deriv)

        return out.item() if scalar else out

    def _sinc_eval(self, x, deriv):
        # Continuation outside [-1, 1] via the sinc-kernel eigenequation:
        #   psi(x) = (1/(pi*mu)) * int_{-1}^{1} sin(c(x-t))/(x-t) psi(t) dt
        # i.e.  psi(x) = (c/mu) * int sinc_pi(c(x-t)/pi) psi(t) dt
        # (np.sinc uses the normalized convention sinc_pi(z) = sin(pi z)/(pi z).)
        c = self.c
        if deriv:
            def k(t):
                d = x - t
                return (c * np.cos(c*d) * d - np.sin(c*d)) / d**2 * self._poly(t)
            val, _ = fixed_quad(k, -1.0, 1.0, n=128)
            return val / (np.pi * self._lam)
        else:
            def k(t):
                return c * np.sinc(c*(x - t) / np.pi) * self._poly(t)
            val, _ = fixed_quad(k, -1.0, 1.0, n=128)
            return val / (np.pi * self._lam)

    def _compute_eigenvalue(self):
        # Sinc-kernel eigenvalue mu:  Q_c[psi](0) = mu * psi(0),
        # with Q_c[psi](x) = int_{-1}^{1} sin(c(x-t))/(pi(x-t)) psi(t) dt.
        # Use np.sinc so t=0 needs no special case.
        c = self.c
        def k(t):
            return c * np.sinc(c*t / np.pi) * self._poly(t)
        n = max(64, 2 * self._poly.degree())
        val, _ = fixed_quad(k, -1.0, 1.0, n=n)
        return (val / np.pi) / self._poly(0.0)


def _demo_self_ft(eps=1e-6, L=32.0, r_c=1.0):
    """Verify Jiang & Greengard (2025) eq. (A25):
        psi_hat_0^c(k) = lambda_0 * psi_0^c(k / c)
    where lambda_0 is the eigenvalue of the F_c operator in eq. (A21).
    By Slepian, lambda_0 = sqrt(2*pi*mu/c) with mu the sinc-kernel
    eigenvalue (what Prolate0._lam stores). FFT length is set per
    Liang et al. (2025) eq. (14): n_f = ceil(c*L/(pi*r_c))."""
    f = Prolate0.from_eps(eps)
    c, mu = f.c, f.eigenvalue
    # A21 eigenvalue. lambda_0 != 1 even when the sinc eigenvalue mu -> 1
    # (perfect band-limiting): F_c maps [-1,1] -> [-c,c], so Parseval picks
    # up a 2*pi/c rescaling. Slepian: F_c F_c^* = (2*pi/c) * (sinc op),
    # hence |lambda_0|^2 = 2*pi*mu/c.
    lambda0 = np.sqrt(2 * np.pi * mu / c)
    n_f = int(np.ceil(c * L / (np.pi * r_c)))
    print(f)
    print(f"L={L}, r_c={r_c}, eq.(14) n_f = ceil(cL/(pi r_c)) = {n_f}")
    print(f"sinc eigenvalue mu={mu:.6f}, F_c eigenvalue lambda_0={lambda0:.6f}")

    # Symmetric grid covering one period: t = j * dt for j = -n_f//2 ... n_f//2 - 1.
    # psi is sampled naturally on [-r_c, r_c]; outside is zero. ifftshift +
    # fft + fftshift is the "centered DFT" idiom -- it routes the t=0 sample
    # (at index n_f//2) to the FFT's expected origin (index 0), and the
    # k=0 frequency back to the center of the output.
    dt = L / n_f
    j = np.arange(n_f) - n_f // 2
    t = j * dt
    psi_t = f(t)

    psi_hat = dt * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi_t)))
    k = (2 * np.pi / L) * j

    # A25: psi_hat(k) = lambda_0 * psi(k/c)  on |k| <= c
    band = np.abs(k) <= c
    err = psi_hat[band].real - lambda0 * f(k[band] / c)
    print(f"samples in band: {band.sum()}")
    print(f"FFT  sup |error|:  {np.max(np.abs(err)):.2e}")
    print(f"FFT  sup |imag|:   {np.max(np.abs(psi_hat[band].imag)):.2e}")

    # Equivalence check: explicit O(n^2) DFT on the same grid.
    #   psi_hat_dft[m] = dt * sum_j psi(t_j) * exp(-i * k_m * t_j)
    # Must match the FFT (centered via the ifftshift/fftshift pair) to roundoff.
    psi_hat_dft = np.zeros(n_f, dtype=complex)
    for m in range(n_f):
        acc = 0.0 + 0.0j
        for jj in range(n_f):
            acc += psi_t[jj] * np.exp(-1j * k[m] * t[jj])
        psi_hat_dft[m] = dt * acc
    print(f"DFT  vs FFT:       {np.max(np.abs(psi_hat_dft - psi_hat)):.2e}")

    # Independent check: direct quadrature of the continuous Fourier integral
    #   psi_hat(k) = int_{-1}^{1} psi(t) e^{-i k t} dt = 2 * int_0^1 psi(t) cos(k t) dt
    # (psi is even and supported in [-1, 1] to machine precision for c >= OUTSIDE_THRESH).
    # This bypasses the FFT grid entirely, so agreement with A25 here is a
    # quadrature-level proof rather than a discrete-sampling artifact.
    k_band = k[band]
    psi_hat_direct = np.empty_like(k_band)
    for i, ki in enumerate(k_band):
        val, _ = fixed_quad(lambda t: f(t) * np.cos(ki * t), 0.0, 1.0, n=256)
        psi_hat_direct[i] = 2.0 * val
    err_direct = psi_hat_direct - lambda0 * f(k_band / c)
    err_vs_fft = psi_hat_direct - psi_hat[band].real
    print(f"quad sup |error|:  {np.max(np.abs(err_direct)):.2e}")
    print(f"quad vs FFT:       {np.max(np.abs(err_vs_fft)):.2e}")


def _demo_compare_gaussian(L=32.0, r_c=1.0):
    """Compare n_f from eq.(14) (PSWF) vs eq.(11) (Gaussian) at matched accuracy.
    Eq.(11): alpha_G = log(1/eps),  n_f^G   = 2 * ceil(log(1/eps) * L / (pi r_c))
    Eq.(14): c per psi_0^c(1)=eps,  n_f^P   =     ceil(c          * L / (pi r_c))
    """
    print(f"\nn_f comparison at L={L}, r_c={r_c}:")
    print(f"{'eps':>8} {'c_pswf':>9} {'a_gauss':>9} "
          f"{'n_f^pswf':>10} {'n_f^gauss':>11} {'ratio (1D)':>11} {'ratio (3D)':>11}")
    for eps in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
        c = Prolate0.from_eps(eps).c
        a_g = np.log(1.0 / eps)
        n_p = int(np.ceil(c   * L / (np.pi * r_c)))
        n_g = 2 * int(np.ceil(a_g * L / (np.pi * r_c)))
        r1 = n_g / n_p
        print(f"{eps:>8.0e} {c:>9.4f} {a_g:>9.4f} "
              f"{n_p:>10d} {n_g:>11d} {r1:>11.2f} {r1**3:>11.2f}")


if __name__ == "__main__":
    _demo_self_ft()
    _demo_compare_gaussian()
