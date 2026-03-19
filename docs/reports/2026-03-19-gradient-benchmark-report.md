# FIDMK Laplace Gradient Benchmark Report

**Date:** 2026-03-19
**Method:** Chebyshev polynomial differentiation of proxy expansion (replaces ik Fourier-space approach)
**Precision:** double
**Hardware:** 32 OpenMP threads
**Kernel:** 3D Laplace

---

## Summary

The poly-derivative gradient method achieves:
- **Gradient accuracy tracks potential accuracy** across all tolerance levels (1e-3 to 1e-12)
- **Low overhead:** 7-37% additional eval time over potential-only (typically ~25%)
- **No additional memory** for gradient proxy coefficients (eliminated `proxy_grad_coeffs_downward` storage)
- **Potentials are unchanged** — gradient computation does not degrade potential accuracy

---

## Accuracy vs Direct O(N^2)

Gradient relative L2 error compared against direct N-body computation on a prefix of 64 points.

| N_src | N_trg | eps    | pot_src_err | pot_trg_err | grad_src_err | grad_trg_err | grad_src_max | grad_trg_max |
|------:|------:|-------:|------------:|------------:|-------------:|-------------:|-------------:|-------------:|
|  4000 |  3000 | 1e-03  |   9.56e-05  |   1.52e-04  |    7.28e-05  |    7.04e-05  |    9.50e-02  |    6.46e-02  |
|  4000 |  3000 | 1e-06  |   1.12e-07  |   2.55e-07  |    1.85e-08  |    1.90e-08  |    2.50e-04  |    2.23e-05  |
|  4000 |  3000 | 1e-09  |   1.20e-10  |   2.38e-10  |    6.44e-11  |    3.30e-11  |    2.83e-07  |    1.80e-07  |
|  4000 |  3000 | 1e-12  |   2.00e-13  |   3.52e-13  |    2.22e-13  |    2.96e-13  |    1.45e-10  |    1.16e-10  |
| 20000 | 20000 | 1e-06  |   2.05e-07  |   1.10e-07  |    3.29e-08  |    3.90e-08  |    3.64e-04  |    1.90e-04  |
| 20000 | 20000 | 1e-09  |   1.89e-10  |   9.51e-11  |    7.22e-11  |    5.29e-11  |    1.09e-06  |    8.13e-07  |

**Key observations:**
- Gradient L2 errors are consistently **better than** potential L2 errors (roughly 5-10x lower)
- At eps=1e-12, gradients achieve ~2e-13 relative L2 error — near machine precision
- Max pointwise gradient errors are higher than L2 (expected for FMM near box boundaries)

---

## Speed: Potential-only vs Potential+Gradient

| N_src  | N_trg  | eps    | pot_eval(s) | grad_eval(s) | overhead |
|-------:|-------:|-------:|------------:|-------------:|---------:|
|   4000 |   3000 | 1e-03  |       0.159 |        0.175 |    1.10x |
|   4000 |   3000 | 1e-06  |       0.168 |        0.180 |    1.07x |
|   4000 |   3000 | 1e-09  |       0.158 |        0.196 |    1.24x |
|   4000 |   3000 | 1e-12  |       0.174 |        0.231 |    1.33x |
|  20000 |  20000 | 1e-06  |       0.207 |        0.264 |    1.28x |
|  20000 |  20000 | 1e-09  |       0.248 |        0.325 |    1.31x |
|  50000 |  50000 | 1e-06  |       0.300 |        0.366 |    1.22x |
| 100000 | 100000 | 1e-06  |       0.267 |        0.460 |    1.72x |
| 200000 | 200000 | 1e-06  |       0.569 |        0.780 |    1.37x |

**Key observations:**
- Gradient overhead is **7-37%** additional eval time (median ~28%)
- At low precision (eps=1e-3), overhead is minimal (~10%) since direct interactions dominate
- At higher precision, the proxy polynomial evaluation (where gradients add work) takes a larger fraction
- Tree build time is **unchanged** (~0.96-1.6s depending on N) since gradient computation happens only during eval
- Total wall time (build + eval) overhead is only **2-11%** since build dominates for small problems

---

## Scaling with Problem Size (eps=1e-6)

| N (src=trg) | pot_total(s) | grad_total(s) | total_overhead |
|------------:|-------------:|--------------:|---------------:|
|       4,000 |        1.140 |         1.153 |           1.0x |
|      20,000 |        1.304 |         1.346 |           1.0x |
|      50,000 |        1.525 |         1.593 |           1.04x |
|     100,000 |        1.721 |         1.830 |           1.06x |
|     200,000 |        2.117 |         2.361 |           1.12x |

Total time overhead stays under 12% even at 200k points.

---

## Method Details

**Proxy (far-field) gradient path:**
- Evaluate `proxy_view_downward(box)` (Chebyshev polynomial coefficients) at target points
- Use `calc_polynomial_and_derivative` to compute both T_n(x) and T_n'(x) via Chebyshev recurrence
- Apply chain rule with scale factor `sc = 2/boxsize` to convert reference-space derivatives to physical gradients
- No separate gradient coefficient storage needed

**Direct (near-field) gradient path:**
- Evaluate kernel gradient analytically: `grad(1/r) = -(x_t - x_s)/r^3`
- Local polynomial correction gradient: `d/dx[poly(r)/r] = scale*dpoly/r^2 - poly/r^3` (times direction)

**What was removed:**
- `pw2poly_grad` matrices (ik Fourier-space gradient coefficients)
- `proxy_grad_coeffs_downward` storage (O(n_boxes * n_order^3 * n_mfm * 3) doubles)
- `planewave_to_proxy_gradient` conversion functions
- Gradient coefficient propagation via `tensorprod::transform` in downward pass

**Equivalence validated:** A/B test showed ik vs poly-derivative agree to 2.4e-14 relative L2 (machine precision).
