# PBC Development Status for 3D Laplace

## Overview

Periodic boundary condition (PBC) support for the 3D Laplace kernel in FIDMK.
The DMK kernel decomposition is (eq 3.25 of Jiang & Greengard 2025):

```
1/r = W_0(r) + D_0(r) + D_1(r) + ... + D_{L-1}(r) + R_L(r)
```

For PBC, each component must correctly account for periodic images.

## Completed

### 1. Direct/residual interactions (commit `3c4206e`)

**Problem:** SCTL's periodic tree provides correct neighbor topology (all 27
neighbor slots filled), but on a single MPI rank the neighbor box indices point
to un-shifted coordinates. The direct evaluator was missing all periodic-image
contributions.

**Fix:**
- Added `list1_shift_` (parallel to `list1_`) in `tree.hpp` storing integer
  periodic shifts per neighbor entry, derived from the SCTL neighbor slot
  encoding: slot `k = d0 + 3*d1 + 9*d2` maps to direction `(d-1)` in each dim.
- `build_direct_interaction_lists()` computes shifts via
  `compute_periodic_shift_from_slot()`.
- `evaluate_direct_interactions()` applies shifts to source positions before
  calling the kernel evaluator.

**Test:** `[DMK] pdmk 3d Laplace PBC direct verification` — verifies at 3/6/9/12
digit precision against an O(N²) loop using the same residual polynomial over
27 periodic images.

### 2. Difference kernels D_1..D_{L-1} + residual R_L (current work)

**Result:** The existing PW code for levels >= 1 already handles PBC correctly.
At these levels, all neighbor slots point to distinct box indices, so the
`neighbor != box` condition in `form_eval_expansions` works without modification.

**Key formula — D_l(r) in real space (3D Laplace):**

```
D_l(r) = [1/(r * c0)] * [Phi(min(1, 2r/boxsize_l)) - Phi(r/boxsize_l)]
```

where:
- `Phi(x) = int_0^x psi_0(t) dt` = `prolate0_fun.int_eval(x)`
- `c0 = prolate0_fun.intvals(beta)[0]` = `int_0^1 psi_0(t) dt`
- `psi_0` is the 0th order prolate spheroidal wave function with bandlimit `beta`

**Derivation:** The code stores the D_l Fourier multiplier as

```
D_hat_l(k) = (4pi/k^2) * (1/c0) * int_0^1 psi_0(u) [cos(k*u*r_s) - cos(k*u*r_b)] du
```

where `r_s = boxsize[l+1]`, `r_b = boxsize[l]`. Applying the 3D inverse spherical
Fourier transform and using the distributional identity
`int_0^inf cos(ak) sin(bk) dk/k = (pi/2) H(b-a)` yields the closed-form above.

**Tests:**
- `[DMK] pdmk 3d Laplace diff+residual vs erfc reference (free space)` —
  6-digit, L2 = 8.5e-7
- `[DMK] pdmk 3d Laplace PBC diff+residual vs PSWF reference` — 4 precisions:

| n_digits | L2 error |
|----------|----------|
| 3        | 6.4e-4   |
| 6        | 9.1e-7   |
| 9        | 7.8e-10  |
| 12       | 8.1e-13  |

### 3. Root level W_0 + D_0: kernel formula validated (commit `3878f1f`)

**Problem:** The combined window + coarsest difference kernel (W_0 + D_0) operates
at the root level and needs periodic Fourier treatment. The current tree code
skips root-level periodic neighbors (`neighbor != box` in `form_eval_expansions`).

**PSWF kernel formula (validated):**

The PSWF eigenvalue relation gives a telescoping property for the DMK difference
kernels. The combined W_0 + D_0 Fourier transform on the periodic grid is:

```
(W_0 + D_0)_hat(k) = (4*pi / psi_0(0)) * psi_0(|k| * sigma_1) / |k|^2
```

where:
- `sigma_1 = boxsize[1] / beta` (scale parameter for level 1)
- `psi_0(x) = pf.eval_val(x)` (PSWF, supported on [-1, 1])
- Grid: `k = (2*pi/L) * (nx, ny, nz)`, `n_modes = ceil(beta/pi) + 2`

This is bandlimited — only ~5-11 modes per dimension are needed (PSWF decays to
zero), making the Fourier sum trivially fast.

**Test:** `[DMK] pdmk 3d Laplace PBC full vs Ewald` — verifies V_partial (tree
D_1..R_L) + V_{W0+D0} (direct PW with PSWF kernel) matches V_Ewald (self-contained
Ewald sum: erfc short-range + Fourier long-range, alpha=10).

| n_digits | L2 error |
|----------|----------|
| 3        | 7.3e-4   |
| 6        | 1.1e-6   |
| 9        | 9.1e-10  |
| 12       | 8.0e-13  |

### 4. Tree integration: periodic pw2poly and downward pass (commit `37579c9`, `ec2b566`)

**Changes to `precompute_fourier_data`:** When `pbc=true`, `window_fourier_data`
is built with periodic grid parameters instead of free-space window parameters:
- `dk = 2*pi/L` (periodic grid spacing)
- `n_pw_periodic = 2*ceil(beta/pi)+3` modes per dimension (stored as tree member)
- `radialft`: PSWF kernel evaluated at periodic k-points via `mk_tensor_product_fourier_transform`
- `poly2pw`/`pw2poly`: built by `calc_planewave_coeff_matrices(boxsize[0], dk, n_pw_periodic, n_order)`

**Changes to `downward_pass`:**
- Root block: uses local PW temp sized for `n_pw_periodic` (not the tree's `n_pw`)
- Level-0 `form_outgoing_expansions` skipped (D_0 already included in periodic root kernel)
- Level-0 `form_eval_expansions` still runs (to propagate `proxy_downward(0)` to children)

**Self-energy correction in `evaluate_direct_interactions`:**
Computes periodic PSWF kernel self-interaction at r=0 via Fourier sum:
`self_periodic = (1/V) * Σ_{k≠0} (4pi/psi_0(0)) * psi_0(|k|*sigma_1) / |k|^2`
Adds `delta = self_periodic - w0_free[0]` to all w0 correction entries.

### 5. Full pipeline verified (commit `d03ffd7`, `2c4af06`)

**Test:** `[DMK] pdmk 3d Laplace PBC full pipeline vs Ewald` — runs full
`pdmk_tree_create`/`pdmk_tree_eval` with `pbc=true` at 3/6/9/12 digits,
comparing potential and gradient for both sources and targets against Ewald.

| n_digits | pot_trg  | grad_src | grad_trg | pot_src  |
|----------|----------|----------|----------|----------|
| 3        | 1.0e-3   | 6.5e-4   | 5.5e-4   | 3.0e-3   |
| 6        | 1.4e-6   | 1.5e-6   | 1.3e-6   | 1.4e-3   |
| 9        | 1.3e-9   | 1.8e-9   | 1.7e-9   | 7.7e-4   |
| 12       | 1.3e-12  | 3.3e-12  | 2.5e-12  | 4.3e-4   |

Target potential, source gradient, and target gradient all achieve eps-level
accuracy. Source potential has ~1e-3 residual (see known issue below).

## Known issue: PBC source potential self-energy

Source potentials have a ~1e-3 relative error from the proxy→PW→proxy round-trip
projection. The finite-rank Chebyshev-to-PW matrices don't exactly preserve
the kernel self-interaction at r=0, so the w0 correction (computed from the
theoretical Fourier sum) over-corrects by ~0.08 per unit charge.

The source gradient is unaffected (the gradient of the smooth self-contribution
is zero by symmetry at r=0).

Possible fix: compute the proxy pipeline's actual self-energy by running a unit
charge through the proxy→PW→kernel→PW→proxy round-trip, rather than using the
theoretical Fourier sum.

## Not yet done

### 6. Source potential self-energy refinement

Fix the ~1e-3 source potential error by computing the proxy pipeline's actual
self-energy rather than the theoretical Fourier sum.

### 7. MPI multi-rank support

Current PBC tests only work on a single rank. The Ewald reference and source/target
gathering use `_owned` data without MPI reduction.

## Key code locations

- `include/dmk/tree.hpp` — `list1_shift_`, tree data structures
- `src/tree.cpp` — `build_direct_interaction_lists()`, `evaluate_direct_interactions()`,
  `form_eval_expansions()`, `build_plane_wave_interaction_lists()`
- `src/fourier_data.cpp` — `laplace_3d_difference_kernel_ft()`,
  `laplace_3d_windowed_kernel_ft()`
- `src/dmk.cpp` — all PBC verification tests
- `include/dmk/fourier_data.hpp` — `FourierData`, `Prolate0Fun`
- `include/dmk.h` — `pdmk_params.use_periodic`

## PSWF reference

The prolate spheroidal wave function machinery lives in `Prolate0Fun`:
- `eval_val(x)` — evaluate ψ₀(x) for x ∈ [-1, 1]
- `int_eval(r)` — evaluate ∫₀ʳ ψ₀(t) dt
- `intvals(beta)` — returns [c0, c1, g0d2, c4] where c0 = ∫₀¹ ψ₀
- `beta` = `procl180_rescale(eps)` — PSWF bandlimit parameter c
