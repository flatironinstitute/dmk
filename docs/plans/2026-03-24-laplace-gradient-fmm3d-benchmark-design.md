# Laplace Gradient FMM3D Benchmark Design

**Date:** 2026-03-24

**Goal:** Add a reproducible benchmark workflow for 3D Laplace gradient evaluation that uses FMM3D to generate a fixed reference dataset once, then benchmarks DMK source and target gradients across tolerance, OpenMP-thread, and MPI-rank sweeps.

## Scope

- Kernel: 3D Laplace
- Outputs: gradients only
- Evaluation points:
  - sources: `N_src = 1e6`
  - targets: `N_trg = 1e4`
- Accuracy levels: requested digits `3, 6, 9, 12`, implemented as `eps = 1e-3, 1e-6, 1e-9, 1e-12`
- OpenMP threads per rank: `1, 2, 4, 8, 16, 32, 64`
- MPI ranks: `1, 2, 4`
- Reference backend: local FMM3D checkout in `/mnt/home/xgao1/codes/FMM3D`

## Non-Goals

- No Hessian benchmarking in this pass.
- No periodic benchmark coverage in this pass.
- No FMM3D performance benchmarking. FMM3D is used only to generate the fixed reference once.

## Approaches Considered

### 1. Inline FMM3D reference inside the benchmark executable

Pros:
- single executable
- no external artifact management

Cons:
- recomputes reference repeatedly
- mixes OpenMP-only FMM3D reference generation into the MPI benchmark loop
- harder to reason about timing contamination

### 2. Two-stage pipeline with stored reference artifacts

Pros:
- matches the requirement exactly
- reference is computed once and reused
- benchmark timing stays focused on DMK
- easy to rerun subsets of the matrix

Cons:
- requires explicit artifact format and storage

### 3. External Julia or Python FMM3D driver

Pros:
- fast to prototype

Cons:
- adds another language/runtime dependency to the benchmark workflow
- weaker fit for this repo than a C++ or repo-local script path

## Chosen Design

Use a two-stage benchmark pipeline.

Stage 1 generates one deterministic dataset and one deterministic FMM3D reference for source and target gradients. Stage 2 runs DMK on the exact same data for every `(eps, mpi_ranks, omp_threads)` configuration and compares against the stored reference.

## Data Model

Store benchmark inputs and reference outputs under:

- `results/laplace_grad_fmm3d/`

Artifacts:

- dataset metadata
- source coordinates
- target coordinates
- source charges
- FMM3D source gradients
- FMM3D target gradients
- raw DMK benchmark CSV
- generated summary tables

The dataset is deterministic and generated with seed `0`, reusing the existing compare-driver style random-point workflow so the benchmark stays consistent with the current DMK performance path.

## Execution Model

### Reference generation

- Build or link the local FMM3D checkout.
- Generate one dataset with `N_src = 1e6` and `N_trg = 1e4`.
- Run FMM3D once at a tighter tolerance than the DMK sweep.
- Compute both:
  - source gradients
  - target gradients
- Normalize by `1 / (4*pi)` so the stored reference matches DMK’s Laplace output convention used by existing comparison code.

### DMK benchmark sweep

For each configuration:

- `eps in {1e-3, 1e-6, 1e-9, 1e-12}`
- `omp_threads in {1, 2, 4, 8, 16, 32, 64}`
- `mpi_ranks in {1, 2, 4}`

Run DMK with:

- `pgh_src = DMK_POTENTIAL_GRAD`
- `pgh_trg = DMK_POTENTIAL_GRAD`
- `kernel = DMK_LAPLACE`
- fixed dataset loaded from disk

Collect:

- tree build time
- evaluation time
- total time
- source-gradient relative L2 error
- target-gradient relative L2 error
- source-gradient max relative error
- target-gradient max relative error
- throughput based on total evaluated points

Use one warm-up run per configuration, then several measured runs and report medians.

## Report

Write the final report to:

- `docs/reports/2026-03-24-laplace-gradient-fmm3d-benchmark.md`

The report includes:

- benchmark setup
- hardware/runtime setup
- reference-generation method
- accuracy tables for source and target gradients
- timing tables for all MPI/OMP/tolerance combinations
- compact observations on scaling and accuracy trends

Include markdown formulas for:

- relative L2 error

  `\[
  \mathrm{relL2}(g, g^{ref}) = \sqrt{\frac{\sum_i \|g_i - g_i^{ref}\|_2^2}{\sum_i \|g_i^{ref}\|_2^2}}
  \]`

- max relative error

  `\[
  \mathrm{maxRel}(g, g^{ref}) = \max_i \frac{\|g_i - g_i^{ref}\|_2}{\|g_i^{ref}\|_2}
  \]`

- speedup

  `\[
  S(p, t) = \frac{T_{1,1}}{T_{p,t}}
  \]`

- parallel efficiency

  `\[
  E(p, t) = \frac{S(p, t)}{p t}
  \]`

## Error Handling

- If reference artifacts already exist, reuse them unless an explicit overwrite flag is set.
- Fail early if the dataset metadata does not match the requested `N_src`, `N_trg`, dimensionality, or seed.
- Record the exact runtime configuration in each output file header.
- Keep raw benchmark outputs even if a subset of runs fail.

## Testing

- Add a small smoke test mode for dataset generation and reference loading.
- Add a small benchmark mode to validate CSV layout and error computation on a reduced problem.
- Verify that the normalization and gradient layout match FMM3D on both sources and targets before launching the full matrix.

## Risks

- FMM3D is OpenMP-only, so reference generation must remain outside the MPI sweep.
- The full matrix is large, so batch-job orchestration and resumable outputs are important.
- Large artifacts may be expensive to rewrite, so the workflow should separate dataset/reference generation from benchmark execution.
