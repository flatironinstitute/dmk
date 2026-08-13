# Changelog

Notable changes to DMK.

## [Unreleased]

First tagged release. Prior history is summarized here rather than reconstructed commit by commit.

### Added

- **C API for fast kernel evaluation.** Tree lifecycle (`pdmk_tree_create`, `pdmk_tree_eval`,
  `pdmk_tree_update_charges`, `pdmk_tree_destroy`) plus the one-shot `pdmk`, each with a
  single-precision `f` counterpart. `pdmk_tree_update_charges` re-evaluates new charges on an
  existing tree without rebuilding the geometry.
- **Kernels.** Yukawa, Laplace, sqrt-Laplace, Stokeslet, Stresslet and Laplace-dipole. See
  `docs/features.rst` for the supported dimension, evaluation type and boundary-condition
  combinations.
- **GPU offload.** `eval_path = DMK_EVAL_PATH_GPU` runs the full pipeline (upward pass, plane-wave
  translations, downward pass, near-field direct) on CUDA for all six kernels in 3D, in both
  precisions, over the full 3--12 digit range in double. Built with `-DDMK_GPU_OFFLOAD=ON`.
- **`gpu_device_id`** in `pdmk_params` selects the CUDA device. A process is pinned to the first
  device it uses; requesting a second one fails with `DMK_ERR_INVALID_ARGUMENT` rather than
  silently reusing JIT modules compiled for the first.
- **`pdmk_direct` / `pdmk_directf`.** Brute-force O(n_src * n_trg) summation behind the same
  argument layout as `pdmk`, including the MPI communicator: sources are gathered across ranks, so
  it is a drop-in reference for validating a tree solve at any scale a direct sum can reach. Runs on
  the CPU or, with `eval_path = DMK_EVAL_PATH_GPU`, on CUDA -- for every kernel in both 2D and 3D
  and both precisions, since an all-pairs sum has none of the tree path's dimension or rank
  restrictions.
- **Periodic boundary conditions** (`use_periodic`) for the scalar kernels (Laplace, Yukawa,
  sqrt-Laplace) in 2D and 3D on the CPU, and in 3D on the GPU.
- **ESP (Ewald summation with prolates)**, an experimental periodic and free-space electrostatics
  solver: `pdmk_esp_plan_create`, `pdmk_esp_eval`, `pdmk_esp_plan_destroy` and the one-shot
  `pdmk_esp`.
- **Structured error handling.** Entry points return `dmk_error`; `*_create` returns `NULL` on
  failure; `pdmk_last_error_message()` gives per-thread detail. No C++ exception escapes the C
  boundary, and unsupported kernel/dimension/eval/path combinations are rejected with
  `DMK_ERR_INVALID_ARGUMENT` before any work is done.
- **Optional JIT kernels** (`-DDMK_USE_JIT=ON`, LLVM via RuFuS) that compile the short-range
  evaluator for the requested precision instead of using the pre-compiled AOT tables.
- **Packaging.** Versioned shared library with SONAME, installed CMake package config for
  `find_package(dmk)`, `dmk.pc` for pkg-config, and a generated `dmk/version.h` exposing
  `pdmk_version_string()`, `pdmk_version()` and `pdmk_git_commit()`.
- **Documentation** built with Sphinx, Doxygen and Breathe, published on Read the Docs.

### Known limitations

- `DMK_POTENTIAL_GRAD_HESSIAN` and `DMK_VELOCITY_PRESSURE` are accepted by the enum but not
  implemented by any path.
- Stokeslet, Stresslet and Laplace-dipole are 3D only.
- The GPU *tree* path is 3D and single-rank; it cannot be combined with multi-rank MPI. The GPU
  direct path has neither restriction.
- Periodic boundary conditions are limited to the scalar kernels.
