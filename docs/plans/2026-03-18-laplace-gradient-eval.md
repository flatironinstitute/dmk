# Laplace Gradient Eval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add API-level Laplace gradient evaluation in the C++ DMK tree so `pdmk_tree_eval`/`pdmk` fill `grad_src` and `grad_trg` when `pgh_src` or `pgh_trg` is `DMK_POTENTIAL_GRAD`.

**Architecture:** Extend the existing particle-data flow to carry gradients alongside potentials, then add the missing Laplace gradient contributions from both the downward local-expansion evaluation and the direct residual/local kernel path. Keep the first change limited to gradients only; do not add Hessian support in this pass.

**Tech Stack:** C++20, SCTL particle tree, doctest, existing DMK proxy/direct kernel code, optional Fortran reference build for parity checks.

---

### Task 1: Add a Failing Laplace Gradient Regression Test

**Files:**
- Modify: `src/dmk.cpp`
- Test: `src/dmk.cpp`

**Step 1: Write the failing test**

Add a doctest that:
- uses `DMK_LAPLACE`
- uses `n_dim = 3`
- sets `params.pgh_src = DMK_POTENTIAL_GRAD`
- sets `params.pgh_trg = DMK_POTENTIAL_GRAD`
- calls `pdmk_tree_eval`
- checks at least a small prefix of `grad_src` and `grad_trg` against direct Laplace gradients computed in the test
- verifies the current implementation leaves gradients wrong or zero before the fix

**Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_all -j4 && ./build/test/test_all --test-case="*[DMK]*Laplace*gradient*"`

Expected: FAIL because Laplace gradients are not populated by the C++ tree path.

**Step 3: Keep the direct reference calculation in the test minimal**

Use the analytic 3D Laplace charge gradient:

```cpp
grad += q * (-(x_t - x_s)) / |x_t - x_s|^3;
```

Skip self interactions with `dr2 == 0`.

**Step 4: Re-run the same test after any cleanup**

Run: `cmake --build build --target test_all -j4 && ./build/test/test_all --test-case="*[DMK]*Laplace*gradient*"`

Expected: still FAIL until implementation lands.

### Task 2: Wire Gradient Storage Through the Tree and Public API

**Files:**
- Modify: `include/dmk/tree.hpp`
- Modify: `src/tree.cpp`
- Modify: `src/dmk.cpp`

**Step 1: Add gradient particle-data storage to the tree**

Add sorted buffers and accessors matching existing potential storage:
- `grad_src_sorted`, `grad_trg_sorted`
- counts/offsets if needed
- `grad_src_view(i_box)`, `grad_trg_view(i_box)`

Keep shapes consistent with existing API layout:
- source gradients: `(n_mfm, DIM, n_src_local_or_halo)`
- target gradients: `(n_mfm, DIM, n_trg_local)`

**Step 2: Register gradient particle data during tree construction**

In the constructor:
- allocate zeroed gradient vectors only when `params.pgh_src >= DMK_POTENTIAL_GRAD` or `params.pgh_trg >= DMK_POTENTIAL_GRAD`
- call `AddParticleData` for source and target gradients
- fetch sorted data handles after refinement the same way potential data is fetched today

**Step 3: Zero gradient buffers in the downward pass**

In `DMKPtTree::downward_pass`, zero gradient arrays alongside potential arrays when gradients are requested.

**Step 4: Return gradients from the public APIs**

In both:
- `dmk::pdmk(...)`
- `dmk::pdmk_tree_eval(...)`

copy out gradient particle data when the corresponding output pointer is non-null and the requested `pgh_*` includes gradients.

**Step 5: Rebuild and run the new test**

Run: `cmake --build build --target test_all -j4 && ./build/test/test_all --test-case="*[DMK]*Laplace*gradient*"`

Expected: FAIL, but now due to missing mathematical contributions rather than missing storage.

### Task 3: Add Gradient Evaluation for Downward Local Expansions

**Files:**
- Modify: `include/dmk/proxy.hpp`
- Modify: `src/proxy.cpp`
- Modify: `src/tree.cpp`

**Step 1: Add proxy-evaluation entry points for gradients**

Extend proxy target evaluation with a gradient-aware routine, for example:
- `eval_targets_grad(...)`

It should evaluate the tensor-product Chebyshev expansion and its first derivatives at target points.

**Step 2: Implement Chebyshev derivative tables locally**

Use the existing polynomial recurrence machinery to build per-target basis values and first derivatives for each coordinate. Apply the chain-rule scale factor `sc` when converting derivative-in-reference-space to physical-space gradient.

**Step 3: Accumulate into gradient buffers**

For each charge dimension and each target:
- add the x/y(/z) derivative into `grad_*`
- preserve existing potential accumulation behavior

**Step 4: Call the gradient-aware evaluator in the tree**

In `DMKPtTree::form_eval_expansions`, when `pgh_src` or `pgh_trg` requests gradients:
- keep calling the existing potential evaluator
- additionally call the new gradient evaluator for source and/or target particles

**Step 5: Re-run the Laplace gradient test**

Run: `cmake --build build --target test_all -j4 && ./build/test/test_all --test-case="*[DMK]*Laplace*gradient*"`

Expected: FAIL remains, but the error should now be attributable to missing direct near-field Laplace gradient contributions.

### Task 4: Add Laplace Gradient Contributions in the Direct Interaction Path

**Files:**
- Modify: `include/dmk/direct.hpp`
- Modify: `src/direct.cpp`
- Modify: `include/dmk/vector_kernels.hpp`
- Modify: `src/tree.cpp`

**Step 1: Extend direct-eval interfaces to accept an optional gradient output**

Update the C++ `direct_eval` signature so the caller can pass:
- potential output view
- optional gradient output view

Keep call sites explicit rather than inferring from null global state.

**Step 2: Implement 3D Laplace local-kernel gradient evaluation**

Follow the Fortran sign convention from `l3ddirectcg`:
- kernel value is `1/r + f_local(r^2)`
- gradient is `-(x_t - x_s)/r^3 + 2 * f_local'(r^2) * (x_t - x_s)`

Use the same polynomial coefficient tables already used for the local potential correction.

**Step 3: Implement 2D Laplace local-kernel gradient evaluation**

For the log kernel:
- potential is `0.5 * log(r^2) + f_local(r^2)`
- gradient is `(x_t - x_s)/r^2 + 2 * f_local'(r^2) * (x_t - x_s)`

Use the same masking rules as the potential path (`thresh`, `d2max`).

**Step 4: Plumb gradients through direct interaction evaluation**

In `DMKPtTree::evaluate_direct_interactions`, when gradients are requested:
- pass `grad_src_view(i_box)` for source self/list1 interactions
- pass `grad_trg_view(i_box)` for target list1 interactions

Do not add self-interaction correction terms to gradients; only potentials need the existing constant correction.

**Step 5: Re-run the focused test**

Run: `cmake --build build --target test_all -j4 && ./build/test/test_all --test-case="*[DMK]*Laplace*gradient*"`

Expected: PASS.

### Task 5: Add a Broader Regression Check

**Files:**
- Modify: `src/dmk.cpp`

**Step 1: Extend an existing DMK regression test**

Add a smaller tolerance-based check that, for Laplace only:
- source gradients match direct gradients on a prefix of particles
- target gradients match direct gradients on a prefix of targets

Prefer a modest sample size to keep runtime acceptable.

**Step 2: Run the targeted regression group**

Run: `cmake --build build --target test_all -j4 && ./build/test/test_all --test-case="*[DMK]*pdmk*"`

Expected: PASS.

### Task 6: Final Verification

**Files:**
- Modify: `src/dmk.cpp`
- Modify: `include/dmk/tree.hpp`
- Modify: `src/tree.cpp`
- Modify: `include/dmk/proxy.hpp`
- Modify: `src/proxy.cpp`
- Modify: `include/dmk/direct.hpp`
- Modify: `src/direct.cpp`
- Modify: `include/dmk/vector_kernels.hpp`

**Step 1: Build the test target from scratch state**

Run: `cmake --build build --target test_all -j4`

Expected: build succeeds with no compile errors.

**Step 2: Run focused Laplace gradient tests**

Run: `./build/test/test_all --test-case="*[DMK]*Laplace*gradient*"`

Expected: PASS.

**Step 3: Run the broader DMK regression slice**

Run: `./build/test/test_all --test-case="*[DMK]*pdmk*"`

Expected: PASS.

**Step 4: Optional parity run if reference build is enabled**

Run: `./build/test/test_all --test-case="*[DMK]*all*"`

Expected: PASS, including any reference-comparison coverage already present in the suite.
