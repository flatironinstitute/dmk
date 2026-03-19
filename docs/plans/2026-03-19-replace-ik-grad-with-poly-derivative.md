# Replace ik Gradient Evaluation with Polynomial Derivative

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `ik` (Fourier-space multiplication) approach for computing Laplace gradients in the FMM tree with direct polynomial differentiation of the downward proxy expansion, eliminating the separate gradient coefficient storage and planewave gradient machinery.

**Architecture:** The current approach precomputes `pw2poly_grad` matrices (pw2poly scaled by `i*k*hpw`), converts plane waves to separate gradient proxy coefficients, propagates them down the tree, and evaluates them at leaf points using `eval_targets`. The replacement evaluates gradients at leaf points by differentiating the potential proxy polynomial directly using `eval_target_gradients`, which already exists and is FD-tested. An A/B validation test confirms equivalence before removing the old code.

**Tech Stack:** C++20, SCTL particle tree, doctest, existing Chebyshev polynomial infrastructure.

---

### Task 1: Add A/B Equivalence Test (ik vs poly-derivative)

This test runs the tree with the current ik method, then runs a second tree evaluation where gradients are computed via polynomial differentiation instead, and asserts the two agree to near machine precision. This validates equivalence before we remove the ik path.

**Note:** The env var `DMK_GRAD_USE_POLY_DERIVATIVE` is read at tree construction time. Since doctest runs test cases serially by default, this is safe. The env var and branching code are temporary — they are removed in Task 3.

**Files:**
- Modify: `src/dmk.cpp` (add test at end, before the `extern "C"` block around line 494)

- [ ] **Step 1: Write the A/B comparison test**

Add this test after the existing `"[DMK] pdmk 3d Laplace gradient"` test case (after line 430 of `src/dmk.cpp`):

```cpp
TEST_CASE("[DMK] Laplace gradient: ik vs poly-derivative equivalence") {
    // This test validates that computing gradients via polynomial differentiation
    // of the downward proxy expansion produces the same results as the current
    // ik (Fourier-space) approach. Both methods are run and compared.
    constexpr int n_dim = 3;
    constexpr int n_src = 4000;
    constexpr int n_trg = 3000;
    constexpr int nd = 1;

    sctl::Vector<double> r_src, r_trg, rnormal, charges, dipstr;
    dmk::util::init_test_data(n_dim, nd, n_src, n_trg, false, true, r_src, r_trg, rnormal, charges, dipstr, 0);

    pdmk_params params;
    params.eps = 1e-7;
    params.n_dim = n_dim;
    params.n_per_leaf = 280;
    params.n_mfm = nd;
    params.pgh_src = DMK_POTENTIAL_GRAD;
    params.pgh_trg = DMK_POTENTIAL_GRAD;
    params.kernel = DMK_LAPLACE;
    params.log_level = SPDLOG_LEVEL_OFF;

    // Run A: current ik method (the default)
    sctl::Vector<double> pot_src_a(n_src * nd), grad_src_a(n_src * nd * n_dim);
    sctl::Vector<double> pot_trg_a(n_trg * nd), grad_trg_a(n_trg * nd * n_dim);
    sctl::Vector<double> hess_src_a(n_src * nd * n_dim * n_dim), hess_trg_a(n_trg * nd * n_dim * n_dim);
    pot_src_a.SetZero(); grad_src_a.SetZero(); pot_trg_a.SetZero(); grad_trg_a.SetZero();
    hess_src_a.SetZero(); hess_trg_a.SetZero();

    pdmk_tree tree_a = pdmk_tree_create(nullptr, params, n_src, &r_src[0], &charges[0],
                                          &rnormal[0], &dipstr[0], n_trg, &r_trg[0]);
    pdmk_tree_eval(tree_a, &pot_src_a[0], &grad_src_a[0], &hess_src_a[0],
                   &pot_trg_a[0], &grad_trg_a[0], &hess_trg_a[0]);
    pdmk_tree_destroy(tree_a);

    // Run B: poly-derivative method (set env var to switch)
    setenv("DMK_GRAD_USE_POLY_DERIVATIVE", "1", 1);
    sctl::Vector<double> pot_src_b(n_src * nd), grad_src_b(n_src * nd * n_dim);
    sctl::Vector<double> pot_trg_b(n_trg * nd), grad_trg_b(n_trg * nd * n_dim);
    sctl::Vector<double> hess_src_b(n_src * nd * n_dim * n_dim), hess_trg_b(n_trg * nd * n_dim * n_dim);
    pot_src_b.SetZero(); grad_src_b.SetZero(); pot_trg_b.SetZero(); grad_trg_b.SetZero();
    hess_src_b.SetZero(); hess_trg_b.SetZero();

    pdmk_tree tree_b = pdmk_tree_create(nullptr, params, n_src, &r_src[0], &charges[0],
                                          &rnormal[0], &dipstr[0], n_trg, &r_trg[0]);
    pdmk_tree_eval(tree_b, &pot_src_b[0], &grad_src_b[0], &hess_src_b[0],
                   &pot_trg_b[0], &grad_trg_b[0], &hess_trg_b[0]);
    pdmk_tree_destroy(tree_b);
    unsetenv("DMK_GRAD_USE_POLY_DERIVATIVE");

    // Potentials should be identical (both methods compute potentials the same way)
    for (int i = 0; i < n_src * nd; ++i)
        CHECK(pot_src_a[i] == doctest::Approx(pot_src_b[i]).epsilon(1e-14));
    for (int i = 0; i < n_trg * nd; ++i)
        CHECK(pot_trg_a[i] == doctest::Approx(pot_trg_b[i]).epsilon(1e-14));

    // Gradients: the two methods should agree to high relative precision
    // (both approximate the same mathematical quantity via different paths)
    auto relative_l2 = [](const sctl::Vector<double> &a, const sctl::Vector<double> &b) {
        double err2 = 0.0, ref2 = 0.0;
        for (int i = 0; i < a.Dim(); ++i) {
            err2 += sctl::pow<2>(a[i] - b[i]);
            ref2 += sctl::pow<2>(a[i]);
        }
        return std::sqrt(err2 / ref2);
    };

    const double src_grad_diff = relative_l2(grad_src_a, grad_src_b);
    const double trg_grad_diff = relative_l2(grad_trg_a, grad_trg_b);
    MESSAGE("src grad ik-vs-poly relative L2: ", src_grad_diff);
    MESSAGE("trg grad ik-vs-poly relative L2: ", trg_grad_diff);
    CHECK(src_grad_diff < 1e-6);
    CHECK(trg_grad_diff < 1e-6);
}
```

- [ ] **Step 2: Add the env var flag to the tree constructor and a `grad_src_view_owned` accessor**

In `src/tree.cpp`, in the `DMKPtTree` constructor (around line 133), add a new member read:

```cpp
// After the existing debug_dump_tree line (line 135):
debug_use_poly_derivative_grad = getenv("DMK_GRAD_USE_POLY_DERIVATIVE") != nullptr;
```

In `include/dmk/tree.hpp`, add the member declaration (after `debug_dump_tree` at line 309):

```cpp
bool debug_use_poly_derivative_grad = false;
```

**CRITICAL:** Also add a new accessor for owned-only source gradients as a 3D view. The existing `grad_src_view` uses `src_counts_with_halo` but `eval_target_gradients` must only write to owned points (matching `r_src_owned_view`). Add this new accessor after `grad_src_flat_view_owned` (after line 204):

```cpp
    ndview<Real, 3> grad_src_view_owned(int i_node) {
        return ndview<Real, 3>({params.n_mfm, DIM, src_counts_owned[i_node]}, grad_src_ptr(i_node));
    }
```

Note: `grad_trg_view` already uses `trg_counts_owned` so it is safe as-is.

- [ ] **Step 3: Wire the poly-derivative path in `form_eval_expansions`**

In `src/tree.cpp`, in `form_eval_expansions` at the leaf evaluation block (lines 957-973), change the gradient evaluation to branch on the flag:

Replace lines 961-963 (source gradient evaluation):
```cpp
                    if (need_grad_src)
                        proxy::eval_targets<Real, DIM>(proxy_grad_view_downward(box), r_src_owned_view(box),
                                                       center_view(box), sc, grad_src_flat_view_owned(box), workspace);
```
With:
```cpp
                    if (need_grad_src) {
                        if (debug_use_poly_derivative_grad)
                            proxy::eval_target_gradients<Real, DIM>(proxy_view_downward(box), r_src_owned_view(box),
                                                                     center_view(box), sc, grad_src_view_owned(box), workspace);
                        else
                            proxy::eval_targets<Real, DIM>(proxy_grad_view_downward(box), r_src_owned_view(box),
                                                           center_view(box), sc, grad_src_flat_view_owned(box), workspace);
                    }
```

Replace lines 968-970 (target gradient evaluation):
```cpp
                    if (need_grad_trg)
                        proxy::eval_targets<Real, DIM>(proxy_grad_view_downward(box), r_trg_owned_view(box),
                                                       center_view(box), sc, grad_trg_flat_view(box), workspace);
```
With:
```cpp
                    if (need_grad_trg) {
                        if (debug_use_poly_derivative_grad)
                            proxy::eval_target_gradients<Real, DIM>(proxy_view_downward(box), r_trg_owned_view(box),
                                                                     center_view(box), sc, grad_trg_view(box), workspace);
                        else
                            proxy::eval_targets<Real, DIM>(proxy_grad_view_downward(box), r_trg_owned_view(box),
                                                           center_view(box), sc, grad_trg_flat_view(box), workspace);
                    }
```

- [ ] **Step 4: Build and run the A/B test**

Run: `cmake --build build --target test_all -j4 && ./build/test/test_all --test-case="*ik vs poly*"`

Expected: PASS. Both methods produce gradients that agree to within `1e-6` relative L2.

- [ ] **Step 5: Also run the existing gradient test to confirm no regression**

Run: `./build/test/test_all --test-case="*Laplace gradient*"`

Expected: PASS (both old and new tests).

- [ ] **Step 6: Commit**

```bash
git add src/dmk.cpp src/tree.cpp include/dmk/tree.hpp
git commit -m "test: add A/B test for ik vs poly-derivative gradient equivalence"
```

---

### Task 2: Add End-to-End Direct Comparison Test for 2D

The existing end-to-end gradient test (`"[DMK] pdmk 3d Laplace gradient"`) only covers 3D. Add a 2D variant that uses the poly-derivative path and validates against direct O(N²) 2D Laplace gradient computation.

**Files:**
- Modify: `src/dmk.cpp` (add test after the A/B test)

- [ ] **Step 1: Write the 2D gradient validation test**

Add after the A/B test:

```cpp
TEST_CASE("[DMK] pdmk 2d Laplace gradient poly-derivative vs direct") {
    constexpr int n_dim = 2;
    constexpr int n_src = 4000;
    constexpr int n_trg = 3000;
    constexpr int nd = 1;
    constexpr double thresh2 = 1e-30;

    sctl::Vector<double> r_src, r_trg, rnormal, charges, dipstr;
    dmk::util::init_test_data(n_dim, nd, n_src, n_trg, false, true, r_src, r_trg, rnormal, charges, dipstr, 0);

    sctl::Vector<double> pot_src(n_src * nd), grad_src(n_src * nd * n_dim);
    sctl::Vector<double> pot_trg(n_trg * nd), grad_trg(n_trg * nd * n_dim);
    sctl::Vector<double> hess_src(n_src * nd * n_dim * n_dim), hess_trg(n_trg * nd * n_dim * n_dim);
    pot_src.SetZero(); grad_src.SetZero(); pot_trg.SetZero(); grad_trg.SetZero();
    hess_src.SetZero(); hess_trg.SetZero();

    setenv("DMK_GRAD_USE_POLY_DERIVATIVE", "1", 1);
    pdmk_params params;
    params.eps = 1e-7;
    params.n_dim = n_dim;
    params.n_per_leaf = 280;
    params.n_mfm = nd;
    params.pgh_src = DMK_POTENTIAL_GRAD;
    params.pgh_trg = DMK_POTENTIAL_GRAD;
    params.kernel = DMK_LAPLACE;
    params.log_level = SPDLOG_LEVEL_OFF;

    pdmk_tree tree = pdmk_tree_create(nullptr, params, n_src, &r_src[0], &charges[0],
                                       &rnormal[0], &dipstr[0], n_trg, &r_trg[0]);
    pdmk_tree_eval(tree, &pot_src[0], &grad_src[0], &hess_src[0],
                   &pot_trg[0], &grad_trg[0], &hess_trg[0]);
    pdmk_tree_destroy(tree);
    unsetenv("DMK_GRAD_USE_POLY_DERIVATIVE");

    // Direct O(N^2) 2D Laplace gradient: grad = q * (x_t - x_s) / |x_t - x_s|^2
    const int n_test_trg = std::min(n_trg, 64);
    std::vector<double> direct_grad_trg(n_test_trg * n_dim, 0.0);
    for (int i_trg = 0; i_trg < n_test_trg; ++i_trg) {
        for (int i_src = 0; i_src < n_src; ++i_src) {
            double dx[2];
            double dr2 = 0.0;
            for (int d = 0; d < n_dim; ++d) {
                dx[d] = r_trg[i_trg * n_dim + d] - r_src[i_src * n_dim + d];
                dr2 += dx[d] * dx[d];
            }
            if (dr2 <= thresh2) continue;
            // grad of 0.5*log(r^2) = (x_t - x_s) / r^2
            for (int d = 0; d < n_dim; ++d)
                direct_grad_trg[d + n_dim * i_trg] += charges[i_src] * dx[d] / dr2;
        }
    }

    std::vector<double> grad_trg_prefix(n_test_trg * n_dim);
    for (int i = 0; i < n_test_trg; ++i)
        for (int d = 0; d < n_dim; ++d)
            grad_trg_prefix[d + n_dim * i] = grad_trg[d + n_dim * i];

    double err2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < n_test_trg * n_dim; ++i) {
        err2 += sctl::pow<2>(grad_trg_prefix[i] - direct_grad_trg[i]);
        ref2 += sctl::pow<2>(direct_grad_trg[i]);
    }
    const double l2_err = std::sqrt(err2 / ref2);
    MESSAGE("2D Laplace gradient poly-derivative L2 error: ", l2_err);
    CHECK(l2_err < 1e-3);
}
```

- [ ] **Step 2: Build and run the new 2D test**

Run: `cmake --build build --target test_all -j4 && ./build/test/test_all --test-case="*2d Laplace gradient*"`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/dmk.cpp
git commit -m "test: add 2D Laplace gradient end-to-end test with poly-derivative"
```

---

### Task 3: Make Poly-Derivative the Default and Remove ik Gradient Code

Now that the A/B test confirms equivalence, remove the ik gradient machinery and make polynomial differentiation the only path.

**Files:**
- Modify: `src/tree.cpp`
- Modify: `include/dmk/tree.hpp`
- Modify: `src/planewave.cpp`
- Modify: `include/dmk/planewave.hpp`

- [ ] **Step 1: Remove the flag and make poly-derivative unconditional in `form_eval_expansions`**

In `src/tree.cpp`, `form_eval_expansions` (around line 957-973), replace the branching code from Task 1 with just the poly-derivative path:

Source gradient (was lines 961-963):
```cpp
                    if (need_grad_src)
                        proxy::eval_target_gradients<Real, DIM>(proxy_view_downward(box), r_src_owned_view(box),
                                                                 center_view(box), sc, grad_src_view_owned(box), workspace);
```

Target gradient (was lines 968-970):
```cpp
                    if (need_grad_trg)
                        proxy::eval_target_gradients<Real, DIM>(proxy_view_downward(box), r_trg_owned_view(box),
                                                                 center_view(box), sc, grad_trg_view(box), workspace);
```

- [ ] **Step 2: Remove `planewave_to_proxy_gradient` calls from the downward pass**

In `src/tree.cpp`, remove the following lines:

In the windowed kernel root block (around lines 1169-1175), remove:
```cpp
        if (need_grad)
            dmk::planewave_to_proxy_gradient<T, DIM>(pw_out_view(0), pw2p, pw2pg, proxy_grad_view_downward(0),
                                                     workspaces_[0]);
```

Also remove the `pw2pg` view construction from the same block (line 1164):
```cpp
        const ndview<std::complex<T>, 2> pw2pg({n_pw, n_order}, &window_fourier_data.pw2poly_grad[0]);
```

In the per-level loop (around lines 1182), remove:
```cpp
        const ndview<std::complex<T>, 2> pw2pg({n_pw, n_order}, &dfd.pw2poly_grad[0]);
```

And remove the `pw2pg` argument from the `form_eval_expansions` call (line 1186).

- [ ] **Step 3: Remove `planewave_to_proxy_gradient` call and gradient coefficients from `form_eval_expansions`**

In `src/tree.cpp` `form_eval_expansions`, remove:

The `pw2poly_grad_view` parameter from the function signature (line 872).

The gradient proxy coefficient zeroing in the `proxy_down_zeroed` block (lines 927-928):
```cpp
                    if (need_grad)
                        proxy_grad_view_downward(box) = 0;
```

The `planewave_to_proxy_gradient` call (lines 933-935):
```cpp
                if (need_grad)
                    dmk::planewave_to_proxy_gradient<Real, DIM>(pw_in_view, pw2poly_view, pw2poly_grad_view,
                                                                proxy_grad_view_downward(box), workspace);
```

The gradient `tensorprod::transform` in the child propagation loop (lines 949-951):
```cpp
                        if (need_grad)
                            tensorprod::transform<Real, DIM>(nd * DIM, add_to_child, proxy_grad_view_downward(box),
                                                             p2c_view, proxy_grad_view_downward(child), workspace);
```

- [ ] **Step 4: Remove gradient coefficient storage from the tree**

In `include/dmk/tree.hpp`, remove these member declarations:
- Line 66: `sctl::Vector<Real> proxy_grad_coeffs_downward;`
- Line 67: `sctl::Vector<sctl::Long> proxy_grad_coeffs_offsets_downward;`

Remove the `proxy_grad_view_downward` and `proxy_grad_ptr_downward` accessor methods (lines 264-275).

Update the `form_eval_expansions` declaration (line 143-145) to remove the `pw2poly_grad_view` parameter:
```cpp
    void form_eval_expansions(const sctl::Vector<int> &boxes, const sctl::Vector<std::complex<Real>> &wpwshift,
                              Real boxsize, const ndview<std::complex<Real>, 2> &pw2poly_view,
                              const sctl::Vector<Real> &p2c);
```

Remove `pw2poly_grad` from `LevelFourierData` (line 90):
```cpp
        sctl::Vector<std::complex<Real>> pw2poly_grad;
```

Remove the `debug_use_poly_derivative_grad` member added in Task 1 (line 309).

- [ ] **Step 5: Remove gradient coefficient allocation from tree construction**

In `src/tree.cpp` `allocate_proxy_coefficients` (around lines 534-589):

Remove `n_grad_coeffs` (line 536):
```cpp
    const int n_grad_coeffs = params.n_mfm * DIM * sctl::pow<DIM>(n_order);
```

Remove `need_grad` (line 537):
```cpp
    const bool need_grad = need_laplace_gradients();
```

Remove `proxy_grad_coeffs_downward` allocation (line 554):
```cpp
    proxy_grad_coeffs_downward.ReInit(need_grad ? n_grad_coeffs * n_proxy_boxes_downward : 0);
```

Remove `proxy_grad_coeffs_offsets_downward` allocation and population (lines 560, 573-588 — the inner parts that deal with grad offsets).

- [ ] **Step 6: Remove `pw2poly_grad` precomputation from `precompute_fourier_data`**

In `src/tree.cpp` `precompute_fourier_data` (around lines 592-632):

Remove from window fourier data block (lines 600, 608-610):
```cpp
    window_fourier_data.pw2poly_grad.ReInit(n_order * n_pw);
    // ...
    dmk::calc_planewave_gradient_coeff_matrix(
        fourier_data.windowed_kernel().hpw, ndview<const std::complex<T>, 2>({n_pw, n_order}, &window_fourier_data.pw2poly[0]),
        ndview<std::complex<T>, 2>({n_pw, n_order}, &window_fourier_data.pw2poly_grad[0]));
```

Remove from per-level block (lines 618, 626-628):
```cpp
        lfd.pw2poly_grad.ReInit(n_order * n_pw);
        // ...
        dmk::calc_planewave_gradient_coeff_matrix(
            fourier_data.difference_kernel(i_level).hpw, ndview<const std::complex<T>, 2>({n_pw, n_order}, &lfd.pw2poly[0]),
            ndview<std::complex<T>, 2>({n_pw, n_order}, &lfd.pw2poly_grad[0]));
```

- [ ] **Step 7: Remove `proxy_grad_coeffs_downward.SetZero()` from `downward_pass`**

In `src/tree.cpp` `downward_pass` (line 1154), remove:
```cpp
    proxy_grad_coeffs_downward.SetZero();
```

Also remove the `need_grad` variable from `downward_pass` (line 1160) and the root-level `need_grad` checks in the windowed kernel block (lines 1169-1175).

- [ ] **Step 8: Remove unused planewave gradient functions**

In `include/dmk/planewave.hpp`, remove:
- `calc_planewave_gradient_coeff_matrix` declaration (lines 10-11)
- `planewave_to_proxy_gradient` declaration (lines 18-22)

In `src/planewave.cpp`, remove:
- `calc_planewave_gradient_coeff_matrix` definition (lines 12-23)
- `pw2proxygrad_2d_component` function (lines 28-68)
- `pw2proxygrad_3d_component` function (lines 70-127)
- `planewave_to_proxy_gradient` definition (lines 239-261)
- All explicit template instantiations for these functions (lines 353-378)

- [ ] **Step 9: Update the A/B test to remove the env var branch**

In `src/dmk.cpp`, update the `"Laplace gradient: ik vs poly-derivative equivalence"` test:
- Remove `setenv`/`unsetenv` calls
- Change it into a simple test that runs the poly-derivative path once and compares against direct O(N²) computation (merge logic from the old A/B test and the existing 3D gradient test), OR simply delete the A/B test since the ik path no longer exists and the `"[DMK] pdmk 3d Laplace gradient"` test already validates against direct computation.

Recommended: delete the A/B test entirely, since the method it compares against is gone.

- [ ] **Step 10: Remove `debug_use_poly_derivative_grad` from tree constructor**

In `src/tree.cpp` constructor (line ~135), remove:
```cpp
    debug_use_poly_derivative_grad = getenv("DMK_GRAD_USE_POLY_DERIVATIVE") != nullptr;
```

- [ ] **Step 11: Build and run all tests**

Run: `cmake --build build --target test_all -j4 && ./build/test/test_all --test-case="*DMK*"`

Expected: All DMK tests PASS. Specifically:
- `[DMK] pdmk 3d Laplace gradient` — PASS (validates against direct O(N²))
- `[DMK] pdmk 2d Laplace gradient poly-derivative vs direct` — PASS (validates 2D)
- `[DMK] proxy eval_target_gradients finite difference` — PASS (validates polynomial differentiation)
- `[DMK] pdmk 3d all` — PASS (no regression on potential-only paths)

- [ ] **Step 12: Commit**

```bash
git add include/dmk/tree.hpp include/dmk/planewave.hpp src/tree.cpp src/planewave.cpp src/dmk.cpp
git commit -m "refactor: replace ik gradient eval with polynomial derivative

Remove the Fourier-space ik gradient machinery (pw2poly_grad, proxy_grad_coeffs_downward,
planewave_to_proxy_gradient) and compute gradients by differentiating the downward proxy
polynomial directly using eval_target_gradients. This simplifies the code and removes
O(n_boxes * n_order^DIM * n_mfm * DIM) storage for gradient coefficients."
```

---

### Task 4: Final Verification

**Files:**
- All modified files from Tasks 1-3

- [ ] **Step 1: Clean rebuild**

Run: `cmake --build build --target test_all -j4`

Expected: build succeeds with no compile errors or warnings related to removed code.

- [ ] **Step 2: Run all DMK tests**

Run: `./build/test/test_all --test-case="*DMK*"`

Expected: All PASS.

- [ ] **Step 3: Run the full test suite**

Run: `./build/test/test_all`

Expected: All PASS. No regressions.
