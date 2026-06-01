# 3D Laplace Dipole Path in pdmk4

This branch adds 3D Laplace dipole-to-potential support to the
`pdmk4.f` point-DMK driver for the case
`ikernel = 1`, `dim = 3`, and `ifdipole = 1`.

## What Changed

`src/pdmk/pdmk4.f` now initializes a second local-kernel coefficient
table for dipole sources in the 3D Laplace case, passes that table
through `pdmk_direct_c`, and uses coefficient-based charge and dipole
local corrections when dipoles are present.

The same file now chooses the 3D Laplace plane-wave order `npw` and
proxy tensor order `norder` from the calibrated smooth-kernel table used
by the adaptive smooth-DMK tests. The table interpolates in
`log10(1/eps)` and has separate charge-only and dipole/combined order
sets. The tight-tolerance dipole endpoint keeps the previous `pdmk4`
plane-wave order needed by the `eps = 1e-12` validation case.
Non-3D-Laplace kernels keep the previous `pdmk4` order choices.

`src/pdmk/pdmk_local.f` adds Fortran wrappers for coefficient-driven
3D Laplace local kernels and the PSWF coefficient initialization
routines used by the dipole path.

`vec-kernels/include/l3d_laplace_dipole_kernel.hpp` adds the vectorized
3D Laplace local dipole kernels. The C ABI declarations and wrappers
are wired through `vec-kernels/include/kernels.h` and
`vec-kernels/src/libkernels2.cpp`, matching the current public
`dmk4_makefile` build path.

`test/pdmk/test_l3d_laplace_dipole_pdmk.f` validates dipole-only and
charge-plus-dipole cases by comparing `pdmk4` output against direct
evaluation. `dmk4_makefile` adds the `test-dipole` target.

## Validation

From the repository root:

```bash
make -f dmk4_makefile test-dipole
```

This uses the same compiler and BLAS settings as `dmk4_makefile`.
Override them with `make.inc` if needed for a local compiler setup.
