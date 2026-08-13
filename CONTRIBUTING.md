# Contributing to DMK

Bug reports, questions and patches are all welcome. Please open an issue on
[GitHub](https://github.com/flatironinstitute/DMK) for anything you are unsure about before
investing much time in a change.

## Getting the source

DMK vendors its dependencies as submodules, so clone recursively:

```bash
git clone --recursive git@github.com:flatironinstitute/DMK
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

## Building and testing

See `docs/install.rst` for the full dependency list and platform notes. A development build on
Flatiron Institute resources:

```bash
module load modules/2.3 python gcc/13 openmpi intel-oneapi-mkl flexiblas
cmake -B build . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBLA_VENDOR=FlexiBLAS -DDMK_BUILD_TESTS=ON
cd build && make -j 10
```

Tests use [doctest](https://github.com/doctest/doctest) and need `-DDMK_BUILD_TESTS=ON`, as above:

```bash
./build/test/test_all                    # everything
./build/test/test_all -tc='*[DMK]*'      # one suite
mpirun -n 2 ./build/test/test_all -nc    # MPI build
```

Many test cases live next to the code they cover, inside `src/*.cpp`, guarded by the
`TEST_CASE_GENERIC` macro from `include/dmk/testing.hpp` (which maps to `MPI_TEST_CASE` when MPI is
enabled). Put a test wherever its subject lives rather than assuming it belongs in `test/`.

### Writing accuracy tests

DMK is an approximation with a requested tolerance, so an accuracy test needs a reference that does
*not* share the approximation:

- compare against `pdmk_direct` (brute-force summation), an analytic result, or a lattice/Ewald sum;
  or against a DMK solve at a far tighter tolerance (e.g. `eps = 1e-12` in double).
- do not compare a run against another run at the same `eps` — that mostly tests reproducibility.
- assert on the relative L2 error, not the pointwise maximum, which can be large wherever the
  potential passes near zero.
- use enough comparison targets for the norm to be meaningful. A few hundred points gives a noisy
  L2 estimate that can look like a tolerance violation when there is none.

`examples/measure_error` sweeps kernels, dimensions, precisions and digit counts against a direct
reference and is the quickest way to check a threshold before baking it into a test.

## Code style

C and C++ sources are formatted with `clang-format` using the checked-in `.clang-format` (LLVM
style, 4-space indent, 120-column limit). Enable the repository hook so staged files are formatted
automatically:

```bash
git config core.hooksPath scripts/hooks
```

Beyond formatting, the house style is: write the minimum code that satisfies the requirement, prefer
functions over classes and flat over nested, and do not add abstractions or error handling beyond
what is needed. Comments should explain why the code is the way it is; a comment describing what
changed relative to some earlier version is noise once the diff is merged.

## Generated sources

Some checked-in files are generated. Do not edit them by hand:

- `src/aot/aot_kernels_*.cpp` — ahead-of-time kernel instantiations, one translation unit per getter.
  After changing the kernel tables in `scripts/generate_aot_kernels.cpp`, regenerate from the project
  root with `./build/scripts/generate_aot_kernels`, and commit the result.
- `include/dmk/bessel.hpp` — produced by `scripts/gen_bessel.py`.

## Changing the C API

`include/dmk.h` is the public contract, along with the CMake package and `dmk.pc`. Within a major
version, changes must be additive: appending to `pdmk_params` or adding entry points is fine;
renaming, reordering or removing is not. Every new entry point should

- return `dmk_error` (or `NULL` for a `*_create`), and route failures through `dmk_guard` so no C++
  exception escapes;
- reject unsupported kernel/dimension/eval/path combinations with `DMK_ERR_INVALID_ARGUMENT` before
  doing any work, rather than throwing from deeper in the call stack; and
- take a `dmk_communicator` if it does any distributed work.

## Documentation

Sphinx sources are in `docs/`; see `docs/README.md` for building locally and for how Read the Docs
versions are wired up. If a change adds or restricts a supported combination, update the table in
`docs/features.rst` in the same pull request.

## Pull requests

Work on a branch and open a pull request against `main`. Please:

- keep the build warning-free and the test suite green (`mpirun -n 2 ./build/test/test_all -nc`);
- add a test for the behaviour you changed;
- add an entry under *Unreleased* in `CHANGELOG.md` for anything user-visible; and
- regenerate any generated sources your change affects.

CI runs on Jenkins (`ci/Jenkinsfile`) in a container built from `ci/Dockerfile`: a Linux CPU build
with g++ and MPI (tests run on 2 ranks), and a Linux GPU build on an A100 with
`-DDMK_GPU_OFFLOAD=on -DDMK_HAVE_MPI=off`. The GPU job sets `DMK_JIT_AUTOTUNE_DISABLE=1`; do the
same when running GPU code locally unless you are specifically measuring autotuned performance,
since autotuning dominates the run time of small problems.

## Releasing

`scripts/release.sh <major.minor.patch>` stamps `VERSION.txt`, syncs the binding manifests, commits
and tags `vX.Y.Z`. It does not push. Before tagging, retitle the *Unreleased* section of
`CHANGELOG.md` and make sure documentation changes are already on `main` — Read the Docs builds tags
as frozen snapshots, so a later docs fix does not apply to an existing tag.
