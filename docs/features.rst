Supported features
==================

DMK supports the three major modes of operation:

- ``pdmk`` -- Point DMK -- A heavily optimized variant of the original tree algorithm from
  `Jiang and Greenguard <https://dx.doi.org/10.1002/cpa.22240>`_. Supports MPI+OpenMP (CPU
  only), pure OpenMP, and GPU evaluation. Tree builds are purely CPU based at the current time,
  though an all GPU implementation is in the works. The GPU implementation is accordingly
  optimally used in "build once eval many" contexts, such as in iterative solvers like GMRES.
- ``esp`` -- A standalone variant of the ``ESP`` (Ewald Summation with Prolate spheroidal wave
  functions) algorithm from `Liang, J., Lu, L., Barnett, A. et
  al. <https://doi.org/10.1038/s41467-026-73232-8>`_. OpenMP or GPU, with parity with the DMK
  tree variant. The GPU path covers every 3D kernel, periodic and free-space; 2D is CPU only.
  MPI+OpenMP hybrid is still a WIP.
- ``direct`` -- Reference free space direct sums. Available with GPU, OpenMP, and OpenMP+MPI.

See the following tables for full support tables.


Kernels
-------

The tree path (``pdmk``, ``pdmk_tree_create``) supports the following combinations. "Eval
types" is the base field ``eval_src`` and ``eval_trg`` may request for that kernel; they are
selected independently. "Gradient" means the kernel also implements ``DMK_POTENTIAL_GRAD``,
which returns the potential and its gradient together.

.. list-table::
   :header-rows: 1
   :widths: 24 8 8 18 14 14 14

   * - Kernel
     - 2D
     - 3D
     - Eval types
     - Gradient
     - Periodic
     - GPU
   * - ``DMK_LAPLACE``
     - yes
     - yes
     - potential
     - yes
     - 2D, 3D
     - 3D
   * - ``DMK_YUKAWA``
     - yes
     - yes
     - potential
     - yes
     - 2D, 3D
     - 3D
   * - ``DMK_SQRT_LAPLACE``
     - yes
     - yes
     - potential
     - yes
     - 2D, 3D
     - 3D
   * - ``DMK_LAPLACE_DIPOLE``
     - no
     - yes
     - potential
     - yes
     - no
     - 3D
   * - ``DMK_STOKESLET``
     - no
     - yes
     - velocity
     - no
     - no
     - 3D
   * - ``DMK_STRESSLET``
     - no
     - yes
     - velocity
     - no
     - no
     - 3D

The Stresslet requires a per-source orientation vector (``normal``, ``n_dim`` components per
source); a null pointer is rejected rather than dereferenced. Every other kernel ignores
``normal``, which may be ``NULL``.

Accuracy and precision
----------------------

Both ``float32`` and ``float64`` precisions are available throughout: every entry point has a
``float`` form suffixed with ``f``.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Precision
     - Useful ``eps`` range
     - Notes
   * - double
     - 1e-2 to 1e-12
     - Full range on both CPU and GPU, all kernels.
   * - float
     - 1e-2 to 1e-6
     - Max error sometimes large at high densities

DMK is tuned to meet the requested tolerance (relative L2 error) under most circumstances for
all supported kernels. As with other fast/approximate methods, there are edge cases that might
not meet the requested tolerance. Internal parameter curves are tuned against 200k points
randomly distributed on a D-dimensional sphere, and typically ``L2_error/eps ~= 0.5``.

Boundary conditions
-------------------

``use_periodic`` applies periodicity in every dimension, over the scalar kernels only (Laplace,
Yukawa, sqrt-Laplace): in 2D and 3D on the CPU, and in 3D on the GPU. The periodic root kernel
has no vector-valued forms currently, so a periodic Stokeslet, Stresslet or Laplace-dipole
request is rejected.

Free-space (non-periodic) evaluation is available for every supported kernel and dimension.

Compute paths
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - ``eval_path``
     - Capabilities
   * - ``DMK_EVAL_PATH_CPU``
     - Default. 2D and 3D, all kernels, MPI + OpenMP, periodic for the scalar kernels.
   * - ``DMK_EVAL_PATH_GPU``
     - Requires ``-DDMK_GPU_OFFLOAD=ON``. 3D only and single-rank only, so it cannot be
       combined with multi-rank MPI. Kernel/grad/periodicity support same as CPU.

``gpu_device_id`` selects the CUDA device. A process is pinned to the first device it uses:
the JIT module caches, autotune records and cached device properties are bound to that
device, so requesting a different one later fails rather than silently reusing modules
compiled for the first. Under MPI, use one device per rank.

Direct summation
----------------

``pdmk_direct`` evaluates the same sum by brute force, with no tree and no approximation, at
O(``n_src`` * ``n_trg``) cost. It exists as the reference for validating a tree solve.

It implements ``DMK_POTENTIAL``, ``DMK_POTENTIAL_GRAD`` and ``DMK_VELOCITY`` for every kernel
in the dimensions listed above, plus Laplace-dipole in 2D, which the tree path does not
support. There is no periodic implementation; ``use_periodic`` is rejected rather than
ignored. Under MPI the sources are gathered across the communicator, so results match a
single-rank run of the same problem.

``eval_path`` selects CPU or GPU, and ``gpu_device_id`` applies as it does for the tree. The
GPU direct kernel is a plain all-pairs sum rather than a plane-wave pipeline, so unlike
``pdmk`` it carries no 3D-only or single-rank restriction: every kernel and dimension above is
available on either path, in both precisions. Sources are gathered on the host first, so a
multi-rank run works with one device per rank.

Both paths evaluate the bare Green's function with correctly-rounded division and square root,
so they agree to round-off but not bit-for-bit -- the summation order differs. Validate against
a tolerance, not equality.

ESP: Ewald summation with prolates
----------------------------------

The ``pdmk_esp*`` entry points are an experimental standalone solver: Ewald summation with a
prolate spheroidal wave function window, over the unit box with either periodic or free-space
boundaries. It is experimental in the sense that matters for callers -- the ``pdmk_esp*`` API and
the ``pdmk_esp_params`` layout are not covered by the project's stability guarantees, and the
combinations below are what is implemented and tested rather than a target for completion.

ESP evaluates at the sources only: one point set, and one ``eval_type`` field rather than the
``eval_src`` / ``eval_trg`` pair of the tree path. Results are written into ``pot_src``
interleaved per source, ``[pot, d/dx, ...]`` for the potential kernels and ``[vx, vy, vz]`` for
the velocity kernels. Both precisions are available: ``pdmk_esp_plan_create`` / ``pdmk_esp_eval``
/ ``pdmk_esp_plan_destroy`` / ``pdmk_esp`` in double, and the ``f``-suffixed forms in float.
Kernel, dimension, evaluation type, boundary condition and accuracy are validated when the plan
is created, so an unsupported combination fails there rather than during evaluation.

.. list-table::
   :header-rows: 1
   :widths: 24 8 8 18 14 14 14

   * - Kernel
     - 2D
     - 3D
     - Eval types
     - Gradient
     - Periodic
     - GPU
   * - ``DMK_LAPLACE``
     - yes
     - yes
     - potential
     - yes
     - 2D, 3D
     - 3D
   * - ``DMK_YUKAWA``
     - yes
     - yes
     - potential
     - yes
     - 2D, 3D
     - 3D
   * - ``DMK_SQRT_LAPLACE``
     - yes
     - yes
     - potential
     - yes
     - 2D, 3D
     - 3D
   * - ``DMK_LAPLACE_DIPOLE``
     - no
     - yes
     - potential
     - yes
     - no
     - 3D
   * - ``DMK_STOKESLET``
     - no
     - yes
     - velocity
     - no
     - no
     - 3D
   * - ``DMK_STRESSLET``
     - no
     - yes
     - velocity
     - no
     - no
     - 3D

Laplace-dipole, Stokeslet and Stresslet are 3D free-space only: a 2D or ``use_periodic`` request
for them is rejected. The scalar kernels (Laplace, Yukawa, sqrt-Laplace) run periodic and
free-space in both 2D and 3D. As on the tree path, the Stresslet reads a per-source orientation
vector from ``normal``, and every other kernel ignores it.

``eval_path = DMK_EVAL_PATH_GPU`` requires ``-DDMK_GPU_OFFLOAD=ON`` and is 3D only, covering
every kernel, both boundary conditions and both precisions; 2D runs on the CPU, and a 2D GPU plan
is rejected. The ``comm`` argument is accepted for signature symmetry with the tree path but is
unused: each call evaluates the whole point set it is handed within one rank, threaded with
OpenMP or offloaded to one device.

``r_c`` is the real-space cutoff, at most 1/3 of the box so the short-range stencil stays within
the neighbouring cells, and ``sigma`` is the FINUFFT upsampling factor for the long-range grid.
Accuracy is set by ``eps``, but the two interact: a tight ``eps`` needs a wide PSWF spread, and a
request beyond FINUFFT's spread-width cap is rejected at plan creation rather than silently
clamped -- loosen ``eps``, raise ``sigma``, or use double precision.

``esp_flags`` selects the short-range strategy; the default combination is the fastest known one.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Effect
   * - ``DMK_ESP_PRUNE_TILE``
     - Sub-cell tile-vs-tile AABB pruning, with source-tile width ``esp_stile``.
   * - ``DMK_ESP_PRUNE_SOURCE``
     - Per-source point-vs-target-box pruning, the finest granularity. On the GPU it takes
       precedence over ``DMK_ESP_PRUNE_TILE``; on the CPU the two are independent.
   * - ``DMK_ESP_N3L``
     - Newton's-third-law reciprocal sum (13-forward half stencil, 27-coloured). CPU only.
   * - ``DMK_ESP_MORTON``
     - Morton within-cell sort, in place of an octant-bin counting sort with ``esp_bins`` bins
       per axis.

``DMK_ESP_N3L``, ``esp_bins`` and ``esp_stile`` are CPU-only; a GPU plan logs which of them it
ignores when it is created.

Parallelism
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Model
     - Notes
   * - OpenMP
     - ``DMK_HAVE_OPENMP``, on by default. Threads within a rank.
   * - MPI
     - ``DMK_HAVE_MPI``, on by default. Points are distributed across ranks; each rank
       passes its own slice and receives potentials for its own points. The GPU tree path is
       single-rank, so it cannot be combined with multi-rank MPI; the GPU direct path gathers
       sources on the host and runs under MPI with one device per rank. ESP ignores the
       communicator.

Platforms
---------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Platform
     - Status
     - Notes
   * - Linux x86-64
     - supported, CI
     - Primary supported platform. CI builds with GCC and with Clang.
   * - macOS (Apple silicon)
     - supported, no CI
     - Not covered by CI. Needs an OpenMP-capable compiler; the default Apple Clang is not.
       Verified with Homebrew LLVM (see :doc:`install`). CPU only.
   * - Windows
     - not supported
     -
   * - CUDA
     - supported, CI
     - Linux only. Requires ``-DDMK_GPU_OFFLOAD=ON``. The CI GPU job is a single-device
       build with MPI off.

Optional components
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Component
     - CMake option
     - Notes
   * - JIT kernels
     - ``DMK_USE_JIT``
     - Off by default. Compiles the short-range evaluator for the requested precision
       instead of using the pre-compiled AOT tables. Needs LLVM; RuFuS targets LLVM 19.
   * - ESP solver
     - always built
     - Experimental periodic and free-space electrostatics solver with its own API
       (``pdmk_esp*``); see the ESP table above and :doc:`api`.
   * - Instrumentation
     - ``DMK_INSTRUMENT``
     - Off by default. Enables SCTL profiler counters.
