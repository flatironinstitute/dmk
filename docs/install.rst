Installation
============

DMK requires a C++20 compiler, CMake (>= 3.20), a BLAS implementation, and optionally an MPI
implementation and/or a CUDA implementation. It is built with CMake and the git submodules in
``extern/``, so clone recursively.

Dependencies
------------

- **BLAS** (required): linear algebra backend.
- **MPI** (optional, ``DMK_HAVE_MPI``, default ON): distributed parallelism.
- **OpenMP** (optional, ``DMK_HAVE_OPENMP``, default ON): shared-memory parallelism.
- **LLVM** (optional, ``DMK_USE_JIT``, default OFF): JIT-compiled kernels; RuFuS targets
  LLVM 19.
- **CUDA** (optional, ``DMK_GPU_OFFLOAD``, default OFF): GPU based ``pdmk`` tree evals,
  ``pdmk`` charge updates, and free space direct sums.

FINUFFT (fetched automatically) provides the FFTs and spread/interp algorithms used by the ESP
periodic solver.

MPI is part of the ABI
~~~~~~~~~~~~~~~~~~~~~~

With ``DMK_HAVE_MPI=ON``, the public header ``dmk.h`` includes ``<mpi.h>`` and
``dmk_communicator`` is ``MPI_Comm``; the installed CMake target propagates ``DMK_HAVE_MPI`` to
consumers, so they see the same declarations. An installed DMK is therefore bound to the MPI
implementation it was built against, and no MPI implementation or version is recorded in the
installed files. Consumers must compile against that same MPI, and a package that ships DMK must
pin it as a dependency. Build with ``-DDMK_HAVE_MPI=OFF`` for a library with no MPI in its
interface.

Building on Flatiron Institute resources
----------------------------------------

.. code-block:: bash

   module load gcc openmpi intel-oneapi-mkl flexiblas

   git clone git@github.com:flatironinstitute/DMK --recursive
   cd DMK
   mkdir build
   cd build

   cmake .. -DCMAKE_BUILD_TYPE=relwithdebinfo -DBLA_VENDOR=FlexiBLAS -DCMAKE_CXX_FLAGS="-march=x86-64-v4"
   make -j 10

Optionally, ``-DDMK_USE_JIT=ON`` enables runtime JIT-generated short-range kernels. This
requires ``LLVM >= 19``. JIT is mostly used as a developer option to make shifting the
``beta(kernel, eps, eval_type)`` curves without having to pay to rebuild the entire direct
evaluator tree. Mild performance bumps are also possible, especially when running on multiple
different hardware.

Building on macOS
-----------------

An OpenMP-capable compiler is required, which the default macOS Clang is not. The build has
been verified with Homebrew's LLVM Clang. Update the LLVM path to wherever Homebrew installed
it.

.. code-block:: bash

   brew install open-mpi llvm openblas cmake
   LLVM_ROOT=/opt/homebrew/Cellar/llvm/21.1.8
   git clone git@github.com:flatironinstitute/dmk --recursive
   mkdir dmk/build
   cd dmk/build
   cmake .. -DCMAKE_CXX_COMPILER="$LLVM_ROOT/bin/clang++" \
     -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
     -DCMAKE_EXE_LINKER_FLAGS="-L$LLVM_ROOT/lib/c++ -Wl,-rpath,$LLVM_ROOT/lib/c++" \
     -DCMAKE_SHARED_LINKER_FLAGS="-L$LLVM_ROOT/lib/c++ -Wl,-rpath,$LLVM_ROOT/lib/c++" \
     -DCMAKE_CXX_FLAGS=-Wno-deprecated -DCMAKE_INSTALL_PREFIX=$PWD/install
   make -j 12

Running the tests
-----------------

Tests are off by default; add ``-DDMK_BUILD_TESTS=ON`` at configure time to build them. The
suite is registered with CTest as a single test named ``test_all``, which is the simplest way to
run it. From the build directory:

.. code-block:: bash

   ctest --output-on-failure

CTest runs the binary under ``mpirun -n 2`` when the build has MPI and invokes it directly
otherwise, and sets ``DMK_JIT_AUTOTUNE_DISABLE=1`` for the run.

The binary can also be run by hand, which is how to pass doctest options such as a test-case
filter or a different rank count:

.. code-block:: bash

   DMK_JIT_AUTOTUNE_DISABLE=1 ./test/test_all # run all tests
   DMK_JIT_AUTOTUNE_DISABLE=1 ./test/test_all -tc='*[DMK]*' # one subset
   DMK_JIT_AUTOTUNE_DISABLE=1 mpirun -np 4 ./test/test_all # MPI-enabled build

``DMK_JIT_AUTOTUNE_DISABLE=1`` matters for GPU builds: the GPU path uses a JIT+autotuning
pipeline that is expensive (~30s) on first call to any device/kernel/precision/eval_type
combination.
