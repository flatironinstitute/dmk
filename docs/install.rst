Installation
============

DMK requires a C++20 compiler and CMake (>= 3.20). It is built with CMake and the git
submodules in ``extern/``, so clone recursively.

Dependencies
------------

- **BLAS** (required): linear algebra backend.
- **MPI** (optional, ``DMK_HAVE_MPI``, default ON): distributed parallelism.
- **OpenMP** (optional, ``DMK_HAVE_OPENMP``, default ON): shared-memory parallelism.
- **LLVM** (optional, ``DMK_USE_JIT``, default OFF): JIT-compiled kernels; RuFuS targets
  LLVM 19.

FINUFFT (fetched automatically) provides the FFTs used by the ESP periodic solver.

Building on Flatiron Institute resources
----------------------------------------

.. code-block:: bash

   module load modules/2.3 python gcc/13 openmpi intel-oneapi-mkl flexiblas

   git clone git@github.com:flatironinstitute/DMK --recursive
   cd DMK
   mkdir build
   cd build

   cmake .. -DCMAKE_BUILD_TYPE=relwithdebinfo -DBLA_VENDOR=FlexiBLAS
   make -j 10

Optionally, ``-DDMK_USE_JIT=ON`` enables runtime JIT-generated short-range kernels. This
requires LLVM; on FI systems, ``module load llvm/19.1.7``.

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

Tests are built by default (``DMK_BUILD_TESTS=ON``). From the build directory:

.. code-block:: bash

   ./test/test_all              # run all tests
   mpirun -np 4 ./test/test_all # MPI-enabled build
