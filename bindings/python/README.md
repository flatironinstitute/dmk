# DMK Python bindings

A single-module `ctypes` binding for the DMK C API (`include/dmk.h`). There is nothing
to compile: the module loads a `libdmk` shared library that you built separately.

## Pointing it at a build

`DMK_LIB` overrides the library name, and may be a full path:

```sh
DMK_LIB=/path/to/build/lib/libdmk.so python -c "import dmk; print(dmk.pdmk_version_string())"
```

Unset, the module loads `libdmk` (then `libdmk.so`, or `libdmk.dylib` on macOS) through the
dynamic loader, so `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` works too. This is the same
resolution the Julia binding uses.

Install with `pip install /path/to/DMK/bindings/python`, or just put the directory on
`PYTHONPATH`.

## Communicators

`comm` is DMK's `dmk_communicator`, i.e. an `MPI_Comm`. `None` means "self" and works against
either build: an MPI-enabled library starts MPI itself on the first call if nobody has, and
finalizes it at exit. To use MPI yourself, initialize it before the first DMK call (importing
`mpi4py.MPI` does this) and pass the communicator's handle value, `MPI.COMM_WORLD.handle`
(`MPI._handleof(MPI.COMM_WORLD)` on mpi4py < 4).

## Usage

```python
import numpy as np
import dmk

n = 10000
r_src = np.random.rand(3 * n)
charge = 2 * np.random.rand(n) - 1
pot = np.zeros(n)

params = dmk.pdmk_init_default_params()
params.n_dim = 3
params.kernel = dmk.Kernel.LAPLACE
params.eps = 1e-6

with dmk.tree_handle(dmk.pdmk_tree_create(None, params, n, r_src, charge, None, 0, None)) as tree:
    dmk.pdmk_tree_eval(tree, pot, None)
    charge[:] = 2 * np.random.rand(n) - 1
    dmk.pdmk_tree_update_charges(tree, charge, None)
    dmk.pdmk_tree_eval(tree, pot, None)
```

`dmk.pdmk(...)` does the same in one call, and `dmk.pdmk_direct(...)` computes the same sum
by brute force as a reference. Functions ending in `f` are the float32 entry points and take
float32 buffers.

Arrays are passed straight to C, so every buffer must be C-contiguous, writable, and of the
precision the entry point expects (float64, or float32 for the `*f` variants). numpy is not
required -- an `array.array` or a ctypes array works as well.

## Device arrays

With `params.eval_path = dmk.EvalPath.GPU` and `params.memory_space = dmk.MemorySpace.DEVICE`,
the tree entry points and `pdmk`/`pdmkf` take device arrays on `params.gpu_device_id` instead:
any C-contiguous array exporting `__cuda_array_interface__` (CuPy, Numba, PyTorch, JAX). All
arrays of such a call are then device arrays, and a tree keeps its memory space for
`pdmk_tree_eval` and `pdmk_tree_update_charges`. `pdmk_direct` and ESP take host arrays only.

```python
import cupy as cp

params.eval_path = dmk.EvalPath.GPU
params.memory_space = dmk.MemorySpace.DEVICE
r_src = cp.random.rand(3 * n)
charge = 2 * cp.random.rand(n) - 1
pot = cp.zeros(n)
dmk.pdmk(None, params, n, r_src, charge, None, 0, None, pot, None)
```

DMK runs on its own CUDA streams. Before the call the binding synchronizes the stream each array
exports, so CuPy and Numba arrays need nothing more. PyTorch exports no stream: synchronize it
first with `torch.cuda.current_stream().synchronize()`, and pass a tensor that requires grad as
`.detach()`.

Every call raises `dmk.DmkError` on failure, carrying the library's own error detail.
