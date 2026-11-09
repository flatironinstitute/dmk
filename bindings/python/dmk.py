"""ctypes binding for the DMK C API.

Point sets are passed as flat, C-contiguous, writable buffers of float64 (the plain
entry points) or float32 (the ``*f`` entry points) -- a numpy array, an ``array.array``
or a ctypes array all work. Positions are interleaved, ``[x0, y0, z0, x1, ...]``, and
particles lie in the unit box [0, 1)^n_dim.

With ``params.memory_space = MemorySpace.DEVICE`` the tree entry points and pdmk/pdmkf take
device arrays instead: any C-contiguous object exporting ``__cuda_array_interface__`` (CuPy,
Numba, PyTorch, JAX, ...). The stream an array exports is synchronized before the call;
PyTorch exports none, so synchronize its stream first
(``torch.cuda.current_stream().synchronize()``).
"""

import ctypes
import os
import sys
from contextlib import contextmanager
from enum import IntEnum, IntFlag


def _load_library():
    """Load libdmk, honouring a DMK_LIB override (a path or a bare name).

    A bare name is resolved by the dynamic loader, i.e. through LD_LIBRARY_PATH,
    DYLD_LIBRARY_PATH or an rpath, as in the Julia binding.
    """
    name = os.environ.get("DMK_LIB", "libdmk")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    candidates = [name] if name.endswith(suffix) else [name, name + suffix]
    details = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            details.append(str(exc))
    raise OSError("could not load libdmk (set DMK_LIB to its path): " + "; ".join(details))


_lib = _load_library()


class Kernel(IntEnum):
    YUKAWA = 0
    LAPLACE = 1
    SQRT_LAPLACE = 2
    STOKESLET = 3
    STRESSLET = 4
    LAPLACE_DIPOLE = 5


class EvalType(IntEnum):
    POTENTIAL = 1
    POTENTIAL_GRAD = 2
    VELOCITY = 3


class EvalPath(IntEnum):
    CPU = 0
    GPU = 1


class MemorySpace(IntEnum):
    HOST = 0
    DEVICE = 1


class ErrorCode(IntEnum):
    SUCCESS = 0
    INVALID_ARGUMENT = 1
    INTERNAL = 2


class LogLevel(IntEnum):
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERR = 4
    CRITICAL = 5
    OFF = 6


class DebugFlags(IntFlag):
    OMIT_PW = 1 << 0
    OMIT_DIRECT = 1 << 1
    DUMP_TREE = 1 << 2
    FORCE_AOT = 1 << 3
    OVERRIDE_BETA = 1 << 4
    OVERRIDE_ORDER = 1 << 5
    USE_PQ = 1 << 6


DEBUG_BETA_SLOT = 0
DEBUG_ORDER_SLOT = 1


class EspFlags(IntFlag):
    PRUNE_TILE = 1 << 0
    PRUNE_SOURCE = 1 << 1
    N3L = 1 << 2
    MORTON = 1 << 3


class PdmkParams(ctypes.Structure):
    _fields_ = [
        ("n_dim", ctypes.c_int),
        ("eps", ctypes.c_double),
        ("kernel", ctypes.c_int),
        ("eval_src", ctypes.c_int),
        ("eval_trg", ctypes.c_int),
        ("fparam", ctypes.c_double),
        ("use_periodic", ctypes.c_int),
        ("n_per_leaf", ctypes.c_int),
        ("log_level", ctypes.c_int),
        ("debug_flags", ctypes.c_uint32),
        ("debug_params", ctypes.c_double * 8),
        ("eval_path", ctypes.c_int),
        ("gpu_device_id", ctypes.c_int),
        ("memory_space", ctypes.c_int),
    ]


class PdmkEspParams(ctypes.Structure):
    """Parameters for the ESP (Ewald Sum with PSWF kernels) path, an experimental feature.

    Defaults are those documented in dmk.h; there is no C-side initializer.
    """

    _fields_ = [
        ("r_c", ctypes.c_double),
        ("eps", ctypes.c_double),
        ("log_level", ctypes.c_int),
        ("kernel", ctypes.c_int),
        ("fparam", ctypes.c_double),
        ("n_dim", ctypes.c_int),
        ("eval_type", ctypes.c_int),
        ("sigma", ctypes.c_double),
        ("use_periodic", ctypes.c_int),
        ("freespace_pad", ctypes.c_double),
        ("esp_flags", ctypes.c_uint32),
        ("esp_bins", ctypes.c_int),
        ("esp_stile", ctypes.c_int),
        ("eval_path", ctypes.c_int),
        ("gpu_device_id", ctypes.c_int),
    ]

    def __init__(self, **kwargs):
        super().__init__(
            r_c=0.05,
            eps=1e-6,
            log_level=LogLevel.OFF,
            kernel=Kernel.LAPLACE,
            fparam=0.0,
            n_dim=3,
            eval_type=EvalType.POTENTIAL,
            sigma=1.35,
            use_periodic=1,
            freespace_pad=0.0,
            esp_flags=EspFlags.PRUNE_SOURCE | EspFlags.N3L | EspFlags.MORTON,
            esp_bins=2,
            esp_stile=0,
            eval_path=EvalPath.CPU,
            gpu_device_id=0,
        )
        for key, value in kwargs.items():
            setattr(self, key, value)


# Both structs are passed to C by value, so a field reorder or type change in dmk.h
# would silently corrupt every call.
assert ctypes.sizeof(PdmkParams) == 136
assert ctypes.sizeof(PdmkEspParams) == 88


class DmkError(RuntimeError):
    pass


# In MPI builds dmk takes an MPI_Comm, which is a pointer under OpenMPI and an int under
# MPICH; either fits c_void_p, so pass the integer handle from mpi4py (e.g.
# MPI.COMM_WORLD.handle). None means "self" in either build.
_Comm = ctypes.c_void_p
_Tree = ctypes.c_void_p
_Plan = ctypes.c_void_p
_Dbl = ctypes.POINTER(ctypes.c_double)
_Flt = ctypes.POINTER(ctypes.c_float)
_Err = ctypes.c_int

_lib.pdmk_version_string.argtypes = []
_lib.pdmk_version_string.restype = ctypes.c_char_p
_lib.pdmk_git_commit.argtypes = []
_lib.pdmk_git_commit.restype = ctypes.c_char_p
_lib.pdmk_version.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3
_lib.pdmk_version.restype = None
_lib.pdmk_last_error_message.argtypes = []
_lib.pdmk_last_error_message.restype = ctypes.c_char_p
_lib.pdmk_init_default_params.argtypes = [ctypes.POINTER(PdmkParams)]
_lib.pdmk_init_default_params.restype = None

_lib.pdmk_tree_create.argtypes = [_Comm, PdmkParams, ctypes.c_int, _Dbl, _Dbl, _Dbl, ctypes.c_int, _Dbl]
_lib.pdmk_tree_create.restype = _Tree
_lib.pdmk_tree_createf.argtypes = [_Comm, PdmkParams, ctypes.c_int, _Flt, _Flt, _Flt, ctypes.c_int, _Flt]
_lib.pdmk_tree_createf.restype = _Tree
_lib.pdmk_tree_eval.argtypes = [_Tree, _Dbl, _Dbl]
_lib.pdmk_tree_eval.restype = _Err
_lib.pdmk_tree_evalf.argtypes = [_Tree, _Flt, _Flt]
_lib.pdmk_tree_evalf.restype = _Err
_lib.pdmk_tree_update_charges.argtypes = [_Tree, _Dbl, _Dbl]
_lib.pdmk_tree_update_charges.restype = _Err
_lib.pdmk_tree_update_chargesf.argtypes = [_Tree, _Flt, _Flt]
_lib.pdmk_tree_update_chargesf.restype = _Err
_lib.pdmk_tree_destroy.argtypes = [_Tree]
_lib.pdmk_tree_destroy.restype = None

_lib.pdmk.argtypes = [_Comm, PdmkParams, ctypes.c_int, _Dbl, _Dbl, _Dbl, ctypes.c_int, _Dbl, _Dbl, _Dbl]
_lib.pdmk.restype = _Err
_lib.pdmkf.argtypes = [_Comm, PdmkParams, ctypes.c_int, _Flt, _Flt, _Flt, ctypes.c_int, _Flt, _Flt, _Flt]
_lib.pdmkf.restype = _Err
_lib.pdmk_direct.argtypes = _lib.pdmk.argtypes
_lib.pdmk_direct.restype = _Err
_lib.pdmk_directf.argtypes = _lib.pdmkf.argtypes
_lib.pdmk_directf.restype = _Err

_lib.pdmk_esp_plan_create.argtypes = [_Comm, PdmkEspParams]
_lib.pdmk_esp_plan_create.restype = _Plan
_lib.pdmk_esp_plan_createf.argtypes = [_Comm, PdmkEspParams]
_lib.pdmk_esp_plan_createf.restype = _Plan
_lib.pdmk_esp_eval.argtypes = [_Comm, _Plan, ctypes.c_int, _Dbl, _Dbl, _Dbl, _Dbl]
_lib.pdmk_esp_eval.restype = _Err
_lib.pdmk_esp_evalf.argtypes = [_Comm, _Plan, ctypes.c_int, _Flt, _Flt, _Flt, _Flt]
_lib.pdmk_esp_evalf.restype = _Err
_lib.pdmk_esp_plan_destroy.argtypes = [_Plan]
_lib.pdmk_esp_plan_destroy.restype = None
_lib.pdmk_esp_plan_destroyf.argtypes = [_Plan]
_lib.pdmk_esp_plan_destroyf.restype = None
_lib.pdmk_esp.argtypes = [_Comm, PdmkEspParams, ctypes.c_int, _Dbl, _Dbl, _Dbl, _Dbl]
_lib.pdmk_esp.restype = _Err
_lib.pdmk_espf.argtypes = [_Comm, PdmkEspParams, ctypes.c_int, _Flt, _Flt, _Flt, _Flt]
_lib.pdmk_espf.restype = _Err

_lib.pdmk_print_profile_data.argtypes = [_Comm, ctypes.c_char]
_lib.pdmk_print_profile_data.restype = _Err


_cuda = None


def _stream_synchronize(stream):
    """cuStreamSynchronize through the driver, as Numba does when it consumes the interface;
    its stream values 1 and 2 are the driver's CU_STREAM_LEGACY and CU_STREAM_PER_THREAD."""
    global _cuda
    if _cuda is None:
        _cuda = ctypes.CDLL("libcuda.so.1")
        _cuda.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
        _cuda.cuStreamSynchronize.restype = ctypes.c_int
    rc = _cuda.cuStreamSynchronize(stream)
    if rc != 0:
        raise DmkError(f"cuStreamSynchronize failed (CUresult {rc})")


def _c_contiguous(shape, strides, itemsize):
    expected = itemsize
    for n, stride in zip(reversed(shape), reversed(strides)):
        if n > 1 and stride != expected:
            return False
        expected *= n
    return True


def _device_ptr(buf, fmt, output):
    """Pointer into a __cuda_array_interface__ array. DMK runs on its own streams, so the stream
    the array exports is synchronized first, as the interface asks of a consumer."""
    cai = buf.__cuda_array_interface__
    typestr, itemsize = ("<f8", 8) if fmt == "d" else ("<f4", 4)
    if cai["typestr"] != typestr:
        raise TypeError(f"expected a '{typestr}' device array, got '{cai['typestr']}'")
    if cai.get("mask") is not None:
        raise TypeError("masked device arrays are not supported")
    strides = cai.get("strides")
    if strides is not None and not _c_contiguous(cai["shape"], strides, itemsize):
        raise TypeError("device arrays must be C-contiguous")
    ptr, readonly = cai["data"]
    if output and readonly:
        raise TypeError("device output array is read-only")
    if cai.get("stream") is not None:
        _stream_synchronize(cai["stream"])
    return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double if fmt == "d" else ctypes.c_float))


def _ptr(buf, fmt, space=MemorySpace.HOST, output=False):
    """Pointer to the first element of `buf`, or NULL for None. The C side does not check
    pointers, so `buf` must live in `space`: a device array for DEVICE, a host buffer for HOST."""
    if buf is None:
        return None
    if hasattr(buf, "__cuda_array_interface__"):
        if space != MemorySpace.DEVICE:
            raise TypeError("device array passed for host memory")
        return _device_ptr(buf, fmt, output)
    if space != MemorySpace.HOST:
        raise TypeError("host buffer passed for device memory")
    view = memoryview(buf)
    # from_buffer below only checks the byte length, so without this a float32 buffer
    # would reach a float64 entry point as half as many elements.
    if view.format != fmt:
        raise TypeError(f"expected a '{fmt}' buffer, got '{view.format}'")
    ctype = ctypes.c_double if fmt == "d" else ctypes.c_float
    return (ctype * (view.nbytes // view.itemsize)).from_buffer(buf)


def pdmk_last_error_message():
    """Detail for the most recent failing call on this thread."""
    return (_lib.pdmk_last_error_message() or b"").decode()


def _check(rc, name):
    if rc != ErrorCode.SUCCESS:
        raise DmkError(f"{name} failed (code {rc}): {pdmk_last_error_message()}")


def _check_handle(handle, name):
    if not handle:
        raise DmkError(f"{name} failed: {pdmk_last_error_message()}")
    return handle


def _require_outputs(name, n_src, pot_src, n_trg, pot_trg):
    """The tree path rejects a NULL output for a non-empty point set; only
    pdmk_direct reads NULL as "skip this point set"."""
    if n_src > 0 and pot_src is None:
        raise DmkError(f"{name}: pot_src is required when n_src > 0")
    if n_trg > 0 and pot_trg is None:
        raise DmkError(f"{name}: pot_trg is required when n_trg > 0")


def pdmk_version_string():
    return _lib.pdmk_version_string().decode()


def pdmk_version():
    major, minor, patch = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    _lib.pdmk_version(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
    return major.value, minor.value, patch.value


def pdmk_git_commit():
    """"unknown" when the library was built outside a git repository."""
    return _lib.pdmk_git_commit().decode()


def pdmk_init_default_params():
    """A PdmkParams filled with the library defaults."""
    params = PdmkParams()
    _lib.pdmk_init_default_params(ctypes.byref(params))
    return params


# The memory_space of each live tree, which eval and update_charges check their arrays against
_tree_space = {}


def pdmk_tree_create(comm, params, n_src, r_src, charge, normal, n_trg, r_trg):
    space = params.memory_space
    tree = _check_handle(
        _lib.pdmk_tree_create(comm, params, n_src, _ptr(r_src, "d", space), _ptr(charge, "d", space),
                              _ptr(normal, "d", space), n_trg, _ptr(r_trg, "d", space)),
        "pdmk_tree_create")
    _tree_space[tree] = space
    return tree


def pdmk_tree_createf(comm, params, n_src, r_src, charge, normal, n_trg, r_trg):
    space = params.memory_space
    tree = _check_handle(
        _lib.pdmk_tree_createf(comm, params, n_src, _ptr(r_src, "f", space), _ptr(charge, "f", space),
                               _ptr(normal, "f", space), n_trg, _ptr(r_trg, "f", space)),
        "pdmk_tree_createf")
    _tree_space[tree] = space
    return tree


def pdmk_tree_eval(tree, pot_src, pot_trg):
    """None is accepted only for a point set that is empty; with points present the
    library returns DMK_ERR_INVALID_ARGUMENT."""
    space = _tree_space[tree]
    _check(_lib.pdmk_tree_eval(tree, _ptr(pot_src, "d", space, True), _ptr(pot_trg, "d", space, True)),
           "pdmk_tree_eval")


def pdmk_tree_evalf(tree, pot_src, pot_trg):
    space = _tree_space[tree]
    _check(_lib.pdmk_tree_evalf(tree, _ptr(pot_src, "f", space, True), _ptr(pot_trg, "f", space, True)),
           "pdmk_tree_evalf")


def pdmk_tree_update_charges(tree, charge, normal):
    space = _tree_space[tree]
    _check(_lib.pdmk_tree_update_charges(tree, _ptr(charge, "d", space), _ptr(normal, "d", space)),
           "pdmk_tree_update_charges")


def pdmk_tree_update_chargesf(tree, charge, normal):
    space = _tree_space[tree]
    _check(_lib.pdmk_tree_update_chargesf(tree, _ptr(charge, "f", space), _ptr(normal, "f", space)),
           "pdmk_tree_update_chargesf")


def pdmk_tree_destroy(tree):
    del _tree_space[tree]
    _lib.pdmk_tree_destroy(tree)


@contextmanager
def tree_handle(tree):
    """Destroy a tree on exit: `with tree_handle(pdmk_tree_create(...)) as tree:`."""
    try:
        yield tree
    finally:
        pdmk_tree_destroy(tree)


def pdmk(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg):
    _require_outputs("pdmk", n_src, pot_src, n_trg, pot_trg)
    space = params.memory_space
    _check(_lib.pdmk(comm, params, n_src, _ptr(r_src, "d", space), _ptr(charge, "d", space),
                     _ptr(normal, "d", space), n_trg, _ptr(r_trg, "d", space),
                     _ptr(pot_src, "d", space, True), _ptr(pot_trg, "d", space, True)), "pdmk")


def pdmkf(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg):
    _require_outputs("pdmkf", n_src, pot_src, n_trg, pot_trg)
    space = params.memory_space
    _check(_lib.pdmkf(comm, params, n_src, _ptr(r_src, "f", space), _ptr(charge, "f", space),
                      _ptr(normal, "f", space), n_trg, _ptr(r_trg, "f", space),
                      _ptr(pot_src, "f", space, True), _ptr(pot_trg, "f", space, True)), "pdmkf")


def pdmk_direct(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg):
    """Brute-force reference for pdmk, O(n_src * n_trg). Either output may be None to
    skip evaluating at that point set."""
    _check(_lib.pdmk_direct(comm, params, n_src, _ptr(r_src, "d"), _ptr(charge, "d"), _ptr(normal, "d"),
                            n_trg, _ptr(r_trg, "d"), _ptr(pot_src, "d"), _ptr(pot_trg, "d")), "pdmk_direct")


def pdmk_directf(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg):
    _check(_lib.pdmk_directf(comm, params, n_src, _ptr(r_src, "f"), _ptr(charge, "f"), _ptr(normal, "f"),
                             n_trg, _ptr(r_trg, "f"), _ptr(pot_src, "f"), _ptr(pot_trg, "f")), "pdmk_directf")


def pdmk_esp_plan_create(comm, params):
    """Experimental."""
    return _check_handle(_lib.pdmk_esp_plan_create(comm, params), "pdmk_esp_plan_create")


def pdmk_esp_plan_createf(comm, params):
    """Experimental."""
    return _check_handle(_lib.pdmk_esp_plan_createf(comm, params), "pdmk_esp_plan_createf")


def pdmk_esp_eval(comm, plan, n, r_src, charges, normal, pot_src):
    """Experimental. `normal` is required by the Stresslet and ignored (may be None)
    otherwise; pot_src is written interleaved per source."""
    _check(_lib.pdmk_esp_eval(comm, plan, n, _ptr(r_src, "d"), _ptr(charges, "d"),
                              _ptr(normal, "d"), _ptr(pot_src, "d")), "pdmk_esp_eval")


def pdmk_esp_evalf(comm, plan, n, r_src, charges, normal, pot_src):
    """Experimental."""
    _check(_lib.pdmk_esp_evalf(comm, plan, n, _ptr(r_src, "f"), _ptr(charges, "f"),
                               _ptr(normal, "f"), _ptr(pot_src, "f")), "pdmk_esp_evalf")


def pdmk_esp_plan_destroy(plan):
    _lib.pdmk_esp_plan_destroy(plan)


def pdmk_esp_plan_destroyf(plan):
    _lib.pdmk_esp_plan_destroyf(plan)


def pdmk_esp(comm, params, n, r_src, charges, normal, pot_src):
    """Experimental. One-shot ESP evaluation; creates and destroys a plan internally."""
    _check(_lib.pdmk_esp(comm, params, n, _ptr(r_src, "d"), _ptr(charges, "d"),
                         _ptr(normal, "d"), _ptr(pot_src, "d")), "pdmk_esp")


def pdmk_espf(comm, params, n, r_src, charges, normal, pot_src):
    """Experimental."""
    _check(_lib.pdmk_espf(comm, params, n, _ptr(r_src, "f"), _ptr(charges, "f"),
                          _ptr(normal, "f"), _ptr(pot_src, "f")), "pdmk_espf")


def pdmk_print_profile_data(comm, type=b"t"):
    """Print profiler output: b"t" for a table, b"h" for CSV headers, b"c" for a CSV
    row (which also resets the counters)."""
    _check(_lib.pdmk_print_profile_data(comm, type), "pdmk_print_profile_data")
