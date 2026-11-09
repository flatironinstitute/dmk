module DMK

import LinearAlgebra
import OpenBLAS32_jll
using DMK_jll: libdmk

function __init__()
    # UCX otherwise claims SIGSEGV, which Julia's GC uses. MPI.jl sets this before MPI_Init, but
    # a serial session has no MPI.jl, and libdmk then starts MPI itself on the first call.
    get!(ENV, "UCX_ERROR_SIGNALS", "SIGILL,SIGBUS,SIGFPE")
    # libdmk calls the LP64 BLAS names; Julia's own OpenBLAS fills only libblastrampoline's ILP64 slot.
    config = LinearAlgebra.BLAS.lbt_get_config()
    if !any(lib -> lib.interface == :lp64, config.loaded_libs)
        LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
    end
    _check_layout(PdmkParams, _PDMK_PARAMS_LAYOUT)
    _check_layout(PdmkEspParams, _PDMK_ESP_PARAMS_LAYOUT)
end

# === Enums (mirror dmk.h) ===

@enum DmkIkernel::Cint begin
    DMK_YUKAWA         = 0
    DMK_LAPLACE        = 1
    DMK_SQRT_LAPLACE   = 2
    DMK_STOKESLET      = 3
    DMK_STRESSLET      = 4
    DMK_LAPLACE_DIPOLE = 5
end

@enum DmkEvalType::Cint begin
    DMK_POTENTIAL      = 1
    DMK_POTENTIAL_GRAD = 2
    DMK_VELOCITY       = 3
end

@enum DmkEvalPath::Cint begin
    DMK_EVAL_PATH_CPU = 0
    DMK_EVAL_PATH_GPU = 1
end

@enum DmkMemorySpace::Cint begin
    DMK_MEMORY_HOST   = 0
    DMK_MEMORY_DEVICE = 1
end

@enum DmkLogLevel::Cint begin
    DMK_LOG_TRACE    = 0
    DMK_LOG_DEBUG    = 1
    DMK_LOG_INFO     = 2
    DMK_LOG_WARN     = 3
    DMK_LOG_ERR      = 4
    DMK_LOG_CRITICAL = 5
    DMK_LOG_OFF      = 6
end

@enum DmkError::Cint begin
    DMK_SUCCESS              = 0
    DMK_ERR_INVALID_ARGUMENT = 1
    DMK_ERR_INTERNAL         = 2
end

# debug_flags bitfield
const DMK_DEBUG_OMIT_PW        = UInt32(1) << 0
const DMK_DEBUG_OMIT_DIRECT    = UInt32(1) << 1
const DMK_DEBUG_DUMP_TREE      = UInt32(1) << 2
const DMK_DEBUG_FORCE_AOT      = UInt32(1) << 3
const DMK_DEBUG_OVERRIDE_BETA  = UInt32(1) << 4
const DMK_DEBUG_OVERRIDE_ORDER = UInt32(1) << 5
const DMK_DEBUG_USE_PQ         = UInt32(1) << 6

const DMK_DEBUG_BETA_SLOT  = 0
const DMK_DEBUG_ORDER_SLOT = 1

# ESP short-range method selection bits for pdmk_esp_params.esp_flags
const DMK_ESP_PRUNE_TILE   = UInt32(1) << 0
const DMK_ESP_PRUNE_SOURCE = UInt32(1) << 1
const DMK_ESP_N3L          = UInt32(1) << 2
const DMK_ESP_MORTON       = UInt32(1) << 3

# === Opaque handles ===

# Opaque tree handle returned by pdmk_tree_create*.
const PdmkTree = Ptr{Cvoid}

# Communicator handle. In MPI builds dmk takes an MPI_Comm; with OpenMPI that's
# a pointer (use MPI.COMM_WORLD.val from MPI.jl). With MPICH/Intel MPI it's a
# Cint and you must reinterpret. C_NULL means "self" in either build.
const DmkCommunicator = Ptr{Cvoid}

# === pdmk_params ===

mutable struct PdmkParams
    n_dim::Cint
    eps::Cdouble
    kernel::DmkIkernel
    eval_src::DmkEvalType
    eval_trg::DmkEvalType
    fparam::Cdouble
    use_periodic::Cint
    n_per_leaf::Cint
    log_level::Cint
    debug_flags::UInt32
    debug_params::NTuple{8,Cdouble}
    eval_path::DmkEvalPath
    gpu_device_id::Cint
    memory_space::DmkMemorySpace
end

# Allocate and fill from the C-side defaults.
function PdmkParams()
    p = PdmkParams(Cint(0), 0.0,
                   DMK_YUKAWA, DMK_POTENTIAL, DMK_POTENTIAL,
                   0.0, Cint(0), Cint(0), Cint(0),
                   UInt32(0), ntuple(_ -> 0.0, 8), DMK_EVAL_PATH_CPU, Cint(0), DMK_MEMORY_HOST)
    pdmk_init_default_params(p)
    return p
end

# === pdmk_esp_params ===

# Particles lie in the unit box [0, 1)^n_dim, as on the tree path. Field order/types mirror
# pdmk_esp_params in dmk.h; the struct is passed to ESP calls by value.
mutable struct PdmkEspParams
    r_c::Cdouble
    eps::Cdouble
    log_level::Cint
    kernel::DmkIkernel
    fparam::Cdouble
    n_dim::Cint
    eval_type::DmkEvalType
    sigma::Cdouble
    use_periodic::Cint
    freespace_pad::Cdouble
    esp_flags::UInt32
    esp_bins::Cint
    esp_stile::Cint
    eval_path::DmkEvalPath
    gpu_device_id::Cint
end

# Defaults from dmk.h (there is no C-side init function for ESP params).
PdmkEspParams() = PdmkEspParams(
    0.05, 1e-6, Cint(6), DMK_LAPLACE, 0.0, Cint(3), DMK_POTENTIAL,
    1.35, Cint(1), 0.0,
    DMK_ESP_PRUNE_SOURCE | DMK_ESP_N3L | DMK_ESP_MORTON, Cint(2), Cint(0),
    DMK_EVAL_PATH_CPU, Cint(0))

# Opaque ESP plan handle returned by pdmk_esp_plan_create*.
const PdmkEspPlan = Ptr{Cvoid}

# === Struct layout guard ===

# Both param structs are passed to C by value, so a field reorder, insertion or type
# change in dmk.h would silently corrupt every call. These are the C sizes and field
# offsets (x86-64 / AArch64 LP64); __init__ checks the Julia mirrors against them.
const _PDMK_PARAMS_LAYOUT = (136, (0, 8, 16, 20, 24, 32, 40, 44, 48, 52, 56, 120, 124, 128))
const _PDMK_ESP_PARAMS_LAYOUT = (88, (0, 8, 16, 20, 24, 32, 36, 40, 48, 56, 64, 68, 72, 76, 80))

function _check_layout(T, layout)
    nbytes, offsets = layout
    fieldcount(T) == length(offsets) ||
        error("$T mirrors dmk.h incorrectly: $(fieldcount(T)) fields, expected $(length(offsets))")
    sizeof(T) == nbytes ||
        error("$T mirrors dmk.h incorrectly: sizeof is $(sizeof(T)), expected $nbytes")
    for (i, offset) in enumerate(offsets)
        fieldoffset(T, i) == offset ||
            error("$T.$(fieldname(T, i)) mirrors dmk.h incorrectly: " *
                  "offset is $(fieldoffset(T, i)), expected $offset")
    end
end

# === ccall wrappers ===

# Pointer argument for every array: host Arrays here and device arrays through ext/DMKCUDAExt.jl, as MPI.jl's
# MPIPtr does. ccall keeps the cconvert result alive for the duration of the call.
primitive type DmkPtr Sys.WORD_SIZE end
Base.cconvert(::Type{DmkPtr}, x::Array) = x
Base.unsafe_convert(::Type{DmkPtr}, x::Array) = reinterpret(DmkPtr, pointer(x))
Base.cconvert(::Type{DmkPtr}, ::Nothing) = reinterpret(DmkPtr, C_NULL)

# The C side does not check pointers, so the tree entry points check each array against the memory space the call
# runs in. The CUDA extension adds device arrays, which it also synchronizes with the caller's stream.
_check_array(::PdmkParams, ::Nothing) = nothing
function _check_array(params::PdmkParams, ::Array)
    params.memory_space == DMK_MEMORY_HOST || error("host array passed with memory_space $(params.memory_space)")
    return nothing
end
_check_arrays(params::PdmkParams, arrays...) = foreach(a -> _check_array(params, a), arrays)

# The params of each live tree, which eval and update_charges check their arrays against
const _tree_params = Dict{PdmkTree,PdmkParams}()
const _tree_params_lock = ReentrantLock()
_tree_params_of(tree::PdmkTree) = @lock _tree_params_lock _tree_params[tree]

pdmk_version_string() = unsafe_string(ccall((:pdmk_version_string, libdmk), Cstring, ()))

# "unknown" when the library was built outside a git repository.
pdmk_git_commit() = unsafe_string(ccall((:pdmk_git_commit, libdmk), Cstring, ()))

function pdmk_version()
    major, minor, patch = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
    ccall((:pdmk_version, libdmk), Cvoid, (Ref{Cint}, Ref{Cint}, Ref{Cint}), major, minor, patch)
    return (Int(major[]), Int(minor[]), Int(patch[]))
end

# Detail for the most recent failing call on this thread.
function pdmk_last_error_message()
    s = ccall((:pdmk_last_error_message, libdmk), Cstring, ())
    return s == C_NULL ? "" : unsafe_string(s)
end

# Throw a Julia exception carrying the C-side message unless rc is DMK_SUCCESS.
function _check(rc::DmkError)
    rc == DMK_SUCCESS && return
    error("DMK error ($rc): " * pdmk_last_error_message())
end

function pdmk_init_default_params(params::PdmkParams)
    ccall((:pdmk_init_default_params, libdmk), Cvoid, (Ref{PdmkParams},), params)
    return params
end

function pdmk_tree_create(comm::DmkCommunicator, params::PdmkParams,
                          n_src::Integer, r_src::DenseVector{Float64},
                          charge::DenseVector{Float64},
                          normal::Union{DenseVector{Float64},Nothing},
                          n_trg::Integer,
                          r_trg::Union{DenseVector{Float64},Nothing})
    _check_arrays(params, r_src, charge, normal, r_trg)
    tree = ccall((:pdmk_tree_create, libdmk), PdmkTree,
                 (DmkCommunicator, PdmkParams, Cint, DmkPtr, DmkPtr, DmkPtr, Cint, DmkPtr),
                 comm, params, n_src, r_src, charge, normal, n_trg, r_trg)
    tree == C_NULL && error("pdmk_tree_create failed: " * pdmk_last_error_message())
    @lock _tree_params_lock _tree_params[tree] = deepcopy(params)
    return tree
end

function pdmk_tree_createf(comm::DmkCommunicator, params::PdmkParams,
                           n_src::Integer, r_src::DenseVector{Float32},
                           charge::DenseVector{Float32},
                           normal::Union{DenseVector{Float32},Nothing},
                           n_trg::Integer,
                           r_trg::Union{DenseVector{Float32},Nothing})
    _check_arrays(params, r_src, charge, normal, r_trg)
    tree = ccall((:pdmk_tree_createf, libdmk), PdmkTree,
                 (DmkCommunicator, PdmkParams, Cint, DmkPtr, DmkPtr, DmkPtr, Cint, DmkPtr),
                 comm, params, n_src, r_src, charge, normal, n_trg, r_trg)
    tree == C_NULL && error("pdmk_tree_createf failed: " * pdmk_last_error_message())
    @lock _tree_params_lock _tree_params[tree] = deepcopy(params)
    return tree
end

# The tree path only accepts a `nothing` output for a point set that is empty; with
# points present C returns DMK_ERR_INVALID_ARGUMENT.
function pdmk_tree_eval(tree::PdmkTree,
                        pot_src::Union{DenseVector{Float64},Nothing},
                        pot_trg::Union{DenseVector{Float64},Nothing})
    _check_arrays(_tree_params_of(tree), pot_src, pot_trg)
    _check(ccall((:pdmk_tree_eval, libdmk), DmkError, (PdmkTree, DmkPtr, DmkPtr), tree, pot_src, pot_trg))
end

function pdmk_tree_evalf(tree::PdmkTree,
                         pot_src::Union{DenseVector{Float32},Nothing},
                         pot_trg::Union{DenseVector{Float32},Nothing})
    _check_arrays(_tree_params_of(tree), pot_src, pot_trg)
    _check(ccall((:pdmk_tree_evalf, libdmk), DmkError, (PdmkTree, DmkPtr, DmkPtr), tree, pot_src, pot_trg))
end

function pdmk_tree_update_charges(tree::PdmkTree, charge::DenseVector{Float64},
                                  normal::Union{DenseVector{Float64},Nothing})
    _check_arrays(_tree_params_of(tree), charge, normal)
    _check(ccall((:pdmk_tree_update_charges, libdmk), DmkError, (PdmkTree, DmkPtr, DmkPtr), tree, charge, normal))
end

function pdmk_tree_update_chargesf(tree::PdmkTree, charge::DenseVector{Float32},
                                   normal::Union{DenseVector{Float32},Nothing})
    _check_arrays(_tree_params_of(tree), charge, normal)
    _check(ccall((:pdmk_tree_update_chargesf, libdmk), DmkError, (PdmkTree, DmkPtr, DmkPtr), tree, charge, normal))
end

function pdmk_tree_destroy(tree::PdmkTree)
    @lock _tree_params_lock pop!(_tree_params, tree)
    ccall((:pdmk_tree_destroy, libdmk), Cvoid, (PdmkTree,), tree)
end

function pdmk_print_profile_data(comm::DmkCommunicator, type::Char)
    _check(ccall((:pdmk_print_profile_data, libdmk), DmkError,
                 (DmkCommunicator, Cchar), comm, Cchar(type)))
end

# ESP calls have no error return; plan_create* signals failure by returning NULL.
function pdmk_esp_plan_create(comm::DmkCommunicator, params::PdmkEspParams)
    plan = ccall((:pdmk_esp_plan_create, libdmk), PdmkEspPlan,
                 (DmkCommunicator, PdmkEspParams), comm, params)
    plan == C_NULL && error("pdmk_esp_plan_create failed: " * pdmk_last_error_message())
    return plan
end

function pdmk_esp_plan_createf(comm::DmkCommunicator, params::PdmkEspParams)
    plan = ccall((:pdmk_esp_plan_createf, libdmk), PdmkEspPlan,
                 (DmkCommunicator, PdmkEspParams), comm, params)
    plan == C_NULL && error("pdmk_esp_plan_createf failed: " * pdmk_last_error_message())
    return plan
end

function pdmk_esp_eval(comm::DmkCommunicator, plan::PdmkEspPlan, n::Integer,
                       r_src::Vector{Float64}, charges::Vector{Float64},
                       normal::Union{Vector{Float64},Nothing},
                       pot_src::Vector{Float64})
    _check(ccall((:pdmk_esp_eval, libdmk), DmkError,
                 (DmkCommunicator, PdmkEspPlan, Cint, DmkPtr, DmkPtr, DmkPtr, DmkPtr),
                 comm, plan, n, r_src, charges, normal, pot_src))
end

function pdmk_esp_evalf(comm::DmkCommunicator, plan::PdmkEspPlan, n::Integer,
                        r_src::Vector{Float32}, charges::Vector{Float32},
                        normal::Union{Vector{Float32},Nothing},
                        pot_src::Vector{Float32})
    _check(ccall((:pdmk_esp_evalf, libdmk), DmkError,
                 (DmkCommunicator, PdmkEspPlan, Cint, DmkPtr, DmkPtr, DmkPtr, DmkPtr),
                 comm, plan, n, r_src, charges, normal, pot_src))
end

function pdmk_esp_plan_destroy(plan::PdmkEspPlan)
    ccall((:pdmk_esp_plan_destroy, libdmk), Cvoid, (PdmkEspPlan,), plan)
end

function pdmk_esp_plan_destroyf(plan::PdmkEspPlan)
    ccall((:pdmk_esp_plan_destroyf, libdmk), Cvoid, (PdmkEspPlan,), plan)
end

function pdmk_esp(comm::DmkCommunicator, params::PdmkEspParams, n::Integer,
                  r_src::Vector{Float64}, charges::Vector{Float64},
                  normal::Union{Vector{Float64},Nothing},
                  pot_src::Vector{Float64})
    _check(ccall((:pdmk_esp, libdmk), DmkError,
                 (DmkCommunicator, PdmkEspParams, Cint, DmkPtr, DmkPtr, DmkPtr, DmkPtr),
                 comm, params, n, r_src, charges, normal, pot_src))
end

function pdmk_espf(comm::DmkCommunicator, params::PdmkEspParams, n::Integer,
                   r_src::Vector{Float32}, charges::Vector{Float32},
                   normal::Union{Vector{Float32},Nothing},
                   pot_src::Vector{Float32})
    _check(ccall((:pdmk_espf, libdmk), DmkError,
                 (DmkCommunicator, PdmkEspParams, Cint, DmkPtr, DmkPtr, DmkPtr, DmkPtr),
                 comm, params, n, r_src, charges, normal, pot_src))
end

function _require_outputs(fn, n_src, pot_src, n_trg, pot_trg)
    n_src > 0 && pot_src === nothing && error("$fn: pot_src is required when n_src > 0")
    n_trg > 0 && pot_trg === nothing && error("$fn: pot_trg is required when n_trg > 0")
end

function pdmk(comm::DmkCommunicator, params::PdmkParams, n_src::Integer,
              r_src::DenseVector{Float64}, charge::DenseVector{Float64},
              normal::Union{DenseVector{Float64},Nothing}, n_trg::Integer,
              r_trg::Union{DenseVector{Float64},Nothing},
              pot_src::Union{DenseVector{Float64},Nothing},
              pot_trg::Union{DenseVector{Float64},Nothing})
    _require_outputs("pdmk", n_src, pot_src, n_trg, pot_trg)
    _check_arrays(params, r_src, charge, normal, r_trg, pot_src, pot_trg)
    _check(ccall((:pdmk, libdmk), DmkError,
                 (DmkCommunicator, PdmkParams, Cint, DmkPtr, DmkPtr, DmkPtr, Cint, DmkPtr, DmkPtr, DmkPtr),
                 comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg))
end

function pdmkf(comm::DmkCommunicator, params::PdmkParams, n_src::Integer,
               r_src::DenseVector{Float32}, charge::DenseVector{Float32},
               normal::Union{DenseVector{Float32},Nothing}, n_trg::Integer,
               r_trg::Union{DenseVector{Float32},Nothing},
               pot_src::Union{DenseVector{Float32},Nothing},
               pot_trg::Union{DenseVector{Float32},Nothing})
    _require_outputs("pdmkf", n_src, pot_src, n_trg, pot_trg)
    _check_arrays(params, r_src, charge, normal, r_trg, pot_src, pot_trg)
    _check(ccall((:pdmkf, libdmk), DmkError,
                 (DmkCommunicator, PdmkParams, Cint, DmkPtr, DmkPtr, DmkPtr, Cint, DmkPtr, DmkPtr, DmkPtr),
                 comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg))
end

function pdmk_direct(comm::DmkCommunicator, params::PdmkParams, n_src::Integer,
                     r_src::Vector{Float64}, charge::Vector{Float64},
                     normal::Union{Vector{Float64},Nothing}, n_trg::Integer,
                     r_trg::Union{Vector{Float64},Nothing},
                     pot_src::Union{Vector{Float64},Nothing},
                     pot_trg::Union{Vector{Float64},Nothing})
    _check(ccall((:pdmk_direct, libdmk), DmkError,
                 (DmkCommunicator, PdmkParams, Cint, DmkPtr, DmkPtr, DmkPtr, Cint, DmkPtr, DmkPtr, DmkPtr),
                 comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg))
end

function pdmk_directf(comm::DmkCommunicator, params::PdmkParams, n_src::Integer,
                      r_src::Vector{Float32}, charge::Vector{Float32},
                      normal::Union{Vector{Float32},Nothing}, n_trg::Integer,
                      r_trg::Union{Vector{Float32},Nothing},
                      pot_src::Union{Vector{Float32},Nothing},
                      pot_trg::Union{Vector{Float32},Nothing})
    _check(ccall((:pdmk_directf, libdmk), DmkError,
                 (DmkCommunicator, PdmkParams, Cint, DmkPtr, DmkPtr, DmkPtr, Cint, DmkPtr, DmkPtr, DmkPtr),
                 comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg))
end

export PdmkParams, PdmkTree, PdmkEspParams, PdmkEspPlan, DmkCommunicator,
       DmkIkernel, DmkEvalType, DmkEvalPath, DmkMemorySpace, DmkLogLevel, DmkError,
       DMK_SUCCESS, DMK_ERR_INVALID_ARGUMENT, DMK_ERR_INTERNAL,
       pdmk_last_error_message,
       DMK_YUKAWA, DMK_LAPLACE, DMK_SQRT_LAPLACE, DMK_STOKESLET, DMK_STRESSLET,
       DMK_LAPLACE_DIPOLE,
       DMK_POTENTIAL, DMK_POTENTIAL_GRAD, DMK_VELOCITY,
       DMK_EVAL_PATH_CPU, DMK_EVAL_PATH_GPU,
       DMK_MEMORY_HOST, DMK_MEMORY_DEVICE,
       DMK_LOG_TRACE, DMK_LOG_DEBUG, DMK_LOG_INFO, DMK_LOG_WARN, DMK_LOG_ERR,
       DMK_LOG_CRITICAL, DMK_LOG_OFF,
       DMK_DEBUG_OMIT_PW, DMK_DEBUG_OMIT_DIRECT, DMK_DEBUG_DUMP_TREE,
       DMK_DEBUG_FORCE_AOT, DMK_DEBUG_OVERRIDE_BETA, DMK_DEBUG_OVERRIDE_ORDER,
       DMK_DEBUG_USE_PQ, DMK_DEBUG_BETA_SLOT, DMK_DEBUG_ORDER_SLOT,
       DMK_ESP_PRUNE_TILE, DMK_ESP_PRUNE_SOURCE, DMK_ESP_N3L, DMK_ESP_MORTON,
       pdmk_version_string, pdmk_version, pdmk_git_commit,
       pdmk_init_default_params,
       pdmk_tree_create, pdmk_tree_createf,
       pdmk_tree_eval, pdmk_tree_evalf,
       pdmk_tree_update_charges, pdmk_tree_update_chargesf,
       pdmk_tree_destroy, pdmk_print_profile_data,
       pdmk_esp_plan_create, pdmk_esp_plan_createf,
       pdmk_esp_eval, pdmk_esp_evalf,
       pdmk_esp_plan_destroy, pdmk_esp_plan_destroyf,
       pdmk_esp, pdmk_espf,
       pdmk, pdmkf,
       pdmk_direct, pdmk_directf

end # module DMK
