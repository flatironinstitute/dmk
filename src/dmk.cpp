#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/direct.hpp>
#include <dmk/error.hpp>
#include <dmk/esp.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/logger.h>
#include <dmk/prolate0_fun.hpp>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/tree.hpp>
#include <dmk/util.hpp>
#include <dmk/version.h>
#include <sctl.hpp>

#include <dmk/nvtx_wrapper.h>
#include <dmk/omp_wrapper.hpp>
#include <dmk/testing.hpp>

#ifdef DMK_GPU_OFFLOAD
#include <dmk/cuda/direct.hpp>
#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/pt/tree.hpp>
// GPU point-tree evaluators join the handle variant; selected at create when
// eval_path == DMK_EVAL_PATH_GPU.
using pdmk_tree_variant =
    std::variant<std::unique_ptr<dmk::DMKPtTree<float, 2>>, std::unique_ptr<dmk::DMKPtTree<float, 3>>,
                 std::unique_ptr<dmk::DMKPtTree<double, 2>>, std::unique_ptr<dmk::DMKPtTree<double, 3>>,
                 std::unique_ptr<dmk::cuda::pt::Tree<float, 2>>, std::unique_ptr<dmk::cuda::pt::Tree<float, 3>>,
                 std::unique_ptr<dmk::cuda::pt::Tree<double, 2>>, std::unique_ptr<dmk::cuda::pt::Tree<double, 3>>>;
#else
using pdmk_tree_variant =
    std::variant<std::unique_ptr<dmk::DMKPtTree<float, 2>>, std::unique_ptr<dmk::DMKPtTree<float, 3>>,
                 std::unique_ptr<dmk::DMKPtTree<double, 2>>, std::unique_ptr<dmk::DMKPtTree<double, 3>>>;
#endif

// The point counts travel with the tree because desort_potentials copies straight into the
// caller's arrays: eval needs them to tell a legitimately empty point set from a null buffer.
struct pdmk_tree_impl {
    pdmk_tree_variant tree;
    int n_src;
    int n_trg;
};

// GpuState is incomplete here, so the handle needs an explicit deleter.
#ifdef DMK_GPU_OFFLOAD
struct pdmk_esp_gpu_deleter {
    void operator()(dmk::GpuState *gpu) const { dmk::esp_destroy_gpu_plan(gpu); }
};
#endif

struct pdmk_esp_plan_impl {
    std::variant<std::unique_ptr<dmk::EspPlan<float>>, std::unique_ptr<dmk::EspPlan<double>>> plan;
#ifdef DMK_GPU_OFFLOAD
    std::unique_ptr<dmk::GpuState, pdmk_esp_gpu_deleter> gpu;
    int gpu_device_id = 0;
#endif
};

namespace dmk {

namespace {
std::string &last_error_buffer() {
    static thread_local std::string buf;
    return buf;
}
} // namespace

#ifdef DMK_HAVE_MPI
/// The C API documents a null communicator as "self". A null handle is not MPI_COMM_NULL -- in Open
/// MPI the latter is a real pointer to a sentinel object -- so both cases have to be tested.
inline MPI_Comm mpi_comm_or_self(dmk_communicator comm) {
    if constexpr (std::is_pointer_v<MPI_Comm>) {
        if (comm == MPI_Comm{})
            return MPI_COMM_SELF;
    }
    return comm == MPI_COMM_NULL ? MPI_COMM_SELF : comm;
}
#endif

void set_last_error(const std::string &msg) { last_error_buffer() = msg; }

const char *last_error_message() { return last_error_buffer().c_str(); }

/// Validate C API inputs before any object is constructed. Throws api_error
/// (DMK_ERR_INVALID_ARGUMENT) so the boundary guard converts to a clean code.
template <typename Real>
void validate_create_args(dmk_communicator comm, const pdmk_params &params, int n_src, const Real *r_src,
                          const Real *charge, const Real *normal, int n_trg, const Real *r_trg) {
    auto fail = [](std::string msg) { throw api_error(DMK_ERR_INVALID_ARGUMENT, std::move(msg)); };

    if (params.n_dim != 2 && params.n_dim != 3)
        fail("Invalid dimension: " + std::to_string(params.n_dim));
    if (params.eps > 1e-2 || params.eps < 1e-12)
        fail("tolerance 'eps' must lie on [1e-12, 1e-2], got " + std::to_string(params.eps));
    // Distances come from differencing coordinates held in Real, so no tolerance below its
    // epsilon is reachable at any tree depth, let alone the depths a real distribution forces.
    if (params.eps < std::numeric_limits<Real>::epsilon())
        fail("tolerance 'eps'=" + std::to_string(params.eps) + " is below the epsilon of the " +
             (sizeof(Real) == 4 ? std::string("single") : std::string("double")) +
             "-precision entry point; use the double-precision entry point or a looser tolerance");
    if (params.n_per_leaf <= 0)
        fail("n_per_leaf must be positive, got " + std::to_string(params.n_per_leaf));
    if (n_src < 0 || n_trg < 0)
        fail("n_src and n_trg must be non-negative");
    if (n_src == 0)
        fail("n_src is zero: nothing to do");
    if (params.kernel < DMK_YUKAWA || params.kernel > DMK_LAPLACE_DIPOLE)
        fail("Invalid kernel: " + std::to_string(int(params.kernel)));
    if (params.kernel == DMK_YUKAWA && params.fparam <= 0.0)
        fail("Invalid yukawa lambda. lambda must be positive, got " + std::to_string(params.fparam));
    if (params.eval_src < DMK_POTENTIAL || params.eval_src > DMK_VELOCITY)
        fail("Invalid eval_src: " + std::to_string(int(params.eval_src)));
    if (params.eval_trg < DMK_POTENTIAL || params.eval_trg > DMK_VELOCITY)
        fail("Invalid eval_trg: " + std::to_string(int(params.eval_trg)));

    // Stokeslet/Stresslet/Laplace-dipole only have 3D evaluators currently
    const bool needs_3d =
        params.kernel == DMK_STOKESLET || params.kernel == DMK_STRESSLET || params.kernel == DMK_LAPLACE_DIPOLE;
    if (needs_3d && params.n_dim != 3)
        fail("kernel " + std::string(util::to_string(params.kernel)) + " is only supported in 3D");

    // Rejected here so the caller gets DMK_ERR_INVALID_ARGUMENT rather than the
    // DMK_ERR_INTERNAL a deeper throw would produce.
    const bool scalar_kernel =
        params.kernel == DMK_LAPLACE || params.kernel == DMK_SQRT_LAPLACE || params.kernel == DMK_YUKAWA;
    if (params.use_periodic && !scalar_kernel)
        fail("periodic boundary conditions are not supported for kernel " +
             std::string(util::to_string(params.kernel)));

    if (params.eval_path == DMK_EVAL_PATH_GPU) {
#ifndef DMK_GPU_OFFLOAD
        fail("eval_path=GPU requires the library to be built with -DDMK_GPU_OFFLOAD=ON");
#else
        if (params.n_dim != 3)
            fail("eval_path=GPU is only supported in 3D (the plane-wave pipeline is 3D-only)");
#ifdef DMK_HAVE_MPI
        const int n_ranks = sctl::Comm(mpi_comm_or_self(comm)).Size();
        if (n_ranks > 1)
            fail("eval_path=GPU is single-rank only (the upward-pass broadcast has no device path), got " +
                 std::to_string(n_ranks) + " ranks");
#endif
        dmk::cuda::pt::bind_gpu_device(params.gpu_device_id);
#endif
    } else if (params.eval_path != DMK_EVAL_PATH_CPU) {
        fail("Invalid eval_path: " + std::to_string(int(params.eval_path)));
    }

    // Reject unsupported kernel/eval-type combinations
    try {
        get_kernel_output_dim(params.n_dim, params.kernel, params.eval_src);
        get_kernel_output_dim(params.n_dim, params.kernel, params.eval_trg);
    } catch (const std::exception &e) {
        fail(e.what());
    }

    if (r_src == nullptr || charge == nullptr)
        fail("r_src and charge must be non-null");
    if (n_trg > 0 && r_trg == nullptr)
        fail("r_trg must be non-null when n_trg > 0");
    // Stresslet reads a per-source normal in build_tree; a null pointer there is
    // an uncatchable segfault rather than an exception.
    if (params.kernel == DMK_STRESSLET && normal == nullptr)
        fail("Stresslet requires a non-null normal array");
}

/// The tree path has no null-output skip: desort_potentials copies into the caller's arrays
/// unconditionally, so a non-empty point set with a null buffer is a write through null.
void validate_eval_outputs(int n_src, const void *pot_src, int n_trg, const void *pot_trg) {
    if (n_src > 0 && pot_src == nullptr)
        throw api_error(DMK_ERR_INVALID_ARGUMENT, "pot_src must be non-null when n_src > 0");
    if (n_trg > 0 && pot_trg == nullptr)
        throw api_error(DMK_ERR_INVALID_ARGUMENT, "pot_trg must be non-null when n_trg > 0");
}

/// Validate ESP plan parameters. EspPlan, short_range and the GPU plan keep their own checks
/// (they are usable directly from C++); running these first means every C API caller is
/// rejected at plan creation with DMK_ERR_INVALID_ARGUMENT rather than at eval time.
void validate_esp_args(const pdmk_esp_params &params) {
    auto fail = [](std::string msg) { throw api_error(DMK_ERR_INVALID_ARGUMENT, std::move(msg)); };

    if (params.n_dim != 2 && params.n_dim != 3)
        fail("Invalid dimension: " + std::to_string(params.n_dim));
    if (params.kernel < DMK_YUKAWA || params.kernel > DMK_LAPLACE_DIPOLE)
        fail("Invalid kernel: " + std::to_string(int(params.kernel)));
    if (params.kernel == DMK_YUKAWA && params.fparam <= 0.0)
        fail("Invalid yukawa lambda. lambda must be positive, got " + std::to_string(params.fparam));
    if (params.eval_type < DMK_POTENTIAL || params.eval_type > DMK_VELOCITY)
        fail("Invalid eval_type: " + std::to_string(int(params.eval_type)));
    if (params.r_c <= 0.0)
        fail("r_c must be positive, got " + std::to_string(params.r_c));
    // Same test the cell lists apply: the 3^DIM-cell stencil needs at least 3 cells per axis,
    // or periodic images get counted twice.
    if (std::floor(1.0 / params.r_c) < 3.0)
        fail("r_c must be <= 1/3, got " + std::to_string(params.r_c));

    // Only the scalar kernels have a periodic ESP symbol, and only 3D evaluators exist for the rest.
    const bool scalar_kernel =
        params.kernel == DMK_YUKAWA || params.kernel == DMK_LAPLACE || params.kernel == DMK_SQRT_LAPLACE;
    if (!scalar_kernel && (params.n_dim != 3 || params.use_periodic))
        fail("ESP kernel " + std::string(util::to_string(params.kernel)) +
             " supports only 3D free-space (use_periodic=0)");

    if (params.eval_path == DMK_EVAL_PATH_GPU) {
#ifndef DMK_GPU_OFFLOAD
        fail("eval_path=GPU requires the library to be built with -DDMK_GPU_OFFLOAD=ON");
#else
        if (params.n_dim != 3)
            fail("ESP eval_path=GPU is only supported in 3D");
        dmk::cuda::pt::bind_gpu_device(params.gpu_device_id);
#endif
    } else if (params.eval_path != DMK_EVAL_PATH_CPU) {
        fail("Invalid eval_path: " + std::to_string(int(params.eval_path)));
    }

    try {
        get_kernel_output_dim(params.n_dim, params.kernel, params.eval_type);
    } catch (const std::exception &e) {
        fail(e.what());
    }
}

/// Validate inputs to the direct path. It shares pdmk's argument layout but none of its
/// tree parameters, and it has no periodic or GPU implementation, so the checks differ from
/// validate_create_args. An eval type is only checked when its output is actually requested:
/// a Stokeslet run that only fills pot_trg must not trip over the default eval_src.
template <typename Real>
void validate_direct_args(const pdmk_params &params, int n_src, const Real *r_src, const Real *charge,
                          const Real *normal, int n_trg, const Real *r_trg, const Real *pot_src, const Real *pot_trg) {
    auto fail = [](std::string msg) { throw api_error(DMK_ERR_INVALID_ARGUMENT, std::move(msg)); };

    if (params.n_dim != 2 && params.n_dim != 3)
        fail("Invalid dimension: " + std::to_string(params.n_dim));
    if (n_src < 0 || n_trg < 0)
        fail("n_src and n_trg must be non-negative");
    if (params.kernel < DMK_YUKAWA || params.kernel > DMK_LAPLACE_DIPOLE)
        fail("Invalid kernel: " + std::to_string(int(params.kernel)));
    if (params.kernel == DMK_YUKAWA && params.fparam <= 0.0)
        fail("Invalid yukawa lambda. lambda must be positive, got " + std::to_string(params.fparam));

    // Only the Stokes kernels are missing a 2D direct evaluator; the dipole has one even
    // though the tree path (validate_create_args) is 3D-only.
    const bool needs_3d = params.kernel == DMK_STOKESLET || params.kernel == DMK_STRESSLET;
    if (needs_3d && params.n_dim != 3)
        fail("kernel " + std::string(util::to_string(params.kernel)) + " is only supported in 3D");

    if (params.use_periodic)
        fail("the direct path has no periodic implementation; use_periodic must be 0");

    // An all-pairs sum has none of the tree path's 3D-only or single-rank restrictions: sources
    // are gathered on the host before any device work.
    if (params.eval_path == DMK_EVAL_PATH_GPU) {
#ifndef DMK_GPU_OFFLOAD
        fail("eval_path=GPU requires the library to be built with -DDMK_GPU_OFFLOAD=ON");
#else
        dmk::cuda::pt::bind_gpu_device(params.gpu_device_id);
#endif
    } else if (params.eval_path != DMK_EVAL_PATH_CPU) {
        fail("Invalid eval_path: " + std::to_string(int(params.eval_path)));
    }

    auto check_eval = [&](dmk_eval_type eval, const char *what) {
        if (eval < DMK_POTENTIAL || eval > DMK_VELOCITY)
            fail(std::string("Invalid ") + what + ": " + std::to_string(int(eval)));
        try {
            get_kernel_output_dim(params.n_dim, params.kernel, eval);
        } catch (const std::exception &e) {
            fail(e.what());
        }
    };
    if (pot_src)
        check_eval(params.eval_src, "eval_src");
    if (pot_trg)
        check_eval(params.eval_trg, "eval_trg");

    if (n_src > 0 && (r_src == nullptr || charge == nullptr))
        fail("r_src and charge must be non-null when n_src > 0");
    if (n_trg > 0 && pot_trg != nullptr && r_trg == nullptr)
        fail("r_trg must be non-null when n_trg > 0 and pot_trg is requested");
    if (params.kernel == DMK_STRESSLET && n_src > 0 && normal == nullptr)
        fail("Stresslet requires a non-null normal array");
}

/// Direct summation needs every source at every target, so each rank's source slice is
/// gathered onto all ranks; targets, and therefore the outputs, stay rank-local.
template <typename Real>
void pdmk_direct(dmk_communicator comm, const pdmk_params &params, int n_src, const Real *r_src, const Real *charge,
                 const Real *normal, int n_trg, const Real *r_trg, Real *pot_src, Real *pot_trg) {
    const int n_dim = params.n_dim;
    const int charge_dim = params.kernel == DMK_STRESSLET ? n_dim : get_kernel_input_dim(n_dim, params.kernel);
    const bool has_normal = params.kernel == DMK_STRESSLET;

    int n_src_global = n_src;
    const Real *r_global = r_src;
    const Real *charge_global = charge;
    const Real *normal_global = normal;
    std::vector<Real> r_gathered, charge_gathered, normal_gathered;

#ifdef DMK_HAVE_MPI
    const MPI_Comm mpi_comm = mpi_comm_or_self(comm);
    const int n_ranks = sctl::Comm(mpi_comm).Size();
    if (n_ranks > 1) {
        const MPI_Datatype mpi_t = std::is_same_v<Real, float> ? MPI_FLOAT : MPI_DOUBLE;
        std::vector<int> n_per_rank(n_ranks);
        MPI_Allgather(&n_src, 1, MPI_INT, n_per_rank.data(), 1, MPI_INT, mpi_comm);
        n_src_global = 0;
        for (int n : n_per_rank)
            n_src_global += n;

        auto gather = [&](const Real *local, int comp_dim, std::vector<Real> &out) {
            std::vector<int> counts(n_ranks), displs(n_ranks);
            for (int i = 0; i < n_ranks; ++i) {
                counts[i] = n_per_rank[i] * comp_dim;
                displs[i] = i ? displs[i - 1] + counts[i - 1] : 0;
            }
            out.resize(size_t(n_src_global) * comp_dim);
            MPI_Allgatherv(local, n_src * comp_dim, mpi_t, out.data(), counts.data(), displs.data(), mpi_t, mpi_comm);
        };
        gather(r_src, n_dim, r_gathered);
        gather(charge, charge_dim, charge_gathered);
        r_global = r_gathered.data();
        charge_global = charge_gathered.data();
        if (has_normal) {
            gather(normal, n_dim, normal_gathered);
            normal_global = normal_gathered.data();
        }
    }
#endif

    // The CPU evaluators accumulate, so the caller's buffer is cleared first; that also gives
    // ranks holding no global sources a well-defined (zero) result.
    auto eval_at = [&](dmk_eval_type eval, int n, const Real *r, Real *pot) {
        const int out_dim = get_kernel_output_dim(n_dim, params.kernel, eval);
        std::fill(pot, pot + size_t(n) * out_dim, Real(0));
        if (n_src_global == 0)
            return;
#ifdef DMK_GPU_OFFLOAD
        if (params.eval_path == DMK_EVAL_PATH_GPU) {
            cuda_helpers::ScopedDevice device_scope(params.gpu_device_id);
            cuda::direct_freespace<Real>(params, eval, n_src_global, r_global, charge_global, normal_global, n, r, pot);
            return;
        }
#endif
        const auto func = get_direct_evaluator<Real>(params.kernel, eval, n_dim, params.fparam);
        parallel_direct_eval(func, n_src_global, r_global, charge_global, normal_global, n, r, pot, n_dim, out_dim);
    };

    if (pot_src && n_src > 0)
        eval_at(params.eval_src, n_src, r_src, pot_src);
    if (pot_trg && n_trg > 0)
        eval_at(params.eval_trg, n_trg, r_trg, pot_trg);
}

template <typename T, int DIM>
void pdmk(dmk_communicator comm, const pdmk_params &params, int n_src, const T *r_src, const T *charge, const T *normal,
          int n_trg, const T *r_trg, T *pot_src, T *pot_trg) {
#ifdef DMK_HAVE_MPI
    const auto &sctl_comm = sctl::Comm(mpi_comm_or_self(comm));
#else
    const auto &sctl_comm = sctl::Comm().Self();
#endif
    auto &logger = dmk::get_logger(sctl_comm, params.log_level);
    auto &rank_logger = dmk::get_rank_logger(sctl_comm, params.log_level);
    logger->info("PDMK called");
    auto st = MY_OMP_GET_WTIME();

    const int kernel_input_dim = get_kernel_input_dim(params.n_dim, params.kernel);

    sctl::Vector<T> r_src_vec(n_src * params.n_dim, const_cast<T *>(r_src), false);
    sctl::Vector<T> r_trg_vec(n_trg * params.n_dim, const_cast<T *>(r_trg), false);
    sctl::Vector<T> charge_vec(n_src * kernel_input_dim, const_cast<T *>(charge), false);
    sctl::Vector<T> normal_vec(n_src * params.n_dim, const_cast<T *>(normal), false);

#ifdef DMK_GPU_OFFLOAD
    if (params.eval_path == DMK_EVAL_PATH_GPU) {
        cuda::pt::Tree<T, DIM> tree(sctl_comm, params, r_src_vec, charge_vec, normal_vec, r_trg_vec);
        tree.eval();
        tree.desort_potentials(pot_src, pot_trg);
    } else
#endif
    {
        DMKPtTree<T, DIM> tree(sctl_comm, params, r_src_vec, charge_vec, normal_vec, r_trg_vec);
        tree.eval();
        tree.desort_potentials(pot_src, pot_trg);
    }

    if (params.log_level <= DMK_LOG_INFO) {
        auto dt = MY_OMP_GET_WTIME() - st;
        int N = n_src + n_trg;
#ifdef DMK_HAVE_MPI
        if (sctl_comm.Rank() == 0)
            MPI_Reduce(MPI_IN_PLACE, &N, 1, MPI_INT, MPI_SUM, 0, mpi_comm_or_self(comm));
        else
            MPI_Reduce(&N, &N, 1, MPI_INT, MPI_SUM, 0, mpi_comm_or_self(comm));
#endif

        logger->info("PDMK finished in {:.4f} seconds ({:.0f} pts/s, {:.0f} pts/s/rank)", dt, N / dt,
                     N / dt / sctl_comm.Size());
    }
}

TEST_CASE_GENERIC("[DMK] pdmk 3d float", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 10000;
    constexpr int n_trg = 10000;
    constexpr int nd = 1;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    sctl::Vector<double> r_src, pot_src, charges, rnormal, pot_trg, r_trg;
    sctl::Vector<float> r_srcf, pot_srcf, chargesf, rnormalf, pot_trgf, r_trgf;
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, 0);
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_srcf, r_trgf, rnormalf, chargesf,
                              0);
    pot_src.ReInit(n_src * nd);
    pot_trg.ReInit(n_trg * nd);
    pot_srcf.ReInit(n_src * nd);
    pot_trgf.ReInit(n_trg * nd);

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 80;
    params.eval_src = DMK_POTENTIAL;
    params.eval_trg = DMK_POTENTIAL;
    params.kernel = DMK_YUKAWA;
    params.fparam = 6.0;
    params.log_level = SPDLOG_LEVEL_OFF;

    pdmk(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0], &pot_src[0], &pot_trg[0]);

    params.eps = 1e-3;
    pdmkf(comm, params, n_src, &r_srcf[0], &chargesf[0], &rnormalf[0], n_trg, &r_trgf[0], &pot_srcf[0], &pot_trgf[0]);

    double l2_err_src{0.0}, l2_err_trg{0.0}, src2{0.0}, trg2{0.0};
    for (int i = 0; i < n_src; ++i) {
        l2_err_src += pot_src[i] ? sctl::pow<2>(pot_src[i] - pot_srcf[i]) : 0.0;
        src2 += pot_src[i] * pot_src[i];
    }
    for (int i = 0; i < n_trg; ++i) {
        l2_err_trg += pot_trg[i] ? sctl::pow<2>(pot_trg[i] - pot_trgf[i]) : 0.0;
        trg2 += pot_trg[i] * pot_trg[i];
    }

    l2_err_src = std::sqrt(l2_err_src / src2);
    l2_err_trg = std::sqrt(l2_err_trg / trg2);
    CHECK(l2_err_src < params.eps);
    CHECK(l2_err_trg < params.eps);
}

TEST_CASE_GENERIC("[DMK] pdmk all", 1) {
    constexpr int n_src = 10000;
    constexpr int nd = 1;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    pdmk_params params;
    params.eps = 1e-6;
    params.eval_src = DMK_POTENTIAL;
    params.eval_trg = DMK_POTENTIAL;
    params.fparam = 6.0;
    params.log_level = SPDLOG_LEVEL_OFF;
    int ndiv[3] = {80, 280, 280};

    const auto test_kernels = {
        DMK_YUKAWA,
        DMK_LAPLACE,
        DMK_SQRT_LAPLACE,
    };

    for (auto n_dim : {2, 3}) {
        params.n_dim = n_dim;
        std::vector<double> r_src, pot_src, charges, rnormal, pot_trg, r_trg;
        dmk::util::init_test_data(n_dim, 1, n_src, 0, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, 0);
        r_trg = r_src;
        std::reverse(r_trg.begin(), r_trg.end());
        r_trg.resize(n_dim * (n_src - set_fixed_charges * 3));
        const int n_trg = r_trg.size() / n_dim;

        for (auto kernel : test_kernels) {
            const std::string kernel_str(util::to_string(kernel));

            SUBCASE((kernel_str + "_" + std::to_string(n_dim)).c_str()) {
                std::vector<double> pot_src(n_src * nd), grad_src(n_src * nd * n_dim),
                    hess_src(n_src * nd * n_dim * n_dim), pot_trg(n_src * nd), grad_trg(n_trg * nd * n_dim),
                    hess_trg(n_trg * nd * n_dim * n_dim);
                params.n_per_leaf = ndiv[int(kernel)];

                params.kernel = kernel;

                const int n_test_src = std::min(n_src, 1000);
                const int n_test_trg = std::min(n_trg, 1000);
                std::vector<double> test_src(n_test_src, 0);
                std::vector<double> test_trg(n_test_trg, 0);
                std::span<const double> r_src_trunc(r_src.data(), n_test_src * n_dim);
                std::span<const double> r_trg_trunc(r_trg.data(), n_test_trg * n_dim);

                compute_direct(n_dim, r_src, charges, std::vector<double>{}, r_src_trunc, test_src, kernel,
                               DMK_POTENTIAL);
                compute_direct(n_dim, r_src, charges, std::vector<double>{}, r_trg_trunc, test_trg, kernel,
                               DMK_POTENTIAL);

                pdmk_tree tree =
                    pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
                pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);

                double err_src{0}, err_trg{0};
                double ref_src{0}, ref_trg{0};
                for (int i = 0; i < n_test_src; ++i) {
                    err_src += sctl::pow<2>(test_src[i] - pot_src[i]);
                    ref_src += sctl::pow<2>(test_src[i]);
                }
                for (int i = 0; i < n_test_trg; ++i) {
                    err_trg += sctl::pow<2>(test_trg[i] - pot_trg[i]);
                    ref_trg += sctl::pow<2>(test_trg[i]);
                }

                err_src = std::sqrt(err_src / ref_src);
                err_trg = std::sqrt(err_trg / ref_trg);

                CHECK(err_src < params.eps);
                CHECK(err_trg < params.eps);

                // Scale charges by 1/2 and re-evaluate. Since the kernel
                // is linear in the charges, potentials should scale by the same factor.
                {
                    const double scale = 2.0;
                    sctl::Vector<double> scaled_charges(charges.size());
                    for (sctl::Long i = 0; i < charges.size(); ++i)
                        scaled_charges[i] = charges[i] * scale;

                    dmk_error rc = pdmk_tree_update_charges(tree, &scaled_charges[0], nullptr);
                    CHECK(rc == DMK_SUCCESS);

                    sctl::Vector<double> pot_src_updated(n_src * nd), pot_trg_updated(n_trg * nd);
                    pdmk_tree_eval(tree, &pot_src_updated[0], &pot_trg_updated[0]);

                    // Check that updated potentials ≈ scale * original potentials
                    double l2_err_src_update = 0.0;
                    double l2_ref_src = 0.0;
                    for (int i = 0; i < n_src; ++i) {
                        double expected = pot_src[i] * scale;
                        CHECK(std::abs(expected - pot_src_updated[i]) < 5 * std::numeric_limits<double>::epsilon());
                    }

                    for (int i = 0; i < n_test_trg; ++i) {
                        double expected = pot_trg[i] * scale;
                        CHECK(std::abs(expected - pot_trg_updated[i]) < 5 * std::numeric_limits<double>::epsilon());
                    }
                }

                pdmk_tree_destroy(tree);
            }
        }
    }
}

TEST_CASE_GENERIC("[DMK] pdmk all float", 1) {
    constexpr int n_src = 10000;
    constexpr int nd = 1;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;
    // eps asks for three digits; the tolerance allows twice that for fp32 round-off in the solve.
    constexpr double tol = 2e-3;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    pdmk_params params;
    params.eps = 1e-3;
    params.eval_src = DMK_POTENTIAL;
    params.eval_trg = DMK_POTENTIAL;
    params.fparam = 6.0;
    params.log_level = SPDLOG_LEVEL_OFF;
    int ndiv[3] = {80, 280, 280};

    const auto test_kernels = {
        DMK_YUKAWA,
        DMK_LAPLACE,
        DMK_SQRT_LAPLACE,
    };

    for (auto n_dim : {2, 3}) {
        params.n_dim = n_dim;
        std::vector<float> r_src, charges, rnormal, r_trg;
        dmk::util::init_test_data(n_dim, 1, n_src, 0, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, 0);
        r_trg = r_src;
        std::reverse(r_trg.begin(), r_trg.end());
        r_trg.resize(n_dim * (n_src - set_fixed_charges * 3));
        const int n_trg = r_trg.size() / n_dim;

        // The reference runs in double over the float-rounded geometry, so it is a higher-accuracy
        // answer to exactly the problem pdmkf is given rather than to a neighbouring one.
        const std::vector<double> r_src_ref(r_src.begin(), r_src.end());
        const std::vector<double> r_trg_ref(r_trg.begin(), r_trg.end());
        const std::vector<double> charges_ref(charges.begin(), charges.end());

        for (auto kernel : test_kernels) {
            const std::string kernel_str(util::to_string(kernel));

            SUBCASE((kernel_str + "_" + std::to_string(n_dim)).c_str()) {
                params.kernel = kernel;
                params.n_per_leaf = ndiv[int(kernel)];

                std::vector<float> pot_src(n_src * nd), pot_trg(n_trg * nd);

                const int n_test_src = std::min(n_src, 1000);
                const int n_test_trg = std::min(n_trg, 1000);
                std::vector<double> test_src, test_trg;
                std::span<const double> r_src_trunc(r_src_ref.data(), n_test_src * n_dim);
                std::span<const double> r_trg_trunc(r_trg_ref.data(), n_test_trg * n_dim);

                compute_direct(n_dim, r_src_ref, charges_ref, std::vector<double>{}, r_src_trunc, test_src, kernel,
                               DMK_POTENTIAL);
                compute_direct(n_dim, r_src_ref, charges_ref, std::vector<double>{}, r_trg_trunc, test_trg, kernel,
                               DMK_POTENTIAL);

                REQUIRE(pdmkf(comm, params, n_src, r_src.data(), charges.data(), rnormal.data(), n_trg, r_trg.data(),
                              pot_src.data(), pot_trg.data()) == DMK_SUCCESS);

                double err_src{0}, err_trg{0};
                double ref_src{0}, ref_trg{0};
                for (int i = 0; i < n_test_src; ++i) {
                    err_src += sctl::pow<2>(test_src[i] - pot_src[i]);
                    ref_src += sctl::pow<2>(test_src[i]);
                }
                for (int i = 0; i < n_test_trg; ++i) {
                    err_trg += sctl::pow<2>(test_trg[i] - pot_trg[i]);
                    ref_trg += sctl::pow<2>(test_trg[i]);
                }

                CHECK(std::sqrt(err_src / ref_src) < tol);
                CHECK(std::sqrt(err_trg / ref_trg) < tol);
            }
        }
    }
}

TEST_CASE_GENERIC("[DMK] pdmk 3d stokeslet velocity", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 2000;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;
    constexpr double thresh2 = 1e-30;
    constexpr int output_dim = n_dim;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    std::vector<double> r_src, charges, rnormal, r_trg;
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, 0);
    charges.resize(n_src * n_dim);
    for (auto &c : charges)
        c = 2 * drand48() - 1.0;

    std::vector<double> vel_src(n_src * output_dim, 0), vel_trg(n_trg * output_dim, 0);

    pdmk_params params;
    params.eps = 1e-3;
    params.n_dim = n_dim;
    params.n_per_leaf = 280;
    params.eval_src = DMK_VELOCITY;
    params.eval_trg = DMK_VELOCITY;
    params.kernel = DMK_STOKESLET;
    params.log_level = SPDLOG_LEVEL_OFF;
    params.debug_flags = 0;

    pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
    pdmk_tree_eval(tree, &vel_src[0], &vel_trg[0]);
    pdmk_tree_destroy(tree);

    const int n_test_src = std::min(n_src, 64);
    const int n_test_trg = std::min(n_trg, 64);
    std::vector<double> vel_src_direct, vel_trg_direct;

    compute_direct(n_dim, r_src, charges, std::vector<double>{}, std::span<double>(r_src.data(), n_test_src * n_dim),
                   vel_src_direct, params.kernel, params.eval_trg);
    compute_direct(n_dim, r_src, charges, std::vector<double>{}, std::span<double>(r_trg.data(), n_test_trg * n_dim),
                   vel_trg_direct, params.kernel, params.eval_trg);

    auto relative_l2_error = [](const auto &approx, const auto &exact) {
        double err2{0.0}, ref2{0.0};
        for (int i = 0; i < exact.size(); ++i) {
            err2 += sctl::pow<2>(approx[i] - exact[i]);
            ref2 += sctl::pow<2>(exact[i]);
        }
        return std::sqrt(err2 / ref2);
    };

    const double l2_err_src = relative_l2_error(vel_src, vel_src_direct);
    const double l2_err_trg = relative_l2_error(vel_trg, vel_trg_direct);

    CHECK(l2_err_src < params.eps);
    CHECK(l2_err_trg < params.eps);
}

TEST_CASE_GENERIC("[DMK] pdmk 3d stresslet velocity", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 2000;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;
    constexpr double thresh2 = 1e-30;
    constexpr int output_dim = n_dim;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    std::vector<double> r_src, charges, rnormal, r_trg;
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, 0);
    charges.resize(n_src * n_dim);
    for (auto &c : charges)
        c = 2 * drand48() - 1.0;

    std::vector<double> vel_src(n_src * output_dim, 0), vel_trg(n_trg * output_dim, 0);

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 280;
    params.eval_src = DMK_VELOCITY;
    params.eval_trg = DMK_VELOCITY;
    params.kernel = DMK_STRESSLET;
    params.log_level = SPDLOG_LEVEL_OFF;
    params.debug_flags = 0;
    params.use_periodic = false;

    pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
    pdmk_tree_eval(tree, &vel_src[0], &vel_trg[0]);
    pdmk_tree_destroy(tree);

    const int n_test_src = std::min(n_src, 64);
    const int n_test_trg = std::min(n_trg, 64);
    std::vector<double> vel_src_direct, vel_trg_direct;

    compute_direct(n_dim, r_src, charges, rnormal, std::span<double>(r_src.data(), n_test_src * n_dim), vel_src_direct,
                   params.kernel, params.eval_trg);
    compute_direct(n_dim, r_src, charges, rnormal, std::span<double>(r_trg.data(), n_test_trg * n_dim), vel_trg_direct,
                   params.kernel, params.eval_trg);

    auto relative_l2_error = [](const auto &approx, const auto &exact) {
        double err2{0.0}, ref2{0.0};
        for (int i = 0; i < exact.size(); ++i) {
            err2 += sctl::pow<2>(approx[i] - exact[i]);
            ref2 += sctl::pow<2>(exact[i]);
        }
        return std::sqrt(err2 / ref2);
    };

    const double l2_err_src = relative_l2_error(vel_src, vel_src_direct);
    const double l2_err_trg = relative_l2_error(vel_trg, vel_trg_direct);

    CHECK(l2_err_src < params.eps);
    CHECK(l2_err_trg < params.eps);
}

TEST_CASE_GENERIC("[DMK] pdmk 3d stresslet update_charges", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 2000;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;
    constexpr int output_dim = n_dim;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    std::vector<double> r_src, charges, rnormal, r_trg;
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, 0);
    charges.resize(n_src * n_dim);
    for (auto &c : charges)
        c = 2 * drand48() - 1.0;

    std::vector<double> vel_src(n_src * output_dim, 0), vel_trg(n_trg * output_dim, 0);

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 280;
    params.eval_src = DMK_VELOCITY;
    params.eval_trg = DMK_VELOCITY;
    params.kernel = DMK_STRESSLET;
    params.log_level = SPDLOG_LEVEL_OFF;

    pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
    REQUIRE(tree != nullptr);
    pdmk_tree_eval(tree, &vel_src[0], &vel_trg[0]);

    // Stresslet is linear in the density for a fixed normal: scaling the density
    // (passing the same normal) must scale the velocity by the same factor. This
    // exercises the stresslet branch of update_charges (charge .outer. normal
    // rebuild + halo broadcast).
    const double scale = 2.0;
    std::vector<double> scaled_charges(charges.size());
    for (size_t i = 0; i < charges.size(); ++i)
        scaled_charges[i] = charges[i] * scale;

    REQUIRE(pdmk_tree_update_charges(tree, scaled_charges.data(), rnormal.data()) == DMK_SUCCESS);

    std::vector<double> vel_src_updated(n_src * output_dim, 0), vel_trg_updated(n_trg * output_dim, 0);
    pdmk_tree_eval(tree, &vel_src_updated[0], &vel_trg_updated[0]);

    auto relative_l2_error = [](const auto &approx, const auto &exact) {
        double err2{0.0}, ref2{0.0};
        for (size_t i = 0; i < exact.size(); ++i) {
            err2 += sctl::pow<2>(approx[i] - exact[i]);
            ref2 += sctl::pow<2>(exact[i]);
        }
        return std::sqrt(err2 / ref2);
    };

    std::vector<double> vel_src_expected(vel_src.size()), vel_trg_expected(vel_trg.size());
    for (size_t i = 0; i < vel_src.size(); ++i)
        vel_src_expected[i] = vel_src[i] * scale;
    for (size_t i = 0; i < vel_trg.size(); ++i)
        vel_trg_expected[i] = vel_trg[i] * scale;

    CHECK(relative_l2_error(vel_src_updated, vel_src_expected) < 1e-12);
    CHECK(relative_l2_error(vel_trg_updated, vel_trg_expected) < 1e-12);

    // A stresslet charge update with a null normal must be rejected, not crash.
    CHECK(pdmk_tree_update_charges(tree, scaled_charges.data(), nullptr) == DMK_ERR_INVALID_ARGUMENT);

    pdmk_tree_destroy(tree);
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace gradient", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 4000;
    constexpr int n_trg = 3000;
    constexpr int nd = 1;
    constexpr bool uniform = false;
    constexpr bool set_fixed_charges = true;
    constexpr double thresh2 = 1e-30;
    constexpr int output_dim = 1 + n_dim;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    sctl::Vector<double> r_src, charges, rnormal, r_trg;
    dmk::util::init_test_data(n_dim, nd, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, charges, 0);

    sctl::Vector<double> pot_src(n_src * output_dim), pot_trg(n_trg * output_dim);
    pot_src.SetZero();
    pot_trg.SetZero();

    pdmk_params params;
    params.eps = 1e-7;
    params.n_dim = n_dim;
    params.n_per_leaf = 280;
    params.eval_src = DMK_POTENTIAL_GRAD;
    params.eval_trg = DMK_POTENTIAL_GRAD;
    params.kernel = DMK_LAPLACE;
    params.log_level = SPDLOG_LEVEL_OFF;

    pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
    pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
    pdmk_tree_destroy(tree);

    const int n_test_src = std::min(n_src, 64);
    const int n_test_trg = std::min(n_trg, 64);
    std::vector<double> direct_grad_src(n_test_src * n_dim, 0.0);
    std::vector<double> direct_grad_trg(n_test_trg * n_dim, 0.0);

    const auto grad_index = [n_dim](int i_pt, int i_dim) { return i_dim + n_dim * i_pt; };
    const auto accumulate_laplace_grad = [&](const double *target, int i_out, std::vector<double> &out) {
        for (int i_src = 0; i_src < n_src; ++i_src) {
            double dx[n_dim];
            double dr2 = 0.0;
            for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                dx[i_dim] = target[i_dim] - r_src[i_src * n_dim + i_dim];
                dr2 += dx[i_dim] * dx[i_dim];
            }
            if (dr2 <= thresh2)
                continue;

            const double rinv = 1.0 / std::sqrt(dr2);
            const double rinv3 = rinv / dr2;
            for (int i_dim = 0; i_dim < n_dim; ++i_dim)
                out[grad_index(i_out, i_dim)] -= charges[i_src] * dx[i_dim] * rinv3;
        }
    };

    for (int i = 0; i < n_test_src; ++i)
        accumulate_laplace_grad(&r_src[i * n_dim], i, direct_grad_src);
    for (int i = 0; i < n_test_trg; ++i)
        accumulate_laplace_grad(&r_trg[i * n_dim], i, direct_grad_trg);

    auto relative_l2_error = [](const auto &approx, const auto &exact) {
        double err2 = 0.0;
        double ref2 = 0.0;
        for (int i = 0; i < exact.size(); ++i) {
            err2 += sctl::pow<2>(approx[i] - exact[i]);
            ref2 += sctl::pow<2>(exact[i]);
        }
        return std::sqrt(err2 / ref2);
    };

    std::vector<double> grad_src_prefix(direct_grad_src.size());
    std::vector<double> grad_trg_prefix(direct_grad_trg.size());
    for (int i = 0; i < n_test_src; ++i)
        for (int i_dim = 0; i_dim < n_dim; ++i_dim)
            grad_src_prefix[grad_index(i, i_dim)] = pot_src[i * output_dim + 1 + i_dim];
    for (int i = 0; i < n_test_trg; ++i)
        for (int i_dim = 0; i_dim < n_dim; ++i_dim)
            grad_trg_prefix[grad_index(i, i_dim)] = pot_trg[i * output_dim + 1 + i_dim];

    const double l2_err_src = relative_l2_error(grad_src_prefix, direct_grad_src);
    const double l2_err_trg = relative_l2_error(grad_trg_prefix, direct_grad_trg);

    CHECK(l2_err_src < 2e-4);
    CHECK(l2_err_trg < 1e-3);
}

TEST_CASE_GENERIC("[DMK] pdmk Laplace dipole", 1) {
    constexpr int n_src = 4000;
    constexpr int n_trg = 3000;
    constexpr bool uniform = true;
    constexpr bool set_fixed_charges = false;
    constexpr double thresh2 = 1e-30;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    for (int n_dim : {3}) {
        const int nd = n_dim; // dipole strength components per source

        sctl::Vector<double> r_src, dipoles, rnormal, r_trg;
        dmk::util::init_test_data(n_dim, nd, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, dipoles,
                                  0);

        sctl::Vector<double> pot_src(n_src), pot_trg(n_trg);
        pot_src.SetZero();
        pot_trg.SetZero();

        pdmk_params params;
        params.eps = 1e-6;
        params.n_dim = n_dim;
        params.n_per_leaf = 280;
        params.eval_src = DMK_POTENTIAL;
        params.eval_trg = DMK_POTENTIAL;
        params.kernel = DMK_LAPLACE_DIPOLE;
        params.log_level = SPDLOG_LEVEL_OFF;

        pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &dipoles[0], &rnormal[0], n_trg, &r_trg[0]);
        pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
        pdmk_tree_destroy(tree);

        const int n_test_src = std::min(n_src, 64);
        const int n_test_trg = std::min(n_trg, 64);
        std::vector<double> direct_pot_src(n_test_src, 0.0);
        std::vector<double> direct_pot_trg(n_test_trg, 0.0);

        const auto accumulate_dipole = [&](const double *target, int i_out, std::vector<double> &out) {
            for (int i_src = 0; i_src < n_src; ++i_src) {
                double dx_arr[3] = {0, 0, 0};
                double dr2 = 0.0;
                for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                    dx_arr[i_dim] = target[i_dim] - r_src[i_src * n_dim + i_dim];
                    dr2 += dx_arr[i_dim] * dx_arr[i_dim];
                }
                if (dr2 <= thresh2)
                    continue;

                double dot = 0.0;
                for (int i_dim = 0; i_dim < n_dim; ++i_dim)
                    dot += dipoles[i_src * nd + i_dim] * dx_arr[i_dim];

                if (n_dim == 3) {
                    const double rinv = 1.0 / std::sqrt(dr2);
                    const double rinv3 = rinv / dr2;
                    out[i_out] += dot * rinv3;
                } else {
                    // 2D: phi = d . grad_s log(R) = -(d.dX)/R^2
                    out[i_out] -= dot / dr2;
                }
            }
        };

        for (int i = 0; i < n_test_src; ++i)
            accumulate_dipole(&r_src[i * n_dim], i, direct_pot_src);
        for (int i = 0; i < n_test_trg; ++i)
            accumulate_dipole(&r_trg[i * n_dim], i, direct_pot_trg);

        auto relative_l2_error = [](const auto &approx, const auto &exact) {
            double err2 = 0.0, ref2 = 0.0;
            for (int i = 0; i < (int)exact.size(); ++i) {
                err2 += sctl::pow<2>(approx[i] - exact[i]);
                ref2 += sctl::pow<2>(exact[i]);
            }
            return std::sqrt(err2 / ref2);
        };

        std::vector<double> pot_src_prefix(direct_pot_src.size()), pot_trg_prefix(direct_pot_trg.size());
        for (int i = 0; i < n_test_src; ++i)
            pot_src_prefix[i] = pot_src[i];
        for (int i = 0; i < n_test_trg; ++i)
            pot_trg_prefix[i] = pot_trg[i];

        const double l2_err_src = relative_l2_error(pot_src_prefix, direct_pot_src);
        const double l2_err_trg = relative_l2_error(pot_trg_prefix, direct_pot_trg);

        CHECK(l2_err_src < 10 * params.eps);
        CHECK(l2_err_trg < 10 * params.eps);
    }
}

TEST_CASE_GENERIC("[DMK] pdmk Laplace dipole gradient", 1) {
    constexpr int n_src = 10000;
    constexpr int n_trg = 10000;
    constexpr bool uniform = true;
    constexpr bool set_fixed_charges = false;
    constexpr double thresh2 = 1e-30;
    constexpr int n_to_compare = 10000;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    for (int n_dim : {3}) {
        const int nd = n_dim; // dipole strength components per source
        const int output_dim = 1 + n_dim;

        sctl::Vector<double> r_src, dipoles, rnormal, r_trg;
        dmk::util::init_test_data(n_dim, nd, n_src, n_trg, uniform, set_fixed_charges, r_src, r_trg, rnormal, dipoles,
                                  0);

        sctl::Vector<double> pot_src(n_src * output_dim), pot_trg(n_trg * output_dim);
        pot_src.SetZero();
        pot_trg.SetZero();

        pdmk_params params;
        params.eps = 1e-6;
        params.n_dim = n_dim;
        params.n_per_leaf = 280;
        params.eval_src = DMK_POTENTIAL_GRAD;
        params.eval_trg = DMK_POTENTIAL_GRAD;
        params.kernel = DMK_LAPLACE_DIPOLE;
        params.log_level = SPDLOG_LEVEL_OFF;

        pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &dipoles[0], &rnormal[0], n_trg, &r_trg[0]);
        pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
        pdmk_tree_destroy(tree);

        const int n_test_src = std::min(n_src, n_to_compare);
        const int n_test_trg = std::min(n_trg, n_to_compare);
        std::vector<double> direct_src(n_test_src * output_dim, 0.0);
        std::vector<double> direct_trg(n_test_trg * output_dim, 0.0);

        const auto accumulate_dipole_grad = [&](const double *target, int i_out, std::vector<double> &out) {
            for (int i_src = 0; i_src < n_src; ++i_src) {
                double dx_arr[3] = {0, 0, 0};
                double dr2 = 0.0;
                for (int i_dim = 0; i_dim < n_dim; ++i_dim) {
                    dx_arr[i_dim] = target[i_dim] - r_src[i_src * n_dim + i_dim];
                    dr2 += dx_arr[i_dim] * dx_arr[i_dim];
                }
                if (dr2 <= thresh2)
                    continue;

                double dot = 0.0;
                for (int i_dim = 0; i_dim < n_dim; ++i_dim)
                    dot += dipoles[i_src * nd + i_dim] * dx_arr[i_dim];

                if (n_dim == 3) {
                    const double rinv = 1.0 / std::sqrt(dr2);
                    const double rinv3 = rinv / dr2;
                    const double rinv5 = rinv3 / dr2;
                    out[i_out * output_dim + 0] += dot * rinv3;
                    for (int i = 0; i < 3; ++i)
                        out[i_out * output_dim + 1 + i] +=
                            dipoles[i_src * nd + i] * rinv3 - 3.0 * dot * dx_arr[i] * rinv5;
                } else {
                    const double r2inv = 1.0 / dr2;
                    const double r4inv = r2inv * r2inv;
                    out[i_out * output_dim + 0] -= dot * r2inv;
                    for (int i = 0; i < 2; ++i)
                        out[i_out * output_dim + 1 + i] +=
                            -dipoles[i_src * nd + i] * r2inv + 2.0 * dot * dx_arr[i] * r4inv;
                }
            }
        };

        for (int i = 0; i < n_test_src; ++i)
            accumulate_dipole_grad(&r_src[i * n_dim], i, direct_src);
        for (int i = 0; i < n_test_trg; ++i)
            accumulate_dipole_grad(&r_trg[i * n_dim], i, direct_trg);

        // Compute relative L2 error over a component range [comp_begin, comp_end),
        auto relative_l2_error_range = [&](const sctl::Vector<double> &approx, const std::vector<double> &exact,
                                           int n_pts, int comp_begin, int comp_end) {
            double err2 = 0.0, ref2 = 0.0;
            for (int i = 0; i < n_pts; ++i) {
                for (int c = comp_begin; c < comp_end; ++c) {
                    const double a = approx[i * output_dim + c];
                    const double e = exact[i * output_dim + c];
                    err2 += sctl::pow<2>(a - e);
                    ref2 += sctl::pow<2>(e);
                }
            }
            return std::sqrt(err2 / ref2);
        };

        const double pot_err_src = relative_l2_error_range(pot_src, direct_src, n_test_src, 0, 1);
        const double grad_err_src = relative_l2_error_range(pot_src, direct_src, n_test_src, 1, output_dim);
        const double pot_err_trg = relative_l2_error_range(pot_trg, direct_trg, n_test_trg, 0, 1);
        const double grad_err_trg = relative_l2_error_range(pot_trg, direct_trg, n_test_trg, 1, output_dim);

        CHECK(pot_err_src < params.eps);
        CHECK(grad_err_src < params.eps);
        CHECK(pot_err_trg < params.eps);
        CHECK(grad_err_trg < params.eps);
    }
}

TEST_CASE_GENERIC("[DMK] pdmk_direct", 1) {
#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif
    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 500;

    sctl::Vector<double> r_src, charges, rnormal, r_trg;
    dmk::util::init_test_data(n_dim, 1, n_src, n_trg, false, false, r_src, r_trg, rnormal, charges, 0);

    pdmk_params params;
    pdmk_init_default_params(&params);
    params.n_dim = n_dim;
    params.kernel = DMK_LAPLACE;
    params.log_level = SPDLOG_LEVEL_OFF;

    auto rel_l2 = [](const std::vector<double> &approx, const std::vector<double> &ref) {
        double err2 = 0.0, ref2 = 0.0;
        for (size_t i = 0; i < approx.size(); ++i) {
            err2 += sctl::pow<2>(approx[i] - ref[i]);
            ref2 += sctl::pow<2>(ref[i]);
        }
        return std::sqrt(err2 / ref2);
    };

    // The tree solve is the approximation, so it is run at a tolerance far tighter than the
    // threshold being asserted; direct summation is exact up to roundoff.
    SUBCASE("agrees with a tight-tolerance tree solve") {
        std::vector<double> pot_src(n_src), pot_trg(n_trg);
        REQUIRE(pdmk_direct(comm, params, n_src, &r_src[0], &charges[0], nullptr, n_trg, &r_trg[0], pot_src.data(),
                            pot_trg.data()) == DMK_SUCCESS);

        pdmk_params tree_params = params;
        tree_params.eps = 1e-12;
        std::vector<double> tree_src(n_src), tree_trg(n_trg);
        REQUIRE(pdmk(comm, tree_params, n_src, &r_src[0], &charges[0], nullptr, n_trg, &r_trg[0], tree_src.data(),
                     tree_trg.data()) == DMK_SUCCESS);

        CHECK(rel_l2(pot_src, tree_src) < 1e-10);
        CHECK(rel_l2(pot_trg, tree_trg) < 1e-10);
    }

    SUBCASE("a NULL output skips that point set") {
        std::vector<double> both_trg(n_trg), only_trg(n_trg);
        REQUIRE(pdmk_direct(comm, params, n_src, &r_src[0], &charges[0], nullptr, n_trg, &r_trg[0], nullptr,
                            both_trg.data()) == DMK_SUCCESS);
        std::vector<double> pot_src(n_src);
        REQUIRE(pdmk_direct(comm, params, n_src, &r_src[0], &charges[0], nullptr, n_trg, &r_trg[0], pot_src.data(),
                            only_trg.data()) == DMK_SUCCESS);
        CHECK(rel_l2(both_trg, only_trg) == 0.0);
    }

    // eval_src is left at the default DMK_POTENTIAL, which the Stokeslet does not support;
    // it must not be validated when no source output is requested.
    SUBCASE("a NULL output also skips its eval type's validation") {
        pdmk_params stokes = params;
        stokes.kernel = DMK_STOKESLET;
        stokes.eval_trg = DMK_VELOCITY;
        std::vector<double> stokes_charges(n_src * n_dim, 1.0);
        std::vector<double> vel_trg(n_trg * n_dim);
        CHECK(pdmk_direct(comm, stokes, n_src, &r_src[0], stokes_charges.data(), nullptr, n_trg, &r_trg[0], nullptr,
                          vel_trg.data()) == DMK_SUCCESS);
    }

    SUBCASE("single precision agrees with double") {
        std::vector<double> pot_trg(n_trg);
        REQUIRE(pdmk_direct(comm, params, n_src, &r_src[0], &charges[0], nullptr, n_trg, &r_trg[0], nullptr,
                            pot_trg.data()) == DMK_SUCCESS);

        std::vector<float> r_srcf(r_src.begin(), r_src.end()), chargesf(charges.begin(), charges.end());
        std::vector<float> r_trgf(r_trg.begin(), r_trg.end()), pot_trgf(n_trg);
        REQUIRE(pdmk_directf(comm, params, n_src, r_srcf.data(), chargesf.data(), nullptr, n_trg, r_trgf.data(),
                             nullptr, pot_trgf.data()) == DMK_SUCCESS);

        std::vector<double> promoted(pot_trgf.begin(), pot_trgf.end());

        // What limits the float run is the coordinate difference, not the kernel: a separation r
        // differenced from coordinates of order one carries relative error eps_f/r, which 1/r
        // passes through unchanged, and each potential is dominated by its nearest source.
        double num = 0.0, den = 0.0;
        for (int t = 0; t < n_trg; ++t) {
            double d2_min = std::numeric_limits<double>::max();
            for (int i = 0; i < n_src; ++i) {
                double d2 = 0.0;
                for (int d = 0; d < n_dim; ++d)
                    d2 += sctl::pow<2>(r_trg[t * n_dim + d] - r_src[i * n_dim + d]);
                d2_min = std::min(d2_min, d2);
            }
            num += sctl::pow<2>(pot_trg[t]) / d2_min;
            den += sctl::pow<2>(pot_trg[t]);
        }
        const double tol = 4 * std::numeric_limits<float>::epsilon() * std::sqrt(num / den);
        CHECK(rel_l2(promoted, pot_trg) < tol);
    }

    SUBCASE("tree-only parameters are rejected rather than ignored") {
        std::vector<double> pot_src(n_src);
        pdmk_params periodic = params;
        periodic.use_periodic = 1;
        CHECK(pdmk_direct(comm, periodic, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr, pot_src.data(),
                          nullptr) == DMK_ERR_INVALID_ARGUMENT);

        pdmk_params bad_path = params;
        bad_path.eval_path = dmk_eval_path(DMK_EVAL_PATH_GPU + 1);
        CHECK(pdmk_direct(comm, bad_path, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr, pot_src.data(),
                          nullptr) == DMK_ERR_INVALID_ARGUMENT);
    }

    SUBCASE("a GPU eval_path is accepted only in a GPU build") {
        std::vector<double> pot_src(n_src);
        pdmk_params gpu = params;
        gpu.eval_path = DMK_EVAL_PATH_GPU;
        const dmk_error err =
            pdmk_direct(comm, gpu, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr, pot_src.data(), nullptr);
#ifdef DMK_GPU_OFFLOAD
        CHECK(err == DMK_SUCCESS);
#else
        CHECK(err == DMK_ERR_INVALID_ARGUMENT);
#endif
    }

    SUBCASE("out-of-range eval types and unimplemented kernel/dim combinations are rejected") {
        std::vector<double> pot_src(n_src * 16);
        pdmk_params bad_eval = params;
        bad_eval.eval_src = dmk_eval_type(99);
        CHECK(pdmk_direct(comm, bad_eval, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr, pot_src.data(),
                          nullptr) == DMK_ERR_INVALID_ARGUMENT);

        pdmk_params stokes_2d = params;
        stokes_2d.n_dim = 2;
        stokes_2d.kernel = DMK_STOKESLET;
        stokes_2d.eval_src = DMK_VELOCITY;
        CHECK(pdmk_direct(comm, stokes_2d, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr, pot_src.data(),
                          nullptr) == DMK_ERR_INVALID_ARGUMENT);
    }

    SUBCASE("null charge with n_src > 0 is rejected") {
        std::vector<double> pot_src(n_src);
        CHECK(pdmk_direct(comm, params, n_src, &r_src[0], nullptr, nullptr, 0, nullptr, pot_src.data(), nullptr) ==
              DMK_ERR_INVALID_ARGUMENT);
    }
}

TEST_CASE_GENERIC("[DMK] error handling", 1) {
#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif
    constexpr int n_dim = 3;
    constexpr int n_src = 1000;

    sctl::Vector<double> r_src, charges, rnormal, r_trg;
    dmk::util::init_test_data(n_dim, 1, n_src, 0, true, false, r_src, r_trg, rnormal, charges, 0);

    pdmk_params params;
    pdmk_init_default_params(&params);
    params.n_dim = n_dim;
    params.kernel = DMK_LAPLACE;
    params.log_level = SPDLOG_LEVEL_OFF;

    SUBCASE("bad dimension returns NULL with message") {
        pdmk_params bad = params;
        bad.n_dim = 5;
        pdmk_tree tree = pdmk_tree_create(comm, bad, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr);
        CHECK(tree == nullptr);
        CHECK(std::string(pdmk_last_error_message()).size() > 0);
    }

    SUBCASE("negative n_src returns NULL") {
        pdmk_tree tree = pdmk_tree_create(comm, params, -1, &r_src[0], &charges[0], nullptr, 0, nullptr);
        CHECK(tree == nullptr);
    }

    SUBCASE("unsupported 2D Stokeslet returns NULL, not an abort") {
        pdmk_params bad = params;
        bad.n_dim = 2;
        bad.kernel = DMK_STOKESLET;
        pdmk_tree tree = pdmk_tree_create(comm, bad, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr);
        CHECK(tree == nullptr);
    }

    SUBCASE("velocity eval on a scalar kernel returns NULL") {
        pdmk_params bad = params;
        bad.eval_trg = DMK_VELOCITY;
        pdmk_tree tree = pdmk_tree_create(comm, bad, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr);
        CHECK(tree == nullptr);
    }

    SUBCASE("out-of-range eval type returns NULL") {
        pdmk_params bad = params;
        bad.eval_src = dmk_eval_type(99);
        bad.eval_trg = dmk_eval_type(99);
        pdmk_tree tree = pdmk_tree_create(comm, bad, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr);
        CHECK(tree == nullptr);
    }

    SUBCASE("out-of-range eval_path returns NULL") {
        pdmk_params bad = params;
        bad.eval_path = dmk_eval_path(7);
        pdmk_tree tree = pdmk_tree_create(comm, bad, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr);
        CHECK(tree == nullptr);
    }

    SUBCASE("Stresslet without a normal returns NULL, not a segfault") {
        pdmk_params bad = params;
        bad.n_dim = 3;
        bad.kernel = DMK_STRESSLET;
        bad.eval_src = DMK_VELOCITY;
        bad.eval_trg = DMK_VELOCITY;
        std::vector<double> stokes_charges(n_src * n_dim, 1.0);
        pdmk_tree tree = pdmk_tree_create(comm, bad, n_src, &r_src[0], stokes_charges.data(), nullptr, 0, nullptr);
        CHECK(tree == nullptr);
    }

    SUBCASE("null charge with n_src > 0 returns NULL") {
        pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], nullptr, nullptr, 0, nullptr);
        CHECK(tree == nullptr);
    }

    SUBCASE("null tree handle to eval is rejected") {
        std::vector<double> pot_src(n_src);
        CHECK(pdmk_tree_eval(nullptr, &pot_src[0], nullptr) == DMK_ERR_INVALID_ARGUMENT);
    }

    SUBCASE("null pot_src with sources present is rejected") {
        pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr);
        REQUIRE(tree != nullptr);
        CHECK(pdmk_tree_eval(tree, nullptr, nullptr) == DMK_ERR_INVALID_ARGUMENT);
        pdmk_tree_destroy(tree);
    }

    SUBCASE("a float tree evaluated in double precision is rejected") {
        std::vector<float> r_srcf(r_src.Dim()), chargesf(charges.Dim());
        std::copy(&r_src[0], &r_src[0] + r_src.Dim(), r_srcf.begin());
        std::copy(&charges[0], &charges[0] + charges.Dim(), chargesf.begin());
        pdmk_tree tree = pdmk_tree_createf(comm, params, n_src, r_srcf.data(), chargesf.data(), nullptr, 0, nullptr);
        REQUIRE(tree != nullptr);
        std::vector<double> pot_src(n_src);
        CHECK(pdmk_tree_eval(tree, pot_src.data(), nullptr) == DMK_ERR_INVALID_ARGUMENT);
        pdmk_tree_destroy(tree);
    }

    SUBCASE("a double tree evaluated in single precision is rejected") {
        pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr);
        REQUIRE(tree != nullptr);
        std::vector<float> pot_src(n_src);
        CHECK(pdmk_tree_evalf(tree, pot_src.data(), nullptr) == DMK_ERR_INVALID_ARGUMENT);
        pdmk_tree_destroy(tree);
    }

    SUBCASE("normal is ignored when updating charges for a scalar kernel") {
        std::vector<double> pot_src(n_src);
        pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], nullptr, 0, nullptr);
        REQUIRE(tree != nullptr);
        CHECK(pdmk_tree_eval(tree, &pot_src[0], nullptr) == DMK_SUCCESS);
        // Laplace has no normal; passing one is harmless and the update succeeds.
        CHECK(pdmk_tree_update_charges(tree, &charges[0], &charges[0]) == DMK_SUCCESS);
        pdmk_tree_destroy(tree);
    }
}

template <typename Real>
inline pdmk_tree pdmk_tree_create(dmk_communicator comm, const pdmk_params &params, int n_src, const Real *r_src,
                                  const Real *charge, const Real *normal, int n_trg, const Real *r_trg) {
    sctl::Profile::reset();
    sctl::Profile::Enable(true);
#ifdef DMK_HAVE_MPI
    const sctl::Comm sctl_comm(mpi_comm_or_self(comm));
#else
    const sctl::Comm sctl_comm;
#endif
    sctl::Profile::Scoped profile("pdmk_tree_create", &sctl_comm);
    const int charge_dim =
        params.kernel == DMK_STRESSLET ? params.n_dim : get_kernel_input_dim(params.n_dim, params.kernel);

    sctl::Vector<Real> r_src_vec(n_src * params.n_dim, const_cast<Real *>(r_src), false);
    sctl::Vector<Real> r_trg_vec(n_trg * params.n_dim, const_cast<Real *>(r_trg), false);
    sctl::Vector<Real> charge_vec(n_src * charge_dim, const_cast<Real *>(charge), false);
    sctl::Vector<Real> normal_vec(n_src * params.n_dim, const_cast<Real *>(normal), false);

    if (params.n_dim != 2 && params.n_dim != 3)
        throw api_error(DMK_ERR_INVALID_ARGUMENT, "Invalid dimension: " + std::to_string(params.n_dim));

#ifdef DMK_GPU_OFFLOAD
    if (params.eval_path == DMK_EVAL_PATH_GPU) {
        if (params.n_dim == 2)
            return new pdmk_tree_impl{pdmk_tree_variant(std::make_unique<dmk::cuda::pt::Tree<Real, 2>>(
                                          sctl_comm, params, r_src_vec, charge_vec, normal_vec, r_trg_vec)),
                                      n_src, n_trg};
        return new pdmk_tree_impl{pdmk_tree_variant(std::make_unique<dmk::cuda::pt::Tree<Real, 3>>(
                                      sctl_comm, params, r_src_vec, charge_vec, normal_vec, r_trg_vec)),
                                  n_src, n_trg};
    }
#endif

    if (params.n_dim == 2)
        return new pdmk_tree_impl{pdmk_tree_variant(std::make_unique<dmk::DMKPtTree<Real, 2>>(
                                      sctl_comm, params, r_src_vec, charge_vec, normal_vec, r_trg_vec)),
                                  n_src, n_trg};
    return new pdmk_tree_impl{pdmk_tree_variant(std::make_unique<dmk::DMKPtTree<Real, 3>>(
                                  sctl_comm, params, r_src_vec, charge_vec, normal_vec, r_trg_vec)),
                              n_src, n_trg};
}

template <typename Real>
inline void pdmk_tree_eval(pdmk_tree tree, Real *pot_src, Real *pot_trg) {
    auto *impl = static_cast<pdmk_tree_impl *>(tree);
    validate_eval_outputs(impl->n_src, pot_src, impl->n_trg, pot_trg);
    std::visit(
        [&](auto &t) {
            using TreeType = std::decay_t<decltype(t)>;
            constexpr bool is_cpu = std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 2>>> ||
                                    std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 3>>>;
#ifdef DMK_GPU_OFFLOAD
            constexpr bool is_gpu = std::is_same_v<TreeType, std::unique_ptr<dmk::cuda::pt::Tree<Real, 2>>> ||
                                    std::is_same_v<TreeType, std::unique_ptr<dmk::cuda::pt::Tree<Real, 3>>>;
#else
            constexpr bool is_gpu = false;
#endif
            if constexpr (is_cpu || is_gpu) {
                const auto &comm = t->GetComm();
                sctl::Profile::Scoped prof("pdmk_tree_eval", &comm);
                nvtxRangePush("pdmk_tree_eval");
                t->eval();
                t->desort_potentials(pot_src, pot_trg);
                nvtxRangePop();
            } else {
                throw api_error(DMK_ERR_INVALID_ARGUMENT, "tree precision does not match eval precision");
            }
        },
        impl->tree);
}

template <typename Real>
inline void pdmk_tree_update_charges(pdmk_tree tree, const Real *charge, const Real *normal) {
    std::visit(
        [&](auto &t) {
            using TreeType = std::decay_t<decltype(t)>;
            constexpr bool is_cpu = std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 2>>> ||
                                    std::is_same_v<TreeType, std::unique_ptr<dmk::DMKPtTree<Real, 3>>>;
#ifdef DMK_GPU_OFFLOAD
            constexpr bool is_gpu = std::is_same_v<TreeType, std::unique_ptr<dmk::cuda::pt::Tree<Real, 2>>> ||
                                    std::is_same_v<TreeType, std::unique_ptr<dmk::cuda::pt::Tree<Real, 3>>>;
#else
            constexpr bool is_gpu = false;
#endif
            if constexpr (is_cpu || is_gpu) {
                t->update_charges(charge, normal);
            } else {
                throw api_error(DMK_ERR_INVALID_ARGUMENT, "tree precision does not match update_charges precision");
            }
        },
        static_cast<pdmk_tree_impl *>(tree)->tree);
}

// Writes pot_src interleaved [pot, dx, dy, dz] per particle, matching pdmk_tree_eval, when the
// plan's eval_type requests gradients; else just pot. Vector-field kernels (Stokeslet/Stresslet)
// write the velocity interleaved [vx, vy, vz] per particle.
template <typename Real>
inline void esp_copy_result(const dmk::PotGrad<Real> &result, int n, Real *pot_src) {
    if (!result.vel_x.empty()) {
        const int dim = result.vel_z.empty() ? 2 : 3;
        for (int i = 0; i < n; ++i) {
            pot_src[i * dim + 0] = result.vel_x[i];
            pot_src[i * dim + 1] = result.vel_y[i];
            if (dim == 3)
                pot_src[i * dim + 2] = result.vel_z[i];
        }
        return;
    }
    if (result.grad_x.empty()) {
        std::copy(result.pot.begin(), result.pot.end(), pot_src);
        return;
    }
    const int dim = result.grad_z.empty() ? 2 : 3; // scaffolding for a future DIM=2 plan
    const int out_dim = 1 + dim;
    for (int i = 0; i < n; ++i) {
        pot_src[i * out_dim + 0] = result.pot[i];
        pot_src[i * out_dim + 1] = result.grad_x[i];
        pot_src[i * out_dim + 2] = result.grad_y[i];
        if (dim == 3)
            pot_src[i * out_dim + 3] = result.grad_z[i];
    }
}

// Dispatches to the plan's own precision -- evalf genuinely runs EspPlan<float>::eval (float
// FFTs/SIMD throughout), it never up-converts through a double plan.
// (A template can't have C language linkage, so this lives here rather than in the extern "C"
// block below, which only holds the non-template pdmk_esp_eval/evalf wrappers.)
#ifdef DMK_GPU_OFFLOAD
// GpuSrStrategy is one enumerator, so the independent CPU pruning bits collapse by precedence.
GpuSrStrategy esp_gpu_strategy(const pdmk_esp_params &params) {
    if (esp_prune_source(params))
        return GpuSrStrategy::PruneSource;
    if (esp_prune_tile(params))
        return GpuSrStrategy::PruneTile;
    return GpuSrStrategy::Dense;
}

GpuSortMode esp_gpu_sort_mode(const pdmk_esp_params &params) {
    return esp_morton(params) ? GpuSortMode::Morton : GpuSortMode::Bins;
}

void esp_report_inert_gpu_tuning(const pdmk_esp_params &params) {
    std::string inert;
    if (esp_n3l(params))
        inert += " DMK_ESP_N3L";
    if (!esp_morton(params) && params.esp_bins != 2)
        inert += " esp_bins";
    if (esp_prune_tile(params) && !esp_prune_source(params) && params.esp_stile != 0)
        inert += " esp_stile";
    if (!inert.empty())
        get_logger(sctl::Comm::Self(), params.log_level)
            ->info("esp: GPU plan ignores CPU-only short-range tuning:{}", inert);
}
#endif

template <typename Real>
pdmk_esp_plan esp_plan_create_impl(pdmk_esp_params params) {
    auto impl = std::make_unique<pdmk_esp_plan_impl>();
    auto plan = std::make_unique<dmk::EspPlan<Real>>(params);

    if (params.eval_path == DMK_EVAL_PATH_GPU) {
#ifdef DMK_GPU_OFFLOAD
        dmk::cuda::pt::bind_gpu_device(params.gpu_device_id);
        cuda_helpers::ScopedDevice device_scope(params.gpu_device_id);
        esp_report_inert_gpu_tuning(params);
        impl->gpu.reset(
            dmk::esp_create_gpu_plan<Real>(plan.get(), esp_gpu_strategy(params), esp_gpu_sort_mode(params)));
        impl->gpu_device_id = params.gpu_device_id;
#else
        throw api_error(DMK_ERR_INVALID_ARGUMENT, "pdmk_esp_params.eval_path is GPU but this build has no GPU support "
                                                  "(configure with -DDMK_GPU_OFFLOAD=ON)");
#endif
    }

    impl->plan = std::move(plan);
    return impl.release();
}

template <typename Real>
inline void pdmk_esp_eval_impl(pdmk_esp_plan plan, int n, const Real *r_src, const Real *charges, const Real *normal,
                               Real *pot_src) {
    if (!plan)
        throw api_error(DMK_ERR_INVALID_ARGUMENT, "null ESP plan handle");
    if (n < 0)
        throw api_error(DMK_ERR_INVALID_ARGUMENT, "n must be non-negative, got " + std::to_string(n));
    if (n > 0 && (r_src == nullptr || charges == nullptr || pot_src == nullptr))
        throw api_error(DMK_ERR_INVALID_ARGUMENT, "r_src, charges and pot_src must be non-null when n > 0");

    auto *impl = static_cast<pdmk_esp_plan_impl *>(plan);
    std::visit(
        [&](auto &p) {
            using PlanType = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<PlanType, std::unique_ptr<dmk::EspPlan<Real>>>) {
                // pack_payload reads normal per source for the Stresslet and nothing else.
                if (p->params.kernel == DMK_STRESSLET && n > 0 && normal == nullptr)
                    throw api_error(DMK_ERR_INVALID_ARGUMENT, "ESP Stresslet requires a non-null normal array");
#ifdef DMK_GPU_OFFLOAD
                if (impl->gpu) {
                    cuda_helpers::ScopedDevice device_scope(impl->gpu_device_id);
                    esp_copy_result<Real>(dmk::esp_eval_gpu(impl->gpu.get(), n, r_src, charges, normal), n, pot_src);
                    return;
                }
#endif
                esp_copy_result<Real>(p->eval(n, r_src, charges, normal), n, pot_src);
            } else
                throw api_error(DMK_ERR_INVALID_ARGUMENT, "ESP plan precision does not match eval precision");
        },
        impl->plan);
}

} // namespace dmk

extern "C" {

const char *pdmk_version_string(void) { return DMK_VERSION_STRING; }

const char *pdmk_git_commit(void) { return DMK_GIT_COMMIT; }

void pdmk_version(int *major, int *minor, int *patch) {
    if (major)
        *major = DMK_VERSION_MAJOR;
    if (minor)
        *minor = DMK_VERSION_MINOR;
    if (patch)
        *patch = DMK_VERSION_PATCH;
}

void pdmk_init_default_params(pdmk_params *params) {
    if (params)
        *params = pdmk_params{};
}

const char *pdmk_last_error_message(void) { return dmk::last_error_message(); }

dmk_error pdmk_print_profile_data(dmk_communicator comm, char type) {
    return dmk::dmk_guard([&] {
#ifdef DMK_HAVE_MPI
        sctl::Comm sctl_comm(dmk::mpi_comm_or_self(comm));
#else
        sctl::Comm sctl_comm;
#endif
        const std::vector<std::string> fields{"t_avg", "t_max",   "t_min",     "f_avg",   "f_max",
                                              "f_min", "f_total", "f/s_total", "custom1", "custom2"};
        if (type == 'h') {
            auto table = sctl::Profile::get_table(fields, &sctl_comm);
            if (sctl_comm.Rank() == 0) {
                std::string sep;
                for (auto &row : table) {
                    for (auto &field : row.second) {
                        std::cout << sep << row.first << "|" << field.first;
                        sep = ",";
                    }
                }
            }
        }
        if (type == 'c') {
            auto table = sctl::Profile::get_table(fields, &sctl_comm);
            if (sctl_comm.Rank() == 0) {
                std::string sep;
                for (auto &row : table) {
                    for (auto &field : row.second) {
                        std::cout << sep << field.second;
                        sep = ",";
                    }
                }
            }
            sctl::Profile::reset();
        } else if (type == 't') {
            sctl::Profile::print(
                &sctl_comm,
                {"t_avg", "t_max", "t_min", "f_avg", "f_max", "f_min", "f_total", "f/s_total", "custom1", "custom2"},
                {"%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.5f", "%.3g", "%.3g"});
        }
    });
}

pdmk_tree pdmk_tree_createf(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src,
                            const float *charge, const float *normal, int n_trg, const float *r_trg) {
    pdmk_tree result = nullptr;
    dmk::dmk_guard([&] {
        dmk::validate_create_args(comm, params, n_src, r_src, charge, normal, n_trg, r_trg);
        result = dmk::pdmk_tree_create(comm, params, n_src, r_src, charge, normal, n_trg, r_trg);
    });
    return result;
}

pdmk_tree pdmk_tree_create(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src,
                           const double *charge, const double *normal, int n_trg, const double *r_trg) {
    pdmk_tree result = nullptr;
    dmk::dmk_guard([&] {
        dmk::validate_create_args(comm, params, n_src, r_src, charge, normal, n_trg, r_trg);
        result = dmk::pdmk_tree_create(comm, params, n_src, r_src, charge, normal, n_trg, r_trg);
    });
    return result;
}

void pdmk_tree_destroy(pdmk_tree tree) {
    if (tree)
        delete static_cast<pdmk_tree_impl *>(tree);
}

dmk_error pdmk_tree_update_charges(pdmk_tree tree, const double *charge, const double *normal) {
    return dmk::dmk_guard([&] {
        if (!tree)
            throw dmk::api_error(DMK_ERR_INVALID_ARGUMENT, "null tree handle");
        if (!charge)
            throw dmk::api_error(DMK_ERR_INVALID_ARGUMENT, "null charge pointer");
        dmk::pdmk_tree_update_charges(tree, charge, normal);
    });
}

dmk_error pdmk_tree_update_chargesf(pdmk_tree tree, const float *charge, const float *normal) {
    return dmk::dmk_guard([&] {
        if (!tree)
            throw dmk::api_error(DMK_ERR_INVALID_ARGUMENT, "null tree handle");
        if (!charge)
            throw dmk::api_error(DMK_ERR_INVALID_ARGUMENT, "null charge pointer");
        dmk::pdmk_tree_update_charges(tree, charge, normal);
    });
}

dmk_error pdmk_tree_evalf(pdmk_tree tree, float *pot_src, float *pot_trg) {
    return dmk::dmk_guard([&] {
        if (!tree)
            throw dmk::api_error(DMK_ERR_INVALID_ARGUMENT, "null tree handle");
        dmk::pdmk_tree_eval(tree, pot_src, pot_trg);
    });
}

dmk_error pdmk_tree_eval(pdmk_tree tree, double *pot_src, double *pot_trg) {
    return dmk::dmk_guard([&] {
        if (!tree)
            throw dmk::api_error(DMK_ERR_INVALID_ARGUMENT, "null tree handle");
        dmk::pdmk_tree_eval(tree, pot_src, pot_trg);
    });
}

dmk_error pdmkf(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src, const float *charge,
                const float *normal, int n_trg, const float *r_trg, float *pot_src, float *pot_trg) {
    return dmk::dmk_guard([&] {
        dmk::validate_create_args(comm, params, n_src, r_src, charge, normal, n_trg, r_trg);
        dmk::validate_eval_outputs(n_src, pot_src, n_trg, pot_trg);
        if (params.n_dim == 2)
            dmk::pdmk<float, 2>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
        else
            dmk::pdmk<float, 3>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
    });
}

dmk_error pdmk(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src, const double *charge,
               const double *normal, int n_trg, const double *r_trg, double *pot_src, double *pot_trg) {
    return dmk::dmk_guard([&] {
        dmk::validate_create_args(comm, params, n_src, r_src, charge, normal, n_trg, r_trg);
        dmk::validate_eval_outputs(n_src, pot_src, n_trg, pot_trg);
        if (params.n_dim == 2)
            dmk::pdmk<double, 2>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
        else
            dmk::pdmk<double, 3>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
    });
}

dmk_error pdmk_directf(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src, const float *charge,
                       const float *normal, int n_trg, const float *r_trg, float *pot_src, float *pot_trg) {
    return dmk::dmk_guard([&] {
        dmk::validate_direct_args(params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
        dmk::pdmk_direct<float>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
    });
}

dmk_error pdmk_direct(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src, const double *charge,
                      const double *normal, int n_trg, const double *r_trg, double *pot_src, double *pot_trg) {
    return dmk::dmk_guard([&] {
        dmk::validate_direct_args(params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
        dmk::pdmk_direct<double>(comm, params, n_src, r_src, charge, normal, n_trg, r_trg, pot_src, pot_trg);
    });
}

pdmk_esp_plan pdmk_esp_plan_create(dmk_communicator /*comm*/, pdmk_esp_params params) {
    pdmk_esp_plan result = nullptr;
    dmk::dmk_guard([&] {
        dmk::validate_esp_args(params);
        result = dmk::esp_plan_create_impl<double>(params);
    });
    return result;
}

pdmk_esp_plan pdmk_esp_plan_createf(dmk_communicator /*comm*/, pdmk_esp_params params) {
    pdmk_esp_plan result = nullptr;
    dmk::dmk_guard([&] {
        dmk::validate_esp_args(params);
        result = dmk::esp_plan_create_impl<float>(params);
    });
    return result;
}

dmk_error pdmk_esp_eval(dmk_communicator /*comm*/, pdmk_esp_plan plan, int n, const double *r_src,
                        const double *charges, const double *normal, double *pot_src) {
    return dmk::dmk_guard([&] { dmk::pdmk_esp_eval_impl<double>(plan, n, r_src, charges, normal, pot_src); });
}

dmk_error pdmk_esp_evalf(dmk_communicator /*comm*/, pdmk_esp_plan plan, int n, const float *r_src, const float *charges,
                         const float *normal, float *pot_src) {
    return dmk::dmk_guard([&] { dmk::pdmk_esp_eval_impl<float>(plan, n, r_src, charges, normal, pot_src); });
}

void pdmk_esp_plan_destroy(pdmk_esp_plan plan) { delete static_cast<pdmk_esp_plan_impl *>(plan); }

void pdmk_esp_plan_destroyf(pdmk_esp_plan plan) { pdmk_esp_plan_destroy(plan); }

// The plan is created inside a guard rather than through pdmk_esp_plan_create, whose NULL
// return would collapse every failure reason into one code.
dmk_error pdmk_esp(dmk_communicator comm, pdmk_esp_params params, int n, const double *r_src, const double *charges,
                   const double *normal, double *pot_src) {
    pdmk_esp_plan plan = nullptr;
    const dmk_error create_err = dmk::dmk_guard([&] {
        dmk::validate_esp_args(params);
        plan = dmk::esp_plan_create_impl<double>(params);
    });
    if (create_err != DMK_SUCCESS)
        return create_err;
    const dmk_error err = pdmk_esp_eval(comm, plan, n, r_src, charges, normal, pot_src);
    pdmk_esp_plan_destroy(plan);
    return err;
}

dmk_error pdmk_espf(dmk_communicator comm, pdmk_esp_params params, int n, const float *r_src, const float *charges,
                    const float *normal, float *pot_src) {
    pdmk_esp_plan plan = nullptr;
    const dmk_error create_err = dmk::dmk_guard([&] {
        dmk::validate_esp_args(params);
        plan = dmk::esp_plan_create_impl<float>(params);
    });
    if (create_err != DMK_SUCCESS)
        return create_err;
    const dmk_error err = pdmk_esp_evalf(comm, plan, n, r_src, charges, normal, pot_src);
    pdmk_esp_plan_destroyf(plan);
    return err;
}
}
