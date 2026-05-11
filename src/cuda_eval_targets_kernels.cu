// Runtime dispatch for the per-box eval_targets kernel. Hand-written switch
// on (DIM, EVAL_LEVEL, N_CHARGE_DIM); n_order is runtime-bound up to
// MAX_N_ORDER. Add new combos here as needed.

#include <dmk/cuda_eval_targets_kernels.cuh>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

namespace {

template <typename Real>
void dispatch(int dim, int eval_level, int n_charge_dim, const EvalTargetsArgs<Real> &args, cudaStream_t stream) {
    constexpr int MAX_N_ORDER = 32;
    if (args.n_order > MAX_N_ORDER)
        throw std::runtime_error("CUDA eval_targets: n_order=" + std::to_string(args.n_order) +
                                 " exceeds MAX_N_ORDER=" + std::to_string(MAX_N_ORDER));

#define DMK_EVAL_TARGETS_CASE(D, E, NCD)                                                                               \
    if (dim == (D) && eval_level == (E) && n_charge_dim == (NCD)) {                                                   \
        launch_eval_targets_kernel<Real, (D), (E), (NCD)>(args, stream);                                  \
        return;                                                                                                        \
    }

    // Single-charge-dim (laplace, sqrt_laplace).
    DMK_EVAL_TARGETS_CASE(2, 1, 1)
    DMK_EVAL_TARGETS_CASE(2, 2, 1)
    DMK_EVAL_TARGETS_CASE(3, 1, 1)
    DMK_EVAL_TARGETS_CASE(3, 2, 1)

    // Three-charge-dim (stokeslet, stresslet — both VELOCITY only, EVAL_LEVEL=1).
    DMK_EVAL_TARGETS_CASE(3, 1, 3)

#undef DMK_EVAL_TARGETS_CASE

    throw std::runtime_error("CUDA eval_targets: unsupported (dim=" + std::to_string(dim) +
                             ", eval_level=" + std::to_string(eval_level) +
                             ", n_charge_dim=" + std::to_string(n_charge_dim) + ")");
}

} // namespace

template <typename Real>
void launch_eval_targets_dispatch(int dim, int eval_level, int n_charge_dim, const EvalTargetsArgs<Real> &args,
                                  cudaStream_t stream) {
    dispatch<Real>(dim, eval_level, n_charge_dim, args, stream);
}

template void launch_eval_targets_dispatch<float>(int, int, int, const EvalTargetsArgs<float> &, cudaStream_t);
template void launch_eval_targets_dispatch<double>(int, int, int, const EvalTargetsArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
