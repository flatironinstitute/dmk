// Runtime dispatch for the per-box eval_targets kernel. Hand-written switch
// on (DIM, EVAL_LEVEL, N_CHARGE_DIM); n_order is runtime-bound up to
// MAX_N_ORDER. Add new combos here as needed.

#include <dmk/cuda/eval_targets_kernels.cuh>

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

template <typename Real>
__global__ void inplace_accumulate_kernel(Real* __restrict__ dst, const Real* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] += src[i];
}

template <typename Real>
void launch_inplace_accumulate(Real* dst, const Real* src, std::size_t n, cudaStream_t stream) {
    if (n == 0)
        return;
    constexpr int block = 256;
    int grid = (int)((n + block - 1) / block);
    inplace_accumulate_kernel<<<grid, block, 0, stream>>>(dst, src, (int)n);
}

template void launch_inplace_accumulate<float>(float*, const float*, std::size_t, cudaStream_t);
template void launch_inplace_accumulate<double>(double*, const double*, std::size_t, cudaStream_t);

template <typename Real>
__global__ void self_correction_kernel(SelfCorrectionArgs<Real> a) {
    int idx = blockIdx.x;
    if (idx >= a.n_direct_work)
        return;

    Real factor = a.correction_factors[idx];
    if (factor == Real{0})
        return;

    int box = a.direct_work[idx];
    if (!a.src_counts_owned[box])
        return;

    int count = a.src_counts_halo[box];
    long pot_off = a.pot_src_offsets[box];
    long chg_off = a.charge_halo_offsets[box];

    for (int i_src = threadIdx.x; i_src < count; i_src += blockDim.x)
        for (int i = 0; i < a.n_input_dim; i++)
            a.pot_src[pot_off + i_src * a.pot_stride + i] -= factor * a.charge_halo[chg_off + i_src * a.n_input_dim + i];
}

template <typename Real>
void launch_self_correction(const SelfCorrectionArgs<Real> &args, cudaStream_t stream) {
    if (args.n_direct_work == 0)
        return;
    constexpr int block = 128;
    self_correction_kernel<<<args.n_direct_work, block, 0, stream>>>(args);
}

template void launch_self_correction<float>(const SelfCorrectionArgs<float> &, cudaStream_t);
template void launch_self_correction<double>(const SelfCorrectionArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
