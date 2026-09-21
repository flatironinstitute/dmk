#include <dmk/cuda/direct.hpp>

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "../pt/launchers.hpp"

#include <dmk/cuda/direct_freespace_kernelargs.hpp>
#include <dmk/cuda/helpers.hpp>
#include <dmk/direct.hpp>
#include <dmk/util.hpp>

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace dmk::cuda {
namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

// Not tuned: a single predictable compile beats a grid sweep on the small problems the
// reference path is usually called on.
constexpr int kBlockSize = 256;
constexpr int kSrcTile = 256;

// 1 = potential / velocity, 2 = potential + gradient, matching pt::direct.
int eval_level_for(dmk_eval_type ev) {
    if (ev == DMK_POTENTIAL || ev == DMK_VELOCITY)
        return 1;
    if (ev == DMK_POTENTIAL_GRAD)
        return 2;
    throw std::runtime_error("direct_freespace: unsupported eval type");
}

// The Stokes kernels have a single output field, so they take no EVAL_LEVEL argument.
std::string evaluator_expr(dmk_ikernel kernel, int dim, int eval_level) {
    const std::string el = "<" + std::to_string(eval_level) + ">";
    if (kernel == DMK_LAPLACE)
        return (dim == 3 ? "LaplaceFreeEvaluator3D" : "LaplaceFreeEvaluator2D") + el;
    if (kernel == DMK_SQRT_LAPLACE)
        return (dim == 3 ? "SqrtLaplaceFreeEvaluator3D" : "SqrtLaplaceFreeEvaluator2D") + el;
    if (kernel == DMK_YUKAWA)
        return (dim == 3 ? "YukawaFreeEvaluator3D" : "YukawaFreeEvaluator2D") + el;
    if (kernel == DMK_LAPLACE_DIPOLE)
        return (dim == 3 ? "LaplaceDipoleFreeEvaluator3D" : "LaplaceDipoleFreeEvaluator2D") + el;
    if (kernel == DMK_STOKESLET && dim == 3)
        return "StokesletFreeEvaluator3D";
    if (kernel == DMK_STRESSLET && dim == 3)
        return "StressletFreeEvaluator3D";
    throw std::runtime_error("direct_freespace: no evaluator for kernel " + std::string(util::to_string(kernel)) +
                             " in " + std::to_string(dim) + "D");
}

} // namespace

template <typename Real>
void direct_freespace(const pdmk_params &params, dmk_eval_type eval, int n_src, const Real *r_src, const Real *charge,
                      const Real *normal, int n_trg, const Real *r_trg, Real *pot) {
    if (n_trg == 0)
        return;

    const int dim = params.n_dim;
    const int eval_level = eval_level_for(eval);
    const int input_dim = get_kernel_input_dim(dim, params.kernel);
    const int normal_dim = params.kernel == DMK_STRESSLET ? dim : 0;
    const int out_dim = get_kernel_output_dim(dim, params.kernel, eval);

    // Allocation must precede the JIT lookup: cuModuleLoadData needs a current context, and
    // neither cuInit nor cudaGetDevice creates one -- the first cudaMalloc is what binds the
    // primary context.
    cuda_helpers::DeviceBuffer<Real> d_r_src, d_charge, d_normal, d_r_trg, d_pot;
    d_r_trg.resize(std::size_t(n_trg) * dim);
    d_r_trg.upload(r_trg, std::size_t(n_trg) * dim);
    d_pot.resize(std::size_t(n_trg) * out_dim);
    if (n_src > 0) {
        d_r_src.resize(std::size_t(n_src) * dim);
        d_r_src.upload(r_src, std::size_t(n_src) * dim);
        d_charge.resize(std::size_t(n_src) * input_dim);
        d_charge.upload(charge, std::size_t(n_src) * input_dim);
        if (normal_dim > 0) {
            d_normal.resize(std::size_t(n_src) * normal_dim);
            d_normal.upload(normal, std::size_t(n_src) * normal_dim);
        }
    }

    // Nothing is baked into the source, so (kernel, dim, eval level) names the module
    // completely. JitKey::to_string hashes only name + real + sm + params.
    const std::string kernel_name = "DirectFreeKernel_" + std::string(util::to_string(params.kernel)) + "_" +
                                    std::to_string(dim) + "d_el" + std::to_string(eval_level);

    static JitCache cache;

    JitKey key;
    key.name = kernel_name;
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params["BLOCK_SIZE"] = kBlockSize;
    key.params["SRC_TILE"] = kSrcTile;

    const std::shared_ptr<jit::JitKernel> kernel = cache.get_kernel_from_source(key, [&] {
        const std::string prelude = "#define DMK_DIRECT_KERNEL_NAME " + kernel_name + "\n\n" +
                                    "#define DMK_DIRECT_EVALUATOR " + evaluator_expr(params.kernel, dim, eval_level) +
                                    "\n\n";
        return pt::make_stage_source("direct/direct_freespace.cu", key, prelude, "DirectFreespace");
    });

    DirectFreespaceArgs<Real> a;
    a.n_src = n_src;
    a.n_trg = n_trg;
    a.lambda = params.fparam;
    a.r_src = d_r_src.data();
    a.charge = d_charge.data();
    a.normal = d_normal.data();
    a.r_trg = d_r_trg.data();
    a.pot = d_pot.data();

    // Must match the carve-up in DirectFreespaceBody.
    const std::size_t shared_bytes = std::size_t(kSrcTile) * (dim + input_dim + normal_dim) * sizeof(Real);
    const int grid = (n_trg + kBlockSize - 1) / kBlockSize;
    kernel->launch(dim3(grid, 1, 1), dim3(kBlockSize, 1, 1), shared_bytes, cudaStream_t{0}, a);

    DMK_CHECK_CUDA(cudaMemcpy(pot, d_pot.data(), d_pot.size_bytes(), cudaMemcpyDeviceToHost));
    cuda_helpers::check_device_errors("direct_freespace");
}

template void direct_freespace<float>(const pdmk_params &, dmk_eval_type, int, const float *, const float *,
                                      const float *, int, const float *, float *);
template void direct_freespace<double>(const pdmk_params &, dmk_eval_type, int, const double *, const double *,
                                       const double *, int, const double *, double *);

} // namespace dmk::cuda
