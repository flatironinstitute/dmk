#include "tensorprod.hpp"

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <string>
#include <vector>

namespace dmk::cuda::pt {
namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

std::size_t tp_shared_bytes(int n_order, int z_tile, std::size_t sizeof_real) {
    const std::size_t n2 = std::size_t(n_order) * std::size_t(n_order);
    return std::size_t(2 * z_tile + 3) * n2 * sizeof_real;
}

} // namespace

template <typename Real>
void launch_tensorprod(dmk::cuda::TensorprodArgs<Real> &args, std::size_t proxy_count, cudaStream_t stream) {
    if (args.n_pairs == 0)
        return;
    static JitCache cache;

    auto launch_one = [&](const TuningParams &p, cudaStream_t st) {
        JitKey key;
        key.name = "PtTensorprodKernel";
        key.real = jit_real_name<Real>();
        key.sm_major = cache.sm_major();
        key.sm_minor = cache.sm_minor();
        key.params = {{"N_ORDER", args.n_order},          {"N_CHARGE_DIM", args.n_charge_dim},
                      {"BLOCK_SIZE", p.at("BLOCK_SIZE")}, {"TENSOR_Z_TILE", p.at("Z_TILE")},
                      {"TENSOR_I_TILE", p.at("I_TILE")},  {"TENSOR_J_TILE", p.at("J_TILE")}};
        auto kernel = cache.get_kernel_from_source(
            key, [&] { return make_stage_source("pt/tensorprod.cu", key, "", "PtTensorprod"); });
        const std::size_t shared = tp_shared_bytes(args.n_order, p.at("Z_TILE"), sizeof(Real));
        set_max_dynamic_smem(*kernel, shared);
        kernel->launch(dim3(args.n_pairs, 1, 1), dim3(p.at("BLOCK_SIZE"), 1, 1), shared, st, args);
    };

    std::ostringstream tune_key;
    tune_key << "PtTensorprod|real=" << jit_real_name<Real>() << "|n_order=" << args.n_order
             << "|n_charge_dim=" << args.n_charge_dim;
    const std::string tk = tune_key.str();

    if (auto cfg = autotune_cached(tk)) {
        launch_one(*cfg, stream);
        return;
    }

    const cudaDeviceProp &prop = device_prop();
    const std::size_t max_shared = device_max_shared_bytes();
    const int n_order = args.n_order;

    const std::vector<TuningParameter> space{
        {"BLOCK_SIZE", {128, 256, 512}}, {"Z_TILE", {1, 2, 4}}, {"I_TILE", {1, 2, 3, 4}}, {"J_TILE", {2, 4, 6}}};
    const TuningParams defaults{{"BLOCK_SIZE", 512}, {"Z_TILE", 2}, {"I_TILE", 2}, {"J_TILE", 4}};

    const auto constraint = [&, n_order](const TuningParams &p) {
        const int bs = p.at("BLOCK_SIZE"), z = p.at("Z_TILE"), it = p.at("I_TILE"), jt = p.at("J_TILE");
        if (bs <= 0 || bs > prop.maxThreadsPerBlock || bs % 32 != 0)
            return false;
        if (z <= 0 || z > n_order || it <= 0 || it > n_order || jt <= 0 || jt > n_order)
            return false;
        if (it * jt > 16)
            return false;
        return tp_shared_bytes(n_order, z, sizeof(Real)) <= max_shared;
    };

    autotuned_launch<Real>(tk, "PtTensorprodKernel", space, defaults, constraint, launch_one, args.proxy_flat,
                           proxy_count, stream);
}

template void launch_tensorprod<float>(dmk::cuda::TensorprodArgs<float> &, std::size_t, cudaStream_t);
template void launch_tensorprod<double>(dmk::cuda::TensorprodArgs<double> &, std::size_t, cudaStream_t);

} // namespace dmk::cuda::pt
