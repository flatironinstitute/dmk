#include "tensorprod.hpp"

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"

#include <cuda_runtime.h>
#include <dmk/cuda/helpers.hpp>

#include <algorithm>
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

// Mirrors the layout in tensorprod_up.cu: child pointers, parity-split matrix, then the x/y/z
// fold buffers for K_TILE slabs.
std::size_t tp_up_shared_bytes(int n, int k_tile, std::size_t sizeof_real) {
    const std::size_t np = ((n + 1) / 2 + 3) / 4 * 4, ld = n | 1;
    return 8 * sizeof(void *) + (2 * n * np + 12 * k_tile * n * ld + 2 * k_tile * n * n) * sizeof_real;
}

} // namespace

template <typename Real>
void launch_tensorprod(dmk::cuda::TensorprodArgs<Real> &args, std::size_t proxy_count, cudaStream_t stream,
                       bool tune_here) {
    if (args.n_pairs == 0)
        return;
    const int grid = args.n_pairs;
    static JitCache cache;

    auto launch_one = [&](const TuningParams &p, cudaStream_t st, bool compile_only) {
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
        if (compile_only)
            return;
        kernel->launch(dim3(grid, args.n_charge_dim, 1), dim3(p.at("BLOCK_SIZE"), 1, 1), shared, st, args);
    };

    std::ostringstream tune_key;
    tune_key << "PtTensorprod|real=" << jit_real_name<Real>() << "|n_order=" << args.n_order
             << "|n_charge_dim=" << args.n_charge_dim;
    tune_key << "|src=" << jit::jit_source_hash("pt/tensorprod.cu");
    const std::string tk = tune_key.str();

    const std::vector<TuningParameter> space{
        {"BLOCK_SIZE", {128, 256, 512}}, {"Z_TILE", {1, 2, 4}}, {"I_TILE", {1, 2, 3, 4}}, {"J_TILE", {2, 4, 6}}};
    const TuningParams defaults{{"BLOCK_SIZE", 512}, {"Z_TILE", 2}, {"I_TILE", 2}, {"J_TILE", 4}};

    if (auto cfg = autotune_cached(tk)) {
        launch_one(*cfg, stream, false);
        return;
    }

    const cudaDeviceProp &prop = device_prop();
    const std::size_t max_shared = device_max_shared_bytes();
    const int n_order = args.n_order;

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

    const auto canonicalize = [&](TuningParams p) {
        return clamp_tiles(std::move(p), {{"Z_TILE", n_order}, {"I_TILE", n_order}, {"J_TILE", n_order}});
    };

    // Measuring a level of a few blocks would pick a config for launch overhead rather than for
    // throughput, and BLOCK_SIZE decides blocks per SM on a kernel that stalls mostly on barriers.
    if (!tune_here) {
        launch_one(jit::nearest_feasible(space, defaults, constraint, canonicalize), stream, false);
        return;
    }

    autotuned_launch<Real>(tk, "PtTensorprodKernel", space, defaults, constraint, launch_one, args.proxy_flat,
                           proxy_count, stream, canonicalize);
}

template <typename Real>
void launch_tensorprod_up(dmk::cuda::TensorprodArgs<Real> &args, std::size_t proxy_count, cudaStream_t stream,
                          bool tune_here) {
    if (args.n_par == 0)
        return;
    static JitCache cache;
    // Never destroyed: a static DeviceBuffer's destructor would free device memory from a
    // static destructor, by which point the CUDA context may already be gone.
    static auto &d_scratch = *new cuda_helpers::DeviceBuffer<Real>();
    const int n_order = args.n_order;
    const std::size_t max_shared = device_max_shared_bytes();
    const bool smem_global = tp_up_shared_bytes(n_order, 1, sizeof(Real)) > max_shared;

    auto launch_one = [&](const TuningParams &p, cudaStream_t st, bool compile_only) {
        JitKey key;
        key.name = "PtTensorprodUpKernel";
        key.real = jit_real_name<Real>();
        key.sm_major = cache.sm_major();
        key.sm_minor = cache.sm_minor();
        key.params = {{"N_ORDER", n_order},
                      {"BLOCK_SIZE", p.at("BLOCK_SIZE")},
                      {"K_TILE", p.at("K_TILE")},
                      {"SMEM_GLOBAL", smem_global ? 1 : 0}};
        auto kernel = cache.get_kernel_from_source(
            key, [&] { return make_stage_source("pt/tensorprod_up.cu", key, "", "PtTensorprodUp"); });
        const std::size_t bytes = tp_up_shared_bytes(n_order, p.at("K_TILE"), sizeof(Real));
        set_max_dynamic_smem(*kernel, smem_global ? 0 : bytes);
        if (compile_only)
            return;
        const dim3 block(p.at("BLOCK_SIZE"), 1, 1);
        if (!smem_global) {
            kernel->launch(dim3(args.n_par, args.n_charge_dim, 1), block, bytes, st, args);
            return;
        }
        // A slot per launched block, so the grid is capped and the level walked in chunks rather
        // than sizing the scratch by the parent count.
        auto ca = args;
        ca.scratch_stride = long((bytes + 15) / 16 * 16 / sizeof(Real));
        const int chunk = std::max(1, std::min(args.n_par, 2 * device_prop().multiProcessorCount / args.n_charge_dim));
        const std::size_t need = std::size_t(ca.scratch_stride) * chunk * args.n_charge_dim;
        if (need > d_scratch.size())
            d_scratch.resize(need);
        ca.scratch = d_scratch.data();
        for (int base = 0; base < args.n_par; base += chunk) {
            ca.par_boxes = args.par_boxes + base;
            ca.par_child_begin = args.par_child_begin + base;
            ca.par_child_count = args.par_child_count + base;
            ca.n_par = std::min(chunk, args.n_par - base);
            kernel->launch(dim3(ca.n_par, args.n_charge_dim, 1), block, 0, st, ca);
        }
    };

    std::ostringstream tune_key;
    tune_key << "PtTensorprodUp|real=" << jit_real_name<Real>() << "|n_order=" << n_order
             << "|n_charge_dim=" << args.n_charge_dim << "|src=" << jit::jit_source_hash("pt/tensorprod_up.cu");
    const std::string tk = tune_key.str();

    const std::vector<TuningParameter> space{{"BLOCK_SIZE", {128, 256, 512, 1024}}, {"K_TILE", {1, 2, 3, 4, 6}}};
    const TuningParams defaults{{"BLOCK_SIZE", 256}, {"K_TILE", smem_global ? 1 : 2}};

    if (auto cfg = autotune_cached(tk)) {
        launch_one(*cfg, stream, false);
        return;
    }

    const cudaDeviceProp &prop = device_prop();
    const auto constraint = [&, n_order](const TuningParams &p) {
        const int bs = p.at("BLOCK_SIZE"), kt = p.at("K_TILE");
        if (bs <= 0 || bs > prop.maxThreadsPerBlock || bs % 32 != 0 || kt <= 0)
            return false;
        return smem_global || tp_up_shared_bytes(n_order, kt, sizeof(Real)) <= max_shared;
    };
    const auto canonicalize = [&](TuningParams p) { return clamp_tiles(std::move(p), {{"K_TILE", n_order}}); };

    if (!tune_here) {
        launch_one(jit::nearest_feasible(space, defaults, constraint, canonicalize), stream, false);
        return;
    }

    autotuned_launch<Real>(tk, "PtTensorprodUpKernel", space, defaults, constraint, launch_one, args.proxy_flat,
                           proxy_count, stream, canonicalize);
}

template void launch_tensorprod<float>(dmk::cuda::TensorprodArgs<float> &, std::size_t, cudaStream_t, bool);
template void launch_tensorprod<double>(dmk::cuda::TensorprodArgs<double> &, std::size_t, cudaStream_t, bool);
template void launch_tensorprod_up<float>(dmk::cuda::TensorprodArgs<float> &, std::size_t, cudaStream_t, bool);
template void launch_tensorprod_up<double>(dmk::cuda::TensorprodArgs<double> &, std::size_t, cudaStream_t, bool);

} // namespace dmk::cuda::pt
