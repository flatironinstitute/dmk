#include "proxy2pw.hpp"

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"

#include <dmk/cuda/helpers.hpp>

#include <algorithm>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace dmk::cuda::pt {
namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

std::size_t p2pw_shared_bytes(int n_order, int n_pw, int z_tile, int ff2_copies, std::size_t sizeof_real) {
    const std::size_t complex_count = std::size_t(z_tile) * std::size_t(n_order) * n_order +
                                      std::size_t(ff2_copies) * z_tile * std::size_t(n_order) * n_pw +
                                      std::size_t(n_order) * n_pw;
    return std::size_t{2} * complex_count * sizeof_real;
}

constexpr int Z_TILE_MIN = 2;

} // namespace

std::size_t proxy2pw_min_shared_bytes(int n_order, int n_pw, int ff2_copies, std::size_t sizeof_real) {
    return p2pw_shared_bytes(n_order, n_pw, Z_TILE_MIN, ff2_copies, sizeof_real);
}

template <typename Real>
void launch_proxy2pw(std::vector<dmk::cuda::Proxy2PwArgs<Real>> &args_h, cudaStream_t stream,
                     std::string_view variant) {
    if (args_h.empty())
        return;
    int max_boxes = 0, max_n_order = 0, max_n_pw = 0;
    for (const auto &a : args_h) {
        max_boxes = std::max(max_boxes, a.n_boxes_at_level);
        max_n_order = std::max(max_n_order, a.n_order);
        max_n_pw = std::max(max_n_pw, a.n_pw);
    }
    if (max_boxes == 0)
        return;

    static JitCache cache;
    static cuda_helpers::DeviceBuffer<dmk::cuda::Proxy2PwArgs<Real>> d_args;
    d_args.upload_async_grow(args_h.data(), args_h.size(), stream);
    const int n_args = static_cast<int>(args_h.size());
    const auto a0 = args_h[0];
    // The fused Stokeslet projector holds one phase-2 buffer per charge dim.
    const int ff2_copies = (a0.multiply_mode == 2) ? a0.n_charge_dim : 1;

    auto launch_one = [&](const TuningParams &p, cudaStream_t st) {
        const std::size_t shared = p2pw_shared_bytes(max_n_order, max_n_pw, p.at("Z_TILE"), ff2_copies, sizeof(Real));

        JitKey key;
        key.name = "PtProxy2PwMultiLevelKernel";
        key.real = jit_real_name<Real>();
        key.sm_major = cache.sm_major();
        key.sm_minor = cache.sm_minor();
        key.params = {{"N_ORDER", a0.n_order},
                      {"N_PW", a0.n_pw},
                      {"N_PW2", a0.n_pw2},
                      {"N_CHARGE_DIM", a0.n_charge_dim},
                      {"P2PW_MULTIPLY", a0.multiply_mode},
                      {"BLOCK_SIZE", p.at("BLOCK_SIZE")},
                      {"PROXY2PW_Z_TILE", p.at("Z_TILE")},
                      {"PROXY2PW_I_TILE", p.at("I_TILE")},
                      {"PROXY2PW_M1_TILE", p.at("M1_TILE")},
                      {"PROXY2PW_M2_TILE", p.at("M2_TILE")},
                      {"MIN_BLOCKS", resident_blocks_per_sm(shared, p.at("BLOCK_SIZE"))}};
        auto kernel = cache.get_kernel_from_source(
            key, [&] { return make_stage_source("pt/proxy2pw.cu", key, "", "PtProxy2Pw"); });
        set_max_dynamic_smem(*kernel, shared);
        const dmk::cuda::Proxy2PwArgs<Real> *dev_args = d_args.data();
        int n = n_args;
        kernel->launch(dim3(max_boxes, n_args, 1), dim3(p.at("BLOCK_SIZE"), 1, 1), shared, st, dev_args, n);
    };

    std::ostringstream tune_key;
    tune_key << "PtProxy2Pw|real=" << jit_real_name<Real>() << "|n_order=" << a0.n_order << "|n_pw=" << a0.n_pw
             << "|n_charge_dim=" << a0.n_charge_dim << "|mul=" << a0.multiply_mode << "|variant=" << variant;
    tune_key << "|src=" << jit::jit_source_hash("pt/proxy2pw.cu");
    const std::string tk = tune_key.str();

    if (auto cfg = autotune_cached(tk)) {
        launch_one(*cfg, stream);
        return;
    }

    const cudaDeviceProp &prop = device_prop();
    const std::size_t max_shared = device_max_shared_bytes();

    const std::vector<TuningParameter> space{{"BLOCK_SIZE", {64, 128, 256}},
                                             {"Z_TILE", {Z_TILE_MIN, 4}},
                                             {"I_TILE", {2, 4}},
                                             {"M1_TILE", {2, 4, 6}},
                                             {"M2_TILE", {2, 4}}};
    // A fused phase 3 holds ff2_copies accumulator tiles at once, in shared and in registers, so
    // its defaults start narrower; the tuner widens them when it runs.
    const bool wide = ff2_copies == 1 && p2pw_shared_bytes(max_n_order, max_n_pw, 4, 1, sizeof(Real)) <= max_shared;
    const TuningParams defaults{{"BLOCK_SIZE", 128},
                                {"Z_TILE", wide ? 4 : Z_TILE_MIN},
                                {"I_TILE", 4},
                                {"M1_TILE", 4},
                                {"M2_TILE", wide ? 4 : 2}};

    const auto constraint = [&](const TuningParams &p) {
        const int bs = p.at("BLOCK_SIZE"), z = p.at("Z_TILE");
        if (bs <= 0 || bs > prop.maxThreadsPerBlock || bs % 32 != 0)
            return false;
        if (z <= 0 || p.at("I_TILE") <= 0 || p.at("M1_TILE") <= 0 || p.at("M2_TILE") <= 0)
            return false;
        return p2pw_shared_bytes(max_n_order, max_n_pw, z, ff2_copies, sizeof(Real)) <= max_shared;
    };

    autotuned_launch<Real>(tk, "PtProxy2PwMultiLevelKernel", space, defaults, constraint, launch_one,
                           /*snapshot_base=*/static_cast<Real *>(nullptr), 0, stream);
}

template void launch_proxy2pw<float>(std::vector<dmk::cuda::Proxy2PwArgs<float>> &, cudaStream_t, std::string_view);
template void launch_proxy2pw<double>(std::vector<dmk::cuda::Proxy2PwArgs<double>> &, cudaStream_t, std::string_view);

} // namespace dmk::cuda::pt
