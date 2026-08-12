#include "pw2proxy.hpp"

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

std::size_t pw2proxy_shared_bytes(int max_n_pw, int max_n_pw2, int max_n_order, int k3_tile, bool pencil_smem,
                                  std::size_t sizeof_real) {
    const int max_k_pad = ((max_n_order + 3) / 4) * 4;
    const int max_phase1_cols = max_n_pw * max_n_pw;
    const std::size_t complex_count = std::size_t(max_n_pw) * std::size_t(max_k_pad) +
                                      std::size_t(k3_tile) * std::size_t(max_phase1_cols) +
                                      std::size_t(k3_tile) * std::size_t(max_n_order) * std::size_t(max_n_pw);
    // The unpacked pencil table leads the block, as int4 so it stays 16-byte aligned.
    const std::size_t pencil_bytes = pencil_smem ? std::size_t(max_n_pw) * max_n_pw2 * 4 * sizeof(int) : 0;
    return pencil_bytes + complex_count * (2 * sizeof_real);
}

} // namespace

template <typename Real>
void launch_pw2proxy(std::vector<dmk::cuda::PwToProxyArgs<Real>> &args_h, Real *proxy_flat, std::size_t proxy_count,
                     cudaStream_t stream, std::string_view variant) {
    if (args_h.empty())
        return;
    int max_boxes = 0, max_n_order = 0, max_n_pw = 0, max_n_pw2 = 0;
    for (const auto &a : args_h) {
        max_boxes = std::max(max_boxes, a.n_boxes_at_level);
        max_n_order = std::max(max_n_order, a.n_order);
        max_n_pw = std::max(max_n_pw, a.n_pw);
        max_n_pw2 = std::max(max_n_pw2, a.n_pw2);
    }
    if (max_boxes == 0)
        return;

    static JitCache cache;
    static cuda_helpers::DeviceBuffer<dmk::cuda::PwToProxyArgs<Real>> d_args;
    d_args.upload_async_grow(args_h.data(), args_h.size(), stream);
    const int n_args = static_cast<int>(args_h.size());
    const auto a0 = args_h[0];

    const std::size_t max_shared = device_max_shared_bytes();
    // Keep the unpacked table only where the narrowest tiling can still afford it; past that it is
    // the difference between fitting in shared and having no runnable config at all.
    const bool pencil_smem =
        pw2proxy_shared_bytes(max_n_pw, max_n_pw2, max_n_order, 1, true, sizeof(Real)) <= max_shared;
    // Past ~9 significant digits the phase buffers alone outgrow any per-block shared limit. The
    // kernel then runs against a private slice of a global buffer instead, at the narrowest tiling
    // so the slice stays small.
    const bool smem_global =
        pw2proxy_shared_bytes(max_n_pw, max_n_pw2, max_n_order, 1, false, sizeof(Real)) > max_shared;
    static cuda_helpers::DeviceBuffer<unsigned char> d_scratch;

    auto launch_one = [&](const TuningParams &p, cudaStream_t st, bool compile_only) {
        const std::size_t shared =
            pw2proxy_shared_bytes(max_n_pw, max_n_pw2, max_n_order, p.at("K3_TILE"), pencil_smem, sizeof(Real));

        // Occupancy is register-bound once the working set leaves shared, and asking for the
        // thread-limited maximum there would only buy blocks by spilling.
        const int min_blocks = smem_global ? 1 : resident_blocks_per_sm(shared, p.at("BLOCK_SIZE"));

        JitKey key;
        key.name = "PtPwToProxyMultiLevelKernel";
        key.real = jit_real_name<Real>();
        key.sm_major = cache.sm_major();
        key.sm_minor = cache.sm_minor();
        key.params = {{"N_ORDER", a0.n_order},
                      {"N_PW", a0.n_pw},
                      {"N_PW2", a0.n_pw2},
                      {"N_CHARGE_DIM", a0.n_charge_dim},
                      {"COL_REG", p.at("COL_REG")},
                      {"K1_TILE", p.at("K1_TILE")},
                      {"K2_TILE", p.at("K2_TILE")},
                      {"K3_TILE", p.at("K3_TILE")},
                      {"KR_TILE", p.at("KR_TILE")},
                      {"BLOCK_SIZE", p.at("BLOCK_SIZE")},
                      {"MIN_BLOCKS", min_blocks},
                      {"PENCIL_SMEM", pencil_smem ? 1 : 0},
                      {"SMEM_GLOBAL", smem_global ? 1 : 0}};
        auto kernel = cache.get_kernel_from_source(
            key, [&] { return make_stage_source("pt/pw2proxy.cu", key, "", "PtPwToProxy"); });
        set_max_dynamic_smem(*kernel, smem_global ? 0 : shared);
        if (compile_only)
            return;
        const dmk::cuda::PwToProxyArgs<Real> *dev_args = d_args.data();
        int n = n_args;
        unsigned char *scratch = nullptr;
        long stride = 0;
        int box_base = 0;
        if (!smem_global) {
            kernel->launch(dim3(max_boxes, n_args, 1), dim3(p.at("BLOCK_SIZE"), 1, 1), shared, st, dev_args, n, scratch,
                           stride, box_base);
            return;
        }
        // A slot per launched block, so the grid is capped and the level walked in chunks rather
        // than sizing the scratch by the box count.
        stride = static_cast<long>((shared + 15) & ~std::size_t{15});
        const int slots_x = std::max(1, std::min(max_boxes, 2 * device_prop().multiProcessorCount / n_args));
        const std::size_t need = static_cast<std::size_t>(stride) * slots_x * n_args;
        if (need > d_scratch.size())
            d_scratch.resize(need);
        scratch = d_scratch.data();
        for (box_base = 0; box_base < max_boxes; box_base += slots_x) {
            const int nx = std::min(slots_x, max_boxes - box_base);
            kernel->launch(dim3(nx, n_args, 1), dim3(p.at("BLOCK_SIZE"), 1, 1), 0, st, dev_args, n, scratch, stride,
                           box_base);
        }
    };

    std::ostringstream tune_key;
    tune_key << "PtPwToProxy|real=" << jit_real_name<Real>() << "|n_order=" << a0.n_order << "|n_pw=" << a0.n_pw
             << "|n_charge_dim=" << a0.n_charge_dim << "|variant=" << variant;
    tune_key << "|src=" << jit::jit_source_hash("pt/pw2proxy.cu");
    const std::string tk = tune_key.str();

    if (auto cfg = autotune_cached(tk)) {
        launch_one(*cfg, stream, false);
        return;
    }

    const cudaDeviceProp &prop = device_prop();

    const std::vector<TuningParameter> space{{"COL_REG", {1, 2}},       {"K1_TILE", {1, 2, 3, 4}},
                                             {"K2_TILE", {2, 3, 4}},    {"K3_TILE", {1, 2, 3, 4}},
                                             {"KR_TILE", {3, 4, 8, 9}}, {"BLOCK_SIZE", {128, 256}}};
    const TuningParams defaults{{"COL_REG", 1}, {"K1_TILE", 2}, {"K2_TILE", 3},
                                {"K3_TILE", 3}, {"KR_TILE", 3}, {"BLOCK_SIZE", 256}};

    const auto constraint = [&](const TuningParams &p) {
        const int bs = p.at("BLOCK_SIZE");
        if (bs <= 0 || bs > prop.maxThreadsPerBlock || bs % 32 != 0)
            return false;
        if (p.at("COL_REG") <= 0 || p.at("K1_TILE") <= 0 || p.at("K2_TILE") <= 0 || p.at("K3_TILE") <= 0 ||
            p.at("KR_TILE") <= 0)
            return false;
        if (smem_global)
            return p.at("K3_TILE") == 1;
        return pw2proxy_shared_bytes(max_n_pw, max_n_pw2, max_n_order, p.at("K3_TILE"), pencil_smem, sizeof(Real)) <=
               max_shared;
    };

    const auto canonicalize = [&](TuningParams p) {
        return clamp_tiles(
            std::move(p),
            {{"K1_TILE", a0.n_order}, {"K2_TILE", a0.n_order}, {"K3_TILE", a0.n_order}, {"KR_TILE", a0.n_pw}});
    };

    autotuned_launch<Real>(tk, "PtPwToProxyMultiLevelKernel", space, defaults, constraint, launch_one, proxy_flat,
                           proxy_count, stream, canonicalize);
}

template void launch_pw2proxy<float>(std::vector<dmk::cuda::PwToProxyArgs<float>> &, float *, std::size_t, cudaStream_t,
                                     std::string_view);
template void launch_pw2proxy<double>(std::vector<dmk::cuda::PwToProxyArgs<double>> &, double *, std::size_t,
                                      cudaStream_t, std::string_view);

} // namespace dmk::cuda::pt
