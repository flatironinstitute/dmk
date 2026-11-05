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

// Must match the kernel's k_pad: A^T rows padded so the last k2 tile stays in its row.
int pw2proxy_k_pad(int n_order, int k2r) { return ((n_order + k2r - 1) / k2r * k2r + 3) / 4 * 4; }

std::size_t pw2proxy_shared_bytes(int max_n_pw, int max_n_pw2, int max_n_order, int k2r, bool pencil_smem, bool slab,
                                  std::size_t sizeof_real) {
    const std::size_t complex_count = std::size_t(max_n_pw) * pw2proxy_k_pad(max_n_order, k2r) +
                                      std::size_t(max_n_pw) * max_n_order +
                                      (slab ? std::size_t(max_n_pw) * max_n_pw : 0);
    // The unpacked pencil table and the per-slab ranges lead the block, as int4 so they stay
    // 16-byte aligned.
    const std::size_t int4_count = (pencil_smem ? std::size_t(max_n_pw) * max_n_pw2 : 0) + max_n_pw2;
    return int4_count * 4 * sizeof(int) + complex_count * (2 * sizeof_real);
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
    // Never destroyed: a static DeviceBuffer's destructor would free device memory from a
    // static destructor, by which point the CUDA context may already be gone.
    static auto &d_args = *new cuda_helpers::DeviceBuffer<dmk::cuda::PwToProxyArgs<Real>>();
    d_args.upload_async_grow(args_h.data(), args_h.size(), stream);
    const int n_args = static_cast<int>(args_h.size());
    const auto a0 = args_h[0];

    const std::size_t max_shared = device_max_shared_bytes();
    // Keep the unpacked table only where it still fits beside everything else; past that it is
    // the difference between fitting in shared and having no runnable config at all.
    const bool pencil_smem =
        pw2proxy_shared_bytes(max_n_pw, max_n_pw2, max_n_order, 1, true, true, sizeof(Real)) <= max_shared;
    // At the highest double-precision digit counts A^T, H and the slab outgrow the per-block shared
    // limit. The kernel then runs against a private slice of a global buffer and reads modes in place.
    const bool smem_global =
        pw2proxy_shared_bytes(max_n_pw, max_n_pw2, max_n_order, 1, false, true, sizeof(Real)) > max_shared;
    // Never destroyed: a static DeviceBuffer's destructor would free device memory from a
    // static destructor, by which point the CUDA context may already be gone.
    static auto &d_scratch = *new cuda_helpers::DeviceBuffer<unsigned char>();

    const auto shared_bytes = [&](const TuningParams &p) {
        return pw2proxy_shared_bytes(max_n_pw, max_n_pw2, max_n_order, p.at("K2R"), pencil_smem, !smem_global,
                                     sizeof(Real));
    };
    // A thread's k3 columns stay in registers while they fit; past that they accumulate in place
    // in the output, which only the high-digit double configs reach.
    const auto acc_reg = [&](const TuningParams &p) {
        const int k2r = p.at("K2R");
        const int items = a0.n_order * ((a0.n_order + k2r - 1) / k2r);
        const int rounds = (items + p.at("BLOCK_SIZE") - 1) / p.at("BLOCK_SIZE");
        return rounds * k2r * a0.n_order * int(sizeof(Real) / sizeof(float)) <= 96;
    };

    auto launch_one = [&](const TuningParams &p, cudaStream_t st, bool compile_only) {
        const std::size_t shared = shared_bytes(p);

        JitKey key;
        key.name = "PtPwToProxyMultiLevelKernel";
        key.real = jit_real_name<Real>();
        key.sm_major = cache.sm_major();
        key.sm_minor = cache.sm_minor();
        key.params = {{"N_ORDER", a0.n_order},
                      {"N_PW", a0.n_pw},
                      {"N_PW2", a0.n_pw2},
                      {"N_CHARGE_DIM", a0.n_charge_dim},
                      {"K2R", p.at("K2R")},
                      {"M2R", p.at("M2R")},
                      {"BLOCK_SIZE", p.at("BLOCK_SIZE")},
                      {"MIN_BLOCKS", p.at("MIN_BLOCKS")},
                      {"ACC_REG", acc_reg(p) ? 1 : 0},
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

    const std::vector<TuningParameter> space{
        {"K2R", {1, 2, 3}}, {"M2R", {1, 2}}, {"BLOCK_SIZE", {128, 192, 256, 384}}, {"MIN_BLOCKS", {1, 2, 4}}};
    const TuningParams defaults{{"K2R", 2}, {"M2R", 1}, {"BLOCK_SIZE", 192}, {"MIN_BLOCKS", 1}};

    const auto constraint = [&](const TuningParams &p) {
        const int bs = p.at("BLOCK_SIZE");
        if (bs <= 0 || bs > prop.maxThreadsPerBlock || bs % 32 != 0)
            return false;
        if (p.at("K2R") <= 0 || p.at("M2R") <= 0)
            return false;
        if (p.at("MIN_BLOCKS") > resident_blocks_per_sm(smem_global ? 0 : shared_bytes(p), bs))
            return false;
        return smem_global || shared_bytes(p) <= max_shared;
    };

    const auto canonicalize = [&](TuningParams p) {
        return clamp_tiles(std::move(p), {{"K2R", a0.n_order}, {"M2R", a0.n_pw}});
    };

    autotuned_launch<Real>(tk, "PtPwToProxyMultiLevelKernel", space, defaults, constraint, launch_one, proxy_flat,
                           proxy_count, stream, canonicalize);
}

template void launch_pw2proxy<float>(std::vector<dmk::cuda::PwToProxyArgs<float>> &, float *, std::size_t, cudaStream_t,
                                     std::string_view);
template void launch_pw2proxy<double>(std::vector<dmk::cuda::PwToProxyArgs<double>> &, double *, std::size_t,
                                      cudaStream_t, std::string_view);

} // namespace dmk::cuda::pt
