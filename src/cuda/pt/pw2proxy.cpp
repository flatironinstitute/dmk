#include "pw2proxy.hpp"

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"

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

std::size_t pw2proxy_shared_bytes(int max_n_pw, int max_n_order, int k3_tile, std::size_t sizeof_real) {
    const int max_k_pad = ((max_n_order + 3) / 4) * 4;
    const int max_phase1_cols = max_n_pw * max_n_pw;
    const std::size_t complex_count = std::size_t(max_n_pw) * std::size_t(max_k_pad) +
                                      std::size_t(k3_tile) * std::size_t(max_phase1_cols) +
                                      std::size_t(k3_tile) * std::size_t(max_n_order) * std::size_t(max_n_pw);
    return complex_count * (2 * sizeof_real);
}

} // namespace

template <typename Real>
void launch_pw2proxy(std::vector<dmk::cuda::PwToProxyArgs<Real>> &args_h, Real *proxy_flat, std::size_t proxy_count,
                     cudaStream_t stream) {
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
    static cuda_helpers::DeviceBuffer<dmk::cuda::PwToProxyArgs<Real>> d_args;
    d_args.upload_async_grow(args_h.data(), args_h.size(), stream);
    const int n_args = static_cast<int>(args_h.size());
    const auto a0 = args_h[0];

    auto launch_one = [&](const TuningParams &p, cudaStream_t st) {
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
                      {"K2_TILE", p.at("K2_TILE")},
                      {"K3_TILE", p.at("K3_TILE")},
                      {"KR_TILE", p.at("KR_TILE")},
                      {"BLOCK_SIZE", p.at("BLOCK_SIZE")}};
        auto kernel = cache.get_kernel_from_source(
            key, [&] { return make_stage_source("pt/pw2proxy.cu", key, "", "PtPwToProxy"); });
        const std::size_t shared = pw2proxy_shared_bytes(max_n_pw, max_n_order, p.at("K3_TILE"), sizeof(Real));
        set_max_dynamic_smem(*kernel, shared);
        const dmk::cuda::PwToProxyArgs<Real> *dev_args = d_args.data();
        int n = n_args;
        kernel->launch(dim3(max_boxes, n_args, 1), dim3(p.at("BLOCK_SIZE"), 1, 1), shared, st, dev_args, n);
    };

    const cudaDeviceProp &prop = device_prop();
    const std::size_t max_shared = device_max_shared_bytes();

    const std::vector<TuningParameter> space{{"COL_REG", {1, 2}},
                                             {"K2_TILE", {2, 4}},
                                             {"K3_TILE", {2, 3, 4}},
                                             {"KR_TILE", {4, 8, 9}},
                                             {"BLOCK_SIZE", {128, 256}}};
    const TuningParams defaults{{"COL_REG", 1}, {"K2_TILE", 2}, {"K3_TILE", 3}, {"KR_TILE", 9}, {"BLOCK_SIZE", 256}};

    const auto constraint = [&](const TuningParams &p) {
        const int bs = p.at("BLOCK_SIZE");
        if (bs <= 0 || bs > prop.maxThreadsPerBlock || bs % 32 != 0)
            return false;
        if (p.at("COL_REG") <= 0 || p.at("K2_TILE") <= 0 || p.at("K3_TILE") <= 0 || p.at("KR_TILE") <= 0)
            return false;
        return pw2proxy_shared_bytes(max_n_pw, max_n_order, p.at("K3_TILE"), sizeof(Real)) <= max_shared;
    };

    std::ostringstream tune_key;
    tune_key << "PtPwToProxy|real=" << jit_real_name<Real>() << "|n_order=" << a0.n_order << "|n_pw=" << a0.n_pw
             << "|n_charge_dim=" << a0.n_charge_dim;

    autotuned_launch<Real>(tune_key.str(), "PtPwToProxyMultiLevelKernel", space, defaults, constraint, launch_one,
                           proxy_flat, proxy_count, stream);
}

template void launch_pw2proxy<float>(std::vector<dmk::cuda::PwToProxyArgs<float>> &, float *, std::size_t,
                                     cudaStream_t);
template void launch_pw2proxy<double>(std::vector<dmk::cuda::PwToProxyArgs<double>> &, double *, std::size_t,
                                      cudaStream_t);

} // namespace dmk::cuda::pt
