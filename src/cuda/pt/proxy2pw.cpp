#include "proxy2pw.hpp"

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

std::size_t p2pw_shared_bytes(int n_order, int n_pw, int z_tile, std::size_t sizeof_real) {
    const std::size_t complex_count =
        std::size_t(z_tile) * (std::size_t(n_order) * n_order + std::size_t(n_order) * n_pw) +
        std::size_t(n_order) * n_pw;
    return std::size_t{2} * complex_count * sizeof_real;
}

} // namespace

template <typename Real>
void launch_proxy2pw(std::vector<dmk::cuda::Proxy2PwArgs<Real>> &args_h, cudaStream_t stream) {
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
    d_args.upload_async(args_h.data(), args_h.size(), stream);
    const int n_args = static_cast<int>(args_h.size());
    const auto a0 = args_h[0];

    auto launch_one = [&](const TuningParams &p, cudaStream_t st) {
        JitKey key;
        key.name = "PtProxy2PwMultiLevelKernel";
        key.real = jit_real_name<Real>();
        key.sm_major = cache.sm_major();
        key.sm_minor = cache.sm_minor();
        key.params = {{"N_ORDER", a0.n_order},
                      {"N_PW", a0.n_pw},
                      {"N_PW2", a0.n_pw2},
                      {"N_CHARGE_DIM", a0.n_charge_dim},
                      {"BLOCK_SIZE", p.at("BLOCK_SIZE")},
                      {"PROXY2PW_Z_TILE", p.at("Z_TILE")},
                      {"PROXY2PW_I_TILE", p.at("I_TILE")},
                      {"PROXY2PW_M1_TILE", p.at("M1_TILE")},
                      {"PROXY2PW_M2_TILE", p.at("M2_TILE")}};
        const std::string source = make_stage_source("pt/proxy2pw.cu", key, "", "PtProxy2Pw");
        auto kernel = cache.get_kernel_from_source(key, source);
        const std::size_t shared = p2pw_shared_bytes(max_n_order, max_n_pw, p.at("Z_TILE"), sizeof(Real));
        set_max_dynamic_smem(*kernel, shared);
        const dmk::cuda::Proxy2PwArgs<Real> *dev_args = d_args.data();
        int n = n_args;
        kernel->launch(dim3(max_boxes, n_args, 1), dim3(p.at("BLOCK_SIZE"), 1, 1), shared, st, dev_args, n);
    };

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    const std::size_t max_shared = prop.sharedMemPerBlockOptin > 0 ? std::size_t(prop.sharedMemPerBlockOptin)
                                                                   : std::size_t(prop.sharedMemPerBlock);

    const std::vector<TuningParameter> space{{"BLOCK_SIZE", {64, 128, 256}},
                                             {"Z_TILE", {2, 4}},
                                             {"I_TILE", {2, 4}},
                                             {"M1_TILE", {2, 4}},
                                             {"M2_TILE", {2, 4}}};
    const TuningParams defaults{{"BLOCK_SIZE", 128}, {"Z_TILE", 4}, {"I_TILE", 4}, {"M1_TILE", 4}, {"M2_TILE", 4}};

    const auto constraint = [&](const TuningParams &p) {
        const int bs = p.at("BLOCK_SIZE"), z = p.at("Z_TILE");
        if (bs <= 0 || bs > prop.maxThreadsPerBlock || bs % 32 != 0)
            return false;
        if (z <= 0 || p.at("I_TILE") <= 0 || p.at("M1_TILE") <= 0 || p.at("M2_TILE") <= 0)
            return false;
        return p2pw_shared_bytes(max_n_order, max_n_pw, z, sizeof(Real)) <= max_shared;
    };

    std::ostringstream tune_key;
    tune_key << "PtProxy2Pw|real=" << jit_real_name<Real>() << "|n_order=" << a0.n_order << "|n_pw=" << a0.n_pw
             << "|n_charge_dim=" << a0.n_charge_dim;

    autotuned_launch<Real>(tune_key.str(), "PtProxy2PwMultiLevelKernel", space, defaults, constraint, launch_one,
                           /*snapshot_base=*/static_cast<Real *>(nullptr), 0, stream);
}

template void launch_proxy2pw<float>(std::vector<dmk::cuda::Proxy2PwArgs<float>> &, cudaStream_t);
template void launch_proxy2pw<double>(std::vector<dmk::cuda::Proxy2PwArgs<double>> &, cudaStream_t);

} // namespace dmk::cuda::pt
