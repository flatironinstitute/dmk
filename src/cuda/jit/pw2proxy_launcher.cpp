#include "pw2proxy_launcher.hpp"

#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/pw2proxy_kernels.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda::jit {
namespace {

std::string make_prelude(const JitKey& key) {
    const int k1_tile   = required_int_param(key, "K1_TILE", "PwToProxy");
    const int col_reg   = required_int_param(key, "COL_REG", "PwToProxy");
    const int k2_tile   = required_int_param(key, "K2_TILE", "PwToProxy");
    const int k3_tile   = required_int_param(key, "K3_TILE", "PwToProxy");
    const int kr_tile   = required_int_param(key, "KR_TILE", "PwToProxy");
    const int blocksize = required_int_param(key, "BLOCK_SIZE", "PwToProxy");
    const int n_order      = required_int_param(key, "N_ORDER", "PwToProxy");
    const int n_pw         = required_int_param(key, "N_PW", "PwToProxy");
    const int n_pw2        = required_int_param(key, "N_PW2", "PwToProxy");
    const int n_charge_dim = required_int_param(key, "N_CHARGE_DIM", "PwToProxy");

    std::ostringstream ss;

    ss << "#include <dmk/cuda/pw2proxy_kernelargs.hpp>\n\n";

    ss << "using dmk::cuda::PwToProxyArgs;\n";

    ss << "using Real = " << key.real << ";\n\n";

    ss << "constexpr int K1_TILE   = " << k1_tile << ";\n";
    ss << "constexpr int COL_REG   = " << col_reg << ";\n";
    ss << "constexpr int K2_TILE   = " << k2_tile << ";\n";
    ss << "constexpr int K3_TILE   = " << k3_tile << ";\n";
    ss << "constexpr int KR_TILE   = " << kr_tile << ";\n";
    ss << "constexpr int N_ORDER      = " << n_order << ";\n";
    ss << "constexpr int N_PW         = " << n_pw << ";\n";
    ss << "constexpr int N_PW2        = " << n_pw2 << ";\n";
    ss << "constexpr int N_CHARGE_DIM = " << n_charge_dim << ";\n";
    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n\n";

    return ss.str();
}

std::size_t pw_to_proxy_shared_bytes(
    int max_n_pw,
    int max_n_pw2,
    int max_n_order,
    int k1_tile,
    std::size_t complex_size
) {
    const int max_k_pad = ((max_n_order + 3) / 4) * 4;
    const int max_phase1_cols = max_n_pw * max_n_pw2;

    const std::size_t complex_count =
        std::size_t(max_n_pw) * std::size_t(max_k_pad) +
        std::size_t(k1_tile) * std::size_t(max_phase1_cols) +
        std::size_t(k1_tile) * std::size_t(max_n_pw2) * std::size_t(max_n_order);

    return complex_count * complex_size;
}

template <typename Real>
void check_pw_to_proxy_shape_or_throw(
    const dmk::cuda::PwToProxyArgs<Real>& a,
    const char* where
) {
    auto fail = [&](const char* field) {
        throw std::runtime_error(
            std::string("PwToProxy JIT: invalid args/shape in ") +
            where +
            " field=" + field +
            " n_boxes_at_level=" + std::to_string(a.n_boxes_at_level) +
            " n_order=" + std::to_string(a.n_order) +
            " n_pw=" + std::to_string(a.n_pw) +
            " n_pw2=" + std::to_string(a.n_pw2) +
            " n_charge_dim=" + std::to_string(a.n_charge_dim) +
            " pw_in_stride=" + std::to_string(a.pw_in_stride) +
            " box_ids=" + std::to_string(reinterpret_cast<std::uintptr_t>(a.box_ids)) +
            " pw_in_pool=" + std::to_string(reinterpret_cast<std::uintptr_t>(a.pw_in_pool)) +
            " pw2poly=" + std::to_string(reinterpret_cast<std::uintptr_t>(a.pw2poly)) +
            " proxy_flat=" + std::to_string(reinterpret_cast<std::uintptr_t>(a.proxy_flat)) +
            " proxy_offsets=" + std::to_string(reinterpret_cast<std::uintptr_t>(a.proxy_offsets))
        );
    };

    if (a.n_boxes_at_level < 0) fail("n_boxes_at_level");
    if (a.n_order <= 0) fail("n_order");
    if (a.n_pw <= 0) fail("n_pw");
    if (a.n_pw2 <= 0) fail("n_pw2");
    if (a.n_charge_dim <= 0) fail("n_charge_dim");
    if (a.n_boxes_at_level > 1 && a.pw_in_stride <= 0) {
        fail("pw_in_stride");
    }
    if (a.box_ids == nullptr) fail("box_ids");
    if (a.pw_in_pool == nullptr) fail("pw_in_pool");
    if (a.pw2poly == nullptr) fail("pw2poly");
    if (a.proxy_flat == nullptr) fail("proxy_flat");
    if (a.proxy_offsets == nullptr) fail("proxy_offsets");
}

void set_dynamic_smem_if_needed(
    const JitKernel& kernel,
    std::size_t shared_bytes,
    const char* label
) {
    if (shared_bytes <= 48 * 1024) {
        return;
    }

    CUresult res = cuFuncSetAttribute(
        kernel.function(),
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        static_cast<int>(shared_bytes)
    );

    if (res != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* msg = nullptr;

        cuGetErrorName(res, &name);
        cuGetErrorString(res, &msg);

        throw std::runtime_error(
            std::string(label) +
            ": cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) failed: " +
            (name ? name : "<unknown>") + ": " +
            (msg ? msg : "<no message>") +
            " shared_bytes=" + std::to_string(shared_bytes)
        );
    }
}

} // namespace

std::string make_pw2proxy_source(const JitKey& key) {
    std::filesystem::path filename;

    if (key.name == "PwToProxyKernel") {
        filename = "pw2proxy.cu";
    } else if (key.name == "PwToProxyMultiLevelKernel") {
        filename = "pw2proxy_multilevel.cu";
    } else {
        throw std::runtime_error(
            "PwToProxy JIT: unknown kernel name: " + key.name
        );
    }

    const SplitSource split = load_split_jit_source(filename.string(), "PwToProxy");

    std::ostringstream generated;

    generated << make_prelude(key);
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real>
void launch_pw_to_proxy_jit(
    JitCache& cache,
    const dmk::cuda::PwToProxyArgs<Real>& args,
    cudaStream_t stream,
    int k1_tile,
    int col_reg,
    int k2_tile,
    int k3_tile,
    int kr_tile,
    int blocksize
) {
    if (args.n_boxes_at_level == 0) {
        return;
    }

    check_pw_to_proxy_shape_or_throw(args, "single");

    JitKey key;
    key.name = "PwToProxyKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"K1_TILE", k1_tile},
        {"COL_REG", col_reg},
        {"K2_TILE", k2_tile},
        {"K3_TILE", k3_tile},
        {"KR_TILE", kr_tile},
        {"N_ORDER", args.n_order},
        {"N_CHARGE_DIM", args.n_charge_dim},
        {"N_PW", args.n_pw},
        {"N_PW2", args.n_pw2},
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes =
        pw_to_proxy_shared_bytes(
            args.n_pw,
            args.n_pw2,
            args.n_order,
            k1_tile,
            sizeof(dmk::cuda_helpers::complx<Real>)
        );

    set_dynamic_smem_if_needed(
        *kernel,
        shared_bytes,
        "launch_pw_to_proxy_jit"
    );

    kernel->launch(
        dim3(args.n_boxes_at_level, 1, 1),
        dim3(blocksize, 1, 1),
        shared_bytes,
        stream,
        args
    );
}

template <typename Real>
void launch_pw_to_proxy_multilevel_jit(
    JitCache& cache,
    const std::vector<dmk::cuda::PwToProxyArgs<Real>>& args_h,
    dmk::cuda::PwToProxyArgs<Real>* d_args_scratch,
    cudaStream_t stream,
    int k1_tile,
    int col_reg,
    int k2_tile,
    int k3_tile,
    int kr_tile,
    int blocksize
) {
    if (args_h.empty()) {
        return;
    }

    int max_boxes = 0;
    int max_n_pw = 0;
    int max_n_pw2 = 0;
    int max_n_order = 0;

    for (const auto& a : args_h) {
        if (a.n_boxes_at_level == 0) {
            continue;
        }

        check_pw_to_proxy_shape_or_throw(a, "multilevel");

        max_boxes = std::max(max_boxes, a.n_boxes_at_level);
        max_n_pw = std::max(max_n_pw, a.n_pw);
        max_n_pw2 = std::max(max_n_pw2, a.n_pw2);
        max_n_order = std::max(max_n_order, a.n_order);
    }

    if (max_boxes == 0) {
        return;
    }

    DMK_CHECK_CUDA(
        cudaMemcpyAsync(
            d_args_scratch,
            args_h.data(),
            args_h.size() * sizeof(dmk::cuda::PwToProxyArgs<Real>),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    JitKey key;
    key.name = "PwToProxyMultiLevelKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"K1_TILE", k1_tile},
        {"COL_REG", col_reg},
        {"K2_TILE", k2_tile},
        {"K3_TILE", k3_tile},
        {"KR_TILE", kr_tile},
        {"N_ORDER", args_h[0].n_order},
        {"N_CHARGE_DIM", args_h[0].n_charge_dim},
        {"N_PW", args_h[0].n_pw},
        {"N_PW2", args_h[0].n_pw2},
        {"BLOCK_SIZE", blocksize},

    };

    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes =
        pw_to_proxy_shared_bytes(
            max_n_pw,
            max_n_pw2,
            max_n_order,
            k1_tile,
            sizeof(dmk::cuda_helpers::complx<Real>)
        );

    set_dynamic_smem_if_needed(
        *kernel,
        shared_bytes,
        "launch_pw_to_proxy_multilevel_jit"
    );

    const int n_args = static_cast<int>(args_h.size());

    kernel->launch(
        dim3(max_boxes, n_args, 1),
        dim3(blocksize, 1, 1),
        shared_bytes,
        stream,
        d_args_scratch,
        n_args
    );
}

template void launch_pw_to_proxy_jit<float>(
    JitCache&,
    const dmk::cuda::PwToProxyArgs<float>&,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int,
    int
);

template void launch_pw_to_proxy_jit<double>(
    JitCache&,
    const dmk::cuda::PwToProxyArgs<double>&,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int,
    int
);

template void launch_pw_to_proxy_multilevel_jit<float>(
    JitCache&,
    const std::vector<dmk::cuda::PwToProxyArgs<float>>&,
    dmk::cuda::PwToProxyArgs<float>*,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int,
    int
);

template void launch_pw_to_proxy_multilevel_jit<double>(
    JitCache&,
    const std::vector<dmk::cuda::PwToProxyArgs<double>>&,
    dmk::cuda::PwToProxyArgs<double>*,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int,
    int
);

} // namespace dmk::cuda::jit

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_pw_to_proxy(const PwToProxyArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_pw_to_proxy_jit<Real>(
        jit_cache,
        args,
        stream,
        6,  // K1_TILE
        2,  // COL_REG
        2,  // K2_TILE
        3,  // K3_TILE
        6,  // KR_TILE
        block_size
    );
}

template void launch_pw_to_proxy<float, 2>(const PwToProxyArgs<float> &, cudaStream_t);
template void launch_pw_to_proxy<float, 3>(const PwToProxyArgs<float> &, cudaStream_t);
template void launch_pw_to_proxy<double, 2>(const PwToProxyArgs<double> &, cudaStream_t);
template void launch_pw_to_proxy<double, 3>(const PwToProxyArgs<double> &, cudaStream_t);

template <typename Real, int DIM>
void launch_pw_to_proxy_multilevel(
    const std::vector<PwToProxyArgs<Real>> &args_h,
    PwToProxyArgs<Real> *d_args_scratch,
    cudaStream_t stream
) {
    if (args_h.empty())
        return;

    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_pw_to_proxy_multilevel_jit<Real>(
        jit_cache,
        args_h,
        d_args_scratch,
        stream,
        18,  // K1_TILE
        1,   // COL_REG
        2,   // K2_TILE
        3,   // K3_TILE
        9,   // KR_TILE
        256  // block_size
    );
}

template void launch_pw_to_proxy_multilevel<float, 2>(const std::vector<PwToProxyArgs<float>> &,
                                                      PwToProxyArgs<float> *, cudaStream_t);
template void launch_pw_to_proxy_multilevel<float, 3>(const std::vector<PwToProxyArgs<float>> &,
                                                      PwToProxyArgs<float> *, cudaStream_t);
template void launch_pw_to_proxy_multilevel<double, 2>(const std::vector<PwToProxyArgs<double>> &,
                                                       PwToProxyArgs<double> *, cudaStream_t);
template void launch_pw_to_proxy_multilevel<double, 3>(const std::vector<PwToProxyArgs<double>> &,
                                                       PwToProxyArgs<double> *, cudaStream_t);

} // namespace dmk::cuda
