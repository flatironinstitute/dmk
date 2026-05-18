#include "pw2proxy_launcher.hpp"

#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_types.hpp"

#ifdef DMK_CUDA_USE_NVRTC_JIT
#include <dmk_jit_config.hpp>
#endif

#include <dmk/cuda/helpers.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda::jit {
namespace {

template <typename Real>
const char* real_name();

template <>
const char* real_name<float>() {
    return "float";
}

template <>
const char* real_name<double>() {
    return "double";
}

int get_required_param(const JitKey& key, const char* name) {
    const auto it = key.params.find(name);

    if (it == key.params.end()) {
        throw std::runtime_error(
            std::string("PwToProxy JIT key missing parameter: ") + name
        );
    }

    return it->second;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "PwToProxy JIT: failed to open source file: " + path.string()
        );
    }

    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct SplitSource {
    std::string header;
    std::string kernel;
};

SplitSource split_at_kernel_start(const std::string& source) {
    constexpr const char* marker = "// KERNEL_START";

    const std::size_t pos = source.find(marker);

    if (pos == std::string::npos) {
        throw std::runtime_error(
            "PwToProxy JIT source is missing // KERNEL_START marker"
        );
    }

    return SplitSource{
        source.substr(0, pos),
        source.substr(pos)
    };
}

std::filesystem::path jit_source_root() {
#ifdef DMK_JIT_SOURCE_DIR
    return std::filesystem::path(DMK_JIT_SOURCE_DIR);
#else
    if (const char* env = std::getenv("DMK_JIT_SOURCE_DIR")) {
        return std::filesystem::path(env);
    }

    return std::filesystem::path("src/cuda/jit_sources");
#endif
}

std::string make_prelude(const JitKey& key) {
    const int k1_tile   = get_required_param(key, "K1_TILE");
    const int col_reg   = get_required_param(key, "COL_REG");
    const int k2_tile   = get_required_param(key, "K2_TILE");
    const int k3_tile   = get_required_param(key, "K3_TILE");
    const int kr_tile   = get_required_param(key, "KR_TILE");
    const int blocksize = get_required_param(key, "BLOCK_SIZE");
    const int n_order      = get_required_param(key, "N_ORDER");
    const int n_pw         = get_required_param(key, "N_PW");
    const int n_pw2        = get_required_param(key, "N_PW2");
    const int n_charge_dim = get_required_param(key, "N_CHARGE_DIM");

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

    const auto source_path = jit_source_root() / filename;

    const std::string file_source = read_text_file(source_path);
    const SplitSource split = split_at_kernel_start(file_source);

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
    key.real = real_name<Real>();
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
    key.real = real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"K1_TILE", k1_tile},
        {"COL_REG", col_reg},
        {"K2_TILE", k2_tile},
        {"K3_TILE", k3_tile},
        {"KR_TILE", kr_tile},
        {"N_ORDER", 0},
        {"N_CHARGE_DIM", 0},
        {"N_PW", 0},
        {"N_PW2", 0},
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