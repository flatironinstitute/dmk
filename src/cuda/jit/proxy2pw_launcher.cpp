#include "proxy2pw_launcher.hpp"

#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_types.hpp"

#ifdef DMK_CUDA_USE_NVRTC_JIT
#include <dmk_jit_config.hpp>
#endif

#include <dmk/cuda/helpers.hpp>

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
            std::string("Proxy2Pw JIT key missing parameter: ") + name
        );
    }

    return it->second;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "Proxy2Pw JIT: failed to open source file: " + path.string()
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
            "Proxy2Pw JIT source is missing // KERNEL_START marker"
        );
    }

    SplitSource out;
    out.header = source.substr(0, pos);
    out.kernel = source.substr(pos);

    return out;
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

std::string make_specialization_constants(const JitKey& key) {
    const int blocksize    = get_required_param(key, "BLOCK_SIZE");
    const int n_order      = get_required_param(key, "N_ORDER");
    const int n_pw         = get_required_param(key, "N_PW");
    const int n_pw2        = get_required_param(key, "N_PW2");
    const int n_charge_dim = get_required_param(key, "N_CHARGE_DIM");
    const std::string real_type = key.real;

    std::ostringstream ss;

    ss << "#include <dmk/cuda/proxy2pw_kernelargs.hpp>\n";
    ss << "using dmk::cuda::Proxy2PwArgs;\n\n";
    ss << "constexpr int N_ORDER      = " << n_order << ";\n";
    ss << "constexpr int N_PW         = " << n_pw << ";\n";
    ss << "constexpr int N_PW2        = " << n_pw2 << ";\n";
    ss << "constexpr int N_CHARGE_DIM = " << n_charge_dim << ";\n";
    ss << "constexpr int BLOCK_SIZE   = " << blocksize << ";\n";
    ss << "using Real = " << real_type << ";\n\n";

    return ss.str();
}

std::size_t proxy2pw_shared_bytes(
    int n_order,
    int n_pw,
    std::size_t sizeof_real
) {
    return
        std::size_t{2} *
        (
            std::size_t(n_order) * std::size_t(n_order) +
            std::size_t(n_order) * std::size_t(n_pw)
        ) *
        sizeof_real;
}

template <typename Real>
void check_proxy2pw_shape_or_throw(
    const dmk::cuda::Proxy2PwArgs<Real>& a
) {
    if (a.n_order <= 0 || a.n_pw <= 0 || a.n_pw2 <= 0 || a.n_charge_dim <= 0) {
        throw std::runtime_error(
            "Proxy2Pw JIT: invalid shape"
        );
    }
}


} // namespace


std::string make_proxy2pw_source(const JitKey& key) {
    std::filesystem::path filename;

    if (key.name == "Proxy2PwKernel") {
        filename = "proxy2pw.cu";
    } else if (key.name == "Proxy2PwMultiLevelKernel") {
        filename = "proxy2pw_multilevel.cu";
    } else {
        throw std::runtime_error(
            "Proxy2Pw JIT: unknown kernel name: " + key.name
        );
    }

    const auto source_path = jit_source_root() / filename;

    const std::string file_source = read_text_file(source_path);
    const SplitSource split = split_at_kernel_start(file_source);

    std::ostringstream generated;

    generated << make_specialization_constants(key);

    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real>
void launch_proxy2pw_jit(
    JitCache& cache,
    const dmk::cuda::Proxy2PwArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
) {
    if (args.n_boxes_at_level == 0) {
        return;
    }

    check_proxy2pw_shape_or_throw(args);

    JitKey key;
    key.name = "Proxy2PwKernel";
    key.real = real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"BLOCK_SIZE", blocksize},
        {"N_ORDER", args.n_order},
        {"N_CHARGE_DIM", args.n_charge_dim},
        {"N_PW", args.n_pw},
        {"N_PW2", args.n_pw2}
    };

    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes =
        proxy2pw_shared_bytes(
            args.n_order,
            args.n_pw,
            sizeof(Real)
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
void launch_proxy2pw_multilevel_jit(
    JitCache& cache,
    const std::vector<dmk::cuda::Proxy2PwArgs<Real>>& pa_h,
    dmk::cuda::Proxy2PwArgs<Real>* d_args_scratch,
    cudaStream_t stream,
    int blocksize
) {
    if (pa_h.empty()) {
        return;
    }

    int max_boxes = 0;
    int max_n_order = 0;
    int max_n_pw = 0;

    for (const auto& pa : pa_h) {
        max_boxes = std::max(max_boxes, pa.n_boxes_at_level);
        max_n_order = std::max(max_n_order, pa.n_order);
        max_n_pw = std::max(max_n_pw, pa.n_pw);
    }

    if (max_boxes == 0) {
        return;
    }

    DMK_CHECK_CUDA(
        cudaMemcpyAsync(
            d_args_scratch,
            pa_h.data(),
            pa_h.size() * sizeof(dmk::cuda::Proxy2PwArgs<Real>),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    JitKey key;
    key.name = "Proxy2PwMultiLevelKernel";
    key.real = real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"BLOCK_SIZE", blocksize},
        {"N_ORDER", 0},
        {"N_CHARGE_DIM", 0},
        {"N_PW", 0},
        {"N_PW2", 0} //unused in the multi-level path
    };

    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes =
        proxy2pw_shared_bytes(
            max_n_order,
            max_n_pw,
            sizeof(Real)
        );

    const int n_args = static_cast<int>(pa_h.size());

    kernel->launch(
        dim3(max_boxes, n_args, 1),
        dim3(blocksize, 1, 1),
        shared_bytes,
        stream,
        d_args_scratch,
        n_args
    );
}

template void launch_proxy2pw_jit<float>(
    JitCache&,
    const dmk::cuda::Proxy2PwArgs<float>&,
    cudaStream_t,
    int
);

template void launch_proxy2pw_jit<double>(
    JitCache&,
    const dmk::cuda::Proxy2PwArgs<double>&,
    cudaStream_t,
    int
);

template void launch_proxy2pw_multilevel_jit<float>(
    JitCache&,
    const std::vector<dmk::cuda::Proxy2PwArgs<float>>&,
    dmk::cuda::Proxy2PwArgs<float>*,
    cudaStream_t,
    int
);

template void launch_proxy2pw_multilevel_jit<double>(
    JitCache&,
    const std::vector<dmk::cuda::Proxy2PwArgs<double>>&,
    dmk::cuda::Proxy2PwArgs<double>*,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit