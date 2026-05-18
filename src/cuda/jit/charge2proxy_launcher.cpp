#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_types.hpp"
#ifdef DMK_CUDA_USE_NVRTC_JIT
#include <dmk_jit_config.hpp>
#endif
#include <dmk/cuda/charge2proxy_kernels.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>
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

auto get_required_param(const JitKey& key, const char* name) {
    const auto it = key.params.find(name);

    if (it == key.params.end()) {
        throw std::runtime_error(
            std::string("Charge2Proxy JIT key missing parameter: ") + name
        );
    }

    return it->second;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "Charge2Proxy JIT: failed to open source file: " + path.string()
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
            "Charge2Proxy JIT source is missing // KERNEL_START marker"
        );
    }

    SplitSource out;
    out.header = source.substr(0, pos);
    out.kernel = source.substr(pos);

    return out;
}


std::string make_specialization_constants(const JitKey& key) {
    const int n_order      = get_required_param(key, "N_ORDER");
    const int n_charge_dim = get_required_param(key, "N_CHARGE_DIM");
    const int chunk        = get_required_param(key, "CHUNK");
    const int i_tile       = get_required_param(key, "I_TILE");
    const int j_tile       = get_required_param(key, "J_TILE");
    const int k_tile       = get_required_param(key, "K_TILE");
    const std::string real_type = key.real;
    std::ostringstream ss;
    ss << "#include <dmk/cuda/charge2proxy_kernelargs.hpp>\n"; 
    ss << "using dmk::cuda::Charge2ProxyArgs;\n\n";
    ss << "constexpr int N_ORDER      = " << n_order << ";\n";
    ss << "constexpr int N_CHARGE_DIM = " << n_charge_dim << ";\n";
    ss << "constexpr int CHUNK        = " << chunk << ";\n";
    ss << "constexpr int I_TILE       = " << i_tile << ";\n";
    ss << "constexpr int J_TILE       = " << j_tile << ";\n";
    ss << "constexpr int K_TILE       = " << k_tile << ";\n";
    ss << "using Real = " << real_type << "; \n\n";
    return ss.str();

}


std::size_t charge2proxy_shared_bytes(int n_order, int n_charge_dim, int chunk, std::size_t sizeof_real) {
    const int ld = chunk + 1;

    return (std::size_t{3} * std::size_t(n_order) * std::size_t(ld) + std::size_t(n_charge_dim) * std::size_t(ld)) * sizeof_real;
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

} // namespace


std::string make_charge2proxy_source(const JitKey& key) {
    const auto source_path = jit_source_root() / "charge2proxy.cu";

    const std::string file_source = read_text_file(source_path); //todo

    std::ostringstream generated;
    const SplitSource split = split_at_kernel_start(file_source);
    generated << split.header << "\n";

    generated << make_specialization_constants(key);

    generated << split.kernel << "\n";


    return generated.str();
}

template <typename Real>
void launch_charge2proxy_jit(
    JitCache& cache,
    const dmk::cuda::Charge2ProxyArgs<Real>& args,
    const int* group_perm,
    int n_launch_groups,
    cudaStream_t stream,
    int chunk,
    int i_tile,
    int j_tile,
    int k_tile,
    int blocksize
) {
    if (args.n_groups == 0 || n_launch_groups == 0) {
        return;
    }

    JitKey key;
    key.name = "Charge2ProxyKernel";
    key.real = real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"N_ORDER", args.n_order},
        {"N_CHARGE_DIM", args.n_charge_dim},
        {"CHUNK", chunk},
        {"I_TILE", i_tile},
        {"J_TILE", j_tile},
        {"K_TILE", k_tile},
        {"BLOCK_SIZE", blocksize},
    };
    
    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes = charge2proxy_shared_bytes(args.n_order, args.n_charge_dim, chunk, sizeof(Real));
    
    kernel->launch(
        dim3(n_launch_groups, 1, 1),
        dim3(blocksize, 1, 1),
        shared_bytes,
        stream,
        args,
        group_perm
    );
}

template void launch_charge2proxy_jit<float>(
    JitCache&,
    const dmk::cuda::Charge2ProxyArgs<float>&,
    const int*,
    int,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int
);

template void launch_charge2proxy_jit<double>(
    JitCache&,
    const dmk::cuda::Charge2ProxyArgs<double>&,
    const int*,
    int,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int
);

}