#include "tensorprod_launcher.hpp"

#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_types.hpp"

#ifdef DMK_CUDA_USE_NVRTC_JIT
#include <dmk_jit_config.hpp>
#endif

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

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
            std::string("Tensorprod JIT key missing parameter: ") + name
        );
    }

    return it->second;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "Tensorprod JIT: failed to open source file: " + path.string()
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
            "Tensorprod JIT source is missing // KERNEL_START marker"
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
    const int n_order   = get_required_param(key, "N_ORDER");
    const int blocksize = get_required_param(key, "BLOCK_SIZE");
    const int n_charge_dim = get_required_param(key, "N_CHARGE_DIM");
    const std::string real_type = key.real;

    std::ostringstream ss;

    ss << "#include <dmk/cuda/tensorprod_kernelargs.hpp>\n";
    ss << "using dmk::cuda::TensorprodArgs;\n\n";

    ss << "constexpr int N_ORDER   = " << n_order << ";\n";
    ss << "constexpr int N_CHARGE_DIM  = " << n_charge_dim <<";\n";
    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n";
    ss << "using Real = " << real_type << ";\n\n";

    return ss.str();
}

} // namespace

std::string make_tensorprod_source(const JitKey& key) {
    const auto source_path = jit_source_root() / "tensorproduct.cu";

    const std::string file_source = read_text_file(source_path);
    const SplitSource split = split_at_kernel_start(file_source);

    std::ostringstream generated;

    generated << split.header << "\n";
    generated << make_specialization_constants(key);
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real>
void launch_tensorprod_jit(
    JitCache& cache,
    const dmk::cuda::TensorprodArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
) {
    if (args.n_pairs == 0) {
        return;
    }

    JitKey key;
    key.name = "TensorprodKernel";
    key.real = real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"N_ORDER", args.n_order},
        {"N_CHARGE_DIM", args.n_charge_dim},
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    kernel->launch(
        dim3(args.n_pairs, 1, 1),
        dim3(blocksize, 1, 1),
        0,
        stream,
        args
    );
}

template void launch_tensorprod_jit<float>(
    JitCache&,
    const dmk::cuda::TensorprodArgs<float>&,
    cudaStream_t,
    int
);

template void launch_tensorprod_jit<double>(
    JitCache&,
    const dmk::cuda::TensorprodArgs<double>&,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit