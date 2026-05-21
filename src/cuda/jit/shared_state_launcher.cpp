#include "shared_state_launcher.hpp"

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
            std::string("SharedState JIT key missing parameter: ") + name
        );
    }

    return it->second;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "SharedState JIT: failed to open source file: " + path.string()
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
            "SharedState JIT source is missing // KERNEL_START marker"
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

std::string make_specialization_constants(const JitKey& key) {
    const int blocksize = get_required_param(key, "BLOCK_SIZE");

    std::ostringstream ss;

    ss << "using Real = " << key.real << ";\n";
    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n\n";

    return ss.str();
}

} // namespace

std::string make_shared_state_source(const JitKey& key) {
    const auto source_path = jit_source_root() / "shared_state.cu";

    const std::string file_source = read_text_file(source_path);
    const SplitSource split = split_at_kernel_start(file_source);

    std::ostringstream generated;

    generated << make_specialization_constants(key) << "\n";
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real>
void launch_accumulate_and_scatter_jit(
    JitCache& cache,
    Real* out,
    const Real* pot_eval,
    const Real* pot_extra,
    const long* scatter_index,
    int dof,
    long n_particles,
    cudaStream_t stream,
    int blocksize
) {
    if (n_particles == 0) {
        return;
    }
    if (blocksize <= 0) {
        throw std::runtime_error(
            "launch_accumulate_and_scatter_jit: invalid blocksize=" + std::to_string(blocksize)
        );
    }

    JitKey key;
    key.name = "AccumulateAndScatterKernel";
    key.real = real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    const long grid = (n_particles + blocksize - 1) / blocksize;

    kernel->launch(
        dim3(grid, 1, 1),
        dim3(blocksize, 1, 1),
        0,
        stream,
        out,
        pot_eval,
        pot_extra,
        scatter_index,
        dof,
        n_particles
    );
}

template void launch_accumulate_and_scatter_jit<float>(
    JitCache&,
    float*,
    const float*,
    const float*,
    const long*,
    int,
    long,
    cudaStream_t,
    int
);

template void launch_accumulate_and_scatter_jit<double>(
    JitCache&,
    double*,
    const double*,
    const double*,
    const long*,
    int,
    long,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit
