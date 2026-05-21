#include "multiply_kernelft_launcher.hpp"

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
            std::string("MultiplyKernelFT JIT key missing parameter: ") + name
        );
    }

    return it->second;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "MultiplyKernelFT JIT: failed to open source file: " + path.string()
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
            "MultiplyKernelFT JIT source is missing // KERNEL_START marker"
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

    ss << "#include <dmk/cuda/multiply_kernelft_kernelargs.hpp>\n";
    ss << "using dmk::cuda::MultiplyCd2pArgs;\n";
    ss << "using dmk::cuda::MultiplyStokeslet3DArgs;\n";
    ss << "using dmk::cuda::MultiplyStresslet3DArgs;\n\n";
    ss << "using Real = " << key.real << ";\n\n";
    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n\n";

    return ss.str();
}

std::size_t stokeslet_shared_bytes(std::size_t sizeof_real) {
    return sizeof_real * 6;
}

void check_block_or_throw(const char* label, int blocksize, int n_boxes_at_level) {
    if (blocksize <= 0) {
        throw std::runtime_error(
            std::string(label) +
            ": invalid blocksize=" + std::to_string(blocksize) +
            " n_boxes_at_level=" + std::to_string(n_boxes_at_level)
        );
    }
}

} // namespace

std::string make_multiply_kernelft_source(const JitKey& key) {
    const auto source_path = jit_source_root() / "multiply_kernelft.cu";

    const std::string file_source = read_text_file(source_path);
    const SplitSource split = split_at_kernel_start(file_source);

    std::ostringstream generated;

    generated << make_specialization_constants(key) << "\n";
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real, int DIM>
void launch_multiply_cd2p_jit(
    JitCache& cache,
    const dmk::cuda::MultiplyCd2pArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
) {
    if (args.n_boxes_at_level == 0) {
        return;
    }

    check_block_or_throw("launch_multiply_cd2p_jit", blocksize, args.n_boxes_at_level);

    JitKey key;
    key.name = "MultiplyCd2pByBoxKernel";
    key.real = real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {
        {"DIM", DIM},
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    kernel->launch(
        dim3(args.n_boxes_at_level, 1, 1),
        dim3(blocksize, 1, 1),
        0,
        stream,
        args
    );
}

template <typename Real>
void launch_multiply_stokeslet_3d_jit(
    JitCache& cache,
    const dmk::cuda::MultiplyStokeslet3DArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
) {
    if (args.n_boxes_at_level == 0) {
        return;
    }

    check_block_or_throw("launch_multiply_stokeslet_3d_jit", blocksize, args.n_boxes_at_level);

    JitKey key;
    key.name = "MultiplyStokeslet3DByBoxKernel";
    key.real = real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    kernel->launch(
        dim3(args.n_boxes_at_level, 1, 1),
        dim3(blocksize, 1, 1),
        stokeslet_shared_bytes(sizeof(Real)),
        stream,
        args
    );
}

template <typename Real>
void launch_multiply_stresslet_3d_jit(
    JitCache& cache,
    const dmk::cuda::MultiplyStresslet3DArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
) {
    if (args.n_boxes_at_level == 0) {
        return;
    }

    check_block_or_throw("launch_multiply_stresslet_3d_jit", blocksize, args.n_boxes_at_level);

    JitKey key;
    key.name = "MultiplyStresslet3DByBoxKernel";
    key.real = real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    kernel->launch(
        dim3(args.n_boxes_at_level, 1, 1),
        dim3(blocksize, 1, 1),
        0,
        stream,
        args
    );
}

template void launch_multiply_cd2p_jit<float, 2>(
    JitCache&,
    const dmk::cuda::MultiplyCd2pArgs<float>&,
    cudaStream_t,
    int
);

template void launch_multiply_cd2p_jit<float, 3>(
    JitCache&,
    const dmk::cuda::MultiplyCd2pArgs<float>&,
    cudaStream_t,
    int
);

template void launch_multiply_cd2p_jit<double, 2>(
    JitCache&,
    const dmk::cuda::MultiplyCd2pArgs<double>&,
    cudaStream_t,
    int
);

template void launch_multiply_cd2p_jit<double, 3>(
    JitCache&,
    const dmk::cuda::MultiplyCd2pArgs<double>&,
    cudaStream_t,
    int
);

template void launch_multiply_stokeslet_3d_jit<float>(
    JitCache&,
    const dmk::cuda::MultiplyStokeslet3DArgs<float>&,
    cudaStream_t,
    int
);

template void launch_multiply_stokeslet_3d_jit<double>(
    JitCache&,
    const dmk::cuda::MultiplyStokeslet3DArgs<double>&,
    cudaStream_t,
    int
);

template void launch_multiply_stresslet_3d_jit<float>(
    JitCache&,
    const dmk::cuda::MultiplyStresslet3DArgs<float>&,
    cudaStream_t,
    int
);

template void launch_multiply_stresslet_3d_jit<double>(
    JitCache&,
    const dmk::cuda::MultiplyStresslet3DArgs<double>&,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit
