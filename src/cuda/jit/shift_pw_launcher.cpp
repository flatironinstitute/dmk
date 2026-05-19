#include "shift_pw_launcher.hpp"

#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_types.hpp"

#ifdef DMK_CUDA_USE_NVRTC_JIT
#include <dmk_jit_config.hpp>
#endif

#include <dmk/cuda/helpers.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
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
            std::string("ShiftPw JIT key missing parameter: ") + name
        );
    }

    return it->second;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "ShiftPw JIT: failed to open source file: " + path.string()
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
            "ShiftPw JIT source is missing // KERNEL_START marker"
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
    const int n_pw_modes         = get_required_param(key, "N_PW_MODES");
    const int n_charge_dim = get_required_param(key, "N_CHARGE_DIM");
    std::ostringstream ss;

    ss << "#include <dmk/cuda/shift_pw_kernelargs.hpp>\n";

    ss << "using dmk::cuda::ShiftPwArgs;\n";

    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n";
    ss << "constexpr int N_PW_MODES   = " << n_pw_modes << ";\n";
    ss << "constexpr int N_CHARGE_DIM = " << n_charge_dim << ";\n";
    ss << "using Real = " << key.real << ";\n\n";

    return ss.str();
}


} // namespace

std::string make_shift_pw_source(const JitKey& key) {
    const auto source_path = jit_source_root() / "shiftpw.cu";

    const std::string file_source = read_text_file(source_path);
    const SplitSource split = split_at_kernel_start(file_source);

    std::ostringstream generated;

    generated << make_specialization_constants(key);
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real>
void launch_shift_pw_multilevel_jit(
    JitCache& cache,
    const std::vector<dmk::cuda::ShiftPwArgs<Real>>& args_h,
    dmk::cuda::ShiftPwArgs<Real>* d_args_scratch,
    cudaStream_t stream,
    int blocksize
) {
    if (args_h.empty()) {
        return;
    }

    int max_boxes = 0;

    for (const auto& a : args_h) {
        if (a.n_boxes_at_level == 0) {
            continue;
        }

        max_boxes = std::max(max_boxes, a.n_boxes_at_level);
    }

    if (max_boxes == 0) {
        return;
    }

    DMK_CHECK_CUDA(
        cudaMemcpyAsync(
            d_args_scratch,
            args_h.data(),
            args_h.size() * sizeof(dmk::cuda::ShiftPwArgs<Real>),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    JitKey key;
    key.name = "ShiftPwKernel";
    key.real = real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"BLOCK_SIZE", blocksize},
        {"N_CHARGE_DIM", args_h[0].n_charge_dim},
        {"N_PW_MODES", args_h[0].n_pw_modes},
    };

    auto kernel = cache.get_kernel(key);

    const int n_args = static_cast<int>(args_h.size());

    kernel->launch(
        dim3(max_boxes, n_args, 1),
        dim3(blocksize, 1, 1),
        0,
        stream,
        d_args_scratch,
        n_args
    );
}

template void launch_shift_pw_multilevel_jit<float>(
    JitCache&,
    const std::vector<dmk::cuda::ShiftPwArgs<float>>&,
    dmk::cuda::ShiftPwArgs<float>*,
    cudaStream_t,
    int
);

template void launch_shift_pw_multilevel_jit<double>(
    JitCache&,
    const std::vector<dmk::cuda::ShiftPwArgs<double>>&,
    dmk::cuda::ShiftPwArgs<double>*,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit