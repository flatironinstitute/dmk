#include "jit_source_utils.hpp"

#ifdef DMK_CUDA_USE_NVRTC_JIT
#include <dmk_jit_config.hpp>
#endif

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dmk::cuda::jit {

int required_int_param(const JitKey& key, const char* name, std::string_view label) {
    const auto it = key.params.find(name);

    if (it == key.params.end()) {
        throw std::runtime_error(
            std::string(label) + " JIT key missing parameter: " + name
        );
    }

    return it->second;
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

std::string read_text_file(const std::filesystem::path& path, std::string_view label) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            std::string(label) + " JIT: failed to open source file: " + path.string()
        );
    }

    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

SplitSource split_at_kernel_start(const std::string& source, std::string_view label) {
    constexpr const char* marker = "// KERNEL_START";

    const std::size_t pos = source.find(marker);

    if (pos == std::string::npos) {
        throw std::runtime_error(
            std::string(label) + " JIT source is missing // KERNEL_START marker"
        );
    }

    return SplitSource{
        source.substr(0, pos),
        source.substr(pos)
    };
}

SplitSource load_split_jit_source(std::string_view filename, std::string_view label) {
    const auto source_path = jit_source_root() / std::filesystem::path(std::string(filename));
    return split_at_kernel_start(read_text_file(source_path, label), label);
}

} // namespace dmk::cuda::jit
