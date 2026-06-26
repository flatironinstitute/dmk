#include "jit_source_utils.hpp"

#include <dmk_jit_config.hpp>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dmk::cuda::jit {
namespace {

std::size_t line_number_at(const std::string& source, std::size_t pos) {
    std::size_t line = 1;

    for (std::size_t i = 0; i < pos; ++i) {
        if (source[i] == '\n') {
            ++line;
        }
    }

    return line;
}

std::size_t first_kernel_source_pos(const std::string& source, std::size_t marker_pos, std::size_t marker_size) {
    std::size_t pos = marker_pos + marker_size;

    while (pos < source.size() && (source[pos] == ' ' || source[pos] == '\t' || source[pos] == '\r')) {
        ++pos;
    }

    if (pos < source.size() && source[pos] == '\n') {
        ++pos;
    }

    return pos;
}

std::string escape_line_directive_path(const std::filesystem::path& path) {
    std::string raw = path.string();
    std::string escaped;
    escaped.reserve(raw.size());

    for (char c : raw) {
        if (c == '\\' || c == '"') {
            escaped.push_back('\\');
        }

        escaped.push_back(c);
    }

    return escaped;
}

std::string line_directive(std::size_t line, const std::filesystem::path& path) {
    std::ostringstream ss;
    ss << "#line " << line << " \"" << escape_line_directive_path(path) << "\"\n";
    return ss.str();
}

} // namespace

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

    const std::size_t kernel_pos = first_kernel_source_pos(source, pos, std::string_view(marker).size());

    return SplitSource{
        source.substr(0, pos),
        source.substr(kernel_pos)
    };
}

SplitSource load_split_jit_source(std::string_view filename, std::string_view label) {
    const auto source_path = jit_source_root() / std::filesystem::path(std::string(filename));
    const std::string source = read_text_file(source_path, label);
    SplitSource split = split_at_kernel_start(source, label);

    constexpr const char* marker = "// KERNEL_START";
    const std::size_t marker_pos = source.find(marker);
    const std::filesystem::path profile_path = std::filesystem::absolute(source_path).lexically_normal();
    const std::size_t kernel_pos =
        first_kernel_source_pos(source, marker_pos, std::string_view(marker).size());

    split.header = line_directive(1, profile_path) + split.header;
    split.kernel = line_directive(line_number_at(source, kernel_pos), profile_path) + split.kernel;

    return split;
}

} // namespace dmk::cuda::jit
