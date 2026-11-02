#include "jit_source_utils.hpp"

#include <dmk_jit_config.hpp>

#include <filesystem>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dmk::cuda::jit {
namespace {

std::size_t line_number_at(const std::string &source, std::size_t pos) {
    std::size_t line = 1;

    for (std::size_t i = 0; i < pos; ++i) {
        if (source[i] == '\n') {
            ++line;
        }
    }

    return line;
}

std::size_t first_kernel_source_pos(const std::string &source, std::size_t marker_pos, std::size_t marker_size) {
    std::size_t pos = marker_pos + marker_size;

    while (pos < source.size() && (source[pos] == ' ' || source[pos] == '\t' || source[pos] == '\r')) {
        ++pos;
    }

    if (pos < source.size() && source[pos] == '\n') {
        ++pos;
    }

    return pos;
}

std::string escape_line_directive_path(const std::filesystem::path &path) {
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

std::string line_directive(std::size_t line, const std::filesystem::path &path) {
    std::ostringstream ss;
    ss << "#line " << line << " \"" << escape_line_directive_path(path) << "\"\n";
    return ss.str();
}

} // namespace

int required_int_param(const JitKey &key, const char *name, std::string_view label) {
    const auto it = key.params.find(name);

    if (it == key.params.end()) {
        throw std::runtime_error(std::string(label) + " JIT key missing parameter: " + name);
    }

    return it->second;
}

SplitSource split_at_kernel_start(const std::string &source, std::string_view label) {
    constexpr const char *marker = "// KERNEL_START";

    const std::size_t pos = source.find(marker);

    if (pos == std::string::npos) {
        throw std::runtime_error(std::string(label) + " JIT source is missing // KERNEL_START marker");
    }

    const std::size_t kernel_pos = first_kernel_source_pos(source, pos, std::string_view(marker).size());

    return SplitSource{source.substr(0, pos), source.substr(kernel_pos)};
}

namespace {
// Physical path of a logical source key. Embedded keys are "<stage>/<file>.cu" -- the
// jit_sources component is stripped, see cmake/embed_jit_sources.cmake -- but on disk the file
// is at "<root>/<stage>/jit_sources/<file>.cu".
std::filesystem::path jit_source_path(std::string_view filename) {
    const std::filesystem::path root(DMK_JIT_SOURCE_DIR);
    const std::filesystem::path logical{filename};

    // Flat keys (no stage component) name a file directly under the root.
    if (!logical.has_parent_path()) {
        return root / logical;
    }

    return root / logical.parent_path() / "jit_sources" / logical.filename();
}

std::string_view jit_source_text(std::string_view filename, std::string_view label) {
    const std::string_view *embedded = find_embedded_jit_source(filename);
    if (!embedded)
        throw std::runtime_error(std::string(label) + " JIT: no embedded source for: " + std::string(filename));
    return *embedded;
}
} // namespace

std::size_t jit_source_hash(std::string_view filename) {
    static std::map<std::string, std::size_t> cache;
    static std::mutex mtx;
    const std::string name(filename);
    std::lock_guard<std::mutex> lock(mtx);
    const auto it = cache.find(name);
    if (it != cache.end())
        return it->second;
    const std::string_view text = jit_source_text(filename, filename);
    std::size_t h = 1469598103934665603ULL;
    for (unsigned char c : text) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    cache.emplace(name, h);
    return h;
}

std::size_t jit_header_hash(std::string_view header_name) {
    static std::map<std::string, std::size_t> cache;
    static std::mutex mtx;
    const std::string name(header_name);
    std::lock_guard<std::mutex> lock(mtx);
    const auto it = cache.find(name);
    if (it != cache.end())
        return it->second;

    const char *text = nullptr;
    const int n = embedded_jit_header_count();
    for (int i = 0; i < n && !text; ++i)
        if (name == embedded_jit_header_name(i))
            text = embedded_jit_header_source(i);
    if (!text)
        throw std::runtime_error("jit_header_hash: not an embedded JIT header: " + name);

    std::size_t h = 1469598103934665603ULL;
    for (const char *c = text; *c; ++c) {
        h ^= static_cast<unsigned char>(*c);
        h *= 1099511628211ULL;
    }
    cache.emplace(name, h);
    return h;
}

SplitSource load_split_jit_source(std::string_view filename, std::string_view label) {
    const std::string source(jit_source_text(filename, label));

    SplitSource split = split_at_kernel_start(source, label);

    // Path baked into the #line directives, which is how -lineinfo consumers like ncu locate
    // the source. It names the source tree's copy, so edit a .cu without rebuilding and the line
    // numbers no longer describe the text.
    constexpr const char *marker = "// KERNEL_START";
    const std::size_t marker_pos = source.find(marker);
    const std::filesystem::path profile_path = jit_source_path(filename);
    const std::size_t kernel_pos = first_kernel_source_pos(source, marker_pos, std::string_view(marker).size());

    split.header = line_directive(1, profile_path) + split.header;
    split.kernel = line_directive(line_number_at(source, kernel_pos), profile_path) + split.kernel;

    return split;
}

} // namespace dmk::cuda::jit
