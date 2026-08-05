#pragma once

#include "jit_types.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace dmk::cuda::jit {

struct SplitSource {
    std::string header;
    std::string kernel;
};

template <typename Real>
const char *jit_real_name();

template <>
inline const char *jit_real_name<float>() {
    return "float";
}

template <>
inline const char *jit_real_name<double>() {
    return "double";
}

int required_int_param(const JitKey &key, const char *name, std::string_view label);

// Non-null when DMK_JIT_SOURCE_DIR is set in the environment, meaning "read .cu files from
// this tree instead of the ones embedded at build time". Every site choosing between disk and
// embedded goes through this so they cannot disagree.
const char *jit_source_override();

// Root that embedded source keys are relative to -- the same directory CMake passes as
// EMBED_ROOT. Overridden by the environment, then by the build-time macro.
std::filesystem::path jit_source_root();

// Physical path of a logical source key. Embedded keys are "<stage>/<file>.cu" -- the
// jit_sources component is stripped, see cmake/embed_jit_sources.cmake -- but on disk the file
// is at "<root>/<stage>/jit_sources/<file>.cu".
std::filesystem::path jit_source_path(std::string_view filename);

const std::string_view *find_embedded_jit_source(std::string_view filename);

int embedded_jit_header_count();

const char *embedded_jit_header_name(int i);

const char *embedded_jit_header_source(int i);

std::string read_text_file(const std::filesystem::path &path, std::string_view label);

SplitSource split_at_kernel_start(const std::string &source, std::string_view label);

SplitSource load_split_jit_source(std::string_view filename, std::string_view label);

} // namespace dmk::cuda::jit
