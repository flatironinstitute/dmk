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

std::filesystem::path jit_source_root();

const std::string_view *find_embedded_jit_source(std::string_view filename);

int embedded_jit_header_count();

const char *embedded_jit_header_name(int i);

const char *embedded_jit_header_source(int i);

std::string read_text_file(const std::filesystem::path &path, std::string_view label);

SplitSource split_at_kernel_start(const std::string &source, std::string_view label);

SplitSource load_split_jit_source(std::string_view filename, std::string_view label);

} // namespace dmk::cuda::jit
