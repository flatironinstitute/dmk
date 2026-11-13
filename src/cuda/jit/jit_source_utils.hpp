#pragma once

#include "jit_types.hpp"

#include <cstddef>
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

const std::string_view *find_embedded_jit_source(std::string_view filename);

int embedded_jit_header_count();

const char *embedded_jit_header_name(int i);

const char *embedded_jit_header_source(int i);

SplitSource split_at_kernel_start(const std::string &source, std::string_view label);

SplitSource load_split_jit_source(std::string_view filename, std::string_view label);

/// Hash of a JIT source file's text, memoized per path. Tuning results are persisted across
/// runs, so a kernel edit that leaves the tune key alone would otherwise keep reusing a config
/// chosen for the old code; folding this into the key retires those entries automatically.
std::size_t jit_source_hash(std::string_view filename);

/// Same, for an embedded device-only header. A source's own hash does not cover what it includes,
/// so a tune key depending on header code must fold this in too. Throws on an unknown name.
std::size_t jit_header_hash(std::string_view header_name);

} // namespace dmk::cuda::jit
