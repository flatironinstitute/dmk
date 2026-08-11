#pragma once

#include "jit_types.hpp"

namespace dmk::cuda::jit {

class JitCompiler {
  public:
    CompiledBinary compile(const std::string &source, const std::string &program_name, int sm_major, int sm_minor,
                           const std::vector<std::string> &extra_options = {},
                           const std::string &name_expression = {}) const;
};

/// Cumulative NVRTC cost per program, at debug level. Times are summed per compile, so the
/// total exceeds elapsed wall whenever a matrix was built in parallel.
void report_jit_compiles();

} // namespace dmk::cuda::jit