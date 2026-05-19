#pragma once

#include "jit_types.hpp"

namespace dmk::cuda::jit {

class JitCompiler {
  public:
    CompiledBinary compile(const std::string &source, const std::string &program_name, int sm_major, int sm_minor,
                           const std::vector<std::string> &extra_options = {},
                           const std::string &name_expression = {}) const;
};

} // namespace dmk::cuda::jit