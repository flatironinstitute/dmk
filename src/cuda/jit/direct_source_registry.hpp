#pragma once

#include "jit_types.hpp"

#include <string>

namespace dmk::cuda::jit {

struct DirectSourceDescriptor {
    std::string coeff_prelude;
    std::string evaluator_expr;
};

int next_direct_kernel_id();

void register_direct_source_descriptor(const std::string &kernel_name, DirectSourceDescriptor descriptor);

const DirectSourceDescriptor &get_direct_source_descriptor(const std::string &kernel_name);

std::string make_direct_by_box_source(const JitKey &key);

} // namespace dmk::cuda::jit