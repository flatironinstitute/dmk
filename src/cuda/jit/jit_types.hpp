#pragma once

#include <map>
#include <string>
#include <vector>

namespace dmk::cuda::jit {

struct JitKey {
    std::string name;
    std::string real;
    int sm_major = 0;
    int sm_minor = 0;

    // compile time params
    std::map<std::string, int> params;

    std::map<std::string, std::string> tags; // to resolve templates
    std::string to_string() const;
};

struct CompiledBinary {
    std::vector<char> image;
    std::string lowered_name;
    bool is_cubin = false;
    std::string log;
};

} // namespace dmk::cuda::jit