#pragma once

#include <map>
#include <string>
#include <vector>

namespace dmk::cuda::jit {

struct JitKey {
    std::string name;
    std::string real; //float or double or what-have-you
    int sm_major = 0;
    int sm_minor = 0;   

    //compile time params
    std::map<std::string, int> params;
    std::string to_string() const;
};


struct CompiledBinary {
    std::vector<char> image;
    bool is_cubin = false;
    std::string log;
};

}