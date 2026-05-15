#include "jit_compiler.hpp"

#include <nvrtc.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda::jit {

namespace {

void throw_nvrtc(
    nvrtcResult res,
    const std::string& where,
    const std::string& log = {}
) {
    std::string msg = where + ": " + nvrtcGetErrorString(res);

    if (!log.empty()) {
        msg += "\nNVRTC log:\n";
        msg += log;
    }

    throw std::runtime_error(msg);
}

std::string get_program_log(nvrtcProgram prog) {
    size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);

    std::string log(log_size, '\0');

    if (log_size > 1) {
        nvrtcGetProgramLog(prog, log.data());
    }

    return log;
}

} //namespace

CompiledBinary JitCompiler::compile(
    const std::string& source,
    const std::string& program_name,
    int sm_major,
    int sm_minor,
    const std::vector<std::string>& extra_options
) const {
    nvrtcProgram prog = nullptr;

    nvrtcResult res = nvrtcCreateProgram(
        &prog,
        source.c_str(),
        program_name.c_str(),
        0,
        nullptr,
        nullptr
    );

    if (res != NVRTC_SUCCESS) {
        throw_nvrtc(res, "nvrtcCreateProgram");
    }

    std::vector<std::string> options_storage;

    options_storage.push_back("--std=c++17");
    options_storage.push_back(
        "--gpu-architecture=sm_" + 
        std::to_string(sm_major) + 
        std::to_string(sm_minor)
    );

    for (const auto& opt : extra_options) {
        options_storage.push_back(opt);
    }

    std::vector<const char*> options;
    options.reserve(options_storage.size());

    for (const auto& opt : options_storage) {
        options.push_back(opt.c_str());
    }

    res = nvrtcCompileProgram(
        prog,
        static_cast<int>(options.size()),
        options.data()
    );


    std::string log = get_program_log(prog);

    if (res != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);
        throw_nvrtc(res, "nvrtcCompileProgram", log);
    }

    CompiledBinary out;
    out.log = log;

    size_t cubin_size = 0;
    res = nvrtcGetCUBINSize(prog, &cubin_size);

    if (res == NVRTC_SUCCESS && cubin_size > 0) {
        out.image.resize(cubin_size);
        res = nvrtcGetCUBIN(prog, out.image.data());

        if (res != NVRTC_SUCCESS) {
            nvrtcDestroyProgram(&prog);
            throw_nvrtc(res, "nvrtcGetCUBIN", log);
        }

        out.is_cubin = true;
    } else {
        size_t ptx_size = 0;
        res = nvrtcGetPTXSize(prog, &ptx_size);

        if (res != NVRTC_SUCCESS) {
            nvrtcDestroyProgram(&prog);
            throw_nvrtc(res, "nvrtcGetPTXSize", log);
        }

        out.image.resize(ptx_size);
        res = nvrtcGetPTX(prog, out.image.data());

        if (res != NVRTC_SUCCESS) {
            nvrtcDestroyProgram(&prog);
            throw_nvrtc(res, "nvrtcGetPTX", log);
        }

        out.is_cubin = false;
    }

    nvrtcDestroyProgram(&prog);
    return out;
}

} // namespace dmk::cuda::jit