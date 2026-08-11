#include "jit_compiler.hpp"

#include "jit_source_utils.hpp"

#include <dmk/logger.h>

#include <nvrtc.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda::jit {

namespace {

struct CompileStat {
    long long count = 0;
    double total_ms = 0;
    double max_ms = 0;
};

std::mutex &stats_mutex() {
    static std::mutex m;
    return m;
}

std::map<std::string, CompileStat> &compile_stats() {
    static std::map<std::string, CompileStat> s;
    return s;
}

void record_compile(const std::string &program_name, double elapsed_ms) {
    {
        std::lock_guard<std::mutex> lock(stats_mutex());
        CompileStat &s = compile_stats()[program_name];
        ++s.count;
        s.total_ms += elapsed_ms;
        s.max_ms = std::max(s.max_ms, elapsed_ms);
    }
    dmk::get_logger()->trace("jit: compiled {} in {:.1f} ms", program_name, elapsed_ms);
}

void throw_nvrtc(nvrtcResult res, const std::string &where, const std::string &log = {}) {
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

} // namespace

void report_jit_compiles() {
    std::vector<std::pair<std::string, CompileStat>> rows;
    {
        std::lock_guard<std::mutex> lock(stats_mutex());
        rows.assign(compile_stats().begin(), compile_stats().end());
    }
    if (rows.empty())
        return;

    std::sort(rows.begin(), rows.end(),
              [](const auto &a, const auto &b) { return a.second.total_ms > b.second.total_ms; });

    long long n = 0;
    double ms = 0;
    for (const auto &[name, s] : rows) {
        n += s.count;
        ms += s.total_ms;
    }

    auto log = dmk::get_logger();
    log->debug("jit: {} NVRTC compiles, {:.2f} s total compile time", n, ms / 1000.0);
    for (const auto &[name, s] : rows)
        log->debug("jit:   {:<34} n={:<5} total={:8.0f} ms  mean={:6.1f}  max={:6.1f}", name, s.count, s.total_ms,
                   s.total_ms / s.count, s.max_ms);
}

CompiledBinary JitCompiler::compile(const std::string &source, const std::string &program_name, int sm_major,
                                    int sm_minor, const std::vector<std::string> &extra_options,
                                    const std::string &name_expression) const {
    const auto compile_start = std::chrono::steady_clock::now();
    nvrtcProgram prog = nullptr;

    std::vector<const char *> header_sources;
    std::vector<const char *> header_names;

    // Reading from a source tree means its own #include lines resolve off disk, so the
    // embedded copies must not shadow them.
    if (!jit_source_override()) {
        const int n = embedded_jit_header_count();

        for (int i = 0; i < n; ++i) {
            header_sources.push_back(embedded_jit_header_source(i));
            header_names.push_back(embedded_jit_header_name(i));
        }
    }

    nvrtcResult res = nvrtcCreateProgram(
        &prog, source.c_str(), program_name.c_str(), static_cast<int>(header_sources.size()),
        header_sources.empty() ? nullptr : header_sources.data(), header_names.empty() ? nullptr : header_names.data());
    if (res != NVRTC_SUCCESS) {
        throw_nvrtc(res, "nvrtcCreateProgram");
    }

    if (!name_expression.empty()) {
        res = nvrtcAddNameExpression(prog, name_expression.c_str());
        if (res != NVRTC_SUCCESS) {
            throw_nvrtc(res, "nvrtcAddNameExpression");
        }
    }

    std::vector<std::string> options_storage;

    options_storage.push_back("--std=c++20");
    options_storage.push_back("--gpu-architecture=sm_" + std::to_string(sm_major) + std::to_string(sm_minor));
    options_storage.push_back("-lineinfo");

    for (const auto &opt : extra_options) {
        options_storage.push_back(opt);
    }

    // Space-separated nvrtc flags for precision/codegen experiments (e.g. "--use_fast_math")
    // without a rebuild. Not part of any cache key, so pair it with DMK_JIT_AUTOTUNE_FORCE=1
    // or a stale tuning entry will be reused for differently-compiled code.
    if (const char *flags = std::getenv("DMK_JIT_NVRTC_FLAGS")) {
        std::istringstream flag_stream(flags);
        std::string flag;
        while (flag_stream >> flag)
            options_storage.push_back(flag);
    }

    std::vector<const char *> options;
    options.reserve(options_storage.size());

    for (const auto &opt : options_storage) {
        options.push_back(opt.c_str());
    }

    res = nvrtcCompileProgram(prog, static_cast<int>(options.size()), options.data());

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
    if (!name_expression.empty()) {
        const char *lowered = nullptr;

        res = nvrtcGetLoweredName(prog, name_expression.c_str(), &lowered);
        if (res != NVRTC_SUCCESS) {
            throw_nvrtc(res, "nvrtcGetLoweredName");
        }
        out.lowered_name = lowered;
    }

    nvrtcDestroyProgram(&prog);

    record_compile(program_name,
                   std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - compile_start).count());

    return out;
}

} // namespace dmk::cuda::jit
