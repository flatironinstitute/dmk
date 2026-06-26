#include "autotune.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace dmk::cuda::jit {
namespace {

std::string json_escape(const std::string &in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

std::optional<std::string> json_string_field(const std::string &line, const std::string &field) {
    const std::string marker = "\"" + field + "\":\"";
    const std::size_t start = line.find(marker);
    if (start == std::string::npos) {
        return std::nullopt;
    }

    std::string out;
    bool escaped = false;
    for (std::size_t i = start + marker.size(); i < line.size(); ++i) {
        const char c = line[i];
        if (escaped) {
            switch (c) {
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            default:
                out += c;
                break;
            }
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            return out;
        } else {
            out += c;
        }
    }
    return std::nullopt;
}

std::optional<double> json_double_field(const std::string &line, const std::string &field) {
    const std::string marker = "\"" + field + "\":";
    const std::size_t start = line.find(marker);
    if (start == std::string::npos) {
        return std::nullopt;
    }

    std::size_t end = start + marker.size();
    while (end < line.size() && line[end] != ',' && line[end] != '}') {
        ++end;
    }

    try {
        return std::stod(line.substr(start + marker.size(), end - start - marker.size()));
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<TuningParams> json_params_field(const std::string &line) {
    const std::string marker = "\"params\":{";
    const std::size_t start = line.find(marker);
    if (start == std::string::npos) {
        return std::nullopt;
    }

    const std::size_t end = line.find('}', start + marker.size());
    if (end == std::string::npos) {
        return std::nullopt;
    }

    TuningParams params;
    std::size_t pos = start + marker.size();
    while (pos < end) {
        while (pos < end && (line[pos] == ',' || line[pos] == ' ')) {
            ++pos;
        }
        if (pos >= end) {
            break;
        }
        if (line[pos] != '"') {
            return std::nullopt;
        }

        const std::size_t name_begin = pos + 1;
        const std::size_t name_end = line.find('"', name_begin);
        if (name_end == std::string::npos || name_end >= end) {
            return std::nullopt;
        }
        const std::string name = line.substr(name_begin, name_end - name_begin);

        pos = name_end + 1;
        if (pos >= end || line[pos] != ':') {
            return std::nullopt;
        }
        ++pos;

        const std::size_t value_begin = pos;
        while (pos < end && line[pos] != ',') {
            ++pos;
        }

        try {
            params[name] = std::stoi(line.substr(value_begin, pos - value_begin));
        } catch (...) {
            return std::nullopt;
        }
    }

    return params;
}

std::optional<CachedTuneResult> parse_cache_line(const std::string &line) {
    auto key = json_string_field(line, "key");
    auto kernel = json_string_field(line, "kernel");
    auto device = json_string_field(line, "device");
    auto runtime_ms = json_double_field(line, "runtime_ms");
    auto params = json_params_field(line);

    if (!key || !kernel || !device || !runtime_ms || !params) {
        return std::nullopt;
    }

    return CachedTuneResult{*key, *kernel, *device, *runtime_ms, *params};
}

void append_json_line(std::ostream &out, const CachedTuneResult &result) {
    out << "{\"key\":\"" << json_escape(result.key) << "\",\"kernel\":\"" << json_escape(result.kernel)
        << "\",\"device\":\"" << json_escape(result.device) << "\",\"runtime_ms\":" << result.runtime_ms
        << ",\"params\":{";

    bool first = true;
    for (const auto &[name, value] : result.params) {
        if (!first) {
            out << ',';
        }
        first = false;
        out << "\"" << json_escape(name) << "\":" << value;
    }
    out << "}}\n";
}

std::string env_or_empty(const char *name) {
    const char *value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

constexpr std::filesystem::perms cache_directory_perms =
    std::filesystem::perms::owner_all | std::filesystem::perms::group_read | std::filesystem::perms::group_exec |
    std::filesystem::perms::others_read | std::filesystem::perms::others_exec;

void create_cache_directories(const std::filesystem::path &path) {
    std::filesystem::path current;
    for (const auto &component : path) {
        current /= component;
        if (current.empty() || std::filesystem::exists(current)) {
            continue;
        }

        if (std::filesystem::create_directory(current)) {
            std::filesystem::permissions(current, cache_directory_perms, std::filesystem::perm_options::replace);
        }
    }
}

} // namespace

JsonTuningCache::JsonTuningCache(std::filesystem::path path) : path_(std::move(path)) {}

std::optional<CachedTuneResult> JsonTuningCache::get(const std::string &key) {
    std::ifstream in(path_);
    if (!in) {
        return std::nullopt;
    }

    std::optional<CachedTuneResult> latest;
    std::string line;
    while (std::getline(in, line)) {
        auto parsed = parse_cache_line(line);
        if (parsed && parsed->key == key) {
            latest = std::move(parsed);
        }
    }
    return latest;
}

void JsonTuningCache::put(const CachedTuneResult &result) {
    if (path_.has_parent_path()) {
        create_cache_directories(path_.parent_path());
    }

    std::ofstream out(path_, std::ios::app);
    if (!out) {
        throw std::runtime_error("JsonTuningCache: failed to open " + path_.string());
    }
    append_json_line(out, result);
}

std::filesystem::path default_tuning_cache_path() {
    const std::string override_path = env_or_empty("DMK_JIT_AUTOTUNE_CACHE");
    if (!override_path.empty()) {
        return std::filesystem::path(override_path);
    }

    const std::string xdg_cache = env_or_empty("XDG_CACHE_HOME");
    if (!xdg_cache.empty()) {
        return std::filesystem::path(xdg_cache) / "dmk" / "jit_autotune.jsonl";
    }

    const std::string home = env_or_empty("HOME");
    if (!home.empty()) {
        return std::filesystem::path(home) / ".cache" / "dmk" / "jit_autotune.jsonl";
    }

    return std::filesystem::path("dmk_jit_autotune.jsonl");
}

std::string current_cuda_device_key() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    std::ostringstream os;
    os << "device=" << device << "|name=" << prop.name << "|sm=" << prop.major << prop.minor
       << "|sms=" << prop.multiProcessorCount;
    return os.str();
}

bool env_flag_enabled(const char *name) {
    const std::string value = env_or_empty(name);
    return value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "ON" || value == "yes" ||
           value == "YES";
}

std::vector<TuningParams> expand_grid(const std::vector<TuningParameter> &space) {
    std::vector<TuningParams> out;
    TuningParams current;

    std::function<void(std::size_t)> visit = [&](std::size_t i) {
        if (i == space.size()) {
            out.push_back(current);
            return;
        }

        const TuningParameter &parameter = space[i];
        if (parameter.values.empty()) {
            return;
        }

        for (int value : parameter.values) {
            current[parameter.name] = value;
            visit(i + 1);
        }
        current.erase(parameter.name);
    };

    visit(0);
    return out;
}

std::string tuning_params_to_string(const TuningParams &params) {
    std::ostringstream os;
    bool first = true;
    for (const auto &[name, value] : params) {
        if (!first) {
            os << ",";
        }
        first = false;
        os << name << "=" << value;
    }
    return os.str();
}

void check_cuda(cudaError_t err, const char *where) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
    }
}

} // namespace dmk::cuda::jit
