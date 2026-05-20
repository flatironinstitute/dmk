#include "direct_source_registry.hpp"

#include "jit_types.hpp"

#ifdef DMK_CUDA_USE_NVRTC_JIT
#include <dmk_jit_config.hpp>
#endif

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace dmk::cuda::jit {
namespace {

std::atomic<int>& direct_kernel_counter() {
    static std::atomic<int> counter{0};
    return counter;
}

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, DirectSourceDescriptor>& registry() {
    static std::unordered_map<std::string, DirectSourceDescriptor> r;
    return r;
}

std::filesystem::path jit_source_root() {
#ifdef DMK_JIT_SOURCE_DIR
    return std::filesystem::path(DMK_JIT_SOURCE_DIR);
#else
    if (const char* env = std::getenv("DMK_JIT_SOURCE_DIR")) {
        return std::filesystem::path(env);
    }

    return std::filesystem::path("src/cuda/jit_sources");
#endif
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "DirectByBox JIT: failed to open source file: " + path.string()
        );
    }

    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct SplitSource {
    std::string header;
    std::string kernel;
};

SplitSource split_at_kernel_start(const std::string& source) {
    constexpr const char* marker = "// KERNEL_START";

    const std::size_t pos = source.find(marker);

    if (pos == std::string::npos) {
        throw std::runtime_error(
            "DirectByBox JIT source is missing // KERNEL_START marker"
        );
    }

    return SplitSource{
        source.substr(0, pos),
        source.substr(pos)
    };
}

const SplitSource& direct_by_box_split_source() {
    static const SplitSource split = [] {
        const auto source_path = jit_source_root() / "direct_kernels.cu";
        return split_at_kernel_start(read_text_file(source_path));
    }();

    return split;
}

int required_param(const JitKey& key, const char* name) {
    const auto it = key.params.find(name);

    if (it == key.params.end()) {
        throw std::runtime_error(
            std::string("DirectByBox JIT key missing param: ") + name
        );
    }

    return it->second;
}

} // namespace

int next_direct_kernel_id() {
    return direct_kernel_counter().fetch_add(1, std::memory_order_relaxed);
}

void register_direct_source_descriptor(
    const std::string& kernel_name,
    DirectSourceDescriptor descriptor
) {
    std::lock_guard<std::mutex> lock(registry_mutex());

    auto& r = registry();

    auto it = r.find(kernel_name);

    if (it == r.end()) {
        r.emplace(kernel_name, std::move(descriptor));
    }
}

const DirectSourceDescriptor& get_direct_source_descriptor(
    const std::string& kernel_name
) {
    std::lock_guard<std::mutex> lock(registry_mutex());

    const auto& r = registry();
    const auto it = r.find(kernel_name);

    if (it == r.end()) {
        throw std::runtime_error(
            "DirectByBox JIT: no source descriptor registered for " +
            kernel_name
        );
    }

    return it->second;
}

std::string make_direct_by_box_source(const JitKey& key) {
    const DirectSourceDescriptor& desc =
        get_direct_source_descriptor(key.name);

    const int src_tile = required_param(key, "SRC_TILE");
    const int blocksize = required_param(key, "BLOCK_SIZE");

    std::ostringstream prelude;

    prelude << "#define DMK_DIRECT_KERNEL_NAME " << key.name << "\n";
    prelude << "using Real = " << key.real << ";\n\n";

    prelude << desc.coeff_prelude << "\n";

    prelude << "#define DMK_DIRECT_EVALUATOR "
            << desc.evaluator_expr << "\n\n";

    prelude << "constexpr int SRC_TILE = " << src_tile << ";\n";
    prelude << "constexpr int BLOCK_SIZE = " << blocksize << ";\n\n";

    const SplitSource& split = direct_by_box_split_source();

    std::ostringstream generated;
    generated << prelude.str();
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

} // namespace dmk::cuda::jit