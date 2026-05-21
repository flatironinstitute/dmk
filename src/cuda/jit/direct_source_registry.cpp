#include "direct_source_registry.hpp"

#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <atomic>
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

const SplitSource& direct_by_box_split_source() {
    static const SplitSource split = load_split_jit_source("direct_kernels.cu", "DirectByBox");

    return split;
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

    const int src_tile = required_int_param(key, "SRC_TILE", "DirectByBox");
    const int blocksize = required_int_param(key, "BLOCK_SIZE", "DirectByBox");

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
