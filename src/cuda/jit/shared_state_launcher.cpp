#include "shared_state_launcher.hpp"

#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <string>

namespace dmk::cuda::jit {
namespace {

std::string make_specialization_constants(const JitKey& key) {
    const int blocksize = required_int_param(key, "BLOCK_SIZE", "SharedState");

    std::ostringstream ss;

    ss << "using Real = " << key.real << ";\n";
    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n\n";

    return ss.str();
}

} // namespace

std::string make_shared_state_source(const JitKey& key) {
    const SplitSource split = load_split_jit_source("shared_state.cu", "SharedState");

    std::ostringstream generated;

    generated << make_specialization_constants(key) << "\n";
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real>
void launch_accumulate_and_scatter_jit(
    JitCache& cache,
    Real* out,
    const Real* pot_eval,
    const Real* pot_extra,
    const long* scatter_index,
    int dof,
    long n_particles,
    cudaStream_t stream,
    int blocksize
) {
    if (n_particles == 0) {
        return;
    }

    JitKey key;
    key.name = "AccumulateAndScatterKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    const long grid = (n_particles + blocksize - 1) / blocksize;

    kernel->launch(
        dim3(grid, 1, 1),
        dim3(blocksize, 1, 1),
        0,
        stream,
        out,
        pot_eval,
        pot_extra,
        scatter_index,
        dof,
        n_particles
    );
}

template void launch_accumulate_and_scatter_jit<float>(
    JitCache&,
    float*,
    const float*,
    const float*,
    const long*,
    int,
    long,
    cudaStream_t,
    int
);

template void launch_accumulate_and_scatter_jit<double>(
    JitCache&,
    double*,
    const double*,
    const double*,
    const long*,
    int,
    long,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit
