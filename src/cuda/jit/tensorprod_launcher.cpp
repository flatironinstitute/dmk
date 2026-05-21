#include "tensorprod_launcher.hpp"

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
    const int n_order   = required_int_param(key, "N_ORDER", "Tensorprod");
    const int blocksize = required_int_param(key, "BLOCK_SIZE", "Tensorprod");
    const int n_charge_dim = required_int_param(key, "N_CHARGE_DIM", "Tensorprod");
    const std::string real_type = key.real;

    std::ostringstream ss;

    ss << "#include <dmk/cuda/tensorprod_kernelargs.hpp>\n";
    ss << "using dmk::cuda::TensorprodArgs;\n\n";

    ss << "constexpr int N_ORDER   = " << n_order << ";\n";
    ss << "constexpr int N_CHARGE_DIM  = " << n_charge_dim <<";\n";
    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n";
    ss << "using Real = " << real_type << ";\n\n";

    return ss.str();
}

} // namespace

std::string make_tensorprod_source(const JitKey& key) {
    const SplitSource split = load_split_jit_source("tensorproduct.cu", "Tensorprod");

    std::ostringstream generated;

    generated << split.header << "\n";
    generated << make_specialization_constants(key);
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real>
void launch_tensorprod_jit(
    JitCache& cache,
    const dmk::cuda::TensorprodArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
) {
    if (args.n_pairs == 0) {
        return;
    }

    JitKey key;
    key.name = "TensorprodKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"N_ORDER", args.n_order},
        {"N_CHARGE_DIM", args.n_charge_dim},
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    kernel->launch(
        dim3(args.n_pairs, 1, 1),
        dim3(blocksize, 1, 1),
        0,
        stream,
        args
    );
}

template void launch_tensorprod_jit<float>(
    JitCache&,
    const dmk::cuda::TensorprodArgs<float>&,
    cudaStream_t,
    int
);

template void launch_tensorprod_jit<double>(
    JitCache&,
    const dmk::cuda::TensorprodArgs<double>&,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit
