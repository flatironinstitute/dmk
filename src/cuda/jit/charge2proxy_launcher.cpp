#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"
#include <dmk/cuda/charge2proxy_kernels.hpp>

#include <cuda_runtime.h>

#include <sstream>
#include <string>
#include <memory>
namespace dmk::cuda::jit {

namespace {

std::string make_specialization_constants(const JitKey& key) {
    const int n_order      = required_int_param(key, "N_ORDER", "Charge2Proxy");
    const int n_charge_dim = required_int_param(key, "N_CHARGE_DIM", "Charge2Proxy");
    const int chunk        = required_int_param(key, "CHUNK", "Charge2Proxy");
    const int i_tile       = required_int_param(key, "I_TILE", "Charge2Proxy");
    const int j_tile       = required_int_param(key, "J_TILE", "Charge2Proxy");
    const int k_tile       = required_int_param(key, "K_TILE", "Charge2Proxy");
    const std::string real_type = key.real;
    std::ostringstream ss;
    ss << "#include <dmk/cuda/charge2proxy_kernelargs.hpp>\n"; 
    ss << "using dmk::cuda::Charge2ProxyArgs;\n\n";
    ss << "constexpr int N_ORDER      = " << n_order << ";\n";
    ss << "constexpr int N_CHARGE_DIM = " << n_charge_dim << ";\n";
    ss << "constexpr int CHUNK        = " << chunk << ";\n";
    ss << "constexpr int I_TILE       = " << i_tile << ";\n";
    ss << "constexpr int J_TILE       = " << j_tile << ";\n";
    ss << "constexpr int K_TILE       = " << k_tile << ";\n";
    ss << "using Real = " << real_type << "; \n\n";
    return ss.str();

}


std::size_t charge2proxy_shared_bytes(int n_order, int n_charge_dim, int chunk, std::size_t sizeof_real) {
    const int ld = chunk + 1;

    return (std::size_t{3} * std::size_t(n_order) * std::size_t(ld) + std::size_t(n_charge_dim) * std::size_t(ld)) * sizeof_real;
}

} // namespace


std::string make_charge2proxy_source(const JitKey& key) {
    const SplitSource split = load_split_jit_source("charge2proxy.cu", "Charge2Proxy");
    std::ostringstream generated;
    generated << split.header << "\n";

    generated << make_specialization_constants(key);

    generated << split.kernel << "\n";


    return generated.str();
}

template <typename Real>
void launch_charge2proxy_jit(
    JitCache& cache,
    const dmk::cuda::Charge2ProxyArgs<Real>& args,
    const int* group_perm,
    int n_launch_groups,
    cudaStream_t stream,
    int chunk,
    int i_tile,
    int j_tile,
    int k_tile,
    int blocksize
) {
    if (args.n_groups == 0 || n_launch_groups == 0) {
        return;
    }

    JitKey key;
    key.name = "Charge2ProxyKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"N_ORDER", args.n_order},
        {"N_CHARGE_DIM", args.n_charge_dim},
        {"CHUNK", chunk},
        {"I_TILE", i_tile},
        {"J_TILE", j_tile},
        {"K_TILE", k_tile},
        {"BLOCK_SIZE", blocksize},
    };
    
    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes = charge2proxy_shared_bytes(args.n_order, args.n_charge_dim, chunk, sizeof(Real));
    
    kernel->launch(
        dim3(n_launch_groups, 1, 1),
        dim3(blocksize, 1, 1),
        shared_bytes,
        stream,
        args,
        group_perm
    );
}

template void launch_charge2proxy_jit<float>(
    JitCache&,
    const dmk::cuda::Charge2ProxyArgs<float>&,
    const int*,
    int,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int
);

template void launch_charge2proxy_jit<double>(
    JitCache&,
    const dmk::cuda::Charge2ProxyArgs<double>&,
    const int*,
    int,
    cudaStream_t,
    int,
    int,
    int,
    int,
    int
);

}
