#include "proxy2pw_launcher.hpp"

#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/proxy2pw_kernels.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmk::cuda::jit {
namespace {

std::string make_specialization_constants(const JitKey& key) {
    const int blocksize    = required_int_param(key, "BLOCK_SIZE", "Proxy2Pw");
    const int n_order      = required_int_param(key, "N_ORDER", "Proxy2Pw");
    const int n_pw         = required_int_param(key, "N_PW", "Proxy2Pw");
    const int n_pw2        = required_int_param(key, "N_PW2", "Proxy2Pw");
    const int n_charge_dim = required_int_param(key, "N_CHARGE_DIM", "Proxy2Pw");
    const std::string real_type = key.real;

    std::ostringstream ss;

    ss << "#include <dmk/cuda/proxy2pw_kernelargs.hpp>\n";
    ss << "using dmk::cuda::Proxy2PwArgs;\n\n";
    ss << "constexpr int N_ORDER      = " << n_order << ";\n";
    ss << "constexpr int N_PW         = " << n_pw << ";\n";
    ss << "constexpr int N_PW2        = " << n_pw2 << ";\n";
    ss << "constexpr int N_CHARGE_DIM = " << n_charge_dim << ";\n";
    ss << "constexpr int BLOCK_SIZE   = " << blocksize << ";\n";
    ss << "using Real = " << real_type << ";\n\n";

    return ss.str();
}

std::size_t proxy2pw_shared_bytes(
    int n_order,
    int n_pw,
    std::size_t sizeof_real
) {
    return
        std::size_t{2} *
        (
            std::size_t(n_order) * std::size_t(n_order) +
            std::size_t(n_order) * std::size_t(n_pw)
        ) *
        sizeof_real;
}

template <typename Real>
void check_proxy2pw_shape_or_throw(
    const dmk::cuda::Proxy2PwArgs<Real>& a
) {
    if (a.n_order <= 0 || a.n_pw <= 0 || a.n_pw2 <= 0 || a.n_charge_dim <= 0) {
        throw std::runtime_error(
            "Proxy2Pw JIT: invalid shape"
        );
    }
}


} // namespace


std::string make_proxy2pw_source(const JitKey& key) {
    std::filesystem::path filename;

    if (key.name == "Proxy2PwKernel") {
        filename = "proxy2pw.cu";
    } else if (key.name == "Proxy2PwMultiLevelKernel") {
        filename = "proxy2pw_multilevel.cu";
    } else {
        throw std::runtime_error(
            "Proxy2Pw JIT: unknown kernel name: " + key.name
        );
    }

    const SplitSource split = load_split_jit_source(filename.string(), "Proxy2Pw");

    std::ostringstream generated;

    generated << make_specialization_constants(key);

    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real>
void launch_proxy2pw_jit(
    JitCache& cache,
    const dmk::cuda::Proxy2PwArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
) {
    if (args.n_boxes_at_level == 0) {
        return;
    }

    check_proxy2pw_shape_or_throw(args);

    JitKey key;
    key.name = "Proxy2PwKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"BLOCK_SIZE", blocksize},
        {"N_ORDER", args.n_order},
        {"N_CHARGE_DIM", args.n_charge_dim},
        {"N_PW", args.n_pw},
        {"N_PW2", args.n_pw2}
    };

    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes =
        proxy2pw_shared_bytes(
            args.n_order,
            args.n_pw,
            sizeof(Real)
        );

    kernel->launch(
        dim3(args.n_boxes_at_level, 1, 1),
        dim3(blocksize, 1, 1),
        shared_bytes,
        stream,
        args
    );

}

template <typename Real>
void launch_proxy2pw_multilevel_jit(
    JitCache& cache,
    const std::vector<dmk::cuda::Proxy2PwArgs<Real>>& pa_h,
    dmk::cuda::Proxy2PwArgs<Real>* d_args_scratch,
    cudaStream_t stream,
    int blocksize
) {
    if (pa_h.empty()) {
        return;
    }

    int max_boxes = 0;
    int max_n_order = 0;
    int max_n_pw = 0;

    for (const auto& pa : pa_h) {
        max_boxes = std::max(max_boxes, pa.n_boxes_at_level);
        max_n_order = std::max(max_n_order, pa.n_order);
        max_n_pw = std::max(max_n_pw, pa.n_pw);
    }

    if (max_boxes == 0) {
        return;
    }

    DMK_CHECK_CUDA(
        cudaMemcpyAsync(
            d_args_scratch,
            pa_h.data(),
            pa_h.size() * sizeof(dmk::cuda::Proxy2PwArgs<Real>),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    JitKey key;
    key.name = "Proxy2PwMultiLevelKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"BLOCK_SIZE", blocksize},
        {"N_ORDER", pa_h[0].n_order},
        {"N_CHARGE_DIM", pa_h[0].n_charge_dim},
        {"N_PW", pa_h[0].n_pw},
        {"N_PW2", pa_h[0].n_pw2}
    };

    auto kernel = cache.get_kernel(key);

    const std::size_t shared_bytes =
        proxy2pw_shared_bytes(
            max_n_order,
            max_n_pw,
            sizeof(Real)
        );

    const int n_args = static_cast<int>(pa_h.size());

    kernel->launch(
        dim3(max_boxes, n_args, 1),
        dim3(blocksize, 1, 1),
        shared_bytes,
        stream,
        d_args_scratch,
        n_args
    );
}

template void launch_proxy2pw_jit<float>(
    JitCache&,
    const dmk::cuda::Proxy2PwArgs<float>&,
    cudaStream_t,
    int
);

template void launch_proxy2pw_jit<double>(
    JitCache&,
    const dmk::cuda::Proxy2PwArgs<double>&,
    cudaStream_t,
    int
);

template void launch_proxy2pw_multilevel_jit<float>(
    JitCache&,
    const std::vector<dmk::cuda::Proxy2PwArgs<float>>&,
    dmk::cuda::Proxy2PwArgs<float>*,
    cudaStream_t,
    int
);

template void launch_proxy2pw_multilevel_jit<double>(
    JitCache&,
    const std::vector<dmk::cuda::Proxy2PwArgs<double>>&,
    dmk::cuda::Proxy2PwArgs<double>*,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_proxy2pw(const Proxy2PwArgs<Real>& args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_proxy2pw_jit<Real>(
        jit_cache,
        args,
        stream,
        block_size
    );
}

template <typename Real, int DIM>
void launch_proxy2pw_multilevel(
    const std::vector<Proxy2PwArgs<Real>>& pa_h,
    Proxy2PwArgs<Real>* d_args_scratch,
    cudaStream_t stream
) {
    if (pa_h.empty())
        return;

    constexpr int block_size = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_proxy2pw_multilevel_jit<Real>(
        jit_cache,
        pa_h,
        d_args_scratch,
        stream,
        block_size
    );
}

template void launch_proxy2pw<float, 2>(const Proxy2PwArgs<float>&, cudaStream_t);
template void launch_proxy2pw<float, 3>(const Proxy2PwArgs<float>&, cudaStream_t);
template void launch_proxy2pw<double, 2>(const Proxy2PwArgs<double>&, cudaStream_t);
template void launch_proxy2pw<double, 3>(const Proxy2PwArgs<double>&, cudaStream_t);

template void launch_proxy2pw_multilevel<float, 2>(const std::vector<Proxy2PwArgs<float>>&, Proxy2PwArgs<float>*,
                                                   cudaStream_t);
template void launch_proxy2pw_multilevel<float, 3>(const std::vector<Proxy2PwArgs<float>>&, Proxy2PwArgs<float>*,
                                                   cudaStream_t);
template void launch_proxy2pw_multilevel<double, 2>(const std::vector<Proxy2PwArgs<double>>&, Proxy2PwArgs<double>*,
                                                    cudaStream_t);
template void launch_proxy2pw_multilevel<double, 3>(const std::vector<Proxy2PwArgs<double>>&, Proxy2PwArgs<double>*,
                                                    cudaStream_t);

} // namespace dmk::cuda
