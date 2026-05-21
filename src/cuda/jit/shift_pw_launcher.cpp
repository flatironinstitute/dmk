#include "shift_pw_launcher.hpp"

#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/helpers.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace dmk::cuda::jit {
namespace {

std::string make_specialization_constants(const JitKey& key) {
    const int blocksize = required_int_param(key, "BLOCK_SIZE", "ShiftPw");
    const int n_pw_modes = required_int_param(key, "N_PW_MODES", "ShiftPw");
    const int n_charge_dim = required_int_param(key, "N_CHARGE_DIM", "ShiftPw");
    std::ostringstream ss;

    ss << "#include <dmk/cuda/shift_pw_kernelargs.hpp>\n";

    ss << "using dmk::cuda::ShiftPwArgs;\n";

    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n";
    ss << "constexpr int N_PW_MODES   = " << n_pw_modes << ";\n";
    ss << "constexpr int N_CHARGE_DIM = " << n_charge_dim << ";\n";
    ss << "using Real = " << key.real << ";\n\n";

    return ss.str();
}


} // namespace

std::string make_shift_pw_source(const JitKey& key) {
    const SplitSource split = load_split_jit_source("shiftpw.cu", "ShiftPw");

    std::ostringstream generated;

    generated << make_specialization_constants(key);
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real>
void launch_shift_pw_jit(
    JitCache& cache,
    const dmk::cuda::ShiftPwArgs<Real>& args,
    cudaStream_t stream,
    int blocksize
) {
    if (args.n_boxes_at_level == 0) {
        return;
    }

    JitKey key;
    key.name = "ShiftPwByBoxKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"BLOCK_SIZE", blocksize},
        {"N_CHARGE_DIM", args.n_charge_dim},
        {"N_PW_MODES", args.n_pw_modes},
    };

    auto kernel = cache.get_kernel(key);

    kernel->launch(
        dim3(args.n_boxes_at_level, 1, 1),
        dim3(blocksize, 1, 1),
        0,
        stream,
        args
    );
}

template <typename Real>
void launch_shift_pw_multilevel_jit(
    JitCache& cache,
    const std::vector<dmk::cuda::ShiftPwArgs<Real>>& args_h,
    dmk::cuda::ShiftPwArgs<Real>* d_args_scratch,
    cudaStream_t stream,
    int blocksize
) {
    if (args_h.empty()) {
        return;
    }

    int max_boxes = 0;

    for (const auto& a : args_h) {
        if (a.n_boxes_at_level == 0) {
            continue;
        }

        max_boxes = std::max(max_boxes, a.n_boxes_at_level);
    }

    if (max_boxes == 0) {
        return;
    }

    DMK_CHECK_CUDA(
        cudaMemcpyAsync(
            d_args_scratch,
            args_h.data(),
            args_h.size() * sizeof(dmk::cuda::ShiftPwArgs<Real>),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    JitKey key;
    key.name = "ShiftPwKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();

    key.params = {
        {"BLOCK_SIZE", blocksize},
        {"N_CHARGE_DIM", args_h[0].n_charge_dim},
        {"N_PW_MODES", args_h[0].n_pw_modes},
    };

    auto kernel = cache.get_kernel(key);

    const int n_args = static_cast<int>(args_h.size());

    kernel->launch(
        dim3(max_boxes, n_args, 1),
        dim3(blocksize, 1, 1),
        0,
        stream,
        d_args_scratch,
        n_args
    );
}

template void launch_shift_pw_jit<float>(
    JitCache&,
    const dmk::cuda::ShiftPwArgs<float>&,
    cudaStream_t,
    int
);

template void launch_shift_pw_jit<double>(
    JitCache&,
    const dmk::cuda::ShiftPwArgs<double>&,
    cudaStream_t,
    int
);

template void launch_shift_pw_multilevel_jit<float>(
    JitCache&,
    const std::vector<dmk::cuda::ShiftPwArgs<float>>&,
    dmk::cuda::ShiftPwArgs<float>*,
    cudaStream_t,
    int
);

template void launch_shift_pw_multilevel_jit<double>(
    JitCache&,
    const std::vector<dmk::cuda::ShiftPwArgs<double>>&,
    dmk::cuda::ShiftPwArgs<double>*,
    cudaStream_t,
    int
);

} // namespace dmk::cuda::jit
