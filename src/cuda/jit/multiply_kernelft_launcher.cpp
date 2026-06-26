#include "multiply_kernelft_launcher.hpp"

#include "jit_cache.hpp"
#include "jit_kernel.hpp"
#include "jit_source_utils.hpp"
#include "jit_types.hpp"

#include <dmk/cuda/multiply_kernelft_kernels.hpp>

#include <cuda_runtime.h>

#include <sstream>
#include <string>

namespace dmk::cuda::jit {
namespace {

std::string make_specialization_constants(const JitKey &key) {
    const int blocksize = required_int_param(key, "BLOCK_SIZE", "MultiplyKernelFT");

    std::ostringstream ss;

    ss << "#include <dmk/cuda/multiply_kernelft_kernelargs.hpp>\n";
    ss << "using dmk::cuda::MultiplyCd2pArgs;\n";
    ss << "using dmk::cuda::MultiplyStokeslet3DArgs;\n";
    ss << "using dmk::cuda::MultiplyStresslet3DArgs;\n\n";
    ss << "using Real = " << key.real << ";\n\n";
    ss << "constexpr int BLOCK_SIZE = " << blocksize << ";\n\n";

    return ss.str();
}

std::size_t stokeslet_shared_bytes(std::size_t sizeof_real) { return sizeof_real * 6; }

} // namespace

std::string make_multiply_kernelft_source(const JitKey &key) {
    const SplitSource split = load_split_jit_source("multiply_kernelft.cu", "MultiplyKernelFT");

    std::ostringstream generated;

    generated << make_specialization_constants(key) << "\n";
    generated << split.header << "\n";
    generated << split.kernel << "\n";

    return generated.str();
}

template <typename Real, int DIM>
void launch_multiply_cd2p_jit(JitCache &cache, const dmk::cuda::MultiplyCd2pArgs<Real> &args, cudaStream_t stream,
                              int blocksize) {
    if (args.n_boxes_at_level == 0) {
        return;
    }

    JitKey key;
    key.name = "MultiplyCd2pByBoxKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {
        {"DIM", DIM},
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    kernel->launch(dim3(args.n_boxes_at_level, 1, 1), dim3(blocksize, 1, 1), 0, stream, args);
}

template <typename Real>
void launch_multiply_stokeslet_3d_jit(JitCache &cache, const dmk::cuda::MultiplyStokeslet3DArgs<Real> &args,
                                      cudaStream_t stream, int blocksize) {
    if (args.n_boxes_at_level == 0) {
        return;
    }

    JitKey key;
    key.name = "MultiplyStokeslet3DByBoxKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    kernel->launch(dim3(args.n_boxes_at_level, 1, 1), dim3(blocksize, 1, 1), stokeslet_shared_bytes(sizeof(Real)),
                   stream, args);
}

template <typename Real>
void launch_multiply_stresslet_3d_jit(JitCache &cache, const dmk::cuda::MultiplyStresslet3DArgs<Real> &args,
                                      cudaStream_t stream, int blocksize) {
    if (args.n_boxes_at_level == 0) {
        return;
    }

    JitKey key;
    key.name = "MultiplyStresslet3DByBoxKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    key.params = {
        {"BLOCK_SIZE", blocksize},
    };

    auto kernel = cache.get_kernel(key);

    kernel->launch(dim3(args.n_boxes_at_level, 1, 1), dim3(blocksize, 1, 1), 0, stream, args);
}

template void launch_multiply_cd2p_jit<float, 2>(JitCache &, const dmk::cuda::MultiplyCd2pArgs<float> &, cudaStream_t,
                                                 int);

template void launch_multiply_cd2p_jit<float, 3>(JitCache &, const dmk::cuda::MultiplyCd2pArgs<float> &, cudaStream_t,
                                                 int);

template void launch_multiply_cd2p_jit<double, 2>(JitCache &, const dmk::cuda::MultiplyCd2pArgs<double> &, cudaStream_t,
                                                  int);

template void launch_multiply_cd2p_jit<double, 3>(JitCache &, const dmk::cuda::MultiplyCd2pArgs<double> &, cudaStream_t,
                                                  int);

template void launch_multiply_stokeslet_3d_jit<float>(JitCache &, const dmk::cuda::MultiplyStokeslet3DArgs<float> &,
                                                      cudaStream_t, int);

template void launch_multiply_stokeslet_3d_jit<double>(JitCache &, const dmk::cuda::MultiplyStokeslet3DArgs<double> &,
                                                       cudaStream_t, int);

template void launch_multiply_stresslet_3d_jit<float>(JitCache &, const dmk::cuda::MultiplyStresslet3DArgs<float> &,
                                                      cudaStream_t, int);

template void launch_multiply_stresslet_3d_jit<double>(JitCache &, const dmk::cuda::MultiplyStresslet3DArgs<double> &,
                                                       cudaStream_t, int);

} // namespace dmk::cuda::jit

namespace dmk::cuda {

template <typename Real, int DIM>
void launch_multiply_cd2p(const MultiplyCd2pArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_multiply_cd2p_jit<Real, DIM>(jit_cache, args, stream, block_size);
}

template <typename Real>
void launch_multiply_stokeslet_3d(const MultiplyStokeslet3DArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_multiply_stokeslet_3d_jit<Real>(jit_cache, args, stream, block_size);
}

template <typename Real>
void launch_multiply_stresslet_3d(const MultiplyStresslet3DArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;

    constexpr int block_size = 128;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_multiply_stresslet_3d_jit<Real>(jit_cache, args, stream, block_size);
}

template void launch_multiply_cd2p<float, 2>(const MultiplyCd2pArgs<float> &, cudaStream_t);
template void launch_multiply_cd2p<float, 3>(const MultiplyCd2pArgs<float> &, cudaStream_t);
template void launch_multiply_cd2p<double, 2>(const MultiplyCd2pArgs<double> &, cudaStream_t);
template void launch_multiply_cd2p<double, 3>(const MultiplyCd2pArgs<double> &, cudaStream_t);
template void launch_multiply_stokeslet_3d<float>(const MultiplyStokeslet3DArgs<float> &, cudaStream_t);
template void launch_multiply_stokeslet_3d<double>(const MultiplyStokeslet3DArgs<double> &, cudaStream_t);
template void launch_multiply_stresslet_3d<float>(const MultiplyStresslet3DArgs<float> &, cudaStream_t);
template void launch_multiply_stresslet_3d<double>(const MultiplyStresslet3DArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
