// Shared-state launchers. Kernel implementation lives in
// src/cuda/jit_sources/shared_state.cu and is compiled with NVRTC.

#include <dmk/cuda/shared_state_kernels.hpp>

#include "cuda/jit/shared_state_launcher.hpp"

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real>
void launch_accumulate_and_scatter(Real *out, const Real *pot_eval, const Real *pot_extra, const long *scatter_index,
                                   int dof, long n_particles, cudaStream_t stream) {
    if (n_particles == 0)
        return;

    constexpr int block = 256;
    static dmk::cuda::jit::JitCache jit_cache;

    dmk::cuda::jit::launch_accumulate_and_scatter_jit<Real>(jit_cache, out, pot_eval, pot_extra, scatter_index, dof,
                                                            n_particles, stream, block);
}

template void launch_accumulate_and_scatter<float>(float *, const float *, const float *, const long *, int, long,
                                                   cudaStream_t);
template void launch_accumulate_and_scatter<double>(double *, const double *, const double *, const long *, int, long,
                                                    cudaStream_t);

} // namespace dmk::cuda
