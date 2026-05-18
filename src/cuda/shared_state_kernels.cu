#include <dmk/cuda/shared_state_kernels.hpp>

namespace dmk::cuda {

template <typename Real>
__global__ void accumulate_and_scatter_kernel(Real *__restrict__ out, const Real *__restrict__ pot_eval,
                                              const Real *__restrict__ pot_extra,
                                              const long *__restrict__ scatter_index, int dof, long n_particles) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n_particles)
        return;
    long src = i * dof;
    long dst = scatter_index[i] * dof;
    for (int j = 0; j < dof; ++j)
        out[dst + j] = pot_eval[src + j] + pot_extra[src + j];
}

template <typename Real>
void launch_accumulate_and_scatter(Real *out, const Real *pot_eval, const Real *pot_extra, const long *scatter_index,
                                   int dof, long n_particles, cudaStream_t stream) {
    if (n_particles == 0)
        return;
    constexpr int block = 256;
    long grid = (n_particles + block - 1) / block;
    accumulate_and_scatter_kernel<<<grid, block, 0, stream>>>(out, pot_eval, pot_extra, scatter_index, dof,
                                                              n_particles);
}

template void launch_accumulate_and_scatter<float>(float *, const float *, const float *, const long *, int, long,
                                                   cudaStream_t);
template void launch_accumulate_and_scatter<double>(double *, const double *, const double *, const long *, int, long,
                                                    cudaStream_t);

} // namespace dmk::cuda
