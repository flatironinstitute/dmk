#include <dmk/cuda/shared_state_kernels.hpp>
#include <stdexcept>

namespace dmk::cuda {

template <typename Real, int DOF>
__global__ void accumulate_and_scatter_kernel(Real *__restrict__ out, const Real *__restrict__ pot_eval,
                                              const Real *__restrict__ pot_extra,
                                              const long *__restrict__ scatter_index, long n_particles) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n_particles)
        return;
    long src = i * DOF;
    long dst = scatter_index[i] * DOF;
    for (int j = 0; j < DOF; ++j)
        out[dst + j] = pot_eval[src + j] + pot_extra[src + j];
}

template <typename Real, int DOF>
__global__ void scatter_forward_kernel(const Real *__restrict__ in, Real *__restrict__ out,
                                       const long *__restrict__ scatter_index, long n_particles) {
    const long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles)
        return;

    const long dst = i * DOF;
    const long src = scatter_index[i] * DOF;
    for (int j = 0; j < DOF; ++j)
        out[dst + j] = in[src + j];
}

template <typename Real>
void launch_scatter_forward(const Real *in, Real *out, const long *scatter_index, long n_particles, int dof,
                            cudaStream_t stream) {
    if (n_particles == 0)
        return;
    constexpr int block = 256;
    long grid = (n_particles + block - 1) / block;
    if (dof == 1)
        return scatter_forward_kernel<Real, 1><<<grid, block, 0, stream>>>(in, out, scatter_index, n_particles);
    if (dof == 2)
        return scatter_forward_kernel<Real, 2><<<grid, block, 0, stream>>>(in, out, scatter_index, n_particles);
    if (dof == 3)
        return scatter_forward_kernel<Real, 3><<<grid, block, 0, stream>>>(in, out, scatter_index, n_particles);

    throw std::runtime_error("Scatter forward sorting not implemented on GPU for kernel/dof combo");
}

template <typename Real, int DIM>
__global__ void scatter_forward_stresslet_kernel(const Real *__restrict__ densities, const Real *__restrict__ normals,
                                                 Real *__restrict__ out, const long *__restrict__ scatter_index,
                                                 long n_particles) {
    const long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles)
        return;

    const long dst = i * DIM * DIM;
    const long src = scatter_index[i] * DIM * DIM;
    for (int k = 0; k < DIM; ++k)
        for (int j = 0; j < DIM; ++j)
            out[dst + k * DIM + j] = densities[src + k] * normals[src + j];
}

template <typename Real>
void launch_scatter_forward_stresslet(const Real *densities, const Real *normals, Real *out, const long *scatter_index,
                                      long n_particles, int dim, cudaStream_t stream) {
    if (n_particles == 0)
        return;
    constexpr int block = 256;
    long grid = (n_particles + block - 1) / block;
    if (dim == 2)
        return scatter_forward_stresslet_kernel<Real, 2>
            <<<grid, block, 0, stream>>>(densities, normals, out, scatter_index, n_particles);
    if (dim == 3)
        return scatter_forward_stresslet_kernel<Real, 3>
            <<<grid, block, 0, stream>>>(densities, normals, out, scatter_index, n_particles);

    throw std::runtime_error("Charge sorting not implemented on GPU for stresslet/DIM combo");
}

template <typename Real>
void launch_accumulate_and_scatter(Real *out, const Real *pot_eval, const Real *pot_extra, const long *scatter_index,
                                   int dof, long n_particles, cudaStream_t stream) {
    if (n_particles == 0)
        return;
    constexpr int block = 256;
    long grid = (n_particles + block - 1) / block;
    if (dof == 1)
        return accumulate_and_scatter_kernel<Real, 1>
            <<<grid, block, 0, stream>>>(out, pot_eval, pot_extra, scatter_index, n_particles);
    if (dof == 2)
        return accumulate_and_scatter_kernel<Real, 2>
            <<<grid, block, 0, stream>>>(out, pot_eval, pot_extra, scatter_index, n_particles);
    if (dof == 3)
        return accumulate_and_scatter_kernel<Real, 3>
            <<<grid, block, 0, stream>>>(out, pot_eval, pot_extra, scatter_index, n_particles);

    throw std::runtime_error("Potential sorting not implemented on GPU for kernel/DIM combo");
}

template void launch_accumulate_and_scatter<float>(float *, const float *, const float *, const long *, int, long,
                                                   cudaStream_t);
template void launch_accumulate_and_scatter<double>(double *, const double *, const double *, const long *, int, long,
                                                    cudaStream_t);
template void launch_scatter_forward<float>(const float *in, float *out, const long *scatter_index, long n_particles,
                                            int dof, cudaStream_t stream);
template void launch_scatter_forward<double>(const double *in, double *out, const long *scatter_index, long n_particles,
                                             int dof, cudaStream_t stream);

template void launch_scatter_forward_stresslet<float>(const float *charges, const float *normals, float *out,
                                                      const long *scatter_index, long n_particles, int dof,
                                                      cudaStream_t stream);
template void launch_scatter_forward_stresslet<double>(const double *charges, const double *normals, double *out,
                                                       const long *scatter_index, long n_particles, int dof,
                                                       cudaStream_t stream);

} // namespace dmk::cuda
