// KERNEL_START

extern "C" __global__ void AccumulateAndScatterKernel(Real *__restrict__ out, const Real *__restrict__ pot_eval,
                                                       const Real *__restrict__ pot_extra,
                                                       const long *__restrict__ scatter_index, int dof,
                                                       long n_particles) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n_particles)
        return;
    long src = i * dof;
    long dst = scatter_index[i] * dof;
    for (int j = 0; j < dof; ++j)
        out[dst + j] = pot_eval[src + j] + pot_extra[src + j];
}

extern "C" __global__ void ScatterForwardKernel(const Real *__restrict__ in, Real *__restrict__ out,
                                                const long *__restrict__ scatter_index, long n_particles, int dof) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n_particles)
        return;

    long dst = i * dof;
    long src = scatter_index[i] * dof;
    for (int j = 0; j < dof; ++j)
        out[dst + j] = in[src + j];
}

extern "C" __global__ void ScatterForwardStressletKernel(const Real *__restrict__ densities,
                                                         const Real *__restrict__ normals, Real *__restrict__ out,
                                                         const long *__restrict__ scatter_index, long n_particles,
                                                         int dim) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n_particles)
        return;

    long dst = i * dim * dim;
    long src = scatter_index[i] * dim;
    for (int k = 0; k < dim; ++k)
        for (int j = 0; j < dim; ++j)
            out[dst + k * dim + j] = densities[src + k] * normals[src + j];
}
