// V2 charge/potential scatter helpers: forward-permute sorted<-user layouts
// (scalar and stresslet outer-product) and the finalize accumulate (near + far
// far-field sum, descattered to user order). The launcher prepends `using Real`
// + BLOCK_SIZE (unused by the bodies, kept uniform with the thin-launcher
// contract).

// KERNEL_START

extern "C" __global__ void PtAccumulateAndScatterKernel(Real *__restrict__ out, const Real *__restrict__ pot_eval,
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

extern "C" __global__ void PtScatterForwardKernel(const Real *__restrict__ in, Real *__restrict__ out,
                                                  const long *__restrict__ scatter_index, long n_particles, int dof) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n_particles)
        return;

    long dst = i * dof;
    long src = scatter_index[i] * dof;
    for (int j = 0; j < dof; ++j)
        out[dst + j] = in[src + j];
}

extern "C" __global__ void PtScatterForwardStressletKernel(const Real *__restrict__ densities,
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
