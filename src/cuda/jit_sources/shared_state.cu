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
