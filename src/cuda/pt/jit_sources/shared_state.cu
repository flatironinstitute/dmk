// V2 charge/potential helpers: the stresslet outer product and the finalize accumulate (near +
// far sum), both in the tree's particle order. The launcher prepends `using Real` + BLOCK_SIZE
// (unused by the bodies, kept uniform with the thin-launcher contract).

// KERNEL_START

extern "C" __global__ void PtAccumulateKernel(Real *__restrict__ out, const Real *__restrict__ pot_eval,
                                              const Real *__restrict__ pot_extra, int dof, long n_particles) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n_particles)
        return;
    long off = i * dof;
    for (int j = 0; j < dof; ++j)
        out[off + j] = pot_eval[off + j] + pot_extra[off + j];
}

extern "C" __global__ void PtStressletChargeKernel(const Real *__restrict__ densities,
                                                   const Real *__restrict__ normals, Real *__restrict__ out,
                                                   long n_particles, int dim) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n_particles)
        return;

    long dst = i * dim * dim;
    long src = i * dim;
    for (int k = 0; k < dim; ++k)
        for (int j = 0; j < dim; ++j)
            out[dst + k * dim + j] = densities[src + k] * normals[src + j];
}

// Clears the proxy slabs of boxes no upward writer touches, so the buffer as a whole never
// needs a memset. One block per box; slab length is uniform.
extern "C" __global__ void PtZeroBoxSlabsKernel(Real *__restrict__ flat, const long *__restrict__ offsets,
                                                const int *__restrict__ boxes, int n_boxes, int slab_reals) {
    if (blockIdx.x >= n_boxes)
        return;
    const long off = offsets[boxes[blockIdx.x]];
    if (off < 0)
        return;
    Real *__restrict__ slab = flat + off;
    for (int i = threadIdx.x; i < slab_reals; i += blockDim.x)
        slab[i] = Real{0};
}
