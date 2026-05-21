// KERNEL_START

extern "C" __global__ void SelfCorrectionKernel(SelfCorrectionArgs<Real> a) {
    int idx = blockIdx.x;
    if (idx >= a.n_direct_work)
        return;

    Real factor = a.correction_factors[idx];
    if (factor == Real{0})
        return;

    int box = a.direct_work[idx];
    if (!a.src_counts_owned[box])
        return;

    int count = a.src_counts_halo[box];
    long pot_off = a.pot_src_offsets[box];
    long chg_off = a.charge_halo_offsets[box];

    for (int i_src = threadIdx.x; i_src < count; i_src += blockDim.x)
        for (int i = 0; i < a.n_input_dim; i++)
            a.pot_src[pot_off + i_src * a.pot_stride + i] -=
                factor * a.charge_halo[chg_off + i_src * a.n_input_dim + i];
}
