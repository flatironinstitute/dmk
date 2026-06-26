template <typename Real>
struct alignas(2 * sizeof(Real)) complx {
    Real r;
    Real i;
};

__device__ __forceinline__ void ShiftPwBody(ShiftPwArgs<Real> a, int box_idx) {
    if (box_idx >= a.n_boxes_at_level)
        return;

    const int *__restrict__ box_ids = a.box_ids;
    const int *__restrict__ neighbors = a.neighbors;
    const long *__restrict__ pw_out_offsets = a.pw_out_offsets;
    const unsigned char *__restrict__ is_global_leaf = a.is_global_leaf;
    const Real *__restrict__ pw_out_flat = a.pw_out_flat;
    const Real *__restrict__ wpwshift = a.wpwshift;
    Real *__restrict__ pw_in_pool = a.pw_in_pool;

    const int box = box_ids[box_idx];

    constexpr int n_pw_modes = N_PW_MODES;
    constexpr int n_charge_dim = N_CHARGE_DIM;
    constexpr int n_neighbors = N_NEIGHBORS;

    const bool box_is_leaf = (is_global_leaf[box] != 0);

    Real *__restrict__ pw_in_real = pw_in_pool + box_idx * a.pw_in_stride;
    complx<Real> *__restrict__ pw_in = reinterpret_cast<complx<Real> *>(pw_in_real);

    const long self_off = pw_out_offsets[box];
    const complx<Real> *__restrict__ self_pw =
        (self_off >= 0) ? reinterpret_cast<const complx<Real> *>(pw_out_flat + 2 * self_off) : nullptr;

    const int *__restrict__ box_neighbors = neighbors + static_cast<long>(box) * n_neighbors;

    for (int m = threadIdx.x; m < n_pw_modes; m += blockDim.x) {
        complx<Real> acc[n_charge_dim];

#pragma unroll
        for (int d = 0; d < n_charge_dim; ++d) {
            const int d_base = d * n_pw_modes;
            if (self_pw)
                acc[d] = self_pw[d_base + m];
            else
                acc[d] = complx<Real>{Real{0}, Real{0}};
        }

#if SHIFT_PW_NEIGHBOR_UNROLL == 0
#pragma unroll
#elif SHIFT_PW_NEIGHBOR_UNROLL == 1
#pragma unroll 1
#elif SHIFT_PW_NEIGHBOR_UNROLL == 3
#pragma unroll 3
#elif SHIFT_PW_NEIGHBOR_UNROLL == 9
#pragma unroll 9
#else
#pragma unroll SHIFT_PW_NEIGHBOR_UNROLL
#endif
        for (int npos = 0; npos < n_neighbors; ++npos) {
            const int neighbor = box_neighbors[npos];
            if (neighbor < 0 || neighbor == box)
                continue;
            if (box_is_leaf && is_global_leaf[neighbor] != 0)
                continue;

            const long nbr_off = pw_out_offsets[neighbor];
            if (nbr_off < 0)
                continue;

            const int ind = n_neighbors - 1 - npos;
            const Real *__restrict__ shift_r = wpwshift + ind * n_pw_modes * 2;
            const Real *__restrict__ shift_i = shift_r + n_pw_modes;
            const Real sr = shift_r[m];
            const Real si = shift_i[m];
            const complx<Real> *__restrict__ nbr_pw =
                reinterpret_cast<const complx<Real> *>(pw_out_flat + 2 * nbr_off);

#pragma unroll
            for (int d = 0; d < n_charge_dim; ++d) {
                const int d_base = d * n_pw_modes;
                const complx<Real> z = nbr_pw[d_base + m];

                acc[d].r += z.r * sr - z.i * si;
                acc[d].i += z.r * si + z.i * sr;
            }
        }

#pragma unroll
        for (int d = 0; d < n_charge_dim; ++d) {
            const int d_base = d * n_pw_modes;
            pw_in[d_base + m] = acc[d];
        }
    }
}

// KERNEL_START

extern "C" __global__ void ShiftPwByBoxKernel(ShiftPwArgs<Real> a) {
    ShiftPwBody(a, blockIdx.x);
}

extern "C" __global__ void ShiftPwKernel(const ShiftPwArgs<Real> *args, int n_args) {
    const int arg_idx = blockIdx.y;
    if (arg_idx >= n_args)
        return;
    ShiftPwBody(args[arg_idx], blockIdx.x);
}
