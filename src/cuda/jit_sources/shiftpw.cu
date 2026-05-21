template <typename Real>
struct alignas(2 * sizeof(Real)) complx {
    Real r;
    Real i;
};

__device__ __forceinline__ void ShiftPwBody(ShiftPwArgs<Real> a, int box_idx) {
    if (box_idx >= a.n_boxes_at_level)
        return;

    const int box = a.box_ids[box_idx];

    const int n_pw_modes = N_PW_MODES;
    const int n_charge_dim = N_CHARGE_DIM;
    const int n_neighbors = a.n_neighbors;

    const bool box_is_leaf = (a.is_global_leaf[box] != 0);

    Real *pw_in_real = a.pw_in_pool + box_idx * a.pw_in_stride;
    complx<Real> *pw_in = reinterpret_cast<complx<Real> *>(pw_in_real);

    const long self_off = a.pw_out_offsets[box];
    const complx<Real> *self_pw =
        (self_off >= 0) ? reinterpret_cast<const complx<Real> *>(a.pw_out_flat + 2 * self_off) : nullptr;

    for (int d = 0; d < n_charge_dim; ++d) {
        const int d_base = d * n_pw_modes;

        for (int m = threadIdx.x; m < n_pw_modes; m += blockDim.x) {
            complx<Real> acc;
            if (self_pw)
                acc = self_pw[d_base + m];
            else
                acc = complx<Real>{Real{0}, Real{0}};

            for (int npos = 0; npos < n_neighbors; ++npos) {
                const int neighbor = a.neighbors[box * n_neighbors + npos];
                if (neighbor < 0 || neighbor == box)
                    continue;
                if (box_is_leaf && a.is_global_leaf[neighbor] != 0)
                    continue;

                const long nbr_off = a.pw_out_offsets[neighbor];
                if (nbr_off < 0)
                    continue;

                const int ind = n_neighbors - 1 - npos;
                const Real *shift_r = a.wpwshift + ind * n_pw_modes * 2;
                const Real *shift_i = shift_r + n_pw_modes;
                const complx<Real> *nbr_pw = reinterpret_cast<const complx<Real> *>(a.pw_out_flat + 2 * nbr_off);

                const complx<Real> z = nbr_pw[d_base + m];
                const Real sr = shift_r[m];
                const Real si = shift_i[m];

                acc.r += z.r * sr - z.i * si;
                acc.i += z.r * si + z.i * sr;
            }

            pw_in[d_base + m] = acc;
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
