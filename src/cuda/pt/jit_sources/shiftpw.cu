// V2 shift_pw (batched multilevel): translates each box's neighbors' outgoing
// plane-wave fields into its own incoming field (per-level pw_in_pool slab).
// The launcher prepends `using Real` + N_PW_MODES / N_CHARGE_DIM / N_NEIGHBORS /
// BLOCK_SIZE / NEIGHBOR_UNROLL. One launch covers all levels via a device array
// of per-level args. Assigns (not additive) into pw_in_pool.

#include <dmk/cuda/shift_pw_kernelargs.hpp>

using dmk::cuda::ShiftPwArgs;
using dmk::cuda::ShiftPwNeighbor;

template <typename Real>
struct alignas(2 * sizeof(Real)) complx {
    Real r;
    Real i;
};

// A neighbor's plane-wave value is read by exactly one thread, once, so there is no reuse
// for L1 to hold on to and no reason to spend a line on it. Worth ~1%: L2 traffic is already
// all payload, so this only stops the streaming values from displacing the shift table, which
// L1 was mostly keeping anyway. .cg rather than .cs because the 89% L2 hit rate is doing real
// work here and .cs would evict-first there too.
__device__ __forceinline__ complx<float> load_stream(const complx<float> *p) {
    const float2 v = __ldcg(reinterpret_cast<const float2 *>(p));
    return complx<float>{v.x, v.y};
}
__device__ __forceinline__ complx<double> load_stream(const complx<double> *p) {
    const double2 v = __ldcg(reinterpret_cast<const double2 *>(p));
    return complx<double>{v.x, v.y};
}

__device__ __forceinline__ void ShiftPwBody(ShiftPwArgs<Real> a, int box_idx) {
    if (box_idx >= a.n_boxes_at_level)
        return;

    const int *__restrict__ box_ids = a.box_ids;
    const long *__restrict__ pw_out_offsets = a.pw_out_offsets;
    const Real *__restrict__ pw_out_flat = a.pw_out_flat;
    const Real *__restrict__ wpwshift = a.wpwshift;
    Real *__restrict__ pw_in_pool = a.pw_in_pool;

    const int box = box_ids[box_idx];

    constexpr int n_pw_modes = N_PW_MODES;
    constexpr int n_charge_dim = N_CHARGE_DIM;

    Real *__restrict__ pw_in_real = pw_in_pool + box_idx * a.pw_in_stride;
    complx<Real> *__restrict__ pw_in = reinterpret_cast<complx<Real> *>(pw_in_real);

    const long self_off = pw_out_offsets[box];
    const complx<Real> *__restrict__ self_pw =
        (self_off >= 0) ? reinterpret_cast<const complx<Real> *>(pw_out_flat + 2 * self_off) : nullptr;

    // Every guard the neighbor loop used to carry -- empty slot, self, leaf-leaf pair,
    // neighbor without an outgoing expansion -- is tree-static, so the host resolved
    // them once and this is a straight run over survivors. Branch-free is the point:
    // with the guards in place ptxas could not hoist a payload load above the branch
    // that decided whether to issue it, so NEIGHBOR_UNROLL widened the body without
    // ever getting two loads in flight.
    const int nbr_begin = a.shift_nbr_offsets[box];
    const int n_nbr = a.shift_nbr_offsets[box + 1] - nbr_begin;
    const ShiftPwNeighbor *__restrict__ nbr_list = a.shift_nbr + nbr_begin;

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

#pragma unroll(NEIGHBOR_UNROLL)
        for (int e = 0; e < n_nbr; ++e) {
            const ShiftPwNeighbor nb = nbr_list[e];

            const Real *__restrict__ shift_r = wpwshift + nb.shift_ind * n_pw_modes * 2;
            const Real *__restrict__ shift_i = shift_r + n_pw_modes;
            const Real sr = shift_r[m];
            const Real si = shift_i[m];
            const complx<Real> *__restrict__ nbr_pw =
                reinterpret_cast<const complx<Real> *>(pw_out_flat + 2 * nb.pw_off);

#pragma unroll
            for (int d = 0; d < n_charge_dim; ++d) {
                const int d_base = d * n_pw_modes;
                const complx<Real> z = load_stream(&nbr_pw[d_base + m]);

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

extern "C" __global__ void PtShiftPwKernel(const ShiftPwArgs<Real> *args, int n_args) {
    const int arg_idx = blockIdx.y;
    if (arg_idx >= n_args)
        return;
    ShiftPwBody(args[arg_idx], blockIdx.x);
}
