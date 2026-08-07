// V2 shift_pw (batched multilevel): translates each box's neighbors' outgoing
// plane-wave fields into its own incoming field (per-level pw_in_pool slab).
// One block covers SHIFT_GROUP consecutive boxes of a level, whose source sets the
// host has merged, so each source slab is fetched once for the whole group.
// The launcher prepends `using Real` + N_PW_MODES / N_PW_LIVE / N_CHARGE_DIM /
// N_NEIGHBORS / SHIFT_GROUP / BLOCK_SIZE / NEIGHBOR_UNROLL. One launch covers all
// levels via a device array of per-level args. Assigns (not additive) into pw_in_pool.

#include <dmk/cuda/shift_pw_kernelargs.hpp>

using dmk::cuda::ShiftPwArgs;
using dmk::cuda::ShiftPwGroupSrc;

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

__device__ __forceinline__ void ShiftPwBody(ShiftPwArgs<Real> a, int group) {
    if (group >= a.n_groups_at_level)
        return;

    const int *__restrict__ box_ids = a.box_ids;
    const long *__restrict__ pw_out_offsets = a.pw_out_offsets;
    const Real *__restrict__ pw_out_flat = a.pw_out_flat;
    const Real *__restrict__ wpwshift = a.wpwshift;
    Real *__restrict__ pw_in_pool = a.pw_in_pool;

    constexpr int n_pw_modes = N_PW_MODES;
    constexpr int n_charge_dim = N_CHARGE_DIM;
    constexpr int group_size = SHIFT_GROUP;

    const int base = group * group_size;
    const int rem = a.n_boxes_at_level - base;
    const int n_mem = rem < group_size ? rem : group_size;

    // Self offsets go to shared rather than registers: group_size longs would cost
    // 2*group_size registers held across the whole mode loop, which at group_size 8 is
    // enough to cost a block of occupancy on top of what the accumulators already take.
    __shared__ long s_self_off[group_size];
    if (threadIdx.x < group_size)
        s_self_off[threadIdx.x] = (threadIdx.x < n_mem) ? pw_out_offsets[box_ids[base + threadIdx.x]] : -1;
    __syncthreads();

    // Every guard the neighbor loop used to carry -- empty slot, self, leaf-leaf pair,
    // neighbor without an outgoing expansion -- is tree-static, so the host resolved
    // them once. It also merged the group's members into one source list, so a source
    // shared by several members is fetched once and applied to each. The payload load
    // is unconditional; only the per-member shift is predicated, and that reads the
    // wpwshift table, which stays in L1.
    const int src_begin = a.group_offsets[group];
    const int n_src = a.group_offsets[group + 1] - src_begin;
    const ShiftPwGroupSrc *__restrict__ src_list = a.group_src + src_begin;

    // Live modes are the front of the slab, which keeps its original stride, so this is the
    // full-slab loop over a shorter prefix. The dead tail is neither read nor written.
    for (int m = threadIdx.x; m < N_PW_LIVE; m += blockDim.x) {
        complx<Real> acc[group_size][n_charge_dim];

#pragma unroll
        for (int t = 0; t < group_size; ++t) {
            const long self_off = s_self_off[t];
#pragma unroll
            for (int d = 0; d < n_charge_dim; ++d) {
                if (self_off >= 0) {
                    const complx<Real> *__restrict__ self_pw =
                        reinterpret_cast<const complx<Real> *>(pw_out_flat + 2 * self_off);
                    acc[t][d] = load_stream(&self_pw[d * n_pw_modes + m]);
                } else {
                    acc[t][d] = complx<Real>{Real{0}, Real{0}};
                }
            }
        }

#pragma unroll(NEIGHBOR_UNROLL)
        for (int e = 0; e < n_src; ++e) {
            const ShiftPwGroupSrc s = src_list[e];
            const complx<Real> *__restrict__ nbr_pw =
                reinterpret_cast<const complx<Real> *>(pw_out_flat + 2 * s.pw_off);

            complx<Real> z[n_charge_dim];
#pragma unroll
            for (int d = 0; d < n_charge_dim; ++d)
                z[d] = load_stream(&nbr_pw[d * n_pw_modes + m]);

#pragma unroll
            for (int t = 0; t < group_size; ++t) {
                const int ind = s.shift_ind[t];
                if (ind < 0)
                    continue;
                const Real *__restrict__ shift_r = wpwshift + ind * n_pw_modes * 2;
                const Real sr = shift_r[m];
                const Real si = shift_r[n_pw_modes + m];
#pragma unroll
                for (int d = 0; d < n_charge_dim; ++d) {
                    acc[t][d].r += z[d].r * sr - z[d].i * si;
                    acc[t][d].i += z[d].r * si + z[d].i * sr;
                }
            }
        }

#pragma unroll
        for (int t = 0; t < group_size; ++t) {
            if (t < n_mem) {
                complx<Real> *__restrict__ pw_in =
                    reinterpret_cast<complx<Real> *>(pw_in_pool + (base + t) * a.pw_in_stride);
#pragma unroll
                for (int d = 0; d < n_charge_dim; ++d)
                    pw_in[d * n_pw_modes + m] = acc[t][d];
            }
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
