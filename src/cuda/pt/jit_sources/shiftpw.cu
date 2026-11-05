// V2 shift_pw (batched multilevel): translates each box's neighbors' outgoing
// plane-wave fields into its own incoming field (per-level pw_in_pool slab).
// One block covers one group -- the plane-wave-evaluating children of one parent --
// over one window of BLOCK_SIZE modes, so each source slab is fetched once for the
// whole group. The launcher prepends `using Real` + N_PW_MODES / N_PW_LIVE /
// N_CHARGE_DIM / BLOCK_SIZE. One launch covers all levels via a device array of
// per-level args. Assigns (not additive) into pw_in_pool. 3D only.

#include <dmk/cuda/shift_pw_kernelargs.hpp>

using dmk::cuda::ShiftPwArgs;
using dmk::cuda::ShiftPwHalo;

template <typename Real>
struct alignas(2 * sizeof(Real)) complx {
    Real r;
    Real i;
};

// A neighbor's plane-wave value is read by exactly one thread, once, so there is no reuse
// for L1 to hold on to and no reason to spend a line on it. .cg rather than .cs because the
// L2 hit rate is doing real work here and .cs would evict-first there too.
__device__ __forceinline__ complx<float> load_stream(const complx<float> *p) {
    const float2 v = __ldcg(reinterpret_cast<const float2 *>(p));
    return complx<float>{v.x, v.y};
}
__device__ __forceinline__ complx<double> load_stream(const complx<double> *p) {
    const double2 v = __ldcg(reinterpret_cast<const double2 *>(p));
    return complx<double>{v.x, v.y};
}

__device__ __forceinline__ complx<Real> cmul(complx<Real> a, complx<Real> b) {
    return complx<Real>{a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}

// x * c^q for q in {-1, 0, 1, 2}, given c and c^2. Every call has a constant q once the halo
// loops unroll, so the selection compiles away.
__device__ __forceinline__ complx<Real> mul_pow(complx<Real> x, complx<Real> c, complx<Real> c2, int q) {
    if (q == 0)
        return x;
    if (q == 1)
        return cmul(x, c);
    if (q == 2)
        return cmul(x, c2);
    return cmul(x, complx<Real>{c.r, -c.i});
}

__device__ __forceinline__ void ShiftPwBody(const ShiftPwArgs<Real> &a, int group, int window) {
    constexpr int n_pw_modes = N_PW_MODES;
    constexpr int n_charge_dim = N_CHARGE_DIM;

    __shared__ long s_off[64];
    __shared__ unsigned s_take[64];
    __shared__ int s_member[8];
    const ShiftPwHalo &g = a.halo[group];
    for (int h = threadIdx.x; h < 64; h += BLOCK_SIZE) {
        s_off[h] = g.pw_off[h];
        s_take[h] = g.take[h];
    }
    if (threadIdx.x < 8)
        s_member[threadIdx.x] = g.member[threadIdx.x];
    __syncthreads();

    // One mode per thread, one window of modes per block. Live modes are the front of the slab,
    // which keeps its original stride; the dead tail is neither read nor written.
    const int m = window * BLOCK_SIZE + threadIdx.x;
    if (m >= N_PW_LIVE)
        return;

    // The mode's phase per axis and its square: every power a halo offset needs.
    const int axes = a.mode_axes[m];
    const complx<Real> *__restrict__ phase = reinterpret_cast<const complx<Real> *>(a.phase);
    complx<Real> c[3], c2[3];
#pragma unroll
    for (int ax = 0; ax < 3; ++ax) {
        c[ax] = phase[(axes >> (8 * ax)) & 0xff];
        c2[ax] = cmul(c[ax], c[ax]);
    }

    const complx<Real> *__restrict__ pw_out = reinterpret_cast<const complx<Real> *>(a.pw_out_flat);
    complx<Real> acc[8][n_charge_dim];
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int d = 0; d < n_charge_dim; ++d)
            acc[t][d] = complx<Real>{Real{0}, Real{0}};

#pragma unroll
    for (int q0 = -1; q0 <= 2; ++q0) {
        const complx<Real> s0 = mul_pow(complx<Real>{Real{1}, Real{0}}, c[0], c2[0], q0);
#pragma unroll
        for (int q1 = -1; q1 <= 2; ++q1) {
            const complx<Real> s01 = mul_pow(s0, c[1], c2[1], q1);
#pragma unroll
            for (int q2 = -1; q2 <= 2; ++q2) {
                const int h = 16 * (q0 + 1) + 4 * (q1 + 1) + (q2 + 1);
                const long off = s_off[h];
                if (off < 0)
                    continue;
                const complx<Real> s = mul_pow(s01, c[2], c2[2], q2);
                complx<Real> zs[n_charge_dim];
#pragma unroll
                for (int d = 0; d < n_charge_dim; ++d)
                    zs[d] = cmul(load_stream(&pw_out[off + d * n_pw_modes + m]), s);
                const unsigned take = s_take[h];
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const int o0 = q0 - (t >> 2), o1 = q1 - ((t >> 1) & 1), o2 = q2 - (t & 1);
                    if (o0 < -1 || o0 > 1 || o1 < -1 || o1 > 1 || o2 < -1 || o2 > 1)
                        continue;
                    if (take & (1u << t)) {
#pragma unroll
                        for (int d = 0; d < n_charge_dim; ++d) {
                            acc[t][d].r += zs[d].r;
                            acc[t][d].i += zs[d].i;
                        }
                    }
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int slot = s_member[t];
        if (slot < 0)
            continue;
        complx<Real> *__restrict__ pw_in = reinterpret_cast<complx<Real> *>(a.pw_in_pool + slot * a.pw_in_stride);
#pragma unroll
        for (int d = 0; d < n_charge_dim; ++d) {
            complx<Real> v = mul_pow(acc[t][d], c[0], c2[0], -(t >> 2));
            v = mul_pow(v, c[1], c2[1], -((t >> 1) & 1));
            pw_in[d * n_pw_modes + m] = mul_pow(v, c[2], c2[2], -(t & 1));
        }
    }
}

// KERNEL_START

// Grid x runs over every level's groups back to back and grid y over mode windows. The window is
// the slower axis, so the blocks resident at any moment share a window, and the groups sharing a
// source slab read the same part of it close together in time, while it is still in L2.
extern "C" __global__ void PtShiftPwKernel(const ShiftPwArgs<Real> *args, int n_args) {
    int group = blockIdx.x;
    int arg_idx = 0;
    while (arg_idx < n_args && group >= args[arg_idx].n_groups_at_level)
        group -= args[arg_idx++].n_groups_at_level;
    if (arg_idx >= n_args)
        return;
    ShiftPwBody(args[arg_idx], group, blockIdx.y);
}
