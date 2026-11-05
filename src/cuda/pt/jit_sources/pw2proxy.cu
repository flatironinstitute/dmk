// V2 pw2proxy (batched multilevel): projects plane-wave modes back onto proxy
// coefficients (additive). The launcher prepends `using Real` + N_ORDER / N_PW
// / N_PW2 / N_CHARGE_DIM / K2R / M2R / BLOCK_SIZE / MIN_BLOCKS / ACC_REG /
// PENCIL_SMEM / SMEM_GLOBAL. One launch covers all levels via a device array of
// per-level args (n_args == 1 for the windowed root). Accumulates into proxy_flat.
//
//   proxy(k1, k2, k3) = 2 Re sum_m3 halve(m3) A(k3, m3) sum_m2 A(k2, m2) sum_m1 A(k1, m1) pw(m1, m2, m3)
//
// A is the level's pw2poly matrix. The field is real, so only half of the m3 axis is stored: 2 Re
// supplies the conjugate half and halve() keeps the self-conjugate plane from counting twice. m3 is
// contracted last, so everything before it stays complex and the final step is real.
//
// The contraction is sum factorization, scheduled after Swirydowicz, Chalmers, Karakus & Warburton,
// "Acceleration of tensor-product operations for high-order finite element methods" (IJHPCA 2019).
// One block per box streams the modes one m3 slab at a time, so pw_in is read exactly once:
//   a: H(k1, m2) = sum_m1 A(k1, m1) pw(m1, m2, m3), over each pencil's live run only
//   b: J(k1, k2) = sum_m2 A(k2, m2) H(k1, m2)
//   c: acc(k1, k2, k3) += Re(halve(m3) A(k3, m3) J(k1, k2))
// A thread owns (k1, K2R consecutive k2) through b and c and keeps their k3 columns in registers.
//
// Modes past the kernel's band edge are zero and are compacted out of pw_in. Each (m2, m3) pencil's
// live m1 form one contiguous run, so step a skips the dead modes exactly rather than masking them.

#include <dmk/cuda/pw2proxy_kernelargs.hpp>

using dmk::cuda::PwToProxyArgs;

template <typename Real>
struct alignas(2 * sizeof(Real)) complx {
    Real r;
    Real i;
};

template <typename Real>
__device__ __forceinline__ complx<Real> complx_zero() {
    return complx<Real>{Real{0}, Real{0}};
}

template <typename Real>
__device__ __forceinline__ complx<Real> complx_load(const Real *__restrict__ p, int idx) {
    return reinterpret_cast<const complx<Real> *>(p)[idx];
}

template <typename Real>
__device__ __forceinline__ void complx_madd(complx<Real> &acc, const complx<Real> a, const complx<Real> b) {
    acc.r = fma(a.r, b.r, acc.r);
    acc.r = fma(-a.i, b.i, acc.r);
    acc.i = fma(a.r, b.i, acc.i);
    acc.i = fma(a.i, b.r, acc.i);
}

template <typename Real>
__device__ __forceinline__ Real complx_real_madd(Real acc, const complx<Real> a, const complx<Real> b) {
    acc = fma(a.r, b.r, acc);
    acc = fma(-a.i, b.i, acc);
    return acc;
}

// KERNEL_START

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
    PtPwToProxyMultiLevelKernel(const PwToProxyArgs<Real> *__restrict__ args, int n_args, unsigned char *scratch,
                                long scratch_stride, int box_base) {
    // Only the global-scratch path splits a level across launches, so the offset costs the
    // shared path nothing.
    const int box_idx = SMEM_GLOBAL ? blockIdx.x + box_base : blockIdx.x;
    const int arg_idx = blockIdx.y;

    if (arg_idx >= n_args)
        return;

    const PwToProxyArgs<Real> a = args[arg_idx];

    if (box_idx >= a.n_boxes_at_level)
        return;

    const int box = a.box_ids[box_idx];

    const long proxy_off = a.proxy_offsets[box];
    if (proxy_off < 0)
        return;

    const bool assign = a.assign != 0;

    constexpr int n_pw = N_PW;
    constexpr int n_pw2 = N_PW2;
    constexpr int n_order = N_ORDER;
    constexpr int n_order2 = n_order * n_order;
    constexpr int n_order3 = n_order2 * n_order;
    constexpr int n_pw_modes = n_pw * n_pw * n_pw2;
    constexpr int n_pencil = n_pw * n_pw2;
    constexpr int k2_tiles = (n_order + K2R - 1) / K2R;
    // A^T rows are padded so a k2 tile never reads past its row; the padding is zero.
    constexpr int k_pad = ((k2_tiles * K2R + 3) / 4) * 4;
    constexpr int n_items = n_order * k2_tiles;
    constexpr int ROUNDS = (n_items + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int SLAB_REG = (n_pw * n_pw + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Live modes sit at the front of the slab, which keeps its full-cube stride. Pencils are
    // packed in cube order, so each m3 slab of live modes is one contiguous run.
    const int *__restrict__ pencil = a.pencil_slots;

    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    // At the largest expansions the working set exceeds the device's per-block shared limit, and
    // the block runs against a private slice of a global buffer instead. SMEM_GLOBAL is baked per
    // module, so only one of these survives compilation and the shared path keeps its LDS loads.
    unsigned char *__restrict__ shared_raw =
        SMEM_GLOBAL ? scratch + (long(blockIdx.y) * gridDim.x + blockIdx.x) * scratch_stride : &dynamic_smem[0];

    // A null table means cube order; writing that case out as an identity run keeps one code path.
    const auto decode_pencil = [&](int p) -> int4 {
        int4 v;
        if (pencil) {
            const int2 pv = reinterpret_cast<const int2 *>(pencil)[p];
            v.x = pv.x;
            v.y = pv.y & 0xffff;
            v.z = pv.y >> 16;
        } else {
            v.x = (p % n_pw) * n_pw + (p / n_pw) * n_pw * n_pw;
            v.y = 0;
            v.z = n_pw;
        }
        v.w = 0;
        return v;
    };

    int4 *__restrict__ s_pencil = reinterpret_cast<int4 *>(shared_raw);
    const auto pencil_at = [&](int p) -> int4 {
        if constexpr (PENCIL_SMEM)
            return s_pencil[p];
        else
            return decode_pencil(p);
    };

    // Per slab: its first and one-past-last slot, then the m2 range holding live pencils.
    int4 *__restrict__ s_range = s_pencil + (PENCIL_SMEM ? n_pencil : 0);
    complx<Real> *__restrict__ s_A_T = reinterpret_cast<complx<Real> *>(s_range + n_pw2);
    complx<Real> *__restrict__ s_H = s_A_T + n_pw * k_pad;
    complx<Real> *__restrict__ s_slab = s_H + n_pw * n_order;

    for (int idx = threadIdx.x; idx < n_pw * k_pad; idx += BLOCK_SIZE) {
        const int m = idx / k_pad;
        const int k = idx - m * k_pad;
        s_A_T[idx] = k < n_order ? complx_load(a.pw2poly, k * n_pw + m) : complx_zero<Real>();
    }

    if constexpr (PENCIL_SMEM) {
        for (int p = threadIdx.x; p < n_pencil; p += BLOCK_SIZE)
            s_pencil[p] = decode_pencil(p);
        __syncthreads();
    }

    for (int m3 = threadIdx.x; m3 < n_pw2; m3 += BLOCK_SIZE) {
        int lo = n_pw, hi = n_pw;
        for (int m2 = 0; m2 < n_pw; ++m2) {
            const int4 pv = pencil_at(m2 + m3 * n_pw);
            if (pv.y < pv.z) {
                lo = min(lo, m2);
                hi = m2 + 1;
            }
        }
        const int4 first = pencil_at(m3 * n_pw);
        const int4 last = pencil_at(m3 * n_pw + n_pw - 1);
        s_range[m3] = int4{first.x + first.y, last.x + last.z, lo, max(lo, hi)};
    }

    __syncthreads();

    const Real *__restrict__ pw_in_box = a.pw_in_pool + box_idx * a.pw_in_stride;
    Real *__restrict__ proxy_box = a.proxy_flat + proxy_off;

    // The next slab is fetched into registers while the current one is reduced, and lands in
    // shared once step a has let go of the buffer. The global-scratch path reads modes in place.
    complx<Real> pre[SLAB_REG];
    const auto fetch = [&](int step) {
        const int d = step / n_pw2;
        const int4 r = s_range[step - d * n_pw2];
        const complx<Real> *__restrict__ src =
            reinterpret_cast<const complx<Real> *>(pw_in_box + 2 * d * n_pw_modes) + r.x;
#pragma unroll
        for (int j = 0; j < SLAB_REG; ++j) {
            const int i = threadIdx.x + j * BLOCK_SIZE;
            pre[j] = i < r.y - r.x ? src[i] : complx_zero<Real>();
        }
    };
    const auto stage = [&](int step) {
        const int4 r = s_range[step % n_pw2];
#pragma unroll
        for (int j = 0; j < SLAB_REG; ++j) {
            const int i = threadIdx.x + j * BLOCK_SIZE;
            if (i < r.y - r.x)
                s_slab[i] = pre[j];
        }
    };

    const int n_steps = a.n_charge_dim * n_pw2;
    if constexpr (!SMEM_GLOBAL) {
        fetch(0);
        stage(0);
        __syncthreads();
    }

    Real acc[ROUNDS][K2R][n_order];

    for (int step = 0; step < n_steps; ++step) {
        const int d = step / n_pw2;
        const int m3 = step - d * n_pw2;
        const int4 r = s_range[m3];
        const int m2_lo = r.z, m2_hi = r.w;
        Real *__restrict__ proxy_d = proxy_box + d * n_order3;

        if constexpr (ACC_REG) {
            if (m3 == 0) {
#pragma unroll
                for (int rd = 0; rd < ROUNDS; ++rd)
#pragma unroll
                    for (int k2r = 0; k2r < K2R; ++k2r)
#pragma unroll
                        for (int k3 = 0; k3 < n_order; ++k3)
                            acc[rd][k2r][k3] = Real{0};
            }
        }

        // Step a. A pencil's slots sit at a fixed offset from its m1, so `base + m1` is the slot
        // relative to the slab start. Each thread takes M2R adjacent pencils, which share every
        // A load over the union of their runs.
        const complx<Real> *__restrict__ slab =
            SMEM_GLOBAL ? reinterpret_cast<const complx<Real> *>(pw_in_box + 2 * d * n_pw_modes) + r.x : s_slab;
        const int n_groups = (m2_hi - m2_lo + M2R - 1) / M2R;
        for (int it = threadIdx.x; it < n_order * n_groups; it += BLOCK_SIZE) {
            const int k1 = it % n_order;
            const int m2_base = m2_lo + (it / n_order) * M2R;

            int base[M2R], lo[M2R], hi[M2R];
            int u_lo = n_pw, u_hi = 0;
#pragma unroll
            for (int q = 0; q < M2R; ++q) {
                lo[q] = hi[q] = base[q] = 0;
                if (m2_base + q < m2_hi) {
                    const int4 pv = pencil_at(m2_base + q + m3 * n_pw);
                    base[q] = pv.x - r.x;
                    lo[q] = pv.y;
                    hi[q] = pv.z;
                    if (pv.y < pv.z) {
                        u_lo = min(u_lo, pv.y);
                        u_hi = max(u_hi, pv.z);
                    }
                }
            }

            complx<Real> h[M2R];
#pragma unroll
            for (int q = 0; q < M2R; ++q)
                h[q] = complx_zero<Real>();

#pragma unroll 4
            for (int m1 = u_lo; m1 < u_hi; ++m1) {
                const complx<Real> am = s_A_T[m1 * k_pad + k1];
#pragma unroll
                for (int q = 0; q < M2R; ++q)
                    if (M2R == 1 || unsigned(m1 - lo[q]) < unsigned(hi[q] - lo[q]))
                        complx_madd(h[q], am, slab[base[q] + m1]);
            }

#pragma unroll
            for (int q = 0; q < M2R; ++q)
                if (m2_base + q < m2_hi)
                    s_H[(m2_base + q) * n_order + k1] = h[q];
        }

        __syncthreads();

        if constexpr (!SMEM_GLOBAL) {
            if (step + 1 < n_steps)
                fetch(step + 1);
        }

        // Steps b and c. Every round's J is formed first so one pass over k3 serves them all and
        // each A(k3, m3) is loaded once. A round past the last item computes on item 0 and is
        // never stored.
        int k1s[ROUNDS], k2b[ROUNDS];
        bool live[ROUNDS];
        complx<Real> J[ROUNDS][K2R];
#pragma unroll
        for (int rd = 0; rd < ROUNDS; ++rd) {
            const int it = threadIdx.x + rd * BLOCK_SIZE;
            live[rd] = it < n_items;
            k1s[rd] = live[rd] ? it % n_order : 0;
            k2b[rd] = live[rd] ? (it / n_order) * K2R : 0;
#pragma unroll
            for (int k2r = 0; k2r < K2R; ++k2r)
                J[rd][k2r] = complx_zero<Real>();
        }

        for (int m2 = m2_lo; m2 < m2_hi; ++m2) {
#pragma unroll
            for (int rd = 0; rd < ROUNDS; ++rd) {
                if (rd > 0 && !live[rd])
                    break;
                const complx<Real> hv = s_H[m2 * n_order + k1s[rd]];
#pragma unroll
                for (int k2r = 0; k2r < K2R; ++k2r)
                    complx_madd(J[rd][k2r], s_A_T[m2 * k_pad + k2b[rd] + k2r], hv);
            }
        }

        // The self-conjugate plane is halved, as the 2 Re below counts it twice.
        const Real scale = (m3 >= n_pw / 2) ? Real{0.5} : Real{1};
#pragma unroll
        for (int rd = 0; rd < ROUNDS; ++rd)
#pragma unroll
            for (int k2r = 0; k2r < K2R; ++k2r) {
                J[rd][k2r].r *= scale;
                J[rd][k2r].i *= scale;
            }

        if constexpr (ACC_REG) {
#pragma unroll
            for (int k3 = 0; k3 < n_order; ++k3) {
                const complx<Real> a3 = s_A_T[m3 * k_pad + k3];
#pragma unroll
                for (int rd = 0; rd < ROUNDS; ++rd)
#pragma unroll
                    for (int k2r = 0; k2r < K2R; ++k2r)
                        acc[rd][k2r][k3] = complx_real_madd(acc[rd][k2r][k3], a3, J[rd][k2r]);
            }

            if (m3 == n_pw2 - 1) {
#pragma unroll
                for (int rd = 0; rd < ROUNDS; ++rd) {
#pragma unroll
                    for (int k2r = 0; k2r < K2R; ++k2r) {
                        if (!live[rd] || k2b[rd] + k2r >= n_order)
                            continue;
#pragma unroll
                        for (int k3 = 0; k3 < n_order; ++k3) {
                            Real *out = proxy_d + k1s[rd] + (k2b[rd] + k2r) * n_order + k3 * n_order2;
                            if (assign)
                                *out = Real{2} * acc[rd][k2r][k3];
                            else
                                *out += Real{2} * acc[rd][k2r][k3];
                        }
                    }
                }
            }
        } else {
            for (int k3 = 0; k3 < n_order; ++k3) {
                const complx<Real> a3 = s_A_T[m3 * k_pad + k3];
#pragma unroll
                for (int rd = 0; rd < ROUNDS; ++rd) {
#pragma unroll
                    for (int k2r = 0; k2r < K2R; ++k2r) {
                        if (!live[rd] || k2b[rd] + k2r >= n_order)
                            continue;
                        Real *out = proxy_d + k1s[rd] + (k2b[rd] + k2r) * n_order + k3 * n_order2;
                        const Real prev = (assign && m3 == 0) ? Real{0} : *out;
                        *out = prev + Real{2} * complx_real_madd(Real{0}, a3, J[rd][k2r]);
                    }
                }
            }
        }

        if constexpr (!SMEM_GLOBAL) {
            if (step + 1 < n_steps)
                stage(step + 1);
        }

        __syncthreads();
    }
}
