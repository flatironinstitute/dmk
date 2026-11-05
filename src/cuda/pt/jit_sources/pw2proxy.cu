// V2 pw2proxy (batched multilevel): projects plane-wave modes back onto proxy
// coefficients (additive). The launcher prepends `using Real` + N_ORDER / N_PW
// / N_PW2 / N_CHARGE_DIM / K2R / BLOCK_SIZE / MIN_BLOCKS / ACC_REG /
// PENCIL_SMEM / SMEM_GLOBAL. One launch covers all levels via a device array of
// per-level args (n_args == 1 for the windowed root). Accumulates into proxy_flat.
//
//   proxy(k1, k2, k3) = 2 Re sum_m3 halve(m3) A(k3, m3) sum_m2 A(k2, m2) sum_m1 A(k1, m1) pw(m1, m2, m3)
//
// A is the level's pw2poly matrix. The field is real, so only half of the m3 axis is stored: 2 Re
// supplies the conjugate half and halve() keeps the self-conjugate plane from counting twice. m3 is
// contracted last, so everything before it stays complex and the final step is real.
//
// A takes modes to Chebyshev coefficients on nodes symmetric about 0, so A(k, m) = alpha(k, m) is real
// and even in m for even k, and A(k, m) = i alpha(k, m) with alpha odd in m for odd k. Pairing each
// mode m < H = N_PW/2 with its mirror N_PW-1-m, even k see only the pair's sum and odd k only its
// difference, against a real table over m <= H.
//
// The contraction is sum factorization, scheduled after Swirydowicz, Chalmers, Karakus & Warburton,
// "Acceleration of tensor-product operations for high-order finite element methods" (IJHPCA 2019).
// One block per box streams the modes one m3 slab at a time, so pw_in is read exactly once:
//   fold: each pencil's m1 and mirror become their sum and difference, in place
//   a: H(k1, m2) = sum_m1 A(k1, m1) pw(m1, m2, m3), for a pencil and its mirror together, kept as
//      their sum and difference in the same way
//   b: J(k1, k2) = sum_m2 A(k2, m2) H(k1, m2)
//   c: acc(k1, k2, k3) += Re(halve(m3) A(k3, m3) J(k1, k2))
// A thread owns (k1, K2R k2 of one parity) through b and c and keeps their k3 columns in registers.
//
// Modes past the kernel's band edge are zero and are compacted out of pw_in. Each (m2, m3) pencil's
// live m1 form one contiguous run symmetric about H, so the fold and step a skip the dead modes
// exactly rather than masking them.

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
__device__ __forceinline__ void complx_madd_real(complx<Real> &acc, Real a, const complx<Real> b) {
    acc.r = fma(a, b.r, acc.r);
    acc.i = fma(a, b.i, acc.i);
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
    constexpr int H = N_PW / 2;
    static_assert(N_PW % 2 == 1 && N_PW2 == H + 1, "modes pair as m and N_PW-1-m around the zero mode H");
    constexpr int n_order = N_ORDER;
    constexpr int n_order2 = n_order * n_order;
    constexpr int n_order3 = n_order2 * n_order;
    constexpr int n_pw_modes = n_pw * n_pw * n_pw2;
    constexpr int n_pencil = n_pw * n_pw2;
    // A k2 tile is K2R coefficients of one parity. The table holds the even k, then the odd k, each
    // padded to whole tiles so a tile never reads into the next group; the padding is zero.
    constexpr int TE = ((n_order + 1) / 2 + K2R - 1) / K2R;
    constexpr int TO = (n_order / 2 + K2R - 1) / K2R;
    constexpr int ODD0 = TE * K2R;
    constexpr int k_pad = ((ODD0 + TO * K2R + 3) / 4) * 4;
    constexpr int n_items = n_order * (TE + TO);
    constexpr int ROUNDS = (n_items + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int SLAB_REG = (n_pw * n_pw + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const auto kcol = [](int k) { return (k & 1) ? ODD0 + k / 2 : k / 2; };

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
    Real *__restrict__ s_alpha = reinterpret_cast<Real *>(s_range + n_pw2);
    complx<Real> *__restrict__ s_H = reinterpret_cast<complx<Real> *>(s_alpha + n_pw2 * k_pad);
    complx<Real> *__restrict__ s_slab = s_H + n_pw * n_order;

    // alpha(m, k) at row m <= H, column kcol(k): Re A for even k, Im A for odd k, which vanishes at m = H.
    for (int idx = threadIdx.x; idx < n_pw2 * k_pad; idx += BLOCK_SIZE) {
        const int m = idx / k_pad;
        const int c = idx - m * k_pad;
        const int k = c < ODD0 ? 2 * c : 2 * (c - ODD0) + 1;
        Real v = Real{0};
        if (k < n_order && !((k & 1) && m == H)) {
            const complx<Real> z = complx_load(a.pw2poly, k * n_pw + m);
            v = (k & 1) ? z.i : z.r;
        }
        s_alpha[idx] = v;
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
    // shared once step a has let go of the buffer. The global-scratch path copies each slab when
    // it starts instead.
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
        // Pencil m2 = p pairs with n_pw-1-p, for p from the low end of the live range up to H.
        const int p_lo = min(m2_lo, n_pw - m2_hi);
        const int n_pairs = m2_lo < m2_hi ? H + 1 - p_lo : 0;
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

        if constexpr (SMEM_GLOBAL) {
            const complx<Real> *__restrict__ src =
                reinterpret_cast<const complx<Real> *>(pw_in_box + 2 * d * n_pw_modes) + r.x;
            for (int i = threadIdx.x; i < r.y - r.x; i += BLOCK_SIZE)
                s_slab[i] = src[i];
            __syncthreads();
        }

        // Fold. A pencil's slots sit at a fixed offset from its m1, so `pen + m1` is the slot; m1 < H
        // takes the sum and its mirror the difference.
        for (int it = threadIdx.x; it < (m2_hi - m2_lo) * H; it += BLOCK_SIZE) {
            const int q = it / H;
            const int m1 = it - q * H;
            const int4 pv = pencil_at(m2_lo + q + m3 * n_pw);
            if (m1 >= pv.y && pv.y < pv.z) {
                complx<Real> *__restrict__ pen = s_slab + (pv.x - r.x);
                const complx<Real> u = pen[m1];
                const complx<Real> v = pen[n_pw - 1 - m1];
                pen[m1] = complx<Real>{u.r + v.r, u.i + v.i};
                pen[n_pw - 1 - m1] = complx<Real>{u.r - v.r, u.i - v.i};
            }
        }

        __syncthreads();

        // Step a. Even k1 read the sums and odd k1 the differences, with A's factor i applied at the end.
        for (int it = threadIdx.x; it < n_order * n_pairs; it += BLOCK_SIZE) {
            const int k1 = it % n_order;
            const int p = p_lo + it / n_order;
            const int pm = n_pw - 1 - p;
            const bool odd = k1 & 1;
            const Real *__restrict__ al = s_alpha + kcol(k1);
            const int4 v0 = pencil_at(p + m3 * n_pw);
            const int4 v1 = pencil_at(pm + m3 * n_pw);
            const complx<Real> *__restrict__ pen0 = s_slab + (v0.x - r.x);
            const complx<Real> *__restrict__ pen1 = s_slab + (v1.x - r.x);
            const int lo0 = v0.y < v0.z ? v0.y : n_pw;
            const int lo1 = (p < H && v1.y < v1.z) ? v1.y : n_pw;

            complx<Real> h0 = complx_zero<Real>(), h1 = complx_zero<Real>();
#pragma unroll 4
            for (int m1 = min(lo0, lo1); m1 <= H; ++m1) {
                const Real am = al[m1 * k_pad];
                const int s = odd ? n_pw - 1 - m1 : m1;
                if (m1 >= lo0)
                    complx_madd_real(h0, am, pen0[s]);
                if (m1 >= lo1)
                    complx_madd_real(h1, am, pen1[s]);
            }
            if (odd) {
                h0 = complx<Real>{-h0.i, h0.r};
                h1 = complx<Real>{-h1.i, h1.r};
            }

            if (p < H) {
                s_H[p * n_order + k1] = complx<Real>{h0.r + h1.r, h0.i + h1.i};
                s_H[pm * n_order + k1] = complx<Real>{h0.r - h1.r, h0.i - h1.i};
            } else {
                s_H[p * n_order + k1] = h0;
            }
        }

        __syncthreads();

        if constexpr (!SMEM_GLOBAL) {
            if (step + 1 < n_steps)
                fetch(step + 1);
        }

        // Steps b and c. Every round's J is formed first so one pass over k3 serves them all and
        // each alpha(m3, k3) is loaded once. A round past the last item computes on item 0 and is
        // never stored.
        int k1s[ROUNDS], j0[ROUNDS], col[ROUNDS];
        bool live[ROUNDS], oddk2[ROUNDS];
        complx<Real> J[ROUNDS][K2R];
#pragma unroll
        for (int rd = 0; rd < ROUNDS; ++rd) {
            const int it = threadIdx.x + rd * BLOCK_SIZE;
            live[rd] = it < n_items;
            const int t = live[rd] ? it / n_order : 0;
            k1s[rd] = live[rd] ? it % n_order : 0;
            oddk2[rd] = t >= TE;
            j0[rd] = (oddk2[rd] ? t - TE : t) * K2R;
            col[rd] = (oddk2[rd] ? ODD0 : 0) + j0[rd];
#pragma unroll
            for (int k2r = 0; k2r < K2R; ++k2r)
                J[rd][k2r] = complx_zero<Real>();
        }

        for (int p = p_lo; p < p_lo + n_pairs; ++p) {
            const Real *__restrict__ arow = s_alpha + p * k_pad;
#pragma unroll
            for (int rd = 0; rd < ROUNDS; ++rd) {
                if (rd > 0 && !live[rd])
                    break;
                const complx<Real> hv = s_H[(oddk2[rd] ? n_pw - 1 - p : p) * n_order + k1s[rd]];
#pragma unroll
                for (int k2r = 0; k2r < K2R; ++k2r)
                    complx_madd_real(J[rd][k2r], arow[col[rd] + k2r], hv);
            }
        }

        // A's factor i for odd k2, and the self-conjugate plane is halved, as the 2 Re below counts it twice.
        const Real scale = (m3 == H) ? Real{0.5} : Real{1};
#pragma unroll
        for (int rd = 0; rd < ROUNDS; ++rd)
#pragma unroll
            for (int k2r = 0; k2r < K2R; ++k2r) {
                const complx<Real> v = J[rd][k2r];
                J[rd][k2r] = oddk2[rd] ? complx<Real>{-scale * v.i, scale * v.r} : complx<Real>{scale * v.r, scale * v.i};
            }

        // Re(A(k3, m3) J) is alpha J.r for even k3 and -alpha J.i for odd k3.
        if constexpr (ACC_REG) {
#pragma unroll
            for (int k3 = 0; k3 < n_order; ++k3) {
                const Real a3 = s_alpha[m3 * k_pad + kcol(k3)];
#pragma unroll
                for (int rd = 0; rd < ROUNDS; ++rd)
#pragma unroll
                    for (int k2r = 0; k2r < K2R; ++k2r)
                        acc[rd][k2r][k3] = fma(a3, (k3 & 1) ? -J[rd][k2r].i : J[rd][k2r].r, acc[rd][k2r][k3]);
            }

            if (m3 == n_pw2 - 1) {
#pragma unroll
                for (int rd = 0; rd < ROUNDS; ++rd) {
#pragma unroll
                    for (int k2r = 0; k2r < K2R; ++k2r) {
                        const int k2 = 2 * (j0[rd] + k2r) + oddk2[rd];
                        if (!live[rd] || k2 >= n_order)
                            continue;
#pragma unroll
                        for (int k3 = 0; k3 < n_order; ++k3) {
                            Real *out = proxy_d + k1s[rd] + k2 * n_order + k3 * n_order2;
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
                const Real a3 = s_alpha[m3 * k_pad + kcol(k3)];
#pragma unroll
                for (int rd = 0; rd < ROUNDS; ++rd) {
#pragma unroll
                    for (int k2r = 0; k2r < K2R; ++k2r) {
                        const int k2 = 2 * (j0[rd] + k2r) + oddk2[rd];
                        if (!live[rd] || k2 >= n_order)
                            continue;
                        Real *out = proxy_d + k1s[rd] + k2 * n_order + k3 * n_order2;
                        const Real prev = (assign && m3 == 0) ? Real{0} : *out;
                        *out = prev + Real{2} * a3 * ((k3 & 1) ? -J[rd][k2r].i : J[rd][k2r].r);
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
