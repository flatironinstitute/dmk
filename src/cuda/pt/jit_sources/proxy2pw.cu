// V2 proxy2pw (batched multilevel): projects upward proxy coefficients onto
// plane-wave modes. The launcher prepends `using Real` + N_ORDER / N_PW /
// N_PW2 / N_CHARGE_DIM / PROXY2PW_{Z,I,M1,M2}_TILE / BLOCK_SIZE. One launch
// covers all levels via a device array of per-level args (n_args == 1 for the
// windowed root). Writes plane-wave modes by assignment (idempotent).

#include <dmk/cuda/proxy2pw_kernelargs.hpp>

using dmk::cuda::Proxy2PwArgs;

template <typename Real>
struct alignas(2 * sizeof(Real)) p2pw_complex {
    Real r;
    Real i;
};

template <typename Real>
__device__ __forceinline__ p2pw_complex<Real> p2pw_zero() {
    return p2pw_complex<Real>{Real{0}, Real{0}};
}

template <typename Real>
__device__ __forceinline__ p2pw_complex<Real> p2pw_load(const Real *__restrict__ p, int idx) {
    return reinterpret_cast<const p2pw_complex<Real> *>(p)[idx];
}

template <typename Real>
__device__ __forceinline__ void p2pw_store(Real *__restrict__ p, int idx, p2pw_complex<Real> v) {
    reinterpret_cast<p2pw_complex<Real> *>(p)[idx] = v;
}

template <typename Real>
__device__ __forceinline__ void p2pw_madd_real(p2pw_complex<Real> &acc, Real a, p2pw_complex<Real> b) {
    acc.r = fma(a, b.r, acc.r);
    acc.i = fma(a, b.i, acc.i);
}

template <typename Real>
__device__ __forceinline__ void p2pw_madd(p2pw_complex<Real> &acc, p2pw_complex<Real> a, p2pw_complex<Real> b) {
    acc.r = fma(a.r, b.r, acc.r);
    acc.r = fma(-a.i, b.i, acc.r);
    acc.i = fma(a.r, b.i, acc.i);
    acc.i = fma(a.i, b.r, acc.i);
}

// A pencil is the m1 run at fixed (m2, m3). `pencil_slots` holds the slot m1 == 0 would land on,
// then the live m1 range as lo | hi<<16. A null table means the slab is in cube order.
__device__ __forceinline__ int pencil_slot(const int *__restrict__ pencil, int p, int m1, int cube_flat) {
    if (!pencil)
        return cube_flat;
    const int2 pv = reinterpret_cast<const int2 *>(pencil)[p];
    const int lo = pv.y & 0xffff;
    const int hi = pv.y >> 16;
    return (m1 >= lo && m1 < hi) ? pv.x + m1 : -1;
}

// KERNEL_START

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
    PtProxy2PwMultiLevelKernel(const Proxy2PwArgs<Real> *__restrict__ a_multilevel, int n_args, unsigned char *scratch,
                               long scratch_stride, int box_base) {
    using Complex = p2pw_complex<Real>;

    // Only the global-scratch path splits a level across launches, so the offset costs the
    // shared path nothing.
    const int box_idx = SMEM_GLOBAL ? blockIdx.x + box_base : blockIdx.x;
    const int arg_idx = blockIdx.y;

    if (arg_idx >= n_args)
        return;

    const Proxy2PwArgs<Real> a = a_multilevel[arg_idx];

    if (box_idx >= a.n_boxes_at_level)
        return;

    // The fused Stokeslet projector mixes the charge dims at one mode, so it needs all of them
    // live at the phase-3 store; that means a phase-2 output per dim. Every other mode keeps one.
    constexpr int D_TILE = (P2PW_MULTIPLY == 2) ? N_CHARGE_DIM : 1;
    static_assert(P2PW_MULTIPLY != 2 || N_CHARGE_DIM == 3, "the fused projector is the 3-vector Stokeslet");
    constexpr int FF2_DIM = PROXY2PW_Z_TILE * N_ORDER * N_PW;

    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    // At the largest expansions the working set exceeds the device's per-block shared limit, and
    // the block runs against a private slice of a global buffer instead. SMEM_GLOBAL is baked per
    // module, so only one of these survives compilation and the shared path keeps its LDS loads.
    unsigned char *__restrict__ shared_raw =
        SMEM_GLOBAL ? scratch + (long(blockIdx.y) * gridDim.x + blockIdx.x) * scratch_stride : &dynamic_smem[0];
    Complex *__restrict__ ff = reinterpret_cast<Complex *>(shared_raw);
    Complex *__restrict__ ff2 = ff + PROXY2PW_Z_TILE * N_ORDER * N_ORDER;
    Complex *__restrict__ poly2pw_s = ff2 + D_TILE * FF2_DIM;

    const int box = a.box_ids[box_idx];

    const long src_off = a.proxy_offsets[box];
    if (src_off < 0)
        return;
    const Real *proxy = a.proxy_flat + src_off;

    const long dst_off_complex = a.dst_offsets ? a.dst_offsets[box] : box_idx * a.dst_stride_complex;
    if (dst_off_complex < 0)
        return;
    Real *pw_dst = a.dst_flat + 2 * dst_off_complex;

    const int n_order = N_ORDER;
    const int n_order2 = n_order * n_order;
    const int n_order3 = n_order2 * n_order;
    const int n_pw = N_PW;
    const int n_pw2 = N_PW2;
    const int n_pw_modes = n_pw * n_pw * n_pw2;

    // Live modes sit at the front of the slab, which keeps its full-cube stride.
    const int *__restrict__ pencil = a.pencil_slots;

    for (int idx = threadIdx.x; idx < n_order * n_pw; idx += blockDim.x)
        poly2pw_s[idx] = p2pw_load(a.poly2pw, idx);

    __syncthreads();

    for (int m3_base = 0; m3_base < n_pw2; m3_base += PROXY2PW_Z_TILE) {
        const int z_count = (m3_base + PROXY2PW_Z_TILE <= n_pw2) ? PROXY2PW_Z_TILE : (n_pw2 - m3_base);

        for (int d_base = 0; d_base < N_CHARGE_DIM; d_base += D_TILE) {
            for (int dd = 0; dd < D_TILE; ++dd) {
                const Real *proxy_d = proxy + (d_base + dd) * n_order3;

                // Phase 1: ff(zr, i, j) = sum_k proxy(i, j, k, d) * poly2pw(m3_base + zr, k).
                for (int ij = threadIdx.x; ij < n_order2; ij += blockDim.x) {
                    Complex acc[PROXY2PW_Z_TILE];
#pragma unroll
                    for (int zr = 0; zr < PROXY2PW_Z_TILE; ++zr)
                        acc[zr] = p2pw_zero<Real>();

                    for (int k = 0; k < n_order; ++k) {
                        const Real p = proxy_d[ij + k * n_order2];
#pragma unroll
                        for (int zr = 0; zr < PROXY2PW_Z_TILE; ++zr) {
                            if (zr < z_count) {
                                const int m3 = m3_base + zr;
                                const Complex b = poly2pw_s[k * n_pw + m3];
                                p2pw_madd_real(acc[zr], p, b);
                            }
                        }
                    }

#pragma unroll
                    for (int zr = 0; zr < PROXY2PW_Z_TILE; ++zr) {
                        if (zr < z_count)
                            ff[zr * n_order2 + ij] = acc[zr];
                    }
                }

                __syncthreads();

                // Phase 2: ff2(dd, zr, i, m2) = sum_j ff(zr, i, j) * poly2pw(m2, j).
                constexpr int I_TILE = PROXY2PW_I_TILE;
                constexpr int M2_TILE = PROXY2PW_M2_TILE;
                const int i_tiles = (n_order + I_TILE - 1) / I_TILE;
                const int m2_tiles = (n_pw + M2_TILE - 1) / M2_TILE;
                const int phase2_tiles = z_count * i_tiles * m2_tiles;
                Complex *__restrict__ ff2_d = ff2 + dd * FF2_DIM;

                for (int tile = threadIdx.x; tile < phase2_tiles; tile += blockDim.x) {
                    int x = tile;
                    const int m2_tile = x % m2_tiles;
                    x /= m2_tiles;
                    const int i_tile = x % i_tiles;
                    const int zr = x / i_tiles;
                    const int i_base = i_tile * I_TILE;
                    const int m2_base = m2_tile * M2_TILE;

                    Complex acc[I_TILE][M2_TILE];
#pragma unroll
                    for (int ii = 0; ii < I_TILE; ++ii) {
#pragma unroll
                        for (int r = 0; r < M2_TILE; ++r)
                            acc[ii][r] = p2pw_zero<Real>();
                    }

#pragma unroll
                    for (int j = 0; j < N_ORDER; ++j) {
                        Complex b[M2_TILE];
#pragma unroll
                        for (int r = 0; r < M2_TILE; ++r) {
                            const int m2 = m2_base + r;
                            b[r] = (m2 < n_pw) ? poly2pw_s[j * n_pw + m2] : p2pw_zero<Real>();
                        }
#pragma unroll
                        for (int ii = 0; ii < I_TILE; ++ii) {
                            const int i = i_base + ii;
                            if (i < n_order) {
                                const Complex f = ff[zr * n_order2 + i + j * n_order];
#pragma unroll
                                for (int r = 0; r < M2_TILE; ++r)
                                    p2pw_madd(acc[ii][r], f, b[r]);
                            }
                        }
                    }

#pragma unroll
                    for (int ii = 0; ii < I_TILE; ++ii) {
                        const int i = i_base + ii;
                        if (i < n_order) {
#pragma unroll
                            for (int r = 0; r < M2_TILE; ++r) {
                                const int m2 = m2_base + r;
                                if (m2 < n_pw)
                                    ff2_d[zr * n_order * n_pw + i + m2 * n_order] = acc[ii][r];
                            }
                        }
                    }
                }

                __syncthreads();
            }

            // Phase 3: pw(m1, m2, m3, d) = sum_i ff2(dd, zr, i, m2) * poly2pw(m1, i), then the
            // kernel-FT multiply in registers (P2PW_MULTIPLY != 0) instead of a second pass.
            constexpr int M1_TILE = PROXY2PW_M1_TILE;
            constexpr int M2_OUT_TILE = PROXY2PW_M2_TILE;
            const int m1_tiles = (n_pw + M1_TILE - 1) / M1_TILE;
            const int phase3_m2_tiles = (n_pw + M2_OUT_TILE - 1) / M2_OUT_TILE;
            const int total_tiles = z_count * phase3_m2_tiles * m1_tiles;

            for (int tile = threadIdx.x; tile < total_tiles; tile += blockDim.x) {
                int x = tile;
                const int m1_tile = x % m1_tiles;
                x /= m1_tiles;
                const int m2_tile = x % phase3_m2_tiles;
                const int zr = x / phase3_m2_tiles;
                const int m3 = m3_base + zr;
                const int m1_base = m1_tile * M1_TILE;
                const int m2_base = m2_tile * M2_OUT_TILE;

                Complex acc[D_TILE][M2_OUT_TILE][M1_TILE];
#pragma unroll
                for (int dd = 0; dd < D_TILE; ++dd) {
#pragma unroll
                    for (int c = 0; c < M2_OUT_TILE; ++c) {
#pragma unroll
                        for (int r = 0; r < M1_TILE; ++r)
                            acc[dd][c][r] = p2pw_zero<Real>();
                    }
                }

#pragma unroll
                for (int i = 0; i < N_ORDER; ++i) {
                    Complex b[M1_TILE];
#pragma unroll
                    for (int r = 0; r < M1_TILE; ++r) {
                        const int m1 = m1_base + r;
                        b[r] = (m1 < n_pw) ? poly2pw_s[i * n_pw + m1] : p2pw_zero<Real>();
                    }
#pragma unroll
                    for (int c = 0; c < M2_OUT_TILE; ++c) {
                        const int m2 = m2_base + c;
                        if (m2 < n_pw) {
#pragma unroll
                            for (int dd = 0; dd < D_TILE; ++dd) {
                                const Complex f = ff2[dd * FF2_DIM + zr * n_order * n_pw + i + m2 * n_order];
#pragma unroll
                                for (int r = 0; r < M1_TILE; ++r)
                                    p2pw_madd(acc[dd][c][r], f, b[r]);
                            }
                        }
                    }
                }

#pragma unroll
                for (int c = 0; c < M2_OUT_TILE; ++c) {
                    const int m2 = m2_base + c;
                    if (m2 < n_pw) {
#pragma unroll
                        for (int r = 0; r < M1_TILE; ++r) {
                            const int m1 = m1_base + r;
                            if (m1 >= n_pw)
                                continue;
                            const int slot =
                                pencil_slot(pencil, m2 + m3 * n_pw, m1, m1 + m2 * n_pw + m3 * n_pw * n_pw);
                            if (slot < 0)
                                continue;

                            if constexpr (P2PW_MULTIPLY == 0) {
#pragma unroll
                                for (int dd = 0; dd < D_TILE; ++dd)
                                    p2pw_store(pw_dst + 2 * (d_base + dd) * n_pw_modes, slot, acc[dd][c][r]);
                            } else if constexpr (P2PW_MULTIPLY == 1) {
                                const Real fk = a.radialft[slot];
#pragma unroll
                                for (int dd = 0; dd < D_TILE; ++dd) {
                                    Complex v = acc[dd][c][r];
                                    v.r *= fk;
                                    v.i *= fk;
                                    p2pw_store(pw_dst + 2 * (d_base + dd) * n_pw_modes, slot, v);
                                }
                            } else {
                                const int npw_half = n_pw / 2;
                                const Real kx = Real(m1 - npw_half) * a.hpw;
                                const Real ky = Real(m2 - npw_half) * a.hpw;
                                const Real kz = Real(m3 - npw_half) * a.hpw;
                                const Real fk = a.radialft[slot];
                                const Real ksq = (kx * kx + ky * ky + kz * kz) * fk;

                                const Complex p0 = acc[0][c][r];
                                const Complex p1 = acc[1][c][r];
                                const Complex p2 = acc[2][c][r];
                                const Real dr = p0.r * kx + p1.r * ky + p2.r * kz;
                                const Real di = p0.i * kx + p1.i * ky + p2.i * kz;

                                Complex o0{dr * (kx * fk) - p0.r * ksq, di * (kx * fk) - p0.i * ksq};
                                Complex o1{dr * (ky * fk) - p1.r * ksq, di * (ky * fk) - p1.i * ksq};
                                Complex o2{dr * (kz * fk) - p2.r * ksq, di * (kz * fk) - p2.i * ksq};

                                // The windowed root keeps a share of the untransformed zero mode.
                                if (a.is_windowed && m1 == npw_half && m2 == npw_half && m3 == npw_half) {
                                    const Real cval = Real(1) / (Real(1.7320508075688772935) + Real(1));
                                    o0.r += cval * p0.r;
                                    o0.i += cval * p0.i;
                                    o1.r += cval * p1.r;
                                    o1.i += cval * p1.i;
                                    o2.r += cval * p2.r;
                                    o2.i += cval * p2.i;
                                }

                                p2pw_store(pw_dst, slot, o0);
                                p2pw_store(pw_dst + 2 * n_pw_modes, slot, o1);
                                p2pw_store(pw_dst + 2 * 2 * n_pw_modes, slot, o2);
                            }
                        }
                    }
                }
            }

            __syncthreads();
        }
    }
}
