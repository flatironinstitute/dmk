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
__device__ __forceinline__ void p2pw_madd_real(
    p2pw_complex<Real> &acc,
    Real a,
    p2pw_complex<Real> b
) {
    acc.r = fma(a, b.r, acc.r);
    acc.i = fma(a, b.i, acc.i);
}

template <typename Real>
__device__ __forceinline__ void p2pw_madd(
    p2pw_complex<Real> &acc,
    p2pw_complex<Real> a,
    p2pw_complex<Real> b
) {
    acc.r = fma(a.r, b.r, acc.r);
    acc.r = fma(-a.i, b.i, acc.r);
    acc.i = fma(a.r, b.i, acc.i);
    acc.i = fma(a.i, b.r, acc.i);
}

// KERNEL_START

extern "C" __global__ void Proxy2PwKernel(Proxy2PwArgs<Real> a) {
    using Complex = p2pw_complex<Real>;

    const int box_idx = blockIdx.x;
    if (box_idx >= a.n_boxes_at_level)
        return;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    Complex *__restrict__ ff = reinterpret_cast<Complex *>(shared_raw);
    Complex *__restrict__ ff2 = ff + PROXY2PW_Z_TILE * N_ORDER * N_ORDER;
    Complex *__restrict__ poly2pw_s = ff2 + PROXY2PW_Z_TILE * N_ORDER * N_PW;

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

    for (int idx = threadIdx.x; idx < n_order * n_pw; idx += blockDim.x)
        poly2pw_s[idx] = p2pw_load(a.poly2pw, idx);

    __syncthreads();

    for (int d = 0; d < N_CHARGE_DIM; ++d) {
        const Real *proxy_d = proxy + d * n_order3;
        Real *pw_d = pw_dst + 2 * d * n_pw_modes;

        for (int m3_base = 0; m3_base < n_pw2; m3_base += PROXY2PW_Z_TILE) {
            const int z_count =
                (m3_base + PROXY2PW_Z_TILE <= n_pw2) ? PROXY2PW_Z_TILE : (n_pw2 - m3_base);

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

            // Phase 2: ff2(zr, i, m2) = sum_j ff(zr, i, j) * poly2pw(m2, j).
            constexpr int I_TILE = PROXY2PW_I_TILE;
            constexpr int M2_TILE = PROXY2PW_M2_TILE;
            const int i_tiles = (n_order + I_TILE - 1) / I_TILE;
            const int m2_tiles = (n_pw + M2_TILE - 1) / M2_TILE;
            const int phase2_tiles = z_count * i_tiles * m2_tiles;

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
                                ff2[zr * n_order * n_pw + i + m2 * n_order] = acc[ii][r];
                        }
                    }
                }
            }

            __syncthreads();

            // Phase 3: pw(m1, m2, m3, d) = sum_i ff2(zr, i, m2) * poly2pw(m1, i).
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

                Complex acc[M2_OUT_TILE][M1_TILE];

#pragma unroll
                for (int c = 0; c < M2_OUT_TILE; ++c) {
#pragma unroll
                    for (int r = 0; r < M1_TILE; ++r)
                        acc[c][r] = p2pw_zero<Real>();
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
                            const Complex f = ff2[zr * n_order * n_pw + i + m2 * n_order];

#pragma unroll
                            for (int r = 0; r < M1_TILE; ++r)
                                p2pw_madd(acc[c][r], f, b[r]);
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
                            if (m1 < n_pw) {
                                const int flat = m1 + m2 * n_pw + m3 * n_pw * n_pw;
                                p2pw_store(pw_d, flat, acc[c][r]);
                            }
                        }
                    }
                }
            }

            __syncthreads();
        }
    }
}
