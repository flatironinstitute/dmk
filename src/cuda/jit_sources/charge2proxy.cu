template <typename Real>
__device__ inline void chebyshev_fill_strided(Real x, Real *out, int n, int stride) {
    Real t0 = Real{1};
    out[0] = t0;
    if (n <= 1)
        return;
    Real t1 = x;
    out[stride] = t1;
    for (int k = 2; k < n; ++k) {
        const Real t2 = Real{2} * x * t1 - t0;
        out[k * stride] = t2;
        t0 = t1;
        t1 = t2;
    }
}

// KERNEL_START

extern "C" __global__ void Charge2ProxyKernel(
    Charge2ProxyArgs<Real> a,
    const int *__restrict__ group_perm
) {
    // constexpr int CHUNK = 128;
    constexpr int LD = CHUNK + 1;
    constexpr int DIM = 3;

    constexpr int N  = N_ORDER;
    const int NC = N_CHARGE_DIM;
    constexpr int N2 = N * N;
    constexpr int N3 = N2 * N;

    constexpr int I_TILES = (N + I_TILE - 1) / I_TILE;
    constexpr int J_TILES = (N + J_TILE - 1) / J_TILE;
    constexpr int K_TILES = (N + K_TILE - 1) / K_TILE;

    const int NTILES = I_TILES * J_TILES * K_TILES * NC;

    extern __shared__ unsigned char shared_raw[];

    Real *poly_x = reinterpret_cast<Real *>(shared_raw);
    Real *poly_y = poly_x + N * LD;
    Real *poly_z = poly_y + N * LD;
    Real *charges_s = poly_z + N * LD;

    const int logical_g = blockIdx.x;
    if (logical_g >= a.n_groups)
        return;

    const int g = group_perm ? group_perm[logical_g] : logical_g;

    const int center_box = a.center_boxes[g];
    const int level = a.levels[g];
    const int sb_off = a.src_box_flat_offsets[g];
    const int n_src_boxes = a.n_src_boxes_per_group[g];

    const Real cx = a.centers[center_box * DIM + 0];
    const Real cy = a.centers[center_box * DIM + 1];
    const Real cz = a.centers[center_box * DIM + 2];

    const Real scale = a.inv_box_scale[level];

    Real *__restrict__ proxy =
        a.proxy_flat + a.proxy_offsets[center_box];

    for (int sbi = 0; sbi < n_src_boxes; ++sbi) {
        const int sb = a.src_boxes_flat[sb_off + sbi];
        const int n_src = a.src_counts_owned[sb];

        if (n_src == 0)
            continue;

        const Real *__restrict__ r_src =
            a.r_src_owned + a.r_src_owned_offsets[sb];

        const Real *__restrict__ charge =
            a.charge_owned + a.charge_owned_offsets[sb];

        for (int s_base = 0; s_base < n_src; s_base += CHUNK) {
            const int n_in_chunk =
                (s_base + CHUNK > n_src) ? (n_src - s_base) : CHUNK;

            // Build Chebyshev tables for this source chunk.
            for (int s = threadIdx.x; s < n_in_chunk; s += blockDim.x) {
                const int sp = s_base + s;

                const Real x = (r_src[sp * DIM + 0] - cx) * scale;
                const Real y = (r_src[sp * DIM + 1] - cy) * scale;
                const Real z = (r_src[sp * DIM + 2] - cz) * scale;

                chebyshev_fill_strided<Real>(x, poly_x + s, N, LD);
                chebyshev_fill_strided<Real>(y, poly_y + s, N, LD);
                chebyshev_fill_strided<Real>(z, poly_z + s, N, LD);
            }

            for (int t = threadIdx.x; t < NC * n_in_chunk; t += blockDim.x) {
                const int s = t / NC;
                const int d = t - s * NC;
                const int sp = s_base + s;

                charges_s[d * LD + s] = charge[d + sp * NC];
            }

            __syncthreads();

            for (int tile = threadIdx.x; tile < NTILES; tile += blockDim.x) {
                int idx = tile;

                const int it = idx % I_TILES;
                idx /= I_TILES;

                const int jt = idx % J_TILES;
                idx /= J_TILES;

                const int kt = idx % K_TILES;
                idx /= K_TILES;

                const int d = idx;

                const int i0 = it * I_TILE;
                const int j0 = jt * J_TILE;
                const int k0 = kt * K_TILE;

                Real acc[K_TILE][I_TILE][J_TILE];

                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    #pragma unroll
                    for (int r = 0; r < I_TILE; ++r) {
                        #pragma unroll
                        for (int c = 0; c < J_TILE; ++c) {
                            acc[kk][r][c] = Real{0};
                        }
                    }
                }

                for (int s = 0; s < n_in_chunk; ++s) {
                    Real xreg[I_TILE];
                    Real yreg[J_TILE];
                    Real zqreg[K_TILE];

                    #pragma unroll
                    for (int r = 0; r < I_TILE; ++r) {
                        const int i = i0 + r;

                        if (i < N)
                            xreg[r] = poly_x[i * LD + s];
                        else
                            xreg[r] = Real{0};
                    }

                    #pragma unroll
                    for (int c = 0; c < J_TILE; ++c) {
                        const int j = j0 + c;

                        if (j < N)
                            yreg[c] = poly_y[j * LD + s];
                        else
                            yreg[c] = Real{0};
                    }

                    const Real q = charges_s[d * LD + s];

                    #pragma unroll
                    for (int kk = 0; kk < K_TILE; ++kk) {
                        const int k = k0 + kk;

                        if (k < N)
                            zqreg[kk] = poly_z[k * LD + s] * q;
                        else
                            zqreg[kk] = Real{0};
                    }

                   
                    #pragma unroll
                    for (int kk = 0; kk < K_TILE; ++kk) {
                        #pragma unroll
                        for (int c = 0; c < J_TILE; ++c) {
                            const Real yzq = yreg[c] * zqreg[kk];

                            #pragma unroll
                            for (int r = 0; r < I_TILE; ++r) {
                                acc[kk][r][c] =
                                    fma(xreg[r], yzq, acc[kk][r][c]);
                            }
                        }
                    }
                }

                #pragma unroll
                for (int kk = 0; kk < K_TILE; ++kk) {
                    const int k = k0 + kk;

                    if (k < N) {
                        #pragma unroll
                        for (int r = 0; r < I_TILE; ++r) {
                            const int i = i0 + r;

                            if (i < N) {
                                #pragma unroll
                                for (int c = 0; c < J_TILE; ++c) {
                                    const int j = j0 + c;

                                    if (j < N) {
                                        proxy[i + j * N + k * N2 + d * N3] += acc[kk][r][c];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            __syncthreads();
        }
    }
}