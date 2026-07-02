// KERNEL_START

extern "C" __global__ void TensorprodKernel(TensorprodArgs<Real> a) {
    constexpr int N = N_ORDER;
    constexpr int N2 = N * N;
    constexpr int N3 = N * N * N;

    const int pair_idx = blockIdx.x;
    if (pair_idx >= a.n_pairs)
        return;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    Real *__restrict__ ff = reinterpret_cast<Real *>(shared_raw);
    Real *__restrict__ ff2 = ff + TENSOR_Z_TILE * N2;
    Real *__restrict__ umat_s = ff2 + TENSOR_Z_TILE * N2;
    Real *__restrict__ umat_x = umat_s;
    Real *__restrict__ umat_y = umat_x + N2;
    Real *__restrict__ umat_z = umat_y + N2;

    const int src_box = a.src_boxes[pair_idx];
    const int dst_box = a.dst_boxes[pair_idx];
    const int oct = a.child_octants[pair_idx];

    const Real *umat_oct = a.umat_flat + oct * 3 * N2;

    for (int idx = threadIdx.x; idx < 3 * N2; idx += blockDim.x)
        umat_s[idx] = umat_oct[idx];

    __syncthreads();

    const Real *src_base = a.proxy_flat + a.proxy_offsets[src_box];
    Real *dst_base = a.proxy_flat + a.proxy_offsets[dst_box];

    for (int d = 0; d < N_CHARGE_DIM; ++d) {
        const Real *fin = src_base + d * N3;
        Real *fout = dst_base + d * N3;

        for (int z_base = 0; z_base < N; z_base += TENSOR_Z_TILE) {
            const int z_count = (z_base + TENSOR_Z_TILE <= N) ? TENSOR_Z_TILE : (N - z_base);

            // Phase 1: ff(zr, i, j) = sum_k fin(i, j, k) * umat_z(z_base + zr, k).
            for (int ij = threadIdx.x; ij < N2; ij += blockDim.x) {
                Real acc[TENSOR_Z_TILE];

#pragma unroll
                for (int zr = 0; zr < TENSOR_Z_TILE; ++zr)
                    acc[zr] = Real{0};

#pragma unroll
                for (int k = 0; k < N; ++k) {
                    const Real f = fin[ij + k * N2];

#pragma unroll
                    for (int zr = 0; zr < TENSOR_Z_TILE; ++zr) {
                        if (zr < z_count) {
                            const int zout = z_base + zr;
                            acc[zr] = fma(f, umat_z[zout + k * N], acc[zr]);
                        }
                    }
                }

#pragma unroll
                for (int zr = 0; zr < TENSOR_Z_TILE; ++zr) {
                    if (zr < z_count)
                        ff[zr * N2 + ij] = acc[zr];
                }
            }

            __syncthreads();

            // Phase 2: ff2(zr, i, jout) = sum_j ff(zr, i, j) * umat_y(jout, j).
            constexpr int I_TILE = TENSOR_I_TILE;
            constexpr int J_TILE = TENSOR_J_TILE;
            const int i_tiles = (N + I_TILE - 1) / I_TILE;
            const int j_tiles = (N + J_TILE - 1) / J_TILE;
            const int phase2_tiles = z_count * i_tiles * j_tiles;

            for (int tile = threadIdx.x; tile < phase2_tiles; tile += blockDim.x) {
                int x = tile;
                const int jout_tile = x % j_tiles;
                x /= j_tiles;
                const int i_tile = x % i_tiles;
                const int zr = x / i_tiles;
                const int i_base = i_tile * I_TILE;
                const int jout_base = jout_tile * J_TILE;

                Real acc[I_TILE][J_TILE];

#pragma unroll
                for (int ii = 0; ii < I_TILE; ++ii) {
#pragma unroll
                    for (int r = 0; r < J_TILE; ++r)
                        acc[ii][r] = Real{0};
                }

#pragma unroll
                for (int j = 0; j < N; ++j) {
                    Real b[J_TILE];

#pragma unroll
                    for (int r = 0; r < J_TILE; ++r) {
                        const int jout = jout_base + r;
                        b[r] = (jout < N) ? umat_y[jout + j * N] : Real{0};
                    }

#pragma unroll
                    for (int ii = 0; ii < I_TILE; ++ii) {
                        const int i = i_base + ii;
                        if (i < N) {
                            const Real f = ff[zr * N2 + i + j * N];

#pragma unroll
                            for (int r = 0; r < J_TILE; ++r)
                                acc[ii][r] = fma(f, b[r], acc[ii][r]);
                        }
                    }
                }

#pragma unroll
                for (int ii = 0; ii < I_TILE; ++ii) {
                    const int i = i_base + ii;
                    if (i < N) {
#pragma unroll
                        for (int r = 0; r < J_TILE; ++r) {
                            const int jout = jout_base + r;
                            if (jout < N)
                                ff2[zr * N2 + i + jout * N] = acc[ii][r];
                        }
                    }
                }
            }

            __syncthreads();

            // Phase 3: fout(iout, jout, z) += sum_i ff2(zr, i, jout) * umat_x(iout, i).
            const int phase3_tiles = z_count * j_tiles * N;

            for (int tile = threadIdx.x; tile < phase3_tiles; tile += blockDim.x) {
                int x = tile;
                const int iout = x % N;
                x /= N;
                const int jout_tile = x % j_tiles;
                const int zr = x / j_tiles;
                const int zout = z_base + zr;
                const int jout_base = jout_tile * J_TILE;

                Real acc[J_TILE];

#pragma unroll
                for (int r = 0; r < J_TILE; ++r)
                    acc[r] = Real{0};

#pragma unroll
                for (int i = 0; i < N; ++i) {
                    const Real b = umat_x[iout + i * N];

#pragma unroll
                    for (int r = 0; r < J_TILE; ++r) {
                        const int jout = jout_base + r;
                        if (jout < N) {
                            const Real f = ff2[zr * N2 + i + jout * N];
                            acc[r] = fma(f, b, acc[r]);
                        }
                    }
                }

#pragma unroll
                for (int r = 0; r < J_TILE; ++r) {
                    const int jout = jout_base + r;
                    if (jout < N) {
                        Real *__restrict__ out = fout + iout + jout * N + zout * N2;
                        if (a.additive_atomic)
                            atomicAdd(out, acc[r]);
                        else
                            *out += acc[r];
                    }
                }
            }

            __syncthreads();
        }
    }
}
