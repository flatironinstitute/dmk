// V2 tensorprod device kernel (parent<->child proxy transfer). The launcher
// prepends `using Real` and the N_ORDER / N_CHARGE_DIM / BLOCK_SIZE /
// TENSOR_Z_TILE / TENSOR_I_TILE / TENSOR_J_TILE constants. Used by both the
// upward sweep (child->parent, umat=c2p, gathered by parent) and the downward
// sweep (parent->child, umat=p2c, one pair per block).

#include <dmk/cuda/tensorprod_kernelargs.hpp>

// KERNEL_START

extern "C" __global__ void PtTensorprodKernel(dmk::cuda::TensorprodArgs<Real> a) {
    constexpr int N = N_ORDER;
    constexpr int N2 = N * N;
    constexpr int N3 = N * N * N;

    const int block_idx = blockIdx.x;
    const bool gather = a.par_boxes != nullptr;
    if (block_idx >= (gather ? a.n_par : a.n_pairs))
        return;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    Real *__restrict__ ff = reinterpret_cast<Real *>(shared_raw);
    Real *__restrict__ ff2 = ff + TENSOR_Z_TILE * N2;
    Real *__restrict__ umat_s = ff2 + TENSOR_Z_TILE * N2;
    Real *__restrict__ umat_x = umat_s;
    Real *__restrict__ umat_y = umat_x + N2;
    Real *__restrict__ umat_z = umat_y + N2;

    const int dst_box = gather ? a.par_boxes[block_idx] : a.dst_boxes[block_idx];
    const int child_begin = gather ? a.par_child_begin[block_idx] : block_idx;
    const int child_end = gather ? child_begin + a.par_child_count[block_idx] : block_idx + 1;

    Real *dst_base = a.proxy_flat + a.proxy_offsets[dst_box];

    // One block owns one charge dim of the destination. The z tiles stay inside the block: they
    // all sweep the same child slab, so splitting them across blocks costs a reread per tile that
    // L1 currently absorbs, and that outweighs anything won on the destination side.
    const int d = blockIdx.y;
    Real *fout = dst_base + d * N3;

    for (int pair_idx = child_begin; pair_idx < child_end; ++pair_idx) {
        // Marked on the parent's first child only, so exactly one pair assigns.
        const bool assign_dst = a.assign_dst && a.assign_dst[pair_idx];
        const Real *umat_oct = a.umat_flat + a.child_octants[pair_idx] * 3 * N2;
        for (int idx = threadIdx.x; idx < 3 * N2; idx += blockDim.x)
            umat_s[idx] = umat_oct[idx];
        __syncthreads();

        const Real *fin = a.proxy_flat + a.proxy_offsets[a.src_boxes[pair_idx]] + d * N3;

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
                        Real *__restrict__ out = fout + iout + jout * N + (z_base + zr) * N2;
                        // atomicAdd, with its result unused, lowers to a fire-and-forget RED that
                        // issues like a store and lets L2 do the read-modify-write. `*out +=`
                        // would stall the warp on a dependent global load. Not about contention:
                        // the destination is owned by this block either way.
                        if (assign_dst)
                            *out = acc[r];
                        else
                            atomicAdd(out, acc[r]);
                    }
                }
            }

            __syncthreads();
        }
    }
}
