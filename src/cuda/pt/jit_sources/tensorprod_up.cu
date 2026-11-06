// Upward tensorprod (children -> parent, c2p), one block per (parent, charge dim). The launcher
// prepends `using Real` and the N_ORDER / BLOCK_SIZE / K_TILE / SMEM_GLOBAL constants.
//
// Along each axis the plus child's matrix is D M D, with M the minus child's and D = diag((-1)^k).
// Two children that differ only along the contracted axis therefore fold into one contraction,
// out(m) = sum_i M(m, i) (a_i + (-1)^(m+i) b_i): M against s = a + D b for even m and t = a - D b
// for odd m. The eight children fold 4 -> 2 -> 1 along x, y, then z, for 7 n^4 FMAs instead of
// 24 n^4. Child z-slabs stream in K_TILE at a time and are read once; x and y run per slab and z
// accumulates the parent in registers.

#include <dmk/cuda/tensorprod_kernelargs.hpp>

// KERNEL_START

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE) PtTensorprodUpKernel(dmk::cuda::TensorprodArgs<Real> a) {
    constexpr int N = N_ORDER;
    constexpr int N2 = N * N;
    constexpr int KT = K_TILE;
    constexpr int NE = (N + 1) / 2; // rows per parity, odd rows padded when N is odd
    constexpr int NP = (NE + 3) / 4 * 4;
    constexpr int LD = N | 1; // odd, so lanes walking j or mx hit distinct banks
    constexpr int XS_ST = 4 * KT * N * LD;
    constexpr int YS_ST = 2 * KT * N * LD;
    constexpr int ZS_ST = KT * N2;
    constexpr int Z_ITEMS = 2 * N2;
    constexpr int ZI = (Z_ITEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int LOADS = (4 * KT * N2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // When one slab's buffers exceed shared memory (high order in double), the block runs against
    // its own slice of global scratch instead.
    extern __shared__ __align__(16) unsigned char shared_raw[];
    unsigned char *smem =
        SMEM_GLOBAL ? reinterpret_cast<unsigned char *>(a.scratch + (long(blockIdx.y) * gridDim.x + blockIdx.x) *
                                                                         a.scratch_stride)
                    : shared_raw;
    const Real **child = reinterpret_cast<const Real **>(smem);           // [octant]
    Real *mp = reinterpret_cast<Real *>(smem + 8 * sizeof(Real *)); // [p][i][r] = M(2r + p, i)
    Real *xs = mp + 2 * N * NP;                                           // [s/t][oy + 2 oz][kt][j][i]
    Real *ys = xs + 2 * XS_ST;                                            // [s/t][oz][kt][mx][j]
    Real *zs = ys + 2 * YS_ST;                                            // [s/t][kt][mx + my N]

    const int par = blockIdx.x;
    const int d = blockIdx.y;
    const int c0 = a.par_child_begin[par];
    const int nc = a.par_child_count[par];

    if (threadIdx.x < 8)
        child[threadIdx.x] = nullptr;
    __syncthreads();
    if (int(threadIdx.x) < nc) {
        const int c = c0 + threadIdx.x;
        child[a.child_octants[c]] = a.proxy_flat + a.proxy_offsets[a.src_boxes[c]] + d * N2 * N;
    }
    for (int idx = threadIdx.x; idx < 2 * N * NP; idx += BLOCK_SIZE) {
        const int r = idx % NP, i = idx / NP % N, p = idx / (NP * N);
        const int m = 2 * r + p;
        mp[idx] = (r < NE && m < N) ? a.umat_flat[m + i * N] : Real{0};
    }
    const bool assign =
        __syncthreads_or(int(threadIdx.x) < nc && a.assign_dst && a.assign_dst[c0 + threadIdx.x]);

    Real acc[ZI][NE];
#pragma unroll
    for (int u = 0; u < ZI; ++u)
#pragma unroll
        for (int r = 0; r < NE; ++r)
            acc[u][r] = Real{0};

    for (int k0 = 0; k0 < N; k0 += KT) {
        const int kc = min(KT, N - k0);

        // Children a = (0, oy, oz) and b = (1, oy, oz) of pair q = oy + 2 oz. All loads issue
        // before any store so they overlap.
        Real va[LOADS], vb[LOADS];
#pragma unroll
        for (int u = 0; u < LOADS; ++u) {
            const int e = threadIdx.x + u * BLOCK_SIZE;
            va[u] = vb[u] = Real{0};
            if (e < 4 * kc * N2) {
                const int ij = e % N2, kt = e / N2 % kc, q = e / (N2 * kc);
                const int off = ij + (k0 + kt) * N2;
                const Real *pa = child[2 * q], *pb = child[2 * q + 1];
                if (pa)
                    va[u] = pa[off];
                if (pb)
                    vb[u] = pb[off];
            }
        }
#pragma unroll
        for (int u = 0; u < LOADS; ++u) {
            const int e = threadIdx.x + u * BLOCK_SIZE;
            if (e < 4 * kc * N2) {
                const int ij = e % N2, kt = e / N2 % kc, q = e / (N2 * kc);
                const int i = ij % N, j = ij / N;
                const Real b = (i & 1) ? -vb[u] : vb[u];
                Real *dst = xs + ((q * KT + kt) * N + j) * LD + i;
                dst[0] = va[u] + b;
                dst[XS_ST] = va[u] - b;
            }
        }
        __syncthreads();

        // x: G_(oy,oz)(mx, j); the two oy then fold along j for y.
        for (int it = threadIdx.x; it < 4 * kc * N; it += BLOCK_SIZE) {
            const int j = it % N, kt = it / N % kc, oz = it / (N * kc) % 2, p = it / (2 * N * kc);
            const Real *w0 = xs + p * XS_ST + ((2 * oz * KT + kt) * N + j) * LD;
            const Real *w1 = w0 + KT * N * LD;
            Real g0[NE], g1[NE];
#pragma unroll
            for (int r = 0; r < NE; ++r)
                g0[r] = g1[r] = Real{0};
#pragma unroll
            for (int i = 0; i < N; ++i) {
                const Real x0 = w0[i], x1 = w1[i];
                const Real *m = mp + (p * N + i) * NP;
#pragma unroll
                for (int r = 0; r < NE; ++r) {
                    g0[r] = fma(m[r], x0, g0[r]);
                    g1[r] = fma(m[r], x1, g1[r]);
                }
            }
            const Real sg = (j & 1) ? Real{-1} : Real{1};
#pragma unroll
            for (int r = 0; r < NE; ++r) {
                const int mx = 2 * r + p;
                if (mx < N) {
                    Real *dst = ys + ((oz * KT + kt) * N + mx) * LD + j;
                    dst[0] = fma(sg, g1[r], g0[r]);
                    dst[YS_ST] = fma(-sg, g1[r], g0[r]);
                }
            }
        }
        __syncthreads();

        // y: H_oz(mx, my); the two oz then fold along k for z.
        for (int it = threadIdx.x; it < 2 * kc * N; it += BLOCK_SIZE) {
            const int mx = it % N, kt = it / N % kc, p = it / (N * kc);
            const Real *w0 = ys + p * YS_ST + (kt * N + mx) * LD;
            const Real *w1 = w0 + KT * N * LD;
            Real h0[NE], h1[NE];
#pragma unroll
            for (int r = 0; r < NE; ++r)
                h0[r] = h1[r] = Real{0};
#pragma unroll
            for (int j = 0; j < N; ++j) {
                const Real x0 = w0[j], x1 = w1[j];
                const Real *m = mp + (p * N + j) * NP;
#pragma unroll
                for (int r = 0; r < NE; ++r) {
                    h0[r] = fma(m[r], x0, h0[r]);
                    h1[r] = fma(m[r], x1, h1[r]);
                }
            }
            const Real sg = ((k0 + kt) & 1) ? Real{-1} : Real{1};
#pragma unroll
            for (int r = 0; r < NE; ++r) {
                const int my = 2 * r + p;
                if (my < N) {
                    Real *dst = zs + kt * N2 + mx + my * N;
                    dst[0] = fma(sg, h1[r], h0[r]);
                    dst[ZS_ST] = fma(-sg, h1[r], h0[r]);
                }
            }
        }
        __syncthreads();

        // z: parent(mx, my, mz) += M(mz, k) w(mx, my).
#pragma unroll
        for (int u = 0; u < ZI; ++u) {
            const int it = threadIdx.x + u * BLOCK_SIZE;
            if (it < Z_ITEMS) {
                const int c = it % N2, p = it / N2;
                for (int kt = 0; kt < kc; ++kt) {
                    const Real w = zs[p * ZS_ST + kt * N2 + c];
                    const Real *m = mp + (p * N + k0 + kt) * NP;
#pragma unroll
                    for (int r = 0; r < NE; ++r)
                        acc[u][r] = fma(m[r], w, acc[u][r]);
                }
            }
        }
    }

    Real *out = a.proxy_flat + a.proxy_offsets[a.par_boxes[par]] + d * N2 * N;
#pragma unroll
    for (int u = 0; u < ZI; ++u) {
        const int it = threadIdx.x + u * BLOCK_SIZE;
        if (it < Z_ITEMS) {
            const int c = it % N2, p = it / N2;
#pragma unroll
            for (int r = 0; r < NE; ++r) {
                const int mz = 2 * r + p;
                if (mz < N) {
                    // A fire-and-forget RED: the block owns the parent, so this is about not
                    // stalling on a dependent load, not about contention.
                    if (assign)
                        out[c + mz * N2] = acc[u][r];
                    else
                        atomicAdd(out + c + mz * N2, acc[u][r]);
                }
            }
        }
    }
}
