// KERNEL_START

extern "C" __global__ void Proxy2PwKernel(Proxy2PwArgs<Real> a) {
    const int box_idx = blockIdx.x;
    if (box_idx >= a.n_boxes_at_level)
        return;

    extern __shared__ unsigned char shared_raw[];
    Real *ff_slab = reinterpret_cast<Real *>(shared_raw);
    Real *ff2_slab = ff_slab + 2 * a.n_order * a.n_order;

    const int box = a.box_ids[box_idx];

    const long src_off = a.proxy_offsets[box];
    if (src_off < 0)
        return;
    const Real *proxy = a.proxy_flat + src_off;

    const long dst_off_complex = a.dst_offsets ? a.dst_offsets[box] : box_idx * a.dst_stride_complex;
    if (dst_off_complex < 0)
        return;
    Real *pw_dst = a.dst_flat + 2 * dst_off_complex;

    const int n_order = a.n_order;
    const int n_order2 = n_order * n_order;
    const int n_order3 = n_order2 * n_order;
    const int n_pw = a.n_pw;
    const int n_pw2 = a.n_pw2;
    const int n_pw_modes = n_pw * n_pw * n_pw2;

    for (int d = 0; d < a.n_charge_dim  ; ++d) {
        const Real *proxy_d = proxy + d * n_order3;
        Real *pw_d = pw_dst + 2 * d * n_pw_modes;

        for (int m3 = 0; m3 < a.n_pw2; ++m3) {
            // Phase 1: ff(i, j) = sum_k proxy(i, j, k, d) * poly2pw(m3, k).
            for (int t = threadIdx.x; t < n_order2; t += blockDim.x) {
                const int i = t % n_order;
                const int j = t / n_order;
                Real sum_r = Real{0}, sum_i = Real{0};
                for (int k = 0; k < n_order; ++k) {
                    const Real p = proxy_d[i + j * n_order + k * n_order2];
                    const Real qr = a.poly2pw[2 * (m3 + k * n_pw)];
                    const Real qi = a.poly2pw[2 * (m3 + k * n_pw) + 1];
                    sum_r += p * qr;
                    sum_i += p * qi;
                }
                ff_slab[2 * (i + j * n_order)] = sum_r;
                ff_slab[2 * (i + j * n_order) + 1] = sum_i;
            }
            __syncthreads();

            // Phase 2: ff2(i, m2) = sum_j ff(i, j) * poly2pw(m2, j).
            for (int t = threadIdx.x; t < n_order * n_pw; t += blockDim.x) {
                const int i = t % n_order;
                const int m2 = t / n_order;
                Real sum_r = Real{0}, sum_i = Real{0};
                for (int j = 0; j < n_order; ++j) {
                    const Real fr = ff_slab[2 * (i + j * n_order)];
                    const Real fi = ff_slab[2 * (i + j * n_order) + 1];
                    const Real qr = a.poly2pw[2 * (m2 + j * n_pw)];
                    const Real qi = a.poly2pw[2 * (m2 + j * n_pw) + 1];
                    sum_r += fr * qr - fi * qi;
                    sum_i += fr * qi + fi * qr;
                }
                ff2_slab[2 * (i + m2 * n_order)] = sum_r;
                ff2_slab[2 * (i + m2 * n_order) + 1] = sum_i;
            }
            __syncthreads();

            // Phase 3: pw(m1, m2, m3, d) = sum_i ff2(i, m2) * poly2pw(m1, i).
            for (int t = threadIdx.x; t < n_pw * n_pw; t += blockDim.x) {
                const int m1 = t % n_pw;
                const int m2 = t / n_pw;
                Real sum_r = Real{0}, sum_i = Real{0};
                for (int i = 0; i < n_order; ++i) {
                    const Real fr = ff2_slab[2 * (i + m2 * n_order)];
                    const Real fi = ff2_slab[2 * (i + m2 * n_order) + 1];
                    const Real qr = a.poly2pw[2 * (m1 + i * n_pw)];
                    const Real qi = a.poly2pw[2 * (m1 + i * n_pw) + 1];
                    sum_r += fr * qr - fi * qi;
                    sum_i += fr * qi + fi * qr;
                }
                const int flat = m1 + m2 * n_pw + m3 * n_pw * n_pw;
                pw_d[2 * flat] = sum_r;
                pw_d[2 * flat + 1] = sum_i;
            }
            __syncthreads();
        }
    }
}

