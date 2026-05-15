// Per-box multiply_kernelFT family: pointwise complex × real multiplies that
// apply the radial kernel-Fourier-transform factor to plane-wave coefficients.

#include <dmk/cuda/multiply_kernelft_kernels.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace dmk::cuda {

// cd2p: pointwise complex × real.
template <typename Real>
__global__ void MultiplyCd2pByBoxKernel(MultiplyCd2pArgs<Real> a) {
    const int box_idx = blockIdx.x;
    if (box_idx >= a.n_boxes_at_level)
        return;
    const int box = a.box_ids[box_idx];
    const long off_complex = a.pw_offsets ? a.pw_offsets[box] : box_idx * a.pw_stride_complex;
    if (off_complex < 0)
        return;
    Real *pw = a.pw_flat + 2 * off_complex;
    const int total = a.n_pw_modes * a.n_charge_dim;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int m = idx % a.n_pw_modes;
        const Real f = a.radialft[m];
        pw[2 * idx] *= f;
        pw[2 * idx + 1] *= f;
    }
}

// Stokeslet 3D. For each (m1, m2, m3):
//     dot = p0*kx + p1*ky + p2*kz   (complex)
//     dd  = (kx² + ky² + kz²) * f
//     pw[d] = dot * (k_d * f) - p_d * dd  for d = x, y, z
//   Windowed variant additionally adds (1/rl) * pre-multiply value at the
//   n0-th mode (the "DC" plane wave at the centre slot).
template <typename Real>
__global__ void MultiplyStokeslet3DByBoxKernel(MultiplyStokeslet3DArgs<Real> a) {
    extern __shared__ unsigned char shared_raw[];
    Real *cvec = reinterpret_cast<Real *>(shared_raw); // 6 reals (3 complex) for windowed correction

    const int box_idx = blockIdx.x;
    if (box_idx >= a.n_boxes_at_level)
        return;
    const int box = a.box_ids[box_idx];
    const long off_complex = a.pw_offsets ? a.pw_offsets[box] : box_idx * a.pw_stride_complex;
    if (off_complex < 0)
        return;
    Real *pw = a.pw_flat + 2 * off_complex;

    const int n_pw = a.n_pw;
    const int n_pw_modes = a.n_pw_modes;
    const int npw_half = n_pw / 2;
    const Real hpw = a.hpw;
    auto ts = [&](int i) { return Real(i - npw_half) * hpw; };

    // n0 is the centre-mode index used by the windowed correction: it
    // corresponds to ix=iy=npw_half, iz=npw_half (the unique "DC" point).
    const int n0 = npw_half + n_pw * npw_half + n_pw * n_pw * npw_half;

    if (a.is_windowed && threadIdx.x == 0) {
        for (int d = 0; d < 3; ++d) {
            cvec[2 * d + 0] = pw[2 * (n0 + d * n_pw_modes)];
            cvec[2 * d + 1] = pw[2 * (n0 + d * n_pw_modes) + 1];
        }
    }
    __syncthreads();

    for (int n_idx = threadIdx.x; n_idx < n_pw_modes; n_idx += blockDim.x) {
        const int ix = n_idx % n_pw;
        const int iy = (n_idx / n_pw) % n_pw;
        const int iz = n_idx / (n_pw * n_pw);
        const Real kx = ts(ix);
        const Real ky = ts(iy);
        const Real kz = ts(iz);
        const Real f = a.radialft[n_idx];
        const Real dd = (kx * kx + ky * ky + kz * kz) * f;

        const int off0 = 2 * n_idx;
        const int off1 = 2 * (n_idx + n_pw_modes);
        const int off2 = 2 * (n_idx + 2 * n_pw_modes);
        const Real p0r = pw[off0], p0i = pw[off0 + 1];
        const Real p1r = pw[off1], p1i = pw[off1 + 1];
        const Real p2r = pw[off2], p2i = pw[off2 + 1];

        const Real dr = p0r * kx + p1r * ky + p2r * kz;
        const Real di = p0i * kx + p1i * ky + p2i * kz;

        pw[off0] = dr * (kx * f) - p0r * dd;
        pw[off0 + 1] = di * (kx * f) - p0i * dd;
        pw[off1] = dr * (ky * f) - p1r * dd;
        pw[off1 + 1] = di * (ky * f) - p1i * dd;
        pw[off2] = dr * (kz * f) - p2r * dd;
        pw[off2 + 1] = di * (kz * f) - p2i * dd;
    }

    if (a.is_windowed) {
        __syncthreads();
        if (threadIdx.x == 0) {
            // rl = sqrt(3) + 1; cval = 1/rl.
            const Real cval = Real(1) / (Real(1.7320508075688772935) + Real(1));
            for (int d = 0; d < 3; ++d) {
                pw[2 * (n0 + d * n_pw_modes)] += cval * cvec[2 * d + 0];
                pw[2 * (n0 + d * n_pw_modes) + 1] += cval * cvec[2 * d + 1];
            }
        }
    }
}

// Stresslet 3D. Reads 9 input tables, writes 3 output tables. For each (m1, m2, m3) and
//   k = (kx, ky, kz), with P_{ij} = src[m, i + 3*j]:
//     u_i = -i f * (k_i * (|k|² tr(P) - 2 k^T P k) + |k|² ((P+P^T)k)_i)
//   (See stresslet_3d_multiply_kernelFT in tree.hpp for the host reference.)
template <typename Real>
__global__ void MultiplyStresslet3DByBoxKernel(MultiplyStresslet3DArgs<Real> a) {
    const int box_idx = blockIdx.x;
    if (box_idx >= a.n_boxes_at_level)
        return;
    const int box = a.box_ids[box_idx];
    const long src_off = a.src_offsets ? a.src_offsets[box] : box_idx * a.src_stride_complex;
    const long dst_off = a.dst_offsets ? a.dst_offsets[box] : box_idx * a.dst_stride_complex;
    if (src_off < 0 || dst_off < 0)
        return;
    const Real *src = a.src_flat + 2 * src_off;
    Real *dst = a.dst_flat + 2 * dst_off;

    const int n_pw = a.n_pw;
    const int n_pw_modes = a.n_pw_modes;
    const int npw_half = n_pw / 2;
    const Real hpw = a.hpw;
    auto ts = [&](int i) { return Real(i - npw_half) * hpw; };

    for (int n_idx = threadIdx.x; n_idx < n_pw_modes; n_idx += blockDim.x) {
        const int ix = n_idx % n_pw;
        const int iy = (n_idx / n_pw) % n_pw;
        const int iz = n_idx / (n_pw * n_pw);
        const Real kx = ts(ix);
        const Real ky = ts(iy);
        const Real kz = ts(iz);
        const Real f = a.radialft[n_idx];
        const Real rksq = kx * kx + ky * ky + kz * kz;
        const Real k[3] = {kx, ky, kz};

        // Load P[i][j] = src[n_idx + n_pw_modes * (i + 3*j)] (complex).
        Real Pr[3][3], Pi[3][3];
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                const int o = 2 * (n_idx + n_pw_modes * (i + 3 * j));
                Pr[i][j] = src[o];
                Pi[i][j] = src[o + 1];
            }
        }

        // k^T P k (complex)
        Real ddr = 0, ddi = 0;
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
                const Real w = k[i] * k[j];
                ddr += Pr[i][j] * w;
                ddi += Pi[i][j] * w;
            }

        // tr(P) (complex)
        Real tr_r = Pr[0][0] + Pr[1][1] + Pr[2][2];
        Real tr_i = Pi[0][0] + Pi[1][1] + Pi[2][2];

        // ((P + P^T) k)_i
        Real prod_r[3]{}, prod_i[3]{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                prod_r[i] += (Pr[i][j] + Pr[j][i]) * k[j];
                prod_i[i] += (Pi[i][j] + Pi[j][i]) * k[j];
            }

        // zz = rksq * tr(P) - 2 * ddot
        const Real zz_r = rksq * tr_r - Real(2) * ddr;
        const Real zz_i = rksq * tr_i - Real(2) * ddi;

        // u_i = -i f * (k_i * zz + rksq * prod_i)
        // multiplying by -i = (0, -1) means: (a + b i) * (0, -1) = (b, -a).
        for (int i = 0; i < 3; ++i) {
            const Real ar = k[i] * zz_r + rksq * prod_r[i];
            const Real ai = k[i] * zz_i + rksq * prod_i[i];
            const int o = 2 * (n_idx + n_pw_modes * i);
            dst[o] = f * ai;
            dst[o + 1] = -f * ar;
        }
    }
}

template <typename Real, int DIM>
void launch_multiply_cd2p(const MultiplyCd2pArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;
    constexpr int block_size = 128;
    MultiplyCd2pByBoxKernel<Real><<<args.n_boxes_at_level, block_size, 0, stream>>>(args);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_multiply_cd2p: ") + cudaGetErrorString(err));
}

template <typename Real>
void launch_multiply_stokeslet_3d(const MultiplyStokeslet3DArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;
    constexpr int block_size = 128;
    const std::size_t shared_bytes = sizeof(Real) * 6; // cvec for windowed correction
    MultiplyStokeslet3DByBoxKernel<Real><<<args.n_boxes_at_level, block_size, shared_bytes, stream>>>(args);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_multiply_stokeslet_3d: ") + cudaGetErrorString(err));
}

template <typename Real>
void launch_multiply_stresslet_3d(const MultiplyStresslet3DArgs<Real> &args, cudaStream_t stream) {
    if (args.n_boxes_at_level == 0)
        return;
    constexpr int block_size = 128;
    MultiplyStresslet3DByBoxKernel<Real><<<args.n_boxes_at_level, block_size, 0, stream>>>(args);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("launch_multiply_stresslet_3d: ") + cudaGetErrorString(err));
}

template void launch_multiply_cd2p<float, 2>(const MultiplyCd2pArgs<float> &, cudaStream_t);
template void launch_multiply_cd2p<float, 3>(const MultiplyCd2pArgs<float> &, cudaStream_t);
template void launch_multiply_cd2p<double, 2>(const MultiplyCd2pArgs<double> &, cudaStream_t);
template void launch_multiply_cd2p<double, 3>(const MultiplyCd2pArgs<double> &, cudaStream_t);
template void launch_multiply_stokeslet_3d<float>(const MultiplyStokeslet3DArgs<float> &, cudaStream_t);
template void launch_multiply_stokeslet_3d<double>(const MultiplyStokeslet3DArgs<double> &, cudaStream_t);
template void launch_multiply_stresslet_3d<float>(const MultiplyStresslet3DArgs<float> &, cudaStream_t);
template void launch_multiply_stresslet_3d<double>(const MultiplyStresslet3DArgs<double> &, cudaStream_t);

} // namespace dmk::cuda
