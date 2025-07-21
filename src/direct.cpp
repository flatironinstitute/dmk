#include <dmk/chebychev.hpp>
#include <dmk/direct.hpp>
#include <dmk/vector_kernels.hpp>

#define VECDIM 4

#ifdef __AVX512F__
#undef VECDIM
#define VECDIM 8
#endif

namespace dmk {

template <typename Real, int DIM>
void yukawa_direct_eval(const ndview<const Real, 2> &r_src, const std::array<std::span<const Real>, DIM> &r_trg,
                        const ndview<const Real, 2> &charges, const ndview<const Real, 1> &coeffs, Real lambda,
                        Real scale, Real center, Real d2max, ndview<Real, 2> &u) {
    const int nsrc = r_src.extent(1);
    const int ntrg = r_trg[0].size();
    const int nd = u.extent(0);
    const int norder = coeffs.extent(0);
    constexpr Real threshq = 1E-30;

    for (int i_trg = 0; i_trg < ntrg; i_trg++) {
        for (int i_src = 0; i_src < nsrc; i_src++) {
            const Real dx = r_trg[0][i_trg] - r_src(0, i_src);
            const Real dy = r_trg[1][i_trg] - r_src(1, i_src);
            Real dd = dx * dx + dy * dy;
            if constexpr (DIM == 3) {
                Real dz = r_trg[2][i_trg] - r_src(2, i_src);
                dd += dz * dz;
            }

            if (dd < threshq || dd > d2max)
                continue;

            const Real r = sqrt(dd);
            const Real xval = r * scale + center;
            const Real fval = chebyshev::evaluate(xval, norder + 1, coeffs.data());
            Real dkval;
            if constexpr (DIM == 2)
                dkval = std::cyl_bessel_k(0, lambda * r);
            if constexpr (DIM == 3)
                dkval = std::exp(-lambda * r) / r;

            const Real factor = dkval + fval;
            for (int i = 0; i < nd; ++i)
                u(i, i_trg) += charges(i, i_src) * factor;
        }
    }
}

template <typename Real, int DIM>
void direct_eval(dmk_ikernel ikernel, const ndview<Real, 2> &r_src, const std::array<std::span<const Real>, DIM> &r_trg,
                 const ndview<Real, 2> &charges, const ndview<Real, 1> &coeffs, const double *kernel_params, Real scale,
                 Real center, Real d2max, ndview<Real, 2> u, int n_digits) {
    constexpr int VECWIDTH = std::is_same_v<Real, float> ? 2 * VECDIM : VECDIM;
    const int nd = charges.extent(0);
    const int ndim = DIM;
    const int nsrc = r_src.extent(1);
    const int ntrg = r_trg[0].size();
    const Real thresh2 = 1E-30; // FIXME
    const Real *trg_ptrs[3] = {r_trg[0].data(), r_trg[1].data(), DIM == 3 ? r_trg[2].data() : nullptr};

    switch (ikernel) {
    case dmk_ikernel::DMK_YUKAWA:
        return yukawa_direct_eval<Real, DIM>(r_src, r_trg, charges, coeffs, *kernel_params, scale, center, d2max, u);
    case dmk_ikernel::DMK_LAPLACE:
        if constexpr (DIM == 2)
            return log_local_kernel_directcp_vec_cpp<Real, VECWIDTH>(
                &nd, &ndim, &n_digits, &scale, &center, &d2max, r_src.data(), &nsrc, charges.data(), trg_ptrs[0],
                trg_ptrs[1], trg_ptrs[2], &ntrg, u.data(), &thresh2);
        if constexpr (DIM == 3)
            return l3d_local_kernel_directcp_vec_cpp<Real, VECWIDTH>(
                &nd, &ndim, &n_digits, &scale, &center, &d2max, r_src.data(), &nsrc, charges.data(), trg_ptrs[0],
                trg_ptrs[1], trg_ptrs[2], &ntrg, u.data(), &thresh2);
    case dmk_ikernel::DMK_SQRT_LAPLACE:
        if constexpr (DIM == 2)
            return l3d_local_kernel_directcp_vec_cpp<Real, VECWIDTH>(
                &nd, &ndim, &n_digits, &scale, &center, &d2max, r_src.data(), &nsrc, charges.data(), trg_ptrs[0],
                trg_ptrs[1], trg_ptrs[2], &ntrg, u.data(), &thresh2);
        if constexpr (DIM == 3)
            return sl3d_local_kernel_directcp_vec_cpp<Real, VECWIDTH>(
                &nd, &ndim, &n_digits, &scale, &center, &d2max, r_src.data(), &nsrc, charges.data(), trg_ptrs[0],
                trg_ptrs[1], trg_ptrs[2], &ntrg, u.data(), &thresh2);
    }
}

template void direct_eval<float, 2>(dmk_ikernel ikernel, const ndview<float, 2> &r_src,
                                    const std::array<std::span<const float>, 2> &r_trg, const ndview<float, 2> &charges,
                                    const ndview<float, 1> &coeffs, const double *kernel_params, float scale,
                                    float center, float d2max, ndview<float, 2> u, int n_digits);
template void direct_eval<float, 3>(dmk_ikernel ikernel, const ndview<float, 2> &r_src,
                                    const std::array<std::span<const float>, 3> &r_trg, const ndview<float, 2> &charges,
                                    const ndview<float, 1> &coeffs, const double *kernel_params, float scale,
                                    float center, float d2max, ndview<float, 2> u, int n_digits);
template void direct_eval<double, 2>(dmk_ikernel ikernel, const ndview<double, 2> &r_src,
                                     const std::array<std::span<const double>, 2> &r_trg,
                                     const ndview<double, 2> &charges, const ndview<double, 1> &coeffs,
                                     const double *kernel_params, double scale, double center, double d2max,
                                     ndview<double, 2> u, int n_digits);
template void direct_eval<double, 3>(dmk_ikernel ikernel, const ndview<double, 2> &r_src,
                                     const std::array<std::span<const double>, 3> &r_trg,
                                     const ndview<double, 2> &charges, const ndview<double, 1> &coeffs,
                                     const double *kernel_params, double scale, double center, double d2max,
                                     ndview<double, 2> u, int n_digits);
} // namespace dmk
