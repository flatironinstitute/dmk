#include <dmk/chebychev.hpp>
#include <dmk/direct.hpp>

#define VECDIM 4

#ifdef __AVX512F__
#undef VECDIM
#define VECDIM 8
#endif

template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void l3d_local_kernel_directcp_vec_cpp(const int32_t *nd, const int32_t *ndim, const int32_t *digits, const Real *rsc,
                                       const Real *cen, const Real *d2max, const Real *sources, const int32_t *ns,
                                       const Real *charge, const Real *xtarg, const Real *ytarg, const Real *ztarg,
                                       const int32_t *nt, Real *pot, const Real *thresh);

namespace dmk {

template <typename Real, int DIM>
void yukawa_direct_eval(const ndview<const Real, 2> &r_src, const std::array<std::span<const Real>, DIM> &r_trg,
                        const ndview<const Real, 2> &charges, const ndview<const Real, 1> &coeffs, Real lambda,
                        Real scale, Real center, Real d2max, const ndview<Real, 2> &u) {
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
            const Real fval = chebyshev::evaluate(xval, norder, coeffs.data_handle());
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
void direct_eval(dmk_ikernel ikernel, const ndview<const Real, 2> &r_src,
                 const std::array<std::span<const Real>, DIM> &r_trg, const ndview<const Real, 2> &charges,
                 const ndview<const Real, 1> &coeffs, const double *kernel_params, Real scale, Real center, Real d2max,
                 const ndview<Real, 2> &u) {
    switch (ikernel) {
    case dmk_ikernel::DMK_YUKAWA:
        yukawa_direct_eval<Real, DIM>(r_src, r_trg, charges, coeffs, *kernel_params, scale, center, d2max, u);
        break;
    case dmk_ikernel::DMK_LAPLACE:
        if constexpr (DIM == 2) {
            throw std::runtime_error("Laplace kernel not implemented in 2D");
        }
        if constexpr (DIM == 3) {
            const int32_t nd = charges.extent(0);
            const int32_t ndim = DIM;
            const int32_t digits = 6; // FIXME
            const int32_t nsrc = r_src.extent(1);
            const int32_t ntrg = r_trg[0].size();
            const Real thresh2 = 1E-30; // FIXME

            return l3d_local_kernel_directcp_vec_cpp<Real, VECDIM>(
                &nd, &ndim, &digits, &scale, &center, &d2max, r_src.data_handle(), &nsrc, charges.data_handle(),
                r_trg[0].data(), r_trg[1].data(), r_trg[2].data(), &ntrg, u.data_handle(), &thresh2);
        }
    case dmk_ikernel::DMK_SQRT_LAPLACE:
        throw std::runtime_error("SQRT Laplace kernel not implementedD");
    }
}

template void direct_eval<float, 2>(dmk_ikernel ikernel, const ndview<const float, 2> &r_src,
                                    const std::array<std::span<const float>, 2> &r_trg,
                                    const ndview<const float, 2> &charges, const ndview<const float, 1> &coeffs,
                                    const double *kernel_params, float scale, float center, float d2max,
                                    const ndview<float, 2> &u);
template void direct_eval<float, 3>(dmk_ikernel ikernel, const ndview<const float, 2> &r_src,
                                    const std::array<std::span<const float>, 3> &r_trg,
                                    const ndview<const float, 2> &charges, const ndview<const float, 1> &coeffs,
                                    const double *kernel_params, float scale, float center, float d2max,
                                    const ndview<float, 2> &u);
template void direct_eval<double, 2>(dmk_ikernel ikernel, const ndview<const double, 2> &r_src,
                                     const std::array<std::span<const double>, 2> &r_trg,
                                     const ndview<const double, 2> &charges, const ndview<const double, 1> &coeffs,
                                     const double *kernel_params, double scale, double center, double d2max,
                                     const ndview<double, 2> &u);
template void direct_eval<double, 3>(dmk_ikernel ikernel, const ndview<const double, 2> &r_src,
                                     const std::array<std::span<const double>, 3> &r_trg,
                                     const ndview<const double, 2> &charges, const ndview<const double, 1> &coeffs,
                                     const double *kernel_params, double scale, double center, double d2max,
                                     const ndview<double, 2> &u);
} // namespace dmk
