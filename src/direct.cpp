#include <dmk/chebychev.hpp>
#include <dmk/direct.hpp>

namespace dmk {

template <typename Real, int DIM>
void yukawa_direct_eval(const ndview<const Real, 2> &r_src, const ndview<const Real, 2> r_trg,
                        const ndview<const Real, 2> &charges, const ndview<const Real, 1> &coeffs, Real lambda,
                        Real scale, Real center, Real d2max, const ndview<Real, 2> &u) {
    const int nsrc = r_src.extent(1);
    const int ntrg = r_trg.extent(0);
    const int nd = u.extent(0);
    const int norder = coeffs.extent(0);
    constexpr Real threshq = 1E-30;

    for (int i_trg = 0; i_trg < ntrg; i_trg++) {
        for (int i_src = 0; i_src < nsrc; i_src++) {
            const Real dx = r_trg(i_trg, 0) - r_src(0, i_src);
            const Real dy = r_trg(i_trg, 1) - r_src(1, i_src);
            Real dd = dx * dx + dy * dy;
            if constexpr (DIM == 3) {
                Real dz = r_trg(i_trg, 2) - r_src(2, i_src);
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
            else if constexpr (DIM == 3)
                dkval = std::exp(-lambda * r) / r;

            const Real factor = dkval + fval;
            for (int i = 0; i < nd; ++i)
                u(i, i_trg) += charges(i, i_src) * factor;
        }
    }
}

template <typename Real, int DIM>
void direct_eval(dmk_ikernel ikernel, const ndview<const Real, 2> &r_src, const ndview<const Real, 2> r_trg,
                 const ndview<const Real, 2> &charges, const ndview<const Real, 1> &coeffs, const Real *kernel_params,
                 Real scale, Real center, Real d2max, const ndview<Real, 2> &u) {
    switch (ikernel) {
    case dmk_ikernel::DMK_YUKAWA:
        yukawa_direct_eval<Real, DIM>(r_src, r_trg, charges, coeffs, *kernel_params, scale, center, d2max, u);
        break;
    case dmk_ikernel::DMK_LAPLACE:
        if constexpr (DIM == 2) {
        }
        if constexpr (DIM == 3) {
        }
        break;
    case dmk_ikernel::DMK_SQRT_LAPLACE:
        if constexpr (DIM == 2) {
        }
        if constexpr (DIM == 3) {
        }
        break;
    }
}

template void direct_eval<double, 2>(dmk_ikernel ikernel, const ndview<const double, 2> &r_src,
                                     const ndview<const double, 2> r_trg, const ndview<const double, 2> &charges,
                                     const ndview<const double, 1> &coeffs, const double *kernel_params, double scale,
                                     double center, double d2max, const ndview<double, 2> &u);
template void direct_eval<double, 3>(dmk_ikernel ikernel, const ndview<const double, 2> &r_src,
                                     const ndview<const double, 2> r_trg, const ndview<const double, 2> &charges,
                                     const ndview<const double, 1> &coeffs, const double *kernel_params, double scale,
                                     double center, double d2max, const ndview<double, 2> &u);
} // namespace dmk
