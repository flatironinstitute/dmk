#include <sctl.hpp>

namespace sctl {
template <class VData, class CType>
VData EvalPolynomial(const VData &x1, const CType &c0, const CType &c1, const CType &c2, const CType &c3,
                     const CType &c4, const CType &c5, const CType &c6, const CType &c7, const CType &c8,
                     const CType &c9, const CType &c10, const CType &c11, const CType &c12, const CType &c13,
                     const CType &c14, const CType &c15, const CType &c16, const CType &c17, const CType &c18,
                     const CType &c19, const CType &c20, const CType &c21) {
    VData x2(mul_intrin<VData>(x1, x1));
    VData x4(mul_intrin<VData>(x2, x2));
    VData x8(mul_intrin<VData>(x4, x4));
    VData x16(mul_intrin<VData>(x8, x8));

    return fma_intrin(
        x16,
        fma_intrin(x4, fma_intrin(x1, set1_intrin<VData>(c21), set1_intrin<VData>(c20)),
                   fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c19), set1_intrin<VData>(c18)),
                              fma_intrin(x1, set1_intrin<VData>(c17), set1_intrin<VData>(c16)))),
        fma_intrin(x8,
                   fma_intrin(x4,
                              fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c15), set1_intrin<VData>(c14)),
                                         fma_intrin(x1, set1_intrin<VData>(c13), set1_intrin<VData>(c12))),
                              fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c11), set1_intrin<VData>(c10)),
                                         fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8)))),
                   fma_intrin(x4,
                              fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)),
                                         fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))),
                              fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)),
                                         fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0))))));
}
}

/* local kernel for the 1/r kernel -- PME (not PSWF) */
template <class Real, sctl::Integer ndim, sctl::Integer digits = -1, 
            sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void l3d_local_kernel_directcp_vec_cpp__rinv_helper(const Real r_cut_sq,
                                                    const Real *sources, 
                                                    const int32_t ns, 
                                                    const Real *charge,
                                                    const Real *xtarg, 
                                                    const Real *ytarg, 
                                                    const Real *ztarg,
                                                    const int32_t nt,
                                                    const Real alpha, 
                                                    Real *offset,
                                                    Real *pot) {
    static constexpr sctl::Integer VecLen = MaxVecLen;
    using Vec = sctl::Vec<Real, VecLen>;

    static constexpr sctl::Integer COORD_DIM = ndim; // ndim[0];
    constexpr sctl::Long nd_ = 1;
    sctl::Long Nsrc = ns; // QUESTION: No array?
    sctl::Long Ntrg = nt; // QUESTION: No array?
    sctl::Long Ntrg_ = ((Ntrg + MaxVecLen - 1) / MaxVecLen) * MaxVecLen;

    alignas(sizeof(Vec)) sctl::StaticArray<Real, 400 * COORD_DIM> buff0;
    alignas(sizeof(Vec)) sctl::StaticArray<Real, 400 * 1> buff1;
    sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
    sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
    if (Ntrg_ > 400) {
        Xt.ReInit(COORD_DIM, Ntrg_);
        Vt.ReInit(nd_, Ntrg_);
    }
    { // Set Xs, Vs, Xt, Vt
        std::memcpy(Xt[0], xtarg, sizeof(Real) * Ntrg);
        std::memcpy(Xt[1], ytarg, sizeof(Real) * Ntrg);
        if constexpr (COORD_DIM > 2)
            std::memcpy(Xt[2], ztarg, sizeof(Real) * Ntrg);
        Vt = 0;
    }

    const Real d2min = 0.0;
    Vec thresh2 = d2min; // QUESTION: No array?
    Vec d2max_vec = r_cut_sq; // QUESTION: No array?
    const Vec alphavec = alpha;
    
    // load charge
    sctl::Matrix<Real> Vs_(Nsrc, nd_, sctl::Ptr2Itr<Real>((Real *)charge, nd_ * Nsrc), false);
    // load the offest vector
    Vec offset_vec[COORD_DIM];
    for (sctl::Integer k = 0; k < COORD_DIM; k++)
        offset_vec[k] = Vec::Load1(&offset[k]);
    
    // load source
    sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real *)sources, COORD_DIM * Nsrc), false);
    for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
        Vec Xtrg[COORD_DIM];
        for (sctl::Integer k = 0; k < COORD_DIM; k++)
            Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);

        // load potential
        Vec Vtrg[nd_];
        for (long i = 0; i < nd_; i++)
            Vtrg[i] = Vec::LoadAligned(&Vt[i][t]);

        for (sctl::Long s = 0; s < Nsrc; s++) {
            Vec dX[COORD_DIM], R2 = Vec::Zero();
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
                dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]) - offset_vec[k];
                R2 += dX[k] * dX[k];
            }

            const auto mask = (R2 > thresh2) & (R2 < d2max_vec);
            if (mask_popcnt_intrin(mask) == 0)
               continue;

            const Vec Rinv = sctl::approx_rsqrt<digits>(R2, mask);

            // // evaluate the erfc() kernel
            // const Vec xtmp = FMA(R2, Rinv, cen_vec) * rsc_vec;
            // const Vec xtmp = R2 * Rinv * alphavec;
            Vec ptmp = sctl::erfc(R2 * Rinv * alphavec);
            // if constexpr (digits <= 4) {
            //     constexpr Real coefs[21] = {9.99986709e-01, -1.12710954e+00, -2.00753066e-02, 5.01288201e-01,
            //                                 -4.08481567e-01,  6.89212929e-01, -1.02479731e+00,  9.18562083e-01,
            //                                 -5.41240270e-01,  2.26164369e-01, -7.01536110e-02,  1.65903972e-02,
            //                                 -3.03508057e-03,  4.32024588e-04, -4.77677386e-05,  4.06403164e-06,
            //                                 -2.61047533e-07,  1.22466686e-08, -3.95958998e-10,  7.88682170e-12,
            //                                 -7.29395441e-14};
            //     ptmp = EvalPolynomial(xtmp.get(), coefs[0], coefs[1], coefs[2], coefs[3], coefs[4], coefs[5], coefs[6], 
            //                             coefs[7], coefs[8], coefs[9], coefs[10], coefs[11], coefs[12], coefs[12], coefs[13],
            //                             coefs[14], coefs[15], coefs[16], coefs[17], coefs[18], coefs[19], coefs[20]);
            // }

            ptmp = ptmp * Rinv;

            for (long i = 0; i < nd_; i++)
                Vtrg[i] += Vec::Load1(&Vs_[s][i]) * ptmp;
        }

        for (long i = 0; i < nd_; i++)
            Vtrg[i].StoreAligned(&Vt[i][t]);
    }

    for (long i = 0; i < Ntrg; i++)
        for (long j = 0; j < nd_; j++)
            pot[i * nd_ + j] += Vt[j][i];
}