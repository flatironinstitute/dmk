#ifndef VECTOR_KERNELS_HPP
#define VECTOR_KERNELS_HPP

#ifndef NDEBUG
#define NDEBUG
#endif
#include <sctl.hpp>

namespace dmk {

template <class Real, class VecType, sctl::Integer DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer SCDIM,
          class uKernel, sctl::Integer digits>
struct uKerHelper {
    template <class CtxType>
    static inline void Eval(VecType *vt, const VecType (&dX)[DIM], const Real *vs, const sctl::Integer nd,
                            const CtxType &ctx) {
        VecType M[KDIM0][KDIM1][SCDIM];
        uKernel::template uKerMatrix<digits>(M, dX, ctx);
        for (sctl::Integer i = 0; i < nd; i++) {
            const Real *vs_ = vs + i * SCDIM;
            for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
                VecType *vt_ = vt + (k1 * nd + i) * SCDIM;
                for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) {
                    const VecType vs0(vs_[(k0 * nd) * SCDIM + 0]);
                    vt_[0] = FMA(M[k0][k1][0], vs0, vt_[0]);
                    if (SCDIM == 2) {
                        const VecType vs1(vs_[(k0 * nd) * SCDIM + 1]);
                        vt_[0] = FMA(M[k0][k1][1], -vs1, vt_[0]);
                        vt_[1] = FMA(M[k0][k1][1], vs0, vt_[1]);
                        vt_[1] = FMA(M[k0][k1][0], vs1, vt_[1]);
                    }
                }
            }
        }
    }
    template <sctl::Integer nd, class CtxType>
    static inline void EvalND(VecType *vt, const VecType (&dX)[DIM], const Real *vs, const CtxType &ctx) {
        VecType M[KDIM0][KDIM1][SCDIM];
        uKernel::template uKerMatrix<digits>(M, dX, ctx);
        for (sctl::Integer i = 0; i < nd; i++) {
            const Real *vs_ = vs + i * SCDIM;
            for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
                VecType *vt_ = vt + (k1 * nd + i) * SCDIM;
                for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) {
                    const VecType vs0(vs_[(k0 * nd) * SCDIM + 0]);
                    vt_[0] = FMA(M[k0][k1][0], vs0, vt_[0]);
                    if (SCDIM == 2) {
                        const VecType vs1(vs_[(k0 * nd) * SCDIM + 1]);
                        vt_[0] = FMA(M[k0][k1][1], -vs1, vt_[0]);
                        vt_[1] = FMA(M[k0][k1][1], vs0, vt_[1]);
                        vt_[1] = FMA(M[k0][k1][0], vs1, vt_[1]);
                    }
                }
            }
        }
    }
};

template <class uKernel>
class GenericKernel : public uKernel {
    static constexpr sctl::Integer VecLen = uKernel::VecLen;
    using VecType = typename uKernel::VecType;
    using Real = typename uKernel::RealType;

    template <sctl::Integer K0, sctl::Integer K1, sctl::Integer Q, sctl::Integer D, class... T>
    static constexpr sctl::Integer get_DIM(void (*uKer)(VecType (&M)[K0][K1][Q], const VecType (&r)[D], T... args)) {
        return D;
    }
    template <sctl::Integer K0, sctl::Integer K1, sctl::Integer Q, sctl::Integer D, class... T>
    static constexpr sctl::Integer get_SCDIM(void (*uKer)(VecType (&M)[K0][K1][Q], const VecType (&r)[D], T... args)) {
        return Q;
    }
    template <sctl::Integer K0, sctl::Integer K1, sctl::Integer Q, sctl::Integer D, class... T>
    static constexpr sctl::Integer get_KDIM0(void (*uKer)(VecType (&M)[K0][K1][Q], const VecType (&r)[D], T... args)) {
        return K0;
    }
    template <sctl::Integer K0, sctl::Integer K1, sctl::Integer Q, sctl::Integer D, class... T>
    static constexpr sctl::Integer get_KDIM1(void (*uKer)(VecType (&M)[K0][K1][Q], const VecType (&r)[D], T... args)) {
        return K1;
    }

    static constexpr sctl::Integer DIM = get_DIM(uKernel::template uKerMatrix<0, GenericKernel>);
    static constexpr sctl::Integer SCDIM = get_SCDIM(uKernel::template uKerMatrix<0, GenericKernel>);
    static constexpr sctl::Integer KDIM0 = get_KDIM0(uKernel::template uKerMatrix<0, GenericKernel>);
    static constexpr sctl::Integer KDIM1 = get_KDIM1(uKernel::template uKerMatrix<0, GenericKernel>);

  public:
    GenericKernel() : ctx_ptr(this) {}

    static constexpr sctl::Integer CoordDim() { return DIM; }
    static constexpr sctl::Integer SrcDim() { return KDIM0 * SCDIM; }
    static constexpr sctl::Integer TrgDim() { return KDIM1 * SCDIM; }

    template <bool enable_openmp = false, sctl::Integer digits = -1>
    void Eval(sctl::Vector<sctl::Vector<Real>> &v_trg_, const sctl::Vector<Real> &r_trg,
              const sctl::Vector<Real> &r_src, const sctl::Vector<sctl::Vector<Real>> &v_src_,
              const sctl::Integer nd) const {
        if (nd == 1)
            EvalHelper<enable_openmp, digits, 1>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 2)
            EvalHelper<enable_openmp, digits, 2>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 3)
            EvalHelper<enable_openmp, digits, 3>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 4)
            EvalHelper<enable_openmp, digits, 4>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 5)
            EvalHelper<enable_openmp, digits, 5>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 6)
            EvalHelper<enable_openmp, digits, 6>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 7)
            EvalHelper<enable_openmp, digits, 7>(v_trg_, r_trg, r_src, v_src_, nd);
        else if (nd == 8)
            EvalHelper<enable_openmp, digits, 8>(v_trg_, r_trg, r_src, v_src_, nd);
        else
            EvalHelper<enable_openmp, digits, 0>(v_trg_, r_trg, r_src, v_src_, nd);
    }

  private:
    template <bool enable_openmp = false, sctl::Integer digits = -1, sctl::Integer ND = 0>
    void EvalHelper(sctl::Vector<sctl::Vector<Real>> &v_trg_, const sctl::Vector<Real> &r_trg,
                    const sctl::Vector<Real> &r_src, const sctl::Vector<sctl::Vector<Real>> &v_src_,
                    const sctl::Integer nd) const {
        static constexpr sctl::Integer digits_ =
            (digits == -1 ? (sctl::Integer)(sctl::TypeTraits<Real>::SigBits * 0.3010299957) : digits);
        auto uKerEval = [this](VecType *vt, const VecType(&dX)[DIM], const Real *vs, const sctl::Integer nd) {
            if (ND > 0)
                uKerHelper<Real, VecType, DIM, KDIM0, KDIM1, SCDIM, uKernel, digits_>::template EvalND<ND>(vt, dX, vs,
                                                                                                           *this);
            else
                uKerHelper<Real, VecType, DIM, KDIM0, KDIM1, SCDIM, uKernel, digits_>::Eval(vt, dX, vs, nd, *this);
        };

        const sctl::Long Ns = r_src.Dim() / DIM;
        const sctl::Long Nt = r_trg.Dim() / DIM;
        SCTL_ASSERT(r_trg.Dim() == Nt * DIM);
        SCTL_ASSERT(r_src.Dim() == Ns * DIM);

        sctl::Vector<sctl::Long> src_cnt(v_src_.Dim()), src_dsp(v_src_.Dim());
        src_dsp = 0;
        sctl::Vector<sctl::Long> trg_cnt(v_trg_.Dim()), trg_dsp(v_trg_.Dim());
        trg_dsp = 0;
        for (sctl::Integer i = 0; i < trg_cnt.Dim(); i++) {
            trg_cnt[i] = v_trg_[i].Dim() / Nt;
            trg_dsp[i] = (i ? trg_dsp[i - 1] + trg_cnt[i - 1] : 0);
        }
        for (sctl::Integer i = 0; i < src_cnt.Dim(); i++) {
            src_cnt[i] = v_src_[i].Dim() / Ns;
            src_dsp[i] = (i ? src_dsp[i - 1] + src_cnt[i - 1] : 0);
        }
        SCTL_ASSERT(src_cnt[src_cnt.Dim() - 1] + src_dsp[src_dsp.Dim() - 1] == SrcDim() * nd);
        SCTL_ASSERT(trg_cnt[trg_cnt.Dim() - 1] + trg_dsp[trg_dsp.Dim() - 1] == TrgDim() * nd);

        sctl::Vector<Real> v_src(Ns * SrcDim() * nd);
        for (sctl::Integer j = 0; j < src_cnt.Dim(); j++) {
            const sctl::Integer src_cnt_ = src_cnt[j];
            const sctl::Integer src_dsp_ = src_dsp[j];
            for (sctl::Integer k = 0; k < src_cnt_; k++) {
                for (sctl::Long i = 0; i < Ns; i++) {
                    v_src[i * SrcDim() * nd + src_dsp_ + k] = v_src_[j][i * src_cnt_ + k];
                }
            }
        }

        const sctl::Long NNt = ((Nt + VecLen - 1) / VecLen) * VecLen;
        {
            const sctl::Matrix<Real> Xs_(Ns, DIM, (sctl::Iterator<Real>)r_src.begin(), false);
            const sctl::Matrix<Real> Vs_(Ns, SrcDim() * nd, (sctl::Iterator<Real>)v_src.begin(), false);

            sctl::Matrix<Real> Xt_(DIM, NNt), Vt_(TrgDim() * nd, NNt);
            for (sctl::Long k = 0; k < DIM; k++) { // Set Xt_
                for (sctl::Long i = 0; i < Nt; i++) {
                    Xt_[k][i] = r_trg[i * DIM + k];
                }
                for (sctl::Long i = Nt; i < NNt; i++) {
                    Xt_[k][i] = 0;
                }
            }
            if (enable_openmp) { // Compute Vt_
#pragma omp parallel for schedule(static)
                for (sctl::Long t = 0; t < NNt; t += VecLen) {
                    VecType xt[DIM], vt[TrgDim() * nd];
                    for (sctl::Integer k = 0; k < TrgDim() * nd; k++)
                        vt[k] = VecType::Zero();
                    for (sctl::Integer k = 0; k < DIM; k++)
                        xt[k] = VecType::LoadAligned(&Xt_[k][t]);

                    for (sctl::Long s = 0; s < Ns; s++) {
                        VecType dX[DIM];
                        for (sctl::Integer k = 0; k < DIM; k++)
                            dX[k] = xt[k] - Xs_[s][k];
                        uKerEval(vt, dX, &Vs_[s][0], nd);
                    }
                    for (sctl::Integer k = 0; k < TrgDim() * nd; k++)
                        vt[k].StoreAligned(&Vt_[k][t]);
                }
            } else {
                for (sctl::Long t = 0; t < NNt; t += VecLen) {
                    VecType xt[DIM], vt[TrgDim() * nd];
                    for (sctl::Integer k = 0; k < TrgDim() * nd; k++)
                        vt[k] = VecType::Zero();
                    for (sctl::Integer k = 0; k < DIM; k++)
                        xt[k] = VecType::LoadAligned(&Xt_[k][t]);

                    for (sctl::Long s = 0; s < Ns; s++) {
                        VecType dX[DIM];
                        for (sctl::Integer k = 0; k < DIM; k++)
                            dX[k] = xt[k] - Xs_[s][k];
                        uKerEval(vt, dX, &Vs_[s][0], nd);
                    }
                    for (sctl::Integer k = 0; k < TrgDim() * nd; k++)
                        vt[k].StoreAligned(&Vt_[k][t]);
                }
            }

            for (sctl::Integer j = 0; j < trg_cnt.Dim(); j++) {
                const sctl::Integer trg_cnt_ = trg_cnt[j];
                const sctl::Integer trg_dsp_ = trg_dsp[j];
                for (sctl::Long i = 0; i < Nt; i++) {
                    for (sctl::Integer k = 0; k < trg_cnt_; k++) {
                        v_trg_[j][i * trg_cnt_ + k] += Vt_[trg_dsp_ + k][i] * uKernel::uKerScaleFactor();
                    }
                }
            }
        }
    }

    void *ctx_ptr;
};

template <class Real, sctl::Integer VecLen_, sctl::Integer chrg, sctl::Integer dipo, sctl::Integer poten,
          sctl::Integer grad>
struct Helmholtz3D {
    static constexpr sctl::Integer VecLen = VecLen_;
    using VecType = sctl::Vec<Real, VecLen>;
    using RealType = Real;

    VecType thresh2;
    VecType zk[2];

    static constexpr Real uKerScaleFactor() { return 1; }
    template <sctl::Integer digits, class CtxType>
    static inline void uKerMatrix(VecType (&M)[chrg + dipo][poten + grad][2], const VecType (&dX)[3],
                                  const CtxType &ctx) {
        using RealType = typename VecType::ScalarType;
        static constexpr sctl::Integer COORD_DIM = 3;

        const VecType &thresh2 = ctx.thresh2;
        const VecType(&zk)[2] = ctx.zk;

        const VecType R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const VecType Rinv = sctl::approx_rsqrt<digits>(R2, (R2 > thresh2));
        const VecType Rinv2 = Rinv * Rinv;

        const VecType R = R2 * Rinv;
        const VecType izkR[2] = {-zk[1] * R, zk[0] * R};

        VecType sin_izkR, cos_izkR;
        sctl::approx_sincos<digits>(sin_izkR, cos_izkR, izkR[1]);
        const VecType exp_izkR = sctl::approx_exp<digits>(izkR[0]);

        // exp(ikr)/r
        const VecType G0 = cos_izkR * exp_izkR * Rinv;
        const VecType G1 = sin_izkR * exp_izkR * Rinv;

        // (1-ikr)*exp(ikr)/r^3
        const VecType H0 = ((izkR[0] - (RealType)1) * G0 - izkR[1] * G1) * Rinv2;
        const VecType H1 = ((izkR[0] - (RealType)1) * G1 + izkR[1] * G0) * Rinv2;

        const VecType tmp0 = (-3.0) * (Rinv * zk[1] + Rinv2) - zk[1] * zk[1] + zk[0] * zk[0];
        const VecType tmp1 = (3.0) * Rinv * zk[0] - zk[0] * zk[1] * (-2.0);
        const VecType J0 = (G0 * tmp0 - G1 * tmp1) * Rinv2;
        const VecType J1 = (G1 * tmp0 + G0 * tmp1) * Rinv2;

        if (chrg && poten) { // charge potential
            M[0][0][0] = G0;
            M[0][0][1] = G1;
        }

        if (chrg && grad) { // charge gradient
            for (sctl::Integer i = 0; i < COORD_DIM; i++) {
                M[0][poten + i][0] = H0 * dX[i];
                M[0][poten + i][1] = H1 * dX[i];
            }
        }

        if (dipo && poten) { // dipole potential
            for (sctl::Integer i = 0; i < COORD_DIM; i++) {
                M[chrg + i][0][0] = -H0 * dX[i];
                M[chrg + i][0][1] = -H1 * dX[i];
            }
        }

        if (dipo && grad) { // dipole gradient
            for (sctl::Integer i = 0; i < COORD_DIM; i++) {
                const VecType J0_dXi = J0 * dX[i];
                const VecType J1_dXi = J1 * dX[i];
                for (sctl::Integer j = 0; j < COORD_DIM; j++) {
                    M[chrg + i][poten + j][0] = (i == j ? J0_dXi * dX[j] - H0 : J0_dXi * dX[j]);
                    M[chrg + i][poten + j][1] = (i == j ? J1_dXi * dX[j] - H1 : J1_dXi * dX[j]);
                }
            }
        }
    }
};

template <class Real, sctl::Integer VecLen, sctl::Integer chrg, sctl::Integer dipo, sctl::Integer poten,
          sctl::Integer grad>
static void EvalHelmholtz(sctl::Vector<sctl::Vector<Real>> &v_trg, const sctl::Vector<Real> &r_trg,
                          const sctl::Vector<Real> &r_src, const sctl::Vector<sctl::Vector<Real>> &v_src,
                          const sctl::Integer nd, const Real *zk, const Real thresh, const sctl::Integer digits) {
    sctl::GenericKernel<Helmholtz3D<Real, VecLen, chrg, dipo, poten, grad>> ker;
    ker.thresh2 = thresh * thresh;
    ker.zk[0] = zk[0];
    ker.zk[1] = zk[1];

    if (digits < 0)
        ker.template Eval<true, -1>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 3)
        ker.template Eval<true, 3>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 6)
        ker.template Eval<true, 6>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 9)
        ker.template Eval<true, 9>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 12)
        ker.template Eval<true, 12>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 15)
        ker.template Eval<true, 15>(v_trg, r_trg, r_src, v_src, nd);
    else
        ker.template Eval<true, -1>(v_trg, r_trg, r_src, v_src, nd);
}

template <class Real, sctl::Integer VecLen_, sctl::Integer chrg, sctl::Integer dipo, sctl::Integer poten,
          sctl::Integer grad>
struct Laplace3D {
    static constexpr sctl::Integer VecLen = VecLen_;
    using VecType = sctl::Vec<Real, VecLen>;
    using RealType = Real;

    VecType thresh2;

    static constexpr Real uKerScaleFactor() { return 1; }
    template <sctl::Integer digits, class CtxType>
    static inline void uKerMatrix(VecType (&M)[chrg + dipo][poten + grad][1], const VecType (&dX)[3],
                                  const CtxType &ctx) {
        using RealType = typename VecType::ScalarType;
        static constexpr sctl::Integer COORD_DIM = 3;

        const VecType &thresh2 = ctx.thresh2;

        const VecType R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const VecType Rinv = sctl::approx_rsqrt<digits>(R2, (R2 > thresh2));
        const VecType Rinv2 = Rinv * Rinv;
        const VecType Rinv3 = Rinv * Rinv2;

        if (chrg && poten) { // charge potential
            M[0][0][0] = Rinv;
        }

        if (chrg && grad) { // charge gradient
            for (sctl::Integer i = 0; i < COORD_DIM; i++) {
                M[0][poten + i][0] = -Rinv3 * dX[i];
            }
        }

        if (dipo && poten) { // dipole potential
            for (sctl::Integer i = 0; i < COORD_DIM; i++) {
                M[chrg + i][0][0] = Rinv3 * dX[i];
            }
        }

        if (dipo && grad) { // dipole gradient
            const VecType J0 = Rinv3 * Rinv2 * (RealType)(-3);
            for (sctl::Integer i = 0; i < COORD_DIM; i++) {
                const VecType J0_dXi = J0 * dX[i];
                for (sctl::Integer j = 0; j < COORD_DIM; j++) {
                    M[chrg + i][poten + j][0] = (i == j ? J0_dXi * dX[j] + Rinv3 : J0_dXi * dX[j]);
                }
            }
        }
    }
};

template <class Real, sctl::Integer VecLen, sctl::Integer chrg, sctl::Integer dipo, sctl::Integer poten,
          sctl::Integer grad>
static void EvalLaplace(sctl::Vector<sctl::Vector<Real>> &v_trg, const sctl::Vector<Real> &r_trg,
                        const sctl::Vector<Real> &r_src, const sctl::Vector<sctl::Vector<Real>> &v_src,
                        const sctl::Integer nd, const Real thresh, const sctl::Integer digits) {
    sctl::GenericKernel<Laplace3D<Real, VecLen, chrg, dipo, poten, grad>> ker;
    ker.thresh2 = thresh * thresh;

    if (digits < 0)
        ker.template Eval<true, -1>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 3)
        ker.template Eval<true, 3>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 6)
        ker.template Eval<true, 6>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 9)
        ker.template Eval<true, 9>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 12)
        ker.template Eval<true, 12>(v_trg, r_trg, r_src, v_src, nd);
    else if (digits <= 15)
        ker.template Eval<true, 15>(v_trg, r_trg, r_src, v_src, nd);
    else
        ker.template Eval<true, -1>(v_trg, r_trg, r_src, v_src, nd);
}

struct Laplace3DLocalPSWF {
    template <class Real, sctl::Integer digits>
    static constexpr auto get_coeffs = []() {
        if constexpr (digits <= 3) {
            return std::array<Real, 7>{1.627823522210361e-01,  -4.553645597616490e-01, 4.171687104204163e-01,
                                       -7.073638602709915e-02, -8.957845614474928e-02, 2.617986644718201e-02,
                                       9.633427876507601e-03};
        } else if constexpr (digits <= 6) {
            return std::array<Real, 13>{5.482525801351582e-02,  -2.616592110444692e-01, 4.862652666337138e-01,
                                        -3.894296348642919e-01, 1.638587821812791e-02,  1.870328434198821e-01,
                                        -8.714171086568978e-02, -3.927020727017803e-02, 3.728187607052319e-02,
                                        3.153734425831139e-03,  -8.651313377285847e-03, 1.725110090795567e-04,
                                        1.034762385284044e-03};
        } else if constexpr (digits <= 9) {
            return std::array<Real, 19>{
                1.835718730962269e-02,  -1.258015846164503e-01, 3.609487248584408e-01,  -5.314579651112283e-01,
                3.447559412892380e-01,  9.664692318551721e-02,  -3.124274531849053e-01, 1.322460720579388e-01,
                9.773007866584822e-02,  -1.021958831082768e-01, -3.812847450976566e-03, 3.858117355875043e-02,
                -8.728545924521301e-03, -9.401196355382909e-03, 4.024549377076924e-03,  1.512806105865091e-03,
                -9.576734877247042e-04, -1.303457547418901e-04, 1.100385844683190e-04};
        } else if constexpr (digits <= 12) {
            return std::array<Real, 25>{
                6.262472576363448e-03,  -5.605742936112479e-02, 2.185890864792949e-01,  -4.717350304955679e-01,
                5.669680214206270e-01,  -2.511606878849214e-01, -2.744523658778361e-01, 4.582527599363415e-01,
                -1.397724810121539e-01, -2.131762135835757e-01, 1.995489373508990e-01,  1.793390341864239e-02,
                -1.035055132403432e-01, 3.035606831075176e-02,  3.153931762550532e-02,  -2.033178627450288e-02,
                -5.406682731236552e-03, 7.543645573618463e-03,  1.437788047407851e-05,  -1.928370882351732e-03,
                2.891658777328665e-04,  3.332996162099811e-04,  -8.397699195938912e-05, -3.015837377517983e-05,
                9.640642701924662e-06};
        }
    };

    template <class VecType>
    struct Context {
        VecType thresh2;
        VecType d2max_vec;
        VecType rsc_vec;
        VecType cen_vec;
    };

    static const std::string &Name() {
        static const std::string name = "Laplace3D-LocalPSWF";
        return name;
    };
    static constexpr sctl::Integer FLOPS() {
        // FIXME should be 'digits' dependent
        return 1;
    }
    template <class Real>
    static constexpr Real uKerScaleFactor() {
        return 1;
    }

    template <sctl::Integer digits, class VecType>
    static void uKerMatrix(VecType (&u)[1][1], const VecType (&dX)[3], const void *ctx_ptr) {
        //        static_assert(digits <= 12, "Laplace3DLocalPSWF only supports up to 12 digits.");
        using RealType = typename VecType::ScalarType;
        static constexpr sctl::Integer COORD_DIM = 3;

        const Context<VecType> &ker = *(Context<VecType> *)(ctx_ptr);

        const auto &thresh2 = ker.thresh2;
        const VecType &d2max_vec = ker.d2max_vec;
        const VecType &rsc_vec = ker.rsc_vec;
        const VecType &cen_vec = ker.cen_vec;
        constexpr auto coefs = get_coeffs<typename VecType::ScalarType, digits>();

        const VecType R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const auto mask = (R2 > thresh2) & (R2 < d2max_vec);
        // Branch misses not worth it for lower precisions. Always do the polynomial expansion then
        if constexpr (digits >= 6) {
            if (!mask_popcnt_intrin(mask)) {
                u[0][0] = VecType::Zero();
                return;
            }
        }
        const VecType Rinv = sctl::approx_rsqrt<digits>(R2, mask);

        // evaluate the PSWF kernel
        const VecType xtmp = FMA(R2, Rinv, cen_vec) * rsc_vec;
        const VecType poly = sctl::EvalPolynomial(xtmp.get(), coefs);

        u[0][0] = poly * Rinv;
    }
};

template <class Real, sctl::Integer ndim, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void EvalLaplaceLocalPSWF(const int32_t *nd, const int32_t *digits, const Real *rsc, const Real *cen, const Real *d2max,
                          const Real *sources, const int32_t *ns, const Real *charge, const Real *rtrg,
                          const int32_t *nt, Real *pot, const Real *thresh2) {
    sctl::GenericKernel<Laplace3DLocalPSWF> ker;
    Laplace3DLocalPSWF::Context<sctl::Vec<Real, MaxVecLen>> ctx;

    ctx.thresh2 = thresh2[0];
    ctx.d2max_vec = d2max[0];
    ctx.rsc_vec = rsc[0];
    ctx.cen_vec = cen[0];

    ker.SetCtxPtr(&ctx);

    sctl::Vector<Real> v_trg(nt[0], pot, false);
    sctl::Vector<Real> r_trg(ndim * nt[0], const_cast<Real *>(rtrg), false);
    sctl::Vector<Real> r_src(ndim * ns[0], const_cast<Real *>(sources), false);
    sctl::Vector<Real> v_src(ns[0], const_cast<Real *>(charge), false);
    sctl::Vector<Real> n_src;

    if (*digits < 0)
        throw std::runtime_error("EvalLaplaceLocalPSFW only supports positive digits up to 12.");
    else if (*digits <= 3)
        ker.template Eval<Real, false, 3>(v_trg, r_trg, r_src, n_src, v_src);
    else if (*digits <= 6)
        ker.template Eval<Real, false, 6>(v_trg, r_trg, r_src, n_src, v_src);
    else if (*digits <= 9)
        ker.template Eval<Real, false, 9>(v_trg, r_trg, r_src, n_src, v_src);
    else if (*digits <= 12)
        ker.template Eval<Real, false, 12>(v_trg, r_trg, r_src, n_src, v_src);
    else
        throw std::runtime_error("EvalLaplaceLocalPSFW only supports up to 12 digits.");
}

template <int n_coeffs_>
struct Laplace3DLocalUnknownCoeffs {
    static constexpr int n_coeffs = n_coeffs_;

    template <class VecType>
    struct Context {
        VecType thresh2;
        VecType d2max_vec;
        VecType rsc_vec;
        VecType cen_vec;
        std::array<typename VecType::ScalarType, n_coeffs> coefs;
    };

    static const std::string &Name() {
        static const std::string name = "Laplace3D-LocalPSWF";
        return name;
    };
    static constexpr sctl::Integer FLOPS() {
        // FIXME should be 'digits' and n_coeffs_ dependent
        return 1;
    }
    template <class Real>
    static constexpr Real uKerScaleFactor() {
        return 1;
    }

    template <sctl::Integer digits, class VecType>
    static void uKerMatrix(VecType (&u)[1][1], const VecType (&dX)[3], const void *ctx_ptr) {
        const Context<VecType> &ctx = *(Context<VecType> *)(ctx_ptr);

        const VecType R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
        const auto mask = (R2 > ctx.thresh2) & (R2 < ctx.d2max_vec);

        // Branch misses not worth it for lower precisions. Always do the polynomial expansion then
        if constexpr (digits >= 6) {
            // Early exit if no targets are in range
            if (!mask_popcnt_intrin(mask)) {
                u[0][0] = VecType::Zero();
                return;
            }
        }
        const VecType Rinv = sctl::approx_rsqrt<digits>(R2, mask);

        // evaluate the PSWF kernel
        const VecType xtmp = FMA(R2, Rinv, ctx.cen_vec) * ctx.rsc_vec;
        VecType poly = VecType::Zero();
        for (int i = n_coeffs - 1; i >= 0; --i)
            poly = FMA(poly, xtmp, VecType::Load1(&ctx.coefs[i]));

        u[0][0] = poly * Rinv;
    }
};

template <class Real, int n_coeffs, int digits, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void EvalLaplaceLocalUnknownCoeffs(int32_t nd, Real rsc, const Real cen, Real d2max, Real thresh2,
                                   const std::array<Real, n_coeffs> &coeffs, const sctl::Vector<Real> &r_src,
                                   const sctl::Vector<Real> &charge, const sctl::Vector<Real> &r_trg,
                                   sctl::Vector<Real> &pot) {
    using KerType = Laplace3DLocalUnknownCoeffs<n_coeffs>;
    using VecType = sctl::Vec<Real, MaxVecLen>;
    typename KerType::template Context<VecType> ctx;
    sctl::GenericKernel<KerType> ker;

    ctx.thresh2 = thresh2;
    ctx.d2max_vec = d2max;
    ctx.rsc_vec = rsc;
    ctx.cen_vec = cen;
    ctx.coefs = coeffs;
    ker.SetCtxPtr(&ctx);

    ker.template Eval<Real, false, digits>(pot, r_trg, r_src, {}, charge);
}

template <typename Real, int N>
std::array<Real, N> vec2arr(const sctl::Vector<Real> &vec) {
    std::array<Real, N> arr{};
    std::copy_n(&vec[0], N, arr.data());
    return arr;
}

template <int digits, typename Real, sctl::Integer MaxVecLen, int... N>
void EvalLaplaceLocalUnknownCoeffs(int32_t nd, Real rsc, const Real cen, Real d2max, Real thresh2,
                                   const sctl::Vector<Real> &coeffs, const sctl::Vector<Real> &r_src,
                                   const sctl::Vector<Real> &charge, const sctl::Vector<Real> &r_trg,
                                   sctl::Vector<Real> &pot) {
    bool matched = false;
    const int n_coeffs = coeffs.Dim();
    ((n_coeffs == N ? (EvalLaplaceLocalUnknownCoeffs<Real, N, digits, MaxVecLen>(
                           nd, rsc, cen, d2max, thresh2, vec2arr<Real, N>(coeffs), r_src, charge, r_trg, pot),
                       matched = true)
                    : false),
     ...);
    if (!matched)
        throw std::runtime_error("Invalid n_coeffs for digits");
}

template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void EvalLaplaceLocalUnknownCoeffs(int32_t nd, int digits, Real rsc, const Real cen, Real d2max, Real thresh2,
                                   const sctl::Vector<Real> &coeffs, const sctl::Vector<Real> &r_src,
                                   const sctl::Vector<Real> &charge, const sctl::Vector<Real> &r_trg,
                                   sctl::Vector<Real> &pot) {
    switch (digits) {
    case 2:
        EvalLaplaceLocalUnknownCoeffs<2, Real, MaxVecLen, 3, 4, 5, 6, 7, 8, 9, 10>(nd, rsc, cen, d2max, thresh2, coeffs,
                                                                                   r_src, charge, r_trg, pot);
        break;
    case 3:
        EvalLaplaceLocalUnknownCoeffs<3, Real, MaxVecLen, 5, 6, 7, 8, 9, 10, 11, 12>(nd, rsc, cen, d2max, thresh2,
                                                                                     coeffs, r_src, charge, r_trg, pot);
        break;
    case 4:
        EvalLaplaceLocalUnknownCoeffs<4, Real, MaxVecLen, 7, 8, 9, 10, 11, 12, 13, 14>(
            nd, rsc, cen, d2max, thresh2, coeffs, r_src, charge, r_trg, pot);
        break;
    case 5:
        EvalLaplaceLocalUnknownCoeffs<5, Real, MaxVecLen, 9, 10, 11, 12, 13, 14, 15, 16>(
            nd, rsc, cen, d2max, thresh2, coeffs, r_src, charge, r_trg, pot);
        break;
    case 6:
        EvalLaplaceLocalUnknownCoeffs<6, Real, MaxVecLen, 11, 12, 13, 14, 15, 16, 17, 18>(
            nd, rsc, cen, d2max, thresh2, coeffs, r_src, charge, r_trg, pot);
        break;
    case 7:
        EvalLaplaceLocalUnknownCoeffs<7, Real, MaxVecLen, 13, 14, 15, 16, 17, 18, 19, 20>(
            nd, rsc, cen, d2max, thresh2, coeffs, r_src, charge, r_trg, pot);
        break;
    case 8:
        EvalLaplaceLocalUnknownCoeffs<8, Real, MaxVecLen, 15, 16, 17, 18, 19, 20, 21, 22>(
            nd, rsc, cen, d2max, thresh2, coeffs, r_src, charge, r_trg, pot);
        break;
    case 9:
        EvalLaplaceLocalUnknownCoeffs<9, Real, MaxVecLen, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26>(
            nd, rsc, cen, d2max, thresh2, coeffs, r_src, charge, r_trg, pot);
        break;
    case 10:
        EvalLaplaceLocalUnknownCoeffs<10, Real, MaxVecLen, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28>(
            nd, rsc, cen, d2max, thresh2, coeffs, r_src, charge, r_trg, pot);
        break;
    case 11:
        EvalLaplaceLocalUnknownCoeffs<11, Real, MaxVecLen, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30>(
            nd, rsc, cen, d2max, thresh2, coeffs, r_src, charge, r_trg, pot);
        break;
    case 12:
        EvalLaplaceLocalUnknownCoeffs<12, Real, MaxVecLen, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32>(
            nd, rsc, cen, d2max, thresh2, coeffs, r_src, charge, r_trg, pot);
        break;
    default:
        throw std::runtime_error("Unsupported digits: " + std::to_string(digits));
    }
}

/* local kernel for the 1/r kernel */
template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>(), sctl::Integer digits = -1,
          sctl::Integer ndim>
void l3d_local_kernel_directcp_vec_cpp_helper(const int32_t *nd, const Real *rsc, const Real *cen, const Real *d2max,
                                              const Real *sources, const int32_t *ns, const Real *charge,
                                              const Real *xtarg, const Real *ytarg, const Real *ztarg,
                                              const int32_t *nt, Real *pot, const Real *thresh) {
    static constexpr sctl::Integer VecLen = MaxVecLen;
    using Vec = sctl::Vec<Real, VecLen>;

    static constexpr sctl::Integer COORD_DIM = ndim; // ndim[0];
    constexpr sctl::Long nd_ = 1;                    // nd[0];
    sctl::Long Nsrc = ns[0];
    sctl::Long Ntrg = nt[0];
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

    static constexpr auto coefs = Laplace3DLocalPSWF::get_coeffs<Real, digits>();

    Vec thresh2 = thresh[0];
    Vec d2max_vec = d2max[0];
    Vec rsc_vec = rsc[0];
    Vec cen_vec = cen[0];
    // load charge
    sctl::Matrix<Real> Vs_(Nsrc, nd_, sctl::Ptr2Itr<Real>((Real *)charge, nd_ * Nsrc), false);
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
                dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
                R2 += dX[k] * dX[k];
            }

            const auto mask = (R2 > thresh2) & (R2 < d2max_vec);
            if constexpr (digits >= 6) {
                if (!mask_popcnt_intrin(mask))
                    continue;
            }

            const Vec Rinv = sctl::approx_rsqrt<digits>(R2, mask);

            // evaluate the PSWF kernel
            const Vec xtmp = FMA(R2, Rinv, cen_vec) * rsc_vec;
            Vec ptmp = sctl::EvalPolynomial(xtmp.get(), coefs);
            ptmp *= Rinv;

            for (long i = 0; i < nd_; i++)
                Vtrg[i] += Vec::Load1(&Vs_[s][i]) * ptmp;
        }

        for (long i = 0; i < nd_; i++)
            Vtrg[i].StoreAligned(&Vt[i][t]);
    }

    for (long i = 0; i < Ntrg; i++)
        for (long j = 0; j < nd_; j++)
            pot[i * nd_ + j] += Vt[j][i];

    constexpr auto horner_flops = [](int n_coeffs) { return 3 * n_coeffs - 1; };
    constexpr auto distance_flops = []() { return 3 * 3 + 3 + 3; };
    constexpr auto inner_loop_flops = [horner_flops, distance_flops]() {
        return horner_flops(coefs.size()) + distance_flops() + 2;
    };
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, inner_loop_flops() * Nsrc * Ntrg + Ntrg);
}

template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void l3d_local_kernel_directcp_vec_cpp(const int32_t *nd, const int32_t *ndim, const int32_t *digits, const Real *rsc,
                                       const Real *cen, const Real *d2max, const Real *sources, const int32_t *ns,
                                       const Real *charge, const Real *xtarg, const Real *ytarg, const Real *ztarg,
                                       const int32_t *nt, Real *pot, const Real *thresh) {
    if (ndim[0] == 3) {
        if (digits[0] <= 3)
            l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 3, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                            xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 6)
            l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 6, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                            xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 9)
            l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 9, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                            xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 12)
            l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 12, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
        else
            l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, -1, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
    }
    if (ndim[0] == 2) {
        if (digits[0] <= 3)
            l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 3, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                            xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 6)
            l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 6, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                            xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 9)
            l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 9, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                            xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 12)
            l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 12, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
        else
            l3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, -1, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
    }
}

template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>(), sctl::Integer digits = -1,
          sctl::Integer ndim>
void sl3d_local_kernel_directcp_vec_cpp_helper(const int32_t *nd, const Real *rsc, const Real *cen, const Real *d2max,
                                               const Real *sources, const int32_t *ns, const Real *charge,
                                               const Real *xtarg, const Real *ytarg, const Real *ztarg,
                                               const int32_t *nt, Real *pot, const Real *thresh) {
    static constexpr sctl::Integer COORD_DIM = ndim; // ndim[0];
    constexpr sctl::Long nd_ = 1;                    // nd[0];
    static constexpr sctl::Integer VecLen = MaxVecLen;
    using Vec = sctl::Vec<Real, VecLen>;

    sctl::Long Nsrc = ns[0];
    sctl::Long Ntrg = nt[0];
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
        if (COORD_DIM > 2) {
            std::memcpy(Xt[2], ztarg, sizeof(Real) * Ntrg);
        }
        Vt = 0;
    }

    constexpr auto coefs = []() {
        if constexpr (digits <= 3) {
            return std::array<Real, 6>{1.072550277328770e-01,  -2.940116755817858e-01, 3.195680735052503e-01,
                                       -1.885147776001495e-01, 7.308229981701020e-02,  -1.746740887691195e-02};
        } else if constexpr (digits <= 6) {
            return std::array<Real, 11>{1.614709231716746e-02,  -8.104482562490173e-02, 1.821202399011004e-01,
                                        -2.445665225927775e-01, 2.214153542697690e-01,  -1.447686339587813e-01,
                                        7.139964303548979e-02,  -2.725881175495876e-02, 8.433287237298649e-03,
                                        -2.361123374269844e-03, 4.843770794874525e-04};
        } else if constexpr (digits <= 9) {
            return std::array<Real, 17>{2.086402113115865e-03,  -1.562089192993565e-02, 5.445041674459949e-02,
                                        -1.178541200828409e-01, 1.783129998458763e-01,  -2.013700177956373e-01,
                                        1.770491043290802e-01,  -1.248556216953823e-01, 7.222027766315242e-02,
                                        -3.487303987632228e-02, 1.426369137364099e-02,  -5.005766772662122e-03,
                                        1.519637697654493e-03,  -3.983110653078808e-04, 9.342903238021878e-05,
                                        -2.223077176699562e-05, 4.041199749483288e-06};
        } else if constexpr (digits <= 12) {
            return std::array<Real, 22>{
                2.585256922145451e-04, -2.586987362995730e-03, 1.228229637064665e-02, -3.689309545219710e-02,
                7.889646803121930e-02, -1.281810865332097e-01, 1.648961579104568e-01, -1.728791019337961e-01,
                1.509077982778982e-01, -1.115149888411611e-01, 7.069833292226477e-02, -3.888103784086078e-02,
                1.872315725194051e-02, -7.957995064673665e-03, 3.005989787130355e-03, -1.015826376419250e-03,
                3.094089416842241e-04, -8.493261780963615e-05, 2.066398691692728e-05, -4.722966945758245e-06,
                1.200827608904831e-06, -2.250099297995689e-07};
        }
    }();

    Vec thresh2 = thresh[0];
    Vec d2max_vec = d2max[0];
    Vec rsc_vec = rsc[0];
    Vec cen_vec = cen[0];
    // load charge
    sctl::Matrix<Real> Vs_(Nsrc, nd_, sctl::Ptr2Itr<Real>((Real *)charge, nd_ * Nsrc), false);
    // load source
    sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real *)sources, COORD_DIM * Nsrc), false);
    for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
        Vec Xtrg[COORD_DIM];
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
        }
        // load potential
        Vec Vtrg[nd_];
        for (long i = 0; i < nd_; i++) {
            Vtrg[i] = Vec::LoadAligned(&Vt[i][t]);
        }
        for (sctl::Long s = 0; s < Nsrc; s++) {
            Vec dX[COORD_DIM], R2 = Vec::Zero();
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
                dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
                R2 += dX[k] * dX[k];
            }

            Vec Rinv = sctl::approx_rsqrt<digits>(R2, (R2 > thresh2) & (R2 < d2max_vec));
            Vec R2inv = Rinv * Rinv;
            // evaluate the PSWF kernel
            Vec xtmp = FMA(R2, rsc_vec, cen_vec);
            Vec ptmp = EvalPolynomial(xtmp.get(), coefs);

            ptmp = ptmp * R2inv;

            for (long i = 0; i < nd_; i++) {
                Vtrg[i] += Vec::Load1(&Vs_[s][i]) * ptmp;
            }
        }
        for (long i = 0; i < nd_; i++) {
            Vtrg[i].StoreAligned(&Vt[i][t]);
        }
    }

    for (long i = 0; i < Ntrg; i++) {
        for (long j = 0; j < nd_; j++) {
            pot[i * nd_ + j] += Vt[j][i];
        }
    }
}

template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void sl3d_local_kernel_directcp_vec_cpp(const int32_t *nd, const int32_t *ndim, const int32_t *digits, const Real *rsc,
                                        const Real *cen, const Real *d2max, const Real *sources, const int32_t *ns,
                                        const Real *charge, const Real *xtarg, const Real *ytarg, const Real *ztarg,
                                        const int32_t *nt, Real *pot, const Real *thresh) {
    if (ndim[0] == 3) {
        if (digits[0] <= 3)
            sl3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 3, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 6)
            sl3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 6, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 9)
            sl3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 9, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 12)
            sl3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 12, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                              xtarg, ytarg, ztarg, nt, pot, thresh);
        else
            sl3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, -1, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                              xtarg, ytarg, ztarg, nt, pot, thresh);
    }
    if (ndim[0] == 2) {
        if (digits[0] <= 3)
            sl3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 3, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 6)
            sl3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 6, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 9)
            sl3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 9, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 12)
            sl3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 12, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                              xtarg, ytarg, ztarg, nt, pot, thresh);
        else
            sl3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, -1, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                              xtarg, ytarg, ztarg, nt, pot, thresh);
    }
}

/* local kernel for the log(r) kernel */
template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>(), sctl::Integer digits = -1,
          sctl::Integer ndim>
void log_local_kernel_directcp_vec_cpp_helper(const int32_t *nd, const Real *rsc, const Real *cen, const Real *d2max,
                                              const Real *sources, const int32_t *ns, const Real *charge,
                                              const Real *xtarg, const Real *ytarg, const Real *ztarg,
                                              const int32_t *nt, Real *pot, const Real *thresh) {
    static constexpr sctl::Integer VecLen = MaxVecLen;
    using Vec = sctl::Vec<Real, VecLen>;
    static constexpr sctl::Integer COORD_DIM = ndim; // ndim[0];
    constexpr sctl::Long nd_ = 1;                    // nd[0];
    sctl::Long Nsrc = ns[0];
    sctl::Long Ntrg = nt[0];
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
        if (COORD_DIM > 2) {
            std::memcpy(Xt[2], ztarg, sizeof(Real) * Ntrg);
        }
        Vt = 0;
    }

    constexpr auto coefs = []() {
        if constexpr (digits <= 3) {
            return std::array<Real, 5>{3.293312412035785e-01, -4.329140084314137e-01, 1.366683635926240e-01,
                                       -4.309918126794055e-02, 1.041106682948322e-02};
        } else if constexpr (digits <= 6) {
            return std::array<Real, 10>{3.449851438836016e-01,  -4.902921365061905e-01, 2.220880572548949e-01,
                                        -1.153716526684871e-01, 5.535102319921498e-02,  -2.281631998557134e-02,
                                        7.843017349311455e-03,  -2.269922867123751e-03, 6.058276012390756e-04,
                                        -1.231943198424746e-04};
        } else if constexpr (digits <= 9) {
            return std::array<Real, 15>{3.464285661408188e-01,  -4.987507216024493e-01, 2.448626056577886e-01,
                                        -1.531215433543293e-01, 9.898265049486202e-02,  -6.064807566030946e-02,
                                        3.368003719471507e-02,  -1.657913779091916e-02, 7.167896958658711e-03,
                                        -2.718119907588121e-03, 9.031254984609993e-04,  -2.605021223678894e-04,
                                        6.764515456758602e-05,  -1.806660493741674e-05, 3.640268744220521e-06};
        } else if constexpr (digits <= 12) {
            return std::array<Real, 19>{
                3.465597422118963e-01, -4.998454957461734e-01, 2.491697186708244e-01, -1.637913095741979e-01,
                1.177474435136638e-01, -8.570242499810476e-02, 6.020769882062124e-02, -3.955364604811165e-02,
                2.382498816767467e-02, -1.301290744915523e-02, 6.410015506457686e-03, -2.841967252293358e-03,
                1.135602935662887e-03, -4.109780616023590e-04, 1.339086505665511e-04, -3.822398669202901e-05,
                1.037153217818392e-05, -3.251884408687046e-06, 7.149918161513096e-07};
        }
    }();

    Vec thresh2 = thresh[0];
    Vec d2max_vec = d2max[0];
    Vec rsc_vec = rsc[0];
    Vec cen_vec = cen[0];
    Vec bsizeinv2_vec = rsc[0] * Real{0.5e0};
    // load charge
    sctl::Matrix<Real> Vs_(Nsrc, nd_, sctl::Ptr2Itr<Real>((Real *)charge, nd_ * Nsrc), false);
    // load source
    sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real *)sources, COORD_DIM * Nsrc), false);
    for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
        Vec Xtrg[COORD_DIM];
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
        }
        // load potential
        Vec Vtrg[nd_];
        for (long i = 0; i < nd_; i++) {
            Vtrg[i] = Vec::LoadAligned(&Vt[i][t]);
        }
        for (sctl::Long s = 0; s < Nsrc; s++) {
            Vec dX[COORD_DIM], R2 = Vec::Zero();
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
                dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
                R2 += dX[k] * dX[k];
            }

            Vec xtmp = FMA(R2, rsc_vec, cen_vec);
            Vec R2sc = R2 * bsizeinv2_vec;
            Vec ptmp = EvalPolynomial(xtmp.get(), coefs);

            ptmp = select((R2 > thresh2) & (R2 < d2max_vec), Real{0.5e0} * sctl::log(R2sc) + ptmp, Vec::Zero());

            for (long i = 0; i < nd_; i++) {
                Vtrg[i] += Vec::Load1(&Vs_[s][i]) * ptmp;
            }
        }
        for (long i = 0; i < nd_; i++) {
            Vtrg[i].StoreAligned(&Vt[i][t]);
        }
    }

    for (long i = 0; i < Ntrg; i++) {
        for (long j = 0; j < nd_; j++) {
            pot[i * nd_ + j] += Vt[j][i];
        }
    }
}

template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void log_local_kernel_directcp_vec_cpp(const int32_t *nd, const int32_t *ndim, const int32_t *digits, const Real *rsc,
                                       const Real *cen, const Real *d2max, const Real *sources, const int32_t *ns,
                                       const Real *charge, const Real *xtarg, const Real *ytarg, const Real *ztarg,
                                       const int32_t *nt, Real *pot, const Real *thresh) {
    if (ndim[0] == 3) {
        if (digits[0] <= 3)
            log_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 3, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                            xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 6)
            log_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 6, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                            xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 9)
            log_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 9, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                            xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 12)
            log_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 12, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
        else
            log_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, -1, 3>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
    }
    if (ndim[0] == 2) {
        if (digits[0] <= 3)
            log_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 3, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                            xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 6)
            log_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 6, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                            xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 9)
            log_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 9, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                            xtarg, ytarg, ztarg, nt, pot, thresh);
        else if (digits[0] <= 12)
            log_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 12, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
        else
            log_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, -1, 2>(nd, rsc, cen, d2max, sources, ns, charge,
                                                                             xtarg, ytarg, ztarg, nt, pot, thresh);
    }
}

/* the Yukawa kernel in 3D exp(-kr)/r, charge to potential */
template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>(), sctl::Integer digits = -1>
void y3ddirectcp_vec_cpp_helper(const int32_t *nd, const Real *rlambda, const Real *d2max, const Real *sources,
                                const int32_t *ns, const Real *charge, const Real *xtarg, const Real *ytarg,
                                const Real *ztarg, const int32_t *nt, Real *pot, const Real *thresh) {
    static constexpr sctl::Integer COORD_DIM = 3; // ndim[0];
    constexpr sctl::Long nd_ = 1;                 // nd[0];
    sctl::Long Nsrc = ns[0];
    sctl::Long Ntrg = nt[0];
    sctl::Long Ntrg_ = ((Ntrg + MaxVecLen - 1) / MaxVecLen) * MaxVecLen;

    sctl::StaticArray<Real, 400 * COORD_DIM> buff0;
    sctl::StaticArray<Real, 400 * 1> buff1;
    sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
    sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
    if (Ntrg_ > 400) {
        Xt.ReInit(COORD_DIM, Ntrg_);
        Vt.ReInit(nd_, Ntrg_);
    }
    { // Set Xs, Vs, Xt, Vt
        std::memcpy(Xt[0], xtarg, sizeof(Real) * Ntrg);
        std::memcpy(Xt[1], ytarg, sizeof(Real) * Ntrg);
        std::memcpy(Xt[2], ztarg, sizeof(Real) * Ntrg);
        Vt = 0;
    }

    static constexpr sctl::Integer VecLen = MaxVecLen;
    using Vec = sctl::Vec<Real, VecLen>;
    Vec thresh2 = thresh[0];
    Vec d2max_vec = d2max[0];
    Vec mrlambda_vec = -rlambda[0];
    // load charge
    sctl::Matrix<Real> Vs_(Nsrc, nd_, sctl::Ptr2Itr<Real>((Real *)charge, nd_ * Nsrc), false);
    // load source
    sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real *)sources, COORD_DIM * Nsrc), false);
    for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
        Vec Xtrg[COORD_DIM];
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
        }
        // load potential
        Vec Vtrg[nd_];
        for (long i = 0; i < nd_; i++) {
            Vtrg[i] = Vec::LoadAligned(&Vt[i][t]);
        }
        for (sctl::Long s = 0; s < Nsrc; s++) {
            Vec dX[COORD_DIM], R2 = Vec::Zero();
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
                dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
                R2 += dX[k] * dX[k];
            }

            Vec Rinv = sctl::approx_rsqrt<digits>(R2, (R2 > thresh2) & (R2 < d2max_vec));

            Vec xtmp = mrlambda_vec * R2 * Rinv;
            Vec ptmp = sctl::approx_exp<digits>(xtmp);

            ptmp = select((R2 > thresh2) & (R2 < d2max_vec), ptmp * Rinv, Vec::Zero());

            for (long i = 0; i < nd_; i++) {
                Vtrg[i] += Vec::Load1(&Vs_[s][i]) * ptmp;
            }
        }
        for (long i = 0; i < nd_; i++) {
            Vtrg[i].StoreAligned(&Vt[i][t]);
        }
    }

    for (long i = 0; i < Ntrg; i++) {
        for (long j = 0; j < nd_; j++) {
            pot[i * nd_ + j] += Vt[j][i];
        }
    }
}

template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void y3ddirectcp_vec_cpp(const int32_t *nd, const int32_t *digits, const Real *rlambda, const Real *d2max,
                         const Real *sources, const int32_t *ns, const Real *charge, const Real *xtarg,
                         const Real *ytarg, const Real *ztarg, const int32_t *nt, Real *pot, const Real *thresh) {
    if (digits[0] <= 3)
        y3ddirectcp_vec_cpp_helper<Real, MaxVecLen, 3>(nd, rlambda, d2max, sources, ns, charge, xtarg, ytarg, ztarg, nt,
                                                       pot, thresh);
    else if (digits[0] <= 6)
        y3ddirectcp_vec_cpp_helper<Real, MaxVecLen, 6>(nd, rlambda, d2max, sources, ns, charge, xtarg, ytarg, ztarg, nt,
                                                       pot, thresh);
    else if (digits[0] <= 9)
        y3ddirectcp_vec_cpp_helper<Real, MaxVecLen, 9>(nd, rlambda, d2max, sources, ns, charge, xtarg, ytarg, ztarg, nt,
                                                       pot, thresh);
    else if (digits[0] <= 12)
        y3ddirectcp_vec_cpp_helper<Real, MaxVecLen, 12>(nd, rlambda, d2max, sources, ns, charge, xtarg, ytarg, ztarg,
                                                        nt, pot, thresh);
    else
        y3ddirectcp_vec_cpp_helper<Real, MaxVecLen, -1>(nd, rlambda, d2max, sources, ns, charge, xtarg, ytarg, ztarg,
                                                        nt, pot, thresh);
}

/* 3D Stokeslet local kernel charge to potential */
#if !defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>(), sctl::Integer digits = -1,
          sctl::Integer ndim>
void st3d_local_kernel_directcp_vec_cpp_helper(const int32_t *nd, const Real *rsc, const Real *cen,
                                               const Real *bsizeinv, const Real *d2min, const Real *d2max,
                                               const Real *sources, const int32_t *ns, const Real *stoklet,
                                               const Real *xtarg, const Real *ytarg, const Real *ztarg,
                                               const int32_t *nt, Real *pot) {
    static constexpr sctl::Integer COORD_DIM = ndim; // ndim[0];
    constexpr sctl::Long nd_ = 1;                    // nd[0];
    sctl::Long Nsrc = ns[0];
    sctl::Long Ntrg = nt[0];
    sctl::Long Ntrg_ = ((Ntrg + MaxVecLen - 1) / MaxVecLen) * MaxVecLen;

    sctl::StaticArray<Real, 400 * COORD_DIM> buff0;
    sctl::StaticArray<Real, 400 * COORD_DIM> buff1;
    sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
    sctl::Matrix<Real> Vt(nd_ * COORD_DIM, Ntrg_, buff1, false);
    if (Ntrg_ > 400) {
        Xt.ReInit(COORD_DIM, Ntrg_);
        Vt.ReInit(nd_ * COORD_DIM, Ntrg_);
    }
    { // Set Xs, Vs, Xt, Vt
        std::memcpy(Xt[0], xtarg, sizeof(Real) * Ntrg);
        std::memcpy(Xt[1], ytarg, sizeof(Real) * Ntrg);
        if (COORD_DIM > 2) {
            std::memcpy(Xt[2], ztarg, sizeof(Real) * Ntrg);
        }
        Vt = 0;
    }

    constexpr auto cdata = []() {
        if constexpr (digits <= 3) {
            return std::make_pair(
                std::array<Real, 10>{6.099528438109667e-01, -2.047157964842887e-01, -2.113126917088419e-01,
                                     7.013284302249340e-01, -3.570510763924770e-01, -2.480337564051550e-01,
                                     2.734718790066922e-01, -2.180464195137854e-02, -6.443538145036971e-02,
                                     2.389731909270994e-02},
                std::array<Real, 10>{3.274814701832230e-01, 4.871068566478242e-01, -2.761957322600402e-01,
                                     -3.536242878724801e-01, 3.632236877209861e-01, 9.518566823755009e-02,
                                     -2.122968176325724e-01, 4.390446486009057e-02, 4.716452147294282e-02,
                                     -2.322990464181018e-02});
        } else if (digits <= 6) {
            return std::make_pair(
                std::array<Real, 16>{
                    5.502056770069008e-01, -2.287529467967572e-01, 2.769691552491858e-01, 3.135322806554014e-01,
                    -1.101558928618569e+00, 7.661126857290709e-01, 5.508339761214055e-01, -1.014479282253664e+00,
                    1.275402352031388e-01, 5.593771332723235e-01, -2.460532805797172e-01, -1.737644276098480e-01,
                    1.123445216568968e-01, 3.020368722716553e-02, -2.028095033637215e-02, -2.228529880707105e-03},
                std::array<Real, 16>{
                    4.330790775204780e-01, 3.458799160696542e-01, -6.228461130271690e-01, 2.169769752603976e-01,
                    7.277536889040245e-01, -8.327819195562342e-01, -2.282968540308179e-01, 8.486133061697511e-01,
                    -2.172465522880920e-01, -4.395930105160376e-01, 2.423098594086845e-01, 1.304989205805818e-01,
                    -1.029623502416652e-01, -2.078744068198703e-02, 1.820880714658577e-02, 1.192640919347133e-03});
        } else if (digits <= 9) {
            return std::make_pair(
                std::array<Real, 24>{
                    5.214362247171502e-01,  -1.479803449544993e-01, 3.772714901685854e-01,  -2.718562296457158e-01,
                    -6.541250825947101e-01, 1.629004008150590e+00,  -9.697385182122231e-01, -1.087412122459157e+00,
                    1.886661054687073e+00,  -3.158514419561095e-01, -1.278575577116810e+00, 8.640602464955245e-01,
                    3.968111895283485e-01,  -6.283634761310652e-01, 1.818983634952905e-02,  2.768163566014041e-01,
                    -7.808238436549593e-02, -8.386066198481187e-02, 3.923618185407533e-02,  1.769007638630430e-02,
                    -1.032650734367779e-02, -2.408001795414074e-03, 1.242092482193819e-03,  1.615914116897001e-04},
                std::array<Real, 24>{
                    4.734026302690363e-01,  1.960139393294324e-01,  -5.732854303746169e-01, 7.193845030721162e-01,
                    7.066873612888103e-02,  -1.307197862207776e+00, 1.190932928328865e+00,  5.891515742124555e-01,
                    -1.660246015308583e+00, 5.086810144494890e-01,  1.022532923554963e+00,  -8.404313009145851e-01,
                    -2.762757730731084e-01, 5.687331825641126e-01,  -4.869207218927012e-02, -2.436440776341837e-01,
                    8.009218231091993e-02,  7.239116359942260e-02,  -3.769113295146279e-02, -1.491303543925510e-02,
                    9.720762594379262e-03,  1.953480016431507e-03,  -1.159739424269745e-03, -1.225811474287557e-04});
        } else if (digits <= 12) {
            return std::make_pair(
                std::array<Real, 30>{
                    5.089666783158919e-01,  -8.185216289157819e-02, 3.054337615746002e-01,  -5.274295383825454e-01,
                    1.113687705200191e-01,  1.278165182983743e+00,  -2.294956424190830e+00, 8.897739252599565e-01,
                    2.162803418889366e+00,  -3.028085759253322e+00, 2.139592256732614e-01,  2.571715095845196e+00,
                    -1.757168309622261e+00, -8.978763339965670e-01, 1.598636771645721e+00,  -1.697334453079356e-01,
                    -8.161171704394112e-01, 3.847774505748418e-01,  2.596417864748551e-01,  -2.477127775378064e-01,
                    -4.064826412338574e-02, 1.042923605433851e-01,  -6.240153075594615e-03, -3.252985552087324e-02,
                    5.677432855285940e-03,  7.641726820425668e-03,  -1.526165191807415e-03, -1.254936307499743e-03,
                    1.686406928990181e-04,  1.090671714677842e-04},
                std::array<Real, 31>{
                    4.893268763647259e-01,  1.014919648418111e-01,  -4.069257263187320e-01, 8.325440110506406e-01,
                    -6.801980196801547e-01, -6.647884379120371e-01, 2.107634955479558e+00,  -1.358153969953307e+00,
                    -1.471983220212153e+00, 2.817883915772575e+00,  -6.093435797359408e-01, -2.137388090106868e+00,
                    1.751272584714726e+00,  6.331984388177745e-01,  -1.461445134944767e+00, 2.466655800895049e-01,
                    7.156713330598122e-01,  -3.831424686835810e-01, -2.136814950036207e-01, 2.348808724228167e-01,
                    2.135702724667653e-02,  -9.755387453273209e-02, 1.695238219459778e-02,  3.057780167943610e-02,
                    -1.245204039542512e-02, -7.366710437323577e-03, 4.842477973194526e-03,  1.268319325937314e-03,
                    -1.163011617009797e-03, -1.173523766826381e-04, 1.345908773360054e-04});
        }
    }();
    constexpr auto cdiag = cdata.first;
    constexpr auto coffd = cdata.second;

    static constexpr sctl::Integer VecLen = MaxVecLen;
    using Vec = sctl::Vec<Real, VecLen>;
    Vec d2min_vec = d2min[0];
    Vec d2max_vec = d2max[0];
    Vec rsc_vec = rsc[0];
    Vec cen_vec = cen[0];
    Vec bsizeinv_vec = bsizeinv[0];

    // load stokslet density
    sctl::Matrix<Real> Vs_(Nsrc, nd_ * COORD_DIM, sctl::Ptr2Itr<Real>((Real *)stoklet, nd_ * Nsrc), false);
    // Vec Vsrc[Nsrc][nd_][COORD_DIM];
    sctl::Vector<Vec> Vsrc(Nsrc * nd_ * COORD_DIM);
    for (sctl::Long s = 0; s < Nsrc; s++) {
        for (long i = 0; i < nd_; i++) {
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0] = Vec::Load1(&Vs_[s][0 * nd_ + i]);
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1] = Vec::Load1(&Vs_[s][1 * nd_ + i]);
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 2] = Vec::Load1(&Vs_[s][2 * nd_ + i]);
        }
    }

    // load source
    sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real *)sources, COORD_DIM * Nsrc), false);
    for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
        Vec Xtrg[COORD_DIM];
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
        }
        // load potential
        Vec Vtrg[nd_][COORD_DIM];
        for (long i = 0; i < nd_; i++) {
            Vtrg[i][0] = Vec::LoadAligned(&Vt[0 * nd_ + i][t]);
            Vtrg[i][1] = Vec::LoadAligned(&Vt[1 * nd_ + i][t]);
            Vtrg[i][2] = Vec::LoadAligned(&Vt[2 * nd_ + i][t]);
        }

        for (sctl::Long s = 0; s < Nsrc; s++) {
            Vec dX[COORD_DIM], R2 = Vec::Zero();
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
                dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
                R2 += dX[k] * dX[k];
            }
            // evaluate the PSWF local kernel
            Vec Rinv = sctl::approx_rsqrt<digits>(R2, (R2 >= d2min_vec) & (R2 <= d2max_vec));
            Vec Rinv3 = Rinv * Rinv * Rinv;
            Vec xtmp = FMA(R2, Rinv, cen_vec) * rsc_vec;
            Vec fdiag = EvalPolynomial(xtmp.get(), cdiag);
            Vec foffd = EvalPolynomial(xtmp.get(), coffd);

            Vec half = 0.5e0;
            Vec Fdiag = (half - fdiag) * Rinv;
            Vec Foffd = (half - foffd) * Rinv3;
            for (long i = 0; i < nd_; i++) {
                Vec Dprod = dX[0] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0] +
                            dX[1] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1] +
                            dX[2] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 2];
                Vec pl = Foffd * Dprod;
                Vtrg[i][0] += pl * dX[0] + Fdiag * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0];
                Vtrg[i][1] += pl * dX[1] + Fdiag * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1];
                Vtrg[i][2] += pl * dX[2] + Fdiag * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 2];
            }
        }
        // store potential
        for (long i = 0; i < nd_; i++) {
            Vtrg[i][0].StoreAligned(&Vt[0 * nd_ + i][t]);
            Vtrg[i][1].StoreAligned(&Vt[1 * nd_ + i][t]);
            Vtrg[i][2].StoreAligned(&Vt[2 * nd_ + i][t]);
        }
    }

    for (long i = 0; i < Ntrg; i++) {
        for (long j = 0; j < nd_ * COORD_DIM; j++) {
            pot[i * nd_ * COORD_DIM + j] += Vt[j][i];
        }
    }
}
#endif

#if defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>(), sctl::Integer digits = -1,
          sctl::Integer ndim>
void st3d_local_kernel_directcp_vec_cpp_helper(const int32_t *nd, const Real *rsc, const Real *cen,
                                               const Real *bsizeinv, const Real *d2min, const Real *d2max,
                                               const Real *sources, const int32_t *ns, const Real *stoklet,
                                               const Real *xtarg, const Real *ytarg, const Real *ztarg,
                                               const int32_t *nt, Real *pot) {
    static constexpr sctl::Integer COORD_DIM = ndim; // ndim[0];
    constexpr sctl::Long nd_ = 1;                    // nd[0];
    sctl::Long Nsrc = ns[0];
    sctl::Long Ntrg = nt[0];
    sctl::Long Ntrg_ = ((Ntrg + MaxVecLen - 1) / MaxVecLen) * MaxVecLen;

    sctl::StaticArray<Real, 400 * COORD_DIM> buff0;
    sctl::StaticArray<Real, 400 * COORD_DIM> buff1;
    sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
    sctl::Matrix<Real> Vt(nd_ * COORD_DIM, Ntrg_, buff1, false);
    if (Ntrg_ > 400) {
        Xt.ReInit(COORD_DIM, Ntrg_);
        Vt.ReInit(nd_ * COORD_DIM, Ntrg_);
    }
    { // Set Xs, Vs, Xt, Vt
        std::memcpy(Xt[0], xtarg, sizeof(Real) * Ntrg);
        std::memcpy(Xt[1], ytarg, sizeof(Real) * Ntrg);
        if (COORD_DIM > 2) {
            std::memcpy(Xt[2], ztarg, sizeof(Real) * Ntrg);
        }
        Vt = 0;
    }

    constexpr auto cdata = []() {
        if constexpr (digits <= 3) {
            return std::make_pair(
                std::array<Real, 10>{6.099528438109667e-01, -2.047157964842887e-01, -2.113126917088419e-01,
                                     7.013284302249340e-01, -3.570510763924770e-01, -2.480337564051550e-01,
                                     2.734718790066922e-01, -2.180464195137854e-02, -6.443538145036971e-02,
                                     2.389731909270994e-02},
                std::array<Real, 10>{3.274814701832230e-01, 4.871068566478242e-01, -2.761957322600402e-01,
                                     -3.536242878724801e-01, 3.632236877209861e-01, 9.518566823755009e-02,
                                     -2.122968176325724e-01, 4.390446486009057e-02, 4.716452147294282e-02,
                                     -2.322990464181018e-02});
        } else if constexpr (digits <= 6) {
            return std::make_pair(
                std::array<Real, 16>{
                    5.502056770069008e-01, -2.287529467967572e-01, 2.769691552491858e-01, 3.135322806554014e-01,
                    -1.101558928618569e+00, 7.661126857290709e-01, 5.508339761214055e-01, -1.014479282253664e+00,
                    1.275402352031388e-01, 5.593771332723235e-01, -2.460532805797172e-01, -1.737644276098480e-01,
                    1.123445216568968e-01, 3.020368722716553e-02, -2.028095033637215e-02, -2.228529880707105e-03},
                std::array<Real, 16>{
                    4.330790775204780e-01, 3.458799160696542e-01, -6.228461130271690e-01, 2.169769752603976e-01,
                    7.277536889040245e-01, -8.327819195562342e-01, -2.282968540308179e-01, 8.486133061697511e-01,
                    -2.172465522880920e-01, -4.395930105160376e-01, 2.423098594086845e-01, 1.304989205805818e-01,
                    -1.029623502416652e-01, -2.078744068198703e-02, 1.820880714658577e-02, 1.192640919347133e-03});
        } else if constexpr (digits <= 9) {
            return std::make_pair(
                std::array<Real, 24>{
                    5.214362247171502e-01,  -1.479803449544993e-01, 3.772714901685854e-01,  -2.718562296457158e-01,
                    -6.541250825947101e-01, 1.629004008150590e+00,  -9.697385182122231e-01, -1.087412122459157e+00,
                    1.886661054687073e+00,  -3.158514419561095e-01, -1.278575577116810e+00, 8.640602464955245e-01,
                    3.968111895283485e-01,  -6.283634761310652e-01, 1.818983634952905e-02,  2.768163566014041e-01,
                    -7.808238436549593e-02, -8.386066198481187e-02, 3.923618185407533e-02,  1.769007638630430e-02,
                    -1.032650734367779e-02, -2.408001795414074e-03, 1.242092482193819e-03,  1.615914116897001e-04},
                std::array<Real, 24>{
                    4.734026302690363e-01,  1.960139393294324e-01,  -5.732854303746169e-01, 7.193845030721162e-01,
                    7.066873612888103e-02,  -1.307197862207776e+00, 1.190932928328865e+00,  5.891515742124555e-01,
                    -1.660246015308583e+00, 5.086810144494890e-01,  1.022532923554963e+00,  -8.404313009145851e-01,
                    -2.762757730731084e-01, 5.687331825641126e-01,  -4.869207218927012e-02, -2.436440776341837e-01,
                    8.009218231091993e-02,  7.239116359942260e-02,  -3.769113295146279e-02, -1.491303543925510e-02,
                    9.720762594379262e-03,  1.953480016431507e-03,  -1.159739424269745e-03, -1.225811474287557e-04});
        } else if constexpr (digits <= 12) {
            return std::make_pair(
                std::array<Real, 30>{
                    5.089666783158919e-01,  -8.185216289157819e-02, 3.054337615746002e-01,  -5.274295383825454e-01,
                    1.113687705200191e-01,  1.278165182983743e+00,  -2.294956424190830e+00, 8.897739252599565e-01,
                    2.162803418889366e+00,  -3.028085759253322e+00, 2.139592256732614e-01,  2.571715095845196e+00,
                    -1.757168309622261e+00, -8.978763339965670e-01, 1.598636771645721e+00,  -1.697334453079356e-01,
                    -8.161171704394112e-01, 3.847774505748418e-01,  2.596417864748551e-01,  -2.477127775378064e-01,
                    -4.064826412338574e-02, 1.042923605433851e-01,  -6.240153075594615e-03, -3.252985552087324e-02,
                    5.677432855285940e-03,  7.641726820425668e-03,  -1.526165191807415e-03, -1.254936307499743e-03,
                    1.686406928990181e-04,  1.090671714677842e-04},
                std::array<Real, 31>{
                    4.893268763647259e-01,  1.014919648418111e-01,  -4.069257263187320e-01, 8.325440110506406e-01,
                    -6.801980196801547e-01, -6.647884379120371e-01, 2.107634955479558e+00,  -1.358153969953307e+00,
                    -1.471983220212153e+00, 2.817883915772575e+00,  -6.093435797359408e-01, -2.137388090106868e+00,
                    1.751272584714726e+00,  6.331984388177745e-01,  -1.461445134944767e+00, 2.466655800895049e-01,
                    7.156713330598122e-01,  -3.831424686835810e-01, -2.136814950036207e-01, 2.348808724228167e-01,
                    2.135702724667653e-02,  -9.755387453273209e-02, 1.695238219459778e-02,  3.057780167943610e-02,
                    -1.245204039542512e-02, -7.366710437323577e-03, 4.842477973194526e-03,  1.268319325937314e-03,
                    -1.163011617009797e-03, -1.173523766826381e-04, 1.345908773360054e-04});
        }
    }();
    constexpr auto cdiag = cdata.first;
    constexpr auto coffd = cdata.second;

    static constexpr sctl::Integer VecLen = MaxVecLen;
    using Vec = sctl::Vec<Real, VecLen>;
    Vec d2min_vec = d2min[0];
    Vec d2max_vec = d2max[0];
    Vec rsc_vec = rsc[0];
    Vec cen_vec = cen[0];
    Vec bsizeinv_vec = bsizeinv[0];

    // load stokslet density
    sctl::Matrix<Real> Vs_(Nsrc, nd_ * COORD_DIM, sctl::Ptr2Itr<Real>((Real *)stoklet, nd_ * Nsrc), false);
    // Vec Vsrc[Nsrc][nd_][COORD_DIM];
    sctl::Vector<Vec> Vsrc(Nsrc * nd_ * COORD_DIM);
    for (sctl::Long s = 0; s < Nsrc; s++) {
        for (long i = 0; i < nd_; i++) {
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0] = Vec::Load1(&Vs_[s][0 * nd_ + i]);
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1] = Vec::Load1(&Vs_[s][1 * nd_ + i]);
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 2] = Vec::Load1(&Vs_[s][2 * nd_ + i]);
        }
    }

    // load source
    sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real *)sources, COORD_DIM * Nsrc), false);

//  TODO: reinit to sctl::Vector is over limit
#define NSRC_LIMIT 1000
    // work array memory stores compressed R2
    Real dX_work[COORD_DIM][NSRC_LIMIT * 8];
    Real R2_work[NSRC_LIMIT * 8];
    // sctl::StaticArray<Real, NSRC_LIMIT*8> R2_work_buff;
    //  why converting to sctl::Vector is a bit slow...
    // sctl::Vector<Real> R2_work(NSRC_LIMIT*8, R2_work_buff, false);
    //  work array memory stores compressed polynomial evals
    Real Fdiag_work[NSRC_LIMIT * 8];
    Real Foffd_work[NSRC_LIMIT * 8];
    // mask array
    typename Vec::MaskType Mask_work[NSRC_LIMIT];
    // sparse starting index of stored Pval
    int start_ind[NSRC_LIMIT];
    // sparse source point index
    int source_ind[NSRC_LIMIT];
    // zero vec
    static Vec zero_tmp = Vec::Zero();
    for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
        Vec Xtrg[COORD_DIM];
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
        }
        // load potential
        Vec Vtrg[nd_][COORD_DIM];
        for (long i = 0; i < nd_; i++) {
            Vtrg[i][0] = Vec::LoadAligned(&Vt[0 * nd_ + i][t]);
            Vtrg[i][1] = Vec::LoadAligned(&Vt[1 * nd_ + i][t]);
            Vtrg[i][2] = Vec::LoadAligned(&Vt[2 * nd_ + i][t]);
        }

        // int total_r = Nsrc*VecLen;
        int valid_r = 0;
        int source_cnt = 0;
        // store compressed R2 to explore 4/3 pi over 27 sparsity
        for (sctl::Long s = 0; s < Nsrc; s++) {
            Vec dX[COORD_DIM], R2 = zero_tmp;
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
                dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
                R2 += dX[k] * dX[k];
            }

            // store sparse data
            Mask_work[source_cnt] = (R2 >= d2min_vec) & (R2 <= d2max_vec);
            int valid_cnt = mask_popcnt(Mask_work[source_cnt]);
            mask_compress_store(Mask_work[source_cnt], dX[0], &dX_work[0][0] + valid_r);
            mask_compress_store(Mask_work[source_cnt], dX[1], &dX_work[1][0] + valid_r);
            mask_compress_store(Mask_work[source_cnt], dX[2], &dX_work[2][0] + valid_r);
            mask_compress_store(Mask_work[source_cnt], R2, &R2_work[0] + valid_r);
            start_ind[source_cnt] = valid_r;
            source_ind[source_cnt] = s;
            source_cnt += (valid_cnt > 0);
            valid_r += valid_cnt;
        }

        // evaluate polynomial on compressed R2 and store in Pval
        for (sctl::Long s = 0; s < valid_r; s += VecLen) {
            Vec R2 = Vec::LoadAligned(&R2_work[0] + s);
            // evaluate the PSWF local kernel
            Vec Rinv = sctl::approx_rsqrt<digits>(R2, (R2 >= d2min_vec) & (R2 <= d2max_vec));
            Vec Rinv3 = Rinv * Rinv * Rinv;
            Vec xtmp = FMA(R2, Rinv, cen_vec) * rsc_vec;

            Vec fdiag = EvalPolynomial(xtmp.get(), cdiag);
            Vec foffd = EvalPolynomial(xtmp.get(), coffd);

            Vec half = 0.5e0;
            Vec Fdiag = (half - fdiag) * Rinv;
            Vec Foffd = (half - foffd) * Rinv3;
            Fdiag.StoreAligned(Fdiag_work + s);
            Foffd.StoreAligned(Foffd_work + s);
        }

        // expand compressed Fdiag_work and Foffd_work, then multiply by charge to accumulate
        for (sctl::Long s_ind = 0; s_ind < source_cnt; s_ind++) {
            int s = source_ind[s_ind];
            int start = start_ind[s_ind];
            Vec Fdiag = mask_expand_load(Mask_work[s_ind], zero_tmp, Fdiag_work + start);
            Vec Foffd = mask_expand_load(Mask_work[s_ind], zero_tmp, Foffd_work + start);
            Vec dX[3];
            dX[0] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[0] + start);
            dX[1] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[1] + start);
            dX[2] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[2] + start);

            for (long i = 0; i < nd_; i++) {
                Vec Dprod = dX[0] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0] +
                            dX[1] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1] +
                            dX[2] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 2];
                Vec pl = Foffd * Dprod;
                Vtrg[i][0] += pl * dX[0] + Fdiag * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0];
                Vtrg[i][1] += pl * dX[1] + Fdiag * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1];
                Vtrg[i][2] += pl * dX[2] + Fdiag * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 2];
            }
        }

        // store potential
        for (long i = 0; i < nd_; i++) {
            Vtrg[i][0].StoreAligned(&Vt[0 * nd_ + i][t]);
            Vtrg[i][1].StoreAligned(&Vt[1 * nd_ + i][t]);
            Vtrg[i][2].StoreAligned(&Vt[2 * nd_ + i][t]);
        }
    }

    for (long i = 0; i < Ntrg; i++) {
        for (long j = 0; j < nd_ * COORD_DIM; j++) {
            pot[i * nd_ * COORD_DIM + j] += Vt[j][i];
        }
    }
}
#endif

template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void st3d_local_kernel_directcp_vec_cpp(const int32_t *nd, const int32_t *ndim, const int32_t *digits, const Real *rsc,
                                        const Real *cen, const Real *bsizeinv, const Real *d2min, const Real *d2max,
                                        const Real *sources, const int32_t *ns, const Real *charge, const Real *xtarg,
                                        const Real *ytarg, const Real *ztarg, const int32_t *nt, Real *pot) {
    if (digits[0] <= 3)
        st3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 3, 3>(nd, rsc, cen, bsizeinv, d2min, d2max, sources,
                                                                         ns, charge, xtarg, ytarg, ztarg, nt, pot);
    else if (digits[0] <= 6)
        st3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 6, 3>(nd, rsc, cen, bsizeinv, d2min, d2max, sources,
                                                                         ns, charge, xtarg, ytarg, ztarg, nt, pot);
    else if (digits[0] <= 9)
        st3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 9, 3>(nd, rsc, cen, bsizeinv, d2min, d2max, sources,
                                                                         ns, charge, xtarg, ytarg, ztarg, nt, pot);
    else if (digits[0] <= 12)
        st3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 12, 3>(nd, rsc, cen, bsizeinv, d2min, d2max, sources,
                                                                          ns, charge, xtarg, ytarg, ztarg, nt, pot);
    else
        st3d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, -1, 3>(nd, rsc, cen, bsizeinv, d2min, d2max, sources,
                                                                          ns, charge, xtarg, ytarg, ztarg, nt, pot);
}

/* 2D Stokeslet local kernel charge to potential */
#if !defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>(), sctl::Integer digits = -1,
          sctl::Integer ndim>
void st2d_local_kernel_directcp_vec_cpp_helper(const int32_t *nd, const Real *rsc, const Real *cen,
                                               const Real *bsizeinv, const Real *d2min, const Real *d2max,
                                               const Real *sources, const int32_t *ns, const Real *stoklet,
                                               const Real *xtarg, const Real *ytarg, const Real *ztarg,
                                               const int32_t *nt, Real *pot) {
    static constexpr sctl::Integer COORD_DIM = ndim; // ndim[0];
    constexpr sctl::Long nd_ = 1;                    // nd[0];
    sctl::Long Nsrc = ns[0];
    sctl::Long Ntrg = nt[0];
    sctl::Long Ntrg_ = ((Ntrg + MaxVecLen - 1) / MaxVecLen) * MaxVecLen;

    sctl::StaticArray<Real, 400 * COORD_DIM> buff0;
    sctl::StaticArray<Real, 400 * COORD_DIM> buff1;
    sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
    sctl::Matrix<Real> Vt(nd_ * COORD_DIM, Ntrg_, buff1, false);
    if (Ntrg_ > 400) {
        Xt.ReInit(COORD_DIM, Ntrg_);
        Vt.ReInit(nd_ * COORD_DIM, Ntrg_);
    }
    { // Set Xs, Vs, Xt, Vt
        std::memcpy(Xt[0], xtarg, sizeof(Real) * Ntrg);
        std::memcpy(Xt[1], ytarg, sizeof(Real) * Ntrg);
        Vt = 0;
    }

    constexpr auto cdata = []() {
        if constexpr (digits <= 3) {
            return std::make_pair(
                std::array<Real, 8>{1.846554578756552e-01, -2.966683058873768e-01, 2.187482696883298e-01,
                                    -1.930445665343498e-01, 1.152120060185630e-01, -4.624359846616554e-02,
                                    3.986446577184038e-02, -2.296552527129039e-02},
                std::array<Real, 8>{4.874864017381395e-01, 5.299044064212663e-02, -1.098956047309594e-01,
                                    1.382693723607076e-01, -9.272099440980629e-02, 3.856825860125045e-02,
                                    -3.459163080563133e-02, 2.031942894002221e-02});
        } else if constexpr (digits <= 6) {
            return std::make_pair(
                std::array<Real, 13>{1.742974848578507e-01, -2.573253389659778e-01, 1.497483703405333e-01,
                                     -1.352127710109274e-01, 1.381360809312198e-01, -1.310180186924878e-01,
                                     1.065981015210400e-01, -7.292086065349250e-02, 4.142209563747640e-02,
                                     -1.964709822677369e-02, 8.808445004019772e-03, -3.879150759306015e-03,
                                     9.924244727038691e-04},
                std::array<Real, 14>{4.989275970042895e-01, 7.861971816869706e-03, -2.698194194185921e-02,
                                     5.785339785044497e-02, -8.735759363134854e-02, 9.922761129588592e-02,
                                     -8.829289346280386e-02, 6.320603843280931e-02, -3.716611638604578e-02,
                                     1.848092169056624e-02, -8.180355441458238e-03, 3.238289075111389e-03,
                                     -9.485654024207217e-04, 1.319073772359878e-04});
        } else if constexpr (digits <= 9) {
            return std::make_pair(
                std::array<Real, 19>{
                    1.733906092107364e-01, -2.510413183117834e-01, 1.299689228965454e-01, -9.836547803209694e-02,
                    9.490748288758954e-02, -1.030238195403018e-01, 1.100358018444409e-01, -1.067108995165202e-01,
                    9.100532141533005e-02, -6.771478903661227e-02, 4.400096234001671e-02, -2.506248463691992e-02,
                    1.264633612426788e-02, -5.743944511033765e-03, 2.297657498033969e-03, -7.636590624624864e-04,
                    2.519553444802364e-04, -1.062770340590954e-04, 2.761976621518952e-05},
                std::array<Real, 19>{
                    4.998914591008903e-01, 1.095588747213744e-03, -5.269955173914959e-03, 1.611115959351134e-02,
                    -3.523064239855683e-02, 5.880539647249017e-02, -7.808753084994935e-02, 8.490446734143470e-02,
                    -7.723140148915500e-02, 5.976142098136458e-02, -3.983048500616800e-02, 2.308293440213639e-02,
                    -1.179441484626060e-02, 5.409162106188744e-03, -2.179104676649507e-03, 7.274978397333333e-04,
                    -2.412513081612349e-04, 1.023726763618472e-04, -2.667316339957590e-05});
        } else if constexpr (digits <= 12) {
            return std::make_pair(
                std::array<Real, 25>{
                    1.732983196849929e-01, -2.501464289483227e-01, 1.258945367320441e-01, -8.683602821002068e-02,
                    7.238738615135265e-02, -7.145254376219257e-02, 7.890525425702721e-02, -8.883419332369469e-02,
                    9.465145231459976e-02, -9.177610256521274e-02, 7.974444833050520e-02, -6.186954815871509e-02,
                    4.292612399557165e-02, -2.672824108341236e-02, 1.500078249666185e-02, -7.624355836844713e-03,
                    3.522043662308565e-03, -1.480734132654379e-03, 5.713150069276215e-04, -2.066574870600848e-04,
                    6.802971183257067e-05, -1.824115587354376e-05, 4.948292077662717e-06, -2.116521218676349e-06,
                    5.505494486025323e-07},
                std::array<Real, 25>{
                    4.999880537322993e-01,  1.524020822224470e-04, -9.356238208061141e-04, 3.686023567244900e-03,
                    -1.048563555422571e-02, 2.297970701545704e-02, -4.042620037941591e-02, 5.873973024957757e-02,
                    -7.199001885198975e-02, 7.563207375223308e-02, -6.899649230772067e-02, 5.523476882716163e-02,
                    -3.914544827522717e-02, 2.474436669033796e-02, -1.404292901409487e-02, 7.197583933083345e-03,
                    -3.345922009486318e-03, 1.414716095992157e-03, -5.492693792558703e-04, 1.988677482898063e-04,
                    -6.492302090540218e-05, 1.771179251074273e-05, -5.107593727824348e-06, 2.048245829228578e-06,
                    -4.835255036164593e-07});
        }
    }();
    constexpr auto cdiag = cdata.first;
    constexpr auto coffd = cdata.second;

    static constexpr sctl::Integer VecLen = MaxVecLen;
    using Vec = sctl::Vec<Real, VecLen>;
    Vec d2min_vec = d2min[0];
    Vec d2max_vec = d2max[0];
    Vec rsc_vec = rsc[0];
    Vec cen_vec = cen[0];
    Vec bsizeinv_vec = bsizeinv[0];
    Vec bsizeinv2_vec = bsizeinv_vec * bsizeinv_vec;

    // load stokslet density
    sctl::Matrix<Real> Vs_(Nsrc, nd_ * COORD_DIM, sctl::Ptr2Itr<Real>((Real *)stoklet, nd_ * Nsrc), false);
    // Vec Vsrc[Nsrc][nd_][COORD_DIM];
    sctl::Vector<Vec> Vsrc(Nsrc * nd_ * COORD_DIM);
    for (sctl::Long s = 0; s < Nsrc; s++) {
        for (long i = 0; i < nd_; i++) {
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0] = Vec::Load1(&Vs_[s][0 * nd_ + i]);
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1] = Vec::Load1(&Vs_[s][1 * nd_ + i]);
        }
    }

    // load source
    sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real *)sources, COORD_DIM * Nsrc), false);
    for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
        Vec Xtrg[COORD_DIM];
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
        }
        // load potential
        Vec Vtrg[nd_][COORD_DIM];
        for (long i = 0; i < nd_; i++) {
            Vtrg[i][0] = Vec::LoadAligned(&Vt[0 * nd_ + i][t]);
            Vtrg[i][1] = Vec::LoadAligned(&Vt[1 * nd_ + i][t]);
        }

        for (sctl::Long s = 0; s < Nsrc; s++) {
            Vec dX[COORD_DIM], R2 = Vec::Zero();
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
                dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
                R2 += dX[k] * dX[k];
            }
            // evaluate the PSWF local kernel

            Vec one = 1.0e0;
            Vec R2inv = select((R2 >= d2min_vec) & (R2 <= d2max_vec), one / R2, Vec::Zero());
            Vec xtmp = FMA(R2, rsc_vec, cen_vec);
            Vec fdiag = EvalPolynomial(xtmp.get(), cdiag);
            Vec foffd = EvalPolynomial(xtmp.get(), coffd);
            Vec R2sc = R2 * bsizeinv2_vec;
            Vec Fdiag = select((R2 >= d2min_vec) & (R2 <= d2max_vec), -0.25e0 * sctl::log(R2sc) - fdiag, Vec::Zero());
            Vec Foffd = (0.5e0 - foffd) * R2inv;

            for (long i = 0; i < nd_; i++) {
                Vec Dprod = dX[0] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0] +
                            dX[1] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1];
                Vec pl = Foffd * Dprod;
                Vtrg[i][0] += pl * dX[0] + Fdiag * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0];
                Vtrg[i][1] += pl * dX[1] + Fdiag * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1];
            }
        }
        // store potential
        for (long i = 0; i < nd_; i++) {
            Vtrg[i][0].StoreAligned(&Vt[0 * nd_ + i][t]);
            Vtrg[i][1].StoreAligned(&Vt[1 * nd_ + i][t]);
        }
    }

    for (long i = 0; i < Ntrg; i++) {
        for (long j = 0; j < nd_ * COORD_DIM; j++) {
            pot[i * nd_ * COORD_DIM + j] += Vt[j][i];
        }
    }
}
#endif

#if defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>(), sctl::Integer digits = -1,
          sctl::Integer ndim>
void st2d_local_kernel_directcp_vec_cpp_helper(const int32_t *nd, const Real *rsc, const Real *cen,
                                               const Real *bsizeinv, const Real *d2min, const Real *d2max,
                                               const Real *sources, const int32_t *ns, const Real *stoklet,
                                               const Real *xtarg, const Real *ytarg, const Real *ztarg,
                                               const int32_t *nt, Real *pot) {
    static constexpr sctl::Integer COORD_DIM = ndim; // ndim[0];
    constexpr sctl::Long nd_ = 1;                    // nd[0];
    sctl::Long Nsrc = ns[0];
    sctl::Long Ntrg = nt[0];
    sctl::Long Ntrg_ = ((Ntrg + MaxVecLen - 1) / MaxVecLen) * MaxVecLen;

    sctl::StaticArray<Real, 400 * COORD_DIM> buff0;
    sctl::StaticArray<Real, 400 * COORD_DIM> buff1;
    sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
    sctl::Matrix<Real> Vt(nd_ * COORD_DIM, Ntrg_, buff1, false);
    if (Ntrg_ > 400) {
        Xt.ReInit(COORD_DIM, Ntrg_);
        Vt.ReInit(nd_ * COORD_DIM, Ntrg_);
    }
    { // Set Xs, Vs, Xt, Vt
        std::memcpy(Xt[0], xtarg, sizeof(Real) * Ntrg);
        std::memcpy(Xt[1], ytarg, sizeof(Real) * Ntrg);
        Vt = 0;
    }

    constexpr auto cdata = []() {
        if constexpr (digits <= 3) {
            return std::make_pair(
                std::array<Real, 8>{1.846554578756552e-01, -2.966683058873768e-01, 2.187482696883298e-01,
                                    -1.930445665343498e-01, 1.152120060185630e-01, -4.624359846616554e-02,
                                    3.986446577184038e-02, -2.296552527129039e-02},
                std::array<Real, 8>{4.874864017381395e-01, 5.299044064212663e-02, -1.098956047309594e-01,
                                    1.382693723607076e-01, -9.272099440980629e-02, 3.856825860125045e-02,
                                    -3.459163080563133e-02, 2.031942894002221e-02});
        } else if constexpr (digits <= 6) {
            return std::make_pair(
                std::array<Real, 13>{1.742974848578507e-01, -2.573253389659778e-01, 1.497483703405333e-01,
                                     -1.352127710109274e-01, 1.381360809312198e-01, -1.310180186924878e-01,
                                     1.065981015210400e-01, -7.292086065349250e-02, 4.142209563747640e-02,
                                     -1.964709822677369e-02, 8.808445004019772e-03, -3.879150759306015e-03,
                                     9.924244727038691e-04},
                std::array<Real, 14>{4.989275970042895e-01, 7.861971816869706e-03, -2.698194194185921e-02,
                                     5.785339785044497e-02, -8.735759363134854e-02, 9.922761129588592e-02,
                                     -8.829289346280386e-02, 6.320603843280931e-02, -3.716611638604578e-02,
                                     1.848092169056624e-02, -8.180355441458238e-03, 3.238289075111389e-03,
                                     -9.485654024207217e-04, 1.319073772359878e-04});
        } else if constexpr (digits <= 9) {
            return std::make_pair(
                std::array<Real, 19>{
                    1.733906092107364e-01, -2.510413183117834e-01, 1.299689228965454e-01, -9.836547803209694e-02,
                    9.490748288758954e-02, -1.030238195403018e-01, 1.100358018444409e-01, -1.067108995165202e-01,
                    9.100532141533005e-02, -6.771478903661227e-02, 4.400096234001671e-02, -2.506248463691992e-02,
                    1.264633612426788e-02, -5.743944511033765e-03, 2.297657498033969e-03, -7.636590624624864e-04,
                    2.519553444802364e-04, -1.062770340590954e-04, 2.761976621518952e-05},
                std::array<Real, 19>{
                    4.998914591008903e-01, 1.095588747213744e-03, -5.269955173914959e-03, 1.611115959351134e-02,
                    -3.523064239855683e-02, 5.880539647249017e-02, -7.808753084994935e-02, 8.490446734143470e-02,
                    -7.723140148915500e-02, 5.976142098136458e-02, -3.983048500616800e-02, 2.308293440213639e-02,
                    -1.179441484626060e-02, 5.409162106188744e-03, -2.179104676649507e-03, 7.274978397333333e-04,
                    -2.412513081612349e-04, 1.023726763618472e-04, -2.667316339957590e-05});
        } else if constexpr (digits <= 12) {
            return std::make_pair(
                std::array<Real, 25>{
                    1.732983196849929e-01, -2.501464289483227e-01, 1.258945367320441e-01, -8.683602821002068e-02,
                    7.238738615135265e-02, -7.145254376219257e-02, 7.890525425702721e-02, -8.883419332369469e-02,
                    9.465145231459976e-02, -9.177610256521274e-02, 7.974444833050520e-02, -6.186954815871509e-02,
                    4.292612399557165e-02, -2.672824108341236e-02, 1.500078249666185e-02, -7.624355836844713e-03,
                    3.522043662308565e-03, -1.480734132654379e-03, 5.713150069276215e-04, -2.066574870600848e-04,
                    6.802971183257067e-05, -1.824115587354376e-05, 4.948292077662717e-06, -2.116521218676349e-06,
                    5.505494486025323e-07},
                std::array<Real, 25>{
                    4.999880537322993e-01,  1.524020822224470e-04, -9.356238208061141e-04, 3.686023567244900e-03,
                    -1.048563555422571e-02, 2.297970701545704e-02, -4.042620037941591e-02, 5.873973024957757e-02,
                    -7.199001885198975e-02, 7.563207375223308e-02, -6.899649230772067e-02, 5.523476882716163e-02,
                    -3.914544827522717e-02, 2.474436669033796e-02, -1.404292901409487e-02, 7.197583933083345e-03,
                    -3.345922009486318e-03, 1.414716095992157e-03, -5.492693792558703e-04, 1.988677482898063e-04,
                    -6.492302090540218e-05, 1.771179251074273e-05, -5.107593727824348e-06, 2.048245829228578e-06,
                    -4.835255036164593e-07});
        }
    }();
    const auto cdiag = cdata.first;
    const auto coffd = cdata.second;

    static constexpr sctl::Integer VecLen = MaxVecLen;
    using Vec = sctl::Vec<Real, VecLen>;
    Vec d2min_vec = d2min[0];
    Vec d2max_vec = d2max[0];
    Vec rsc_vec = rsc[0];
    Vec cen_vec = cen[0];
    Vec bsizeinv_vec = bsizeinv[0];
    Vec bsizeinv2_vec = bsizeinv_vec * bsizeinv_vec;

    // load stokslet density
    sctl::Matrix<Real> Vs_(Nsrc, nd_ * COORD_DIM, sctl::Ptr2Itr<Real>((Real *)stoklet, nd_ * Nsrc), false);
    // Vec Vsrc[Nsrc][nd_][COORD_DIM];
    sctl::Vector<Vec> Vsrc(Nsrc * nd_ * COORD_DIM);
    for (sctl::Long s = 0; s < Nsrc; s++) {
        for (long i = 0; i < nd_; i++) {
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0] = Vec::Load1(&Vs_[s][0 * nd_ + i]);
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1] = Vec::Load1(&Vs_[s][1 * nd_ + i]);
        }
    }

    // load source
    sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real *)sources, COORD_DIM * Nsrc), false);

//  TODO: reinit to sctl::Vector is over limit
#define NSRC_LIMIT 1000
    // work array memory stores compressed R2
    Real dX_work[COORD_DIM][NSRC_LIMIT * 8];
    Real R2_work[NSRC_LIMIT * 8];
    // sctl::StaticArray<Real, NSRC_LIMIT*8> R2_work_buff;
    //  why converting to sctl::Vector is a bit slow...
    // sctl::Vector<Real> R2_work(NSRC_LIMIT*8, R2_work_buff, false);
    //  work array memory stores compressed polynomial evals
    Real Fdiag_work[NSRC_LIMIT * 8];
    Real Foffd_work[NSRC_LIMIT * 8];
    // mask array
    typename Vec::MaskType Mask_work[NSRC_LIMIT];
    // sparse starting index of stored Pval
    int start_ind[NSRC_LIMIT];
    // sparse source point index
    int source_ind[NSRC_LIMIT];
    // zero vec
    static Vec zero_tmp = Vec::Zero();
    for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
        Vec Xtrg[COORD_DIM];
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
        }
        // load potential
        Vec Vtrg[nd_][COORD_DIM];
        for (long i = 0; i < nd_; i++) {
            Vtrg[i][0] = Vec::LoadAligned(&Vt[0 * nd_ + i][t]);
            Vtrg[i][1] = Vec::LoadAligned(&Vt[1 * nd_ + i][t]);
        }

        // int total_r = Nsrc*VecLen;
        int valid_r = 0;
        int source_cnt = 0;
        // store compressed R2 to explore 4/3 pi over 27 sparsity
        for (sctl::Long s = 0; s < Nsrc; s++) {
            Vec dX[COORD_DIM], R2 = zero_tmp;
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
                dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
                R2 += dX[k] * dX[k];
            }

            // store sparse data
            Mask_work[source_cnt] = (R2 >= d2min_vec) & (R2 <= d2max_vec);
            int valid_cnt = mask_popcnt(Mask_work[source_cnt]);
            mask_compress_store(Mask_work[source_cnt], dX[0], &dX_work[0][0] + valid_r);
            mask_compress_store(Mask_work[source_cnt], dX[1], &dX_work[1][0] + valid_r);
            mask_compress_store(Mask_work[source_cnt], R2, &R2_work[0] + valid_r);
            start_ind[source_cnt] = valid_r;
            source_ind[source_cnt] = s;
            source_cnt += (valid_cnt > 0);
            valid_r += valid_cnt;
        }

        // evaluate polynomial on compressed R2 and store in Pval
        for (sctl::Long s = 0; s < valid_r; s += VecLen) {
            Vec R2 = Vec::LoadAligned(&R2_work[0] + s);
            // evaluate the PSWF local kernel
            Vec one = 1.0e0;
            Vec R2inv = select((R2 >= d2min_vec) & (R2 <= d2max_vec), one / R2, Vec::Zero());
            Vec xtmp = FMA(R2, rsc_vec, cen_vec);
            Vec fdiag = EvalPolynomial(xtmp.get(), cdiag);
            Vec foffd = EvalPolynomial(xtmp.get(), coffd);
            Vec R2sc = R2 * bsizeinv2_vec;
            Vec Fdiag = select((R2 >= d2min_vec) & (R2 <= d2max_vec), -0.25e0 * sctl::log(R2sc) - fdiag, Vec::Zero());
            Vec Foffd = (0.5e0 - foffd) * R2inv;
            Fdiag.StoreAligned(Fdiag_work + s);
            Foffd.StoreAligned(Foffd_work + s);
        }

        // expand compressed Fdiag_work and Foffd_work, then multiply by charge to accumulate
        for (sctl::Long s_ind = 0; s_ind < source_cnt; s_ind++) {
            int s = source_ind[s_ind];
            int start = start_ind[s_ind];
            Vec Fdiag = mask_expand_load(Mask_work[s_ind], zero_tmp, Fdiag_work + start);
            Vec Foffd = mask_expand_load(Mask_work[s_ind], zero_tmp, Foffd_work + start);
            Vec dX[2];
            dX[0] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[0] + start);
            dX[1] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[1] + start);

            for (long i = 0; i < nd_; i++) {
                Vec Dprod = dX[0] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0] +
                            dX[1] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1];
                Vec pl = Foffd * Dprod;
                Vtrg[i][0] += pl * dX[0] + Fdiag * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0];
                Vtrg[i][1] += pl * dX[1] + Fdiag * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1];
            }
        }

        // store potential
        for (long i = 0; i < nd_; i++) {
            Vtrg[i][0].StoreAligned(&Vt[0 * nd_ + i][t]);
            Vtrg[i][1].StoreAligned(&Vt[1 * nd_ + i][t]);
        }
    }

    for (long i = 0; i < Ntrg; i++) {
        for (long j = 0; j < nd_ * COORD_DIM; j++) {
            pot[i * nd_ * COORD_DIM + j] += Vt[j][i];
        }
    }
}
#endif

template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void st2d_local_kernel_directcp_vec_cpp(const int32_t *nd, const int32_t *ndim, const int32_t *digits, const Real *rsc,
                                        const Real *cen, const Real *bsizeinv, const Real *d2min, const Real *d2max,
                                        const Real *sources, const int32_t *ns, const Real *charge, const Real *xtarg,
                                        const Real *ytarg, const Real *ztarg, const int32_t *nt, Real *pot) {
    if (digits[0] <= 3)
        st2d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 3, 2>(nd, rsc, cen, bsizeinv, d2min, d2max, sources,
                                                                         ns, charge, xtarg, ytarg, ztarg, nt, pot);
    else if (digits[0] <= 6)
        st2d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 6, 2>(nd, rsc, cen, bsizeinv, d2min, d2max, sources,
                                                                         ns, charge, xtarg, ytarg, ztarg, nt, pot);
    else if (digits[0] <= 9)
        st2d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 9, 2>(nd, rsc, cen, bsizeinv, d2min, d2max, sources,
                                                                         ns, charge, xtarg, ytarg, ztarg, nt, pot);
    else if (digits[0] <= 12)
        st2d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, 12, 2>(nd, rsc, cen, bsizeinv, d2min, d2max, sources,
                                                                          ns, charge, xtarg, ytarg, ztarg, nt, pot);
    else
        st2d_local_kernel_directcp_vec_cpp_helper<Real, MaxVecLen, -1, 2>(nd, rsc, cen, bsizeinv, d2min, d2max, sources,
                                                                          ns, charge, xtarg, ytarg, ztarg, nt, pot);
}

/* 2D Log local kernel dipole to potential */
#if !defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>(), sctl::Integer digits = -1,
          sctl::Integer ndim>
void log_local_kernel_directdp_vec_cpp_helper(const int32_t *nd, const Real *rsc, const Real *cen, const Real *d2min,
                                              const Real *d2max, const Real *sources, const int32_t *ns,
                                              const Real *dipvec, const Real *xtarg, const Real *ytarg,
                                              const int32_t *nt, Real *pot) {
    static constexpr sctl::Integer COORD_DIM = ndim; // ndim[0];
    constexpr sctl::Long nd_ = 1;                    // nd[0];
    sctl::Long Nsrc = ns[0];
    sctl::Long Ntrg = nt[0];
    sctl::Long Ntrg_ = ((Ntrg + MaxVecLen - 1) / MaxVecLen) * MaxVecLen;

    sctl::StaticArray<Real, 400 * COORD_DIM> buff0;
    sctl::StaticArray<Real, 400> buff1;
    sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
    sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
    if (Ntrg_ > 400) {
        Xt.ReInit(COORD_DIM, Ntrg_);
        Vt.ReInit(nd_, Ntrg_);
    }
    { // Set Xs, Vs, Xt, Vt
        std::memcpy(Xt[0], xtarg, sizeof(Real) * Ntrg);
        std::memcpy(Xt[1], ytarg, sizeof(Real) * Ntrg);
        Vt = 0;
    }

    constexpr auto coefs = []() {
        if constexpr (digits <= 3) {
            return std::array<Real, 7>{9.334285864639144e-01, 2.169971553925394e-01,  -3.006861735844956e-01,
                                       2.360122612383026e-01, -1.211621477124991e-01, 4.683707884859314e-02,
                                       -1.169643948451524e-02};
        } else if constexpr (digits <= 6) {
            return std::array<Real, 12>{9.922973647646393e-01,  4.517941536621922e-02,  -1.212953577289340e-01,
                                        1.989164113250858e-01,  -2.245055284970060e-01, 1.865422171066172e-01,
                                        -1.193543856601842e-01, 6.036222021672452e-02,  -2.433775183640557e-02,
                                        8.334389683592744e-03,  -2.804387273230964e-03, 6.652266634776942e-04};
        } else if constexpr (digits <= 9) {
            return std::array<Real, 18>{9.990603915706698e-01,  7.897602108744739e-03,  -3.122669887520646e-02,
                                        7.746531848059268e-02,  -1.357139968250016e-01, 1.792398837538570e-01,
                                        -1.860703309800472e-01, 1.563404194632586e-01,  -1.086719295561396e-01,
                                        6.356400646965495e-02,  -3.171623791444703e-02, 1.366331708296591e-02,
                                        -5.141772077832021e-03, 1.695376406990721e-03,  -4.834996012361570e-04,
                                        1.266855672999249e-04,  -3.592580267758308e-05, 7.390593201473847e-06};
        } else if constexpr (digits <= 12) {
            return std::array<Real, 23>{
                9.998827855396359e-01,  1.279729485988923e-03, -6.667320227000562e-03, 2.210425656192649e-02,
                -5.247441065369307e-02, 9.517926389227593e-02, -1.374603602142662e-01, 1.626721895736130e-01,
                -1.611253399469532e-01, 1.357864486500207e-01, -9.865004461170723e-02, 6.245699337710517e-02,
                -3.477478798108834e-02, 1.716122552233221e-02, -7.556538868258740e-03, 2.985503961942498e-03,
                -1.065227945902495e-03, 3.461493082345737e-04, -1.017890137358330e-04, 2.630178122754457e-05,
                -6.571760731648724e-06, 1.937885323237026e-06, -3.943162901113174e-07};
        }
    }();

    static constexpr sctl::Integer VecLen = MaxVecLen;
    using Vec = sctl::Vec<Real, VecLen>;
    Vec d2min_vec = d2min[0];
    Vec d2max_vec = d2max[0];
    Vec rsc_vec = rsc[0];
    Vec cen_vec = cen[0];

    // load dipoles
    sctl::Matrix<Real> Vs_(Nsrc, nd_ * COORD_DIM, sctl::Ptr2Itr<Real>((Real *)dipvec, nd_ * Nsrc), false);
    // Vec Vsrc[Nsrc][nd_][COORD_DIM];
    sctl::Vector<Vec> Vsrc(Nsrc * nd_ * COORD_DIM);
    for (sctl::Long s = 0; s < Nsrc; s++) {
        for (long i = 0; i < nd_; i++) {
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0] = Vec::Load1(&Vs_[s][0 * nd_ + i]);
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1] = Vec::Load1(&Vs_[s][1 * nd_ + i]);
        }
    }

    // load source
    sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real *)sources, COORD_DIM * Nsrc), false);
    for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
        Vec Xtrg[COORD_DIM];
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
        }
        // load potential
        Vec Vtrg[nd_];
        for (long i = 0; i < nd_; i++) {
            Vtrg[i] = Vec::LoadAligned(&Vt[i][t]);
        }

        for (sctl::Long s = 0; s < Nsrc; s++) {
            Vec dX[COORD_DIM], R2 = Vec::Zero();
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
                dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
                R2 += dX[k] * dX[k];
            }
            // evaluate the PSWF local kernel
            Vec one = 1.0e0;
            Vec R2inv = select((R2 >= d2min_vec) & (R2 <= d2max_vec), one / R2, Vec::Zero());
            Vec xtmp = FMA(R2, rsc_vec, cen_vec);
            Vec fres = EvalPolynomial(xtmp.get(), coefs);

            Vec Fval = (fres - one) * R2inv;

            for (long i = 0; i < nd_; i++) {
                Vec Dprod = dX[0] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0] +
                            dX[1] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1];
                Vtrg[i] += Fval * Dprod;
            }
        }
        // store potential
        for (long i = 0; i < nd_; i++) {
            Vtrg[i].StoreAligned(&Vt[i][t]);
        }
    }

    for (long i = 0; i < Ntrg; i++) {
        for (long j = 0; j < nd_; j++) {
            pot[i * nd_ + j] += Vt[j][i];
        }
    }
}
#endif

#if defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>(), sctl::Integer digits = -1,
          sctl::Integer ndim>
void log_local_kernel_directdp_vec_cpp_helper(const int32_t *nd, const Real *rsc, const Real *cen, const Real *d2min,
                                              const Real *d2max, const Real *sources, const int32_t *ns,
                                              const Real *dipvec, const Real *xtarg, const Real *ytarg,
                                              const int32_t *nt, Real *pot) {
    static constexpr sctl::Integer COORD_DIM = ndim; // ndim[0];
    constexpr sctl::Long nd_ = 1;                    // nd[0];
    sctl::Long Nsrc = ns[0];
    sctl::Long Ntrg = nt[0];
    sctl::Long Ntrg_ = ((Ntrg + MaxVecLen - 1) / MaxVecLen) * MaxVecLen;

    sctl::StaticArray<Real, 400 * COORD_DIM> buff0;
    sctl::StaticArray<Real, 400> buff1;
    sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
    sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
    if (Ntrg_ > 400) {
        Xt.ReInit(COORD_DIM, Ntrg_);
        Vt.ReInit(nd_, Ntrg_);
    }
    { // Set Xs, Vs, Xt, Vt
        std::memcpy(Xt[0], xtarg, sizeof(Real) * Ntrg);
        std::memcpy(Xt[1], ytarg, sizeof(Real) * Ntrg);
        Vt = 0;
    }

    constexpr auto coefs = []() {
        if constexpr (digits <= 3) {
            return std::array<Real, 7>{9.334285864639144e-01, 2.169971553925394e-01,  -3.006861735844956e-01,
                                       2.360122612383026e-01, -1.211621477124991e-01, 4.683707884859314e-02,
                                       -1.169643948451524e-02};
        } else if constexpr (digits <= 6) {
            return std::array<Real, 12>{9.922973647646393e-01,  4.517941536621922e-02,  -1.212953577289340e-01,
                                        1.989164113250858e-01,  -2.245055284970060e-01, 1.865422171066172e-01,
                                        -1.193543856601842e-01, 6.036222021672452e-02,  -2.433775183640557e-02,
                                        8.334389683592744e-03,  -2.804387273230964e-03, 6.652266634776942e-04};
        } else if constexpr (digits <= 9) {
            return std::array<Real, 18>{9.990603915706698e-01,  7.897602108744739e-03,  -3.122669887520646e-02,
                                        7.746531848059268e-02,  -1.357139968250016e-01, 1.792398837538570e-01,
                                        -1.860703309800472e-01, 1.563404194632586e-01,  -1.086719295561396e-01,
                                        6.356400646965495e-02,  -3.171623791444703e-02, 1.366331708296591e-02,
                                        -5.141772077832021e-03, 1.695376406990721e-03,  -4.834996012361570e-04,
                                        1.266855672999249e-04,  -3.592580267758308e-05, 7.390593201473847e-06};
        } else if constexpr (digits <= 12) {
            return std::array<Real, 23>{
                9.998827855396359e-01,  1.279729485988923e-03, -6.667320227000562e-03, 2.210425656192649e-02,
                -5.247441065369307e-02, 9.517926389227593e-02, -1.374603602142662e-01, 1.626721895736130e-01,
                -1.611253399469532e-01, 1.357864486500207e-01, -9.865004461170723e-02, 6.245699337710517e-02,
                -3.477478798108834e-02, 1.716122552233221e-02, -7.556538868258740e-03, 2.985503961942498e-03,
                -1.065227945902495e-03, 3.461493082345737e-04, -1.017890137358330e-04, 2.630178122754457e-05,
                -6.571760731648724e-06, 1.937885323237026e-06, -3.943162901113174e-07};
        }
    }();

    static constexpr sctl::Integer VecLen = MaxVecLen;
    using Vec = sctl::Vec<Real, VecLen>;
    Vec d2min_vec = d2min[0];
    Vec d2max_vec = d2max[0];
    Vec rsc_vec = rsc[0];
    Vec cen_vec = cen[0];

    // load stokslet density
    sctl::Matrix<Real> Vs_(Nsrc, nd_ * COORD_DIM, sctl::Ptr2Itr<Real>((Real *)dipvec, nd_ * Nsrc), false);
    // Vec Vsrc[Nsrc][nd_][COORD_DIM];
    sctl::Vector<Vec> Vsrc(Nsrc * nd_ * COORD_DIM);
    for (sctl::Long s = 0; s < Nsrc; s++) {
        for (long i = 0; i < nd_; i++) {
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0] = Vec::Load1(&Vs_[s][0 * nd_ + i]);
            Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1] = Vec::Load1(&Vs_[s][1 * nd_ + i]);
        }
    }

    // load source
    sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real *)sources, COORD_DIM * Nsrc), false);

//  TODO: reinit to sctl::Vector is over limit
#define NSRC_LIMIT 1000
    // work array memory stores compressed R2
    Real dX_work[COORD_DIM][NSRC_LIMIT * 8];
    Real R2_work[NSRC_LIMIT * 8];
    // sctl::StaticArray<Real, NSRC_LIMIT*8> R2_work_buff;
    //  why converting to sctl::Vector is a bit slow...
    // sctl::Vector<Real> R2_work(NSRC_LIMIT*8, R2_work_buff, false);
    //  work array memory stores compressed polynomial evals
    Real Fres_work[NSRC_LIMIT * 8];
    // mask array
    typename Vec::MaskType Mask_work[NSRC_LIMIT];
    // sparse starting index of stored Pval
    int start_ind[NSRC_LIMIT];
    // sparse source point index
    int source_ind[NSRC_LIMIT];
    // zero vec
    static Vec zero_tmp = Vec::Zero();
    for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
        Vec Xtrg[COORD_DIM];
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
        }
        // load potential
        Vec Vtrg[nd_];
        for (long i = 0; i < nd_; i++) {
            Vtrg[i] = Vec::LoadAligned(&Vt[i][t]);
        }

        // int total_r = Nsrc*VecLen;
        int valid_r = 0;
        int source_cnt = 0;
        // store compressed R2 to explore 4/3 pi over 27 sparsity
        for (sctl::Long s = 0; s < Nsrc; s++) {
            Vec dX[COORD_DIM], R2 = zero_tmp;
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
                dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
                R2 += dX[k] * dX[k];
            }

            // store sparse data
            Mask_work[source_cnt] = (R2 >= d2min_vec) & (R2 <= d2max_vec);
            int valid_cnt = mask_popcnt(Mask_work[source_cnt]);
            mask_compress_store(Mask_work[source_cnt], dX[0], &dX_work[0][0] + valid_r);
            mask_compress_store(Mask_work[source_cnt], dX[1], &dX_work[1][0] + valid_r);
            mask_compress_store(Mask_work[source_cnt], R2, &R2_work[0] + valid_r);
            start_ind[source_cnt] = valid_r;
            source_ind[source_cnt] = s;
            source_cnt += (valid_cnt > 0);
            valid_r += valid_cnt;
        }

        // evaluate polynomial on compressed R2 and store in Pval
        for (sctl::Long s = 0; s < valid_r; s += VecLen) {
            Vec R2 = Vec::LoadAligned(&R2_work[0] + s);
            // evaluate the PSWF local kernel
            Vec one = 1.0e0;
            Vec R2inv = select((R2 >= d2min_vec) & (R2 <= d2max_vec), one / R2, Vec::Zero());
            Vec xtmp = FMA(R2, rsc_vec, cen_vec);
            Vec fres = EvalPolynomial(xtmp.get(), coefs);

            Vec Fres = (fres - one) * R2inv;
            Fres.StoreAligned(Fres_work + s);
        }

        // expand compressed Fdiag_work and Foffd_work, then multiply by charge to accumulate
        for (sctl::Long s_ind = 0; s_ind < source_cnt; s_ind++) {
            int s = source_ind[s_ind];
            int start = start_ind[s_ind];
            Vec Fres = mask_expand_load(Mask_work[s_ind], zero_tmp, Fres_work + start);
            Vec dX[2];
            dX[0] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[0] + start);
            dX[1] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[1] + start);

            for (long i = 0; i < nd_; i++) {
                Vec Dprod = dX[0] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 0] +
                            dX[1] * Vsrc[s * nd_ * COORD_DIM + i * COORD_DIM + 1];
                Vtrg[i] += Fres * Dprod;
            }
        }

        // store potential
        for (long i = 0; i < nd_; i++) {
            Vtrg[i].StoreAligned(&Vt[i][t]);
        }
    }

    for (long i = 0; i < Ntrg; i++) {
        for (long j = 0; j < nd_; j++) {
            pot[i * nd_ + j] += Vt[j][i];
        }
    }
}
#endif

template <class Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void log_local_kernel_directdp_vec_cpp(const int32_t *nd, const int32_t *ndim, const int32_t *digits, const Real *rsc,
                                       const Real *cen, const Real *d2min, const Real *d2max, const Real *sources,
                                       const int32_t *ns, const Real *dipvec, const Real *xtarg, const Real *ytarg,
                                       const int32_t *nt, Real *pot) {
    if (digits[0] <= 3)
        log_local_kernel_directdp_vec_cpp_helper<Real, MaxVecLen, 3, 2>(nd, rsc, cen, d2min, d2max, sources, ns, dipvec,
                                                                        xtarg, ytarg, nt, pot);
    else if (digits[0] <= 6)
        log_local_kernel_directdp_vec_cpp_helper<Real, MaxVecLen, 6, 2>(nd, rsc, cen, d2min, d2max, sources, ns, dipvec,
                                                                        xtarg, ytarg, nt, pot);
    else if (digits[0] <= 9)
        log_local_kernel_directdp_vec_cpp_helper<Real, MaxVecLen, 9, 2>(nd, rsc, cen, d2min, d2max, sources, ns, dipvec,
                                                                        xtarg, ytarg, nt, pot);
    else if (digits[0] <= 12)
        log_local_kernel_directdp_vec_cpp_helper<Real, MaxVecLen, 12, 2>(nd, rsc, cen, d2min, d2max, sources, ns,
                                                                         dipvec, xtarg, ytarg, nt, pot);
    else
        log_local_kernel_directdp_vec_cpp_helper<Real, MaxVecLen, -1, 2>(nd, rsc, cen, d2min, d2max, sources, ns,
                                                                         dipvec, xtarg, ytarg, nt, pot);
}
} // namespace dmk

#endif
