#ifndef _VEC_KERNELS_HPP_
#define _VEC_KERNELS_HPP_

#define NDEBUG
#define NSRC_LIMIT 3000
#include <sctl.hpp>
#include <sctllog.h>
// for stokesdmk7.f
template <class Real, class VecType, sctl::Integer DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer SCDIM, class uKernel, sctl::Integer digits> struct uKerHelper {
  template <class CtxType> static inline void Eval(VecType* vt, const VecType (&dX)[DIM], const Real* vs, const sctl::Integer nd, const CtxType& ctx) {
    VecType M[KDIM0][KDIM1][SCDIM];
    uKernel::template uKerMatrix<digits>(M, dX, ctx);
    for (sctl::Integer i = 0; i < nd; i++) {
      const Real* vs_ = vs+i*SCDIM;
      for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
        VecType* vt_ = vt+(k1*nd+i)*SCDIM;
        for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) {
          const VecType vs0(vs_[(k0*nd)*SCDIM+0]);
          vt_[0] = FMA(M[k0][k1][0], vs0, vt_[0]);
          if (SCDIM == 2) {
            const VecType vs1(vs_[(k0*nd)*SCDIM+1]);
            vt_[0] = FMA(M[k0][k1][1],-vs1, vt_[0]);
            vt_[1] = FMA(M[k0][k1][1], vs0, vt_[1]);
            vt_[1] = FMA(M[k0][k1][0], vs1, vt_[1]);
          }
        }
      }
    }
  }
  template <sctl::Integer nd, class CtxType> static inline void EvalND(VecType* vt, const VecType (&dX)[DIM], const Real* vs, const CtxType& ctx) {
    VecType M[KDIM0][KDIM1][SCDIM];
    uKernel::template uKerMatrix<digits>(M, dX, ctx);
    for (sctl::Integer i = 0; i < nd; i++) {
      const Real* vs_ = vs+i*SCDIM;
      for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
        VecType* vt_ = vt+(k1*nd+i)*SCDIM;
        for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) {
          const VecType vs0(vs_[(k0*nd)*SCDIM+0]);
          vt_[0] = FMA(M[k0][k1][0], vs0, vt_[0]);
          if (SCDIM == 2) {
            const VecType vs1(vs_[(k0*nd)*SCDIM+1]);
            vt_[0] = FMA(M[k0][k1][1],-vs1, vt_[0]);
            vt_[1] = FMA(M[k0][k1][1], vs0, vt_[1]);
            vt_[1] = FMA(M[k0][k1][0], vs1, vt_[1]);
          }
        }
      }
    }
  }
};

template <class uKernel> class GenericKernel : public uKernel {
    static constexpr sctl::Integer VecLen = uKernel::VecLen;
    using VecType = typename uKernel::VecType;
    using Real = typename uKernel::RealType;

    template <sctl::Integer K0, sctl::Integer K1, sctl::Integer Q, sctl::Integer D, class ...T> static constexpr sctl::Integer get_DIM  (void (*uKer)(VecType (&M)[K0][K1][Q], const VecType (&r)[D], T... args)) { return D;  }
    template <sctl::Integer K0, sctl::Integer K1, sctl::Integer Q, sctl::Integer D, class ...T> static constexpr sctl::Integer get_SCDIM(void (*uKer)(VecType (&M)[K0][K1][Q], const VecType (&r)[D], T... args)) { return Q;  }
    template <sctl::Integer K0, sctl::Integer K1, sctl::Integer Q, sctl::Integer D, class ...T> static constexpr sctl::Integer get_KDIM0(void (*uKer)(VecType (&M)[K0][K1][Q], const VecType (&r)[D], T... args)) { return K0; }
    template <sctl::Integer K0, sctl::Integer K1, sctl::Integer Q, sctl::Integer D, class ...T> static constexpr sctl::Integer get_KDIM1(void (*uKer)(VecType (&M)[K0][K1][Q], const VecType (&r)[D], T... args)) { return K1; }

    static constexpr sctl::Integer DIM   = get_DIM  (uKernel::template uKerMatrix<0,GenericKernel>);
    static constexpr sctl::Integer SCDIM = get_SCDIM(uKernel::template uKerMatrix<0,GenericKernel>);
    static constexpr sctl::Integer KDIM0 = get_KDIM0(uKernel::template uKerMatrix<0,GenericKernel>);
    static constexpr sctl::Integer KDIM1 = get_KDIM1(uKernel::template uKerMatrix<0,GenericKernel>);

  public:

    GenericKernel() : ctx_ptr(this) {}

    static constexpr sctl::Integer CoordDim() {
      return DIM;
    }
    static constexpr sctl::Integer SrcDim() {
      return KDIM0*SCDIM;
    }
    static constexpr sctl::Integer TrgDim() {
      return KDIM1*SCDIM;
    }

    template <bool enable_openmp=false, sctl::Integer digits=-1> void Eval(sctl::Vector<sctl::Vector<Real>>& v_trg_, const sctl::Vector<Real>& r_trg, const sctl::Vector<Real>& r_src, const sctl::Vector<sctl::Vector<Real>>& v_src_, const sctl::Integer nd) const {
      if      (nd == 1) EvalHelper<enable_openmp, digits, 1>(v_trg_, r_trg, r_src, v_src_, nd);
      else if (nd == 2) EvalHelper<enable_openmp, digits, 2>(v_trg_, r_trg, r_src, v_src_, nd);
      else if (nd == 3) EvalHelper<enable_openmp, digits, 3>(v_trg_, r_trg, r_src, v_src_, nd);
      else if (nd == 4) EvalHelper<enable_openmp, digits, 4>(v_trg_, r_trg, r_src, v_src_, nd);
      else if (nd == 5) EvalHelper<enable_openmp, digits, 5>(v_trg_, r_trg, r_src, v_src_, nd);
      else if (nd == 6) EvalHelper<enable_openmp, digits, 6>(v_trg_, r_trg, r_src, v_src_, nd);
      else if (nd == 7) EvalHelper<enable_openmp, digits, 7>(v_trg_, r_trg, r_src, v_src_, nd);
      else if (nd == 8) EvalHelper<enable_openmp, digits, 8>(v_trg_, r_trg, r_src, v_src_, nd);
      else  EvalHelper<enable_openmp, digits, 0>(v_trg_, r_trg, r_src, v_src_, nd);
    }

  private:

    template <bool enable_openmp=false, sctl::Integer digits=-1, sctl::Integer ND=0> void EvalHelper(sctl::Vector<sctl::Vector<Real>>& v_trg_, const sctl::Vector<Real>& r_trg, const sctl::Vector<Real>& r_src, const sctl::Vector<sctl::Vector<Real>>& v_src_, const sctl::Integer nd) const {
      static constexpr sctl::Integer digits_ = (digits==-1 ? (sctl::Integer)(sctl::TypeTraits<Real>::SigBits*0.3010299957) : digits);
      auto uKerEval = [this](VecType* vt, const VecType (&dX)[DIM], const Real* vs, const sctl::Integer nd) {
        if (ND > 0) uKerHelper<Real,VecType,DIM,KDIM0,KDIM1,SCDIM,uKernel,digits_>::template EvalND<ND>(vt, dX, vs, *this);
        else uKerHelper<Real,VecType,DIM,KDIM0,KDIM1,SCDIM,uKernel,digits_>::Eval(vt, dX, vs, nd, *this);
      };

      const sctl::Long Ns = r_src.Dim() / DIM;
      const sctl::Long Nt = r_trg.Dim() / DIM;
      SCTL_ASSERT(r_trg.Dim() == Nt*DIM);
      SCTL_ASSERT(r_src.Dim() == Ns*DIM);

      sctl::Vector<sctl::Long> src_cnt(v_src_.Dim()), src_dsp(v_src_.Dim()); src_dsp = 0;
      sctl::Vector<sctl::Long> trg_cnt(v_trg_.Dim()), trg_dsp(v_trg_.Dim()); trg_dsp = 0;
      for (sctl::Integer i = 0; i < trg_cnt.Dim(); i++) {
        trg_cnt[i] = v_trg_[i].Dim()/Nt;
        trg_dsp[i] = (i ? trg_dsp[i-1]+trg_cnt[i-1] : 0);
      }
      for (sctl::Integer i = 0; i < src_cnt.Dim(); i++) {
        src_cnt[i] = v_src_[i].Dim()/Ns;
        src_dsp[i] = (i ? src_dsp[i-1]+src_cnt[i-1] : 0);
      }
      SCTL_ASSERT(src_cnt[src_cnt.Dim()-1] + src_dsp[src_dsp.Dim()-1] == SrcDim()*nd);
      SCTL_ASSERT(trg_cnt[trg_cnt.Dim()-1] + trg_dsp[trg_dsp.Dim()-1] == TrgDim()*nd);

      sctl::Vector<Real> v_src(Ns*SrcDim()*nd);
      for (sctl::Integer j = 0; j < src_cnt.Dim(); j++) {
        const sctl::Integer src_cnt_ = src_cnt[j];
        const sctl::Integer src_dsp_ = src_dsp[j];
        for (sctl::Integer k = 0; k < src_cnt_; k++) {
          for (sctl::Long i = 0; i < Ns; i++) {
            v_src[i*SrcDim()*nd+src_dsp_+k] = v_src_[j][i*src_cnt_+k];
          }
        }
      }

      const sctl::Long NNt = ((Nt + VecLen - 1) / VecLen) * VecLen;
      //if (NNt == VecLen) {
      //  VecType xt[DIM], vt[KDIM1], xs[DIM];
      //  Real vs[KDIM0];
      //  for (sctl::Integer k = 0; k < KDIM1; k++) vt[k] = VecType::Zero();
      //  for (sctl::Integer k = 0; k < DIM; k++) {
      //    alignas(sizeof(VecType)) sctl::StaticArray<Real,VecLen> Xt;
      //    VecType::Zero().StoreAligned(&Xt[0]);
      //    for (sctl::Integer i = 0; i < Nt; i++) Xt[i] = r_trg[i*DIM+k];
      //    xt[k] = VecType::LoadAligned(&Xt[0]);
      //  }
      //  for (sctl::Long s = 0; s < Ns; s++) {
      //    for (sctl::Integer k = 0; k < DIM; k++) xs[k] = VecType::Load1(&r_src[s*DIM+k]);
      //    for (sctl::Integer k = 0; k < KDIM0; k++) vs[k] = v_src[s*KDIM0+k];
      //    uKerEval(vt, xt, xs, vs, nd);
      //  }
      //  for (sctl::Integer k = 0; k < KDIM1; k++) {
      //    alignas(sizeof(VecType)) sctl::StaticArray<Real,VecLen> out;
      //    vt[k].StoreAligned(&out[0]);
      //    for (sctl::Long t = 0; t < Nt; t++) {
      //      v_trg[t*KDIM1+k] += out[t] * uKernel::uKerScaleFactor();
      //    }
      //  }
      //} else
      {
        const sctl::Matrix<Real> Xs_(Ns, DIM, (sctl::Iterator<Real>)r_src.begin(), false);
        const sctl::Matrix<Real> Vs_(Ns, SrcDim()*nd, (sctl::Iterator<Real>)v_src.begin(), false);

        sctl::Matrix<Real> Xt_(DIM, NNt), Vt_(TrgDim()*nd, NNt);
        for (sctl::Long k = 0; k < DIM; k++) { // Set Xt_
          for (sctl::Long i = 0; i < Nt; i++) {
            Xt_[k][i] = r_trg[i*DIM+k];
          }
          for (sctl::Long i = Nt; i < NNt; i++) {
            Xt_[k][i] = 0;
          }
        }
        if (enable_openmp) { // Compute Vt_
          #pragma omp parallel for schedule(static)
          for (sctl::Long t = 0; t < NNt; t += VecLen) {
            VecType xt[DIM], vt[TrgDim()*nd];
            for (sctl::Integer k = 0; k < TrgDim()*nd; k++) vt[k] = VecType::Zero();
            for (sctl::Integer k = 0; k < DIM; k++) xt[k] = VecType::LoadAligned(&Xt_[k][t]);

            for (sctl::Long s = 0; s < Ns; s++) {
              VecType dX[DIM];
              for (sctl::Integer k = 0; k < DIM; k++) dX[k] = xt[k] - Xs_[s][k];
              uKerEval(vt, dX, &Vs_[s][0], nd);
            }
            for (sctl::Integer k = 0; k < TrgDim()*nd; k++) vt[k].StoreAligned(&Vt_[k][t]);
          }
        } else {
          for (sctl::Long t = 0; t < NNt; t += VecLen) {
            VecType xt[DIM], vt[TrgDim()*nd];
            for (sctl::Integer k = 0; k < TrgDim()*nd; k++) vt[k] = VecType::Zero();
            for (sctl::Integer k = 0; k < DIM; k++) xt[k] = VecType::LoadAligned(&Xt_[k][t]);

            for (sctl::Long s = 0; s < Ns; s++) {
              VecType dX[DIM];
              for (sctl::Integer k = 0; k < DIM; k++) dX[k] = xt[k] - Xs_[s][k];
              uKerEval(vt, dX, &Vs_[s][0], nd);
            }
            for (sctl::Integer k = 0; k < TrgDim()*nd; k++) vt[k].StoreAligned(&Vt_[k][t]);
          }
        }

        for (sctl::Integer j = 0; j < trg_cnt.Dim(); j++) {
          const sctl::Integer trg_cnt_ = trg_cnt[j];
          const sctl::Integer trg_dsp_ = trg_dsp[j];
          for (sctl::Long i = 0; i < Nt; i++) {
            for (sctl::Integer k = 0; k < trg_cnt_; k++) {
              v_trg_[j][i*trg_cnt_+k] += Vt_[trg_dsp_+k][i] * uKernel::uKerScaleFactor();
            }
          }
        }
      }

    }

    void* ctx_ptr;
};


template <class Real, sctl::Integer VecLen_, sctl::Integer chrg, sctl::Integer dipo, sctl::Integer poten, sctl::Integer grad> struct Helmholtz3D {
  static constexpr sctl::Integer VecLen = VecLen_;
  using VecType = sctl::Vec<Real, VecLen>;
  using RealType = Real;

  VecType thresh2;
  VecType zk[2];

  static constexpr Real uKerScaleFactor() {
    return 1;
  }
  template <sctl::Integer digits, class CtxType> static inline void uKerMatrix(VecType (&M)[chrg+dipo][poten+grad][2], const VecType (&dX)[3], const CtxType& ctx) {
    using RealType = typename VecType::ScalarType;
    static constexpr sctl::Integer COORD_DIM = 3;

    const VecType& thresh2 = ctx.thresh2;
    const VecType (&zk)[2] = ctx.zk;

    const VecType R2 = dX[0]*dX[0]+dX[1]*dX[1]+dX[2]*dX[2];
    const VecType Rinv = sctl::approx_rsqrt<digits>(R2, (R2 > thresh2));
    const VecType Rinv2 = Rinv * Rinv;

    const VecType R = R2 * Rinv;
    const VecType izkR[2] = {-zk[1]*R, zk[0]*R};

    VecType sin_izkR, cos_izkR;
    sctl::approx_sincos<digits>(sin_izkR, cos_izkR, izkR[1]);
    const VecType exp_izkR = sctl::approx_exp<digits>(izkR[0]);

    // exp(ikr)/r
    const VecType G0 = cos_izkR * exp_izkR * Rinv;
    const VecType G1 = sin_izkR * exp_izkR * Rinv;

    // (1-ikr)*exp(ikr)/r^3
    const VecType H0 = ((izkR[0]-(RealType)1)*G0 - izkR[1]*G1) * Rinv2;
    const VecType H1 = ((izkR[0]-(RealType)1)*G1 + izkR[1]*G0) * Rinv2;

    const VecType tmp0 = (-3.0)*(Rinv*zk[1]+Rinv2) - zk[1]*zk[1] + zk[0]*zk[0];
    const VecType tmp1 = (3.0)*Rinv*zk[0] - zk[0]*zk[1]*(-2.0);
    const VecType J0 = (G0 * tmp0 - G1 * tmp1)*Rinv2;
    const VecType J1 = (G1 * tmp0 + G0 * tmp1)*Rinv2;

    if (chrg && poten) { // charge potential
      M[0][0][0] =  G0;
      M[0][0][1] =  G1;
    }

    if (chrg && grad) { // charge gradient
      for (sctl::Integer i = 0; i < COORD_DIM; i++){
        M[0][poten+i][0] = H0*dX[i];
        M[0][poten+i][1] = H1*dX[i];
      }
    }

    if (dipo && poten) { // dipole potential
      for (sctl::Integer i = 0; i < COORD_DIM; i++){
        M[chrg+i][0][0] = -H0*dX[i];
        M[chrg+i][0][1] = -H1*dX[i];
      }
    }

    if (dipo && grad) { // dipole gradient
      for (sctl::Integer i = 0; i < COORD_DIM; i++){
        const VecType J0_dXi = J0*dX[i];
        const VecType J1_dXi = J1*dX[i];
        for (sctl::Integer j = 0; j < COORD_DIM; j++){
          M[chrg+i][poten+j][0] = (i==j ? J0_dXi*dX[j]-H0 : J0_dXi*dX[j]);
          M[chrg+i][poten+j][1] = (i==j ? J1_dXi*dX[j]-H1 : J1_dXi*dX[j]);
        }
      }
    }
  }
};

template <class Real, sctl::Integer VecLen, sctl::Integer chrg, sctl::Integer dipo, sctl::Integer poten, sctl::Integer grad> static void EvalHelmholtz(sctl::Vector<sctl::Vector<Real>>& v_trg, const sctl::Vector<Real>& r_trg, const sctl::Vector<Real>& r_src, const sctl::Vector<sctl::Vector<Real>>& v_src, const sctl::Integer nd, const Real* zk, const Real thresh, const sctl::Integer digits) {
  GenericKernel<Helmholtz3D<Real,VecLen, chrg,dipo,poten,grad>> ker;
  ker.thresh2 = thresh*thresh;
  ker.zk[0] = zk[0];
  ker.zk[1] = zk[1];

  if (digits < 0) ker.template Eval<true, -1>(v_trg, r_trg, r_src, v_src, nd);
  else if (digits <= 3) ker.template Eval<true, 3>(v_trg, r_trg, r_src, v_src, nd);
  else if (digits <= 6) ker.template Eval<true, 6>(v_trg, r_trg, r_src, v_src, nd);
  else if (digits <= 9) ker.template Eval<true, 9>(v_trg, r_trg, r_src, v_src, nd);
  else if (digits <=12) ker.template Eval<true,12>(v_trg, r_trg, r_src, v_src, nd);
  else if (digits <=15) ker.template Eval<true,15>(v_trg, r_trg, r_src, v_src, nd);
  else ker.template Eval<true,-1>(v_trg, r_trg, r_src, v_src, nd);
}


template <class Real, sctl::Integer VecLen_, sctl::Integer chrg, sctl::Integer dipo, sctl::Integer poten, sctl::Integer grad> struct Laplace3D {
  static constexpr sctl::Integer VecLen = VecLen_;
  using VecType = sctl::Vec<Real, VecLen>;
  using RealType = Real;

  VecType thresh2;

  static constexpr Real uKerScaleFactor() {
    return 1;
  }
  template <sctl::Integer digits, class CtxType> static inline void uKerMatrix(VecType (&M)[chrg+dipo][poten+grad][1], const VecType (&dX)[3], const CtxType& ctx) {
    using RealType = typename VecType::ScalarType;
    static constexpr sctl::Integer COORD_DIM = 3;

    const VecType& thresh2 = ctx.thresh2;

    const VecType R2 = dX[0]*dX[0]+dX[1]*dX[1]+dX[2]*dX[2];
    const VecType Rinv = sctl::approx_rsqrt<digits>(R2, (R2 > thresh2));
    const VecType Rinv2 = Rinv * Rinv;
    const VecType Rinv3 = Rinv * Rinv2;

    if (chrg && poten) { // charge potential
      M[0][0][0] = Rinv;
    }

    if (chrg && grad) { // charge gradient
      for (sctl::Integer i = 0; i < COORD_DIM; i++){
        M[0][poten+i][0] = -Rinv3*dX[i];
      }
    }

    if (dipo && poten) { // dipole potential
      for (sctl::Integer i = 0; i < COORD_DIM; i++){
        M[chrg+i][0][0] = Rinv3*dX[i];
      }
    }

    if (dipo && grad) { // dipole gradient
      const VecType J0 = Rinv3 * Rinv2 * (RealType)(-3);
      for (sctl::Integer i = 0; i < COORD_DIM; i++){
        const VecType J0_dXi = J0*dX[i];
        for (sctl::Integer j = 0; j < COORD_DIM; j++){
          M[chrg+i][poten+j][0] = (i==j ? J0_dXi*dX[j]+Rinv3 : J0_dXi*dX[j]);
        }
      }
    }
  }
};

template <class Real, sctl::Integer VecLen, sctl::Integer chrg, sctl::Integer dipo, sctl::Integer poten, sctl::Integer grad> static void EvalLaplace(sctl::Vector<sctl::Vector<Real>>& v_trg, const sctl::Vector<Real>& r_trg, const sctl::Vector<Real>& r_src, const sctl::Vector<sctl::Vector<Real>>& v_src, const sctl::Integer nd, const Real thresh, const sctl::Integer digits) {
  GenericKernel<Laplace3D<Real,VecLen, chrg,dipo,poten,grad>> ker;
  ker.thresh2 = thresh*thresh;

  if (digits < 0) ker.template Eval<true, -1>(v_trg, r_trg, r_src, v_src, nd);
  else if (digits <= 3) ker.template Eval<true, 3>(v_trg, r_trg, r_src, v_src, nd);
  else if (digits <= 6) ker.template Eval<true, 6>(v_trg, r_trg, r_src, v_src, nd);
  else if (digits <= 9) ker.template Eval<true, 9>(v_trg, r_trg, r_src, v_src, nd);
  else if (digits <=12) ker.template Eval<true,12>(v_trg, r_trg, r_src, v_src, nd);
  else if (digits <=15) ker.template Eval<true,15>(v_trg, r_trg, r_src, v_src, nd);
  else ker.template Eval<true,-1>(v_trg, r_trg, r_src, v_src, nd);
}





/* local kernel for the 1/r kernel */
#if !defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim>void l3d_local_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc,const Real* cen, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot, const Real* thresh) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;


  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
	if (COORD_DIM>2) {
	  std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
	}
    Vt = 0;
  }
  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec thresh2 = thresh[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  // load charge
  sctl::Matrix<Real> Vs_(Nsrc, nd_,sctl::Ptr2Itr<Real>((Real*)charge, nd_*Nsrc),false);
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
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
        R2 += dX[k]*dX[k];
      }
	  

      Vec Rinv = sctl::approx_rsqrt<digits>(R2, (R2 > thresh2) & (R2 < d2max_vec));
      // evaluate the PSWF kernel
      Vec xtmp = FMA(R2,Rinv,cen_vec)*rsc_vec;
      Vec ptmp;
      if (digits <= 3) {
		constexpr Real coefs[7] = {1.627823522210361e-01, -4.553645597616490e-01, 4.171687104204163e-01, -7.073638602709915e-02, -8.957845614474928e-02, 2.617986644718201e-02, 9.633427876507601e-03};
	ptmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6]);
      } else if (digits <= 6) {
		constexpr Real coefs[13] = {5.482525801351582e-02, -2.616592110444692e-01, 4.862652666337138e-01, -3.894296348642919e-01, 1.638587821812791e-02, 1.870328434198821e-01, -8.714171086568978e-02, -3.927020727017803e-02, 3.728187607052319e-02, 3.153734425831139e-03, -8.651313377285847e-03, 1.725110090795567e-04, 1.034762385284044e-03};
	ptmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12]);		
      }  else if (digits <= 9) {
		constexpr Real coefs[19] = {1.835718730962269e-02, -1.258015846164503e-01, 3.609487248584408e-01, -5.314579651112283e-01, 3.447559412892380e-01, 9.664692318551721e-02, -3.124274531849053e-01, 1.322460720579388e-01, 9.773007866584822e-02, -1.021958831082768e-01, -3.812847450976566e-03, 3.858117355875043e-02, -8.728545924521301e-03, -9.401196355382909e-03, 4.024549377076924e-03, 1.512806105865091e-03, -9.576734877247042e-04, -1.303457547418901e-04, 1.100385844683190e-04};
	ptmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18]);
      }  else if (digits <= 12) {
		constexpr Real coefs[25] = {6.262472576363448e-03, -5.605742936112479e-02, 2.185890864792949e-01, -4.717350304955679e-01, 5.669680214206270e-01, -2.511606878849214e-01, -2.744523658778361e-01, 4.582527599363415e-01, -1.397724810121539e-01, -2.131762135835757e-01, 1.995489373508990e-01, 1.793390341864239e-02, -1.035055132403432e-01, 3.035606831075176e-02, 3.153931762550532e-02, -2.033178627450288e-02, -5.406682731236552e-03, 7.543645573618463e-03, 1.437788047407851e-05, -1.928370882351732e-03, 2.891658777328665e-04, 3.332996162099811e-04, -8.397699195938912e-05, -3.015837377517983e-05, 9.640642701924662e-06};
	ptmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18],coefs[19],coefs[20],coefs[21],coefs[22],coefs[23],coefs[24]);
      }
	  
      ptmp = ptmp*Rinv;

      for (long i = 0; i < nd_; i++) {
		Vtrg[i] += Vec::Load1(&Vs_[s][i])*ptmp;
      }
    }
    for (long i = 0; i < nd_; i++) {
	  Vtrg[i].StoreAligned(&Vt[i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j = 0; j < nd_; j++) {
      pot[i*nd_+j] += Vt[j][i];
    }
  }
}
#endif




#if defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim>void l3d_local_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc,const Real* cen, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot, const Real* thresh) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;


  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
	if (COORD_DIM>2) {
	  std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
	}
    Vt = 0;
  }
  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec thresh2 = thresh[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  // load charge
  sctl::Matrix<Real> Vs_(Nsrc, nd_,sctl::Ptr2Itr<Real>((Real*)charge, nd_*Nsrc),false);
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);

  //#pragma omp parallel for schedule(static)
  // TODO: reinit to sctl::Vector is over limit
  // work array memory stores compressed R2
  Real R2_work[NSRC_LIMIT*8];
  //sctl::StaticArray<Real, NSRC_LIMIT*8> R2_work_buff;
  // why converting to sctl::Vector is a bit slow...
  //sctl::Vector<Real> R2_work(NSRC_LIMIT*8, R2_work_buff, false);
  // work array memory stores compressed polynomial evals
  Real Pval_work[NSRC_LIMIT*8];
  // mask array
  typename Vec::MaskType Mask_work[NSRC_LIMIT];
  // sparse starting index of stored Pval
  int start_ind[NSRC_LIMIT];
  // sparse source point index
  int source_ind[NSRC_LIMIT];
  // zero vec
  static Vec zero_tmp = Vec::Zero();
  
  #pragma omp parallel for schedule(static)
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

	int valid_r = 0;
    int source_cnt = 0;
    // store compressed R2 to explore 4/3 pi over 27 sparsity
	for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = zero_tmp;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }
	  // store sparse data
      Mask_work[source_cnt] = R2 < d2max_vec;
      int valid_cnt =  mask_popcnt(Mask_work[source_cnt]);
      mask_compress_store(Mask_work[source_cnt], R2, &R2_work[0]+valid_r);
      start_ind[source_cnt] = valid_r;
      source_ind[source_cnt] = s;
      source_cnt += (valid_cnt>0);
      valid_r += valid_cnt;
    }

    // evaluate polynomial on compressed R2 and store in Pval
    for (sctl::Long s = 0; s < valid_r; s += VecLen) {
      Vec R2 = Vec::LoadAligned(&R2_work[0]+s);
      // evaluate the PSWF local kernel
      Vec rtmp;
      Vec Rinv = sctl::approx_rsqrt<digits>(R2, R2 > thresh2);
      // evaluate the PSWF kernel
      Vec xtmp = FMA(R2,Rinv,cen_vec)*rsc_vec;
      if (digits <= 3) {
		constexpr Real coefs[7] = {1.627823522210361e-01, -4.553645597616490e-01, 4.171687104204163e-01, -7.073638602709915e-02, -8.957845614474928e-02, 2.617986644718201e-02, 9.633427876507601e-03};
		rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6]);
      } else if (digits <= 6) {
		constexpr Real coefs[13] = {5.482525801351582e-02, -2.616592110444692e-01, 4.862652666337138e-01, -3.894296348642919e-01, 1.638587821812791e-02, 1.870328434198821e-01, -8.714171086568978e-02, -3.927020727017803e-02, 3.728187607052319e-02, 3.153734425831139e-03, -8.651313377285847e-03, 1.725110090795567e-04, 1.034762385284044e-03};
		rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12]);		
      }  else if (digits <= 9) {
		constexpr Real coefs[19] = {1.835718730962269e-02, -1.258015846164503e-01, 3.609487248584408e-01, -5.314579651112283e-01, 3.447559412892380e-01, 9.664692318551721e-02, -3.124274531849053e-01, 1.322460720579388e-01, 9.773007866584822e-02, -1.021958831082768e-01, -3.812847450976566e-03, 3.858117355875043e-02, -8.728545924521301e-03, -9.401196355382909e-03, 4.024549377076924e-03, 1.512806105865091e-03, -9.576734877247042e-04, -1.303457547418901e-04, 1.100385844683190e-04};
		rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18]);
      }  else if (digits <= 12) {
		constexpr Real coefs[25] = {6.262472576363448e-03, -5.605742936112479e-02, 2.185890864792949e-01, -4.717350304955679e-01, 5.669680214206270e-01, -2.511606878849214e-01, -2.744523658778361e-01, 4.582527599363415e-01, -1.397724810121539e-01, -2.131762135835757e-01, 1.995489373508990e-01, 1.793390341864239e-02, -1.035055132403432e-01, 3.035606831075176e-02, 3.153931762550532e-02, -2.033178627450288e-02, -5.406682731236552e-03, 7.543645573618463e-03, 1.437788047407851e-05, -1.928370882351732e-03, 2.891658777328665e-04, 3.332996162099811e-04, -8.397699195938912e-05, -3.015837377517983e-05, 9.640642701924662e-06};
		rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18],coefs[19],coefs[20],coefs[21],coefs[22],coefs[23],coefs[24]);
      }
	  
      Vec ptmp = rtmp*Rinv;
      ptmp.StoreAligned(Pval_work+s);
    }

    // expand compressed Pval then multiply by charge to accumulate
    for (sctl::Long s_ind = 0; s_ind < source_cnt; s_ind++) {
      int s = source_ind[s_ind];
      int start = start_ind[s_ind];
      Vec ptmp = mask_expand_load(Mask_work[s_ind], zero_tmp, Pval_work+start);
      for (long i = 0; i < nd_; i++) {
		Vtrg[i] += Vec::Load1(&Vs_[s][i])*ptmp;
      }
    }
    // store potential
    for (long i = 0; i < nd_; i++) {
	  Vtrg[i].StoreAligned(&Vt[i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j = 0; j < nd_; j++) {
      pot[i*nd_+j] += Vt[j][i];
    }
  }
}
#endif



template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void l3d_local_kernel_directcp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const Real* rsc,const Real* cen, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot, const Real* thresh) {
  if (ndim[0] == 3) {
    if (digits[0] <= 3) l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 6) l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 9) l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 12) l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);  
    else l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
  }
  if (ndim[0] == 2) {
    if (digits[0] <= 3) l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 6) l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 9) l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 12) l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);  
    else l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
  }
}




#if !defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void l3d_near_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    if (COORD_DIM>2) {
      std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
    }
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];

  // load charge
  sctl::Matrix<Real> Vs_(Nsrc, nd_,sctl::Ptr2Itr<Real>((Real*)charge, nd_*Nsrc),false);
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
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
        R2 += dX[k]*dX[k];
      }

      // evaluate the PSWF local kernel
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      Vec rtmp;
      if (digits<=3) {
	constexpr Real coefs[9] = {2.146941209293174e-02, -6.263795886222773e-02, 8.000991991853681e-02, -6.788435300350132e-02, 4.424154750495320e-02, -1.495578200922307e-02, 6.246832281171233e-03, -1.726850983519235e-02, 1.079907579714659e-02};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8]);
      } else if (digits <= 6) {
	constexpr Real coefs[16] = {1.340418974956820e-03, -6.599369969180820e-03, 1.490307518448090e-02, -2.093949273676980e-02, 2.107881727833481e-02, -1.675447756809429e-02, 1.153573427436465e-02, -7.167326866171437e-03, 3.494340256858195e-03, -1.811569682012156e-03, 2.526431600085065e-03, -1.709903001756345e-03, -7.760281837689070e-04, 6.225228333113239e-04, 7.224764067524717e-04, -4.656557370053271e-04};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15]);
      } else if (digits <= 9) {
	constexpr Real coefs[23] = {7.439068818728897e-05, -5.262656940614313e-04, 1.747940646794798e-03, -3.655452046318296e-03, 5.458534689045880e-03, -6.274347452854087e-03, 5.861446056118720e-03, -4.651936315076755e-03, 3.265895020991306e-03, -2.116549594822989e-03, 1.295019564795384e-03, -7.104931793603499e-04, 3.963779800852606e-04, -3.715088788968398e-04, 2.481566542480207e-04, 9.891447519677762e-05, -9.709134019402767e-05, -2.149126143671141e-04, 1.548732897108437e-04, 1.000150136693176e-04, -7.093300485336296e-05, -3.465097946298012e-05, 2.257702407959528e-05};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18],coefs[19],coefs[20],coefs[21],coefs[22]);
      } else if (digits <= 12) {
	constexpr Real coefs[30] = {4.096568781361293e-06, -3.795272362992590e-05, 1.673411559121917e-04, -4.694154825486451e-04, 9.456836355880467e-04, -1.465958880864274e-03, 1.831740439198150e-03, -1.911093729553248e-03, 1.714472186521308e-03, -1.357720764610499e-03, 9.725782737362134e-04, -6.452430396558342e-04, 4.062759476187977e-04, -2.464801686610534e-04, 1.415660349296740e-04, -8.151213564287058e-05, 6.088776174428736e-05, -3.897054187608593e-05, -5.199322510991006e-06, 8.621087085929069e-06, 3.728698952551457e-05, -2.821687742332601e-05, -2.924205296217508e-05, 2.220073809144907e-05, 2.144585933194776e-05, -1.522711053780768e-05, -8.222228416343569e-06, 5.646501904266188e-06, 1.761327439453453e-06, -1.149448507931083e-06};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18],coefs[19],coefs[20],coefs[21],coefs[22],coefs[23],coefs[24],coefs[25],coefs[26],coefs[27],coefs[28],coefs[29]);
      }
      Vec ptmp = select(R2 < d2max_vec, rtmp*bsizeinv_vec, Vec::Zero());
      for (long i = 0; i < nd_; i++) {
	Vtrg[i] += Vec::Load1(&Vs_[s][i])*ptmp;
      }
    }
    for (long i = 0; i < nd_; i++) {
      Vtrg[i].StoreAligned(&Vt[i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j = 0; j < nd_; j++) {
      pot[i*nd_+j] += Vt[j][i];
    }
  }
}
#endif




#if defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void l3d_near_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    if (COORD_DIM>2) {
      std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
    }
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];

  // load charge
  sctl::Matrix<Real> Vs_(Nsrc, nd_,sctl::Ptr2Itr<Real>((Real*)charge, nd_*Nsrc),false);
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);

  //#pragma omp parallel for schedule(static)
  // TODO: reinit to sctl::Vector is over limit
  //  #define NSRC_LIMIT 400
  // work array memory stores compressed R2
  Real R2_work[NSRC_LIMIT*8];
  //sctl::StaticArray<Real, NSRC_LIMIT*8> R2_work_buff;
  // why converting to sctl::Vector is a bit slow...
  //sctl::Vector<Real> R2_work(NSRC_LIMIT*8, R2_work_buff, false);
  // work array memory stores compressed polynomial evals
  Real Pval_work[NSRC_LIMIT*8];
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

    //int total_r = Nsrc*VecLen;
    int valid_r = 0;
    int source_cnt = 0;
    // store compressed R2 to explore 4/3 pi over 27 sparsity
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = zero_tmp;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }

      // store sparse data
      Mask_work[source_cnt] = R2 < d2max_vec;
      int valid_cnt =  mask_popcnt(Mask_work[source_cnt]);
      mask_compress_store(Mask_work[source_cnt], R2, &R2_work[0]+valid_r);
      start_ind[source_cnt] = valid_r;
      source_ind[source_cnt] = s;
      source_cnt += (valid_cnt>0);
      valid_r += valid_cnt;
    }

    // evaluate polynomial on compressed R2 and store in Pval
    for (sctl::Long s = 0; s < valid_r; s += VecLen) {
      Vec R2 = Vec::LoadAligned(&R2_work[0]+s);
      // evaluate the PSWF local kernel
      Vec rtmp;
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      if (digits<=3) {
	constexpr Real coefs[9] = {2.146941209293174e-02, -6.263795886222773e-02, 8.000991991853681e-02, -6.788435300350132e-02, 4.424154750495320e-02, -1.495578200922307e-02, 6.246832281171233e-03, -1.726850983519235e-02, 1.079907579714659e-02};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8]);
      } else if (digits <= 6) {
	constexpr Real coefs[16] = {1.340418974956820e-03, -6.599369969180820e-03, 1.490307518448090e-02, -2.093949273676980e-02, 2.107881727833481e-02, -1.675447756809429e-02, 1.153573427436465e-02, -7.167326866171437e-03, 3.494340256858195e-03, -1.811569682012156e-03, 2.526431600085065e-03, -1.709903001756345e-03, -7.760281837689070e-04, 6.225228333113239e-04, 7.224764067524717e-04, -4.656557370053271e-04};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15]);
      } else if (digits <= 9) {
	constexpr Real coefs[23] = {7.439068818728897e-05, -5.262656940614313e-04, 1.747940646794798e-03, -3.655452046318296e-03, 5.458534689045880e-03, -6.274347452854087e-03, 5.861446056118720e-03, -4.651936315076755e-03, 3.265895020991306e-03, -2.116549594822989e-03, 1.295019564795384e-03, -7.104931793603499e-04, 3.963779800852606e-04, -3.715088788968398e-04, 2.481566542480207e-04, 9.891447519677762e-05, -9.709134019402767e-05, -2.149126143671141e-04, 1.548732897108437e-04, 1.000150136693176e-04, -7.093300485336296e-05, -3.465097946298012e-05, 2.257702407959528e-05};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18],coefs[19],coefs[20],coefs[21],coefs[22]);
      } else if (digits <= 12) {
	constexpr Real coefs[30] = {4.096568781361293e-06, -3.795272362992590e-05, 1.673411559121917e-04, -4.694154825486451e-04, 9.456836355880467e-04, -1.465958880864274e-03, 1.831740439198150e-03, -1.911093729553248e-03, 1.714472186521308e-03, -1.357720764610499e-03, 9.725782737362134e-04, -6.452430396558342e-04, 4.062759476187977e-04, -2.464801686610534e-04, 1.415660349296740e-04, -8.151213564287058e-05, 6.088776174428736e-05, -3.897054187608593e-05, -5.199322510991006e-06, 8.621087085929069e-06, 3.728698952551457e-05, -2.821687742332601e-05, -2.924205296217508e-05, 2.220073809144907e-05, 2.144585933194776e-05, -1.522711053780768e-05, -8.222228416343569e-06, 5.646501904266188e-06, 1.761327439453453e-06, -1.149448507931083e-06};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18],coefs[19],coefs[20],coefs[21],coefs[22],coefs[23],coefs[24],coefs[25],coefs[26],coefs[27],coefs[28],coefs[29]);
      }
      Vec ptmp = rtmp*bsizeinv_vec;
      ptmp.StoreAligned(Pval_work+s);
    }

    // expand compressed Pval then multiply by charge to accumulate
    for (sctl::Long s_ind = 0; s_ind < source_cnt; s_ind++) {
      int s = source_ind[s_ind];
      int start = start_ind[s_ind];
      Vec ptmp = mask_expand_load(Mask_work[s_ind], zero_tmp, Pval_work+start);
      for (long i = 0; i < nd_; i++) {
	Vtrg[i] += Vec::Load1(&Vs_[s][i])*ptmp;
      }
    }

    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i].StoreAligned(&Vt[i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j = 0; j < nd_; j++) {
      pot[i*nd_+j] += Vt[j][i];
    }
  }
}
#endif




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void l3d_near_kernel_directcp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  if (ndim[0] == 3) {
    if (digits[0] <= 3) l3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 6) l3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 9) l3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 12) l3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else l3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  }
  if (ndim[0] == 2) {
    if (digits[0] <= 3) l3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 6) l3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 9) l3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 12) l3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else l3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  }
}





template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim>void sl3d_local_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc,const Real* cen, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot, const Real* thresh) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;


  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    if (COORD_DIM>2) {
      std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
    }
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec thresh2 = thresh[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  // load charge
  sctl::Matrix<Real> Vs_(Nsrc, nd_,sctl::Ptr2Itr<Real>((Real*)charge, nd_*Nsrc),false);
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
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
        R2 += dX[k]*dX[k];
      }

      Vec Rinv = sctl::approx_rsqrt<digits>(R2, (R2 > thresh2) & (R2 < d2max_vec));
      Vec R2inv=Rinv*Rinv;
      // evaluate the PSWF kernel
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      Vec ptmp;
      if (digits <= 3) {
	constexpr Real coefs[6] = {1.072550277328770e-01, -2.940116755817858e-01, 3.195680735052503e-01, -1.885147776001495e-01, 7.308229981701020e-02, -1.746740887691195e-02};
	ptmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5]);
      } else if (digits <= 6) {
	constexpr Real coefs[11] = {1.614709231716746e-02, -8.104482562490173e-02, 1.821202399011004e-01, -2.445665225927775e-01, 2.214153542697690e-01, -1.447686339587813e-01, 7.139964303548979e-02, -2.725881175495876e-02, 8.433287237298649e-03, -2.361123374269844e-03, 4.843770794874525e-04};
	ptmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10]);		
      }  else if (digits <= 9) {
	constexpr Real coefs[17] = {2.086402113115865e-03, -1.562089192993565e-02, 5.445041674459949e-02, -1.178541200828409e-01, 1.783129998458763e-01, -2.013700177956373e-01, 1.770491043290802e-01, -1.248556216953823e-01, 7.222027766315242e-02, -3.487303987632228e-02, 1.426369137364099e-02, -5.005766772662122e-03, 1.519637697654493e-03, -3.983110653078808e-04, 9.342903238021878e-05, -2.223077176699562e-05, 4.041199749483288e-06};
	ptmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16]);
      }  else if (digits <= 12) {
	constexpr Real coefs[22] = {2.585256922145451e-04, -2.586987362995730e-03, 1.228229637064665e-02, -3.689309545219710e-02, 7.889646803121930e-02, -1.281810865332097e-01, 1.648961579104568e-01, -1.728791019337961e-01, 1.509077982778982e-01, -1.115149888411611e-01, 7.069833292226477e-02, -3.888103784086078e-02, 1.872315725194051e-02, -7.957995064673665e-03, 3.005989787130355e-03, -1.015826376419250e-03, 3.094089416842241e-04, -8.493261780963615e-05, 2.066398691692728e-05, -4.722966945758245e-06, 1.200827608904831e-06, -2.250099297995689e-07};
	ptmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18],coefs[19],coefs[20],coefs[21]);
      }
	  
      ptmp = ptmp*R2inv;

      for (long i = 0; i < nd_; i++) {
	Vtrg[i] += Vec::Load1(&Vs_[s][i])*ptmp;
      }
    }
    for (long i = 0; i < nd_; i++) {
      Vtrg[i].StoreAligned(&Vt[i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j = 0; j < nd_; j++) {
      pot[i*nd_+j] += Vt[j][i];
    }
  }
}




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void sl3d_local_kernel_directcp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const Real* rsc,const Real* cen, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot, const Real* thresh) {
  if (ndim[0] == 3) {
    if (digits[0] <= 3) sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 6) sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 9) sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 12) sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);  
    else sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
  }
  if (ndim[0] == 2) {
    if (digits[0] <= 3) sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 6) sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 9) sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 12) sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);  
    else sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
  }
}




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void sl3d_near_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    if (COORD_DIM>2) {
      std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
    }
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0]*bsizeinv[0];

  // load charge
  sctl::Matrix<Real> Vs_(Nsrc, nd_,sctl::Ptr2Itr<Real>((Real*)charge, nd_*Nsrc),false);
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
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
        R2 += dX[k]*dX[k];
      }
	  
      // evaluate the PSWF local kernel
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      Vec rtmp;
      if (digits<=3) {
	constexpr Real coefs[9] = {8.173887886459996e-02, -2.451274687320377e-01, 3.277193048641895e-01, -3.062447627681784e-01, 2.189036013807557e-01, -5.720936576940738e-02, 1.938154100021000e-02, -1.170456117266738e-01, 7.803629856633656e-02};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8]);
      } else if (digits <= 6) {
	constexpr Real coefs[16] = {6.698297444417106e-03, -3.336488967202680e-02, 7.653327462540263e-02, -1.097982329138708e-01, 1.135180590069429e-01, -9.346163757569020e-02, 6.802392826055811e-02, -4.473461412598995e-02, 2.053582116124963e-02, -1.050937392785068e-02, 2.140236275524626e-02, -1.541390853346769e-02, -8.563795340776453e-03, 6.850981352078830e-03, 6.850974278963085e-03, -4.567316486905559e-03};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15]);
      } else if (digits <= 9) {
	constexpr Real coefs[23] = {4.469280876409076e-04, -3.180300344022710e-03, 1.064042494973956e-02, -2.245742997736247e-02, 3.392748565479964e-02, -3.958160892988580e-02, 3.767970892048522e-02, -3.060599931494992e-02, 2.210711084762564e-02, -1.488479343040319e-02, 9.479554346392999e-03, -5.164827611128249e-03, 2.921689534493682e-03, -3.474436592339673e-03, 2.476406820890942e-03, 1.453581302051929e-03, -1.340755280231744e-03, -2.481010442893649e-03, 1.837785547647935e-03, 1.202914582812853e-03, -8.687714213235775e-04, -4.009715343213927e-04, 2.673143161473490e-04};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18],coefs[19],coefs[20],coefs[21],coefs[22]);
      } else if (digits <= 12) {
	constexpr Real coefs[30] = {2.819652392720140e-05, -2.621417646486990e-04, 1.160666482044811e-03, -3.272221873150037e-03, 6.632530755659602e-03, -1.035849178023596e-02, 1.306264335216134e-02, -1.378426535688983e-02, 1.254085460183132e-02, -1.010339264570685e-02, 7.387432833369574e-03, -5.020801942566828e-03, 3.255681802450250e-03, -2.037634681848246e-03, 1.183450016729645e-03, -6.948989883377235e-04, 6.016054650414167e-04, -4.097810677297087e-04, -1.395540857050542e-04, 1.619848136232838e-04, 4.871200845424362e-04, -3.817087672355335e-04, -4.033367785609698e-04, 3.126972055405020e-04, 2.889390772224942e-04, -2.101222128203517e-04, -1.114695751311956e-04, 7.819135134923272e-05, 2.346749533899128e-05, -1.564034027978778e-05};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18],coefs[19],coefs[20],coefs[21],coefs[22],coefs[23],coefs[24],coefs[25],coefs[26],coefs[27],coefs[28],coefs[29]);
      }
      Vec ptmp = select(R2 < d2max_vec, rtmp*bsizeinv_vec, Vec::Zero());
      for (long i = 0; i < nd_; i++) {
	Vtrg[i] += Vec::Load1(&Vs_[s][i])*ptmp;
      }
    }
    for (long i = 0; i < nd_; i++) {
      Vtrg[i].StoreAligned(&Vt[i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j = 0; j < nd_; j++) {
      pot[i*nd_+j] += Vt[j][i];
    }
  }
}



template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void sl3d_near_kernel_directcp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  if (ndim[0] == 3) {
    if (digits[0] <= 3) sl3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 6) sl3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 9) sl3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 12) sl3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else sl3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  }
  if (ndim[0] == 2) {
    if (digits[0] <= 3) sl3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 6) sl3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 9) sl3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 12) sl3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else sl3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  }
}  





/* local kernel for the log(r) kernel */
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim>void log_local_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc,const Real* cen, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot, const Real* thresh) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;


  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    if (COORD_DIM>2) {
      std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
    }
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec thresh2 = thresh[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv2_vec = rsc[0]*0.5e0;
  // load charge
  sctl::Matrix<Real> Vs_(Nsrc, nd_,sctl::Ptr2Itr<Real>((Real*)charge, nd_*Nsrc),false);
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
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
        R2 += dX[k]*dX[k];
      }
	  
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      Vec R2sc = R2*bsizeinv2_vec;
	  
      Vec ptmp;
      if (digits <= 3) {
	constexpr Real coefs[5] = {3.293312412035785e-01, -4.329140084314137e-01, 1.366683635926240e-01, -4.309918126794055e-02, 1.041106682948322e-02};
	ptmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4]);
      } else if (digits <= 6) {
	constexpr Real coefs[10] = {3.449851438836016e-01, -4.902921365061905e-01, 2.220880572548949e-01, -1.153716526684871e-01, 5.535102319921498e-02, -2.281631998557134e-02, 7.843017349311455e-03, -2.269922867123751e-03, 6.058276012390756e-04, -1.231943198424746e-04};
	ptmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9]);		
      }  else if (digits <= 9) {
	constexpr Real coefs[15] = {3.464285661408188e-01, -4.987507216024493e-01, 2.448626056577886e-01, -1.531215433543293e-01, 9.898265049486202e-02, -6.064807566030946e-02, 3.368003719471507e-02, -1.657913779091916e-02, 7.167896958658711e-03, -2.718119907588121e-03, 9.031254984609993e-04, -2.605021223678894e-04, 6.764515456758602e-05, -1.806660493741674e-05, 3.640268744220521e-06};
	ptmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14]);
      }  else if (digits <= 12) {
	constexpr Real coefs[19] = {3.465597422118963e-01, -4.998454957461734e-01, 2.491697186708244e-01, -1.637913095741979e-01, 1.177474435136638e-01, -8.570242499810476e-02, 6.020769882062124e-02, -3.955364604811165e-02, 2.382498816767467e-02, -1.301290744915523e-02, 6.410015506457686e-03, -2.841967252293358e-03, 1.135602935662887e-03, -4.109780616023590e-04, 1.339086505665511e-04, -3.822398669202901e-05, 1.037153217818392e-05, -3.251884408687046e-06, 7.149918161513096e-07};
	ptmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18]);
      }
	  
      ptmp = select((R2 > thresh2) & (R2 < d2max_vec), 0.5e0*sctl::veclog(R2sc)+ptmp, Vec::Zero());

      for (long i = 0; i < nd_; i++) {
	Vtrg[i] += Vec::Load1(&Vs_[s][i])*ptmp;
      }
    }
    for (long i = 0; i < nd_; i++) {
      Vtrg[i].StoreAligned(&Vt[i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j = 0; j < nd_; j++) {
      pot[i*nd_+j] += Vt[j][i];
    }
  }
}




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void log_local_kernel_directcp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const Real* rsc,const Real* cen, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot, const Real* thresh) {
  if (ndim[0] == 3) {
    if (digits[0] <= 3) log_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 6) log_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 9) log_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 12) log_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);  
    else log_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
  }
  if (ndim[0] == 2) {
    if (digits[0] <= 3) log_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 6) log_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 9) log_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
    else if (digits[0] <= 12) log_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);  
    else log_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,2>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
  }
  
}




/* near kernel for the log(r) kernel */
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void log_near_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    if (COORD_DIM>2) {
      std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
    }
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];

  // load charge
  sctl::Matrix<Real> Vs_(Nsrc, nd_,sctl::Ptr2Itr<Real>((Real*)charge, nd_*Nsrc),false);
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
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
        R2 += dX[k]*dX[k];
      }
	  
      // evaluate the PSWF local kernel
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      Vec rtmp;
      if (digits<=3) {
	constexpr Real coefs[9] = {-6.111367193489408e-03, 2.084211725904822e-02, -2.814992265313088e-02, 2.317145716583081e-02, -1.430284775818347e-02, 5.484439507096733e-03, -2.395905697203788e-03, 4.176782420780789e-03, -2.438403456299273e-03};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8]);
      } else if (digits <= 6) {
	constexpr Real coefs[16] = {-3.161280140115538e-04, 1.632058640226754e-03, -3.844349799468478e-03, 5.577306481275532e-03, -5.721277634228778e-03, 4.559290205715725e-03, -3.071201618386827e-03, 1.839973139774339e-03, -9.187363825782615e-04, 4.702946355513738e-04, -4.916708204871587e-04, 3.113194482571232e-04, 1.070558573237174e-04, -8.783390147456941e-05, -1.223387576644629e-04, 7.612193606978956e-05};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15]);
      } else if (digits <= 9) {
	constexpr Real coefs[23] = {-1.480526527085856e-05, 1.082575403525449e-04, -3.708464597095037e-04, 7.973459217364091e-04, -1.218840652219571e-03, 1.426110326278787e-03, -1.346497983981166e-03, 1.070950178086426e-03, -7.460456670321969e-04, 4.735956561368585e-04, -2.815377216492961e-04, 1.536203375356456e-04, -8.389205404246250e-05, 6.548395440955711e-05, -4.099342560057078e-05, -8.984326635160672e-06, 1.037717522160677e-05, 2.985184255602257e-05, -2.088390732763368e-05, -1.318992946343009e-05, 9.188925611986615e-06, 4.773485622111551e-06, -3.037662502869409e-06};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18],coefs[19],coefs[20],coefs[21],coefs[22]);
      } else if (digits <= 12) {
	constexpr Real coefs[30] = {-7.165864640578160e-07, 6.809707052032247e-06, -3.076702821749118e-05, 8.831526892853401e-05, -1.817299603051672e-04, 2.870664593138785e-04, -3.644248407478640e-04, 3.848464506263178e-04, -3.478587716561872e-04, 2.760386920690674e-04, -1.969154474120812e-04, 1.292205534423863e-04, -7.985953340655639e-05, 4.731188923708227e-05, -2.676160929679155e-05, 1.503835307815370e-05, -1.006229608427533e-05, 6.142572231615496e-06, -2.491704739248160e-07, -5.618278360449910e-07, -4.572199184794956e-06, 3.401527197667564e-06, 3.337274919079647e-06, -2.528651319888553e-06, -2.523197146331313e-06, 1.767893834880852e-06, 9.598676342648105e-07, -6.502428959720419e-07, -2.096821845043451e-07, 1.345351847703569e-07};
	rtmp = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18],coefs[19],coefs[20],coefs[21],coefs[22],coefs[23],coefs[24],coefs[25],coefs[26],coefs[27],coefs[28],coefs[29]);
      }
      Vec ptmp = select(R2 < d2max_vec, rtmp, Vec::Zero());
      for (long i = 0; i < nd_; i++) {
	Vtrg[i] += Vec::Load1(&Vs_[s][i])*ptmp;
      }
    }
    for (long i = 0; i < nd_; i++) {
      Vtrg[i].StoreAligned(&Vt[i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j = 0; j < nd_; j++) {
      pot[i*nd_+j] += Vt[j][i];
    }
  }
}




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void log_near_kernel_directcp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  if (ndim[0] == 3) {
    if (digits[0] <= 3) log_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 6) log_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 9) log_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 12) log_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else log_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,3>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  }
  if (ndim[0] == 2) {
    if (digits[0] <= 3) log_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 6) log_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 9) log_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else if (digits[0] <= 12) log_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
    else log_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,2>(nd,rsc,cen,bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  }
}





/* the Yukawa kernel in 3D exp(-kr)/r, charge to potential */
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1>void y3ddirectcp_vec_cpp_helper(const int32_t* nd, const Real* rlambda, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot, const Real* thresh) {
  static constexpr sctl::Integer COORD_DIM = 3; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;


  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec thresh2 = thresh[0];
  Vec d2max_vec = d2max[0];
  Vec mrlambda_vec = -rlambda[0];
  // load charge
  sctl::Matrix<Real> Vs_(Nsrc, nd_,sctl::Ptr2Itr<Real>((Real*)charge, nd_*Nsrc),false);
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
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
        R2 += dX[k]*dX[k];
      }

      Vec Rinv = sctl::approx_rsqrt<digits>(R2, (R2 > thresh2) & (R2 < d2max_vec));

      Vec xtmp = mrlambda_vec*R2*Rinv;
      Vec ptmp = sctl::approx_exp<digits>(xtmp);
	  
      ptmp = select((R2 > thresh2) & (R2 < d2max_vec), ptmp*Rinv, Vec::Zero());

      for (long i = 0; i < nd_; i++) {
	Vtrg[i] += Vec::Load1(&Vs_[s][i])*ptmp;
      }
    }
    for (long i = 0; i < nd_; i++) {
      Vtrg[i].StoreAligned(&Vt[i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j = 0; j < nd_; j++) {
      pot[i*nd_+j] += Vt[j][i];
    }
  }
}




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void y3ddirectcp_vec_cpp(const int32_t* nd, const int32_t* digits,const Real* rlambda, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot, const Real* thresh) {
  if (digits[0] <= 3) y3ddirectcp_vec_cpp_helper<Real,MaxVecLen,3>(nd,rlambda,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
  else if (digits[0] <= 6) y3ddirectcp_vec_cpp_helper<Real,MaxVecLen,6>(nd,rlambda,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
  else if (digits[0] <= 9) y3ddirectcp_vec_cpp_helper<Real,MaxVecLen,9>(nd,rlambda,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
  else if (digits[0] <= 12) y3ddirectcp_vec_cpp_helper<Real,MaxVecLen,12>(nd,rlambda,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);  
  else y3ddirectcp_vec_cpp_helper<Real,MaxVecLen,-1>(nd,rlambda,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
}









/* 3D Stokeslet local kernel charge to potential */ 
#if !defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void st3d_local_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* stoklet, const Real* xtarg, const Real* ytarg, const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_*COORD_DIM, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_*COORD_DIM, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    if (COORD_DIM>2) {
      std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
    }
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];

  // load stokslet density
  sctl::Matrix<Real> Vs_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)stoklet, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vsrc(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vs_[s][0*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vs_[s][1*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+2] = Vec::Load1(&Vs_[s][2*nd_+i]);
    }
  }
  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
  for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
    Vec Xtrg[COORD_DIM];
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
    }
    // load potential
    Vec Vtrg[nd_][COORD_DIM];
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0] = Vec::LoadAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1] = Vec::LoadAligned(&Vt[1*nd_+i][t]);
      Vtrg[i][2] = Vec::LoadAligned(&Vt[2*nd_+i][t]);
    }
	
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = Vec::Zero();
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }
      // evaluate the PSWF local kernel
      Vec fdiag,foffd;
      Vec Rinv = sctl::approx_rsqrt<digits>(R2, (R2 >= d2min_vec) & (R2 <= d2max_vec));
      Vec Rinv3 = Rinv*Rinv*Rinv;
      Vec xtmp = FMA(R2,Rinv,cen_vec)*rsc_vec;
      if (digits<=3) {
		constexpr Real cdiag[9] = {6.303821890482977e-01, -1.364095475417093e-01, -3.781608653672008e-01, 6.362529703372491e-01, -8.617349467259916e-02, -3.092434934988564e-01, 1.118908784199183e-01, 5.966367591689842e-02, -2.772850523913949e-02};
		constexpr Real coffd[9] = {2.760710825268998e-01, 4.908133061808407e-01, -1.130230492845359e-01, -3.984116608334281e-01, 1.683117126918866e-01, 1.970831792617346e-01, -1.057765964495930e-01, -3.973887101646317e-02, 2.420205320151863e-02};
		
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8]);
      } else if (digits <= 6) {
		constexpr Real cdiag[15] = {5.719333574905636e-01, -2.496724574238847e-01, 1.207785090817141e-01, 5.951913468068477e-01, -9.591803087061828e-01, 1.530350585989323e-01, 7.211477019762759e-01, -4.518818911516000e-01, -2.335227591803760e-01, 2.800595595157066e-01, 2.337511861970532e-02, -9.035575151775754e-02, 7.750233147587158e-03, 1.362457951608498e-02, -2.281603211550020e-03};
		constexpr Real coffd[15] = {3.992031999348375e-01, 4.224022514485368e-01, -5.431782158129892e-01, -9.225906219437770e-02, 7.538029820209597e-01, -3.314764311243446e-01, -4.914503503937011e-01, 4.289247197902905e-01, 1.428114781589459e-01, -2.429483974712088e-01, -3.450665352106919e-03, 7.691485471620529e-02, -1.011313983819307e-02, -1.155835253079296e-02, 2.374462321377280e-03};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14]);
      } else if (digits <= 9) {
		constexpr Real cdiag[23] = {5.336982837541028e-01, -1.923809969882605e-01, 3.613844754648635e-01, 7.226134515824434e-03, -1.000100631165442e+00, 1.323435114443506e+00, -2.908230111938132e-02, -1.357152528231017e+00, 9.086256748961907e-01, 5.115877503981630e-01, -8.212053204413041e-01, 3.715620995282090e-02, 3.992853539103454e-01, -1.361592775255129e-01, -1.255362817947336e-01, 7.755520134600147e-02, 2.580817925469603e-02, -2.631415338771603e-02, -2.899096334601836e-03, 5.653827934679612e-03, -1.755741482156739e-05, -6.072822602618544e-04, 3.922121777577559e-05};
		constexpr Real coffd[23] = {4.568730328579675e-01, 2.692062479410559e-01, -6.305907240288979e-01, 5.029030927057596e-01, 4.935844961188628e-01, -1.216959102244955e+00, 3.637509383886227e-01, 1.014173106769019e+00, -9.049312762383398e-01, -3.133549644887098e-01, 7.252755755553277e-01, -9.057874610248320e-02, -3.396279100577519e-01, 1.380407710805294e-01, 1.041246200311221e-01, -7.307099239034189e-02, -2.050338053280143e-02, 2.425714884864761e-02, 1.961816788423221e-03, -5.170155775203966e-03, 1.289467641072406e-04, 5.535934606086812e-04, -4.613586952473270e-05};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22]);
      } else if (digits <= 12) {
		constexpr Real cdiag[29] = {5.130718055892454e-01, -1.072019100994833e-01, 3.470691404445909e-01, -4.596201113166043e-01, -2.114973576121068e-01, 1.566794296454659e+00, -1.866984282972543e+00, -1.448819975760944e-01, 2.420606991476392e+00, -1.811558332742166e+00, -9.386333837861963e-01, 2.027163684363858e+00, -4.120892008857107e-01, -1.113079875287754e+00, 7.125810795318408e-01, 3.237828259750088e-01, -4.595912177831991e-01, -4.503843888727495e-03, 1.926703125566291e-01, -4.596560166419295e-02, -5.833930172843793e-02, 2.597428610397401e-02, 1.298025923006207e-02, -8.425015508692013e-03, -2.032701850112062e-03, 1.685102455667220e-03, 1.957357308128849e-04, -1.635072694625705e-04, -7.877941243350506e-06};
		constexpr Real coffd[29] = {4.842053330529956e-01, 1.360683826354384e-01, -4.831375230780703e-01, 8.270679209529221e-01, -3.857605078008048e-01, -1.054135375677016e+00, 1.876590130459041e+00, -3.981479011415701e-01, -1.913797625904047e+00, 1.842661320373685e+00, 5.452189728678314e-01, -1.804407834350390e+00, 5.271928407782402e-01, 9.345688662727589e-01, -6.930779983432558e-01, -2.482476268410352e-01, 4.245214655196321e-01, -1.455497961532614e-02, -1.741010361682811e-01, 4.776760393292534e-02, 5.192981388404406e-02, -2.521830673185832e-02, -1.136767994648835e-02, 8.013719359951210e-03, 1.738933104206808e-03, -1.589429484738503e-03, -1.611329644219950e-04, 1.536403142381459e-04, 6.014539394527675e-06};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22],cdiag[23],cdiag[24],cdiag[25],cdiag[26],cdiag[27],cdiag[28]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24],coffd[25],coffd[26],coffd[27],coffd[28]);
      }
      Vec half = 0.5e0;
      Vec Fdiag = (half-fdiag)*Rinv;
      Vec Foffd = (half-foffd)*Rinv3;
      for (long i = 0; i < nd_; i++) {
        Vec Dprod = dX[0]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] + dX[2]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+2];
        Vec pl = Foffd*Dprod;
        Vtrg[i][0] += pl*dX[0] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0];
        Vtrg[i][1] += pl*dX[1] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vtrg[i][2] += pl*dX[2] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+2];
      }
    }
    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0].StoreAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1].StoreAligned(&Vt[1*nd_+i][t]);
      Vtrg[i][2].StoreAligned(&Vt[2*nd_+i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_*COORD_DIM; j++) {
      pot[i*nd_*COORD_DIM+j] += Vt[j][i];
    }
  }
}
#endif




#if defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void st3d_local_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* stoklet, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_*COORD_DIM, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_*COORD_DIM, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    if (COORD_DIM>2) {
      std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
    }
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];

  // load stokslet density
  sctl::Matrix<Real> Vs_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)stoklet, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vsrc(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vs_[s][0*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vs_[s][1*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+2] = Vec::Load1(&Vs_[s][2*nd_+i]);
    }
  }
  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);

  //#pragma omp parallel for schedule(static)
  // TODO: reinit to sctl::Vector is over limit
  //  #define NSRC_LIMIT 1000
  // work array memory stores compressed R2
  Real dX_work[COORD_DIM][NSRC_LIMIT*8];
  Real R2_work[NSRC_LIMIT*8];
  //sctl::StaticArray<Real, NSRC_LIMIT*8> R2_work_buff;
  // why converting to sctl::Vector is a bit slow...
  //sctl::Vector<Real> R2_work(NSRC_LIMIT*8, R2_work_buff, false);
  // work array memory stores compressed polynomial evals
  Real Fdiag_work[NSRC_LIMIT*8];
  Real Foffd_work[NSRC_LIMIT*8];
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
      Vtrg[i][0] = Vec::LoadAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1] = Vec::LoadAligned(&Vt[1*nd_+i][t]);
      Vtrg[i][2] = Vec::LoadAligned(&Vt[2*nd_+i][t]);
    }

    //int total_r = Nsrc*VecLen;
    int valid_r = 0;
    int source_cnt = 0;
    // store compressed R2 to explore 4/3 pi over 27 sparsity
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = zero_tmp;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }

      // store sparse data
      Mask_work[source_cnt] = (R2 >= d2min_vec) & (R2 <= d2max_vec);
      int valid_cnt =  mask_popcnt(Mask_work[source_cnt]);
      mask_compress_store(Mask_work[source_cnt], dX[0], &dX_work[0][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], dX[1], &dX_work[1][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], dX[2], &dX_work[2][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], R2, &R2_work[0]+valid_r);
      start_ind[source_cnt] = valid_r;
      source_ind[source_cnt] = s;
      source_cnt += (valid_cnt>0);
      valid_r += valid_cnt;
    }

    // evaluate polynomial on compressed R2 and store in Pval
    for (sctl::Long s = 0; s < valid_r; s += VecLen) {
      Vec R2 = Vec::LoadAligned(&R2_work[0]+s);
      // evaluate the PSWF local kernel
      Vec fdiag,foffd;
      Vec Rinv = sctl::approx_rsqrt<digits>(R2, (R2 >= d2min_vec) & (R2 <= d2max_vec));
      Vec Rinv3 = Rinv*Rinv*Rinv;
      Vec xtmp = FMA(R2,Rinv,cen_vec)*rsc_vec;
      if (digits<=3) {
		constexpr Real cdiag[9] = {6.303821890482977e-01, -1.364095475417093e-01, -3.781608653672008e-01, 6.362529703372491e-01, -8.617349467259916e-02, -3.092434934988564e-01, 1.118908784199183e-01, 5.966367591689842e-02, -2.772850523913949e-02};
		constexpr Real coffd[9] = {2.760710825268998e-01, 4.908133061808407e-01, -1.130230492845359e-01, -3.984116608334281e-01, 1.683117126918866e-01, 1.970831792617346e-01, -1.057765964495930e-01, -3.973887101646317e-02, 2.420205320151863e-02};
		
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8]);
      } else if (digits <= 6) {
		constexpr Real cdiag[15] = {5.719333574905636e-01, -2.496724574238847e-01, 1.207785090817141e-01, 5.951913468068477e-01, -9.591803087061828e-01, 1.530350585989323e-01, 7.211477019762759e-01, -4.518818911516000e-01, -2.335227591803760e-01, 2.800595595157066e-01, 2.337511861970532e-02, -9.035575151775754e-02, 7.750233147587158e-03, 1.362457951608498e-02, -2.281603211550020e-03};
		constexpr Real coffd[15] = {3.992031999348375e-01, 4.224022514485368e-01, -5.431782158129892e-01, -9.225906219437770e-02, 7.538029820209597e-01, -3.314764311243446e-01, -4.914503503937011e-01, 4.289247197902905e-01, 1.428114781589459e-01, -2.429483974712088e-01, -3.450665352106919e-03, 7.691485471620529e-02, -1.011313983819307e-02, -1.155835253079296e-02, 2.374462321377280e-03};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14]);
      } else if (digits <= 9) {
		constexpr Real cdiag[23] = {5.336982837541028e-01, -1.923809969882605e-01, 3.613844754648635e-01, 7.226134515824434e-03, -1.000100631165442e+00, 1.323435114443506e+00, -2.908230111938132e-02, -1.357152528231017e+00, 9.086256748961907e-01, 5.115877503981630e-01, -8.212053204413041e-01, 3.715620995282090e-02, 3.992853539103454e-01, -1.361592775255129e-01, -1.255362817947336e-01, 7.755520134600147e-02, 2.580817925469603e-02, -2.631415338771603e-02, -2.899096334601836e-03, 5.653827934679612e-03, -1.755741482156739e-05, -6.072822602618544e-04, 3.922121777577559e-05};
		constexpr Real coffd[23] = {4.568730328579675e-01, 2.692062479410559e-01, -6.305907240288979e-01, 5.029030927057596e-01, 4.935844961188628e-01, -1.216959102244955e+00, 3.637509383886227e-01, 1.014173106769019e+00, -9.049312762383398e-01, -3.133549644887098e-01, 7.252755755553277e-01, -9.057874610248320e-02, -3.396279100577519e-01, 1.380407710805294e-01, 1.041246200311221e-01, -7.307099239034189e-02, -2.050338053280143e-02, 2.425714884864761e-02, 1.961816788423221e-03, -5.170155775203966e-03, 1.289467641072406e-04, 5.535934606086812e-04, -4.613586952473270e-05};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22]);
      } else if (digits <= 12) {
		constexpr Real cdiag[29] = {5.130718055892454e-01, -1.072019100994833e-01, 3.470691404445909e-01, -4.596201113166043e-01, -2.114973576121068e-01, 1.566794296454659e+00, -1.866984282972543e+00, -1.448819975760944e-01, 2.420606991476392e+00, -1.811558332742166e+00, -9.386333837861963e-01, 2.027163684363858e+00, -4.120892008857107e-01, -1.113079875287754e+00, 7.125810795318408e-01, 3.237828259750088e-01, -4.595912177831991e-01, -4.503843888727495e-03, 1.926703125566291e-01, -4.596560166419295e-02, -5.833930172843793e-02, 2.597428610397401e-02, 1.298025923006207e-02, -8.425015508692013e-03, -2.032701850112062e-03, 1.685102455667220e-03, 1.957357308128849e-04, -1.635072694625705e-04, -7.877941243350506e-06};
		constexpr Real coffd[29] = {4.842053330529956e-01, 1.360683826354384e-01, -4.831375230780703e-01, 8.270679209529221e-01, -3.857605078008048e-01, -1.054135375677016e+00, 1.876590130459041e+00, -3.981479011415701e-01, -1.913797625904047e+00, 1.842661320373685e+00, 5.452189728678314e-01, -1.804407834350390e+00, 5.271928407782402e-01, 9.345688662727589e-01, -6.930779983432558e-01, -2.482476268410352e-01, 4.245214655196321e-01, -1.455497961532614e-02, -1.741010361682811e-01, 4.776760393292534e-02, 5.192981388404406e-02, -2.521830673185832e-02, -1.136767994648835e-02, 8.013719359951210e-03, 1.738933104206808e-03, -1.589429484738503e-03, -1.611329644219950e-04, 1.536403142381459e-04, 6.014539394527675e-06};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22],cdiag[23],cdiag[24],cdiag[25],cdiag[26],cdiag[27],cdiag[28]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24],coffd[25],coffd[26],coffd[27],coffd[28]);
      }
      Vec half = 0.5e0;
      Vec Fdiag = (half-fdiag)*Rinv;
      Vec Foffd = (half-foffd)*Rinv3;
      Fdiag.StoreAligned(Fdiag_work+s);
      Foffd.StoreAligned(Foffd_work+s);
    }

    // expand compressed Fdiag_work and Foffd_work, then multiply by charge to accumulate
    for (sctl::Long s_ind = 0; s_ind < source_cnt; s_ind++) {
      int s = source_ind[s_ind];
      int start = start_ind[s_ind];
      Vec Fdiag = mask_expand_load(Mask_work[s_ind], zero_tmp, Fdiag_work+start);
      Vec Foffd = mask_expand_load(Mask_work[s_ind], zero_tmp, Foffd_work+start);
      Vec dX[3];
      dX[0] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[0]+start);
      dX[1] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[1]+start);
      dX[2] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[2]+start);

      for (long i = 0; i < nd_; i++) {
        Vec Dprod = dX[0]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] + dX[2]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+2];
        Vec pl = Foffd*Dprod;
        Vtrg[i][0] += pl*dX[0] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0];
        Vtrg[i][1] += pl*dX[1] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vtrg[i][2] += pl*dX[2] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+2];
      }
    }

    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0].StoreAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1].StoreAligned(&Vt[1*nd_+i][t]);
      Vtrg[i][2].StoreAligned(&Vt[2*nd_+i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_*COORD_DIM; j++) {
      pot[i*nd_*COORD_DIM+j] += Vt[j][i];
    }
  }
}
#endif




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void st3d_local_kernel_directcp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  if (digits[0] <= 3) st3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 6) st3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 9) st3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 12) st3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else st3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
}


/* 3D Stokeslet near kernel charge to potential */ 
#if !defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void st3d_near_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* stoklet, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_*COORD_DIM, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_*COORD_DIM, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    if (COORD_DIM>2) {
      std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
    }
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];
  Vec bsizeinv3_vec = bsizeinv_vec*bsizeinv_vec*bsizeinv_vec;

  // load stokslet density
  sctl::Matrix<Real> Vs_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)stoklet, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vsrc(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vs_[s][0*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vs_[s][1*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+2] = Vec::Load1(&Vs_[s][2*nd_+i]);
    }
  }
  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
  for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
    Vec Xtrg[COORD_DIM];
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
    }
    // load potential
    Vec Vtrg[nd_][COORD_DIM];
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0] = Vec::LoadAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1] = Vec::LoadAligned(&Vt[1*nd_+i][t]);
      Vtrg[i][2] = Vec::LoadAligned(&Vt[2*nd_+i][t]);
    }
	
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = Vec::Zero();
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }
      // evaluate the PSWF local kernel
      Vec fdiag,foffd;
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      if (digits<=3) {
	constexpr Real cdiag[8] = {-1.200465301327137e-02, 4.400592204092247e-02, -7.288597161347318e-02, 6.078910968844339e-02, -3.108708030232192e-02, 1.003402339570593e-02, 5.363405306153075e-03, -5.553171339387891e-03};
	constexpr Real coffd[11] = {2.250430121704944e-02, -9.787611559588733e-02, 2.070217916734389e-01, -2.524368328199907e-01, 2.259705057475586e-01, -2.420893817698530e-01, 1.866923371452701e-01, 3.395770252841146e-02, -4.311241235638091e-02, -1.308211487415907e-01, 9.152265431281228e-02};
		
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10]);
      } else if (digits <= 6) {
	constexpr Real cdiag[16] = {-6.778413276129699e-04, 4.066402381071667e-03, -1.124864783439655e-02, 1.920489141603694e-02, -2.281183769709533e-02, 1.999165544932320e-02, -1.322112536620177e-02, 6.698004916394447e-03, -2.867817976329824e-03, 8.503272900193856e-04, 6.879027750537969e-04, -6.923660302115852e-04, -4.270882823565761e-04, 3.190690821977045e-04, 3.599672200216620e-04, -2.326863819796524e-04};
	constexpr Real coffd[18] = { 1.184705266078750e-03, -7.928021809420677e-03, 2.512862663913759e-02, -5.082440691714102e-02, 7.474069197148014e-02, -8.588843617624770e-02, 7.996358530102793e-02, -6.499276257962841e-02, 5.539840177127529e-02, -4.014685171673929e-02, 4.879357438144119e-03, 1.419883695499086e-03, 3.525871341680183e-02, -2.706860456161926e-02, -1.886722540638105e-02, 1.460477431730375e-02, 9.998150833375779e-03, -6.859402138560325e-03};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17]);
      } else if (digits <= 9) {
	constexpr Real cdiag[23] = {-3.878153424079992e-05, 3.256189735159013e-04, -1.291899812828380e-03, 3.229916915849101e-03, -5.725266461380479e-03, 7.674125798699759e-03, -8.098740367932972e-03, 6.913185387623120e-03, -4.856062649177749e-03, 2.823670090265469e-03, -1.355279681198315e-03, 5.449639991689835e-04, -1.573565756195452e-04, -6.142949960309105e-05, 8.529221725570496e-05, 6.031578891512924e-05, -5.124161335103055e-05, -1.068570833585420e-04, 7.728626890477778e-05, 5.004058618345081e-05, -3.546124492943787e-05, -1.732639740560593e-05, 1.128665971177788e-05};
	constexpr Real coffd[25] = {6.624075337874347e-05, -6.004788860368634e-04, 2.607469373283419e-03, -7.253882663868505e-03, 1.460090147812115e-02, -2.278998801679410e-02, 2.890979755628365e-02, -3.090332841865437e-02, 2.864081241685065e-02, -2.346242023342941e-02, 1.758898775589444e-02, -1.313353046939937e-02, 9.021722692595935e-03, -3.282346255657025e-03, 1.420184381040870e-03, -6.380299711307833e-03, 5.025686336175834e-03, 4.958529166778002e-03, -4.180405898190081e-03, -5.347809413213879e-03, 4.033537744888153e-03, 2.501762336680086e-03, -1.818842040363272e-03, -6.956860848979441e-04, 4.733863147248488e-04};
		
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24]);
      } else if (digits <= 12) {
	constexpr Real cdiag[30] = {-2.365198302088124e-06, 2.532509802787800e-05, -1.296537952505377e-04, 4.232094084463001e-04, -9.910012567778351e-04, 1.775947306400662e-03, -2.537394110037294e-03, 2.972748898833850e-03, -2.914410596015076e-03, 2.426822952762970e-03, -1.734662064238163e-03, 1.070629855889581e-03, -5.701347107608747e-04, 2.599775140854510e-04, -9.984675852515150e-05, 2.602133849193166e-05, 7.916312005211445e-06, -1.011119573757128e-05, -8.536886328363358e-06, 4.830410119964790e-06, 2.259222340348041e-05, -1.585169034452381e-05, -1.796469429729689e-05, 1.393079426628784e-05, 1.248640412589774e-05, -9.761834712979377e-06, -4.660085362534675e-06, 3.676347054719938e-06, 9.569006500073318e-07, -7.168872115696289e-07};
	constexpr Real coffd[33] = {3.987091389679245e-06, -4.530466653601684e-05, 2.482202699268798e-04, -8.757112651021776e-04, 2.242385653041884e-03, -4.456033431322906e-03, 7.179047693407825e-03, -9.679175689803102e-03, 1.119660676257098e-02, -1.134724857533237e-02, 1.026356197723929e-02, -8.432004750793295e-03, 6.400243446867002e-03, -4.543110589277678e-03, 3.046931278095421e-03, -2.036384585843810e-03, 1.395901058514606e-03, -7.057178364754547e-04, 1.946535621651056e-04, -5.652424121139273e-04, 7.913110889513830e-04, 3.587169277761613e-04, -8.099287633106095e-04, -6.702740168842681e-04, 9.785690594905075e-04, 5.615069843258193e-04, -7.287094291575096e-04, -3.648193776827285e-04, 3.973573611639653e-04, 1.343712111701797e-04, -1.293553078167434e-04, -2.606246726396383e-05, 2.171173793246976e-05};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22],cdiag[23],cdiag[24],cdiag[25],cdiag[26],cdiag[27],cdiag[28],cdiag[29]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24],coffd[25],coffd[26],coffd[27],coffd[28],coffd[29],coffd[30],coffd[31],coffd[32]);
      }

      Vec Fdiag = select((R2 >= d2min_vec) & (R2 <= d2max_vec), fdiag*bsizeinv_vec, Vec::Zero());
      Vec Foffd = select((R2 >= d2min_vec) & (R2 <= d2max_vec), foffd*bsizeinv3_vec, Vec::Zero());
	  
      for (long i = 0; i < nd_; i++) {
        Vec Dprod = dX[0]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] + dX[2]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+2];
        Vec pl = Foffd*Dprod;
        Vtrg[i][0] += pl*dX[0] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0];
        Vtrg[i][1] += pl*dX[1] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vtrg[i][2] += pl*dX[2] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+2];
      }
    }
    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0].StoreAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1].StoreAligned(&Vt[1*nd_+i][t]);
      Vtrg[i][2].StoreAligned(&Vt[2*nd_+i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_*COORD_DIM; j++) {
      pot[i*nd_*COORD_DIM+j] += Vt[j][i];
    }
  }
}
#endif




#if defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void st3d_near_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* stoklet, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_*COORD_DIM, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_*COORD_DIM, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    if (COORD_DIM>2) {
      std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
    }
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];
  Vec bsizeinv3_vec = bsizeinv_vec*bsizeinv_vec*bsizeinv_vec;

  // load stokslet density
  sctl::Matrix<Real> Vs_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)stoklet, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vsrc(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vs_[s][0*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vs_[s][1*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+2] = Vec::Load1(&Vs_[s][2*nd_+i]);
    }
  }
  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);

  //#pragma omp parallel for schedule(static)
  // TODO: reinit to sctl::Vector is over limit
  //  #define NSRC_LIMIT 1000
  // work array memory stores compressed R2
  Real dX_work[COORD_DIM][NSRC_LIMIT*8];
  Real R2_work[NSRC_LIMIT*8];
  //sctl::StaticArray<Real, NSRC_LIMIT*8> R2_work_buff;
  // why converting to sctl::Vector is a bit slow...
  //sctl::Vector<Real> R2_work(NSRC_LIMIT*8, R2_work_buff, false);
  // work array memory stores compressed polynomial evals
  Real Fdiag_work[NSRC_LIMIT*8];
  Real Foffd_work[NSRC_LIMIT*8];
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
      Vtrg[i][0] = Vec::LoadAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1] = Vec::LoadAligned(&Vt[1*nd_+i][t]);
      Vtrg[i][2] = Vec::LoadAligned(&Vt[2*nd_+i][t]);
    }

    //int total_r = Nsrc*VecLen;
    int valid_r = 0;
    int source_cnt = 0;
    // store compressed R2 to explore 4/3 pi over 27 sparsity
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = zero_tmp;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }

      // store sparse data
      Mask_work[source_cnt] = (R2>= d2min_vec) & (R2 <= d2max_vec);
      int valid_cnt =  mask_popcnt(Mask_work[source_cnt]);
      mask_compress_store(Mask_work[source_cnt], dX[0], &dX_work[0][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], dX[1], &dX_work[1][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], dX[2], &dX_work[2][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], R2, &R2_work[0]+valid_r);
      start_ind[source_cnt] = valid_r;
      source_ind[source_cnt] = s;
      source_cnt += (valid_cnt>0);
      valid_r += valid_cnt;
    }

    // evaluate polynomial on compressed R2 and store in Pval
    for (sctl::Long s = 0; s < valid_r; s += VecLen) {
      Vec R2 = Vec::LoadAligned(&R2_work[0]+s);
      // evaluate the PSWF local kernel
      Vec fdiag,foffd;
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      if (digits<=3) {
	constexpr Real cdiag[8] = {-1.200465301327137e-02, 4.400592204092247e-02, -7.288597161347318e-02, 6.078910968844339e-02, -3.108708030232192e-02, 1.003402339570593e-02, 5.363405306153075e-03, -5.553171339387891e-03};
	constexpr Real coffd[11] = {2.250430121704944e-02, -9.787611559588733e-02, 2.070217916734389e-01, -2.524368328199907e-01, 2.259705057475586e-01, -2.420893817698530e-01, 1.866923371452701e-01, 3.395770252841146e-02, -4.311241235638091e-02, -1.308211487415907e-01, 9.152265431281228e-02};
		
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10]);
      } else if (digits <= 6) {
	constexpr Real cdiag[16] = {-6.778413276129699e-04, 4.066402381071667e-03, -1.124864783439655e-02, 1.920489141603694e-02, -2.281183769709533e-02, 1.999165544932320e-02, -1.322112536620177e-02, 6.698004916394447e-03, -2.867817976329824e-03, 8.503272900193856e-04, 6.879027750537969e-04, -6.923660302115852e-04, -4.270882823565761e-04, 3.190690821977045e-04, 3.599672200216620e-04, -2.326863819796524e-04};
	constexpr Real coffd[18] = { 1.184705266078750e-03, -7.928021809420677e-03, 2.512862663913759e-02, -5.082440691714102e-02, 7.474069197148014e-02, -8.588843617624770e-02, 7.996358530102793e-02, -6.499276257962841e-02, 5.539840177127529e-02, -4.014685171673929e-02, 4.879357438144119e-03, 1.419883695499086e-03, 3.525871341680183e-02, -2.706860456161926e-02, -1.886722540638105e-02, 1.460477431730375e-02, 9.998150833375779e-03, -6.859402138560325e-03};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17]);
      } else if (digits <= 9) {
	constexpr Real cdiag[23] = {-3.878153424079992e-05, 3.256189735159013e-04, -1.291899812828380e-03, 3.229916915849101e-03, -5.725266461380479e-03, 7.674125798699759e-03, -8.098740367932972e-03, 6.913185387623120e-03, -4.856062649177749e-03, 2.823670090265469e-03, -1.355279681198315e-03, 5.449639991689835e-04, -1.573565756195452e-04, -6.142949960309105e-05, 8.529221725570496e-05, 6.031578891512924e-05, -5.124161335103055e-05, -1.068570833585420e-04, 7.728626890477778e-05, 5.004058618345081e-05, -3.546124492943787e-05, -1.732639740560593e-05, 1.128665971177788e-05};
	constexpr Real coffd[25] = {6.624075337874347e-05, -6.004788860368634e-04, 2.607469373283419e-03, -7.253882663868505e-03, 1.460090147812115e-02, -2.278998801679410e-02, 2.890979755628365e-02, -3.090332841865437e-02, 2.864081241685065e-02, -2.346242023342941e-02, 1.758898775589444e-02, -1.313353046939937e-02, 9.021722692595935e-03, -3.282346255657025e-03, 1.420184381040870e-03, -6.380299711307833e-03, 5.025686336175834e-03, 4.958529166778002e-03, -4.180405898190081e-03, -5.347809413213879e-03, 4.033537744888153e-03, 2.501762336680086e-03, -1.818842040363272e-03, -6.956860848979441e-04, 4.733863147248488e-04};
		
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24]);
      } else if (digits <= 12) {
	constexpr Real cdiag[30] = {-2.365198302088124e-06, 2.532509802787800e-05, -1.296537952505377e-04, 4.232094084463001e-04, -9.910012567778351e-04, 1.775947306400662e-03, -2.537394110037294e-03, 2.972748898833850e-03, -2.914410596015076e-03, 2.426822952762970e-03, -1.734662064238163e-03, 1.070629855889581e-03, -5.701347107608747e-04, 2.599775140854510e-04, -9.984675852515150e-05, 2.602133849193166e-05, 7.916312005211445e-06, -1.011119573757128e-05, -8.536886328363358e-06, 4.830410119964790e-06, 2.259222340348041e-05, -1.585169034452381e-05, -1.796469429729689e-05, 1.393079426628784e-05, 1.248640412589774e-05, -9.761834712979377e-06, -4.660085362534675e-06, 3.676347054719938e-06, 9.569006500073318e-07, -7.168872115696289e-07};
	constexpr Real coffd[33] = {3.987091389679245e-06, -4.530466653601684e-05, 2.482202699268798e-04, -8.757112651021776e-04, 2.242385653041884e-03, -4.456033431322906e-03, 7.179047693407825e-03, -9.679175689803102e-03, 1.119660676257098e-02, -1.134724857533237e-02, 1.026356197723929e-02, -8.432004750793295e-03, 6.400243446867002e-03, -4.543110589277678e-03, 3.046931278095421e-03, -2.036384585843810e-03, 1.395901058514606e-03, -7.057178364754547e-04, 1.946535621651056e-04, -5.652424121139273e-04, 7.913110889513830e-04, 3.587169277761613e-04, -8.099287633106095e-04, -6.702740168842681e-04, 9.785690594905075e-04, 5.615069843258193e-04, -7.287094291575096e-04, -3.648193776827285e-04, 3.973573611639653e-04, 1.343712111701797e-04, -1.293553078167434e-04, -2.606246726396383e-05, 2.171173793246976e-05};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22],cdiag[23],cdiag[24],cdiag[25],cdiag[26],cdiag[27],cdiag[28],cdiag[29]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24],coffd[25],coffd[26],coffd[27],coffd[28],coffd[29],coffd[30],coffd[31],coffd[32]);
      }
      Vec Fdiag = fdiag*bsizeinv_vec;
      Vec Foffd = foffd*bsizeinv3_vec;

      Fdiag.StoreAligned(Fdiag_work+s);
      Foffd.StoreAligned(Foffd_work+s);
    }

    // expand compressed Fdiag_work and Foffd_work, then multiply by charge to accumulate
    for (sctl::Long s_ind = 0; s_ind < source_cnt; s_ind++) {
      int s = source_ind[s_ind];
      int start = start_ind[s_ind];
      Vec Fdiag = mask_expand_load(Mask_work[s_ind], zero_tmp, Fdiag_work+start);
      Vec Foffd = mask_expand_load(Mask_work[s_ind], zero_tmp, Foffd_work+start);
      Vec dX[3];
      dX[0] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[0]+start);
      dX[1] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[1]+start);
      dX[2] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[2]+start);

      for (long i = 0; i < nd_; i++) {
        Vec Dprod = dX[0]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] + dX[2]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+2];
        Vec pl = Foffd*Dprod;
        Vtrg[i][0] += pl*dX[0] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0];
        Vtrg[i][1] += pl*dX[1] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vtrg[i][2] += pl*dX[2] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+2];
      }
    }

    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0].StoreAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1].StoreAligned(&Vt[1*nd_+i][t]);
      Vtrg[i][2].StoreAligned(&Vt[2*nd_+i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_*COORD_DIM; j++) {
      pot[i*nd_*COORD_DIM+j] += Vt[j][i];
    }
  }
}
#endif




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void st3d_near_kernel_directcp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  if (digits[0] <= 3) st3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 6) st3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 9) st3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 12) st3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else st3d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
}












/* 2D Stokeslet local kernel charge to potential */ 
#if !defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void st2d_local_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* stoklet, const Real* xtarg, const Real* ytarg, const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_*COORD_DIM, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_*COORD_DIM, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];
  Vec bsizeinv2_vec = bsizeinv_vec*bsizeinv_vec;

  // load stokslet density
  sctl::Matrix<Real> Vs_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)stoklet, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vsrc(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vs_[s][0*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vs_[s][1*nd_+i]);
    }
  }
  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
  for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
    Vec Xtrg[COORD_DIM];
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
    }
    // load potential
    Vec Vtrg[nd_][COORD_DIM];
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0] = Vec::LoadAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1] = Vec::LoadAligned(&Vt[1*nd_+i][t]);
    }
	
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = Vec::Zero();
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }
      // evaluate the PSWF local kernel
      Vec fdiag,foffd;
      Vec one = 1.0e0;
      Vec R2inv = select((R2 >= d2min_vec) & (R2 <= d2max_vec), one/R2, Vec::Zero());	  
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      if (digits<=3) {
		constexpr Real cdiag[7] = { 1.959500857289356e-01, -3.284830946520660e-01, 2.375632650021020e-01, -1.647203776448787e-01, 8.742169743139379e-02, -3.893621673575272e-02, 1.136533379790958e-02};
		constexpr Real coffd[7] = {4.747691879437245e-01, 9.107713275042888e-02, -1.415654634495663e-01, 1.244870306229794e-01, -7.297850943103942e-02, 3.433763040218109e-02, -1.028068542798326e-02};
		
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6]);
      } else if (digits <= 6) {
		constexpr Real cdiag[13] = {1.760428366601711e-01, -2.667829654918805e-01, 1.716498771087513e-01, -1.618522666623655e-01, 1.515113469728826e-01, -1.203016457244573e-01, 7.794073952339156e-02, -4.126503902810891e-02, 1.798142000816606e-02, -6.442785881522806e-03, 1.989366825238204e-03, -6.020610620502964e-04, 1.312483270865528e-04};
		constexpr Real coffd[13] = {4.970489286389045e-01, 1.825850433000789e-02, -5.195229155580422e-02, 9.071250314080681e-02, -1.094953661935410e-01, 9.763910501821899e-02, -6.719282408462778e-02, 3.684960249355154e-02, -1.641919442780065e-02, 5.970360092846794e-03, -1.864318303460788e-03, 5.698889654928152e-04, -1.249637175924079e-04};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12]);
      } else if (digits <= 9) {
		constexpr Real cdiag[18] = {1.736143915441559e-01, -2.528524511664132e-01, 1.366833301793841e-01, -1.133134260344975e-01, 1.166157315304277e-01, -1.230131219471640e-01, 1.177354397255480e-01, -9.743725260814573e-02, 6.892764420306721e-02, -4.176646335575018e-02, 2.181773229782813e-02, -9.911049860722446e-03, 3.956822639098690e-03, -1.386800731893345e-03, 4.180703747636016e-04, -1.173474051654466e-04, 3.687848095523805e-05, -8.127868795781978e-06};
		constexpr Real coffd[18] = {4.996554864314352e-01, 3.024707944061178e-03, -1.252563561387401e-02, 3.262923591163785e-02, -6.018124183652912e-02, 8.388364991601044e-02, -9.211786685803622e-02, 8.205920867617028e-02, -6.060047230317656e-02, 3.773113221913438e-02, -2.007273358627204e-02, 9.237261128936858e-03, -3.723656940433528e-03, 1.314598064800165e-03, -3.983824058195751e-04, 1.123601701848997e-04, -3.549687892245856e-05, 7.845964177022328e-06};
		
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17]);
      } else if (digits <= 12) {
		constexpr Real cdiag[23] = {1.733152750766768e-01, -2.503328206628244e-01, 1.268600527552057e-01, -8.995976388336457e-02, 7.942081624384922e-02, -8.300977523275155e-02, 9.284334062963850e-02, -1.004012699509259e-01, 9.895312778087256e-02, -8.674351810930167e-02, 6.718781305488747e-02, -4.602798682590235e-02, 2.800089153537579e-02, -1.520313372523674e-02, 7.401950829354575e-03, -3.244080946487315e-03, 1.289503633216782e-03, -4.705846217802901e-04, 1.548748350614914e-04, -4.353472422735648e-05, 1.223128856508993e-05, -4.436710696609225e-06, 1.027730831992812e-06};
		constexpr Real coffd[24] = {4.999703955247617e-01, 3.476228991981245e-04, -1.954359598159400e-03, 7.015028488193076e-03, -1.808914320462718e-02, 3.575334826061605e-02, -5.644243240756530e-02, 7.323212818702446e-02, -7.975711091506987e-02, 7.411111289451824e-02, -5.952438854906328e-02, 4.176782845892292e-02, -2.583528706359139e-02, 1.419532411899560e-02, -6.975222066653686e-03, 3.082523135581995e-03, -1.230656390390550e-03, 4.466771457316551e-04, -1.489299099830532e-04, 4.513695239438677e-05, -1.187371208288823e-05, 3.051187377423047e-06, -9.917075658449903e-07, 2.182714524678886e-07};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23]);
      }
      Vec R2sc = R2*bsizeinv2_vec;
      Vec Fdiag = select((R2 >= d2min_vec) & (R2 <= d2max_vec), -0.25e0*sctl::veclog(R2sc)-fdiag, Vec::Zero());	  
      Vec Foffd = (0.5e0-foffd)*R2inv;

      for (long i = 0; i < nd_; i++) {
        Vec Dprod = dX[0]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vec pl = Foffd*Dprod;
        Vtrg[i][0] += pl*dX[0] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0];
        Vtrg[i][1] += pl*dX[1] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
      }
    }
    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0].StoreAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1].StoreAligned(&Vt[1*nd_+i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_*COORD_DIM; j++) {
      pot[i*nd_*COORD_DIM+j] += Vt[j][i];
    }
  }
}
#endif




#if defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void st2d_local_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* stoklet, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_*COORD_DIM, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_*COORD_DIM, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];
  Vec bsizeinv2_vec = bsizeinv_vec*bsizeinv_vec;

  // load stokslet density
  sctl::Matrix<Real> Vs_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)stoklet, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vsrc(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vs_[s][0*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vs_[s][1*nd_+i]);
    }
  }
  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);

  //#pragma omp parallel for schedule(static)
  // TODO: reinit to sctl::Vector is over limit
  // #define NSRC_LIMIT 1000
  // work array memory stores compressed R2
  Real dX_work[COORD_DIM][NSRC_LIMIT*8];
  Real R2_work[NSRC_LIMIT*8];
  //sctl::StaticArray<Real, NSRC_LIMIT*8> R2_work_buff;
  // why converting to sctl::Vector is a bit slow...
  //sctl::Vector<Real> R2_work(NSRC_LIMIT*8, R2_work_buff, false);
  // work array memory stores compressed polynomial evals
  Real Fdiag_work[NSRC_LIMIT*8];
  Real Foffd_work[NSRC_LIMIT*8];
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
      Vtrg[i][0] = Vec::LoadAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1] = Vec::LoadAligned(&Vt[1*nd_+i][t]);
    }

    //int total_r = Nsrc*VecLen;
    int valid_r = 0;
    int source_cnt = 0;
    // store compressed R2 to explore 4/3 pi over 27 sparsity
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = zero_tmp;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }

      // store sparse data
      Mask_work[source_cnt] = (R2 >= d2min_vec) & (R2 <= d2max_vec);
      int valid_cnt =  mask_popcnt(Mask_work[source_cnt]);
      mask_compress_store(Mask_work[source_cnt], dX[0], &dX_work[0][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], dX[1], &dX_work[1][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], R2, &R2_work[0]+valid_r);
      start_ind[source_cnt] = valid_r;
      source_ind[source_cnt] = s;
      source_cnt += (valid_cnt>0);
      valid_r += valid_cnt;
    }

    // evaluate polynomial on compressed R2 and store in Pval
    for (sctl::Long s = 0; s < valid_r; s += VecLen) {
      Vec R2 = Vec::LoadAligned(&R2_work[0]+s);
      // evaluate the PSWF local kernel
      Vec fdiag,foffd;
      Vec one = 1.0e0;
      Vec R2inv = select((R2 >= d2min_vec) & (R2 <= d2max_vec), one/R2, Vec::Zero());	  
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      if (digits<=3) {
		constexpr Real cdiag[7] = { 1.959500857289356e-01, -3.284830946520660e-01, 2.375632650021020e-01, -1.647203776448787e-01, 8.742169743139379e-02, -3.893621673575272e-02, 1.136533379790958e-02};
		constexpr Real coffd[7] = {4.747691879437245e-01, 9.107713275042888e-02, -1.415654634495663e-01, 1.244870306229794e-01, -7.297850943103942e-02, 3.433763040218109e-02, -1.028068542798326e-02};
		
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6]);
      } else if (digits <= 6) {
		constexpr Real cdiag[13] = {1.760428366601711e-01, -2.667829654918805e-01, 1.716498771087513e-01, -1.618522666623655e-01, 1.515113469728826e-01, -1.203016457244573e-01, 7.794073952339156e-02, -4.126503902810891e-02, 1.798142000816606e-02, -6.442785881522806e-03, 1.989366825238204e-03, -6.020610620502964e-04, 1.312483270865528e-04};
		constexpr Real coffd[13] = {4.970489286389045e-01, 1.825850433000789e-02, -5.195229155580422e-02, 9.071250314080681e-02, -1.094953661935410e-01, 9.763910501821899e-02, -6.719282408462778e-02, 3.684960249355154e-02, -1.641919442780065e-02, 5.970360092846794e-03, -1.864318303460788e-03, 5.698889654928152e-04, -1.249637175924079e-04};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12]);
      } else if (digits <= 9) {
		constexpr Real cdiag[18] = {1.736143915441559e-01, -2.528524511664132e-01, 1.366833301793841e-01, -1.133134260344975e-01, 1.166157315304277e-01, -1.230131219471640e-01, 1.177354397255480e-01, -9.743725260814573e-02, 6.892764420306721e-02, -4.176646335575018e-02, 2.181773229782813e-02, -9.911049860722446e-03, 3.956822639098690e-03, -1.386800731893345e-03, 4.180703747636016e-04, -1.173474051654466e-04, 3.687848095523805e-05, -8.127868795781978e-06};
		constexpr Real coffd[18] = {4.996554864314352e-01, 3.024707944061178e-03, -1.252563561387401e-02, 3.262923591163785e-02, -6.018124183652912e-02, 8.388364991601044e-02, -9.211786685803622e-02, 8.205920867617028e-02, -6.060047230317656e-02, 3.773113221913438e-02, -2.007273358627204e-02, 9.237261128936858e-03, -3.723656940433528e-03, 1.314598064800165e-03, -3.983824058195751e-04, 1.123601701848997e-04, -3.549687892245856e-05, 7.845964177022328e-06};
		
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17]);
      } else if (digits <= 12) {
		constexpr Real cdiag[23] = {1.733152750766768e-01, -2.503328206628244e-01, 1.268600527552057e-01, -8.995976388336457e-02, 7.942081624384922e-02, -8.300977523275155e-02, 9.284334062963850e-02, -1.004012699509259e-01, 9.895312778087256e-02, -8.674351810930167e-02, 6.718781305488747e-02, -4.602798682590235e-02, 2.800089153537579e-02, -1.520313372523674e-02, 7.401950829354575e-03, -3.244080946487315e-03, 1.289503633216782e-03, -4.705846217802901e-04, 1.548748350614914e-04, -4.353472422735648e-05, 1.223128856508993e-05, -4.436710696609225e-06, 1.027730831992812e-06};
		constexpr Real coffd[24] = {4.999703955247617e-01, 3.476228991981245e-04, -1.954359598159400e-03, 7.015028488193076e-03, -1.808914320462718e-02, 3.575334826061605e-02, -5.644243240756530e-02, 7.323212818702446e-02, -7.975711091506987e-02, 7.411111289451824e-02, -5.952438854906328e-02, 4.176782845892292e-02, -2.583528706359139e-02, 1.419532411899560e-02, -6.975222066653686e-03, 3.082523135581995e-03, -1.230656390390550e-03, 4.466771457316551e-04, -1.489299099830532e-04, 4.513695239438677e-05, -1.187371208288823e-05, 3.051187377423047e-06, -9.917075658449903e-07, 2.182714524678886e-07};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23]);
      }
      Vec R2sc = R2*bsizeinv2_vec;
      Vec Fdiag = select((R2 >= d2min_vec) & (R2 <= d2max_vec), -0.25e0*sctl::veclog(R2sc)-fdiag, Vec::Zero());	  
      Vec Foffd = (0.5e0-foffd)*R2inv;
      Fdiag.StoreAligned(Fdiag_work+s);
      Foffd.StoreAligned(Foffd_work+s);
    }

    // expand compressed Fdiag_work and Foffd_work, then multiply by charge to accumulate
    for (sctl::Long s_ind = 0; s_ind < source_cnt; s_ind++) {
      int s = source_ind[s_ind];
      int start = start_ind[s_ind];
      Vec Fdiag = mask_expand_load(Mask_work[s_ind], zero_tmp, Fdiag_work+start);
      Vec Foffd = mask_expand_load(Mask_work[s_ind], zero_tmp, Foffd_work+start);
      Vec dX[2];
      dX[0] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[0]+start);
      dX[1] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[1]+start);

      for (long i = 0; i < nd_; i++) {
        Vec Dprod = dX[0]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vec pl = Foffd*Dprod;
        Vtrg[i][0] += pl*dX[0] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0];
        Vtrg[i][1] += pl*dX[1] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
      }
    }

    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0].StoreAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1].StoreAligned(&Vt[1*nd_+i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_*COORD_DIM; j++) {
      pot[i*nd_*COORD_DIM+j] += Vt[j][i];
    }
  }
}
#endif




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void st2d_local_kernel_directcp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  if (digits[0] <= 3) st2d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 6) st2d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 9) st2d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 12) st2d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else st2d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
}









/* 2D Stokeslet near kernel charge to potential */ 
#if !defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void st2d_near_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* stoklet, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_*COORD_DIM, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_*COORD_DIM, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];
  Vec bsizeinv2_vec = bsizeinv_vec*bsizeinv_vec;

  // load stokslet density
  sctl::Matrix<Real> Vs_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)stoklet, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vsrc(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vs_[s][0*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vs_[s][1*nd_+i]);
    }
  }
  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
  for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
    Vec Xtrg[COORD_DIM];
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
    }
    // load potential
    Vec Vtrg[nd_][COORD_DIM];
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0] = Vec::LoadAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1] = Vec::LoadAligned(&Vt[1*nd_+i][t]);
    }
	
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = Vec::Zero();
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }
      // evaluate the PSWF local kernel
      Vec fdiag,foffd;
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      if (digits<=3) {
	constexpr Real cdiag[8] = {-3.998414794184642e-03, 1.324285664942096e-02, -1.893609240136285e-02, 1.988081279390220e-02, -1.565735340857958e-02, 6.009474290405808e-03, 1.461928833976798e-03, -1.654777943685357e-03};
	constexpr Real coffd[10] = {6.825370013492135e-03, -2.738568594707985e-02, 4.963206753307496e-02, -6.582798379855469e-02, 8.027100286820954e-02, -6.519957095909519e-02, 7.550079217492817e-03, 1.268089291030193e-03, 3.828786857898434e-02, -2.579301318235034e-02};
	
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9]);
      } else if (digits <= 6) {
	constexpr Real cdiag[15] = {-1.456047650277214e-04, 9.079956726538666e-04, -2.623397818356276e-03, 4.663208241908292e-03, -5.730901904803670e-03, 5.200683558903413e-03, -3.641728717601099e-03, 2.050744701335512e-03, -8.785210517186211e-04, 1.146038553901061e-04, 7.700211098051572e-05, 1.081512431354119e-04, -6.706057402490595e-05, -9.559983599910285e-05, 6.069825378925619e-05};
	constexpr Real coffd[18] = {2.429808372877533e-04, -1.671550929931438e-03, 5.450474945249396e-03, -1.127345545629040e-02, 1.679652911724135e-02, -1.944228593275497e-02, 1.827746384664138e-02, -1.484220332387097e-02, 1.185852189617975e-02, -8.227913132868213e-03, 2.123327526893051e-03, -6.471398957193231e-04, 5.603797617073294e-03, -4.121918190402536e-03, -2.664246417934463e-03, 2.029835104534074e-03, 1.522474967786489e-03, -1.014967217619112e-03};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17]);
      } else if (digits <= 9) {
	constexpr Real cdiag[22] = {-7.110615341771953e-06, 6.141875873809334e-05, -2.506676253528090e-04, 6.445589936989885e-04, -1.175033860647445e-03, 1.620124721429269e-03, -1.759441713321558e-03, 1.546330114969403e-03, -1.119598997773590e-03, 6.758915878568045e-04, -3.444264068102575e-04, 1.441352916633147e-04, -3.528985235015155e-05, -9.486598365419032e-07, -1.180126115423290e-05, 7.380015531148850e-06, 2.069507010796023e-05, -1.475518926368035e-05, -9.433880958019434e-06, 6.602811307928483e-06, 3.757785599178625e-06, -2.386711531075020e-06};
	constexpr Real coffd[25] = { 1.173975955150262e-05, -1.088357519001305e-04, 4.826945683628972e-04, -1.369185474159939e-03, 2.804273486454630e-03, -4.442865965632523e-03, 5.702647228061498e-03, -6.142361398128255e-03, 5.709935004220391e-03, -4.679442074016084e-03, 3.484512488815244e-03, -2.508099104179134e-03, 1.672185985854015e-03, -7.400863791144803e-04, 3.740867712550477e-04, -9.033823107353169e-04, 6.780284981485554e-04, 5.849023635865252e-04, -4.919195500725383e-04, -6.794312442009345e-04, 5.011871121452637e-04, 3.118709922271007e-04, -2.227512396633936e-04, -8.910545106079971e-05, 5.940131461123234e-05};
		
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24]);
      } else if (digits <= 12) {
	constexpr Real cdiag[30] = {-3.857407689956136e-07, 4.223762137705732e-06, -2.211255054653232e-05, 7.380540793879963e-05, -1.767076372234046e-04, 3.237659530274397e-04, -4.729289704973748e-04, 5.664945924034799e-04, -5.679462362790733e-04, 4.838714850249418e-04, -3.542882471926727e-04, 2.245805649065638e-04, -1.233427319614584e-04, 5.835230132168445e-05, -2.405055485496506e-05, 8.294431040922999e-06, 1.695951845876735e-07, -2.690159478561324e-06, -1.654500265729366e-06, 2.730582486100622e-06, 3.722435990228163e-06, -4.236963197620414e-06, -2.930026230415609e-06, 3.253831542690141e-06, 1.968228491238935e-06, -1.917779603045695e-06, -7.142152721166487e-07, 6.456833673045986e-07, 1.395323993802861e-07, -1.120740129435944e-07};
	constexpr Real coffd[32] = {6.329756000924419e-07, -7.327697464512041e-06, 4.087585448236432e-05, -1.467017507678289e-04, 3.817558577488382e-04, -7.699843732206189e-04, 1.257177653830602e-03, -1.714577678688572e-03, 2.001731684123898e-03, -2.041946413992746e-03, 1.853558605303895e-03, -1.522120939480519e-03, 1.146782939466555e-03, -8.072732448832567e-04, 5.492678428509865e-04, -3.563406094010927e-04, 1.887594181488638e-04, -1.050871686050720e-04, 1.464128935127148e-04, -1.092658069019802e-04, -8.433862896306294e-05, 8.037073345345367e-05, 1.522514982941662e-04, -1.215832904612562e-04, -1.215793564674542e-04, 9.448183907894859e-05, 7.654193769017407e-05, -5.612862382307245e-05, -2.735793161494765e-05, 1.930096294532848e-05, 5.198557164750587e-06, -3.487738839117130e-06};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22],cdiag[23],cdiag[24],cdiag[25],cdiag[26],cdiag[27],cdiag[28],cdiag[29]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24],coffd[25],coffd[26],coffd[27],coffd[28],coffd[29],coffd[30],coffd[31]);
      }
      
      Vec Fdiag = select((R2 >= d2min_vec) & (R2 <= d2max_vec), fdiag, Vec::Zero());
      Vec Foffd = select((R2 >= d2min_vec) & (R2 <= d2max_vec), foffd*bsizeinv2_vec, Vec::Zero());
	  
      for (long i = 0; i < nd_; i++) {
        Vec Dprod = dX[0]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vec pl = Foffd*Dprod;
        Vtrg[i][0] += pl*dX[0] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0];
        Vtrg[i][1] += pl*dX[1] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
      }
    }
    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0].StoreAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1].StoreAligned(&Vt[1*nd_+i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_*COORD_DIM; j++) {
      pot[i*nd_*COORD_DIM+j] += Vt[j][i];
    }
  }
}
#endif




#if defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void st2d_near_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* stoklet, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_*COORD_DIM, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_*COORD_DIM, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];
  Vec bsizeinv2_vec = bsizeinv_vec*bsizeinv_vec;

  // load stokslet density
  sctl::Matrix<Real> Vs_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)stoklet, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vsrc(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vs_[s][0*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vs_[s][1*nd_+i]);
    }
  }
  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);

  //#pragma omp parallel for schedule(static)
  // TODO: reinit to sctl::Vector is over limit
  // #define NSRC_LIMIT 1000
  // work array memory stores compressed R2
  Real dX_work[COORD_DIM][NSRC_LIMIT*8];
  Real R2_work[NSRC_LIMIT*8];
  //sctl::StaticArray<Real, NSRC_LIMIT*8> R2_work_buff;
  // why converting to sctl::Vector is a bit slow...
  //sctl::Vector<Real> R2_work(NSRC_LIMIT*8, R2_work_buff, false);
  // work array memory stores compressed polynomial evals
  Real Fdiag_work[NSRC_LIMIT*8];
  Real Foffd_work[NSRC_LIMIT*8];
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
      Vtrg[i][0] = Vec::LoadAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1] = Vec::LoadAligned(&Vt[1*nd_+i][t]);
    }

    //int total_r = Nsrc*VecLen;
    int valid_r = 0;
    int source_cnt = 0;
    // store compressed R2 to explore 4/3 pi over 27 sparsity
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = zero_tmp;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }

      // store sparse data
      Mask_work[source_cnt] = (R2>= d2min_vec) & (R2 <= d2max_vec);
      int valid_cnt =  mask_popcnt(Mask_work[source_cnt]);
      mask_compress_store(Mask_work[source_cnt], dX[0], &dX_work[0][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], dX[1], &dX_work[1][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], R2, &R2_work[0]+valid_r);
      start_ind[source_cnt] = valid_r;
      source_ind[source_cnt] = s;
      source_cnt += (valid_cnt>0);
      valid_r += valid_cnt;
    }

    // evaluate polynomial on compressed R2 and store in Pval
    for (sctl::Long s = 0; s < valid_r; s += VecLen) {
      Vec R2 = Vec::LoadAligned(&R2_work[0]+s);
      // evaluate the PSWF local kernel
      Vec fdiag,foffd;
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      if (digits<=3) {
	constexpr Real cdiag[8] = {-3.998414794184642e-03, 1.324285664942096e-02, -1.893609240136285e-02, 1.988081279390220e-02, -1.565735340857958e-02, 6.009474290405808e-03, 1.461928833976798e-03, -1.654777943685357e-03};
	constexpr Real coffd[10] = {6.825370013492135e-03, -2.738568594707985e-02, 4.963206753307496e-02, -6.582798379855469e-02, 8.027100286820954e-02, -6.519957095909519e-02, 7.550079217492817e-03, 1.268089291030193e-03, 3.828786857898434e-02, -2.579301318235034e-02};
		
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9]);
      } else if (digits <= 6) {
	constexpr Real cdiag[15] = {-1.456047650277214e-04, 9.079956726538666e-04, -2.623397818356276e-03, 4.663208241908292e-03, -5.730901904803670e-03, 5.200683558903413e-03, -3.641728717601099e-03, 2.050744701335512e-03, -8.785210517186211e-04, 1.146038553901061e-04, 7.700211098051572e-05, 1.081512431354119e-04, -6.706057402490595e-05, -9.559983599910285e-05, 6.069825378925619e-05};
	constexpr Real coffd[18] = {2.429808372877533e-04, -1.671550929931438e-03, 5.450474945249396e-03, -1.127345545629040e-02, 1.679652911724135e-02, -1.944228593275497e-02, 1.827746384664138e-02, -1.484220332387097e-02, 1.185852189617975e-02, -8.227913132868213e-03, 2.123327526893051e-03, -6.471398957193231e-04, 5.603797617073294e-03, -4.121918190402536e-03, -2.664246417934463e-03, 2.029835104534074e-03, 1.522474967786489e-03, -1.014967217619112e-03};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17]);
      } else if (digits <= 9) {
	constexpr Real cdiag[22] = {-7.110615341771953e-06, 6.141875873809334e-05, -2.506676253528090e-04, 6.445589936989885e-04, -1.175033860647445e-03, 1.620124721429269e-03, -1.759441713321558e-03, 1.546330114969403e-03, -1.119598997773590e-03, 6.758915878568045e-04, -3.444264068102575e-04, 1.441352916633147e-04, -3.528985235015155e-05, -9.486598365419032e-07, -1.180126115423290e-05, 7.380015531148850e-06, 2.069507010796023e-05, -1.475518926368035e-05, -9.433880958019434e-06, 6.602811307928483e-06, 3.757785599178625e-06, -2.386711531075020e-06};
	constexpr Real coffd[25] = { 1.173975955150262e-05, -1.088357519001305e-04, 4.826945683628972e-04, -1.369185474159939e-03, 2.804273486454630e-03, -4.442865965632523e-03, 5.702647228061498e-03, -6.142361398128255e-03, 5.709935004220391e-03, -4.679442074016084e-03, 3.484512488815244e-03, -2.508099104179134e-03, 1.672185985854015e-03, -7.400863791144803e-04, 3.740867712550477e-04, -9.033823107353169e-04, 6.780284981485554e-04, 5.849023635865252e-04, -4.919195500725383e-04, -6.794312442009345e-04, 5.011871121452637e-04, 3.118709922271007e-04, -2.227512396633936e-04, -8.910545106079971e-05, 5.940131461123234e-05};
		
	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24]);
      } else if (digits <= 12) {
	constexpr Real cdiag[30] = {-3.857407689956136e-07, 4.223762137705732e-06, -2.211255054653232e-05, 7.380540793879963e-05, -1.767076372234046e-04, 3.237659530274397e-04, -4.729289704973748e-04, 5.664945924034799e-04, -5.679462362790733e-04, 4.838714850249418e-04, -3.542882471926727e-04, 2.245805649065638e-04, -1.233427319614584e-04, 5.835230132168445e-05, -2.405055485496506e-05, 8.294431040922999e-06, 1.695951845876735e-07, -2.690159478561324e-06, -1.654500265729366e-06, 2.730582486100622e-06, 3.722435990228163e-06, -4.236963197620414e-06, -2.930026230415609e-06, 3.253831542690141e-06, 1.968228491238935e-06, -1.917779603045695e-06, -7.142152721166487e-07, 6.456833673045986e-07, 1.395323993802861e-07, -1.120740129435944e-07};
	constexpr Real coffd[32] = {6.329756000924419e-07, -7.327697464512041e-06, 4.087585448236432e-05, -1.467017507678289e-04, 3.817558577488382e-04, -7.699843732206189e-04, 1.257177653830602e-03, -1.714577678688572e-03, 2.001731684123898e-03, -2.041946413992746e-03, 1.853558605303895e-03, -1.522120939480519e-03, 1.146782939466555e-03, -8.072732448832567e-04, 5.492678428509865e-04, -3.563406094010927e-04, 1.887594181488638e-04, -1.050871686050720e-04, 1.464128935127148e-04, -1.092658069019802e-04, -8.433862896306294e-05, 8.037073345345367e-05, 1.522514982941662e-04, -1.215832904612562e-04, -1.215793564674542e-04, 9.448183907894859e-05, 7.654193769017407e-05, -5.612862382307245e-05, -2.735793161494765e-05, 1.930096294532848e-05, 5.198557164750587e-06, -3.487738839117130e-06};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22],cdiag[23],cdiag[24],cdiag[25],cdiag[26],cdiag[27],cdiag[28],cdiag[29]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24],coffd[25],coffd[26],coffd[27],coffd[28],coffd[29],coffd[30],coffd[31]);
      }
      Vec Fdiag = fdiag;
      Vec Foffd = foffd*bsizeinv2_vec;

      Fdiag.StoreAligned(Fdiag_work+s);
      Foffd.StoreAligned(Foffd_work+s);
    }

    // expand compressed Fdiag_work and Foffd_work, then multiply by charge to accumulate
    for (sctl::Long s_ind = 0; s_ind < source_cnt; s_ind++) {
      int s = source_ind[s_ind];
      int start = start_ind[s_ind];
      Vec Fdiag = mask_expand_load(Mask_work[s_ind], zero_tmp, Fdiag_work+start);
      Vec Foffd = mask_expand_load(Mask_work[s_ind], zero_tmp, Foffd_work+start);
      Vec dX[2];
      dX[0] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[0]+start);
      dX[1] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[1]+start);

      for (long i = 0; i < nd_; i++) {
        Vec Dprod = dX[0]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vec pl = Foffd*Dprod;
        Vtrg[i][0] += pl*dX[0] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0];
        Vtrg[i][1] += pl*dX[1] + Fdiag*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
      }
    }

    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0].StoreAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1].StoreAligned(&Vt[1*nd_+i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_*COORD_DIM; j++) {
      pot[i*nd_*COORD_DIM+j] += Vt[j][i];
    }
  }
}
#endif




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void st2d_near_kernel_directcp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  if (digits[0] <= 3) st2d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 6) st2d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 9) st2d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 12) st2d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
  else st2d_near_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot);
}











/* 2D Log local kernel dipole to potential */ 
#if !defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void log_local_kernel_directdp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* dipvec, const Real* xtarg, const Real* ytarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];

  // load dipoles
  sctl::Matrix<Real> Vs_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)dipvec, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vsrc(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vs_[s][0*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vs_[s][1*nd_+i]);
    }
  }
  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
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
        R2 += dX[k]*dX[k];
      }
      // evaluate the PSWF local kernel
      Vec fres;
      Vec one = 1.0e0;
      Vec R2inv = select((R2 >= d2min_vec) & (R2 <= d2max_vec), one/R2, Vec::Zero());	  
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      if (digits<=3) {
		constexpr Real coefs[7] = {9.334285864639144e-01, 2.169971553925394e-01, -3.006861735844956e-01, 2.360122612383026e-01, -1.211621477124991e-01, 4.683707884859314e-02, -1.169643948451524e-02};
		fres = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6]);
      } else if (digits <= 6) {
		constexpr Real coefs[12] = {9.922973647646393e-01, 4.517941536621922e-02, -1.212953577289340e-01, 1.989164113250858e-01, -2.245055284970060e-01, 1.865422171066172e-01, -1.193543856601842e-01, 6.036222021672452e-02, -2.433775183640557e-02, 8.334389683592744e-03, -2.804387273230964e-03, 6.652266634776942e-04};
		fres = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11]);
      } else if (digits <= 9) {
		constexpr Real coefs[18] = {9.990603915706698e-01, 7.897602108744739e-03, -3.122669887520646e-02, 7.746531848059268e-02, -1.357139968250016e-01, 1.792398837538570e-01, -1.860703309800472e-01, 1.563404194632586e-01, -1.086719295561396e-01, 6.356400646965495e-02, -3.171623791444703e-02, 1.366331708296591e-02, -5.141772077832021e-03, 1.695376406990721e-03, -4.834996012361570e-04, 1.266855672999249e-04, -3.592580267758308e-05, 7.390593201473847e-06};
		fres = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17]);
      } else if (digits <= 12) {
		constexpr Real coefs[23] = {9.998827855396359e-01, 1.279729485988923e-03, -6.667320227000562e-03, 2.210425656192649e-02, -5.247441065369307e-02, 9.517926389227593e-02, -1.374603602142662e-01, 1.626721895736130e-01, -1.611253399469532e-01, 1.357864486500207e-01, -9.865004461170723e-02, 6.245699337710517e-02, -3.477478798108834e-02, 1.716122552233221e-02, -7.556538868258740e-03, 2.985503961942498e-03, -1.065227945902495e-03, 3.461493082345737e-04, -1.017890137358330e-04, 2.630178122754457e-05, -6.571760731648724e-06, 1.937885323237026e-06, -3.943162901113174e-07};
		fres = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18],coefs[19],coefs[20],coefs[21],coefs[22]);
      }
      Vec Fval = (fres-one)*R2inv;

      for (long i = 0; i < nd_; i++) {
        Vec Dprod = dX[0]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vtrg[i] += Fval*Dprod;
      }
    }
    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i].StoreAligned(&Vt[i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_; j++) {
      pot[i*nd_+j] += Vt[j][i];
    }
  }
}
#endif




#if defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void log_local_kernel_directdp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* dipvec, const Real* xtarg,const Real* ytarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];

  // load stokslet density
  sctl::Matrix<Real> Vs_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)dipvec, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vsrc(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vs_[s][0*nd_+i]);
      Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vs_[s][1*nd_+i]);
    }
  }
  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);

  //#pragma omp parallel for schedule(static)
  // TODO: reinit to sctl::Vector is over limit
  // #define NSRC_LIMIT 1000
  // work array memory stores compressed R2
  Real dX_work[COORD_DIM][NSRC_LIMIT*8];
  Real R2_work[NSRC_LIMIT*8];
  //sctl::StaticArray<Real, NSRC_LIMIT*8> R2_work_buff;
  // why converting to sctl::Vector is a bit slow...
  //sctl::Vector<Real> R2_work(NSRC_LIMIT*8, R2_work_buff, false);
  // work array memory stores compressed polynomial evals
  Real Fres_work[NSRC_LIMIT*8];
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

    //int total_r = Nsrc*VecLen;
    int valid_r = 0;
    int source_cnt = 0;
    // store compressed R2 to explore 4/3 pi over 27 sparsity
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = zero_tmp;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }

      // store sparse data
      Mask_work[source_cnt] = (R2 >= d2min_vec) & (R2 <= d2max_vec);
      int valid_cnt =  mask_popcnt(Mask_work[source_cnt]);
      mask_compress_store(Mask_work[source_cnt], dX[0], &dX_work[0][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], dX[1], &dX_work[1][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], R2, &R2_work[0]+valid_r);
      start_ind[source_cnt] = valid_r;
      source_ind[source_cnt] = s;
      source_cnt += (valid_cnt>0);
      valid_r += valid_cnt;
    }

    // evaluate polynomial on compressed R2 and store in Pval
    for (sctl::Long s = 0; s < valid_r; s += VecLen) {
      Vec R2 = Vec::LoadAligned(&R2_work[0]+s);
      // evaluate the PSWF local kernel
      Vec fres;
      Vec one = 1.0e0;
      Vec R2inv = select((R2 >= d2min_vec) & (R2 <= d2max_vec), one/R2, Vec::Zero());	  
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      if (digits<=3) {
		constexpr Real coefs[7] = {9.334285864639144e-01, 2.169971553925394e-01, -3.006861735844956e-01, 2.360122612383026e-01, -1.211621477124991e-01, 4.683707884859314e-02, -1.169643948451524e-02};
		fres = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6]);
      } else if (digits <= 6) {
		constexpr Real coefs[12] = {9.922973647646393e-01, 4.517941536621922e-02, -1.212953577289340e-01, 1.989164113250858e-01, -2.245055284970060e-01, 1.865422171066172e-01, -1.193543856601842e-01, 6.036222021672452e-02, -2.433775183640557e-02, 8.334389683592744e-03, -2.804387273230964e-03, 6.652266634776942e-04};
		fres = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11]);
      } else if (digits <= 9) {
		constexpr Real coefs[18] = {9.990603915706698e-01, 7.897602108744739e-03, -3.122669887520646e-02, 7.746531848059268e-02, -1.357139968250016e-01, 1.792398837538570e-01, -1.860703309800472e-01, 1.563404194632586e-01, -1.086719295561396e-01, 6.356400646965495e-02, -3.171623791444703e-02, 1.366331708296591e-02, -5.141772077832021e-03, 1.695376406990721e-03, -4.834996012361570e-04, 1.266855672999249e-04, -3.592580267758308e-05, 7.390593201473847e-06};
		fres = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17]);
      } else if (digits <= 12) {
		constexpr Real coefs[23] = {9.998827855396359e-01, 1.279729485988923e-03, -6.667320227000562e-03, 2.210425656192649e-02, -5.247441065369307e-02, 9.517926389227593e-02, -1.374603602142662e-01, 1.626721895736130e-01, -1.611253399469532e-01, 1.357864486500207e-01, -9.865004461170723e-02, 6.245699337710517e-02, -3.477478798108834e-02, 1.716122552233221e-02, -7.556538868258740e-03, 2.985503961942498e-03, -1.065227945902495e-03, 3.461493082345737e-04, -1.017890137358330e-04, 2.630178122754457e-05, -6.571760731648724e-06, 1.937885323237026e-06, -3.943162901113174e-07};
		fres = EvalPolynomial(xtmp.get(),coefs[0],coefs[1],coefs[2],coefs[3],coefs[4],coefs[5],coefs[6],coefs[7],coefs[8],coefs[9],coefs[10],coefs[11],coefs[12],coefs[13],coefs[14],coefs[15],coefs[16],coefs[17],coefs[18],coefs[19],coefs[20],coefs[21],coefs[22]);
      }
      Vec Fres = (fres-one)*R2inv;
      Fres.StoreAligned(Fres_work+s);
    }

    // expand compressed Fdiag_work and Foffd_work, then multiply by charge to accumulate
    for (sctl::Long s_ind = 0; s_ind < source_cnt; s_ind++) {
      int s = source_ind[s_ind];
      int start = start_ind[s_ind];
      Vec Fres = mask_expand_load(Mask_work[s_ind], zero_tmp, Fres_work+start);
      Vec dX[2];
      dX[0] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[0]+start);
      dX[1] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[1]+start);

      for (long i = 0; i < nd_; i++) {
        Vec Dprod = dX[0]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsrc[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vtrg[i] += Fres*Dprod;
      }
    }

    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i].StoreAligned(&Vt[i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_; j++) {
      pot[i*nd_+j] += Vt[j][i];
    }
  }
}
#endif




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void log_local_kernel_directdp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits, const Real* rsc, const Real* cen, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* dipvec, const Real* xtarg, const Real* ytarg, const int32_t* nt, Real* pot) {
	if (digits[0] <= 3) log_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,3,2>(nd,rsc,cen,d2min,d2max,sources,ns,dipvec,xtarg,ytarg,nt,pot);
	else if (digits[0] <= 6) log_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,6,2>(nd,rsc,cen,d2min,d2max,sources,ns,dipvec,xtarg,ytarg,nt,pot);
	else if (digits[0] <= 9) log_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,9,2>(nd,rsc,cen,d2min,d2max,sources,ns,dipvec,xtarg,ytarg,nt,pot);
	else if (digits[0] <= 12) log_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,12,2>(nd,rsc,cen,d2min,d2max,sources,ns,dipvec,xtarg,ytarg,nt,pot);
	else log_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,-1,2>(nd,rsc,cen,d2min,d2max,sources,ns,dipvec,xtarg,ytarg,nt,pot);
}



/* 3D Stresslet local kernel to potential */ 
#if !defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void st3d_local_kernel_directdp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* strslet, const Real* strsvec,const Real* xtarg, const Real* ytarg, const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_*COORD_DIM, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_*COORD_DIM, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    if (COORD_DIM>2) {
      std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
    }
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];

  // load strslet and strsvec
  sctl::Matrix<Real> Vsl_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)strslet, nd_*Nsrc),false);
  sctl::Matrix<Real> Vsv_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)strsvec, nd_*Nsrc),false);  
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vslet(Nsrc*nd_*COORD_DIM);
  sctl::Vector<Vec> Vsvec(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vsl_[s][0*nd_+i]);
      Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vsl_[s][1*nd_+i]);
      Vslet[s*nd_*COORD_DIM+i*COORD_DIM+2] = Vec::Load1(&Vsl_[s][2*nd_+i]);
    }
  }
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vsv_[s][0*nd_+i]);
      Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vsv_[s][1*nd_+i]);
      Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+2] = Vec::Load1(&Vsv_[s][2*nd_+i]);
    }
  }
  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
  for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
    Vec Xtrg[COORD_DIM];
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
    }
    // load potential
    Vec Vtrg[nd_][COORD_DIM];
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0] = Vec::LoadAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1] = Vec::LoadAligned(&Vt[1*nd_+i][t]);
      Vtrg[i][2] = Vec::LoadAligned(&Vt[2*nd_+i][t]);
    }
	
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = Vec::Zero();
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }
      // evaluate the PSWF local kernel
      Vec fdiag,foffd;
      Vec Rinv = sctl::approx_rsqrt<digits>(R2, (R2 >= d2min_vec) & (R2 <= d2max_vec));
      Vec Rinv3 = Rinv*Rinv*Rinv;
	  Vec Rinv5 = Rinv3*Rinv*Rinv;
      Vec xtmp = FMA(R2,Rinv,cen_vec)*rsc_vec;
      if (digits<=3) {
		constexpr Real cdiag[10] = {-4.909552162703472e-01, -2.638326207042311e-01, 1.426957484649616e+00, 5.133112420191036e-01, -1.692721168557828e+00, -3.331428832347214e-01, 9.764073959118829e-01, 7.923558379145755e-02, -2.299454085259451e-01, -5.708623886772283e-03};
		constexpr Real coffd[10] = {-3.875806562298824e-01, 4.031306201463891e-01, 3.626294455986704e-01, -2.307948583865092e-01, -3.959286768274001e-01, 9.858855313084947e-02, 2.196925355210589e-01, -3.006434523801740e-02, -5.244641630714497e-02, 5.535941142357893e-03};
		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9]);
      } else if (digits <= 6) {
		constexpr Real cdiag[18] = {-4.223999157922261e-01, 6.639496106000218e-01, 1.362863153121536e+00, -2.738405892193764e+00, -1.352748482947856e+00, 4.607335078399908e+00, -8.963343367930998e-02, -4.158601207960919e+00, 1.166122847649966e+00, 2.274176040024465e+00, -1.031865672685651e+00, -8.228724052420864e-01, 4.845591499488178e-01, 2.060153768358927e-01, -1.351013559403134e-01, -3.479488210553128e-02, 1.816899726122462e-02, 3.163540467786774e-03};
		constexpr Real coffd[16] = {-2.415969568205024e-01, 6.437162555004452e-01, -8.886684055725748e-02, -1.004963550079139e+00, 3.023900125189125e-01, 1.203207169828373e+00, -5.173536934661076e-01, -9.517824419372273e-01, 5.159051120129571e-01, 5.006553080725374e-01, -3.140960612583585e-01, -1.762122302038716e-01, 1.120404160726096e-01, 4.001650697669312e-02, -1.843399330899232e-02, -4.648881387204539e-03};

		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15]);
      } else if (digits <= 9) {
		constexpr Real cdiag[25] = {-2.692062485564092e-01, 9.919752014657248e-01, -2.475276651563518e-01, -3.483047290967188e+00, 4.110450253556166e+00, 3.902288269852157e+00, -9.281592572398081e+00, 1.402859806439762e-01, 1.005854710632259e+01, -4.433065768464961e+00, -6.250745674629706e+00, 5.074721901153145e+00, 2.262821078806491e+00, -3.261653666605643e+00, -3.236650597906456e-01, 1.443628453346002e+00, -1.363581893372792e-01, -4.733239006426434e-01, 1.088198030976763e-01, 1.163379837058996e-01, -3.899147580748667e-02, -1.996054535142191e-02, 8.304694659040938e-03, 1.813315378685804e-03, -8.561172730301156e-04};
		constexpr Real coffd[24] = {-1.328623832934818e-01, 5.998646490238386e-01, -7.130999555454819e-01, -6.581127232402704e-01, 1.863735047664648e+00, 8.380501392006251e-02, -2.730121970220487e+00, 1.060918785747619e+00, 2.448001823516434e+00, -1.790934790189596e+00, -1.358784546977841e+00, 1.600552255725127e+00, 4.163703074190580e-01, -9.480185933976362e-01, -7.857097865269225e-03, 4.061524503251803e-01, -5.947496832482102e-02, -1.312977911169639e-01, 3.145743836336123e-02, 3.205149253369654e-02, -8.373574144115992e-03, -5.476849877595668e-03, 1.009857050121354e-03, 4.960781625413800e-04};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22],cdiag[23],cdiag[24]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23]);
      } else if (digits <= 12) {
		constexpr Real cdiag[33] = {-1.360683826314615e-01, 8.302066635143668e-01, -1.514928718394161e+00, -9.381617318207349e-01, 6.813719028589369e+00, -5.988863853057117e+00, -8.472508766732407e+00, 1.809741443831257e+01, -1.273523386015356e+00, -2.203611105917701e+01, 1.439588582795162e+01, 1.352189057132570e+01, -1.847340430858263e+01, -2.444667103472067e+00, 1.341798582270320e+01, -3.074993686699973e+00, -6.521211003969547e+00, 3.398385460000120e+00, 2.180822019664342e+00, -1.978605521170600e+00, -4.466586996459809e-01, 8.228831786637951e-01, 4.803782852447806e-03, -2.660309603180980e-01, 3.972099950624761e-02, 6.873638809031379e-02, -1.933391187598765e-02, -1.395009222278865e-02, 5.664769109722911e-03, 2.024357518311426e-03, -1.064954429644160e-03, -1.570495227838364e-04, 9.988186457121628e-05};
		  constexpr Real coffd[31] = {-6.115079449085273e-02, 4.128039371348927e-01, -9.881137625293366e-01, 5.143473444562541e-01, 1.885479167316791e+00, -3.050423360463870e+00, -9.475794240844918e-01, 5.634324267214952e+00, -2.338305998487992e+00, -5.502717038119693e+00, 5.343853465909301e+00, 2.702947876599122e+00, -5.630648085837302e+00, 1.193882076389876e-01, 3.779767599610939e+00, -1.272232357794287e+00, -1.749826750933352e+00, 1.115712801512132e+00, 5.543234357850793e-01, -6.071871801557336e-01, -9.960689541071586e-02, 2.430395362921445e-01, -6.286886193265678e-03, -7.494048833905255e-02, 1.166351111896802e-02, 1.752280059736222e-02, -4.375057367724366e-03, -2.820066991262138e-03, 8.900506072677671e-04, 2.337204059585929e-04, -8.357502520084380e-05};

		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22],cdiag[23],cdiag[24],cdiag[25],cdiag[26],cdiag[27],cdiag[28],cdiag[29],cdiag[30],cdiag[31],cdiag[32]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24],coffd[25],coffd[26],coffd[27],coffd[28],coffd[29],coffd[30]);
      }
      Vec six = 6.0e0;
      Vec Fdiag = -fdiag*Rinv3;
      Vec Foffd = six*foffd*Rinv5;
      for (long i = 0; i < nd_; i++) {
        Vec rdotq = dX[0]*Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1] + dX[2]*Vslet[s*nd_*COORD_DIM+i*COORD_DIM+2];
        Vec rdotn = dX[0]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1] + dX[2]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+2];
        Vec qdotn = Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0] + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1] + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+2]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+2]; 

        Vec pl = Foffd*rdotq*rdotn;
        Vtrg[i][0] += pl*dX[0] + Fdiag*(dX[0]*qdotn + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0]*rdotn + Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0]*rdotq);
		Vtrg[i][1] += pl*dX[1] + Fdiag*(dX[1]*qdotn + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1]*rdotn + Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1]*rdotq);
		Vtrg[i][2] += pl*dX[2] + Fdiag*(dX[2]*qdotn + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+2]*rdotn + Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+2]*rdotq);
      }
    }
    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0].StoreAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1].StoreAligned(&Vt[1*nd_+i][t]);
      Vtrg[i][2].StoreAligned(&Vt[2*nd_+i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_*COORD_DIM; j++) {
      pot[i*nd_*COORD_DIM+j] += Vt[j][i];
    }
  }
}
#endif




#if defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void st3d_local_kernel_directdp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* strslet, const Real* strsvec,const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_*COORD_DIM, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_*COORD_DIM, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    if (COORD_DIM>2) {
      std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
    }
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];

  // load strslet and strsvec
  sctl::Matrix<Real> Vsl_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)strslet, nd_*Nsrc),false);
  sctl::Matrix<Real> Vsv_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)strsvec, nd_*Nsrc),false);  
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vslet(Nsrc*nd_*COORD_DIM);
  sctl::Vector<Vec> Vsvec(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vsl_[s][0*nd_+i]);
      Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vsl_[s][1*nd_+i]);
      Vslet[s*nd_*COORD_DIM+i*COORD_DIM+2] = Vec::Load1(&Vsl_[s][2*nd_+i]);
    }
  }
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vsv_[s][0*nd_+i]);
      Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vsv_[s][1*nd_+i]);
      Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+2] = Vec::Load1(&Vsv_[s][2*nd_+i]);
    }
  }
  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);

  //#pragma omp parallel for schedule(static)
  // TODO: reinit to sctl::Vector is over limit
  //  #define NSRC_LIMIT 1000
  // work array memory stores compressed R2
  Real dX_work[COORD_DIM][NSRC_LIMIT*8];
  Real R2_work[NSRC_LIMIT*8];
  //sctl::StaticArray<Real, NSRC_LIMIT*8> R2_work_buff;
  // why converting to sctl::Vector is a bit slow...
  //sctl::Vector<Real> R2_work(NSRC_LIMIT*8, R2_work_buff, false);
  // work array memory stores compressed polynomial evals
  Real Fdiag_work[NSRC_LIMIT*8];
  Real Foffd_work[NSRC_LIMIT*8];
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
      Vtrg[i][0] = Vec::LoadAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1] = Vec::LoadAligned(&Vt[1*nd_+i][t]);
      Vtrg[i][2] = Vec::LoadAligned(&Vt[2*nd_+i][t]);
    }

    //int total_r = Nsrc*VecLen;
    int valid_r = 0;
    int source_cnt = 0;
    // store compressed R2 to explore 4/3 pi over 27 sparsity
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = zero_tmp;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }

      // store sparse data
      Mask_work[source_cnt] = (R2 >= d2min_vec) & (R2 <= d2max_vec);
      int valid_cnt =  mask_popcnt(Mask_work[source_cnt]);
      mask_compress_store(Mask_work[source_cnt], dX[0], &dX_work[0][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], dX[1], &dX_work[1][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], dX[2], &dX_work[2][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], R2, &R2_work[0]+valid_r);
      start_ind[source_cnt] = valid_r;
      source_ind[source_cnt] = s;
      source_cnt += (valid_cnt>0);
      valid_r += valid_cnt;
    }

    // evaluate polynomial on compressed R2 and store in Pval
    for (sctl::Long s = 0; s < valid_r; s += VecLen) {
      Vec R2 = Vec::LoadAligned(&R2_work[0]+s);
      // evaluate the PSWF local kernel
      Vec fdiag,foffd;
      Vec Rinv = sctl::approx_rsqrt<digits>(R2, (R2 >= d2min_vec) & (R2 <= d2max_vec));
      Vec Rinv3 = Rinv*Rinv*Rinv;
	  Vec Rinv5 = Rinv3*Rinv*Rinv;
      Vec xtmp = FMA(R2,Rinv,cen_vec)*rsc_vec;
      if (digits<=3) {
		constexpr Real cdiag[10] = {-4.909552162703472e-01, -2.638326207042311e-01, 1.426957484649616e+00, 5.133112420191036e-01, -1.692721168557828e+00, -3.331428832347214e-01, 9.764073959118829e-01, 7.923558379145755e-02, -2.299454085259451e-01, -5.708623886772283e-03};
		constexpr Real coffd[10] = {-3.875806562298824e-01, 4.031306201463891e-01, 3.626294455986704e-01, -2.307948583865092e-01, -3.959286768274001e-01, 9.858855313084947e-02, 2.196925355210589e-01, -3.006434523801740e-02, -5.244641630714497e-02, 5.535941142357893e-03};
		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9]);
      } else if (digits <= 6) {
		constexpr Real cdiag[18] = {-4.223999157922261e-01, 6.639496106000218e-01, 1.362863153121536e+00, -2.738405892193764e+00, -1.352748482947856e+00, 4.607335078399908e+00, -8.963343367930998e-02, -4.158601207960919e+00, 1.166122847649966e+00, 2.274176040024465e+00, -1.031865672685651e+00, -8.228724052420864e-01, 4.845591499488178e-01, 2.060153768358927e-01, -1.351013559403134e-01, -3.479488210553128e-02, 1.816899726122462e-02, 3.163540467786774e-03};
		constexpr Real coffd[16] = {-2.415969568205024e-01, 6.437162555004452e-01, -8.886684055725748e-02, -1.004963550079139e+00, 3.023900125189125e-01, 1.203207169828373e+00, -5.173536934661076e-01, -9.517824419372273e-01, 5.159051120129571e-01, 5.006553080725374e-01, -3.140960612583585e-01, -1.762122302038716e-01, 1.120404160726096e-01, 4.001650697669312e-02, -1.843399330899232e-02, -4.648881387204539e-03};

		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15]);
      } else if (digits <= 9) {
		constexpr Real cdiag[25] = {-2.692062485564092e-01, 9.919752014657248e-01, -2.475276651563518e-01, -3.483047290967188e+00, 4.110450253556166e+00, 3.902288269852157e+00, -9.281592572398081e+00, 1.402859806439762e-01, 1.005854710632259e+01, -4.433065768464961e+00, -6.250745674629706e+00, 5.074721901153145e+00, 2.262821078806491e+00, -3.261653666605643e+00, -3.236650597906456e-01, 1.443628453346002e+00, -1.363581893372792e-01, -4.733239006426434e-01, 1.088198030976763e-01, 1.163379837058996e-01, -3.899147580748667e-02, -1.996054535142191e-02, 8.304694659040938e-03, 1.813315378685804e-03, -8.561172730301156e-04};
		constexpr Real coffd[24] = {-1.328623832934818e-01, 5.998646490238386e-01, -7.130999555454819e-01, -6.581127232402704e-01, 1.863735047664648e+00, 8.380501392006251e-02, -2.730121970220487e+00, 1.060918785747619e+00, 2.448001823516434e+00, -1.790934790189596e+00, -1.358784546977841e+00, 1.600552255725127e+00, 4.163703074190580e-01, -9.480185933976362e-01, -7.857097865269225e-03, 4.061524503251803e-01, -5.947496832482102e-02, -1.312977911169639e-01, 3.145743836336123e-02, 3.205149253369654e-02, -8.373574144115992e-03, -5.476849877595668e-03, 1.009857050121354e-03, 4.960781625413800e-04};

	fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22],cdiag[23],cdiag[24]);
	foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23]);
      } else if (digits <= 12) {
		constexpr Real cdiag[33] = {-1.360683826314615e-01, 8.302066635143668e-01, -1.514928718394161e+00, -9.381617318207349e-01, 6.813719028589369e+00, -5.988863853057117e+00, -8.472508766732407e+00, 1.809741443831257e+01, -1.273523386015356e+00, -2.203611105917701e+01, 1.439588582795162e+01, 1.352189057132570e+01, -1.847340430858263e+01, -2.444667103472067e+00, 1.341798582270320e+01, -3.074993686699973e+00, -6.521211003969547e+00, 3.398385460000120e+00, 2.180822019664342e+00, -1.978605521170600e+00, -4.466586996459809e-01, 8.228831786637951e-01, 4.803782852447806e-03, -2.660309603180980e-01, 3.972099950624761e-02, 6.873638809031379e-02, -1.933391187598765e-02, -1.395009222278865e-02, 5.664769109722911e-03, 2.024357518311426e-03, -1.064954429644160e-03, -1.570495227838364e-04, 9.988186457121628e-05};
		  constexpr Real coffd[31] = {-6.115079449085273e-02, 4.128039371348927e-01, -9.881137625293366e-01, 5.143473444562541e-01, 1.885479167316791e+00, -3.050423360463870e+00, -9.475794240844918e-01, 5.634324267214952e+00, -2.338305998487992e+00, -5.502717038119693e+00, 5.343853465909301e+00, 2.702947876599122e+00, -5.630648085837302e+00, 1.193882076389876e-01, 3.779767599610939e+00, -1.272232357794287e+00, -1.749826750933352e+00, 1.115712801512132e+00, 5.543234357850793e-01, -6.071871801557336e-01, -9.960689541071586e-02, 2.430395362921445e-01, -6.286886193265678e-03, -7.494048833905255e-02, 1.166351111896802e-02, 1.752280059736222e-02, -4.375057367724366e-03, -2.820066991262138e-03, 8.900506072677671e-04, 2.337204059585929e-04, -8.357502520084380e-05};

		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22],cdiag[23],cdiag[24],cdiag[25],cdiag[26],cdiag[27],cdiag[28],cdiag[29],cdiag[30],cdiag[31],cdiag[32]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24],coffd[25],coffd[26],coffd[27],coffd[28],coffd[29],coffd[30]);
      }
      Vec six = 6.0e0;
      Vec Fdiag = -fdiag*Rinv3;
      Vec Foffd = six*foffd*Rinv5;
      Fdiag.StoreAligned(Fdiag_work+s);
      Foffd.StoreAligned(Foffd_work+s);
    }

    // expand compressed Fdiag_work and Foffd_work, then multiply by charge to accumulate
    for (sctl::Long s_ind = 0; s_ind < source_cnt; s_ind++) {
      int s = source_ind[s_ind];
      int start = start_ind[s_ind];
      Vec Fdiag = mask_expand_load(Mask_work[s_ind], zero_tmp, Fdiag_work+start);
      Vec Foffd = mask_expand_load(Mask_work[s_ind], zero_tmp, Foffd_work+start);
      Vec dX[3];
      dX[0] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[0]+start);
      dX[1] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[1]+start);
      dX[2] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[2]+start);

      for (long i = 0; i < nd_; i++) {
        Vec rdotq = dX[0]*Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1] + dX[2]*Vslet[s*nd_*COORD_DIM+i*COORD_DIM+2];
        Vec rdotn = dX[0]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1] + dX[2]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+2];
        Vec qdotn = Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0] + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1] + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+2]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+2]; 

        Vec pl = Foffd*rdotq*rdotn;
        Vtrg[i][0] += pl*dX[0] + Fdiag*(dX[0]*qdotn + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0]*rdotn + Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0]*rdotq);
		Vtrg[i][1] += pl*dX[1] + Fdiag*(dX[1]*qdotn + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1]*rdotn + Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1]*rdotq);
		Vtrg[i][2] += pl*dX[2] + Fdiag*(dX[2]*qdotn + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+2]*rdotn + Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+2]*rdotq);
      }
    }

    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0].StoreAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1].StoreAligned(&Vt[1*nd_+i][t]);
      Vtrg[i][2].StoreAligned(&Vt[2*nd_+i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_*COORD_DIM; j++) {
      pot[i*nd_*COORD_DIM+j] += Vt[j][i];
    }
  }
}
#endif




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void st3d_local_kernel_directdp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* strslet, const Real* strsvec, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  if (digits[0] <= 3) st3d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,3,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,strslet,strsvec,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 6) st3d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,6,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,strslet,strsvec,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 9) st3d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,9,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,strslet,strsvec,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 12) st3d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,12,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,strslet,strsvec,xtarg,ytarg,ztarg,nt,pot);
  else st3d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,-1,3>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,strslet,strsvec,xtarg,ytarg,ztarg,nt,pot);
}


/* 2D Stresslet local kernel charge to potential */ 
#if !defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void st2d_local_kernel_directdp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* strslet, const Real* strsvec,const Real* xtarg, const Real* ytarg, const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_*COORD_DIM, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_*COORD_DIM, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];
  Vec bsizeinv2_vec = bsizeinv_vec*bsizeinv_vec;

  // load strslet and strsvec 
  sctl::Matrix<Real> Vsl_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)strslet, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vslet(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vsl_[s][0*nd_+i]);
      Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vsl_[s][1*nd_+i]);
    }
  }
  sctl::Matrix<Real> Vsv_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)strsvec, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vsvec(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vsv_[s][0*nd_+i]);
      Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vsv_[s][1*nd_+i]);
    }
  }
  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);
  #pragma omp parallel for schedule(static)
  for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
    Vec Xtrg[COORD_DIM];
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
    }
    // load potential
    Vec Vtrg[nd_][COORD_DIM];
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0] = Vec::LoadAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1] = Vec::LoadAligned(&Vt[1*nd_+i][t]);
    }
	
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = Vec::Zero();
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }
      // evaluate the PSWF local kernel
      Vec fdiag,foffd;
      Vec one = 1.0e0;
      Vec R2inv = select((R2 >= d2min_vec) & (R2 <= d2max_vec), one/R2, Vec::Zero());	  
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      if (digits<=3) {
		constexpr Real cdiag[9] = {-1.816189882130478e-01, 3.842027435699696e-01, -1.939109312588435e-01, -1.716118942514620e-01, 2.873134185691492e-01, -1.874594193993935e-01, 7.941576483320495e-02, -2.746385545682173e-02, 6.463142588475671e-03};
		constexpr Real coffd[8] = {-2.321311056270709e-01, 5.658165638396627e-01, -4.754260725108626e-01, 8.168572884381946e-02, 1.332774714714928e-01, -1.274312822936841e-01, 7.178067915417725e-02, -2.252264191473135e-02};
		
		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7]);
      } else if (digits <= 6) {
		constexpr Real cdiag[14] = {-3.651714691751001e-02, 1.712921814324308e-01, -3.364542389921445e-01, 3.316949847849875e-01, -1.005829690260044e-01, -1.701956931755001e-01, 2.911627263819399e-01, -2.525685651033986e-01, 1.536479028814323e-01, -7.156531970086269e-02, 2.630614118984020e-02, -8.147113414077967e-03, 2.432826549753706e-03, -5.152202849797587e-04};
		constexpr Real coffd[14] = {-4.241928963970104e-02, 2.078093408212323e-01, -4.403588221037121e-01, 5.131157706605827e-01, -3.195737014137147e-01, 2.511628010918487e-02, 1.567770782158370e-01, -1.789851198247047e-01, 1.208095140190344e-01, -5.943166666775834e-02, 2.257750458949382e-02, -7.161681761024496e-03, 2.182899112222358e-03, -4.677291225405612e-04};

		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13]);
      } else if (digits <= 9) {
		constexpr Real cdiag[20] = {-6.049415849462836e-03, 4.405313434892306e-02, -1.456728721643744e-01, 2.856740997471844e-01, -3.573867768060327e-01, 2.665844513443985e-01, -4.341039188262985e-02, -1.792670738082169e-01, 2.904159363750097e-01, -2.775365663752623e-01, 1.983541077350943e-01, -1.142088781772825e-01, 5.493687700913488e-02, -2.258906159192373e-02, 8.081033878698825e-03, -2.529748884141591e-03, 6.847719911099448e-04, -1.711325594366820e-04, 4.672452856180342e-05, -9.229225679696638e-06};
		constexpr Real coffd[19] = {-6.738442962491431e-03, 5.010254956812221e-02, -1.707241472964444e-01, 3.509326117058103e-01, -4.777491563636959e-01, 4.343510447940209e-01, -2.276471945170291e-01, -1.514300539173696e-02, 1.692204890573921e-01, -2.020987899068024e-01, 1.581927595341201e-01, -9.567202283927670e-02, 4.751651269561565e-02, -2.005576260889796e-02, 7.257615707082721e-03, -2.217351064146555e-03, 6.279933284361051e-04, -1.992794130956099e-04, 4.356562476459658e-05};
		
		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18]);
      } else if (digits <= 12) {
		constexpr Real cdiag[26] = {-6.952457984474164e-04, 7.122192588827236e-03, -3.427273252636015e-02, 1.026229752368365e-01, -2.128203372116609e-01, 3.197756910515241e-01, -3.479406059706922e-01, 2.508641818652326e-01, -5.788620705375092e-02, -1.435137429135196e-01, 2.715949737423533e-01, -2.988387008678568e-01, 2.509715852539839e-01, -1.737912987560399e-01, 1.028205025439143e-01, -5.305857474896497e-02, 2.421410546007963e-02, -9.870473424179238e-03, 3.620654361748521e-03, -1.204469434908262e-03, 3.668941521439705e-04, -1.015726305624298e-04, 2.473045492479286e-05, -5.877766625417282e-06, 1.682589697793447e-06, -3.302018285467057e-07};
		constexpr Real coffd[25] = {-7.544547489348741e-04, 7.817438387701992e-03, -3.818145171931508e-02, 1.166530321632586e-01, -2.489986237838498e-01, 3.912823891168687e-01, -4.608254676889533e-01, 3.973284161123167e-01, -2.174004592931065e-01, 4.708660392780617e-03, 1.525463726784126e-01, -2.153039240964150e-01, 1.993003657554928e-01, -1.453978004685617e-01, 8.887161443583116e-02, -4.689972980757773e-02, 2.175029227703364e-02, -8.967982783929074e-03, 3.325446073063176e-03, -1.123186507811625e-03, 3.413650852053252e-04, -8.981020891951627e-05, 2.343420237593818e-05, -7.502301741624253e-06, 1.566724677104502e-06};

		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22],cdiag[23],cdiag[24],cdiag[25]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24]);
      }
      Vec two = 2.0e0;
      Vec Fdiag = -fdiag*R2inv;
      Vec Foffd = two*foffd*R2inv*R2inv;

      for (long i = 0; i < nd_; i++) {
		Vec rdotq = dX[0]*Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vec rdotn = dX[0]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vec qdotn = Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0] + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1];
		
        Vec pl = Foffd*rdotq*rdotn;
        Vtrg[i][0] += pl*dX[0] + Fdiag*(dX[0]*qdotn + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0]*rdotn + Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0]*rdotq);
        Vtrg[i][1] += pl*dX[1] + Fdiag*(dX[1]*qdotn + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1]*rdotn + Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1]*rdotq);
      }
    }
    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0].StoreAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1].StoreAligned(&Vt[1*nd_+i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_*COORD_DIM; j++) {
      pot[i*nd_*COORD_DIM+j] += Vt[j][i];
    }
  }
}
#endif




#if defined(__AVX512DQ__)
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim> void st2d_local_kernel_directdp_vec_cpp_helper(const int32_t* nd, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* strslet, const Real* strsvec, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100*COORD_DIM> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_*COORD_DIM, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_*COORD_DIM, Ntrg_);
  }
  { // Set Xs, Vs, Xt, Vt
    std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
    std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
    Vt = 0;
  }

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec d2min_vec = d2min[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];
  Vec bsizeinv_vec = bsizeinv[0];
  Vec bsizeinv2_vec = bsizeinv_vec*bsizeinv_vec;

  // load strslet and strsvec 
  sctl::Matrix<Real> Vsl_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)strslet, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vslet(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vsl_[s][0*nd_+i]);
      Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vsl_[s][1*nd_+i]);
    }
  }
  sctl::Matrix<Real> Vsv_(Nsrc, nd_*COORD_DIM,sctl::Ptr2Itr<Real>((Real*)strsvec, nd_*Nsrc),false);
  //Vec Vsrc[Nsrc][nd_][COORD_DIM];
  sctl::Vector<Vec> Vsvec(Nsrc*nd_*COORD_DIM);
  for (sctl::Long s = 0; s < Nsrc; s++) {
    for (long i = 0; i < nd_; i++) {
      Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0] = Vec::Load1(&Vsv_[s][0*nd_+i]);
      Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1] = Vec::Load1(&Vsv_[s][1*nd_+i]);
    }
  }

  
  // load source
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM, sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc),false);

  //#pragma omp parallel for schedule(static)
  // TODO: reinit to sctl::Vector is over limit
  // #define NSRC_LIMIT 1000
  // work array memory stores compressed R2
  Real dX_work[COORD_DIM][NSRC_LIMIT*8];
  Real R2_work[NSRC_LIMIT*8];
  //sctl::StaticArray<Real, NSRC_LIMIT*8> R2_work_buff;
  // why converting to sctl::Vector is a bit slow...
  //sctl::Vector<Real> R2_work(NSRC_LIMIT*8, R2_work_buff, false);
  // work array memory stores compressed polynomial evals
  Real Fdiag_work[NSRC_LIMIT*8];
  Real Foffd_work[NSRC_LIMIT*8];
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
      Vtrg[i][0] = Vec::LoadAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1] = Vec::LoadAligned(&Vt[1*nd_+i][t]);
    }

    //int total_r = Nsrc*VecLen;
    int valid_r = 0;
    int source_cnt = 0;
    // store compressed R2 to explore 4/3 pi over 27 sparsity
    for (sctl::Long s = 0; s < Nsrc; s++) {
      Vec dX[COORD_DIM], R2 = zero_tmp;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        dX[k] = Xtrg[k] - Vec::Load1(&Xs_[s][k]);
        R2 += dX[k]*dX[k];
      }

      // store sparse data
      Mask_work[source_cnt] = (R2 >= d2min_vec) & (R2 <= d2max_vec);
      int valid_cnt =  mask_popcnt(Mask_work[source_cnt]);
      mask_compress_store(Mask_work[source_cnt], dX[0], &dX_work[0][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], dX[1], &dX_work[1][0]+valid_r);
      mask_compress_store(Mask_work[source_cnt], R2, &R2_work[0]+valid_r);
      start_ind[source_cnt] = valid_r;
      source_ind[source_cnt] = s;
      source_cnt += (valid_cnt>0);
      valid_r += valid_cnt;
    }

    // evaluate polynomial on compressed R2 and store in Pval
    for (sctl::Long s = 0; s < valid_r; s += VecLen) {
      Vec R2 = Vec::LoadAligned(&R2_work[0]+s);
      // evaluate the PSWF local kernel
      Vec fdiag,foffd;
      Vec one = 1.0e0;
      Vec R2inv = select((R2 >= d2min_vec) & (R2 <= d2max_vec), one/R2, Vec::Zero());	  
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
      if (digits<=3) {
		constexpr Real cdiag[9] = {-1.816189882130478e-01, 3.842027435699696e-01, -1.939109312588435e-01, -1.716118942514620e-01, 2.873134185691492e-01, -1.874594193993935e-01, 7.941576483320495e-02, -2.746385545682173e-02, 6.463142588475671e-03};
		constexpr Real coffd[8] = {-2.321311056270709e-01, 5.658165638396627e-01, -4.754260725108626e-01, 8.168572884381946e-02, 1.332774714714928e-01, -1.274312822936841e-01, 7.178067915417725e-02, -2.252264191473135e-02};
		
		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7]);
      } else if (digits <= 6) {
		constexpr Real cdiag[14] = {-3.651714691751001e-02, 1.712921814324308e-01, -3.364542389921445e-01, 3.316949847849875e-01, -1.005829690260044e-01, -1.701956931755001e-01, 2.911627263819399e-01, -2.525685651033986e-01, 1.536479028814323e-01, -7.156531970086269e-02, 2.630614118984020e-02, -8.147113414077967e-03, 2.432826549753706e-03, -5.152202849797587e-04};
		constexpr Real coffd[14] = {-4.241928963970104e-02, 2.078093408212323e-01, -4.403588221037121e-01, 5.131157706605827e-01, -3.195737014137147e-01, 2.511628010918487e-02, 1.567770782158370e-01, -1.789851198247047e-01, 1.208095140190344e-01, -5.943166666775834e-02, 2.257750458949382e-02, -7.161681761024496e-03, 2.182899112222358e-03, -4.677291225405612e-04};

		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13]);
      } else if (digits <= 9) {
		constexpr Real cdiag[20] = {-6.049415849462836e-03, 4.405313434892306e-02, -1.456728721643744e-01, 2.856740997471844e-01, -3.573867768060327e-01, 2.665844513443985e-01, -4.341039188262985e-02, -1.792670738082169e-01, 2.904159363750097e-01, -2.775365663752623e-01, 1.983541077350943e-01, -1.142088781772825e-01, 5.493687700913488e-02, -2.258906159192373e-02, 8.081033878698825e-03, -2.529748884141591e-03, 6.847719911099448e-04, -1.711325594366820e-04, 4.672452856180342e-05, -9.229225679696638e-06};
		constexpr Real coffd[19] = {-6.738442962491431e-03, 5.010254956812221e-02, -1.707241472964444e-01, 3.509326117058103e-01, -4.777491563636959e-01, 4.343510447940209e-01, -2.276471945170291e-01, -1.514300539173696e-02, 1.692204890573921e-01, -2.020987899068024e-01, 1.581927595341201e-01, -9.567202283927670e-02, 4.751651269561565e-02, -2.005576260889796e-02, 7.257615707082721e-03, -2.217351064146555e-03, 6.279933284361051e-04, -1.992794130956099e-04, 4.356562476459658e-05};
		
		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18]);
      } else if (digits <= 12) {
		constexpr Real cdiag[26] = {-6.952457984474164e-04, 7.122192588827236e-03, -3.427273252636015e-02, 1.026229752368365e-01, -2.128203372116609e-01, 3.197756910515241e-01, -3.479406059706922e-01, 2.508641818652326e-01, -5.788620705375092e-02, -1.435137429135196e-01, 2.715949737423533e-01, -2.988387008678568e-01, 2.509715852539839e-01, -1.737912987560399e-01, 1.028205025439143e-01, -5.305857474896497e-02, 2.421410546007963e-02, -9.870473424179238e-03, 3.620654361748521e-03, -1.204469434908262e-03, 3.668941521439705e-04, -1.015726305624298e-04, 2.473045492479286e-05, -5.877766625417282e-06, 1.682589697793447e-06, -3.302018285467057e-07};
		constexpr Real coffd[25] = {-7.544547489348741e-04, 7.817438387701992e-03, -3.818145171931508e-02, 1.166530321632586e-01, -2.489986237838498e-01, 3.912823891168687e-01, -4.608254676889533e-01, 3.973284161123167e-01, -2.174004592931065e-01, 4.708660392780617e-03, 1.525463726784126e-01, -2.153039240964150e-01, 1.993003657554928e-01, -1.453978004685617e-01, 8.887161443583116e-02, -4.689972980757773e-02, 2.175029227703364e-02, -8.967982783929074e-03, 3.325446073063176e-03, -1.123186507811625e-03, 3.413650852053252e-04, -8.981020891951627e-05, 2.343420237593818e-05, -7.502301741624253e-06, 1.566724677104502e-06};

		fdiag = EvalPolynomial(xtmp.get(),cdiag[0],cdiag[1],cdiag[2],cdiag[3],cdiag[4],cdiag[5],cdiag[6],cdiag[7],cdiag[8],cdiag[9],cdiag[10],cdiag[11],cdiag[12],cdiag[13],cdiag[14],cdiag[15],cdiag[16],cdiag[17],cdiag[18],cdiag[19],cdiag[20],cdiag[21],cdiag[22],cdiag[23],cdiag[24],cdiag[25]);
		foffd = EvalPolynomial(xtmp.get(),coffd[0],coffd[1],coffd[2],coffd[3],coffd[4],coffd[5],coffd[6],coffd[7],coffd[8],coffd[9],coffd[10],coffd[11],coffd[12],coffd[13],coffd[14],coffd[15],coffd[16],coffd[17],coffd[18],coffd[19],coffd[20],coffd[21],coffd[22],coffd[23],coffd[24]);
      }
      Vec two = 2.0e0;
      Vec Fdiag = -fdiag*R2inv;
      Vec Foffd = two*foffd*R2inv*R2inv;
      Fdiag.StoreAligned(Fdiag_work+s);
      Foffd.StoreAligned(Foffd_work+s);
    }

    // expand compressed Fdiag_work and Foffd_work, then multiply by charge to accumulate
    for (sctl::Long s_ind = 0; s_ind < source_cnt; s_ind++) {
      int s = source_ind[s_ind];
      int start = start_ind[s_ind];
      Vec Fdiag = mask_expand_load(Mask_work[s_ind], zero_tmp, Fdiag_work+start);
      Vec Foffd = mask_expand_load(Mask_work[s_ind], zero_tmp, Foffd_work+start);
      Vec dX[2];
      dX[0] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[0]+start);
      dX[1] = mask_expand_load(Mask_work[s_ind], zero_tmp, dX_work[1]+start);

      for (long i = 0; i < nd_; i++) {
		Vec rdotq = dX[0]*Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vec rdotn = dX[0]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0] + dX[1]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1];
        Vec qdotn = Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0] + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1]*Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1];

        Vec pl = Foffd*rdotq*rdotn;
        Vtrg[i][0] += pl*dX[0] + Fdiag*(dX[0]*qdotn + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+0]*rdotn + Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+0]*rdotq);
        Vtrg[i][1] += pl*dX[1] + Fdiag*(dX[1]*qdotn + Vslet[s*nd_*COORD_DIM+i*COORD_DIM+1]*rdotn + Vsvec[s*nd_*COORD_DIM+i*COORD_DIM+1]*rdotq);
      }
    }

    // store potential
    for (long i = 0; i < nd_; i++) {
      Vtrg[i][0].StoreAligned(&Vt[0*nd_+i][t]);
      Vtrg[i][1].StoreAligned(&Vt[1*nd_+i][t]);
    }
  }

  for (long i = 0; i < Ntrg; i++) {
    for (long j=0; j < nd_*COORD_DIM; j++) {
      pot[i*nd_*COORD_DIM+j] += Vt[j][i];
    }
  }
}
#endif




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void st2d_local_kernel_directdp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits, const Real* rsc, const Real* cen, const Real* bsizeinv, const Real* d2min, const Real* d2max, const Real* sources,const int32_t* ns, const Real* strslet, const Real* strsvec, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot) {
  if (digits[0] <= 3) st2d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,3,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,strslet,strsvec,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 6) st2d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,6,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,strslet,strsvec,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 9) st2d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,9,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,strslet,strsvec,xtarg,ytarg,ztarg,nt,pot);
  else if (digits[0] <= 12) st2d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,12,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,strslet,strsvec,xtarg,ytarg,ztarg,nt,pot);
  else st2d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,-1,2>(nd,rsc,cen,bsizeinv,d2min,d2max,sources,ns,strslet,strsvec,xtarg,ytarg,ztarg,nt,pot);
}



#endif //_VEC_KERNELS_HPP_
