#ifndef _VEC_KERNELS_HPP_
#define _VEC_KERNELS_HPP_

#define NDEBUG
#include <sctl.hpp>
#include <sctllog.h>

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
template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(), sctl::Integer digits=-1, sctl::Integer ndim>void l3d_local_kernel_directcp_vec_cpp_helper(const int32_t* nd, const Real* rsc,const Real* cen, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot, const Real* thresh) {
  static constexpr sctl::Integer COORD_DIM = ndim; //ndim[0];
  constexpr sctl::Long nd_ = 1; //nd[0];
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;


  sctl::StaticArray<Real,400*COORD_DIM> buff0;
  sctl::StaticArray<Real,400*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 400) {
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
		//		std::cout << Xtrg[k] <<"\n"<< Xs_[s][k]<<"\n" << dX[k]<< "\n";
        R2 += dX[k]*dX[k];
      }
	  

      Vec Rinv = sctl::approx_rsqrt<digits>(R2, (R2 > thresh2) & (R2 < d2max_vec));
	  // std::cout << Rinv <<"\n";
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




template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()> void l3d_local_kernel_directcp_vec_cpp(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const Real* rsc,const Real* cen, const Real* d2max, const Real* sources,const int32_t* ns, const Real* charge, const Real* xtarg,const Real* ytarg,const Real* ztarg, const int32_t* nt, Real* pot, const Real* thresh) {
  if (ndim[0] == 3) {
	if (digits[0] <= 3) l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,3>(nd,rsc,cen,
																			d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
	else if (digits[0] <= 6) l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
	else if (digits[0] <= 9) l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
	else if (digits[0] <= 12) l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);  
	else l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
  }
  if (ndim[0] == 2) {
	if (digits[0] <= 3) l3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,2>(nd,rsc,cen,
																			d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
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

  sctl::StaticArray<Real,400*COORD_DIM> buff0;
  sctl::StaticArray<Real,400*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 400) {
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

  sctl::StaticArray<Real,400*COORD_DIM> buff0;
  sctl::StaticArray<Real,400*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 400) {
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
  #define NSRC_LIMIT 1000
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


  sctl::StaticArray<Real,400*COORD_DIM> buff0;
  sctl::StaticArray<Real,400*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 400) {
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
		//		std::cout << Xtrg[k] <<"\n"<< Xs_[s][k]<<"\n" << dX[k]<< "\n";
        R2 += dX[k]*dX[k];
      }
	  

      Vec Rinv = sctl::approx_rsqrt<digits>(R2, (R2 > thresh2) & (R2 < d2max_vec));
	  Vec R2inv=Rinv*Rinv;
	  // std::cout << Rinv <<"\n";
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
	if (digits[0] <= 3) sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,3>(nd,rsc,cen,
																			d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
	else if (digits[0] <= 6) sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,6,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
	else if (digits[0] <= 9) sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,9,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
	else if (digits[0] <= 12) sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,12,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);  
	else sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,-1,3>(nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
  }
  if (ndim[0] == 2) {
	if (digits[0] <= 3) sl3d_local_kernel_directcp_vec_cpp_helper<Real,MaxVecLen,3,2>(nd,rsc,cen,
																			d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,pot,thresh);
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

  sctl::StaticArray<Real,400*COORD_DIM> buff0;
  sctl::StaticArray<Real,400*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 400) {
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


  sctl::StaticArray<Real,400*COORD_DIM> buff0;
  sctl::StaticArray<Real,400*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 400) {
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
		//		std::cout << Xtrg[k] <<"\n"<< Xs_[s][k]<<"\n" << dX[k]<< "\n";
        R2 += dX[k]*dX[k];
      }
	  
      Vec xtmp = FMA(R2,rsc_vec,cen_vec);
	  Vec R2sc=R2*bsizeinv2_vec;
	  
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

  sctl::StaticArray<Real,400*COORD_DIM> buff0;
  sctl::StaticArray<Real,400*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 400) {
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


  sctl::StaticArray<Real,400*COORD_DIM> buff0;
  sctl::StaticArray<Real,400*1> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 400) {
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



#endif //_VEC_KERNELS_HPP_
