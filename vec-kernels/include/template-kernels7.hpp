#ifndef _VEC_KERNELS_HPP_
#define _VEC_KERNELS_HPP_

#define NDEBUG
#define NSRC_LIMIT 3000
#include <sctl.hpp>
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
      Vec Fdiag = select((R2 >= d2min_vec) & (R2 <= d2max_vec), -0.25e0*log(R2sc)-fdiag, Vec::Zero());	  
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
      Vec Fdiag = select((R2 >= d2min_vec) & (R2 <= d2max_vec), -0.25e0*log(R2sc)-fdiag, Vec::Zero());	  
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
