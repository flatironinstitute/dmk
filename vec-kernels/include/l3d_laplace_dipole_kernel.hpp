#ifndef _L3D_LAPLACE_DIPOLE_KERNEL_HPP_
#define _L3D_LAPLACE_DIPOLE_KERNEL_HPP_

template <class Vec, class Real, size_t N>
static inline void l3d_eval_poly_der(const Vec& x,
    const Real (&coefs)[N], Vec& p, Vec& dp) {
  p = Vec(coefs[N-1]);
  dp = Vec::Zero();
  for (int i = (int)N-2; i >= 0; i--) {
    dp = FMA(dp, x, p);
    p = FMA(p, x, Vec(coefs[i]));
  }
}

template <class Vec, class Real>
static inline Vec l3d_eval_poly_dyn(const Vec& x, const Real* coefs,
    int ncoefs) {
  Vec p = Vec(coefs[ncoefs-1]);
  for (int i = ncoefs-2; i >= 0; i--) {
    p = FMA(p, x, Vec(coefs[i]));
  }
  return p;
}

template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(),
    sctl::Integer digits=-1, sctl::Integer ndim>
void l3d_local_kernel_directdp_vec_cpp_helper(const int32_t* nd,
    const Real* rsc, const Real* cen, const Real* d2max,
    const Real* sources, const int32_t* ns, const Real* dipvec,
    const Real* xtarg, const Real* ytarg, const Real* ztarg,
    const int32_t* nt, Real* pot, const Real* thresh) {
  static constexpr sctl::Integer COORD_DIM = ndim;
  constexpr sctl::Long nd_ = 1;
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,300> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }

  std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
  std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
  if (COORD_DIM > 2) std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
  Vt = 0;

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec thresh2 = thresh[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];

  sctl::Matrix<Real> Vs_(Nsrc, nd_*COORD_DIM,
      sctl::Ptr2Itr<Real>((Real*)dipvec, nd_*COORD_DIM*Nsrc), false);
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM,
      sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc), false);

  #pragma omp parallel for schedule(static)
  for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
    Vec Xtrg[COORD_DIM];
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
    }

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

      Vec Rinv = sctl::approx_rsqrt<digits>(R2,
          (R2 > thresh2) & (R2 < d2max_vec));
      Vec Rinv2 = Rinv*Rinv;
      Vec Rinv3 = Rinv2*Rinv;
      Vec xtmp = FMA(R2, Rinv, cen_vec)*rsc_vec;
      Vec ptmp, dptmp;

      if (digits <= 3) {
        constexpr Real coefs[7] = {1.627823522210361e-01,
        -4.553645597616490e-01, 4.171687104204163e-01,
        -7.073638602709915e-02, -8.957845614474928e-02,
        2.617986644718201e-02, 9.633427876507601e-03};
        l3d_eval_poly_der(xtmp, coefs, ptmp, dptmp);
      } else if (digits <= 6) {
        constexpr Real coefs[13] = {5.482525801351582e-02,
        -2.616592110444692e-01, 4.862652666337138e-01,
        -3.894296348642919e-01, 1.638587821812791e-02,
        1.870328434198821e-01, -8.714171086568978e-02,
        -3.927020727017803e-02, 3.728187607052319e-02,
        3.153734425831139e-03, -8.651313377285847e-03,
        1.725110090795567e-04, 1.034762385284044e-03};
        l3d_eval_poly_der(xtmp, coefs, ptmp, dptmp);
      } else if (digits <= 9) {
        constexpr Real coefs[19] = {1.835718730962269e-02,
        -1.258015846164503e-01, 3.609487248584408e-01,
        -5.314579651112283e-01, 3.447559412892380e-01,
        9.664692318551721e-02, -3.124274531849053e-01,
        1.322460720579388e-01, 9.773007866584822e-02,
        -1.021958831082768e-01, -3.812847450976566e-03,
        3.858117355875043e-02, -8.728545924521301e-03,
        -9.401196355382909e-03, 4.024549377076924e-03,
        1.512806105865091e-03, -9.576734877247042e-04,
        -1.303457547418901e-04, 1.100385844683190e-04};
        l3d_eval_poly_der(xtmp, coefs, ptmp, dptmp);
      } else {
        constexpr Real coefs[25] = {6.262472576363448e-03,
        -5.605742936112479e-02, 2.185890864792949e-01,
        -4.717350304955679e-01, 5.669680214206270e-01,
        -2.511606878849214e-01, -2.744523658778361e-01,
        4.582527599363415e-01, -1.397724810121539e-01,
        -2.131762135835757e-01, 1.995489373508990e-01,
        1.793390341864239e-02, -1.035055132403432e-01,
        3.035606831075176e-02, 3.153931762550532e-02,
        -2.033178627450288e-02, -5.406682731236552e-03,
        7.543645573618463e-03, 1.437788047407851e-05,
        -1.928370882351732e-03, 2.891658777328665e-04,
        3.332996162099811e-04, -8.397699195938912e-05,
        -3.015837377517983e-05, 9.640642701924662e-06};
        l3d_eval_poly_der(xtmp, coefs, ptmp, dptmp);
      }

      Vec dfac = ptmp*Rinv3 - dptmp*rsc_vec*Rinv2;
      for (long i = 0; i < nd_; i++) {
        Vec dotp = dX[0]*Vec::Load1(&Vs_[s][0*nd_+i]) +
            dX[1]*Vec::Load1(&Vs_[s][1*nd_+i]);
        if (COORD_DIM > 2) {
          dotp += dX[2]*Vec::Load1(&Vs_[s][2*nd_+i]);
        }
        Vtrg[i] += dotp*dfac;
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

template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()>
void l3d_local_kernel_directdp_vec_cpp(const int32_t* nd,
    const int32_t* ndim, const int32_t* digits, const Real* rsc,
    const Real* cen, const Real* d2max, const Real* sources,
    const int32_t* ns, const Real* dipvec, const Real* xtarg,
    const Real* ytarg, const Real* ztarg, const int32_t* nt,
    Real* pot, const Real* thresh) {
  if (ndim[0] == 3) {
    if (digits[0] <= 3)
      l3d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,3,3>
          (nd,rsc,cen,d2max,sources,ns,dipvec,xtarg,ytarg,ztarg,nt,
          pot,thresh);
    else if (digits[0] <= 6)
      l3d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,6,3>
          (nd,rsc,cen,d2max,sources,ns,dipvec,xtarg,ytarg,ztarg,nt,
          pot,thresh);
    else if (digits[0] <= 9)
      l3d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,9,3>
          (nd,rsc,cen,d2max,sources,ns,dipvec,xtarg,ytarg,ztarg,nt,
          pot,thresh);
    else
      l3d_local_kernel_directdp_vec_cpp_helper<Real,MaxVecLen,12,3>
          (nd,rsc,cen,d2max,sources,ns,dipvec,xtarg,ytarg,ztarg,nt,
          pot,thresh);
  }
}

template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(),
    sctl::Integer digits=-1, sctl::Integer ndim>
void l3d_local_kernel_directcp_coef_vec_cpp_helper(const int32_t* nd,
    const Real* rsc, const Real* cen, const Real* d2max,
    const Real* sources, const int32_t* ns, const Real* charge,
    const Real* xtarg, const Real* ytarg, const Real* ztarg,
    const int32_t* nt, const int32_t* ncoefs, const Real* coefs,
    Real* pot, const Real* thresh) {
  static constexpr sctl::Integer COORD_DIM = ndim;
  constexpr sctl::Long nd_ = 1;
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,300> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }

  std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
  std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
  if (COORD_DIM > 2) std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
  Vt = 0;

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec thresh2 = thresh[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];

  sctl::Matrix<Real> Vs_(Nsrc, nd_,
      sctl::Ptr2Itr<Real>((Real*)charge, nd_*Nsrc), false);
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM,
      sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc), false);

  #pragma omp parallel for schedule(static)
  for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
    Vec Xtrg[COORD_DIM];
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
    }

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

      Vec Rinv = sctl::approx_rsqrt<digits>(R2,
          (R2 > thresh2) & (R2 < d2max_vec));
      Vec xtmp = FMA(R2, Rinv, cen_vec)*rsc_vec;
      Vec ptmp = l3d_eval_poly_dyn<Vec,Real>(xtmp, coefs, ncoefs[0]);
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

template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()>
void l3d_local_kernel_directcp_coef_vec_cpp(const int32_t* nd,
    const int32_t* ndim, const int32_t* digits, const Real* rsc,
    const Real* cen, const Real* d2max, const Real* sources,
    const int32_t* ns, const Real* charge, const Real* xtarg,
    const Real* ytarg, const Real* ztarg, const int32_t* nt,
    const int32_t* ncoefs, const Real* coefs, Real* pot,
    const Real* thresh) {
  if (ndim[0] == 3) {
    if (digits[0] <= 3)
      l3d_local_kernel_directcp_coef_vec_cpp_helper<Real,MaxVecLen,3,3>
          (nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,
          ncoefs,coefs,pot,thresh);
    else if (digits[0] <= 6)
      l3d_local_kernel_directcp_coef_vec_cpp_helper<Real,MaxVecLen,6,3>
          (nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,
          ncoefs,coefs,pot,thresh);
    else if (digits[0] <= 9)
      l3d_local_kernel_directcp_coef_vec_cpp_helper<Real,MaxVecLen,9,3>
          (nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,
          ncoefs,coefs,pot,thresh);
    else
      l3d_local_kernel_directcp_coef_vec_cpp_helper<Real,MaxVecLen,12,3>
          (nd,rsc,cen,d2max,sources,ns,charge,xtarg,ytarg,ztarg,nt,
          ncoefs,coefs,pot,thresh);
  }
}

template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>(),
    sctl::Integer digits=-1, sctl::Integer ndim>
void l3d_local_kernel_directdp_coef_vec_cpp_helper(const int32_t* nd,
    const Real* rsc, const Real* cen, const Real* d2max,
    const Real* sources, const int32_t* ns, const Real* dipvec,
    const Real* xtarg, const Real* ytarg, const Real* ztarg,
    const int32_t* nt, const int32_t* ncoefs, const Real* coefs,
    const int32_t* ncoefsd, const Real* coefsd, Real* pot,
    const Real* thresh) {
  static constexpr sctl::Integer COORD_DIM = ndim;
  constexpr sctl::Long nd_ = 1;
  sctl::Long Nsrc = ns[0];
  sctl::Long Ntrg = nt[0];
  sctl::Long Ntrg_ = ((Ntrg+MaxVecLen-1)/MaxVecLen)*MaxVecLen;

  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,300> buff0;
  alignas(sizeof(Real)*MaxVecLen) sctl::StaticArray<Real,100> buff1;
  sctl::Matrix<Real> Xt(COORD_DIM, Ntrg_, buff0, false);
  sctl::Matrix<Real> Vt(nd_, Ntrg_, buff1, false);
  if (Ntrg_ > 100) {
    Xt.ReInit(COORD_DIM, Ntrg_);
    Vt.ReInit(nd_, Ntrg_);
  }

  std::memcpy(Xt[0], xtarg, sizeof(Real)*Ntrg);
  std::memcpy(Xt[1], ytarg, sizeof(Real)*Ntrg);
  if (COORD_DIM > 2) std::memcpy(Xt[2], ztarg, sizeof(Real)*Ntrg);
  Vt = 0;

  static constexpr sctl::Integer VecLen = MaxVecLen;
  using Vec = sctl::Vec<Real,VecLen>;
  Vec thresh2 = thresh[0];
  Vec d2max_vec = d2max[0];
  Vec rsc_vec = rsc[0];
  Vec cen_vec = cen[0];

  sctl::Matrix<Real> Vs_(Nsrc, nd_*COORD_DIM,
      sctl::Ptr2Itr<Real>((Real*)dipvec, nd_*COORD_DIM*Nsrc), false);
  sctl::Matrix<Real> Xs_(Nsrc, COORD_DIM,
      sctl::Ptr2Itr<Real>((Real*)sources, COORD_DIM*Nsrc), false);

  #pragma omp parallel for schedule(static)
  for (sctl::Long t = 0; t < Ntrg_; t += VecLen) {
    Vec Xtrg[COORD_DIM];
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      Xtrg[k] = Vec::LoadAligned(&Xt[k][t]);
    }

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

      Vec Rinv = sctl::approx_rsqrt<digits>(R2,
          (R2 > thresh2) & (R2 < d2max_vec));
      Vec Rinv3 = Rinv*Rinv*Rinv;
      Vec xtmp = FMA(R2, Rinv, cen_vec)*rsc_vec;
      Vec dptmp = l3d_eval_poly_dyn<Vec,Real>(xtmp, coefsd,
          ncoefsd[0]);

      Vec dfac = dptmp*Rinv3;
      for (long i = 0; i < nd_; i++) {
        Vec dotp = dX[0]*Vec::Load1(&Vs_[s][0*nd_+i]) +
            dX[1]*Vec::Load1(&Vs_[s][1*nd_+i]);
        if (COORD_DIM > 2) {
          dotp += dX[2]*Vec::Load1(&Vs_[s][2*nd_+i]);
        }
        Vtrg[i] += dotp*dfac;
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

template <class Real, sctl::Integer MaxVecLen=sctl::DefaultVecLen<Real>()>
void l3d_local_kernel_directdp_coef_vec_cpp(const int32_t* nd,
    const int32_t* ndim, const int32_t* digits, const Real* rsc,
    const Real* cen, const Real* d2max, const Real* sources,
    const int32_t* ns, const Real* dipvec, const Real* xtarg,
    const Real* ytarg, const Real* ztarg, const int32_t* nt,
    const int32_t* ncoefs, const Real* coefs, const int32_t* ncoefsd,
    const Real* coefsd, Real* pot, const Real* thresh) {
  if (ndim[0] == 3) {
    if (digits[0] <= 3)
      l3d_local_kernel_directdp_coef_vec_cpp_helper<Real,MaxVecLen,3,3>
          (nd,rsc,cen,d2max,sources,ns,dipvec,xtarg,ytarg,ztarg,nt,
          ncoefs,coefs,ncoefsd,coefsd,pot,thresh);
    else if (digits[0] <= 6)
      l3d_local_kernel_directdp_coef_vec_cpp_helper<Real,MaxVecLen,6,3>
          (nd,rsc,cen,d2max,sources,ns,dipvec,xtarg,ytarg,ztarg,nt,
          ncoefs,coefs,ncoefsd,coefsd,pot,thresh);
    else if (digits[0] <= 9)
      l3d_local_kernel_directdp_coef_vec_cpp_helper<Real,MaxVecLen,9,3>
          (nd,rsc,cen,d2max,sources,ns,dipvec,xtarg,ytarg,ztarg,nt,
          ncoefs,coefs,ncoefsd,coefsd,pot,thresh);
    else
      l3d_local_kernel_directdp_coef_vec_cpp_helper<Real,MaxVecLen,12,3>
          (nd,rsc,cen,d2max,sources,ns,dipvec,xtarg,ytarg,ztarg,nt,
          ncoefs,coefs,ncoefsd,coefsd,pot,thresh);
  }
}

#endif
