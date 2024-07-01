#ifndef FORTRAN_H
#define FORTRAN_H

extern "C" {
void meshnd_(const int *, const double *, const int *, double *);

double besk0_(double *);
double besk1_(double *);
double besj0_(double *);
double besj1_(double *);

void prol0ini_(int *ier, double *beta, double *wprolate, double *rlam20, double *rkhi, int *lenw, int *keep, int *ltot);
void prol0eva_(const double *, const double *, double *, double *);
void prolate_intvals_(const double *, const double *, double *, double *, double *, double *);
void legeexps_(const int *, const int *, double *, double *, double *, double *);
void chebexps_(const int *, const int *, double *, double *, double *, double *);
void hank103_(const double _Complex *, double _Complex *, double _Complex *, const int *);
double hkrand_(int *);

void pdmk_charge2proxycharge_(const int *ndim, const int *nd, const int *norder, const int *ns, const double *sources,
                              const double *charge, const double *cen, const double *sc, double *coefs);
void dmk_proxycharge2pw_(const int *ndim, const int *nd, const int *n, const double *coefs, const int *npw,
                         const double *tab_coefs2pw, double *pwexp);
void mk_pw_translation_matrices_(const int *ndim, const double *xmin, const int *npw, const double *ts, const int *nmax,
                                 double *wshift);
void dmk_find_pwshift_ind_(const int *ndim, const int *iperiod, const double *tcenter, const double *scenter,
                           const double *bs0, const double *xmin, const int *nmax, int *ind);
void dmk_shiftpw_(const int *nd, const int *nexp, const double *pwexp1, double *pwexp2, const double *wshift);
void dmk_pw2proxypot_(const int *ndim, const int *nd, const int *n, const int *npw, const double *pwexp,
                      const double *tab_pw2coefs, double *coefs);
void pdmk_ortho_evalt_nd_(const int *ndim, const int *nd, const int *norder, const double *coefs, const int *nt,
                          const double *targ, const double *cen, const double *sc, double *pot);
void pdmk_direct_c_(const int *nd, const int *dim, const int *ikernel, const double *rpars, const int *ndigits,
                    const double *rsc, const double *cen, const int *ifself, const int *ncoefs, const double *coefs,
                    const double *d2max, const int *istart, const int *iend, const double *source, const int *ifcharge,
                    const double *charge, const int *ifdipole, const double *dipvec, const int *jstart, const int *jend,
                    const int *ntarget, const double *ctarg, const int *ifpgh, double *pot, double *grad, double *hess);
void yukawa_windowed_kernel_value_at_zero_(const int *dim, const double *rpars, const double *beta, const double *bsize,
                                           const double *rl, const double *wprolate, double *fval);
}

#endif
