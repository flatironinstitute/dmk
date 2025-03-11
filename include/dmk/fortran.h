#ifndef FORTRAN_H
#define FORTRAN_H

extern "C" {
// Used in production
void prol0ini_(int *ier, double *beta, double *wprolate, double *rlam20, double *rkhi, int *lenw, int *keep, int *ltot);
void prol0eva_(const double *, const double *, double *, double *);
void prolate_intvals_(const double *, const double *, double *, double *, double *, double *);
void legeexps_(const int *, const int *, double *, double *, double *, double *);

// Only used for tests
void pdmk_charge2proxycharge_(const int *ndim, const int *nd, const int *norder, const int *ns, const double *sources,
                              const double *charge, const double *cen, const double *sc, double *coefs);
void dmk_proxycharge2pw_(const int *ndim, const int *nd, const int *n, const double *coefs, const int *npw,
                         const double *tab_coefs2pw, double *pwexp);
void dmk_pw2proxypot_(const int *ndim, const int *nd, const int *n, const int *npw, const double *pwexp,
                      const double *tab_pw2coefs, double *coefs);
void pdmk_ortho_evalt_nd_(const int *ndim, const int *nd, const int *norder, const double *coefs, const int *nt,
                          const double *targ, const double *cen, const double *sc, double *pot);
void pdmk_(const int *nd, const int *dim, const double *eps, const int *ikernel, const double *rpars,
           const int *iperiod, const int *ns, const double *sources, const int *ifcharge, const double *charge,
           const int *ifdipole, const double *rnormal, const double *dipstr, const int *ifpgh, double *pot,
           double *grad, double *hess, const int *nt, const double *targ, const int *ifpghtarg, double *pottarg,
           double *gradtarg, double *hesstarg, double *tottimeinfo);
}

#endif
