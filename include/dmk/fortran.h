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
}

#endif
