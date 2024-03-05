#ifndef FORTRAN_H
#define FORTRAN_H

extern "C" {
double besk0_(double *);
double besk1_(double *);
double besj0_(double *);
double besj1_(double *);

void prol0ini_(int *ier, double *beta, double *wprolate, double *rlam20, double *rkhi, int *lenw, int *keep, int *ltot);
void prol0eva_(const double *, const double *, double *, double *);
}

#endif
