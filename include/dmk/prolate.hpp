#ifndef PROLATE_HPP
#define PROLATE_HPP

namespace dmk {

// Function declarations
void prolc180(double eps, double &c);
void prosinin(double c, const double *ts, const double *whts, const double *fs, double x, int n, double &rint,
              double &derrint);
void prolps0i(int &ier, double c, double *w, int lenw, int &nterms, int &ltot, double &rkhi);
void prolfun0(int &ier, int n, double c, double *as, double *bs, double *cs, double *xk, double *u, double *v,
              double *w, double eps, int &nterms, double &rkhi);
void prolcoef(double rlam, int k, double c, double &alpha0, double &beta0, double &gamma0, double &alpha, double &beta,
              double &gamma);
void prolmatr(double *as, double *bs, double *cs, int n, double c, double rlam, int ifsymm, int ifodd);
void prolql1(int N, double *D, double *E, int &IERR);
void prolfact(double *a, double *b, double *c, int n, double *u, double *v, double *w);
void prolsolv(const double *u, const double *v, const double *w, int n, double *rhs);
void prol0ini(int &ier, double c, double *w, double &rlam20, double &rkhi, int lenw, int &keep, int &ltot);
void prol0eva(double x, const double *w, double &psi0, double &derpsi0);
void prol0int0r(const double *w, double r, double &val);
void prolate_intvals(double beta, double *wprolate, double &c0, double &c1, double &c2, double &c4);
} // namespace dmk
  // 
#endif // PROLATE_HPP
