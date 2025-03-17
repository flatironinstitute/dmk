#ifndef LEGEEXPS_HPP
#define LEGEEXPS_HPP
#include <vector>

namespace dmk {

template <typename Real>
void legeexps(int itype, int n, Real *x, std::vector<std::vector<Real>> &u, std::vector<std::vector<Real>> &v,
              Real *whts);
template <typename Real>
void legeexev(Real x, Real &val, const Real *pexp, int n);
template <typename Real>
void legerts(int itype, int n, Real *ts, Real *whts);
template <typename Real>
void legeFDER(Real x, Real &val, Real &der, const Real *pexp, int n);
template <typename Real>
void legeexev(Real x, Real &val, const Real *pexp, int n);
template <typename Real>
void legeinte(Real *polin, int n, Real *polout);

} // namespace dmk

#endif // LEGEEXPS_HPP
