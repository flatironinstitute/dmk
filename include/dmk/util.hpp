#ifndef UTIL_HPP
#define UTIL_HPP

#include <dmk/types.hpp>
#include <sctl.hpp>
#include <type_traits>

namespace dmk::util {
template <class...>
constexpr std::false_type always_false{};

template <typename Real>
void mesh_nd(int dim, Real *in, int size, Real *out);

template <typename Real>
void mesh_nd(int dim, const ndview<const Real, 1> &in, const ndview<Real, 2> &out);

template <typename Real>
void mk_tensor_product_fourier_transform(int dim, int npw, int nfourier, Real *fhat, int nexp, Real *pswfft);

template <typename Real>
void mk_tensor_product_fourier_transform(int dim, int npw, const ndview<const Real, 1> &fhat,
                                         const ndview<Real, 1> &pswfft);

template <typename T>
inline T int_pow(T base, int exp) {
    T result{1};
    for (int i = 0; i < exp; i++)
        result *= base;
    return result;
}

template <typename Real>
void init_test_data(int n_dim, int nd, int n_src, int n_trg, bool uniform, bool set_fixed_charges,
                    sctl::Vector<Real> &r_src, sctl::Vector<Real> &r_trg, sctl::Vector<Real> &rnormal,
                    sctl::Vector<Real> &charges, sctl::Vector<Real> &dipstr, long seed);
} // namespace dmk::util

#endif
