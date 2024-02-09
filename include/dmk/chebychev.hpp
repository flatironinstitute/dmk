#ifndef CHEBYCHEV_HPP
#define CHEBYCHEV_HPP

#include <sctl.hpp>

#include <Eigen/Core>
#include <Eigen/LU>

namespace dmk::chebyshev {
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using Vector = Eigen::Vector<T, Eigen::Dynamic>;

template <typename T>
using VectorRef = Eigen::Ref<Vector<T>>;

template <typename T>
using LU = Eigen::PartialPivLU<Matrix<T>>;

template <typename T>
Vector<T> get_cheb_nodes(int order, T lb, T ub) {
    Vector<T> nodes(order);
    const T mean = 0.5 * (lb + ub);
    const T hw = 0.5 * (ub - lb);
    for (int i = 0; i < order; ++i)
        nodes[i] = mean + hw * cos(M_PI * (i + 0.5) / order);

    return nodes;
}

template <typename T>
Matrix<T> calc_vandermonde(int order) {
    Matrix<T> V(order, order);
    Vector<T> cosarray = get_cheb_nodes<T>(order, -1.0, 1.0);

    for (int j = 0; j < order; ++j) {
        V(0, j) = 1;
        V(1, j) = cosarray(j);
    }

    for (int i = 2; i < order; ++i) {
        for (int j = 0; j < order; ++j) {
            V(i, j) = T(2) * V(i - 1, j) * cosarray(j) - V(i - 2, j);
        }
    }

    return V.transpose();
}

template <typename T>
inline T cheb_eval_1d(int order, const T x, const T *c) {
    // note (RB): uses clenshaw's method to avoid direct calculation of recurrence relation of
    // T_i, where res = \Sum_i T_i c_i
    const T x2 = 2 * x;

    T c0 = c[order - 1];
    T c1 = c[order - 2];
    for (int i = 2; i < order; ++i) {
        T tmp = c1;
        c1 = c[order - i - 1] - c0;
        c0 = tmp + c0 * x2;
    }

    return c1 + c0 * x;
}

template <typename T, int VecLen>
inline void cheb_eval_1d(int order, int N, T lb, T ub, const T *__restrict x_p, const T *__restrict c_p,
                         T *__restrict res) {
    // note (RB): uses clenshaw's method to avoid direct calculation of recurrence relation of
    // T_i, where res = \Sum_i T_i c_i
    using vec_t = sctl::Vec<T, VecLen>;
    const int remainder = N % VecLen;
    N -= remainder;

    vec_t inv_width = T(1.0f / (ub - lb));
    vec_t four_mean = T(2.0 * (ub + lb));
    for (int i = 0; i < N; i += VecLen) {
        vec_t x2 = vec_t::Load(x_p + i);
        x2 = (T(4.0) * x2 - four_mean) * inv_width;

        vec_t c0 = c_p[order - 1];
        vec_t c1 = c_p[order - 2];
        for (int i = 2; i < order; ++i) {
            vec_t tmp = c1;
            c1 = c_p[order - i - 1] - c0;
            c0 = tmp + c0 * x2;
        }

        c0 = c1 + T(0.5) * c0 * x2;
        c0.Store(res + i);
    }

    for (int i = N; i < N + remainder; ++i)
        res[i] = cheb_eval_1d(order, x_p[i], c_p);
}

template <typename T>
Vector<T> cheb_eval(int dim, int order, const VectorRef<T> &x, const VectorRef<T> &coeffs) {
    if (dim == 1) {
        const int N = x / order;
        Vector<T> res(N);
        for (int i = 0; i < N; ++i)
            res[i] = cheb_eval_1d(order, x[i], coeffs.data());

        return res;
    }

    return Vector<T>();
}

template <typename T, typename FT>
Vector<T> fit(int dim, int order, FT &&func, const VectorRef<T> &lb, const VectorRef<T> &ub) {
    Matrix<T> lu = calc_vandermonde<T>(order);
    LU<T> vlu(lu);

    if (dim == 1) {
        Vector<T> xvec = get_cheb_nodes<T>(order, lb[0], ub[0]);
        Vector<T> F(order);

        for (int i = 0; i < order; ++i)
            F[i] = func(&xvec[i]);
        return vlu.solve(F);
    }

    return Vector<T>();
}

} // namespace dmk::chebyshev
#endif
