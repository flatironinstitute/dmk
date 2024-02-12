#ifndef CHEBYCHEV_HPP
#define CHEBYCHEV_HPP

#include <sctl.hpp>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

namespace dmk::chebyshev {

template <typename T>
using Matrix = Eigen::MatrixX<T>;

template <typename T>
using Vector = Eigen::VectorX<T>;

template <typename T>
using VectorRef = Eigen::Ref<Vector<T>>;

template <typename T>
using LU = Eigen::PartialPivLU<Matrix<T>>;

template <typename T>
inline Vector<T> get_cheb_nodes(int order, T lb, T ub) {
    Vector<T> nodes(order);
    const T mean = 0.5 * (lb + ub);
    const T hw = 0.5 * (ub - lb);
    for (int i = 0; i < order; ++i)
        nodes[i] = mean + hw * cos(M_PI * (i + 0.5) / order);

    return nodes;
}

template <typename T>
inline Matrix<T> calc_vandermonde(const VectorRef<T> &nodes) {
    const int order = nodes.size();
    Matrix<T> V(order, order);

    for (int j = 0; j < order; ++j) {
        V(0, j) = 1;
        V(1, j) = nodes(j);
    }

    for (int i = 2; i < order; ++i) {
        for (int j = 0; j < order; ++j) {
            V(i, j) = T(2) * V(i - 1, j) * nodes(j) - V(i - 2, j);
        }
    }

    return V.transpose();
}

template <typename T>
inline Matrix<T> calc_vandermonde(int order) {
    Vector<T> cosarray = get_cheb_nodes<T>(order, -1.0, 1.0);
    return calc_vandermonde<T>(cosarray);
}

template <typename T>
inline const std::pair<Matrix<T>, LU<T>> &get_vandermonde_and_LU(int order) {
    static std::vector<std::pair<Matrix<T>, LU<T>>> vander_lus(128);
    if (!vander_lus[order].first.size()) {
        Matrix<T> vander = calc_vandermonde<T>(order);
        vander_lus[order] = std::make_pair(vander, LU<T>(vander));
    }

    return vander_lus[order];
}

template <typename T>
inline T cheb_eval(T x, int order, T lb, T ub, const T *c) {
    // note (RB): uses clenshaw's method to avoid direct calculation of recurrence relation of
    // T_i, where res = \Sum_i T_i c_i

    const T x2 = 2.0 * (T{2.0} * x - (ub + lb)) / (ub - lb);
    T c0 = c[order - 1];
    T c1 = c[order - 2];
    for (int i = 2; i < order; ++i) {
        T tmp = c1;
        c1 = c[order - i - 1] - c0;
        c0 = tmp + c0 * x2;
    }

    return c1 + T{0.5} * c0 * x2;
}

template <typename T>
inline T cheb_eval(T x, int order, const T *c) {
    // note (RB): uses clenshaw's method to avoid direct calculation of recurrence relation of
    // T_i, where res = \Sum_i T_i c_i
    const T x2 = T{2.0} * x;

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
inline void cheb_eval(int order, int N, T lb, T ub, const T *__restrict x_p, const T *__restrict c_p,
                      T *__restrict res) {
    // note (RB): uses clenshaw's method to avoid direct calculation of recurrence relation of
    // T_i, where res = \Sum_i T_i c_i
    using vec_t = sctl::Vec<T, VecLen>;
    const int remainder = N % VecLen;
    N -= remainder;

    const vec_t inv_width = T(1.0f / (ub - lb));
    const vec_t four_mean = T(2.0 * (ub + lb));
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
        res[i] = cheb_eval(x_p[i], order, c_p);
}

template <typename T>
inline std::pair<Matrix<T>, Matrix<T>> parent_to_child_matrices(int order) {
    auto &[_, u] = get_vandermonde_and_LU<T>(order);
    Vector<T> x = get_cheb_nodes<T>(order, -1.0, 1.0);

    // Shifted box positions. vec.array() allows element-wise/broadcasting ops
    Vector<T> xm = 0.5 * x.array() - 0.5;
    Vector<T> xp = 0.5 * x.array() + 0.5;

    // minus and plus "vandermonde" matrices
    Matrix<T> vm = calc_vandermonde<T>(xm);
    Matrix<T> vp = calc_vandermonde<T>(xp);

    return std::make_pair(u.solve(vm), u.solve(vp));
}

template <typename T>
inline Vector<T> fit(int order, T (*func)(T), T lb, T ub) {
    auto &[_, u] = get_vandermonde_and_LU<T>(order);

    Vector<T> xvec = get_cheb_nodes<T>(order, lb, ub);
    Vector<T> F(order);

    for (int i = 0; i < order; ++i)
        F[i] = func(xvec[i]);

    return u.solve(F);
}

template <typename T>
inline void fit(int order, T (*func)(T), T lb, T ub, T *coeffs) {
    Eigen::Map<Vector<T>> res(coeffs, order);
    res = fit(order, func, lb, ub);
}

} // namespace dmk::chebyshev
#endif
