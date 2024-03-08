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
using CVectorRef = Eigen::Ref<const Vector<T>>;

template <typename T>
using MatrixRef = Eigen::Ref<Matrix<T>>;

template <typename T>
using CMatrixRef = Eigen::Ref<const Matrix<T>>;

template <typename T>
using LU = Eigen::PartialPivLU<Matrix<T>>;

template <typename T>
inline Vector<T> get_cheb_nodes(int order, T lb, T ub) {
    Vector<T> nodes(order);
    const T mean = 0.5 * (lb + ub);
    const T hw = 0.5 * (ub - lb);
    for (int i = 0; i < order; ++i)
        nodes[i] = mean + hw * cos(M_PI * ((order - i - 1) + 0.5) / order);

    return nodes;
}

template <typename T>
inline Matrix<T> calc_vandermonde(const CVectorRef<T> &nodes) {
    const int order = nodes.size();
    Matrix<T> V(order, order);

    for (int i = 0; i < order; ++i) {
        V(i, 0) = T{1};
        V(i, 1) = nodes(i);
    }

    for (int j = 2; j < order; ++j)
        for (int i = 0; i < order; ++i)
            V(i, j) = T(2) * V(i, j - 1) * nodes(i) - V(i, j - 2);

    return V;
}

template <typename T>
inline void calc_polynomial(int order, T x, VectorRef<T> poly) {
    poly[0] = 1.0;
    poly[1] = x;

    for (int i = 2; i < order; ++i)
        poly[i] = T{2} * x * poly[i - 1] - poly[i - 2];
}

template <typename T>
inline void calc_polynomial(int order, T x, T *poly_) {
    Eigen::Map<Vector<T>> poly(poly_, order);
    calc_polynomial<T>(order, x, poly);
}

template <typename T>
inline Vector<T> calc_polynomial(int order, T x) {
    Vector<T> poly(order);
    calc_polynomial<T>(order, x, poly);
    return poly;
}

template <typename T>
inline void calc_polynomial(int order, const CVectorRef<T> &x, MatrixRef<T> poly) {
    // Memory bottlenecked. Not really worth optimizing
    const int ns = x.rows();
    for (int i = 0; i < ns; ++i) {
        poly(0, i) = T{1.0};
        poly(1, i) = x[i];
        for (int j = 2; j < order; ++j)
            poly(j, i) = T{2} * poly(j - 1, i) * x[i] - poly(j - 2, i);
    }
}

template <typename T>
inline void calc_polynomial(int order, int n_poly, const T *x, T *poly) {
    Eigen::Map<const Vector<T>> x_(x, n_poly);
    Eigen::Map<Matrix<T>> poly_(poly, order, n_poly);
    calc_polynomial<T>(order, x_, poly_);
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
inline T evaluate(T x, int order, T lb, T ub, const T *c) {
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
inline T evaluate(T x, int order, const T *c) {
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
inline void evaluate(int order, int N, T lb, T ub, const T *__restrict x_p, const T *__restrict c_p,
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
        res[i] = evaluate(x_p[i], order, c_p);
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
