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

} // namespace dmk::chebyshev

namespace dmk::chebyshev::fixed {

template <typename T, int ORDER>
Eigen::Matrix<T, ORDER, ORDER> calc_vandermonde() {
    Eigen::Matrix<T, ORDER, ORDER> V;
    Eigen::Vector<T, ORDER> cosarray_;
    for (int i = 0; i < ORDER; ++i)
        cosarray_[ORDER - i - 1] = cos(M_PI * (i + 0.5) / ORDER);

    for (int j = 0; j < ORDER; ++j) {
        V(0, j) = 1;
        V(1, j) = cosarray_(j);
    }

    for (int i = 2; i < ORDER; ++i) {
        for (int j = 0; j < ORDER; ++j) {
            V(i, j) = T(2) * V(i - 1, j) * cosarray_(j) - V(i - 2, j);
        }
    }

    return V.transpose();
}

/// @brief Evaluate chebyshev polynomial given a box and a point inside that box
/// @tparam DIM dim of chebyshev polynomial to evaluate
/// @tparam ORDER order of chebyshev polynomial to evaluate
/// @param[in] x position of point to evaluate (pre-normalized on interval from -1:1)
/// @param[in] coeffs_raw flat vector of coefficients
/// @returns value of interpolating function at x
template <int DIM, int ORDER, typename T = double>
inline T cheb_eval(const Eigen::Vector<T, DIM> &x, const double *coeffs_raw);

template <int ORDER, typename T>
inline T cheb_eval(T x, const T *c) {
    // note (RB): uses clenshaw's method to avoid direct calculation of recurrence relation of
    // T_i, where res = \Sum_i T_i c_i
    const T x2 = 2 * x;

    T c0 = c[0];
    T c1 = c[1];
    for (int i = 2; i < ORDER; ++i) {
        T tmp = c1;
        c1 = c[i] - c0;
        c0 = tmp + c0 * x2;
    }

    return c1 + c0 * x;
}

template <int ORDER, typename T = double>
inline T cheb_eval(const Eigen::Vector<T, 1> &x, const T *c) {
    return cheb_eval<ORDER, T>(x[0], c);
}

template <int ORDER, typename T = double>
inline T cheb_eval(const Eigen::Vector<T, 2> &x, const T *coeffs_raw) {
    // note (RB): There is code to do this with clenshaw's method (twice), but it doesn't seem
    // faster (isolated tests shows it's 3x faster, but that doesn't bear fruit in production
    // and this is, imho, clearer)
    Eigen::Matrix<T, 2, ORDER> Tns;
    Tns.col(0).setOnes();
    Tns.col(1) = x;
    for (int i = 2; i < ORDER; ++i)
        Tns.col(i) = 2 * x.array() * Tns.col(i - 1).array() - Tns.col(i - 2).array();

    Eigen::Map<const Eigen::Matrix<T, ORDER, ORDER>> coeffs(coeffs_raw);

    return Tns.row(0).transpose().dot(coeffs * Tns.row(1).transpose());
}

template <int ORDER, typename T = double>
inline T cheb_eval(const Eigen::Vector<T, 3> &x, const T *coeffs_raw) {
    Eigen::Vector<T, ORDER> Tn[3];
    Tn[0][0] = Tn[1][0] = Tn[2][0] = 1.0;
    for (int i = 0; i < 3; ++i) {
        Tn[i][1] = x[i];
        for (int j = 2; j < ORDER; ++j)
            Tn[i][j] = 2 * x[i] * Tn[i][j - 1] - Tn[i][j - 2];
    }

    T res = 0.0;
    using map_t = Eigen::Map<const Eigen::Matrix<T, ORDER, ORDER>>;
    for (int i = 0; i < ORDER; ++i)
        res += Tn[0][i] * Tn[1].dot(map_t(coeffs_raw + i * ORDER * ORDER) * Tn[2]);
    return res;
}

template <int ORDER, typename T = double>
inline Eigen::Vector<T, ORDER> get_cheb_nodes(T lb, T ub) {
    Eigen::Vector<T, ORDER> res;
    const T mean = 0.5 * (lb + ub);
    const T hw = 0.5 * (ub - lb);
    for (int i = 0; i < ORDER; ++i)
        res[ORDER - i - 1] = mean + hw * cos(M_PI * (i + 0.5) / ORDER);

    return res;
}

template <int DIM, int ORDER, typename T = double, typename FT>
inline Eigen::Vector<T, ORDER> fit(FT &&func, const VectorRef<T> &lb, const VectorRef<T> &ub) {
    Eigen::PartialPivLU<Eigen::Matrix<T, ORDER, ORDER>> VLU(calc_vandermonde<T, ORDER>());

    if constexpr (DIM == 1) {
        Eigen::Vector<T, ORDER> xvec = get_cheb_nodes<ORDER, T>(lb[0], ub[0]);

        Eigen::Vector<T, ORDER> F;
        for (int i = 0; i < ORDER; ++i)
            F[i] = func(&xvec[i]);
        return Eigen::Reverse(VLU.solve(F));
    }
    if constexpr (DIM == 2) {
        Eigen::Vector<T, ORDER> xvec = get_cheb_nodes<ORDER, T>(lb[0], ub[0]);
        Eigen::Vector<T, ORDER> yvec = get_cheb_nodes<ORDER, T>(lb[1], ub[1]);

        Eigen::Matrix<T, ORDER, ORDER> F;
        for (int j = 0; j < ORDER; ++j) {
            for (int i = 0; i < ORDER; ++i) {
                double x[2] = {xvec[i], yvec[j]};
                F(i, j) = func(x);
            }
        }

        Eigen::Matrix<T, ORDER, ORDER> coeffs = VLU.solve(F);
        return VLU.solve(coeffs.transpose()).transpose();
    }
}

} // namespace dmk::chebyshev::fixed

namespace dmk::chebyshev::dynamic {

template <typename T>
Vector<T> get_cheb_nodes(int order, T lb, T ub) {
    Vector<T> nodes(order);
    const T mean = 0.5 * (lb + ub);
    const T hw = 0.5 * (ub - lb);
    for (int i = 0; i < order; ++i)
        nodes[order - i - 1] = mean + hw * cos(M_PI * (i + 0.5) / order);

    return nodes;
}

template <typename T>
Matrix<T> calc_vandermonde(int order) {
    Matrix<T> V(order, order);
    Vector<T> cosarray_ = get_cheb_nodes(order, -1.0, 1.0);

    for (int j = 0; j < order; ++j) {
        V(0, j) = 1;
        V(1, j) = cosarray_(j);
    }

    for (int i = 2; i < order; ++i) {
        for (int j = 0; j < order; ++j) {
            V(i, j) = T(2) * V(i - 1, j) * cosarray_(j) - V(i - 2, j);
        }
    }

    return V.transpose();
}

template <typename T>
inline T cheb_eval_1d(int order, const T x, const T *c) {
    // note (RB): uses clenshaw's method to avoid direct calculation of recurrence relation of
    // T_i, where res = \Sum_i T_i c_i
    const T x2 = 2 * x;

    T c0 = c[0];
    T c1 = c[1];
    for (int i = 2; i < order; ++i) {
        T tmp = c1;
        c1 = c[i] - c0;
        c0 = tmp + c0 * x2;
    }

    return c1 + c0 * x;
}

template <typename T, int VecLen>
inline void cheb_eval_1d(int order, int N, double lb, double ub, const T *__restrict x_p, const T *__restrict c_p,
                         T *__restrict res) {
    // note (RB): uses clenshaw's method to avoid direct calculation of recurrence relation of
    // T_i, where res = \Sum_i T_i c_i
    using vec_t = sctl::Vec<T, VecLen>;
    const int remainder = N % VecLen;
    N -= remainder;

    const vec_t inv_width = 1.0 / (ub - lb);
    const vec_t four_mean = 2.0 * (ub + lb);
    for (int i = 0; i < N; i += VecLen) {
        vec_t x2 = vec_t::Load(x_p + i);
        x2 = (4.0 * x2 - four_mean) * inv_width;

        vec_t c0 = c_p[0];
        vec_t c1 = c_p[1];
        for (int i = 2; i < order; ++i) {
            vec_t tmp = c1;
            c1 = c_p[i] - c0;
            c0 = tmp + c0 * x2;
        }

        c0 = c1 + 0.5 * c0 * x2;
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
        return Eigen::Reverse(vlu.solve(F));
    }

    return Vector<T>();
}

} // namespace dmk::chebyshev::dynamic
#endif
