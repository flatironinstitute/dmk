#ifndef CHEBYCHEV_HPP
#define CHEBYCHEV_HPP
#include <doctest/doctest.h>

#include <sctl.hpp>
#include <tuple>

#include <nda/nda.hpp>

namespace dmk::chebyshev {

template <typename T>
using Matrix = nda::matrix<T, nda::F_layout>;

template <typename T>
using Vector = nda::vector<T>;

template <typename T>
struct LU {
    Matrix<T> lu;
    Vector<int> pivots;

    LU() = default;

    LU(auto &&A) {
        // Perform LU decomposition
        lu = A;
        nda::lapack::getrf(lu, pivots);
    }

    auto solve(const auto &b) const {
        auto res = b;
        nda::lapack::getrs(lu, res, pivots);
        return res;
    }
};

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
inline Matrix<T> calc_vandermonde(const auto &nodes) {
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
inline void calc_polynomial(int order, T x, auto &&poly) {
    poly[0] = 1.0;
    poly[1] = x;

    for (int i = 2; i < order; ++i)
        poly[i] = T{2} * x * poly[i - 1] - poly[i - 2];
}

template <typename T>
inline void calc_polynomial(int order, T x, T *poly_) {
    calc_polynomial<T>(order, x, nda::vector_view<T>({order}, poly_));
}

template <typename T, int order>
inline void calc_polynomial(T x, T *poly) {
    poly[0] = 1.0;
    poly[1] = x;

    for (int i = 2; i < order; ++i)
        poly[i] = T{2} * x * poly[i - 1] - poly[i - 2];
}

template <typename T>
inline auto get_polynomial_calculator(int order) {
    if (order == 9)
        return calc_polynomial<T, 9>;
    if (order == 18)
        return calc_polynomial<T, 18>;
    if (order == 28)
        return calc_polynomial<T, 28>;
    if (order == 38)
        return calc_polynomial<T, 38>;
    throw std::runtime_error("Unsupported order: " + std::to_string(order));
}

template <typename T>
inline Vector<T> calc_polynomial(int order, T x) {
    Vector<T> poly(order);
    calc_polynomial<T>(order, x, poly);
    return poly;
}

template <typename T>
inline void calc_polynomial(int order, const auto &x, auto &&poly) {
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
    nda::vector_const_view<T> x_({n_poly}, x);
    nda::matrix_view<T, nda::F_layout> poly_({order, n_poly}, poly);
    calc_polynomial<T>(order, x_, poly_);
}

template <typename T>
inline Matrix<T> calc_vandermonde(int order) {
    Vector<T> cosarray = get_cheb_nodes<T>(order, -1.0, 1.0);
    return calc_vandermonde<T>(cosarray);
}

template <typename T>
inline const std::pair<Matrix<T>, LU<T>> &get_vandermonde_and_LU(int order) {
    static sctl::Vector<std::pair<Matrix<T>, LU<T>>> vander_lus(128);
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
    Vector<T> xm = 0.5 * x - 0.5;
    Vector<T> xp = 0.5 * x + 0.5;

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
    nda::basic_array_view<T, 1, nda::F_layout> res({order}, coeffs);
    res = fit(order, func, lb, ub);
}

template <typename T>
std::pair<sctl::Vector<T>, sctl::Vector<T>> get_c2p_p2c_matrices(int dim, int order) {
    std::array<nda::matrix<T, nda::F_layout, nda::heap<>>, 2> c2p_mp;
    std::array<nda::matrix<T, nda::F_layout, nda::heap<>>, 2> p2c_mp;

    std::tie(p2c_mp[0], p2c_mp[1]) = dmk::chebyshev::parent_to_child_matrices<T>(order);
    c2p_mp[0] = nda::transpose(p2c_mp[0]);
    c2p_mp[1] = nda::transpose(p2c_mp[1]);

    const int mc = std::pow(2, dim);

    std::pair<sctl::Vector<T>, sctl::Vector<T>> res;
    auto &[c2p, p2c] = res;

    c2p.ReInit(order * order * dim * mc);
    p2c.ReInit(order * order * dim * mc);

    const int blocksize = order * order;
    const int matsize = order * order * sizeof(T);
    if (dim == 1) {
        memcpy(&c2p[0], c2p_mp[0].data(), matsize);
        memcpy(&c2p[0] + blocksize, p2c_mp[1].data(), matsize);

        memcpy(&p2c[0], p2c_mp[0].data(), matsize);
        memcpy(&p2c[0] + blocksize, p2c_mp[1].data(), matsize);
    }
    if (dim == 2) {
        int offset = 0;
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                for (auto idx : {i, j}) {
                    memcpy(&c2p[0] + offset, c2p_mp[idx].data(), matsize);
                    memcpy(&p2c[0] + offset, p2c_mp[idx].data(), matsize);
                    offset += blocksize;
                }
            }
        }
    }

    if (dim == 3) {
        int offset = 0;
        for (int k = 0; k < 2; ++k) {
            for (int j = 0; j < 2; ++j) {
                for (int i = 0; i < 2; ++i) {
                    for (auto idx : {i, j, k}) {
                        memcpy(&c2p[0] + offset, c2p_mp[idx].data(), matsize);
                        memcpy(&p2c[0] + offset, p2c_mp[idx].data(), matsize);
                        offset += blocksize;
                    }
                }
            }
        }
    }

    return res;
}

TEST_CASE_TEMPLATE("[DMK] chebyshev fit", T, float, double) {
    T lb{-1.0}, ub{1.0};
    auto testfunc = [](T x) -> T { return std::sin(x * x * x) + 0.5; };

    for (const int order : {5, 9, 16, 24}) {
        CAPTURE(order);
        // return by value
        Vector<T> coeffs_rbv = fit<T>(order, testfunc, lb, ub);
        // pass by reference
        Vector<T> coeffs_pbr(order);
        fit<T>(order, testfunc, lb, ub, coeffs_pbr.data());

        CHECK(coeffs_rbv == coeffs_pbr);
    }
}

TEST_CASE_TEMPLATE("[DMK] chebyshev interpolation", T, float, double) {
    // Check that automatic interpolation by passing bounds works the same as the implicit [-1.0, 1.0]
    T lb{-1.0}, ub{1.0};
    auto testfunc = [](T x) -> T { return std::sin(x * x * x) + 0.5; };

    for (const int order : {5, 9, 16, 24}) {
        Vector<T> coeffs = fit<T>(order, testfunc, lb, ub);
        CAPTURE(order);

        for (T x = lb; x <= ub; x += (ub - lb) / 9) {
            T res = evaluate(x, order, coeffs.data());
            T res_alt = evaluate(x, order, lb, ub, coeffs.data());
            CHECK(std::fabs(res - res_alt) <= std::numeric_limits<T>::epsilon());
        }
    }
}

TEST_CASE_TEMPLATE("[DMK] chebyshev translation", T, float, double) {
    // Check that parent->child translation matrices work to reasonable precision
    // Larger bounds shifts -> larger errors.
    T lb{-1.3}, ub{1.2};
    T mid = lb + 0.5 * (ub - lb);

    auto testfunc = [](T x) -> T { return std::sin(x * x * x) + 0.5; };

    for (const int order : {5, 9, 16, 24}) {
        CAPTURE(order);

        Matrix<T> tm, tp;
        std::tie(tm, tp) = parent_to_child_matrices<T>(order);
        Vector<T> coeffs = fit<T>(order, testfunc, lb, ub);
        Vector<T> coeffs_m = tm * coeffs;
        Vector<T> coeffs_p = tp * coeffs;

        for (T x = lb; x <= mid; x += (mid - lb) / 9) {
            T res = evaluate<T>(x, order, lb, ub, coeffs.data());
            T res_alt = evaluate<T>(x, order, lb, mid, coeffs_m.data());
            CHECK(std::fabs(res - res_alt) <= 5 * std::numeric_limits<T>::epsilon());
        }

        for (T x = mid; x <= ub; x += (ub - mid) / 9) {
            T res = evaluate<T>(x, order, lb, ub, coeffs.data());
            T res_alt = evaluate<T>(x, order, mid, ub, coeffs_p.data());
            CHECK(std::fabs(res - res_alt) <= 5 * std::numeric_limits<T>::epsilon());
        }
    }
}

} // namespace dmk::chebyshev
#endif
