#ifndef UTIL_HPP
#define UTIL_HPP

#include <dmk/omp_wrapper.hpp>
#include <dmk/types.hpp>
#include <random>
#include <sctl.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>

#include <dmk/bessel.hpp>

namespace dmk::util {
template <class...>
constexpr std::false_type always_false{};

// Orders 0 and 1 only (all DMK needs). Evaluated in double regardless of the
// argument type, matching the promotion behavior of std::cyl_bessel_* that this
// replaced, so float builds still get double-accurate generation coefficients.
static inline double cyl_bessel_k(int nu, double x) { return nu == 0 ? dmk::bessel::k0(x) : dmk::bessel::k1(x); }

static inline double cyl_bessel_j(int nu, double x) { return nu == 0 ? dmk::bessel::j0(x) : dmk::bessel::j1(x); }

template <typename T, size_t StackSize>
class StackOrHeapBuffer {
    alignas(64) std::array<T, StackSize> stack_buffer_;
    T *heap_buffer_ = nullptr;
    T *data_;

  public:
    StackOrHeapBuffer(size_t required_size) {
        if (required_size <= StackSize) {
            data_ = stack_buffer_.data();
        } else {
            size_t alloc_size = required_size * sizeof(T);
            // aligned_alloc requires size to be a multiple of alignment
            alloc_size = (alloc_size + 63) & ~size_t{63};
            heap_buffer_ = static_cast<T *>(std::aligned_alloc(64, alloc_size));
            data_ = heap_buffer_;
        }
    }

    T *data() { return data_; }
    const T *data() const { return data_; }

    ~StackOrHeapBuffer() {
        if (heap_buffer_) {
            std::free(heap_buffer_);
        }
    }
};

// Canonical lowercase-snake names, used everywhere a kernel/eval-type is shown
// or parsed (CLI args, table/CSV output, error messages). The inverse parsers
// below are case-insensitive and ignore a leading "DMK_", so the C enum spelling
// also round-trips.
constexpr std::array<std::string_view, 6> ikernel_names = {
    "yukawa", "laplace", "sqrt_laplace", "stokeslet", "stresslet", "laplace_dipole",
};

constexpr std::array<std::string_view, 3> return_names = {
    "potential",
    "potential_grad",
    "velocity",
};

constexpr std::string_view to_string(dmk_ikernel k) noexcept {
    auto idx = static_cast<int>(k);
    if (idx >= 0 && idx < static_cast<int>(ikernel_names.size()))
        return ikernel_names[idx];
    return "unknown_kernel";
}

constexpr std::string_view to_string(dmk_eval_type k) noexcept {
    auto idx = static_cast<int>(k) - 1;
    if (idx >= 0 && idx < static_cast<int>(return_names.size()))
        return return_names[idx];
    return "unknown_eval_type";
}

// Case-insensitive equality, ignoring an optional leading "DMK_" on either side and treating '-'
// as '_', so "stokeslet", "DMK_STOKESLET" and "sqrt-laplace" all match their canonical spellings.
constexpr bool name_matches(std::string_view a, std::string_view b) noexcept {
    auto strip = [](std::string_view s) { return s.substr(0, 4) == "DMK_" ? s.substr(4) : s; };
    auto lower = [](char c) -> char {
        if (c >= 'A' && c <= 'Z')
            return static_cast<char>(c - 'A' + 'a');
        return c == '-' ? '_' : c;
    };
    a = strip(a);
    b = strip(b);
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (lower(a[i]) != lower(b[i]))
            return false;
    return true;
}

// Inverse of to_string. Accepts the canonical name with or without the "DMK_"
// prefix, case-insensitively (e.g. "DMK_YUKAWA", "yukawa"). Returns nullopt if
// no kernel matches.
constexpr std::optional<dmk_ikernel> ikernel_from_string(std::string_view s) noexcept {
    for (int i = 0; i < static_cast<int>(ikernel_names.size()); ++i)
        if (name_matches(s, ikernel_names[i]))
            return static_cast<dmk_ikernel>(i);
    return std::nullopt;
}

// Inverse of to_string for eval types (enum starts at 1). Same matching rules.
constexpr std::optional<dmk_eval_type> eval_type_from_string(std::string_view s) noexcept {
    for (int i = 0; i < static_cast<int>(return_names.size()); ++i)
        if (name_matches(s, return_names[i]))
            return static_cast<dmk_eval_type>(i + 1);
    return std::nullopt;
}

double calc_bandlimiting(const pdmk_params &p);

template <typename Real>
void mesh_nd(int dim, Real *in, int size, Real *out);

template <typename Real>
void mesh_nd(int dim, const ndview<const Real, 1> &in, ndview<Real, 2> out);

template <typename Real>
void mk_tensor_product_fourier_transform(int dim, int npw, int nfourier, Real *fhat, int nexp, Real *pswfft);

template <typename Real>
void mk_tensor_product_fourier_transform(int dim, int npw, const ndview<Real, 1> &fhat, ndview<Real, 1> pswfft);

template <typename Real, int ORDER>
inline Real dot_product(Real *a, Real *b) {
    // LOL I KNOW BUT IT'S FASTER
    Real res{0.0};
    for (int i = 0; i < ORDER; ++i)
        res += a[i] * b[i];

    return res;
}

template <typename T, int... Is>
inline auto get_opt_dot_impl(int n_order, std::integer_sequence<int, Is...>) {
    using fn_t = decltype(&dmk::util::dot_product<T, 0>);
    fn_t result = nullptr;
    (void)((Is + min_proxy_order == n_order ? (result = &dmk::util::dot_product<T, Is + min_proxy_order>, true)
                                            : false) ||
           ...);
    if (!result)
        throw std::runtime_error("Invalid order " + std::to_string(n_order));
    return result;
}

template <typename T>
inline auto get_opt_dot(int n_order) {
    return get_opt_dot_impl<T>(n_order, std::make_integer_sequence<int, n_proxy_orders>{});
}

template <typename T>
inline T int_pow(T base, int exp) {
    T result{1};
    for (int i = 0; i < exp; i++)
        result *= base;
    return result;
}

inline bool env_is_set(const char *name) {
    const char *val = getenv(name);
    return val != nullptr && val[0] != '\0' && std::string_view(val) != "0";
}

template <typename T>
concept HasResize = requires(T t, size_t n) { t.resize(n); };

inline auto size_to = [](auto &v, size_t n) {
    if constexpr (HasResize<decltype(v)>)
        v.resize(n);
    else
        v.ReInit(n);
};

// std::default_random_engine and std::uniform_real_distribution are implementation dependent
// one seed with them can draw different points under libstdc++ and libc++.
// mt19937 is specified exactly, so we just scale the integer output by 2^-32 to live on [0, 1)
struct TestRng {
    explicit TestRng(uint32_t seed) : eng(seed) {}
    inline double operator()() { return eng() * 0x1p-32; }
    std::mt19937 eng;
};

template <typename Real>
struct UniformVolume {
    UniformVolume(int n_dim, Real side_length, long seed) : DIM(n_dim), L(side_length), rng(seed) {};

    inline void operator()(Real *res) {
        const Real shift = 0.5 * (1 - L);
        for (int i = 0; i < DIM; ++i)
            res[i] = dist() * L + shift;
    };

    inline Real dist() { return rng(); };
    constexpr int n_dim() { return DIM; }

    const int DIM;
    const Real L;
    TestRng rng;
};

// Volume fill whose density rises linearly along x, from sampling x as sqrt(u).
template <typename Real>
struct GradedVolume {
    GradedVolume(int n_dim, Real side_length, long seed) : DIM(n_dim), L(side_length), rng(seed) {};

    inline void operator()(Real *res) {
        const Real shift = 0.5 * (1 - L);
        res[0] = std::sqrt(dist()) * L + shift;
        for (int i = 1; i < DIM; ++i)
            res[i] = dist() * L + shift;
    };

    inline Real dist() { return rng(); };
    constexpr int n_dim() { return DIM; }

    const int DIM;
    const Real L;
    TestRng rng;
};

// A sphere (circle in 2D) deformed into radial lobes: the radius is a function of angle, dipping
// inward over `polar_modes` lobes in the polar angle and `azimuthal_modes` lobes in the azimuth, so
// surface density and the leaf occupancy that follows from it vary around the surface instead of
// being uniform over a shell. The deformation is one-sided -- the radius spans [radius - amplitude,
// radius]. amplitude = 0 degenerates to an exact sphere.
template <typename Real>
struct LobedNSphere {
    LobedNSphere(int n_dim, Real radius, long seed, Real amplitude = Real(0.12), int polar_modes = 6,
                 int azimuthal_modes = 0)
        : DIM(n_dim), R(radius), A(amplitude), N_polar(polar_modes), N_azim(azimuthal_modes), rng(seed) {};

    inline void operator()(Real *res) {
        if (DIM == 2) {
            const Real phi = dist() * 2 * M_PI;
            const Real rr = radius_at(phi, Real{0});
            res[0] = rr * cos(phi) + 0.5;
            res[1] = rr * sin(phi) + 0.5;
        } else if (DIM == 3) {
            const Real theta = dist() * M_PI;
            const Real phi = dist() * 2 * M_PI;
            const Real rr = radius_at(theta, phi);
            const Real ct = cos(theta), st = sin(theta);
            res[0] = rr * st * cos(phi) + 0.5;
            res[1] = rr * st * sin(phi) + 0.5;
            res[2] = rr * ct + 0.5;
        }
    };

    inline Real radius_at(Real polar, Real azim) const {
        const Real lobe = std::cos(N_polar * polar) * (N_azim ? std::cos(N_azim * azim) : Real{1});
        return R - A * (Real{1} - lobe) / 2;
    }

    inline Real dist() { return rng(); };
    constexpr int n_dim() { return DIM; }

    const int DIM;
    const Real R;
    const Real A;
    const int N_polar;
    const int N_azim;
    TestRng rng;
};

template <typename Real>
struct NCubePartialFacet {
    NCubePartialFacet(int n_dim, Real side_length_, Real band_width_, long seed)
        : DIM(n_dim), L(side_length_), W(band_width_), rng(seed) {};

    inline void operator()(Real *res) {
        const Real h = Real{0.5} * L;

        if (DIM == 2) {
            using vec_type = sctl::Vec<Real, 2>;
            vec_type res_vec;

            // Sample points along boundary
            const Real lpos = dist();
            const vec_type p = [&]() -> vec_type {
                const int segment = 4 * lpos;
                const Real t = (Real{4} * lpos - segment) * L;
                if (segment == 0)
                    return {t, Real{0}};
                else if (segment == 1)
                    return {t, L};
                else if (segment == 2)
                    return {Real{0}, t};
                else
                    return {L, t};
            }() - h;
            p.Store(res);
        } else if (DIM == 3) {
            const int normal_axis = 3 * dist();
            const Real normal_sign = dist() < 0.5 ? -1 : 1;

            // Distance from the nearest edge.
            const Real u = dist();
            const Real d = 0.5 * (L - std::sqrt(L * L - 4.0 * u * (L * W - W * W)));

            // Pick which of the 4 edges of the face we're near.
            const int edge_axis = 2 * dist();

            // Pick which side of that coordinate.
            const Real edge_sign = dist() < 0.5 ? -1 : 1;

            // Position along the edge.
            const Real t = (2.0 * dist() - 1.0) * (h - d);

            for (int i = 0; i < 3; i++)
                res[i] = 0.5;

            // The coordinate normal to the face.
            res[normal_axis] += normal_sign * h;

            // The two coordinates within the face.
            const int a = (normal_axis + 1) % 3;
            const int b = (normal_axis + 2) % 3;

            if (edge_axis == 0) {
                res[a] += edge_sign * (h - d);
                res[b] += t;
            } else {
                res[a] += t;
                res[b] += edge_sign * (h - d);
            }
        }
    };

    inline Real dist() { return rng(); }
    constexpr int n_dim() { return DIM; };

    const int DIM;
    const Real L, W;
    TestRng rng;
};

inline void init_test_data(int n_dim, int nd, int n_src, int n_trg, auto point_generator, bool set_fixed_charges,
                           auto &r_src, auto &r_trg, auto &r_normal, auto &charges) {
    using Real = std::decay_t<decltype(r_src)>::value_type;
    size_to(r_src, n_dim * n_src);
    size_to(r_trg, n_dim * n_trg);
    size_to(charges, nd * n_src);
    size_to(r_normal, n_dim * n_src);

    // Redraw until the point is interior at the working precision (a double draw < 1 can
    // round UP to 1.0 in float, landing on the box boundary) and distinct from every point
    // emitted so far. Coincident points (near-duplicates collapsing below an ulp) and
    // boundary points otherwise corrupt the tree / far-field.
    std::unordered_set<std::string> seen;
    auto emit = [&](Real *p) {
        for (;;) {
            point_generator(p);
            bool interior = true;
            for (int j = 0; j < n_dim; ++j)
                interior &= (p[j] > Real(0) && p[j] < Real(1));
            if (interior && seen.insert(std::string(reinterpret_cast<const char *>(p), n_dim * sizeof(Real))).second)
                return;
        }
    };

    for (int i = 0; i < n_src; ++i) {
        emit(&r_src[i * n_dim]);

        // Unit normals (sphere-distributed) — required for stresslet, harmless otherwise.
        if (n_dim == 2) {
            const Real phi_n = point_generator.dist() * 2 * M_PI;
            r_normal[i * 2 + 0] = std::cos(phi_n);
            r_normal[i * 2 + 1] = std::sin(phi_n);
        } else if (n_dim == 3) {
            const Real theta_n = point_generator.dist() * M_PI;
            const Real ct_n = std::cos(theta_n), st_n = std::sin(theta_n);
            const Real phi_n = point_generator.dist() * 2 * M_PI;
            r_normal[i * 3 + 0] = st_n * std::cos(phi_n);
            r_normal[i * 3 + 1] = st_n * std::sin(phi_n);
            r_normal[i * 3 + 2] = ct_n;
        }

        for (int j = 0; j < nd; ++j) {
            charges[i * nd + j] = point_generator.dist() - 0.5;
        }
    }

    for (int i_trg = 0; i_trg < n_trg; ++i_trg)
        emit(&r_trg[i_trg * n_dim]);

    if (set_fixed_charges && n_src > 0)
        for (int i = 0; i < n_dim; ++i)
            r_src[i] = 0.0;
    if (set_fixed_charges && n_src > 1)
        for (int i = n_dim; i < 2 * n_dim; ++i)
            r_src[i] = 1 - std::numeric_limits<Real>::epsilon();
    if (set_fixed_charges && n_src > 2)
        for (int i = 2 * n_dim; i < 3 * n_dim; ++i)
            r_src[i] = 0.05;
}

enum class Distribution : int {
    Uniform = 0,
    LobedNSphere = 1,
    NCubePartialFacet = 2,
    GradedVolume = 3,
};

inline void init_test_data(int n_dim, int nd, int n_src, int n_trg, Distribution dist, bool set_fixed_charges,
                           auto &r_src, auto &r_trg, auto &r_normal, auto &charges, long seed) {
    using Real = std::decay_t<decltype(r_src)>::value_type;
    if (dist == Distribution::LobedNSphere)
        return init_test_data(n_dim, nd, n_src, n_trg, LobedNSphere<Real>(n_dim, 0.95 * 0.5, seed), set_fixed_charges,
                              r_src, r_trg, r_normal, charges);
    if (dist == Distribution::NCubePartialFacet)
        return init_test_data(n_dim, nd, n_src, n_trg, NCubePartialFacet<Real>(n_dim, 0.95, 0.02, seed),
                              set_fixed_charges, r_src, r_trg, r_normal, charges);
    constexpr Real almost_one = Real(1) - std::numeric_limits<Real>::epsilon();
    if (dist == Distribution::GradedVolume)
        return init_test_data(n_dim, nd, n_src, n_trg, GradedVolume<Real>(n_dim, almost_one, seed), set_fixed_charges,
                              r_src, r_trg, r_normal, charges);
    return init_test_data(n_dim, nd, n_src, n_trg, UniformVolume<Real>(n_dim, almost_one, seed), set_fixed_charges,
                          r_src, r_trg, r_normal, charges);
}

inline void init_test_data(int n_dim, int nd, int n_src, int n_trg, bool uniform, bool set_fixed_charges, auto &r_src,
                           auto &r_trg, auto &rnormal, auto &charges, long seed) {
    return init_test_data(n_dim, nd, n_src, n_trg, uniform ? Distribution::Uniform : Distribution::LobedNSphere,
                          set_fixed_charges, r_src, r_trg, rnormal, charges, seed);
}

template <typename T>
inline void vec_mul(T *__restrict__ dst, const T *__restrict__ a, const T *__restrict__ b, int n) {
    using Vec = sctl::Vec<T, sctl::DefaultVecLen<T>()>;
    constexpr int N = Vec::Size();
    int i = 0;
    for (; i + N <= n; i += N) {
        Vec va = Vec::Load(a + i);
        Vec vb = Vec::Load(b + i);
        (va * vb).Store(dst + i);
    }
    for (; i < n; ++i)
        dst[i] = a[i] * b[i];
}

template <typename T>
inline void vec_mul_broadcast(T *__restrict__ dst, const T *__restrict__ a, T b, int n) {
    using Vec = sctl::Vec<T, sctl::DefaultVecLen<T>()>;
    constexpr int N = Vec::Size();
    Vec vb(b);
    int i = 0;
    for (; i + N <= n; i += N) {
        Vec va = Vec::Load(a + i);
        (va * vb).Store(dst + i);
    }
    for (; i < n; ++i)
        dst[i] = a[i] * b;
}

template <typename T>
inline void vec_fma(T *__restrict__ dst, const T *__restrict__ a, const T *__restrict__ b, int n) {
    using Vec = sctl::Vec<T, sctl::DefaultVecLen<T>()>;
    constexpr int N = Vec::Size();
    int i = 0;
    for (; i + N <= n; i += N) {
        Vec vd = Vec::Load(dst + i);
        Vec va = Vec::Load(a + i);
        Vec vb = Vec::Load(b + i);
        FMA(va, vb, vd).Store(dst + i);
    }
    for (; i < n; ++i)
        dst[i] += a[i] * b[i];
}

template <typename T>
inline void vec_fma_3(T *__restrict__ dst, const T *__restrict__ a, const T *__restrict__ b, const T *__restrict__ c,
                      int n) {
    using Vec = sctl::Vec<T, sctl::DefaultVecLen<T>()>;
    constexpr int N = Vec::Size();
    int i = 0;
    for (; i + N <= n; i += N) {
        Vec vd = Vec::Load(dst + i);
        Vec va = Vec::Load(a + i);
        Vec vb = Vec::Load(b + i);
        Vec vc = Vec::Load(c + i);
        FMA(va * vb, vc, vd).Store(dst + i);
    }
    for (; i < n; ++i)
        dst[i] += a[i] * b[i] * c[i];
}

// Fused inner kernel: computes all 4 accumulations in a single pass over n elements,
// sharing py*tf between potential and grad_x.
//   pot[k] += py[k] * tf[k] * px[k]
//   gx[k]  += py[k] * tf[k] * dpx[k]      (reuses py*tf)
//   gy[k]  += dpy[k] * tf[k] * px[k]
//   gz[k]  += py[k] * tf_z[k] * px[k]
template <typename T>
inline void vec_fma_3_grad(T *__restrict__ pot, T *__restrict__ gx, T *__restrict__ gy, T *__restrict__ gz,
                           const T *__restrict__ py, const T *__restrict__ dpy, const T *__restrict__ tf,
                           const T *__restrict__ tf_z, const T *__restrict__ px, const T *__restrict__ dpx, int n) {
    using Vec = sctl::Vec<T, sctl::DefaultVecLen<T>()>;
    constexpr int N = Vec::Size();
    int i = 0;
    for (; i + N <= n; i += N) {
        Vec vpy = Vec::Load(py + i);
        Vec vdpy = Vec::Load(dpy + i);
        Vec vtf = Vec::Load(tf + i);
        Vec vtf_z = Vec::Load(tf_z + i);
        Vec vpx = Vec::Load(px + i);
        Vec vdpx = Vec::Load(dpx + i);

        Vec py_tf = vpy * vtf;

        FMA(py_tf, vpx, Vec::Load(pot + i)).Store(pot + i);
        FMA(py_tf, vdpx, Vec::Load(gx + i)).Store(gx + i);
        FMA(vdpy * vtf, vpx, Vec::Load(gy + i)).Store(gy + i);
        FMA(vpy * vtf_z, vpx, Vec::Load(gz + i)).Store(gz + i);
    }
    for (; i < n; ++i) {
        T py_tf = py[i] * tf[i];
        pot[i] += py_tf * px[i];
        gx[i] += py_tf * dpx[i];
        gy[i] += dpy[i] * tf[i] * px[i];
        gz[i] += py[i] * tf_z[i] * px[i];
    }
}

// 2D fused inner kernel for the EVAL_LEVEL == 2 proxy accumulator.
//   pot[k] += t[k]   * px[k]
//   gx[k]  += t[k]   * dpx[k]    (reuses t)
//   gy[k]  += t_y[k] * px[k]     (reuses px)
template <typename T>
inline void vec_fma_2_grad(T *__restrict__ pot, T *__restrict__ gx, T *__restrict__ gy, const T *__restrict__ t,
                           const T *__restrict__ t_y, const T *__restrict__ px, const T *__restrict__ dpx, int n) {
    using Vec = sctl::Vec<T, sctl::DefaultVecLen<T>()>;
    constexpr int N = Vec::Size();
    int i = 0;
    for (; i + N <= n; i += N) {
        Vec vt = Vec::Load(t + i);
        Vec vty = Vec::Load(t_y + i);
        Vec vpx = Vec::Load(px + i);
        Vec vdpx = Vec::Load(dpx + i);

        FMA(vt, vpx, Vec::Load(pot + i)).Store(pot + i);
        FMA(vt, vdpx, Vec::Load(gx + i)).Store(gx + i);
        FMA(vty, vpx, Vec::Load(gy + i)).Store(gy + i);
    }
    for (; i < n; ++i) {
        pot[i] += t[i] * px[i];
        gx[i] += t[i] * dpx[i];
        gy[i] += t_y[i] * px[i];
    }
}

#if defined(__AVX512F__)
inline void complex_deinterleave(const __m512 &lo, const __m512 &hi, __m512 &real, __m512 &imag) {
    const __m512i idx_r = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i idx_i = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    real = _mm512_permutex2var_ps(lo, idx_r, hi);
    imag = _mm512_permutex2var_ps(lo, idx_i, hi);
}
inline void complex_interleave(const __m512 &real, const __m512 &imag, __m512 &lo, __m512 &hi) {
    const __m512i idx_lo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    const __m512i idx_hi = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
    lo = _mm512_permutex2var_ps(real, idx_lo, imag);
    hi = _mm512_permutex2var_ps(real, idx_hi, imag);
}
inline void complex_deinterleave(const __m512d &lo, const __m512d &hi, __m512d &real, __m512d &imag) {
    const __m512i idx_r = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i idx_i = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    real = _mm512_permutex2var_pd(lo, idx_r, hi);
    imag = _mm512_permutex2var_pd(lo, idx_i, hi);
}
inline void complex_interleave(const __m512d &real, const __m512d &imag, __m512d &lo, __m512d &hi) {
    const __m512i idx_lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i idx_hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    lo = _mm512_permutex2var_pd(real, idx_lo, imag);
    hi = _mm512_permutex2var_pd(real, idx_hi, imag);
}
#endif

#ifdef __AVX2__
inline void complex_deinterleave(const __m256 &lo, const __m256 &hi, __m256 &real, __m256 &imag) {
    // lo = [r0,i0,r1,i1,r2,i2,r3,i3], hi = [r4,i4,r5,i5,r6,i6,r7,i7]
    __m256 a = _mm256_shuffle_ps(lo, hi, 0x88); // [r0,r1,r4,r5,r2,r3,r6,r7]
    __m256 b = _mm256_shuffle_ps(lo, hi, 0xDD); // [i0,i1,i4,i5,i2,i3,i6,i7]
    real = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(a), 0xD8));
    imag = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(b), 0xD8));
}
inline void complex_interleave(const __m256 &real, const __m256 &imag, __m256 &lo, __m256 &hi) {
    lo = _mm256_unpacklo_ps(real, imag);              // [r0,i0,r1,i1,r4,i4,r5,i5]
    hi = _mm256_unpackhi_ps(real, imag);              // [r2,i2,r3,i3,r6,i6,r7,i7]
    __m256 t0 = _mm256_permute2f128_ps(lo, hi, 0x20); // [r0,i0,r1,i1,r2,i2,r3,i3]
    __m256 t1 = _mm256_permute2f128_ps(lo, hi, 0x31); // [r4,i4,r5,i5,r6,i6,r7,i7]
    lo = t0;
    hi = t1;
}
inline void complex_deinterleave(const __m256d &lo, const __m256d &hi, __m256d &real, __m256d &imag) {
    real = _mm256_permute4x64_pd(_mm256_unpacklo_pd(lo, hi), 0xD8);
    imag = _mm256_permute4x64_pd(_mm256_unpackhi_pd(lo, hi), 0xD8);
}
inline void complex_interleave(const __m256d &real, const __m256d &imag, __m256d &lo, __m256d &hi) {
    lo = _mm256_unpacklo_pd(real, imag);
    hi = _mm256_unpackhi_pd(real, imag);
    __m256d t0 = _mm256_permute2f128_pd(lo, hi, 0x20);
    __m256d t1 = _mm256_permute2f128_pd(lo, hi, 0x31);
    lo = t0;
    hi = t1;
}
#endif

} // namespace dmk::util

#endif
