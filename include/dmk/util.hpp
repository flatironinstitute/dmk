#ifndef UTIL_HPP
#define UTIL_HPP

#include <dmk/omp_wrapper.hpp>
#include <dmk/types.hpp>
#include <sctl.hpp>

#include <array>
#include <cmath>
#include <string_view>
#include <type_traits>

#ifndef __cpp_lib_math_special_functions
#include <vendor/bessel.hpp>
#endif

#ifndef __cpp_lib_math_special_functions
#include <vendor/bessel.hpp>
#endif

namespace dmk::util {
template <class...>
constexpr std::false_type always_false{};

static inline auto cyl_bessel_k(auto nu, auto x) {
#ifdef __cpp_lib_math_special_functions
    return std::cyl_bessel_k(nu, x);
#else
    return bessel::cyl_k(nu, x);
#endif
}

static inline auto cyl_bessel_j(auto nu, auto x) {
#ifdef __cpp_lib_math_special_functions
    return std::cyl_bessel_j(nu, x);
#else
    return bessel::cyl_j(nu, x);
#endif
}

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

constexpr std::array<std::string_view, 4> ikernel_names = {
    "DMK_YUKAWA",
    "DMK_LAPLACE",
    "DMK_SQRT_LAPLACE",
    "DMK_STOKESLET",
};

constexpr std::array<std::string_view, 5> return_names = {
    "DMK_POTENTIAL", "DMK_POTENTIAL_GRAD", "DMK_POTENTIAL_GRAD_HESSIAN", "DMK_VELOCITY", "DMK_VELOCITY_PRESSURE",
};

constexpr std::string_view to_string(dmk_ikernel k) noexcept {
    auto idx = static_cast<int>(k);
    if (idx >= 0 && idx < static_cast<int>(ikernel_names.size()))
        return ikernel_names[idx];
    return "DMK_IKERNEL_UNKNOWN";
}

constexpr std::string_view to_string(dmk_eval_type k) noexcept {
    auto idx = static_cast<int>(k) - 1;
    if (idx >= 0 && idx < static_cast<int>(ikernel_names.size()))
        return return_names[idx];
    return "DMK_RETURN_TYPE_UNKNOWN";
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
    (void)((Is + 5 == n_order ? (result = &dmk::util::dot_product<T, Is + 5>, true) : false) || ...);
    if (!result)
        throw std::runtime_error("Invalid order " + std::to_string(n_order));
    return result;
}

template <typename T>
inline auto get_opt_dot(int n_order) {
    return get_opt_dot_impl<T>(n_order, std::make_integer_sequence<int, 41>{});
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

inline void init_test_data(int n_dim, int nd, int n_src, int n_trg, bool uniform, bool set_fixed_charges, auto &r_src,
                           auto &r_trg, auto &rnormal, auto &charges, long seed) {
    using Real = std::decay_t<decltype(r_src)>::value_type;
    size_to(r_src, n_dim * n_src);
    size_to(r_trg, n_dim * n_trg);
    size_to(charges, nd * n_src);
    size_to(rnormal, n_dim * n_src);

    double rin = 0.45;
    double wrig = 0.12;
    double rwig = 0;
    int nwig = 6;
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> rng;

    for (int i = 0; i < n_src; ++i) {
        if (!uniform) {
            if (n_dim == 2) {
                const double phi = rng(eng) * 2 * M_PI;
                r_src[i * 2 + 0] = 0.5 * (cos(phi) + 1.0);
                r_src[i * 2 + 1] = 0.5 * (sin(phi) + 1.0);
            }
            if (n_dim == 3) {
                double theta = rng(eng) * M_PI;
                double rr = rin + rwig * cos(nwig * theta);
                double ct = cos(theta);
                double st = sin(theta);
                double phi = rng(eng) * 2 * M_PI;
                double cp = cos(phi);
                double sp = sin(phi);

                r_src[i * 3 + 0] = rr * st * cp + 0.5;
                r_src[i * 3 + 1] = rr * st * sp + 0.5;
                r_src[i * 3 + 2] = rr * ct + 0.5;
            }
        } else {
            for (int j = 0; j < n_dim; ++j)
                r_src[i * n_dim + j] = rng(eng);
        }

        for (int j = 0; j < n_dim; ++j)
            rnormal[i * n_dim + j] = rng(eng);

        for (int j = 0; j < nd; ++j) {
            charges[i * nd + j] = rng(eng) - 0.5;
        }
    }

    for (int i_trg = 0; i_trg < n_trg; ++i_trg) {
        if (!uniform) {
            if (n_dim == 2) {
                double phi = rng(eng) * 2 * M_PI;
                r_trg[i_trg * 2 + 0] = 0.5 * (cos(phi) + 1.0);
                r_trg[i_trg * 2 + 1] = 0.5 * (sin(phi) + 1.0);
            }
            if (n_dim == 3) {
                double theta = rng(eng) * M_PI;
                double rr = rin + rwig * cos(nwig * theta);
                double ct = cos(theta);
                double st = sin(theta);
                double phi = rng(eng) * 2 * M_PI;
                double cp = cos(phi);
                double sp = sin(phi);

                r_trg[i_trg * 3 + 0] = rr * st * cp + 0.5;
                r_trg[i_trg * 3 + 1] = rr * st * sp + 0.5;
                r_trg[i_trg * 3 + 2] = rr * ct + 0.5;
            }
        } else {
            for (int j = 0; j < n_dim; ++j)
                r_trg[i_trg * n_dim + j] = rng(eng);
        }
    }

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
