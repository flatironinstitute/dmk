#pragma once

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include <xsimd/xsimd.hpp>

// ============================================================
// TYPE ALIASES (replacing FINUFFT's BIGINT/UBIGINT macros)
// ============================================================
using BIGINT  = int64_t;
using UBIGINT = uint64_t;

// ============================================================
// CHANGED: replacing MY_OMP_GET_THREAD_NUM() macro
// ============================================================
inline int my_omp_get_thread_num() { return omp_get_thread_num(); }

// ============================================================
// fold_rescale: copied verbatim from FINUFFT utils.hpp
// Maps coordinate x from [-pi, pi) to [0, N)
// Works for both scalar T and xsimd::batch<T> via using namespace xsimd
// ============================================================
static constexpr double INV_2PI = 0.15915494309189535; // 1/(2*pi)

template<typename T>
inline T fold_rescale(const T x, const UBIGINT N) noexcept {
    using namespace std;
    using namespace xsimd;
    const T result = fma(x, T(INV_2PI), T(0.5)); // x/(2pi) + 0.5
    return (result - floor(result)) * T(N);
}

// ============================================================
// bin_sort_multithread_impl
// Copied verbatim from FINUFFT spread.hpp anonymous namespace.
// CHANGED:
//   - removed "using namespace finufft::spreadinterp"
//   - MY_OMP_GET_THREAD_NUM() -> my_omp_get_thread_num()
//   - BIGINT/UBIGINT replaced by type aliases above
// Everything else is identical to FINUFFT.
// ============================================================
template<typename T, int ndims>
inline void bin_sort_multithread_impl(
    std::vector<BIGINT> &ret, UBIGINT M,
    const T *kx, const T *ky, const T *kz,
    UBIGINT N1, UBIGINT N2, UBIGINT N3,
    double bin_size_x, double bin_size_y, double bin_size_z,
    int nthr)
{
    static_assert(ndims >= 1 && ndims <= 3, "ndims must be 1, 2, or 3");
    using simd_type                 = xsimd::batch<T>;
    using arch_t                    = typename simd_type::arch_type;
    static constexpr auto simd_size = simd_type::size;
    static constexpr auto alignment = arch_t::alignment();

    static constexpr auto to_array = [](const auto &vec) constexpr noexcept {
        using VT = decltype(std::decay_t<decltype(vec)>());
        alignas(alignment) std::array<typename VT::value_type, VT::size> arr{};
        vec.store_aligned(arr.data());
        return arr;
    };

    const auto nbins1             = BIGINT(T(N1) / bin_size_x + 1);
    const auto nbins2             = ndims > 1 ? BIGINT(T(N2) / bin_size_y + 1) : 1;
    const auto nbins3             = ndims > 2 ? BIGINT(T(N3) / bin_size_z + 1) : 1;
    const auto nbins              = nbins1 * nbins2 * nbins3;
    const auto inv_bin_size_x_vec = simd_type(1.0 / bin_size_x);
    const auto inv_bin_size_y_vec = simd_type(1.0 / bin_size_y);
    const auto inv_bin_size_z_vec = simd_type(1.0 / bin_size_z);
    const auto inv_bin_size_x     = T(1.0 / bin_size_x);
    const auto inv_bin_size_y     = T(1.0 / bin_size_y);
    const auto inv_bin_size_z     = T(1.0 / bin_size_z);

    auto compute_bins = [&](UBIGINT offset) {
        const auto i1 = xsimd::to_int(
            fold_rescale(simd_type::load_unaligned(kx + offset), N1) * inv_bin_size_x_vec);
        auto bin = i1;
        if constexpr (ndims > 1) {
            const auto i2 = xsimd::to_int(
                fold_rescale(simd_type::load_unaligned(ky + offset), N2) * inv_bin_size_y_vec);
            bin = i1 + nbins1 * i2;
        }
        if constexpr (ndims > 2) {
            const auto i3 = xsimd::to_int(
                fold_rescale(simd_type::load_unaligned(kz + offset), N3) * inv_bin_size_z_vec);
            bin = bin + nbins1 * nbins2 * i3;
        }
        return bin;
    };

    auto compute_bin_scalar = [&](UBIGINT idx) {
        auto bin = BIGINT(fold_rescale<T>(kx[idx], N1) * inv_bin_size_x);
        if constexpr (ndims > 1)
            bin += nbins1 * BIGINT(fold_rescale<T>(ky[idx], N2) * inv_bin_size_y);
        if constexpr (ndims > 2)
            bin += nbins1 * nbins2 * BIGINT(fold_rescale<T>(kz[idx], N3) * inv_bin_size_z);
        return bin;
    };

    if (nthr == 0) fprintf(stderr, "[%s] nthr (%d) must be positive!\n", __func__, nthr);
    int nt = (int)std::min(M, UBIGINT(nthr));
    std::vector<UBIGINT> brk(nt + 1);
    for (int t = 0; t <= nt; ++t) brk[t] = (UBIGINT)(0.5 + M * t / (double)nt);

    std::vector<std::vector<uint32_t>> counts(nt);
    std::vector<uint32_t> bin_offset(nbins);
    std::vector<uint32_t> thread_totals(nt);

#pragma omp parallel num_threads(nt)
    {
        const int t            = my_omp_get_thread_num(); // CHANGED from MY_OMP_GET_THREAD_NUM()
        const auto chunk_start = brk[t];
        const auto chunk_end   = brk[t + 1];
        const auto chunk_simd  =
            chunk_start + ((chunk_end - chunk_start) & UBIGINT(-simd_size));

        counts[t].resize(nbins, 0);
        auto &my_counts = counts[t];

        UBIGINT i;
        for (i = chunk_start; i < chunk_simd; i += simd_size) {
            const auto bin       = compute_bins(i);
            const auto bin_array = to_array(bin);
            for (std::size_t j = 0; j < simd_size; ++j) ++my_counts[bin_array[j]];
        }
        for (; i < chunk_end; i++) ++my_counts[compute_bin_scalar(i)];

#pragma omp barrier

        const BIGINT bin_chunk = (nbins + nt - 1) / nt;
        const BIGINT bin_start = t * bin_chunk;
        const BIGINT bin_end   = std::min(bin_start + bin_chunk, nbins);
        uint32_t running       = 0;
        for (BIGINT b = bin_start; b < bin_end; ++b) {
            uint32_t total = 0;
            for (int tt = 0; tt < nt; ++tt) total += counts[tt][b];
            bin_offset[b] = running;
            running += total;
        }
        thread_totals[t] = running;

#pragma omp barrier

#pragma omp single
        std::exclusive_scan(thread_totals.begin(), thread_totals.end(),
                            thread_totals.begin(), uint32_t{0});

        const uint32_t thread_prefix = thread_totals[t];
        for (BIGINT b = bin_start; b < bin_end; ++b) {
            uint32_t off = bin_offset[b] + thread_prefix;
            for (int tt = 0; tt < nt; ++tt) {
                uint32_t tmp  = counts[tt][b];
                counts[tt][b] = off;
                off += tmp;
            }
        }

#pragma omp barrier

        for (i = chunk_start; i < chunk_simd; i += simd_size) {
            const auto bin       = compute_bins(i);
            const auto bin_array = to_array(bin);
            for (std::size_t j = 0; j < simd_size; ++j) {
                ret[my_counts[bin_array[j]]] = BIGINT(j + i);
                ++my_counts[bin_array[j]];
            }
        }
        for (; i < chunk_end; i++) {
            const auto bin      = compute_bin_scalar(i);
            ret[my_counts[bin]] = BIGINT(i);
            ++my_counts[bin];
        }
    }
}