#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dmk.h>
#include <dmk/aot_kernels.hpp>
#include <dmk/direct.hpp>
#include <dmk/esp.hpp>
#include <dmk/prolate.hpp>
#include <dmk/prolate0_fun.hpp>
#include <dmk/types.hpp>
#include <dmk/util.hpp>
#include <ducc0/fft/fft.h>
#include <finufft.h>
#include <finufft_common/kernel.h>
#include <omp.h>
#include <sctl.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

// Row-major flattening of a DIM-dimensional grid multi-index (matches the historical
// ix*n_f*n_f + iy*n_f + iz for DIM=3).
template <int DIM>
static inline int grid_idx(const std::array<int, DIM> &idx, int n_f) {
    int r = 0;
    for (int d = 0; d < DIM; ++d)
        r = r * n_f + idx[d];
    return r;
}

template <typename Real, int DIM>
static void fftn(const std::vector<std::complex<Real>> &in, std::vector<std::complex<Real>> &out, int n) {
    out = in;
    const std::vector<size_t> shape(DIM, static_cast<size_t>(n));
    std::vector<size_t> axes(DIM);
    for (int d = 0; d < DIM; ++d)
        axes[d] = d;
    ducc0::vfmav<std::complex<Real>> v(out.data(), shape);
    ducc0::c2c(v, v, axes, true, Real(1.0));
}

template <typename Real, int DIM>
static void ifftn(const std::vector<std::complex<Real>> &in, std::vector<std::complex<Real>> &out, int n) {
    out = in;
    const std::vector<size_t> shape(DIM, static_cast<size_t>(n));
    std::vector<size_t> axes(DIM);
    for (int d = 0; d < DIM; ++d)
        axes[d] = d;
    ducc0::vfmav<std::complex<Real>> v(out.data(), shape);
    Real fct = Real(1.0);
    for (int d = 0; d < DIM; ++d)
        fct /= Real(n);
    ducc0::c2c(v, v, axes, false, fct);
}

namespace dmk {

PSWFKernel::PSWFKernel(double eps_, double c_, int lenw) : eps(eps_), c(c_) {
    pswf = dmk::Prolate0Fun(c, lenw);

    scale = 1.0 / pswf.eval_val(0.0);

    double mu = pswf.rlam20 / M_PI;
    lambda0 = std::sqrt(2.0 * M_PI * mu / c);

    c0 = pswf.int_eval(1.0) * scale;

    int nc = 24;
    auto pswf_lambda = [&](double x) { return pswf.eval_val(std::abs(x)) * scale; };
    pswf_poly_coeffs = finufft::kernel::poly_fit<double>(pswf_lambda, nc);

    nc = 20;
    auto pswf_int_lambda = [&](double t) { return pswf.int_eval(t) * scale; };
    pswf_int_poly_coeffs = finufft::kernel::poly_fit<double>(pswf_int_lambda, nc);
}

template <typename Real, int DIM>
static inline std::array<int, DIM> particle_cell(const Vec3T<Real, DIM> &r, Real L, int n_cells) {
    const Real cell_size = L / n_cells;
    std::array<int, DIM> ci;
    for (int d = 0; d < DIM; ++d) {
        ci[d] = static_cast<int>(std::floor((r[d] + L / 2) / cell_size)) % n_cells;
        if (ci[d] < 0)
            ci[d] += n_cells;
    }
    return ci;
}

// Row-major cell coordinate -> linear cell index (axis 0 outermost); shared by the cell list and
// the neighbour-cell lookups in short_range.
template <int DIM>
static inline int cell_linear_index(const std::array<int, DIM> &ci, int nc) {
    int c = 0;
    for (int d = 0; d < DIM; ++d)
        c = c * nc + ci[d];
    return c;
}

// Morton (Z-order) key from an integer coordinate, used to spatially sort particles within a cell
// so that consecutive VecLen-runs form geometrically compact tiles. part1by2_64 (3-way bit spread)
// serves DIM=3; part1by1_64 (2-way spread) serves DIM=2.
static constexpr uint64_t part1by2_64(uint64_t x) {
    x &= 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffffULL;
    x = (x | x << 16) & 0x1f0000ff0000ffULL;
    x = (x | x << 8) & 0x100f00f00f00f00fULL;
    x = (x | x << 4) & 0x10c30c30c30c30c3ULL;
    x = (x | x << 2) & 0x1249249249249249ULL;
    return x;
}
static constexpr uint64_t part1by1_64(uint64_t x) {
    x &= 0xffffffffULL;
    x = (x | x << 16) & 0x0000ffff0000ffffULL;
    x = (x | x << 8) & 0x00ff00ff00ff00ffULL;
    x = (x | x << 4) & 0x0f0f0f0f0f0f0f0fULL;
    x = (x | x << 2) & 0x3333333333333333ULL;
    x = (x | x << 1) & 0x5555555555555555ULL;
    return x;
}
// 8-bit-chunk spread tables: kSpreadN[v] = partNby._64(v). A kMortonBits (<=21) coordinate is spread
// via three chunk lookups instead of the dependent shift/mask chain.
inline constexpr std::array<uint64_t, 256> kSpread3 = [] {
    std::array<uint64_t, 256> t{};
    for (int v = 0; v < 256; ++v)
        t[v] = part1by2_64(uint64_t(v));
    return t;
}();
inline constexpr std::array<uint64_t, 256> kSpread2 = [] {
    std::array<uint64_t, 256> t{};
    for (int v = 0; v < 256; ++v)
        t[v] = part1by1_64(uint64_t(v));
    return t;
}();
template <int DIM>
static inline uint64_t morton(const std::array<uint64_t, DIM> &c) {
    if constexpr (DIM == 3) {
        auto s = [](uint64_t x) { // part1by2_64(x) for x < 2^21, via 8+8+5-bit chunks
            return kSpread3[x & 0xff] | (kSpread3[(x >> 8) & 0xff] << 24) | (kSpread3[(x >> 16) & 0x1f] << 48);
        };
        return (s(c[0]) << 2) | (s(c[1]) << 1) | s(c[2]);
    } else if constexpr (DIM == 2) {
        auto s = [](uint64_t x) { // part1by1_64(x) for x < 2^21, via 8+8+5-bit chunks
            return kSpread2[x & 0xff] | (kSpread2[(x >> 8) & 0xff] << 16) | (kSpread2[(x >> 16) & 0x1f] << 32);
        };
        return (s(c[0]) << 1) | s(c[1]);
    }
}

// Cell list: particles sorted into DIM-dimensional cubic cells for cache-friendly traversal.
template <typename Real, int DIM>
struct CellList {
    int n_cells;
    std::vector<int> cell_start; // size n_cells^DIM + 1 (exclusive prefix sum)
    std::vector<Real> rs, qs;    // particle data reordered by cell
    std::vector<int> orig;       // orig[slot] = original particle index
};

// Scratch buffers reused across cells by the within-cell spatial sorts.
template <typename Real, int DIM>
struct SortScratch {
    std::vector<Real> tr, tq;                          // reordered coords / charges
    std::vector<int> to;                               // reordered origin indices
    std::vector<int> key, off;                         // bins: per-particle bucket key / bucket offsets
    std::vector<std::pair<uint64_t, int>> mkey, mkey2; // morton: (code, local index) pairs, radix ping-pong
};

// Reorder particles [b, b+len) by z-order within the cell (lower corner lo, width h), tightening
// every VecLen-run's AABB. Tightest tiles, but a comparison sort per cell.
template <typename Real, int DIM>
static void sort_cell_morton(CellList<Real, DIM> &cl, int b, int len, const std::array<Real, DIM> &lo, Real h,
                             SortScratch<Real, DIM> &s) {
    // At hundreds of points per cell, kMortonBits levels/axis fully resolve the cell (each point
    // lands in its own sub-box), so the tiles stay as tight as a fine sort while the code fits in
    // DIM*kMortonBits <= 16 bits -- sortable with a 2-pass byte radix (no comparisons/mispredicts).
    constexpr int kMortonBits = 16 / DIM;
    constexpr int n_pass = (DIM * kMortonBits + 7) / 8;
    const Real q = Real(1 << kMortonBits) / h;
    s.mkey.resize(len);
    s.mkey2.resize(len);
    s.tr.resize(DIM * len);
    s.tq.resize(len);
    s.to.resize(len);
    for (int i = 0; i < len; ++i) {
        const int slot = b + i;
        std::array<uint64_t, DIM> qc;
        for (int d = 0; d < DIM; ++d)
            qc[d] = uint32_t(
                std::min(Real((1 << kMortonBits) - 1), std::max(Real(0), (cl.rs[DIM * slot + d] - lo[d]) * q)));
        s.mkey[i] = {morton<DIM>(qc), i};
    }

    // LSD byte radix on the Morton code, ping-ponging between mkey and mkey2.
    auto *src = &s.mkey, *dst = &s.mkey2;
    for (int p = 0; p < n_pass; ++p) {
        const int sh = 8 * p;
        int hist[256] = {0};
        for (int i = 0; i < len; ++i)
            ++hist[((*src)[i].first >> sh) & 0xff];
        int acc = 0;
        for (int k = 0; k < 256; ++k) {
            const int cnt = hist[k];
            hist[k] = acc;
            acc += cnt;
        }
        for (int i = 0; i < len; ++i) {
            const int dgt = ((*src)[i].first >> sh) & 0xff;
            (*dst)[hist[dgt]++] = (*src)[i];
        }
        std::swap(src, dst);
    }
    const auto &sorted = *src; // final result buffer after the last swap
    for (int i = 0; i < len; ++i) {
        const int slot = b + sorted[i].second;
        for (int d = 0; d < DIM; ++d)
            s.tr[DIM * i + d] = cl.rs[DIM * slot + d];
        s.tq[i] = cl.qs[slot];
        s.to[i] = cl.orig[slot];
    }
    for (int i = 0; i < len; ++i) {
        for (int d = 0; d < DIM; ++d)
            cl.rs[DIM * (b + i) + d] = s.tr[DIM * i + d];
        cl.qs[b + i] = s.tq[i];
        cl.orig[b + i] = s.to[i];
    }
}

// Counting sort of the same particles into bins^DIM spatial sub-boxes (bins=2 -> octants/quadrants).
// No comparisons; looser tiles than morton but cheaper to build.
template <typename Real, int DIM>
static void sort_cell_bins(CellList<Real, DIM> &cl, int b, int len, const std::array<Real, DIM> &lo, Real h, int bins,
                           SortScratch<Real, DIM> &s) {
    int nbuckets = 1;
    for (int d = 0; d < DIM; ++d)
        nbuckets *= bins;
    const Real scale = Real(bins) / h;
    s.key.resize(len);
    s.off.assign(nbuckets, 0);
    s.tr.resize(DIM * len);
    s.tq.resize(len);
    s.to.resize(len);
    for (int i = 0; i < len; ++i) {
        const int slot = b + i;
        std::array<int, DIM> bidx;
        for (int d = 0; d < DIM; ++d)
            bidx[d] = std::min(bins - 1, std::max(0, int((cl.rs[DIM * slot + d] - lo[d]) * scale)));
        int key = 0;
        for (int d = DIM - 1; d >= 0; --d)
            key = key * bins + bidx[d];
        s.key[i] = key;
        ++s.off[key];
    }
    int acc = 0;
    for (int k = 0; k < nbuckets; ++k) {
        const int cnt = s.off[k];
        s.off[k] = acc;
        acc += cnt;
    }
    for (int i = 0; i < len; ++i) {
        const int slot = b + i;
        const int dst = s.off[s.key[i]]++;
        for (int d = 0; d < DIM; ++d)
            s.tr[DIM * dst + d] = cl.rs[DIM * slot + d];
        s.tq[dst] = cl.qs[slot];
        s.to[dst] = cl.orig[slot];
    }
    for (int i = 0; i < len; ++i) {
        for (int d = 0; d < DIM; ++d)
            cl.rs[DIM * (b + i) + d] = s.tr[DIM * i + d];
        cl.qs[b + i] = s.tq[i];
        cl.orig[b + i] = s.to[i];
    }
}

template <typename Real, int DIM>
inline CellList<Real, DIM> build_cell_list(const Real *r_src, const Real *charges, int n, int nc,
                                           const pdmk_esp_params &params, int min_sort_len = 2) {
    sctl::Profile::Scoped profile("build_cell_list");
    CellList<Real, DIM> cl;
    cl.n_cells = nc;
    int ncells = 1;
    for (int d = 0; d < DIM; ++d)
        ncells *= nc;
    const Real L_r = Real(params.L);

    auto cell_of = [&](const Real *r) {
        std::array<Real, DIM> rr;
        for (int d = 0; d < DIM; ++d)
            rr[d] = r[d];
        return cell_linear_index<DIM>(particle_cell<Real, DIM>(rr, L_r, nc), nc);
    };

    // pass 1: count per cell
    std::vector<int> count(ncells, 0), cidx(n);
    for (int j = 0; j < n; ++j) {
        int c = cell_of(r_src + DIM * j);
        cidx[j] = c;
        ++count[c];
    }

    // pass 2: exclusive prefix sum -> cell start offsets
    cl.cell_start.assign(ncells + 1, 0);
    for (int c = 0; c < ncells; ++c)
        cl.cell_start[c + 1] = cl.cell_start[c] + count[c];

    // pass 3: scatter into sorted arrays
    cl.rs.resize(DIM * n);
    cl.qs.resize(n);
    cl.orig.resize(n);
    std::vector<int> cursor(cl.cell_start.begin(), cl.cell_start.end() - 1);
    for (int j = 0; j < n; ++j) {
        int slot = cursor[cidx[j]]++;
        for (int d = 0; d < DIM; ++d)
            cl.rs[DIM * slot + d] = r_src[DIM * j + d];
        cl.qs[slot] = charges[j];
        cl.orig[slot] = j;
    }

    // pass 4: spatially reorder particles within each cell so consecutive VecLen-runs form compact
    // tiles, enabling the geometric source pruning in short_range. esp_morton() picks the z-order
    // sort; otherwise the cheaper bins^DIM counting sort. Both keep the tile count fixed and only
    // tighten each tile's extent.
    if (esp_spatial_sort(params)) {
        sctl::Profile::Scoped sort("spatial_sort");
        const Real h = L_r / Real(nc);
        const Real half_L = L_r / Real(2);
        const bool morton_sort = esp_morton(params);
        SortScratch<Real, DIM> s;
        for (int c = 0; c < ncells; ++c) {
            const int b = cl.cell_start[c], e = cl.cell_start[c + 1], len = e - b;
            if (len <= min_sort_len)
                continue;
            std::array<int, DIM> ci_axes;
            int rem = c;
            for (int d = DIM - 1; d >= 0; --d) {
                ci_axes[d] = rem % nc;
                rem /= nc;
            }
            std::array<Real, DIM> lo;
            for (int d = 0; d < DIM; ++d)
                lo[d] = Real(ci_axes[d]) * h - half_L; // cell lower corner
            if (morton_sort)
                sort_cell_morton<Real, DIM>(cl, b, len, lo, h, s);
            else
                sort_cell_bins<Real, DIM>(cl, b, len, lo, h, params.esp_bins, s);
        }
    }
    return std::move(cl);
}

// Shared read-only state handed to each short-range method. The four methods differ only in how
// they enumerate and cull source/target pairs; everything below is common.
template <typename Real, int DIM>
struct SRCtx {
    const CellList<Real, DIM> &cl;
    const int *nbc_tab;  // nbc_tab[c*3+d] = neighbour cell coord for delta d in {-1,0,+1}
    const Real *off_tab; // off_tab[c*3+d] = periodic image shift for that neighbour
    const residual_evaluator_func<Real> &evaluator;
    const residual_evaluator_range_func<Real> &range_evaluator;
    Real r_c_sq, rsc, cen;
    int nc, out_dim, stile;
    Real *pg_sorted; // interleaved [pot] or [pot, d/dx, ...] accumulator (out_dim = 1+DIM), cell-sorted order
};

// Enumerate the pow(3,DIM) neighbour-cell offsets around a home cell, given per-axis neighbour-cell
// index / periodic-shift slices (each length 3, indexed by delta in {0,1,2} <-> {-1,0,+1}). Calls
// fn(digit, nb_index, shift) for every neighbour: `digit` is the DIM-length delta tuple (each in
// {0,1,2}), `nb_index` is the neighbour's linear cell index, `shift` is its per-axis periodic offset.
template <typename Real, int DIM, typename Fn>
static inline void for_each_neighbor(const std::array<const int *, DIM> &nbc, const std::array<const Real *, DIM> &off,
                                     int nc, Fn &&fn) {
    constexpr int n_nbr = [] {
        int p = 1;
        for (int d = 0; d < DIM; ++d)
            p *= 3;
        return p;
    }();
    for (int lin = 0; lin < n_nbr; ++lin) {
        std::array<int, DIM> digit, nb_axes;
        std::array<Real, DIM> shift;
        int rem = lin;
        for (int d = DIM - 1; d >= 0; --d) {
            digit[d] = rem % 3;
            rem /= 3;
            nb_axes[d] = nbc[d][digit[d]];
            shift[d] = off[d][digit[d]];
        }
        fn(digit, cell_linear_index<DIM>(nb_axes, nc), shift);
    }
}

// True for the forward half of the pow(3,DIM)-1 non-centre neighbour offsets: the delta tuple is
// lexicographically greater than the centre (delta=0 on every axis), axis 0 most significant. Used
// by short_range_n3l so each unordered cross-cell pair is visited exactly once.
template <int DIM>
static inline bool is_forward(const std::array<int, DIM> &digit) {
    for (int d = 0; d < DIM; ++d) {
        if (digit[d] > 1)
            return true;
        if (digit[d] < 1)
            return false;
    }
    return false; // the centre cell itself
}

// Dense path: gather all pow(3,DIM) neighbour cells and evaluate the full block (no pruning).
template <typename Real, int DIM, int VecLen>
static void short_range_dense(const SRCtx<Real, DIM> &ctx) {
    const auto &cl = ctx.cl;
    const int nc = ctx.nc, out_dim = ctx.out_dim;
    const Real rsc = ctx.rsc, cen = ctx.cen, r_c_sq = ctx.r_c_sq;
    int ncells = 1;
    for (int d = 0; d < DIM; ++d)
        ncells *= nc;
#pragma omp parallel
    {
        std::vector<Real> r_src_g, charge_g;
        // Flat loop over the DIM-dimensional cell grid: OpenMP collapse(N) needs a compile-time N,
        // so decode the linear cell index into per-axis coordinates instead (dynamic scheduling
        // over all cells is unaffected).
#pragma omp for schedule(dynamic)
        for (int home = 0; home < ncells; ++home) {
            const int hbeg = cl.cell_start[home], hend = cl.cell_start[home + 1];
            const int n_trg = hend - hbeg;
            if (n_trg == 0)
                continue;

            std::array<int, DIM> home_axes;
            int rem = home;
            for (int d = DIM - 1; d >= 0; --d) {
                home_axes[d] = rem % nc;
                rem /= nc;
            }
            std::array<const int *, DIM> nbc_axes;
            std::array<const Real *, DIM> off_axes;
            for (int d = 0; d < DIM; ++d) {
                nbc_axes[d] = ctx.nbc_tab + home_axes[d] * 3;
                off_axes[d] = ctx.off_tab + home_axes[d] * 3;
            }

            const Real *__restrict__ r_trg_ptr = cl.rs.data() + DIM * hbeg;

            int n_src = 0;
            for_each_neighbor<Real, DIM>(nbc_axes, off_axes, nc, [&](const auto &, int nb, const auto &) {
                n_src += cl.cell_start[nb + 1] - cl.cell_start[nb];
            });

            r_src_g.resize(DIM * n_src);
            charge_g.resize(n_src);
            Real *__restrict__ r_src_ptr = r_src_g.data();
            Real *__restrict__ charge_ptr = charge_g.data();
            int r_i = 0, c_i = 0;
            for_each_neighbor<Real, DIM>(nbc_axes, off_axes, nc, [&](const auto &, int nb, const auto &shift) {
                const int nbeg = cl.cell_start[nb], nend = cl.cell_start[nb + 1];
                for (int b = nbeg; b < nend; ++b) {
                    for (int d = 0; d < DIM; ++d)
                        r_src_ptr[r_i++] = cl.rs[b * DIM + d] + shift[d];
                    charge_ptr[c_i++] = cl.qs[b];
                }
            });

            ctx.evaluator(rsc, cen, r_c_sq, Real(0), n_src, r_src_ptr, charge_ptr, nullptr, n_trg, r_trg_ptr,
                          ctx.pg_sorted + out_dim * hbeg);
        }
    }
}

// Sub-cell geometric pruning (tile-vs-tile). Sources and targets are each grouped into VecLen-wide
// tiles; one AABB test per (target-tile, source-tile) pair skips up to VecLen*VecLen interactions.
// Surviving source tiles are passed as disjoint ranges to the range-list evaluator, which reads
// them in-place (the dense kernel's internal d2max mask still enforces the exact cutoff).
template <typename Real, int DIM, int VecLen>
static void short_range_prune_tile(const SRCtx<Real, DIM> &ctx) {
    const auto &cl = ctx.cl;
    const int nc = ctx.nc, out_dim = ctx.out_dim, stile = ctx.stile;
    const Real rsc = ctx.rsc, cen = ctx.cen, r_c_sq = ctx.r_c_sq;
    int ncells = 1;
    for (int d = 0; d < DIM; ++d)
        ncells *= nc;
#pragma omp parallel
    {
        std::vector<Real> r_src_soa, charge_g; // sources gathered SoA (axis-major, stride n_src)
        // Source-tile AABBs, SoA so the tile-vs-tile test vectorizes; d2buf holds its output.
        std::array<std::vector<Real>, DIM> slo, shi;
        std::vector<Real> d2buf;
        std::vector<int> tile_s0, tile_sn; // cell-aligned source tiles (start, length)
        std::vector<int> surv_s0, surv_sn; // surviving source tiles per target-tile
#pragma omp for schedule(dynamic)
        for (int home = 0; home < ncells; ++home) {
            const int hbeg = cl.cell_start[home], hend = cl.cell_start[home + 1];
            const int n_trg = hend - hbeg;
            if (n_trg == 0)
                continue;

            std::array<int, DIM> home_axes;
            int rem = home;
            for (int d = DIM - 1; d >= 0; --d) {
                home_axes[d] = rem % nc;
                rem /= nc;
            }
            std::array<const int *, DIM> nbc_axes;
            std::array<const Real *, DIM> off_axes;
            for (int d = 0; d < DIM; ++d) {
                nbc_axes[d] = ctx.nbc_tab + home_axes[d] * 3;
                off_axes[d] = ctx.off_tab + home_axes[d] * 3;
            }

            const Real *__restrict__ r_trg_ptr = cl.rs.data() + DIM * hbeg;

            int n_src = 0;
            for_each_neighbor<Real, DIM>(nbc_axes, off_axes, nc, [&](const auto &, int nb, const auto &) {
                n_src += cl.cell_start[nb + 1] - cl.cell_start[nb];
            });

            r_src_soa.resize(DIM * n_src);
            charge_g.resize(n_src);
            Real *__restrict__ r_src_ptr = r_src_soa.data();
            Real *__restrict__ charge_ptr = charge_g.data();
            tile_s0.clear();
            tile_sn.clear();
            int c_i = 0;
            for_each_neighbor<Real, DIM>(nbc_axes, off_axes, nc, [&](const auto &, int nb, const auto &shift) {
                const int nbeg = cl.cell_start[nb], nend = cl.cell_start[nb + 1];
                const int seg0 = c_i;
                for (int b = nbeg; b < nend; ++b) {
                    for (int d = 0; d < DIM; ++d)
                        r_src_ptr[d * n_src + c_i] = cl.rs[b * DIM + d] + shift[d];
                    charge_ptr[c_i] = cl.qs[b];
                    ++c_i;
                }
                // Split this cell's contribution into VecLen source tiles that never cross a cell
                // boundary, so each tile's AABB stays compact.
                for (int st = seg0; st < c_i; st += stile) {
                    tile_s0.push_back(st);
                    tile_sn.push_back(std::min(stile, c_i - st));
                }
            });

            const int n_stiles = static_cast<int>(tile_s0.size());
            for (int d = 0; d < DIM; ++d) {
                slo[d].resize(n_stiles);
                shi[d].resize(n_stiles);
            }
            d2buf.resize(n_stiles);
            for (int st = 0; st < n_stiles; ++st) {
                const int s0 = tile_s0[st];
                const int sn = tile_sn[st];
                std::array<Real, DIM> lo, hi;
                for (int d = 0; d < DIM; ++d)
                    lo[d] = hi[d] = r_src_ptr[d * n_src + s0];
                for (int i = 1; i < sn; ++i)
                    for (int d = 0; d < DIM; ++d) {
                        const Real v = r_src_ptr[d * n_src + (s0 + i)];
                        lo[d] = std::min(lo[d], v);
                        hi[d] = std::max(hi[d], v);
                    }
                for (int d = 0; d < DIM; ++d) {
                    slo[d][st] = lo[d];
                    shi[d][st] = hi[d];
                }
            }

            surv_s0.resize(n_stiles);
            surv_sn.resize(n_stiles);
            for (int t0 = 0; t0 < n_trg; t0 += VecLen) {
                const int tn = std::min(VecLen, n_trg - t0);
                const Real *__restrict__ tptr = r_trg_ptr + DIM * t0;

                Real lo[DIM], hi[DIM];
                for (int d = 0; d < DIM; ++d)
                    lo[d] = hi[d] = tptr[d];
                for (int i = 1; i < tn; ++i)
                    for (int d = 0; d < DIM; ++d) {
                        const Real v = tptr[DIM * i + d];
                        lo[d] = std::min(lo[d], v);
                        hi[d] = std::max(hi[d], v);
                    }

                // Branchless squared box-distance of every source tile to this target tile; SoA
                // loads with no gather or branch, so it vectorizes cleanly.
#pragma omp simd
                for (int st = 0; st < n_stiles; ++st) {
                    Real d2 = Real(0);
                    for (int d = 0; d < DIM; ++d) {
                        const Real delta = std::max(Real(0), std::max(slo[d][st] - hi[d], lo[d] - shi[d][st]));
                        d2 += delta * delta;
                    }
                    d2buf[st] = d2;
                }

                // Branchless compaction of surviving source tiles into disjoint ranges: write
                // unconditionally, advance the cursor only on a survivor.
                int *__restrict__ s0_ptr = surv_s0.data();
                int *__restrict__ sn_ptr = surv_sn.data();
                int m = 0;
                for (int st = 0; st < n_stiles; ++st) {
                    s0_ptr[m] = tile_s0[st];
                    sn_ptr[m] = tile_sn[st];
                    m += (d2buf[st] <= r_c_sq);
                }

                ctx.range_evaluator(rsc, cen, r_c_sq, Real(0), n_src, r_src_ptr, charge_ptr, nullptr, m, s0_ptr, sn_ptr,
                                    tn, tptr, ctx.pg_sorted + out_dim * (hbeg + t0), nullptr, nullptr);
            }
        }
    }
}

// Per-source pruning (point-vs-target-box). Repack sources to SoA so the per-source box-distance
// test vectorizes, then for each target tile keep the individual sources within r_c of its AABB and
// feed them to the range evaluator as unit-length ranges -- no gather/copy, and the survivor
// fraction tracks the true in-range fraction far better than tile granularity.
template <typename Real, int DIM, int VecLen>
static void short_range_prune_source(const SRCtx<Real, DIM> &ctx) {
    using RealVec = sctl::Vec<Real, VecLen>;
    const auto &cl = ctx.cl;
    const int nc = ctx.nc, out_dim = ctx.out_dim;
    const Real rsc = ctx.rsc, cen = ctx.cen, r_c_sq = ctx.r_c_sq;
    int ncells = 1;
    for (int d = 0; d < DIM; ++d)
        ncells *= nc;
#pragma omp parallel
    {
        std::vector<Real> r_src_soa, charge_g; // sources gathered SoA (axis-major, stride n_src)
        std::vector<int> surv_s0, surv_sn;
#pragma omp for schedule(dynamic)
        for (int home = 0; home < ncells; ++home) {
            const int hbeg = cl.cell_start[home], hend = cl.cell_start[home + 1];
            const int n_trg = hend - hbeg;
            if (n_trg == 0)
                continue;

            std::array<int, DIM> home_axes;
            int rem = home;
            for (int d = DIM - 1; d >= 0; --d) {
                home_axes[d] = rem % nc;
                rem /= nc;
            }
            std::array<const int *, DIM> nbc_axes;
            std::array<const Real *, DIM> off_axes;
            for (int d = 0; d < DIM; ++d) {
                nbc_axes[d] = ctx.nbc_tab + home_axes[d] * 3;
                off_axes[d] = ctx.off_tab + home_axes[d] * 3;
            }

            const Real *__restrict__ r_trg_ptr = cl.rs.data() + DIM * hbeg;

            int n_src = 0;
            for_each_neighbor<Real, DIM>(nbc_axes, off_axes, nc, [&](const auto &, int nb, const auto &) {
                n_src += cl.cell_start[nb + 1] - cl.cell_start[nb];
            });

            // Gather neighbour sources straight into SoA (axis d block at r_src_ptr + d*n_src): the
            // per-source box-distance cull reads each axis contiguously and the evaluator reads the
            // same layout, so no separate AoS->SoA repack is needed.
            r_src_soa.resize(DIM * n_src);
            charge_g.resize(n_src);
            Real *__restrict__ r_src_ptr = r_src_soa.data();
            Real *__restrict__ charge_ptr = charge_g.data();
            int c_i = 0;
            for_each_neighbor<Real, DIM>(nbc_axes, off_axes, nc, [&](const auto &, int nb, const auto &shift) {
                const int nbeg = cl.cell_start[nb], nend = cl.cell_start[nb + 1];
                for (int b = nbeg; b < nend; ++b) {
                    for (int d = 0; d < DIM; ++d)
                        r_src_ptr[d * n_src + c_i] = cl.rs[b * DIM + d] + shift[d];
                    charge_ptr[c_i] = cl.qs[b];
                    ++c_i;
                }
            });

            surv_s0.resize(n_src);
            if (static_cast<int>(surv_sn.size()) < n_src)
                surv_sn.resize(n_src, 1); // unit lengths; all entries stay 1 across cells
            int *__restrict__ idx_ptr = surv_s0.data();

            for (int t0 = 0; t0 < n_trg; t0 += VecLen) {
                const int tn = std::min(VecLen, n_trg - t0);
                const Real *__restrict__ tptr = r_trg_ptr + DIM * t0;

                Real lo[DIM], hi[DIM];
                for (int d = 0; d < DIM; ++d)
                    lo[d] = hi[d] = tptr[d];
                for (int i = 1; i < tn; ++i)
                    for (int d = 0; d < DIM; ++d) {
                        const Real v = tptr[DIM * i + d];
                        lo[d] = std::min(lo[d], v);
                        hi[d] = std::max(hi[d], v);
                    }

                // Fused per-source test + compaction, fully SIMD: box-distance via SCTL Vec,
                // compare to r_c^2, and compress the surviving source indices with a masked
                // vcompress -- no d2 scratch, no scalar filter. Two half-width d2 masks drive one
                // full-width int32 vcompress. Scalar tail only.
                std::array<RealVec, DIM> lov, hiv;
                for (int d = 0; d < DIM; ++d) {
                    lov[d] = RealVec(lo[d]);
                    hiv[d] = RealVec(hi[d]);
                }
                const RealVec zero = RealVec::Zero(), rc2v(r_c_sq);
                auto box_d2 = [&](int i) {
                    RealVec sum = zero;
                    for (int d = 0; d < DIM; ++d) {
                        const RealVec x = RealVec::Load(&r_src_ptr[d * n_src + i]);
                        const RealVec delta = sctl::max(zero, sctl::max(lov[d] - x, x - hiv[d]));
                        sum = sctl::FMA(delta, delta, sum);
                    }
                    return sum;
                };
                int m = 0, s = 0;
                for (; s + 2 * VecLen <= n_src; s += 2 * VecLen)
                    m += sctl::mask_compress_iota_store2(box_d2(s) <= rc2v, box_d2(s + VecLen) <= rc2v, s, idx_ptr + m);
                for (; s + VecLen <= n_src; s += VecLen)
                    m += sctl::mask_compress_iota_store(box_d2(s) <= rc2v, s, idx_ptr + m);
                for (; s < n_src; ++s) {
                    Real d2 = Real(0);
                    for (int d = 0; d < DIM; ++d) {
                        const Real sd = r_src_ptr[d * n_src + s];
                        const Real delta = std::max(Real(0), std::max(lo[d] - sd, sd - hi[d]));
                        d2 += delta * delta;
                    }
                    idx_ptr[m] = s;
                    m += (d2 <= r_c_sq);
                }

                ctx.range_evaluator(rsc, cen, r_c_sq, Real(0), n_src, r_src_ptr, charge_ptr, nullptr, m, idx_ptr,
                                    surv_sn.data(), tn, tptr, ctx.pg_sorted + out_dim * (hbeg + t0), nullptr, nullptr);
            }
        }
    }
}

// Newton's-third-law reciprocal path. Each cross-cell pair is evaluated once (forward half-stencil)
// and scattered to both endpoints; the home cell stays full (both endpoints are targets there, so
// no reciprocal). Home cells are pow(3,DIM)-coloured so reciprocal writes into neighbour cells never
// race. Uses the per-source cull. Requires nc divisible by 3 (enforced by short_range).
template <typename Real, int DIM, int VecLen>
static void short_range_n3l(const SRCtx<Real, DIM> &ctx) {
    using RealVec = sctl::Vec<Real, VecLen>;
    const auto &cl = ctx.cl;
    const int nc = ctx.nc, out_dim = ctx.out_dim;
    const Real rsc = ctx.rsc, cen = ctx.cen, r_c_sq = ctx.r_c_sq;
    constexpr int n_colors = [] {
        int p = 1;
        for (int d = 0; d < DIM; ++d)
            p *= 3;
        return p;
    }();
    const int nb_per_axis = nc / 3; // nc is a multiple of 3, enforced by short_range
    int cells_per_color = 1;
    for (int d = 0; d < DIM; ++d)
        cells_per_color *= nb_per_axis;
#pragma omp parallel
    {
        // Gathered forward-neighbour sources (SoA, axis-major, stride n_fwd), charges, cell-sorted
        // origin indices, the per-source reciprocal accumulator, and the self-block SoA source coords.
        std::vector<Real> src_soa, fwd_soa, fwd_q, pg_src;
        std::vector<int> fwd_origin, surv_s0, surv_sn;

        // Per-source point-vs-target-box cull (same test as prune_source), factored so the self and
        // forward blocks share it. s_soa is axis-major (axis d block at s_soa + d*nsrc). Appends
        // surviving source indices to idx_ptr.
        auto cull_box = [&](const Real *__restrict__ s_soa, int nsrc, const Real lo[DIM], const Real hi[DIM],
                            int *__restrict__ idx_ptr) -> int {
            std::array<RealVec, DIM> lov, hiv;
            for (int d = 0; d < DIM; ++d) {
                lov[d] = RealVec(lo[d]);
                hiv[d] = RealVec(hi[d]);
            }
            const RealVec zero = RealVec::Zero(), rc2v(r_c_sq);
            auto box_d2 = [&](int i) {
                RealVec sum = zero;
                for (int d = 0; d < DIM; ++d) {
                    const RealVec x = RealVec::Load(&s_soa[d * nsrc + i]);
                    const RealVec delta = sctl::max(zero, sctl::max(lov[d] - x, x - hiv[d]));
                    sum = sctl::FMA(delta, delta, sum);
                }
                return sum;
            };
            int m = 0, s = 0;
            for (; s + 2 * VecLen <= nsrc; s += 2 * VecLen)
                m += sctl::mask_compress_iota_store2(box_d2(s) <= rc2v, box_d2(s + VecLen) <= rc2v, s, idx_ptr + m);
            for (; s + VecLen <= nsrc; s += VecLen)
                m += sctl::mask_compress_iota_store(box_d2(s) <= rc2v, s, idx_ptr + m);
            for (; s < nsrc; ++s) {
                Real d2 = Real(0);
                for (int d = 0; d < DIM; ++d) {
                    const Real sd = s_soa[d * nsrc + s];
                    const Real delta = std::max(Real(0), std::max(lo[d] - sd, sd - hi[d]));
                    d2 += delta * delta;
                }
                idx_ptr[m] = s;
                m += (d2 <= r_c_sq);
            }
            return m;
        };
        auto trg_box = [](const Real *tptr, int tn, Real lo[DIM], Real hi[DIM]) {
            for (int d = 0; d < DIM; ++d)
                lo[d] = hi[d] = tptr[d];
            for (int i = 1; i < tn; ++i)
                for (int d = 0; d < DIM; ++d) {
                    const Real v = tptr[DIM * i + d];
                    lo[d] = std::min(lo[d], v);
                    hi[d] = std::max(hi[d], v);
                }
        };

        // n_colors-colour so no two concurrently-processed home cells share a write cell (home +
        // its forward neighbours span the +-1 shell; stride 3 keeps same-colour cells >= 3 apart).
        for (int color = 0; color < n_colors; ++color) {
            std::array<int, DIM> c0;
            int crem = color;
            for (int d = DIM - 1; d >= 0; --d) {
                c0[d] = crem % 3;
                crem /= 3;
            }
            // Enumerate this colour's stride-3-aligned cells directly (idx decodes into per-axis
            // block indices) rather than scanning every cell and filtering, so the pow(3,DIM)-colour
            // pass over all colours still touches each cell exactly once in total.
#pragma omp for schedule(dynamic)
            for (int idx = 0; idx < cells_per_color; ++idx) {
                std::array<int, DIM> home_axes;
                int rem = idx;
                for (int d = DIM - 1; d >= 0; --d) {
                    home_axes[d] = c0[d] + 3 * (rem % nb_per_axis);
                    rem /= nb_per_axis;
                }
                const int home = cell_linear_index<DIM>(home_axes, nc);
                const int hbeg = cl.cell_start[home], hend = cl.cell_start[home + 1];
                const int n_trg = hend - hbeg;
                if (n_trg == 0)
                    continue;

                const Real *__restrict__ r_trg_ptr = cl.rs.data() + DIM * hbeg;
                const Real *__restrict__ q_trg_all = cl.qs.data() + hbeg;
                std::array<const int *, DIM> nbc_axes;
                std::array<const Real *, DIM> off_axes;
                for (int d = 0; d < DIM; ++d) {
                    nbc_axes[d] = ctx.nbc_tab + home_axes[d] * 3;
                    off_axes[d] = ctx.off_tab + home_axes[d] * 3;
                }

                // Count forward-neighbour sources so the surv buffers can be sized once.
                int n_fwd = 0;
                for_each_neighbor<Real, DIM>(nbc_axes, off_axes, nc, [&](const auto &digit, int nb, const auto &) {
                    if (!is_forward<DIM>(digit))
                        return;
                    n_fwd += cl.cell_start[nb + 1] - cl.cell_start[nb];
                });

                const int nmax = std::max(n_trg, n_fwd);
                if (static_cast<int>(surv_s0.size()) < nmax)
                    surv_s0.resize(nmax);
                if (static_cast<int>(surv_sn.size()) < nmax)
                    surv_sn.resize(nmax, 1);
                int *__restrict__ idx_ptr = surv_s0.data();

                // Self block: home cell is its own source, evaluated full (both endpoints are home
                // targets, so no reciprocal). Repack home coords SoA for both the cull and evaluator.
                src_soa.resize(DIM * n_trg);
                for (int s = 0; s < n_trg; ++s)
                    for (int d = 0; d < DIM; ++d)
                        src_soa[d * n_trg + s] = r_trg_ptr[DIM * s + d];
                for (int t0 = 0; t0 < n_trg; t0 += VecLen) {
                    const int tn = std::min(VecLen, n_trg - t0);
                    const Real *__restrict__ tptr = r_trg_ptr + DIM * t0;
                    Real lo[DIM], hi[DIM];
                    trg_box(tptr, tn, lo, hi);
                    const int m = cull_box(src_soa.data(), n_trg, lo, hi, idx_ptr);
                    ctx.range_evaluator(rsc, cen, r_c_sq, Real(0), n_trg, src_soa.data(), q_trg_all, nullptr, m,
                                        idx_ptr, surv_sn.data(), tn, tptr, ctx.pg_sorted + out_dim * (hbeg + t0),
                                        nullptr, nullptr);
                }

                if (n_fwd == 0)
                    continue;

                // Forward block: gather the forward-half-stencil cells straight into SoA (+ charge +
                // origin), then evaluate N3L (forward onto home targets, reciprocal onto pg_src indexed
                // by forward source).
                fwd_soa.resize(DIM * n_fwd);
                fwd_q.resize(n_fwd);
                fwd_origin.resize(n_fwd);
                int c_i = 0;
                for_each_neighbor<Real, DIM>(nbc_axes, off_axes, nc, [&](const auto &digit, int nb, const auto &shift) {
                    if (!is_forward<DIM>(digit))
                        return;
                    const int nbeg = cl.cell_start[nb], nend = cl.cell_start[nb + 1];
                    for (int b = nbeg; b < nend; ++b) {
                        for (int d = 0; d < DIM; ++d)
                            fwd_soa[d * n_fwd + c_i] = cl.rs[b * DIM + d] + shift[d];
                        fwd_q[c_i] = cl.qs[b];
                        fwd_origin[c_i] = b;
                        ++c_i;
                    }
                });
                pg_src.assign(out_dim * n_fwd, Real(0));
                const Real *__restrict__ fwd_q_ptr = fwd_q.data();
                for (int t0 = 0; t0 < n_trg; t0 += VecLen) {
                    const int tn = std::min(VecLen, n_trg - t0);
                    const Real *__restrict__ tptr = r_trg_ptr + DIM * t0;
                    Real lo[DIM], hi[DIM];
                    trg_box(tptr, tn, lo, hi);
                    const int m = cull_box(fwd_soa.data(), n_fwd, lo, hi, idx_ptr);
                    ctx.range_evaluator(rsc, cen, r_c_sq, Real(0), n_fwd, fwd_soa.data(), fwd_q_ptr, nullptr, m,
                                        idx_ptr, surv_sn.data(), tn, tptr, ctx.pg_sorted + out_dim * (hbeg + t0),
                                        q_trg_all + t0, pg_src.data());
                }
                // Scatter reciprocal contributions back onto the forward sources; safe because
                // same-colour home cells write disjoint neighbour cells.
                for (int s = 0; s < n_fwd; ++s) {
                    const int b = fwd_origin[s];
                    for (int k = 0; k < out_dim; ++k)
                        ctx.pg_sorted[out_dim * b + k] += pg_src[out_dim * s + k];
                }
            }
        }
    }
}

// force[d] is the d-th force-component output span (fx, fy, ... for d in [0,DIM)); unused (may be
// empty) when eval_type == DMK_POTENTIAL. Chosen over separate fx/fy/fz parameters so this driver
// and its four strategies stay DIM-generic; esp_eval builds this array from PotForce's spans.
template <typename Real>
template <int DIM>
void EspPlan<Real>::short_range(int n, const Real *r_src, const Real *charges, std::span<Real> pot,
                                std::array<std::span<Real>, DIM> force) {
    if constexpr (DIM != 3)
        throw std::runtime_error("ESP short-range for DIM=2 is not implemented");

    sctl::Profile::Scoped short_range("short_range");
    // pow(3,DIM)-cell stencil requires nc >= 3 so periodic images aren't double-counted
    int nc = static_cast<int>(std::floor(params.L / params.r_c));
    if (nc < 3)
        throw std::runtime_error("short_range_fast requires r_c <= L/3 (nc >= 3)");
    // Stride-3 periodic colouring is conflict-free only if each axis length is divisible by the
    // stride; round nc down to a multiple of 3 (cells grow slightly, still >= r_c, still >= 3).
    if (esp_n3l(params))
        nc -= nc % 3;

    constexpr int MaxVecLen = sctl::DefaultVecLen<Real>();

    CellList<Real, DIM> cl = build_cell_list<Real, DIM>(r_src, charges, n, nc, params, MaxVecLen);

    const Real L = Real(params.L);
    const Real r_c_sq = Real(params.r_c) * Real(params.r_c);
    const bool want_force = (params.eval_type >= DMK_POTENTIAL_GRAD);
    const int out_dim = want_force ? 1 + DIM : 1;

    const Real rsc = Real(2.0 / params.r_c);
    const Real cen = Real(-params.r_c / 2.0);

    // Interleaved [pot] or [pot, d/dx, ...] per particle (out_dim = 1+DIM when forces are wanted),
    // in cell-sorted order.
    std::vector<Real> pg_sorted(out_dim * n, Real(0));

    // For each cell coordinate c in [0, nc) and each delta d in {-1,0,+1} (stored as d=0,1,2):
    // nbc_tab[c*3+d] = neighbor cell index, off_tab[c*3+d] = image shift to subtract.
    // Precomputed once; all DIM axes share the same table since they share nc.
    std::vector<int> nbc_tab(nc * 3);
    std::vector<Real> off_tab(nc * 3);
    for (int c = 0; c < nc; ++c) {
        for (int d = 0; d < 3; ++d) {
            int ci = c + d - 1;
            if (ci < 0) {
                nbc_tab[c * 3 + d] = ci + nc;
                off_tab[c * 3 + d] = -L;
            } else if (ci >= nc) {
                nbc_tab[c * 3 + d] = ci - nc;
                off_tab[c * 3 + d] = L;
            } else {
                nbc_tab[c * 3 + d] = ci;
                off_tab[c * 3 + d] = Real(0);
            }
        }
    }

    const SRCtx<Real, DIM> ctx{cl,
                               nbc_tab.data(),
                               off_tab.data(),
                               evaluator,
                               range_evaluator,
                               r_c_sq,
                               rsc,
                               cen,
                               nc,
                               out_dim,
                               params.esp_stile > 0 ? params.esp_stile : MaxVecLen,
                               pg_sorted.data()};

    // Each method enumerates source/target pairs its own way; they all accumulate into pg_sorted.
    if (esp_n3l(params))
        short_range_n3l<Real, DIM, MaxVecLen>(ctx);
    else if (esp_prune_source(params))
        short_range_prune_source<Real, DIM, MaxVecLen>(ctx);
    else if (esp_prune_tile(params))
        short_range_prune_tile<Real, DIM, MaxVecLen>(ctx);
    else
        short_range_dense<Real, DIM, MaxVecLen>(ctx);

    for (int a = 0; a < n; ++a) {
        const int orig = cl.orig[a];
        pot[orig] += pg_sorted[out_dim * a + 0];
        if (want_force) {
            const Real q = cl.qs[a];
            for (int d = 0; d < DIM; ++d)
                force[d][orig] += -q * pg_sorted[out_dim * a + 1 + d];
        }
    }
}

template <int DIM>
static inline double S_hat(const PSWFKernel &pswf, double r_c, const std::array<double, DIM> &k_vec) {
    double k_mag_sq = 0.0;
    for (int d = 0; d < DIM; ++d)
        k_mag_sq += k_vec[d] * k_vec[d];
    const double k_mag = std::sqrt(k_mag_sq);
    return pswf.pswf_hat(k_mag * r_c) / (2.0 * k_mag * k_mag) / pswf.c0;
}

// The far-field split's 1/k^2 structure (Fourier transform of the Laplacian Green's function) is
// dimension-independent by construction, so this generalizes mechanically to DIM=2 -- no new
// physics needed here (unlike short_range's near-field correction).
template <typename Real>
template <int DIM>
std::vector<double> EspPlan<Real>::precompute_scaling_coefficients() const {
    const int nf = n_f;
    std::vector<int> k_idx(nf);
    for (int i = 0; i < nf; ++i)
        k_idx[i] = (i <= nf / 2) ? i : i - nf;

    // 1-D phi_hat values
    std::vector<double> phi_hat_1d(nf);
    for (int i = 0; i < nf; ++i) {
        double k_vec = 2.0 * M_PI * k_idx[i] / params.L;
        double arg = k_vec * (P * h) / 2.0;
        phi_hat_1d[i] = (P * h / 2.0) * pswf.pswf_hat(arg);
    }

    int ntot = 1;
    double L_pow_dim = 1.0;
    for (int d = 0; d < DIM; ++d) {
        ntot *= nf;
        L_pow_dim *= params.L;
    }

    std::vector<double> p(ntot, 0.0);
    for (int lin = 0; lin < ntot; ++lin) {
        std::array<int, DIM> gidx;
        int rem = lin;
        for (int d = DIM - 1; d >= 0; --d) {
            gidx[d] = rem % nf;
            rem /= nf;
        }
        std::array<double, DIM> k_vec;
        double ph = 1.0;
        bool all_zero = true;
        for (int d = 0; d < DIM; ++d) {
            k_vec[d] = 2.0 * M_PI * k_idx[gidx[d]] / params.L;
            all_zero &= (k_vec[d] == 0.0);
            ph *= phi_hat_1d[gidx[d]];
        }
        if (all_zero)
            continue;

        const double s = S_hat<DIM>(pswf, params.r_c, k_vec);
        p[grid_idx<DIM>(gidx, nf)] = s / (L_pow_dim * ph * ph * static_cast<double>(ntot));
    }
    return p;
}

// Long-range contribution via FINUFFT spreading/interpolation. DIM=3 is exercised by every existing
// test/caller; DIM=2 is unverified scaffolding -- short_range throws for DIM=2 before esp_eval ever
// reaches this function, so the DIM=2 branch below has never actually run.
template <typename Real>
template <int DIM>
void EspPlan<Real>::long_range(int n, const Real *r_src, const Real *charges, std::span<Real> pot,
                               std::array<std::span<Real>, DIM> force) {
    sctl::Profile::Scoped long_range("long_range");
    const bool want_force = (params.eval_type >= DMK_POTENTIAL_GRAD);
    const int nf = n_f;
    int ntot = 1;
    for (int d = 0; d < DIM; ++d)
        ntot *= nf;

    const Real scale = Real(2.0 * M_PI / params.L);
    auto &coord = lr_coord;
    auto &c = lr_c;
    for (int d = 0; d < DIM; ++d)
        coord[d].resize(n);
    c.resize(n);
#pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        for (int d = 0; d < DIM; ++d)
            coord[d][j] = r_src[DIM * j + d] * scale;
        c[j] = {charges[j], Real(0)};
    }

    finufft_opts opts;
    if constexpr (std::is_same_v<Real, float>)
        finufftf_default_opts(&opts);
    else
        finufft_default_opts(&opts);
    opts.spreadinterponly = 1;
    opts.upsampfac = params.sigma;
    const Real tol = Real(pswf.eps);

    // NU points <-> uniform grid, dispatched on both precision (finufft*/finufftf*) and dimension
    // (finufft3d*/finufft2d*, which differ in arity, not just template parameter).
    auto nufft1 = [&](std::complex<Real> *cj, std::complex<Real> *out) {
        int ier = [&]() {
            if constexpr (DIM == 3) {
                if constexpr (std::is_same_v<Real, float>)
                    return finufftf3d1(n, coord[0].data(), coord[1].data(), coord[2].data(), cj, +1, tol, nf, nf, nf,
                                       out, &opts);
                else
                    return finufft3d1(n, coord[0].data(), coord[1].data(), coord[2].data(), cj, +1, tol, nf, nf, nf,
                                      out, &opts);
            } else {
                if constexpr (std::is_same_v<Real, float>)
                    return finufftf2d1(n, coord[0].data(), coord[1].data(), cj, +1, tol, nf, nf, out, &opts);
                else
                    return finufft2d1(n, coord[0].data(), coord[1].data(), cj, +1, tol, nf, nf, out, &opts);
            }
        }();
        if (ier > 1)
            throw std::runtime_error("finufft NUFFT spread failed, ier=" + std::to_string(ier));
    };
    auto nufft2 = [&](std::complex<Real> *cj, const std::complex<Real> *grid) {
        int ier = [&]() {
            if constexpr (DIM == 3) {
                if constexpr (std::is_same_v<Real, float>)
                    return finufftf3d2(n, coord[0].data(), coord[1].data(), coord[2].data(), cj, +1, tol, nf, nf, nf,
                                       grid, &opts);
                else
                    return finufft3d2(n, coord[0].data(), coord[1].data(), coord[2].data(), cj, +1, tol, nf, nf, nf,
                                      grid, &opts);
            } else {
                if constexpr (std::is_same_v<Real, float>)
                    return finufftf2d2(n, coord[0].data(), coord[1].data(), cj, +1, tol, nf, nf, grid, &opts);
                else
                    return finufft2d2(n, coord[0].data(), coord[1].data(), cj, +1, tol, nf, nf, grid, &opts);
            }
        }();
        if (ier > 1)
            throw std::runtime_error("finufft NUFFT interp failed, ier=" + std::to_string(ier));
    };

    // 1. Spread: NU points -> uniform grid (type 1). b is zeroed since spreading only writes grid
    // points near NU sources while the forward FFT below reads all ntot entries.
    auto &b = lr_b;
    b.assign(ntot, std::complex<Real>(0));
    {
        sctl::Profile::Scoped prof("lr_spread");
        nufft1(c.data(), b.data());
    }

    // 2. Forward FFT (in place; b now holds the spectrum)
    {
        sctl::Profile::Scoped prof("lr_fft_fwd");
        fftn<Real, DIM>(b, b, nf);
    }

    // 3. Diagonal scaling: u_hat[0] is the far-field potential spectrum (b * scaling_coeffs). For the
    // force, the gradient spectra follow from the ik method: F = -q*grad(u), grad(u)_hat_k = i*k*u_hat_k,
    // so u_hat[1+axis] = i*k_axis*u_hat[0] -- a complex-by-imaginary product written as a real swap+scale
    // rather than a full std::complex multiply.
    const int out_dim = want_force ? 1 + DIM : 1;
    auto &u_hat = lr_u_hat;
    for (int k = 0; k < out_dim; ++k)
        u_hat[k].resize(ntot);

    if (!want_force) {
#pragma omp parallel for
        for (int idx = 0; idx < ntot; ++idx)
            u_hat[0][idx] = b[idx] * scaling_coeffs[idx];
    } else {
        // Per-axis wavenumbers scale*k_idx. Walking the grid with nested loops makes each grid index a
        // loop counter, avoiding the O(ntot) modulo/division reconstruction of the multi-index.
        std::vector<Real> kvals(nf);
        for (int i = 0; i < nf; ++i)
            kvals[i] = scale * Real((i <= nf / 2) ? i : i - nf);

        auto grad_hat = [](const std::complex<Real> &ph, Real km) {
            return std::complex<Real>(-ph.imag() * km, ph.real() * km); // i*km*ph
        };
        // DMK's grid_idx (row-major, axis DIM-1 fastest) and FINUFFT's internal grid storage
        // (column-major, axis 0 fastest) disagree on which loop variable is which physical axis;
        if constexpr (DIM == 3) {
#pragma omp parallel for
            for (int g0 = 0; g0 < nf; ++g0)
                for (int g1 = 0; g1 < nf; ++g1) {
                    const int base = (g0 * nf + g1) * nf;
                    for (int g2 = 0; g2 < nf; ++g2) {
                        const int idx = base + g2;
                        const std::complex<Real> ph = b[idx] * scaling_coeffs[idx];
                        u_hat[0][idx] = ph;
                        u_hat[1][idx] = grad_hat(ph, kvals[g2]); // force axis 0
                        u_hat[2][idx] = grad_hat(ph, kvals[g1]); // force axis 1
                        u_hat[3][idx] = grad_hat(ph, kvals[g0]); // force axis 2
                    }
                }
        } else {
#pragma omp parallel for
            for (int g0 = 0; g0 < nf; ++g0) {
                const int base = g0 * nf;
                for (int g1 = 0; g1 < nf; ++g1) {
                    const int idx = base + g1;
                    const std::complex<Real> ph = b[idx] * scaling_coeffs[idx];
                    u_hat[0][idx] = ph;
                    u_hat[1][idx] = grad_hat(ph, kvals[g1]); // force axis 0
                    u_hat[2][idx] = grad_hat(ph, kvals[g0]); // force axis 1
                }
            }
        }
    }

    // 4-5. Inverse FFT + interpolate. Each output-component spectrum is Hermitian, so its inverse
    // transform is a real grid; we carry two real components in one complex transform by packing
    // component A into the real channel and B into the imaginary channel as A + i*B. ifft(A + i*B) =
    // gridA + i*gridB with gridA, gridB real, and because spreadinterponly interpolation is a
    // real-kernel gather the interpolated coefficient's real part is component A at the target and its
    // imaginary part is component B -- one complex iFFT + interp delivers two real components.
    auto &out = lr_out;
    out.resize(n);
    auto ifft_interp = [&](std::vector<std::complex<Real>> &g) { // g -> real/imag fields in `out`
        {
            sctl::Profile::Scoped prof("lr_fft_inv");
            ifftn<Real, DIM>(g, g, nf); // in place
        }
        sctl::Profile::Scoped prof("lr_interp");
        nufft2(out.data(), g.data());
    };
    // Component 0 is the potential; components 1.. are force axes 0.. (accumulated as -q*field).
    auto accumulate = [&](int k, int j, Real field) {
        if (k == 0)
            pot[j] += field;
        else
            force[k - 1][j] += -charges[j] * field;
    };

    for (int k = 0; k < out_dim; k += 2) {
        const bool paired = (k + 1 < out_dim);
        if (paired)
#pragma omp parallel for
            for (int idx = 0; idx < ntot; ++idx) // pack A + i*B = {A.re - B.im, A.im + B.re}
                u_hat[k][idx] = {u_hat[k][idx].real() - u_hat[k + 1][idx].imag(),
                                 u_hat[k][idx].imag() + u_hat[k + 1][idx].real()};
        ifft_interp(u_hat[k]);
#pragma omp parallel for
        for (int j = 0; j < n; ++j) {
            accumulate(k, j, out[j].real());
            if (paired)
                accumulate(k + 1, j, out[j].imag());
        }
    }
}

// Self-interaction correction — subtracts directly from the provided potential span. 3D-only
// formula (bakes in the 4*pi solid-angle constant); not templated on DIM. Never reached for DIM=2
// today since short_range throws first (see short_range's kernel-dispatch site).
template <typename Real>
void EspPlan<Real>::self_interaction(int n, const Real *charges, std::span<Real> pot) {
    Real factor = Real(pswf(0.0) / (params.r_c * 4.0 * M_PI * pswf.c0));
    for (int i = 0; i < n; ++i)
        pot[i] -= charges[i] * factor;
}

template <typename Real>
EspPlan<Real>::EspPlan(const pdmk_esp_params &params_, int n_dim_)
    : n_digits(esp_digits_from_eps(params_.eps)), n_dim(n_dim_), params(params_) {
    const Real eps_d = std::pow(10.0, -Real(n_digits));
    const double sigma = params.sigma;
    P = esp_P_from_eps(eps_d, sigma, n_dim);
    const double c = esp_pswf_c_from_P(sigma, P);
    pswf = PSWFKernel(eps_d, c);
    n_f = static_cast<int>(std::ceil(pswf.c * params.L / (M_PI * params.r_c)));
    h = params.L / n_f;
    const std::vector<double> sc =
        n_dim == 2 ? precompute_scaling_coefficients<2>() : precompute_scaling_coefficients<3>();
    scaling_coeffs.assign(sc.begin(), sc.end());

    if (n_dim == 3) {
        constexpr int MaxVecLen = sctl::DefaultVecLen<Real>();
        evaluator = get_esp_3d_kernel<Real, MaxVecLen>(params.eval_type, n_digits);
        range_evaluator = get_esp_3d_kernel_ranges<Real, MaxVecLen>(params.eval_type, n_digits);
#ifdef DMK_USE_JIT
        if (!util::env_is_set("DMK_DEBUG_FORCE_AOT"))
            evaluator = make_esp_evaluator_jit<Real>(params.eval_type, n_digits, sigma, 3);
#endif
    }
}

template <typename Real>
PotForce<Real> EspPlan<Real>::eval(int n, const Real *r_src, const Real *charges) {
    sctl::Profile::Scoped esp_eval("esp_eval");
    const bool want_force = (params.eval_type >= DMK_POTENTIAL_GRAD);
    const int slots = want_force ? 1 + n_dim : 1;

    // Reuse the plan's typed workspace; zero-initialize the active region.
    if (static_cast<int>(buf.size()) < slots * n)
        buf.resize(slots * n);
    std::fill(buf.begin(), buf.begin() + slots * n, Real(0));

    std::span<Real> pot_sp(buf.data(), n);
    std::span<Real> fx_sp(buf.data() + n, want_force ? n : 0);
    std::span<Real> fy_sp(buf.data() + 2 * n, want_force ? n : 0);
    std::span<Real> fz_sp(buf.data() + 3 * n, (want_force && n_dim == 3) ? n : 0);

    // Runtime n_dim -> compile-time DIM dispatch (mirrors src/aot_evaluator.cpp's pattern). DIM=3 is
    // the only dimension with a working short-range kernel; short_range throws for DIM=2 before
    // touching long_range/self_interaction (self_interaction's 3D-only formula is therefore never
    // exercised for DIM=2).
    if (n_dim == 3) {
        std::array<std::span<Real>, 3> force{fx_sp, fy_sp, fz_sp};
        short_range<3>(n, r_src, charges, pot_sp, force);
        long_range<3>(n, r_src, charges, pot_sp, force);
        self_interaction(n, charges, pot_sp);
    } else if (n_dim == 2) {
        std::array<std::span<Real>, 2> force{fx_sp, fy_sp};
        short_range<2>(n, r_src, charges, pot_sp, force);
        long_range<2>(n, r_src, charges, pot_sp, force);
        self_interaction(n, charges, pot_sp);
    } else {
        throw std::runtime_error("ESP: unsupported n_dim=" + std::to_string(n_dim));
    }

    return {pot_sp, fx_sp, fy_sp, fz_sp};
}

template struct EspPlan<float>;
template struct EspPlan<double>;
} // namespace dmk
