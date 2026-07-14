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
#include <iostream>
#include <omp.h>
#include <sctl.hpp>
#include <span>
#include <stdexcept>
#include <vector>

using Vec3 = std::array<double, 3>;
using CGrid = std::vector<std::complex<double>>;
using DGrid = std::vector<double>;

static inline int grid_idx(int ix, int iy, int iz, int n_f) { return ix * n_f * n_f + iy * n_f + iz; }

static void fftn_3d(const CGrid &in, CGrid &out, int n) {
    out = in;
    ducc0::vfmav<std::complex<double>> v(out.data(), {(size_t)n, (size_t)n, (size_t)n});
    ducc0::c2c(v, v, {0, 1, 2}, true, 1.0);
}

static void ifftn_3d(const CGrid &in, CGrid &out, int n) {
    out = in;
    ducc0::vfmav<std::complex<double>> v(out.data(), {(size_t)n, (size_t)n, (size_t)n});
    ducc0::c2c(v, v, {0, 1, 2}, false, 1.0 / ((double)n * n * n));
}

namespace dmk {
struct PSWFKernel {
    dmk::Prolate0Fun pswf;
    double eps;
    double c;
    double lambda0;
    double c0;
    double scale;
    std::vector<double> pswf_poly_coeffs;
    std::vector<double> pswf_int_poly_coeffs;

    PSWFKernel() = default;
    explicit PSWFKernel(double eps_, double c_, int lenw = 8000) : eps(eps_), c(c_) {
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

    double operator()(double x) const { return pswf.eval_val(x) * scale; }

    double integral_eval(double t) const { return pswf.int_eval(t) * scale; }

    double integral(double a, double b) const {
        double va = (a == 0.0) ? 0.0 : integral_eval(a);
        double vb = integral_eval(b);
        return vb - va;
    }

    double pswf_hat(double k) const {
        const double x = k / c;
        return std::fabs(x) > 1 ? 0.0 : lambda0 * (*this)(x);
    }
};

struct ESPParams {
    double L;
    double r_c;
    double sigma;
    int P;
    int n_f;
    double h;
    double lambda0;
    double c;
    double c0;
    int n;

    ESPParams() = default;
    ESPParams(double L_, double r_c_, double sigma_, int P_, const PSWFKernel &pswf, int n_)
        : L(L_), r_c(r_c_), sigma(sigma_), P(P_), n(n_) {
        n_f = static_cast<int>(std::ceil(pswf.c * L / (M_PI * r_c)));
        h = L / n_f;
        lambda0 = pswf.lambda0;
        c = pswf.c;
        c0 = pswf.c0;
    }
};

struct CellIndex {
    int x, y, z;
};

template <typename Real>
static inline CellIndex particle_cell(const Vec3T<Real> &r, Real L, int n_cells) {
    Real cell_size = L / n_cells;
    CellIndex ci;
    ci.x = static_cast<int>(std::floor((r[0] + L / 2) / cell_size)) % n_cells;
    ci.y = static_cast<int>(std::floor((r[1] + L / 2) / cell_size)) % n_cells;
    ci.z = static_cast<int>(std::floor((r[2] + L / 2) / cell_size)) % n_cells;
    if (ci.x < 0)
        ci.x += n_cells;
    if (ci.y < 0)
        ci.y += n_cells;
    if (ci.z < 0)
        ci.z += n_cells;
    return ci;
}

// Morton (Z-order) key from a 3D integer coordinate, used to spatially sort particles
// within a cell so that consecutive VecLen-runs form geometrically compact tiles.
static inline uint64_t part1by2_64(uint64_t x) {
    x &= 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffffULL;
    x = (x | x << 16) & 0x1f0000ff0000ffULL;
    x = (x | x << 8) & 0x100f00f00f00f00fULL;
    x = (x | x << 4) & 0x10c30c30c30c30c3ULL;
    x = (x | x << 2) & 0x1249249249249249ULL;
    return x;
}
static inline uint64_t morton3(uint64_t x, uint64_t y, uint64_t z) {
    return (part1by2_64(x) << 2) | (part1by2_64(y) << 1) | part1by2_64(z);
}

// Cell list: particles sorted into cubic cells for cache-friendly traversal.
template <typename Real>
struct CellList {
    int n_cells;
    std::vector<int> cell_start; // size n_cells^3 + 1 (exclusive prefix sum)
    std::vector<Real> rs, qs;    // particle data reordered by cell
    std::vector<int> orig;       // orig[slot] = original particle index
};

template <typename Real>
static CellList<Real> build_cell_list(const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges,
                                      const ESPParams &params, int nc, bool spatial_sort, int min_sort_len = 2) {
    sctl::Profile::Scoped profile("build_cell_list");
    CellList<Real> cl;
    cl.n_cells = nc;
    const int ncells = nc * nc * nc;
    const int n = params.n;
    const Real L_r = Real(params.L);

    auto cell_of = [&](const Vec3T<Real> &r) {
        CellIndex ci = particle_cell<Real>(r, L_r, nc);
        return (ci.x * nc + ci.y) * nc + ci.z;
    };

    // pass 1: count per cell
    std::vector<int> count(ncells, 0), cidx(n);
    for (int j = 0; j < n; ++j) {
        int c = cell_of(r_src[j]);
        cidx[j] = c;
        ++count[c];
    }

    // pass 2: exclusive prefix sum -> cell start offsets
    cl.cell_start.assign(ncells + 1, 0);
    for (int c = 0; c < ncells; ++c)
        cl.cell_start[c + 1] = cl.cell_start[c] + count[c];

    // pass 3: scatter into sorted arrays
    cl.rs.resize(3 * n);
    cl.qs.resize(n);
    cl.orig.resize(n);
    std::vector<int> cursor(cl.cell_start.begin(), cl.cell_start.end() - 1);
    for (int j = 0; j < n; ++j) {
        int slot = cursor[cidx[j]]++;
        cl.rs[3 * slot + 0] = r_src[j][0];
        cl.rs[3 * slot + 1] = r_src[j][1];
        cl.rs[3 * slot + 2] = r_src[j][2];
        cl.qs[slot] = charges[j];
        cl.orig[slot] = j;
    }

    // pass 4: Spatially reorder particles within each cell so consecutive VecLen-runs form
    // compact tiles, enabling geometric source pruning in short_range. Two env-selected
    // strategies; both keep the tile/range count fixed and only tighten each tile's extent:
    //   DMK_ESP_BINS=b (default 2): counting sort into b^3 spatial sub-boxes, no comparisons.
    //                               b=2 is the octant sort; larger b tightens tiles, still cheap.
    //   DMK_ESP_MORTON: full z-order (Morton) sort within the cell -- tightest tiles, but a
    //                   comparison sort per cell (notably slower to build).
    if (spatial_sort) {
        sctl::Profile::Scoped sort("spatial_sort");

        const bool morton = util::env_is_set("DMK_ESP_MORTON");
        const int bins = [] {
            const char *e = std::getenv("DMK_ESP_BINS");
            const int v = e ? std::atoi(e) : 2;
            return v > 0 ? v : 2;
        }();
        const int nbuckets = bins * bins * bins;

        const Real h = L_r / Real(nc);
        const Real half_L = L_r / Real(2);

        std::vector<Real> tr, tq; // scratch for the in-place reorder
        std::vector<int> to;
        std::vector<int> key;        // per-particle bucket (bins) or sort permutation (morton)
        std::vector<int> off;        // bucket offsets (bins)
        std::vector<uint64_t> mcode; // morton codes (morton)

        // Interleave the low 21 bits of three integers into a 63-bit Morton code.
        auto spread = [](uint64_t v) {
            v &= 0x1fffffULL;
            v = (v | (v << 32)) & 0x1f00000000ffffULL;
            v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
            v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
            v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
            v = (v | (v << 2)) & 0x1249249249249249ULL;
            return v;
        };

        for (int c = 0; c < ncells; ++c) {
            const int b = cl.cell_start[c], e = cl.cell_start[c + 1], len = e - b;
            if (len <= min_sort_len)
                continue;

            const int cz = c % nc;
            const int cy = (c / nc) % nc;
            const int cx = c / (nc * nc);
            const Real lo_x = Real(cx) * h - half_L; // cell lower corner
            const Real lo_y = Real(cy) * h - half_L;
            const Real lo_z = Real(cz) * h - half_L;

            tr.resize(3 * len);
            tq.resize(len);
            to.resize(len);

            if (morton) {
                // Quantize local coordinates to 21 bits/axis, then stable-sort indices by code.
                const Real q = Real(uint32_t(1) << 21) / h;
                mcode.resize(len);
                key.resize(len);
                for (int i = 0; i < len; ++i) {
                    const int s = b + i;
                    const auto qx =
                        uint32_t(std::min(Real((1u << 21) - 1), std::max(Real(0), (cl.rs[3 * s + 0] - lo_x) * q)));
                    const auto qy =
                        uint32_t(std::min(Real((1u << 21) - 1), std::max(Real(0), (cl.rs[3 * s + 1] - lo_y) * q)));
                    const auto qz =
                        uint32_t(std::min(Real((1u << 21) - 1), std::max(Real(0), (cl.rs[3 * s + 2] - lo_z) * q)));
                    mcode[i] = spread(qx) | (spread(qy) << 1) | (spread(qz) << 2);
                    key[i] = i;
                }
                std::sort(key.begin(), key.end(), [&](int a, int bb) { return mcode[a] < mcode[bb]; });
                for (int i = 0; i < len; ++i) {
                    const int s = b + key[i];
                    tr[3 * i + 0] = cl.rs[3 * s + 0];
                    tr[3 * i + 1] = cl.rs[3 * s + 1];
                    tr[3 * i + 2] = cl.rs[3 * s + 2];
                    tq[i] = cl.qs[s];
                    to[i] = cl.orig[s];
                }
            } else {
                // Counting sort into bins^3 spatial sub-boxes (b=2 -> octants).
                const Real scale = Real(bins) / h;
                key.resize(len);
                off.assign(nbuckets, 0);
                for (int i = 0; i < len; ++i) {
                    const int s = b + i;
                    const int bx = std::min(bins - 1, std::max(0, int((cl.rs[3 * s + 0] - lo_x) * scale)));
                    const int by = std::min(bins - 1, std::max(0, int((cl.rs[3 * s + 1] - lo_y) * scale)));
                    const int bz = std::min(bins - 1, std::max(0, int((cl.rs[3 * s + 2] - lo_z) * scale)));
                    const int bucket = bx + bins * (by + bins * bz);
                    key[i] = bucket;
                    ++off[bucket];
                }
                int acc = 0;
                for (int k = 0; k < nbuckets; ++k) {
                    const int cnt = off[k];
                    off[k] = acc;
                    acc += cnt;
                }
                for (int i = 0; i < len; ++i) {
                    const int s = b + i;
                    const int dst = off[key[i]]++;
                    tr[3 * dst + 0] = cl.rs[3 * s + 0];
                    tr[3 * dst + 1] = cl.rs[3 * s + 1];
                    tr[3 * dst + 2] = cl.rs[3 * s + 2];
                    tq[dst] = cl.qs[s];
                    to[dst] = cl.orig[s];
                }
            }

            for (int i = 0; i < len; ++i) {
                cl.rs[3 * (b + i) + 0] = tr[3 * i + 0];
                cl.rs[3 * (b + i) + 1] = tr[3 * i + 1];
                cl.rs[3 * (b + i) + 2] = tr[3 * i + 2];
                cl.qs[b + i] = tq[i];
                cl.orig[b + i] = to[i];
            }
        }
    }
    return cl;
}

template <typename Real>
static void short_range(const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges,
                        const ESPParams &params, int n_digits, const PSWFKernel &pswf, dmk_eval_type eval_type,
                        std::span<Real> pot, std::span<Real> fx, std::span<Real> fy, std::span<Real> fz) {
    sctl::Profile::Scoped short_range("short_range");
    // 27-cell stencil requires nc >= 3 so periodic images aren't double-counted
    int nc = static_cast<int>(std::floor(params.L / params.r_c));
    if (nc < 3)
        throw std::runtime_error("short_range_fast requires r_c <= L/3 (nc >= 3)");

    // DMK_ESP_PRUNE selects the (experimental) geometric pruning path; nonzero modes also need
    // the within-cell sort. 0 (unset): original dense path, verbatim. 1: sub-cell tile-vs-tile
    // AABB pruning. 2: per-source point-vs-target-box pruning (finest granularity, survivors fed
    // to the range evaluator as unit-length ranges). Set-but-nonnumeric maps to mode 1.
    const int prune = [] {
        const char *e = std::getenv("DMK_ESP_PRUNE");
        if (!e || !*e)
            return 0;
        const int v = std::atoi(e);
        return v > 0 ? v : 1;
    }();
    // DMK_ESP_N3L: Newton's-third-law reciprocal path. Each cross-cell pair is evaluated once
    // (13-forward-neighbour half stencil) and scattered to both endpoints; the home cell stays
    // full (both endpoints are targets there, so no reciprocal is needed). Home cells are
    // 27-coloured so reciprocal writes into neighbour cells never race. Uses the per-source cull.
    const bool n3l = util::env_is_set("DMK_ESP_N3L");
    // Stride-3 periodic colouring is conflict-free only if each axis length is divisible by the
    // stride; round nc down to a multiple of 3 (cells grow slightly, still >= r_c, still >= 3).
    if (n3l)
        nc -= nc % 3;
    // One-off phase breakdown of the pruned short-range loop (gather / AABB / test / eval).
    // sctl::Profile can't scope inside this loop cleanly; omp_get_wtime accumulators do the job.
    const bool phase_timing = util::env_is_set("DMK_ESP_PHASE_TIMING");
    // Source-test tile width, decoupled from SIMD width (the kernel broadcasts sources one at
    // a time). Smaller -> tighter source AABBs -> more culling, at the cost of more AABB tests.
    constexpr int MaxVecLen = sctl::DefaultVecLen<Real>();

    const int stile = [] {
        const char *e = std::getenv("DMK_ESP_STILE");
        const int v = e ? std::atoi(e) : MaxVecLen;
        return v > 0 ? v : MaxVecLen;
    }();

    CellList<Real> cl = build_cell_list<Real>(r_src, charges, params, nc, prune != 0 || n3l, MaxVecLen);

    const Real L = Real(params.L);
    const Real r_c_sq = Real(params.r_c) * Real(params.r_c);
    const int n = params.n;
    const bool want_force = (eval_type >= DMK_POTENTIAL_GRAD);
    const int out_dim = want_force ? 4 : 1;
    auto evaluator = get_esp_3d_kernel<Real, MaxVecLen>(eval_type, n_digits);
    auto range_evaluator = get_esp_3d_kernel_ranges<Real, MaxVecLen>(eval_type, n_digits);
#ifdef DMK_USE_JIT
    if (!util::env_is_set("DMK_DEBUG_FORCE_AOT"))
        evaluator = make_esp_evaluator_jit<Real>(eval_type, n_digits, params.sigma, 3);
#endif
    const Real rsc = Real(2.0 / params.r_c);
    const Real cen = Real(-params.r_c / 2.0);

    // Interleaved [pot] or [pot, d/dx, d/dy, d/dz] per particle, in cell-sorted order.
    std::vector<Real> pg_sorted(out_dim * n, Real(0));

    // For each cell coordinate c in [0, nc) and each delta d in {-1,0,+1} (stored as d=0,1,2):
    // nbc_tab[c*3+d] = neighbor cell index, off_tab[c*3+d] = image shift to subtract.
    // Precomputed once; all three dimensions share the same table since they share nc.
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

    double sum_gather = 0, sum_aabb = 0, sum_test = 0, sum_eval = 0; // phase_timing totals

#pragma omp parallel
    {
        std::vector<Real> r_src_g, charge_g;
        // Source-tile AABBs, SoA so the tile-vs-tile test vectorizes; d2buf holds its output.
        std::vector<Real> slo_x, slo_y, slo_z, shi_x, shi_y, shi_z, d2buf;
        std::vector<int> tile_s0, tile_sn; // cell-aligned source tiles (start, length)
        std::vector<int> surv_s0, surv_sn; // surviving source tiles / source indices per target-tile
        // Mode 2 (per-source): SoA source coords.
        std::vector<Real> src_x, src_y, src_z;
        // N3L: gathered forward-neighbour sources (AoS + SoA), their charges, cell-sorted origin
        // indices, and the per-source reciprocal accumulator.
        std::vector<Real> r_fwd, fwd_q, fwd_x, fwd_y, fwd_z, pg_src;
        std::vector<int> fwd_origin;
        double tg = 0, ta = 0, tt = 0, te = 0; // per-thread phase times (phase_timing)
        if (n3l) {
            using RealVec = sctl::Vec<Real, MaxVecLen>;
            // Per-source point-vs-target-box cull (same test as prune==2), factored so the self
            // and forward blocks share it. Appends surviving source indices to idx_ptr.
            auto cull_box = [&](const std::vector<Real> &sx, const std::vector<Real> &sy, const std::vector<Real> &sz,
                                int nsrc, const Real lo[3], const Real hi[3], int *__restrict__ idx_ptr) -> int {
                const RealVec hi0v(hi[0]), hi1v(hi[1]), hi2v(hi[2]);
                const RealVec lo0v(lo[0]), lo1v(lo[1]), lo2v(lo[2]);
                const RealVec zero = RealVec::Zero(), rc2v(r_c_sq);
                auto box_d2 = [&](int i) {
                    const RealVec x = RealVec::Load(&sx[i]);
                    const RealVec y = RealVec::Load(&sy[i]);
                    const RealVec z = RealVec::Load(&sz[i]);
                    const RealVec dx = sctl::max(zero, sctl::max(lo0v - x, x - hi0v));
                    const RealVec dy = sctl::max(zero, sctl::max(lo1v - y, y - hi1v));
                    const RealVec dz = sctl::max(zero, sctl::max(lo2v - z, z - hi2v));
                    return sctl::FMA(dx, dx, sctl::FMA(dy, dy, dz * dz));
                };
                int m = 0, s = 0;
                for (; s + 2 * MaxVecLen <= nsrc; s += 2 * MaxVecLen)
                    m += sctl::mask_compress_iota_store2(box_d2(s) <= rc2v, box_d2(s + MaxVecLen) <= rc2v, s,
                                                         idx_ptr + m);
                for (; s + MaxVecLen <= nsrc; s += MaxVecLen)
                    m += sctl::mask_compress_iota_store(box_d2(s) <= rc2v, s, idx_ptr + m);
                for (; s < nsrc; ++s) {
                    const Real dx = std::max(Real(0), std::max(lo[0] - sx[s], sx[s] - hi[0]));
                    const Real dy = std::max(Real(0), std::max(lo[1] - sy[s], sy[s] - hi[1]));
                    const Real dz = std::max(Real(0), std::max(lo[2] - sz[s], sz[s] - hi[2]));
                    idx_ptr[m] = s;
                    m += (dx * dx + dy * dy + dz * dz <= r_c_sq);
                }
                return m;
            };
            auto trg_box = [](const Real *tptr, int tn, Real lo[3], Real hi[3]) {
                for (int k = 0; k < 3; ++k)
                    lo[k] = hi[k] = tptr[k];
                for (int i = 1; i < tn; ++i)
                    for (int k = 0; k < 3; ++k) {
                        const Real v = tptr[3 * i + k];
                        lo[k] = std::min(lo[k], v);
                        hi[k] = std::max(hi[k], v);
                    }
            };
            // Forward half of the 26 neighbours: deltas lexicographically greater than 0, so each
            // unordered cross-cell pair is visited exactly once (the backward half is covered when
            // those cells are home). dx/dy/dz in {0,1,2} encode delta {-1,0,+1}.
            auto is_forward = [](int dx, int dy, int dz) {
                return dx > 1 || (dx == 1 && (dy > 1 || (dy == 1 && dz > 1)));
            };

            // 27-colour so no two concurrently-processed home cells share a write cell (home + its
            // forward neighbours span the +-1 shell; stride 3 keeps same-colour cells >= 3 apart).
            for (int color = 0; color < 27; ++color) {
                const int c0x = color / 9, c0y = (color / 3) % 3, c0z = color % 3;
#pragma omp for schedule(dynamic) collapse(3)
                for (int cx = c0x; cx < nc; cx += 3)
                    for (int cy = c0y; cy < nc; cy += 3)
                        for (int cz = c0z; cz < nc; cz += 3) {
                            const int home = (cx * nc + cy) * nc + cz;
                            const int hbeg = cl.cell_start[home], hend = cl.cell_start[home + 1];
                            const int n_trg = hend - hbeg;
                            if (n_trg == 0)
                                continue;

                            const Real *__restrict__ r_trg_ptr = cl.rs.data() + 3 * hbeg;
                            const Real *__restrict__ q_trg_all = cl.qs.data() + hbeg;
                            const int *__restrict__ nbc_cx = &nbc_tab[cx * 3];
                            const int *__restrict__ nbc_cy = &nbc_tab[cy * 3];
                            const int *__restrict__ nbc_cz = &nbc_tab[cz * 3];
                            const Real *__restrict__ off_cx = &off_tab[cx * 3];
                            const Real *__restrict__ off_cy = &off_tab[cy * 3];
                            const Real *__restrict__ off_cz = &off_tab[cz * 3];

                            // Count forward-neighbour sources so the surv buffers can be sized once.
                            int n_fwd = 0;
                            for (int dx = 0; dx < 3; ++dx)
                                for (int dy = 0; dy < 3; ++dy)
                                    for (int dz = 0; dz < 3; ++dz) {
                                        if (!is_forward(dx, dy, dz))
                                            continue;
                                        const int nb = (nbc_cx[dx] * nc + nbc_cy[dy]) * nc + nbc_cz[dz];
                                        n_fwd += cl.cell_start[nb + 1] - cl.cell_start[nb];
                                    }

                            const int nmax = std::max(n_trg, n_fwd);
                            if (static_cast<int>(surv_s0.size()) < nmax)
                                surv_s0.resize(nmax);
                            if (static_cast<int>(surv_sn.size()) < nmax)
                                surv_sn.resize(nmax, 1);
                            int *__restrict__ idx_ptr = surv_s0.data();

                            // Self block: home cell is its own source, evaluated full (both
                            // endpoints are home targets, so no reciprocal). SoA-repack for the cull.
                            src_x.resize(n_trg);
                            src_y.resize(n_trg);
                            src_z.resize(n_trg);
                            for (int s = 0; s < n_trg; ++s) {
                                src_x[s] = r_trg_ptr[3 * s + 0];
                                src_y[s] = r_trg_ptr[3 * s + 1];
                                src_z[s] = r_trg_ptr[3 * s + 2];
                            }
                            for (int t0 = 0; t0 < n_trg; t0 += MaxVecLen) {
                                const int tn = std::min(MaxVecLen, n_trg - t0);
                                const Real *__restrict__ tptr = r_trg_ptr + 3 * t0;
                                Real lo[3], hi[3];
                                trg_box(tptr, tn, lo, hi);
                                const int m = cull_box(src_x, src_y, src_z, n_trg, lo, hi, idx_ptr);
                                range_evaluator(rsc, cen, r_c_sq, Real(0), n_trg, r_trg_ptr, q_trg_all, nullptr, m,
                                                idx_ptr, surv_sn.data(), tn, tptr,
                                                pg_sorted.data() + out_dim * (hbeg + t0), nullptr, nullptr);
                            }

                            if (n_fwd == 0)
                                continue;

                            // Forward block: gather the 13 forward cells (AoS + charge + origin),
                            // repack SoA, then evaluate N3L (forward onto home targets, reciprocal
                            // onto pg_src indexed by forward source).
                            r_fwd.resize(3 * n_fwd);
                            fwd_q.resize(n_fwd);
                            fwd_origin.resize(n_fwd);
                            int r_i = 0, c_i = 0;
                            for (int dx = 0; dx < 3; ++dx) {
                                const Real off_x = off_cx[dx];
                                for (int dy = 0; dy < 3; ++dy) {
                                    const Real off_y = off_cy[dy];
                                    for (int dz = 0; dz < 3; ++dz) {
                                        if (!is_forward(dx, dy, dz))
                                            continue;
                                        const Real off_z = off_cz[dz];
                                        const int nb = (nbc_cx[dx] * nc + nbc_cy[dy]) * nc + nbc_cz[dz];
                                        const int nbeg = cl.cell_start[nb], nend = cl.cell_start[nb + 1];
                                        for (int b = nbeg; b < nend; ++b) {
                                            r_fwd[r_i++] = cl.rs[b * 3 + 0] + off_x;
                                            r_fwd[r_i++] = cl.rs[b * 3 + 1] + off_y;
                                            r_fwd[r_i++] = cl.rs[b * 3 + 2] + off_z;
                                            fwd_q[c_i] = cl.qs[b];
                                            fwd_origin[c_i] = b;
                                            ++c_i;
                                        }
                                    }
                                }
                            }
                            fwd_x.resize(n_fwd);
                            fwd_y.resize(n_fwd);
                            fwd_z.resize(n_fwd);
                            for (int s = 0; s < n_fwd; ++s) {
                                fwd_x[s] = r_fwd[3 * s + 0];
                                fwd_y[s] = r_fwd[3 * s + 1];
                                fwd_z[s] = r_fwd[3 * s + 2];
                            }
                            pg_src.assign(out_dim * n_fwd, Real(0));
                            const Real *__restrict__ r_fwd_ptr = r_fwd.data();
                            const Real *__restrict__ fwd_q_ptr = fwd_q.data();
                            for (int t0 = 0; t0 < n_trg; t0 += MaxVecLen) {
                                const int tn = std::min(MaxVecLen, n_trg - t0);
                                const Real *__restrict__ tptr = r_trg_ptr + 3 * t0;
                                Real lo[3], hi[3];
                                trg_box(tptr, tn, lo, hi);
                                const int m = cull_box(fwd_x, fwd_y, fwd_z, n_fwd, lo, hi, idx_ptr);
                                range_evaluator(rsc, cen, r_c_sq, Real(0), n_fwd, r_fwd_ptr, fwd_q_ptr, nullptr, m,
                                                idx_ptr, surv_sn.data(), tn, tptr,
                                                pg_sorted.data() + out_dim * (hbeg + t0), q_trg_all + t0,
                                                pg_src.data());
                            }
                            // Scatter reciprocal contributions back onto the forward sources; safe
                            // because same-colour home cells write disjoint neighbour cells.
                            for (int s = 0; s < n_fwd; ++s) {
                                const int b = fwd_origin[s];
                                for (int k = 0; k < out_dim; ++k)
                                    pg_sorted[out_dim * b + k] += pg_src[out_dim * s + k];
                            }
                        }
            }
        } else
#pragma omp for schedule(dynamic) collapse(3)
            for (int cx = 0; cx < nc; ++cx)
                for (int cy = 0; cy < nc; ++cy)
                    for (int cz = 0; cz < nc; ++cz) {
                        const int home = (cx * nc + cy) * nc + cz;
                        const int hbeg = cl.cell_start[home], hend = cl.cell_start[home + 1];
                        const int n_trg = hend - hbeg;
                        if (n_trg == 0)
                            continue;

                        double _t = phase_timing ? omp_get_wtime() : 0.0;

                        const Real *__restrict__ r_trg_ptr = cl.rs.data() + 3 * hbeg;

                        const int *__restrict__ nbc_cx = &nbc_tab[cx * 3];
                        const int *__restrict__ nbc_cy = &nbc_tab[cy * 3];
                        const int *__restrict__ nbc_cz = &nbc_tab[cz * 3];
                        const Real *__restrict__ off_cx = &off_tab[cx * 3];
                        const Real *__restrict__ off_cy = &off_tab[cy * 3];
                        const Real *__restrict__ off_cz = &off_tab[cz * 3];

                        int n_src = 0;
                        for (int dx = 0; dx < 3; ++dx)
                            for (int dy = 0; dy < 3; ++dy)
                                for (int dz = 0; dz < 3; ++dz) {
                                    const int nb = (nbc_cx[dx] * nc + nbc_cy[dy]) * nc + nbc_cz[dz];
                                    const int nbeg = cl.cell_start[nb], nend = cl.cell_start[nb + 1];
                                    n_src += nend - nbeg;
                                }

                        r_src_g.resize(3 * n_src);
                        charge_g.resize(n_src);
                        Real *__restrict__ r_src_ptr = r_src_g.data();
                        Real *__restrict__ charge_ptr = charge_g.data();
                        tile_s0.clear();
                        tile_sn.clear();
                        int r_i = 0, c_i = 0;
                        for (int dx = 0; dx < 3; ++dx) {
                            const Real off_x = off_cx[dx];
                            const int nbc_x = nbc_cx[dx];
                            for (int dy = 0; dy < 3; ++dy) {
                                const Real off_y = off_cy[dy];
                                const int nbc_y = nbc_cy[dy];
                                for (int dz = 0; dz < 3; ++dz) {
                                    const Real off_z = off_cz[dz];
                                    const int nbc_z = nbc_cz[dz];
                                    const int nb = (nbc_x * nc + nbc_y) * nc + nbc_z;
                                    const int nbeg = cl.cell_start[nb], nend = cl.cell_start[nb + 1];
                                    const int seg0 = c_i;
                                    for (int b = nbeg; b < nend; ++b) {
                                        r_src_ptr[r_i++] = cl.rs[b * 3 + 0] + off_x;
                                        r_src_ptr[r_i++] = cl.rs[b * 3 + 1] + off_y;
                                        r_src_ptr[r_i++] = cl.rs[b * 3 + 2] + off_z;
                                        charge_ptr[c_i++] = cl.qs[b];
                                    }
                                    // Split this cell's contribution into VecLen source tiles that
                                    // never cross a cell boundary, so each tile's AABB stays compact.
                                    if (prune == 1)
                                        for (int s = seg0; s < c_i; s += stile) {
                                            tile_s0.push_back(s);
                                            tile_sn.push_back(std::min(stile, c_i - s));
                                        }
                                }
                            }
                        }
                        if (phase_timing) {
                            const double now = omp_get_wtime();
                            tg += now - _t;
                            _t = now;
                        }

                        if (prune == 0) {
                            evaluator(rsc, cen, r_c_sq, Real(0), n_src, r_src_ptr, charge_ptr, nullptr, n_trg,
                                      r_trg_ptr, pg_sorted.data() + out_dim * hbeg);
                            if (phase_timing)
                                te += omp_get_wtime() - _t;
                            continue;
                        }

                        if (prune == 2) {
                            // Per-source pruning (point-vs-target-box). Repack sources to SoA so the
                            // per-source box-distance test vectorizes, then for each target tile keep the
                            // individual sources within r_c of its AABB and feed them to the range
                            // evaluator as unit-length ranges (surv_sn all 1) -- no gather/copy, and the
                            // survivor fraction tracks the true in-range fraction far better than tiles.
                            src_x.resize(n_src);
                            src_y.resize(n_src);
                            src_z.resize(n_src);
                            for (int s = 0; s < n_src; ++s) {
                                src_x[s] = r_src_ptr[3 * s + 0];
                                src_y[s] = r_src_ptr[3 * s + 1];
                                src_z[s] = r_src_ptr[3 * s + 2];
                            }
                            surv_s0.resize(n_src);
                            if (static_cast<int>(surv_sn.size()) < n_src)
                                surv_sn.resize(n_src, 1); // unit lengths; all entries stay 1 across cells
                            int *__restrict__ idx_ptr = surv_s0.data();

                            if (phase_timing) {
                                const double now = omp_get_wtime();
                                ta += now - _t;
                                _t = now;
                            }

                            for (int t0 = 0; t0 < n_trg; t0 += MaxVecLen) {
                                const int tn = std::min(MaxVecLen, n_trg - t0);
                                const Real *__restrict__ tptr = r_trg_ptr + 3 * t0;

                                Real lo[3], hi[3];
                                for (int k = 0; k < 3; ++k)
                                    lo[k] = hi[k] = tptr[k];
                                for (int i = 1; i < tn; ++i)
                                    for (int k = 0; k < 3; ++k) {
                                        const Real v = tptr[3 * i + k];
                                        lo[k] = std::min(lo[k], v);
                                        hi[k] = std::max(hi[k], v);
                                    }

                                // Fused per-source test + compaction, fully SIMD: box-distance via SCTL
                                // Vec, compare to r_c^2, and compress the surviving source indices with a
                                // masked vcompress -- no d2 scratch, no scalar filter. Two half-width d2
                                // masks drive one full-width int32 vcompress, so a double (8-lane) chunk
                                // pair costs a single compress/iota/popcount. Scalar tail only.
                                using RealVec = sctl::Vec<Real, MaxVecLen>;
                                const RealVec hi0v(hi[0]), hi1v(hi[1]), hi2v(hi[2]);
                                const RealVec lo0v(lo[0]), lo1v(lo[1]), lo2v(lo[2]);
                                const RealVec zero = RealVec::Zero(), rc2v(r_c_sq);
                                auto box_d2 = [&](int i) {
                                    const RealVec x = RealVec::Load(&src_x[i]);
                                    const RealVec y = RealVec::Load(&src_y[i]);
                                    const RealVec z = RealVec::Load(&src_z[i]);
                                    const RealVec dx = sctl::max(zero, sctl::max(lo0v - x, x - hi0v));
                                    const RealVec dy = sctl::max(zero, sctl::max(lo1v - y, y - hi1v));
                                    const RealVec dz = sctl::max(zero, sctl::max(lo2v - z, z - hi2v));
                                    return sctl::FMA(dx, dx, sctl::FMA(dy, dy, dz * dz));
                                };
                                int m = 0, s = 0;
                                for (; s + 2 * MaxVecLen <= n_src; s += 2 * MaxVecLen)
                                    m += sctl::mask_compress_iota_store2(box_d2(s) <= rc2v,
                                                                         box_d2(s + MaxVecLen) <= rc2v, s, idx_ptr + m);
                                for (; s + MaxVecLen <= n_src; s += MaxVecLen)
                                    m += sctl::mask_compress_iota_store(box_d2(s) <= rc2v, s, idx_ptr + m);
                                for (; s < n_src; ++s) {
                                    const Real dx = std::max(Real(0), std::max(lo[0] - src_x[s], src_x[s] - hi[0]));
                                    const Real dy = std::max(Real(0), std::max(lo[1] - src_y[s], src_y[s] - hi[1]));
                                    const Real dz = std::max(Real(0), std::max(lo[2] - src_z[s], src_z[s] - hi[2]));
                                    idx_ptr[m] = s;
                                    m += (dx * dx + dy * dy + dz * dz <= r_c_sq);
                                }

                                if (phase_timing) {
                                    const double now = omp_get_wtime();
                                    tt += now - _t;
                                    _t = now;
                                }

                                range_evaluator(rsc, cen, r_c_sq, Real(0), n_src, r_src_ptr, charge_ptr, nullptr, m,
                                                idx_ptr, surv_sn.data(), tn, tptr,
                                                pg_sorted.data() + out_dim * (hbeg + t0), nullptr, nullptr);

                                if (phase_timing) {
                                    const double now = omp_get_wtime();
                                    te += now - _t;
                                    _t = now;
                                }
                            }
                            continue;
                        }

                        // Sub-cell geometric pruning (tile-vs-tile). Sources and targets are each
                        // grouped into VecLen-wide tiles; one AABB test per (target-tile, source-tile)
                        // pair skips up to VecLen*VecLen interactions. Surviving source tiles are passed
                        // as a list of disjoint ranges to the range-list evaluator, which reads them
                        // in-place from the gathered array (the dense kernel's internal d2max mask still
                        // enforces the exact cutoff on the survivors).
                        const int n_stiles = static_cast<int>(tile_s0.size());
                        slo_x.resize(n_stiles);
                        slo_y.resize(n_stiles);
                        slo_z.resize(n_stiles);
                        shi_x.resize(n_stiles);
                        shi_y.resize(n_stiles);
                        shi_z.resize(n_stiles);
                        d2buf.resize(n_stiles);
                        for (int st = 0; st < n_stiles; ++st) {
                            const int s0 = tile_s0[st];
                            const int sn = tile_sn[st];
                            Real lo0 = r_src_ptr[3 * s0 + 0], lo1 = r_src_ptr[3 * s0 + 1], lo2 = r_src_ptr[3 * s0 + 2];
                            Real hi0 = lo0, hi1 = lo1, hi2 = lo2;
                            for (int i = 1; i < sn; ++i) {
                                const Real x = r_src_ptr[3 * (s0 + i) + 0];
                                const Real y = r_src_ptr[3 * (s0 + i) + 1];
                                const Real z = r_src_ptr[3 * (s0 + i) + 2];
                                lo0 = std::min(lo0, x), hi0 = std::max(hi0, x);
                                lo1 = std::min(lo1, y), hi1 = std::max(hi1, y);
                                lo2 = std::min(lo2, z), hi2 = std::max(hi2, z);
                            }
                            slo_x[st] = lo0, slo_y[st] = lo1, slo_z[st] = lo2;
                            shi_x[st] = hi0, shi_y[st] = hi1, shi_z[st] = hi2;
                        }

                        if (phase_timing) {
                            const double now = omp_get_wtime();
                            ta += now - _t;
                            _t = now;
                        }

                        surv_s0.resize(n_stiles);
                        surv_sn.resize(n_stiles);
                        for (int t0 = 0; t0 < n_trg; t0 += MaxVecLen) {
                            const int tn = std::min(MaxVecLen, n_trg - t0);
                            const Real *__restrict__ tptr = r_trg_ptr + 3 * t0;

                            Real lo[3], hi[3];
                            for (int k = 0; k < 3; ++k)
                                lo[k] = hi[k] = tptr[k];
                            for (int i = 1; i < tn; ++i)
                                for (int k = 0; k < 3; ++k) {
                                    const Real v = tptr[3 * i + k];
                                    lo[k] = std::min(lo[k], v);
                                    hi[k] = std::max(hi[k], v);
                                }

                            // Branchless squared box-distance of every source tile to this target tile;
                            // SoA loads with no gather or branch, so it vectorizes cleanly.
                            const Real hi0 = hi[0], hi1 = hi[1], hi2 = hi[2], lo0 = lo[0], lo1 = lo[1], lo2 = lo[2];
#pragma omp simd
                            for (int st = 0; st < n_stiles; ++st) {
                                const Real dx = std::max(Real(0), std::max(slo_x[st] - hi0, lo0 - shi_x[st]));
                                const Real dy = std::max(Real(0), std::max(slo_y[st] - hi1, lo1 - shi_y[st]));
                                const Real dz = std::max(Real(0), std::max(slo_z[st] - hi2, lo2 - shi_z[st]));
                                d2buf[st] = dx * dx + dy * dy + dz * dz;
                            }

                            // Branchless compaction of surviving source tiles into disjoint ranges:
                            // write unconditionally, advance the cursor only on a survivor. Avoids the
                            // ~80%-taken unpredictable branch and push_back overhead of the naive filter.
                            int *__restrict__ s0_ptr = surv_s0.data();
                            int *__restrict__ sn_ptr = surv_sn.data();
                            int m = 0;
                            for (int st = 0; st < n_stiles; ++st) {
                                s0_ptr[m] = tile_s0[st];
                                sn_ptr[m] = tile_sn[st];
                                m += (d2buf[st] <= r_c_sq);
                            }

                            if (phase_timing) {
                                const double now = omp_get_wtime();
                                tt += now - _t;
                                _t = now;
                            }

                            range_evaluator(rsc, cen, r_c_sq, Real(0), n_src, r_src_ptr, charge_ptr, nullptr, m, s0_ptr,
                                            sn_ptr, tn, tptr, pg_sorted.data() + out_dim * (hbeg + t0), nullptr,
                                            nullptr);

                            if (phase_timing) {
                                const double now = omp_get_wtime();
                                te += now - _t;
                                _t = now;
                            }
                        }
                    }

        if (phase_timing) {
#pragma omp critical
            {
                sum_gather += tg;
                sum_aabb += ta;
                sum_test += tt;
                sum_eval += te;
            }
        }
    }

    if (phase_timing)
        std::printf("# esp phase_timing (thread-summed s): gather=%.4f aabb=%.4f test=%.4f eval=%.4f\n", sum_gather,
                    sum_aabb, sum_test, sum_eval);

    for (int a = 0; a < n; ++a) {
        const int orig = cl.orig[a];
        pot[orig] += pg_sorted[out_dim * a + 0];
        if (want_force) {
            const Real q = cl.qs[a];
            fx[orig] += -q * pg_sorted[out_dim * a + 1];
            fy[orig] += -q * pg_sorted[out_dim * a + 2];
            fz[orig] += -q * pg_sorted[out_dim * a + 3];
        }
    }
}

static inline double S_hat(const PSWFKernel &pswf, const ESPParams &params, const Vec3 &k_vec) {
    double k_mag = std::sqrt(k_vec[0] * k_vec[0] + k_vec[1] * k_vec[1] + k_vec[2] * k_vec[2]);
    return pswf.pswf_hat(k_mag * params.r_c) / (2.0 * k_mag * k_mag) / params.c0;
}

static DGrid precompute_scaling_coefficients(const PSWFKernel &pswf, const ESPParams &params) {
    int nf = params.n_f;
    std::vector<int> k_idx(nf);
    for (int i = 0; i < nf; ++i)
        k_idx[i] = (i <= nf / 2) ? i : i - nf;

    // 1-D phi_hat values
    std::vector<double> phi_hat_1d(nf);
    for (int i = 0; i < nf; ++i) {
        double k_vec = 2.0 * M_PI * k_idx[i] / params.L;
        double arg = k_vec * (params.P * params.h) / 2.0;
        phi_hat_1d[i] = (params.P * params.h / 2.0) * pswf.pswf_hat(arg);
    }

    DGrid p(nf * nf * nf, 0.0);
    for (int ix = 0; ix < nf; ++ix)
        for (int iy = 0; iy < nf; ++iy)
            for (int iz = 0; iz < nf; ++iz) {
                Vec3 k_vec = {2.0 * M_PI * k_idx[ix] / params.L, 2.0 * M_PI * k_idx[iy] / params.L,
                              2.0 * M_PI * k_idx[iz] / params.L};
                if (k_vec[0] == 0.0 && k_vec[1] == 0.0 && k_vec[2] == 0.0)
                    continue;

                double s = S_hat(pswf, params, k_vec);
                double ph = phi_hat_1d[ix] * phi_hat_1d[iy] * phi_hat_1d[iz];
                p[grid_idx(ix, iy, iz, nf)] =
                    s / (params.L * params.L * params.L * ph * ph * static_cast<double>(nf * nf * nf));
            }
    return p;
}

// Long-range contribution via FINUFFT spreading/interpolation
template <typename Real>
static void long_range(const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges, const PSWFKernel &pswf,
                       const ESPParams &params, const DGrid &scaling_coeffs, dmk_eval_type eval_type,
                       std::span<Real> pot, std::span<Real> fx, std::span<Real> fy, std::span<Real> fz) {
    sctl::Profile::Scoped long_range("long_range");
    const bool want_force = (eval_type >= DMK_POTENTIAL_GRAD);
    int n = params.n;
    int nf = params.n_f;
    int ntot = nf * nf * nf;

    double scale = 2.0 * M_PI / params.L;
    std::vector<double> x(n), y(n), z(n);
    for (int j = 0; j < n; j++) {
        x[j] = double(r_src[j][0]) * scale;
        y[j] = double(r_src[j][1]) * scale;
        z[j] = double(r_src[j][2]) * scale;
    }

    std::vector<std::complex<double>> c(n);
    for (int j = 0; j < n; j++)
        c[j] = {double(charges[j]), 0.0};

    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.spreadinterponly = 1;
    opts.upsampfac = params.sigma;
    double tol = pswf.eps;

    // 1. Spread: NU points -> uniform grid (type 1)
    std::vector<std::complex<double>> b(ntot, 0.0);
    int ier = finufft3d1(n, x.data(), y.data(), z.data(), c.data(), +1, tol, nf, nf, nf, b.data(), &opts);
    if (ier > 1)
        throw std::runtime_error("finufft3d1 spread failed, ier=" + std::to_string(ier));

    // 2. Forward FFT
    CGrid b_hat(ntot);
    fftn_3d(b, b_hat, nf);

    // 3. Diagonal scaling (precomputed in plan)
    CGrid pot_hat(ntot);
#pragma omp parallel for
    for (int idx = 0; idx < ntot; ++idx)
        pot_hat[idx] = b_hat[idx] * scaling_coeffs[idx];

    // 4. Inverse FFT
    CGrid grid_pot(ntot);
    ifftn_3d(pot_hat, grid_pot, nf);

    // 5. Interpolate: uniform grid -> NU points (type 2)
    std::vector<std::complex<double>> pot_c(n);
    ier = finufft3d2(n, x.data(), y.data(), z.data(), pot_c.data(), +1, tol, nf, nf, nf, grid_pot.data(), &opts);
    if (ier > 1)
        throw std::runtime_error("finufft3d2 interp failed, ier=" + std::to_string(ier));

    for (int j = 0; j < n; j++)
        pot[j] += Real(pot_c[j].real());

    if (!want_force)
        return;

    std::vector<int> k_idx(nf);
    for (int i = 0; i < nf; ++i)
        k_idx[i] = (i <= nf / 2) ? i : i - nf;

    // ik method: F = -q*grad(u), and grad(u)_hat_k = i*k*u_hat_k, so the force spectrum is
    // obtained from the potential spectrum (b_hat * scaling_coeffs, i.e. pot_hat) by multiplying
    // by -i*k component-wise.
    std::complex<double> coeff_grad = std::complex<double>(0, 1) * (2.0 * M_PI / params.L);
    CGrid f_hat_x(ntot), f_hat_y(ntot), f_hat_z(ntot);
#pragma omp parallel for collapse(3)
    for (int ix = 0; ix < nf; ++ix)
        for (int iy = 0; iy < nf; ++iy)
            for (int iz = 0; iz < nf; ++iz) {
                const int idx = grid_idx(ix, iy, iz, nf);
                const std::complex<double> scaled = b_hat[idx] * scaling_coeffs[idx];
                // DMK's grid_idx (row-major, z fastest) and FINUFFT's internal grid storage
                // (column-major, x fastest) disagree on which loop variable is which physical axis;
                // ix here lines up with FINUFFT's z slot and iz lines up with its x slot
                f_hat_x[idx] = scaled * coeff_grad * double(k_idx[iz]);
                f_hat_y[idx] = scaled * coeff_grad * double(k_idx[iy]);
                f_hat_z[idx] = scaled * coeff_grad * double(k_idx[ix]);
            }

    CGrid grid_force_x(ntot), grid_force_y(ntot), grid_force_z(ntot);
    ifftn_3d(f_hat_x, grid_force_x, nf);
    ifftn_3d(f_hat_y, grid_force_y, nf);
    ifftn_3d(f_hat_z, grid_force_z, nf);

    std::vector<std::complex<double>> force_x_c(n), force_y_c(n), force_z_c(n);
    ier =
        finufft3d2(n, x.data(), y.data(), z.data(), force_x_c.data(), +1, tol, nf, nf, nf, grid_force_x.data(), &opts);
    if (ier > 1)
        throw std::runtime_error("finufft3d2 interp failed, ier=" + std::to_string(ier));
    ier =
        finufft3d2(n, x.data(), y.data(), z.data(), force_y_c.data(), +1, tol, nf, nf, nf, grid_force_y.data(), &opts);
    if (ier > 1)
        throw std::runtime_error("finufft3d2 interp failed, ier=" + std::to_string(ier));
    ier =
        finufft3d2(n, x.data(), y.data(), z.data(), force_z_c.data(), +1, tol, nf, nf, nf, grid_force_z.data(), &opts);
    if (ier > 1)
        throw std::runtime_error("finufft3d2 interp failed, ier=" + std::to_string(ier));

    for (int j = 0; j < n; j++) {
        fx[j] += -charges[j] * Real(force_x_c[j].real());
        fy[j] += -charges[j] * Real(force_y_c[j].real());
        fz[j] += -charges[j] * Real(force_z_c[j].real());
    }
}

// Self-interaction correction — subtracts directly from the provided potential span.
template <typename Real>
static void self_interaction(const std::vector<Real> &charges, const PSWFKernel &pswf, const ESPParams &params,
                             std::span<Real> pot) {
    Real factor = Real(pswf(0.0) / (params.r_c * 4.0 * M_PI * params.c0));
    for (int i = 0; i < params.n; ++i)
        pot[i] -= charges[i] * factor;
}

struct EspPlan {
    int n_digits;
    PSWFKernel pswf;
    ESPParams params_base;
    DGrid scaling_coeffs;
    dmk_eval_type eval_type;     // baked in at creation; DMK_POTENTIAL skips all force computation
    std::vector<double> dbl_buf; // reused output workspace for esp_eval<double>
    std::vector<float> flt_buf;  // reused output workspace for esp_eval<float>

    EspPlan(double L, double r_c, double eps, double sigma, dmk_eval_type eval_type_)
        : n_digits(esp_digits_from_eps(eps)), eval_type(eval_type_) {
        const double eps_d = std::pow(10.0, -double(n_digits));
        const int P = esp_P_from_eps(eps_d, sigma);
        const double c = esp_pswf_c_from_P(sigma, P);
        pswf = PSWFKernel(eps_d, c);
        params_base = ESPParams(L, r_c, sigma, P, pswf, 0);
        scaling_coeffs = precompute_scaling_coefficients(pswf, params_base);
    }
};

EspPlan *esp_create_plan(double L, double r_c, double eps, double sigma, dmk_eval_type eval_type) {
    return new EspPlan(L, r_c, eps, sigma, eval_type);
}

void esp_destroy_plan(EspPlan *plan) { delete plan; }

template <typename Real>
PotForce<Real> esp_eval(EspPlan *plan, const std::vector<Vec3T<Real>> &r_src, const std::vector<Real> &charges) {
    sctl::Profile::Scoped esp_eval("esp_eval");
    const int n = static_cast<int>(charges.size());
    ESPParams params = plan->params_base;
    params.n = n;
    const bool want_force = (plan->eval_type >= DMK_POTENTIAL_GRAD);
    const int slots = want_force ? 4 : 1;

    // Reuse the plan's typed workspace; zero-initialize the active region.
    auto &buf = [&]() -> std::vector<Real> & {
        if constexpr (std::is_same_v<Real, double>)
            return plan->dbl_buf;
        else
            return plan->flt_buf;
    }();
    if (static_cast<int>(buf.size()) < slots * n)
        buf.resize(slots * n);
    std::fill(buf.begin(), buf.begin() + slots * n, Real(0));

    std::span<Real> pot_sp(buf.data(), n);
    std::span<Real> fx_sp(buf.data() + n, want_force ? n : 0);
    std::span<Real> fy_sp(buf.data() + 2 * n, want_force ? n : 0);
    std::span<Real> fz_sp(buf.data() + 3 * n, want_force ? n : 0);

    short_range<Real>(r_src, charges, params, plan->n_digits, plan->pswf, plan->eval_type, pot_sp, fx_sp, fy_sp, fz_sp);
    long_range<Real>(r_src, charges, plan->pswf, params, plan->scaling_coeffs, plan->eval_type, pot_sp, fx_sp, fy_sp,
                     fz_sp);
    self_interaction<Real>(charges, plan->pswf, params, pot_sp);

    return {pot_sp, fx_sp, fy_sp, fz_sp};
}

template PotForce<float> esp_eval<float>(EspPlan *, const std::vector<Vec3T<float>> &, const std::vector<float> &);
template PotForce<double> esp_eval<double>(EspPlan *, const std::vector<Vec3T<double>> &, const std::vector<double> &);

} // namespace dmk
