#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/omp_wrapper.hpp>
#include <dmk/periodic_reference.hpp>
#include <dmk/util.hpp>
#include <sctl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef DMK_GPU_OFFLOAD
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#endif

#ifdef DMK_HAVE_MPI
#include <mpi.h>
#define MYCOMM MPI_COMM_WORLD
#else
#define MYCOMM nullptr
#endif

enum class Solver { DMK, ESP };

typedef enum : int {
    DMK_UNIFORM = 0,
    DMK_NSPHERESURFACE = 1,
    DMK_NCUBEPARTIALFACET = 2,
} dmk_distribution;

struct Config {
    Solver solver = Solver::DMK;

    // Shared
    int n_src = 100'000;
    int n_dim = 3;
    double eps = 1e-5;
    char prec = 'f';
    dmk_ikernel kernel = DMK_LAPLACE;
    double fparam = 6.0;
    int n_runs = 100;
    int n_direct = 10'000;
    int log_level = DMK_LOG_OFF;
    bool with_grad = false;
    dmk_eval_path eval_path = DMK_EVAL_PATH_CPU;
    int gpu_device_id = 0;
    bool use_periodic = false;

    // DMK only
    int n_trg = 0;
    int n_per_leaf = 280;
    dmk_distribution dist = DMK_UNIFORM;
    bool enable_direct = true;
    long seed = 0;
    int n_show_outliers = 0; // print top-N worst points per block to stderr (0 = off)
    bool pin_host = false;
    bool bench_build = false;
    bool bench_eval = true;
    bool bench_update_charges = false;

    // ESP only
    double L = 1.0;
    double r_c = -1.0;
    double sigma = 1.35;
    bool sigma_set = false;
    bool bench_plan = false;
    bool bench_forces = false;
    bool check_forces = false;
    bool skip_cpu_baseline = false;
    double freespace_pad = 0;
    uint32_t esp_flags = DMK_ESP_PRUNE_SOURCE | DMK_ESP_N3L | DMK_ESP_MORTON;
    int esp_bins = 2;
    int esp_stile = 0;
};

// ---------------------------------------------------------------------------
// Shared kernel traits, timing and error metrics
// ---------------------------------------------------------------------------

inline bool is_velocity_kernel(dmk_ikernel k) { return k == DMK_STOKESLET || k == DMK_STRESSLET; }
inline bool is_scalar_kernel(dmk_ikernel k) { return k == DMK_LAPLACE || k == DMK_YUKAWA || k == DMK_SQRT_LAPLACE; }
inline bool needs_normal(dmk_ikernel k) { return k == DMK_STRESSLET; }

inline dmk_eval_type get_eval_type(dmk_ikernel kernel, bool with_grad) {
    if (is_velocity_kernel(kernel))
        return DMK_VELOCITY;
    return with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
}

struct TimingResult {
    double elapsed;
    double pts_per_sec;
    double pts_per_sec_per_rank;
    double pts_per_sec_per_thread;
};

struct ErrorMetrics {
    double l2_rel;
    double max_rel;
};

inline int local_count(int n, int np, int r) { return n / np + (r < (n % np) ? 1 : 0); }

TimingResult make_timing(double elapsed, int n_total, int n_per_rank, int n_threads) {
    return {elapsed, n_total / elapsed, n_per_rank / elapsed, n_per_rank / elapsed / n_threads};
}

dmk_ikernel parse_kernel(const char *s) {
    if (auto kernel = dmk::util::ikernel_from_string(s))
        return *kernel;
    throw std::runtime_error("Unknown kernel: " + std::string(s));
}

template <typename Real>
ErrorMetrics compute_error(const std::vector<Real> &computed, const std::vector<Real> &reference, int rank, int np,
                           int kdim = 1, int comp_begin = 0, int comp_end = -1) {
    if (comp_end < 0)
        comp_end = kdim;
    double local_err2 = 0.0, local_ref2 = 0.0, local_maxre = 0.0;

    const size_t n_pts = reference.size() / kdim;
    for (size_t p = 0; p < n_pts; ++p) {
        for (int c = comp_begin; c < comp_end; ++c) {
            const size_t i = p * kdim + c;
            double diff = double(computed[i]) - double(reference[i]);
            double ref = double(reference[i]);
            local_err2 += diff * diff;
            local_ref2 += ref * ref;
            if (std::abs(ref) > 0.0)
                local_maxre = std::max(local_maxre, std::abs(diff / ref));
        }
    }

#ifdef DMK_HAVE_MPI
    double glob_err2 = 0.0, glob_ref2 = 0.0, glob_maxre = 0.0;
    MPI_Allreduce(&local_err2, &glob_err2, 1, MPI_DOUBLE, MPI_SUM, MYCOMM);
    MPI_Allreduce(&local_ref2, &glob_ref2, 1, MPI_DOUBLE, MPI_SUM, MYCOMM);
    MPI_Allreduce(&local_maxre, &glob_maxre, 1, MPI_DOUBLE, MPI_MAX, MYCOMM);
#else
    double glob_err2 = local_err2, glob_ref2 = local_ref2, glob_maxre = local_maxre;
#endif

    return {std::sqrt(glob_err2 / glob_ref2), glob_maxre};
}

template <typename Real>
double l2_rel_err(std::span<const Real> pot, const std::vector<double> &ref) {
    const int n = static_cast<int>(ref.size());
    double err2 = 0, ref2 = 0;
    for (int i = 0; i < n; ++i) {
        const double d = double(pot[i]) - ref[i];
        err2 += d * d;
        ref2 += ref[i] * ref[i];
    }
    return std::sqrt(err2 / ref2);
}

// ESP potentials of a charge-neutral system carry an arbitrary additive constant, so both sides are
// mean-subtracted before comparison.
template <typename Real>
double l2_rel_err_gauged(const std::vector<Real> &a, const std::vector<Real> &b) {
    const int n = static_cast<int>(std::min(a.size(), b.size()));
    if (n == 0)
        return std::numeric_limits<double>::quiet_NaN();
    double a_mean = 0, b_mean = 0;
    for (int i = 0; i < n; ++i) {
        a_mean += double(a[i]);
        b_mean += double(b[i]);
    }
    a_mean /= n;
    b_mean /= n;
    double err2 = 0, ref2 = 0;
    for (int i = 0; i < n; ++i) {
        const double d = (double(a[i]) - a_mean) - (double(b[i]) - b_mean);
        const double r = double(b[i]) - b_mean;
        err2 += d * d;
        ref2 += r * r;
    }
    return std::sqrt(err2 / ref2);
}

// ---------------------------------------------------------------------------
// DMK point tree
// ---------------------------------------------------------------------------

// Page-locks a caller-owned output buffer. pdmk_tree_eval takes a host pointer, so the GPU path's
// result copy is staged by the driver unless the caller's memory is already page-locked.
template <typename Real>
bool pin_host_buffer([[maybe_unused]] std::vector<Real> &buf, bool enable) {
    if (!enable || buf.empty())
        return false;
#ifdef DMK_GPU_OFFLOAD
    const cudaError_t rc = cudaHostRegister(buf.data(), buf.size() * sizeof(Real), cudaHostRegisterDefault);
    if (rc == cudaSuccess)
        return true;
    std::cerr << "warning: cudaHostRegister failed (" << cudaGetErrorString(rc) << "), continuing unpinned\n";
#else
    std::cerr << "warning: --pin needs a DMK_GPU_OFFLOAD build, continuing unpinned\n";
#endif
    return false;
}

template <typename Real>
void unpin_host_buffer([[maybe_unused]] std::vector<Real> &buf, [[maybe_unused]] bool pinned) {
#ifdef DMK_GPU_OFFLOAD
    if (pinned)
        cudaHostUnregister(buf.data());
#endif
}

// Brute-force nearest-source distance to a point at r_pts[p*n_dim..]. O(n_src). Diagnostics
// only. If r_pts and r_src are the same array, also skip the self-pair (R == 0).
template <typename Real>
double nearest_source_dist(const std::vector<Real> &r_pts, int p, const std::vector<Real> &r_src, int n_dim,
                           bool same_set) {
    const int n_src = int(r_src.size() / n_dim);
    double best2 = std::numeric_limits<double>::infinity();
    for (int s = 0; s < n_src; ++s) {
        double d2 = 0.0;
        for (int d = 0; d < n_dim; ++d) {
            const double dx = double(r_pts[p * n_dim + d]) - double(r_src[s * n_dim + d]);
            d2 += dx * dx;
        }
        if (same_set && d2 == 0.0)
            continue; // self pair
        if (d2 < best2)
            best2 = d2;
    }
    return std::sqrt(best2);
}

template <typename Real>
void print_outliers(const std::vector<Real> &computed, const std::vector<Real> &reference,
                    const std::vector<Real> &r_pts, const std::vector<Real> &r_src, bool same_set,
                    const std::vector<Real> &pt_charges, int charge_dim, int n_dim, int kdim, int comp_begin,
                    int comp_end, int n_show, const std::string &label, int rank, std::ostream &os) {
    if (rank != 0 || n_show <= 0 || reference.empty() || computed.empty())
        return;

    const int n_pts = std::min(computed.size() / kdim, reference.size() / kdim);
    std::vector<std::pair<double, int>> ranked; // (|diff|, point_index)
    ranked.reserve(n_pts);
    for (int p = 0; p < n_pts; ++p) {
        double diff2 = 0.0;
        for (int c = comp_begin; c < comp_end; ++c) {
            const double d = double(computed[p * kdim + c]) - double(reference[p * kdim + c]);
            diff2 += d * d;
        }
        ranked.emplace_back(std::sqrt(diff2), p);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) { return a.first > b.first; });

    const bool has_charges = !pt_charges.empty() && charge_dim > 0;
    const int n = std::min(n_show, int(ranked.size()));
    os << "# " << label << " top " << n << " outliers (by |dmk - ref| over comps " << comp_begin << ".." << comp_end
       << "):\n";
    std::vector<double> nn_outliers(n);
    for (int k = 0; k < n; ++k) {
        const int p = ranked[k].second;
        const double absdiff = ranked[k].first;

        double refnorm = 0.0;
        for (int c = comp_begin; c < comp_end; ++c) {
            const double e = double(reference[p * kdim + c]);
            refnorm += e * e;
        }
        refnorm = std::sqrt(refnorm);
        const double relerr = refnorm > 0 ? absdiff / refnorm : std::numeric_limits<double>::infinity();

        const double r_nn = nearest_source_dist(r_pts, p, r_src, n_dim, same_set);
        nn_outliers[k] = r_nn;

        os << "#   [" << k << "] idx=" << p << " pos=(";
        for (int d = 0; d < n_dim; ++d)
            os << (d ? ", " : "") << double(r_pts[p * n_dim + d]);
        os << ") |diff|=" << absdiff << " |ref|=" << refnorm << " relerr=" << relerr;
        os << " r_nn_src=" << r_nn;
        os << "\n#       dmk=(";
        for (int c = comp_begin; c < comp_end; ++c)
            os << (c > comp_begin ? ", " : "") << double(computed[p * kdim + c]);
        os << ")\n#       ref=(";
        for (int c = comp_begin; c < comp_end; ++c)
            os << (c > comp_begin ? ", " : "") << double(reference[p * kdim + c]);
        os << ")\n";

        // If the point has an associated input vector (e.g. source dipole), print it and the
        // per-component ratio (diff_i / d_i). For an additive self-correction `c · d`, these
        // ratios should equal a single constant across all sources.
        if (has_charges) {
            os << "#       d=(";
            for (int c = 0; c < charge_dim; ++c)
                os << (c ? ", " : "") << double(pt_charges[p * charge_dim + c]);
            os << ")";
            // Ratio diff_i / d_i across the matching component range. Best when
            // (comp_end - comp_begin) == charge_dim (e.g. dipole grad: 3 grad components, 3 d).
            const int n_match = std::min(charge_dim, comp_end - comp_begin);
            if (n_match > 0) {
                os << "\n#       diff_i / d_i = (";
                for (int i = 0; i < n_match; ++i) {
                    const double d_i = double(pt_charges[p * charge_dim + i]);
                    const double diff_i =
                        double(computed[p * kdim + (comp_begin + i)]) - double(reference[p * kdim + (comp_begin + i)]);
                    const double ratio = (std::abs(d_i) > 0) ? diff_i / d_i : std::numeric_limits<double>::quiet_NaN();
                    os << (i ? ", " : "") << ratio;
                }
                os << ")";
            }
            os << "\n";
        }
    }

    // Summary: compare outlier nn-distance distribution vs the full set.
    const int sample = std::min(n_pts, 1024);
    std::vector<double> nn_all;
    nn_all.reserve(sample);
    for (int p = 0; p < sample; ++p)
        nn_all.push_back(nearest_source_dist(r_pts, p, r_src, n_dim, same_set));
    std::sort(nn_outliers.begin(), nn_outliers.end());
    std::sort(nn_all.begin(), nn_all.end());
    auto med = [](const std::vector<double> &v) { return v.empty() ? 0.0 : v[v.size() / 2]; };
    os << "#   nn_src distance summary: outliers min=" << (nn_outliers.empty() ? 0.0 : nn_outliers.front())
       << " median=" << med(nn_outliers) << " | sample(" << sample
       << ") min=" << (nn_all.empty() ? 0.0 : nn_all.front()) << " median=" << med(nn_all) << "\n";
}

template <typename Real>
void generate_and_scatter(int n_dim, int charge_dim, size_t n_src, size_t n_trg, dmk_distribution dist,
                          bool set_fixed_charges, std::vector<Real> &r_src, std::vector<Real> &r_trg,
                          std::vector<Real> &charges, std::vector<Real> &normals, long seed, int rank, int np) {
    std::vector<Real> r_src_all, r_trg_all, charges_all, normals_all;

    if (rank == 0) {
        using namespace dmk::util;
        constexpr Real almost_one = 1.0 - std::numeric_limits<Real>::epsilon();
        switch (dist) {
        case DMK_UNIFORM:
            init_test_data(n_dim, charge_dim, int(n_src), int(n_trg), UniformVolume<Real>(n_dim, almost_one, seed),
                           set_fixed_charges, r_src_all, r_trg_all, normals_all, charges_all);
            break;
        case DMK_NSPHERESURFACE:
            init_test_data(n_dim, charge_dim, int(n_src), int(n_trg),
                           dmk::util::NSphereSurface<Real>(n_dim, 0.95 * 0.5, seed), set_fixed_charges, r_src_all,
                           r_trg_all, normals_all, charges_all);
            break;
        case DMK_NCUBEPARTIALFACET:
            init_test_data(n_dim, charge_dim, int(n_src), int(n_trg),
                           dmk::util::NCubePartialFacet<Real>(n_dim, 0.95, 0.02, seed), set_fixed_charges, r_src_all,
                           r_trg_all, normals_all, charges_all);
            break;
        }
    }

#ifdef DMK_HAVE_MPI
    const auto mpi_t = std::is_same_v<Real, float> ? MPI_FLOAT : MPI_DOUBLE;
    auto scatter = [&](const std::vector<Real> &src_all, std::vector<Real> &dst_local, size_t n_total, int stride) {
        std::vector<int> counts(np), displs(np);
        for (int i = 0; i < np; ++i)
            counts[i] = local_count(n_total, np, i) * stride;
        displs[0] = 0;
        for (int i = 1; i < np; ++i)
            displs[i] = displs[i - 1] + counts[i - 1];
        dst_local.resize(size_t(local_count(n_total, np, rank)) * stride);
        MPI_Scatterv(rank == 0 ? const_cast<Real *>(src_all.data()) : nullptr, counts.data(), displs.data(), mpi_t,
                     dst_local.data(), int(dst_local.size()), mpi_t, 0, MYCOMM);
    };
    scatter(r_src_all, r_src, n_src, n_dim);
    scatter(r_trg_all, r_trg, n_trg, n_dim);
    scatter(charges_all, charges, n_src, charge_dim);
    scatter(normals_all, normals, n_src, n_dim);
#else
    r_src = std::move(r_src_all);
    r_trg = std::move(r_trg_all);
    charges = std::move(charges_all);
    normals = std::move(normals_all);
#endif
}

template <typename Real>
void run_direct(const Config &cfg, int n_dim, int charge_dim, const std::vector<Real> &r_src,
                const std::vector<Real> &charges, const std::vector<Real> &normals, const std::vector<Real> &r_trg,
                std::vector<Real> &pot, int rank, int np) {
    const int n_src_local = r_src.size() / n_dim;
    int n_trg_local = r_trg.size() / n_dim;

#ifdef DMK_HAVE_MPI
    // Gather all sources to all ranks
    int n_src_global = 0;
    MPI_Allreduce(&n_src_local, &n_src_global, 1, MPI_INT, MPI_SUM, MYCOMM);

    auto mpi_t = std::is_same_v<Real, float> ? MPI_FLOAT : MPI_DOUBLE;

    std::vector<int> recv_cnts_r(np), recv_disp_r(np);
    std::vector<int> recv_cnts_c(np), recv_disp_c(np);
    std::vector<int> recv_cnts_n(np), recv_disp_n(np);
    {
        int send_cnt_r = n_src_local * n_dim;
        int send_cnt_n = n_src_local * n_dim;
        int send_cnt_c = n_src_local * charge_dim;
        MPI_Allgather(&send_cnt_r, 1, MPI_INT, recv_cnts_r.data(), 1, MPI_INT, MYCOMM);
        MPI_Allgather(&send_cnt_n, 1, MPI_INT, recv_cnts_n.data(), 1, MPI_INT, MYCOMM);
        MPI_Allgather(&send_cnt_c, 1, MPI_INT, recv_cnts_c.data(), 1, MPI_INT, MYCOMM);
        recv_disp_r[0] = recv_disp_c[0] = recv_disp_n[0] = 0;
        for (int i = 1; i < np; ++i) {
            recv_disp_r[i] = recv_disp_r[i - 1] + recv_cnts_r[i - 1];
            recv_disp_n[i] = recv_disp_n[i - 1] + recv_cnts_n[i - 1];
            recv_disp_c[i] = recv_disp_c[i - 1] + recv_cnts_c[i - 1];
        }
    }

    std::vector<Real> glb_r_src(n_src_global * n_dim);
    std::vector<Real> glb_normals(n_src_global * n_dim);
    std::vector<Real> glb_charges(n_src_global * charge_dim);
    MPI_Allgatherv(r_src.data(), n_src_local * n_dim, mpi_t, glb_r_src.data(), recv_cnts_r.data(), recv_disp_r.data(),
                   mpi_t, MYCOMM);
    MPI_Allgatherv(normals.data(), n_src_local * n_dim, mpi_t, glb_normals.data(), recv_cnts_n.data(),
                   recv_disp_n.data(), mpi_t, MYCOMM);
    MPI_Allgatherv(charges.data(), n_src_local * charge_dim, mpi_t, glb_charges.data(), recv_cnts_c.data(),
                   recv_disp_c.data(), mpi_t, MYCOMM);
#else
    int n_src_global = n_src_local;
    const auto &glb_r_src = r_src;
    const auto &glb_charges = charges;
    const auto &glb_normals = normals;
#endif

    // Convert sources to double for reference evaluation
    std::vector<double> r_src_d(glb_r_src.begin(), glb_r_src.end());
    std::vector<double> normals_d(glb_normals.begin(), glb_normals.end());
    std::vector<double> charges_d(glb_charges.begin(), glb_charges.end());
    std::vector<double> r_trg_d(r_trg.begin(), r_trg.end());

    // Evaluate: each rank handles its own local targets
    const auto eval_level = get_eval_type(cfg.kernel, cfg.with_grad);
    const int kdim = dmk::get_kernel_output_dim(n_dim, cfg.kernel, eval_level);
    std::vector<double> pot_d(n_trg_local * kdim, 0.0);

    const auto eval = dmk::get_direct_evaluator<double>(cfg.kernel, eval_level, n_dim, cfg.fparam);
    dmk::parallel_direct_eval<double>(eval, n_src_global, r_src_d.data(), charges_d.data(), normals_d.data(),
                                      n_trg_local, r_trg_d.data(), pot_d.data(), n_dim, kdim);

    pot.resize(n_trg_local * kdim);
    for (size_t i = 0; i < pot_d.size(); ++i)
        pot[i] = pot_d[i];
}

template <typename Real>
double run_dmk(pdmk_tree tree, std::vector<Real> &pot_src, std::vector<Real> &pot_trg, int n_src_per_rank,
               int n_trg_per_rank, int kdim, int rank, int np) {
    pot_src.resize(size_t(n_src_per_rank) * kdim);
    pot_trg.resize(size_t(n_trg_per_rank) * kdim);

#ifdef DMK_HAVE_MPI
    MPI_Barrier(MYCOMM);
#endif

    Real *pot_trg_ptr = n_trg_per_rank > 0 ? pot_trg.data() : nullptr;
    double st = MY_OMP_GET_WTIME();
    int rc;
    if constexpr (std::is_same_v<Real, float>)
        rc = pdmk_tree_evalf(tree, pot_src.data(), pot_trg_ptr);
    else
        rc = pdmk_tree_eval(tree, pot_src.data(), pot_trg_ptr);
    double ft = MY_OMP_GET_WTIME();

    if (rc != 0) {
        std::cerr << "pdmk_tree_eval failed with rc=" << rc << ": " << pdmk_last_error_message() << "\n";
        std::exit(1);
    }
    return ft - st;
}

template <typename Real>
double run_update_charges(pdmk_tree tree, const std::vector<Real> &charges, const Real *normal) {
#ifdef DMK_HAVE_MPI
    MPI_Barrier(MYCOMM);
#endif

    double st = omp_get_wtime();
    int rc;
    if constexpr (std::is_same_v<Real, float>)
        rc = pdmk_tree_update_chargesf(tree, charges.data(), normal);
    else
        rc = pdmk_tree_update_charges(tree, charges.data(), normal);
    double ft = omp_get_wtime();

    if (rc != 0) {
        std::cerr << "pdmk_tree_update_charges failed with rc=" << rc << ": " << pdmk_last_error_message() << "\n";
        std::exit(1);
    }
    return ft - st;
}

void print_build_csv_header(std::ostream &os) { os << "build_time,build_pts_s,build_pts_s_rank,build_pts_s_thread"; }

void print_build_csv_row(const TimingResult &t, std::ostream &os) {
    os << t.elapsed << "," << t.pts_per_sec << "," << t.pts_per_sec_per_rank << "," << t.pts_per_sec_per_thread;
}

void print_update_csv_header(std::ostream &os) {
    os << "update_time,update_pts_s,update_pts_s_rank,update_pts_s_thread";
}

void print_update_csv_row(const TimingResult &t, std::ostream &os) {
    os << t.elapsed << "," << t.pts_per_sec << "," << t.pts_per_sec_per_rank << "," << t.pts_per_sec_per_thread;
}

void print_dmk_config_comment(const Config &cfg, int np, int n_threads, std::ostream &os) {
    const std::string_view kernel_str = dmk::util::to_string(cfg.kernel);
    os << "# mpi_ranks:            " << np << "\n"
       << "# omp_threads_per_rank: " << n_threads << "\n"
       << "# n_src:                " << cfg.n_src << "\n"
       << "# n_trg:                " << cfg.n_trg << "\n"
       << "# n_dim:                " << cfg.n_dim << "\n"
       << "# kernel:               " << kernel_str << "\n"
       << "# fparam:               " << cfg.fparam << "\n"
       << "# with_grad:            " << cfg.with_grad << "\n"
       << "# precision:            " << (cfg.prec == 'd' ? "double" : "float") << "\n"
       << "# dist:                 " << cfg.dist << "\n"
       << "# seed:                 " << cfg.seed << "\n"
       << "# eps:                  " << cfg.eps << "\n"
       << "# n_per_leaf:           " << cfg.n_per_leaf << "\n"
       << "# n_runs:               " << cfg.n_runs << "\n"
       << "# direct_enabled:       " << cfg.enable_direct << "\n"
       << "# n_direct:             " << cfg.n_direct << "\n"
       << "# n_show_outliers:      " << cfg.n_show_outliers << "\n"
       << "# log_level:            " << cfg.log_level << "\n"
       << "# eval_path:            " << cfg.eval_path << "\n"
       << "# bench_build:          " << cfg.bench_build << "\n"
       << "# bench_eval:           " << cfg.bench_eval << "\n"
       << "# bench_update_charges: " << cfg.bench_update_charges << "\n";
}

struct ErrorBlock {
    bool have = false;
    ErrorMetrics pot{};
    ErrorMetrics grad{}; // only meaningful when with_grad
};

void print_csv_header_block(std::ostream &os, const std::string &prefix, bool with_grad) {
    os << "," << prefix << "_l2_rel_err," << prefix << "_max_rel_err";
    if (with_grad)
        os << "," << prefix << "_l2_rel_err_grad," << prefix << "_max_rel_err_grad";
}

void print_csv_block(std::ostream &os, const ErrorBlock &b, bool with_grad) {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    auto emit = [&](const ErrorMetrics &e) {
        if (b.have)
            os << "," << e.l2_rel << "," << e.max_rel;
        else
            os << "," << nan << "," << nan;
    };
    emit(b.pot);
    if (with_grad)
        emit(b.grad);
}

void print_csv_header(std::ostream &os, bool with_grad, bool with_trg) {
    os << "dmk_time,dmk_pts_s,dmk_pts_s_rank,dmk_pts_s_thread";
    print_csv_header_block(os, "src", with_grad);
    if (with_trg)
        print_csv_header_block(os, "trg", with_grad);
}

void print_csv_row(const TimingResult &t, const ErrorBlock &src, const ErrorBlock *trg, bool with_grad,
                   std::ostream &os) {
    os << t.elapsed << "," << t.pts_per_sec << "," << t.pts_per_sec_per_rank << "," << t.pts_per_sec_per_thread;
    print_csv_block(os, src, with_grad);
    if (trg)
        print_csv_block(os, *trg, with_grad);
}

template <typename Real>
void run_dmk_benchmark(const Config &cfg) {
    int rank = 0, np = 1;
#ifdef DMK_HAVE_MPI
    MPI_Comm_rank(MYCOMM, &rank);
    MPI_Comm_size(MYCOMM, &np);
#endif

    const int n_dim = cfg.n_dim;
    const int n_threads = MY_OMP_GET_MAX_THREADS();
    const int n_src = cfg.n_src;
    const int n_trg = cfg.n_trg;
    const int n_src_per_rank = local_count(n_src, np, rank);
    const int n_trg_per_rank = local_count(n_trg, np, rank);
    const bool with_trg = n_trg > 0;

    pdmk_params params{};
    params.eps = cfg.eps;
    params.n_dim = n_dim;
    params.n_per_leaf = cfg.n_per_leaf;
    params.log_level = cfg.log_level;
    params.kernel = cfg.kernel;
    params.eval_path = cfg.eval_path;
    params.gpu_device_id = cfg.gpu_device_id;
    params.use_periodic = cfg.use_periodic;
    params.eval_src = get_eval_type(cfg.kernel, cfg.with_grad);
    params.eval_trg = params.eval_src;
    if (cfg.kernel == DMK_YUKAWA)
        params.fparam = cfg.fparam;

    const int charge_dim = dmk::get_kernel_input_dim(n_dim, params.kernel);
    const int pot_dim = dmk::get_kernel_output_dim(n_dim, cfg.kernel, params.eval_src);

    std::vector<Real> r_src, r_trg, charges, normals;
    generate_and_scatter<Real>(n_dim, charge_dim, n_src, n_trg, cfg.dist, true, r_src, r_trg, charges, normals,
                               cfg.seed, rank, np);

    auto create_tree = [&]() -> pdmk_tree {
        const Real *r_trg_ptr = with_trg ? r_trg.data() : nullptr;
        pdmk_tree tree;
        if constexpr (std::is_same_v<Real, float>)
            tree = pdmk_tree_createf(MYCOMM, params, n_src_per_rank, r_src.data(), charges.data(), normals.data(),
                                     n_trg_per_rank, r_trg_ptr);
        else
            tree = pdmk_tree_create(MYCOMM, params, n_src_per_rank, r_src.data(), charges.data(), normals.data(),
                                    n_trg_per_rank, r_trg_ptr);
        if (!tree)
            throw std::runtime_error(pdmk_last_error_message());
        return tree;
    };

    if (cfg.bench_build) {
        if (rank == 0) {
            print_dmk_config_comment(cfg, np, n_threads, std::cout);
            print_build_csv_header(std::cout);
            std::cout << std::flush;
        }
        for (int run = 0; run < cfg.n_runs; ++run) {
            sctl::Profile::reset();
#ifdef DMK_HAVE_MPI
            MPI_Barrier(MYCOMM);
#endif
            double st = MY_OMP_GET_WTIME();
            pdmk_tree tree = create_tree();
            double ft = MY_OMP_GET_WTIME();
            pdmk_tree_destroy(tree);

            TimingResult t = make_timing(ft - st, n_src, n_src_per_rank, n_threads);

            if (run == 0) {
                if (rank == 0)
                    std::cout << ",";
                pdmk_print_profile_data(MYCOMM, 'h');
                if (rank == 0)
                    std::cout << "\n";
            }

            if (rank == 0) {
                print_build_csv_row(t, std::cout);
                std::cout << ",";
            }
            pdmk_print_profile_data(MYCOMM, 'c');
            if (rank == 0)
                std::cout << "\n" << std::flush;
        }
    }

    pdmk_tree tree = nullptr;
    if (cfg.bench_update_charges || cfg.bench_eval)
        tree = create_tree();

    if (cfg.bench_update_charges) {
        if (rank == 0) {
            if (!cfg.bench_build)
                print_dmk_config_comment(cfg, np, n_threads, std::cout);
            print_update_csv_header(std::cout);
            std::cout << std::flush;
        }
        for (int run = 0; run < cfg.n_runs; ++run) {
            sctl::Profile::reset();
            const Real *normal_ptr = (cfg.kernel == DMK_STRESSLET) ? normals.data() : nullptr;
            double dt = run_update_charges<Real>(tree, charges, normal_ptr);
            TimingResult t = make_timing(dt, n_src, n_src_per_rank, n_threads);

            if (run == 0) {
                if (rank == 0)
                    std::cout << ",";
                pdmk_print_profile_data(MYCOMM, 'h');
                if (rank == 0)
                    std::cout << "\n";
            }

            if (rank == 0) {
                print_update_csv_row(t, std::cout);
                std::cout << ",";
            }
            pdmk_print_profile_data(MYCOMM, 'c');
            if (rank == 0)
                std::cout << "\n" << std::flush;
        }
    }

    if (!cfg.bench_eval) {
        if (tree)
            pdmk_tree_destroy(tree);
        return;
    }

    // Direct reference at source positions, and at target positions if requested.
    std::vector<Real> pot_direct_src, pot_direct_trg;
    if (cfg.enable_direct && cfg.n_direct > 0) {
        const int n_direct_global = std::min(cfg.n_direct, n_src);
        const int n_direct_per_rank = local_count(n_direct_global, np, rank);

        if (n_direct_global == n_src) {
            run_direct(cfg, n_dim, charge_dim, r_src, charges, normals, r_src, pot_direct_src, rank, np);
        } else {
            std::vector<Real> r_eval(r_src.begin(), r_src.begin() + size_t(n_direct_per_rank) * n_dim);
            run_direct(cfg, n_dim, charge_dim, r_src, charges, normals, r_eval, pot_direct_src, rank, np);
        }

        if (with_trg) {
            const int n_direct_trg_global = std::min(n_direct_global, n_trg);
            const int n_direct_trg_per_rank = local_count(n_direct_trg_global, np, rank);
            std::vector<Real> r_eval(r_trg.begin(), r_trg.begin() + size_t(n_direct_trg_per_rank) * n_dim);
            run_direct(cfg, n_dim, charge_dim, r_src, charges, normals, r_eval, pot_direct_trg, rank, np);
        }
    }

#ifdef DMK_HAVE_MPI
    MPI_Barrier(MYCOMM);
#endif

    if (rank == 0) {
        if (!cfg.bench_build && !cfg.bench_update_charges)
            print_dmk_config_comment(cfg, np, n_threads, std::cout);
        print_csv_header(std::cout, cfg.with_grad, with_trg);
        std::cout << std::flush;
    }

    auto fill_block = [&](const std::vector<Real> &pot_dmk, const std::vector<Real> &pot_dir, ErrorBlock &out) {
        if (pot_dmk.empty() || pot_dir.empty())
            return;
        int n_compare = std::min(int(pot_dir.size()), int(pot_dmk.size()));
        n_compare = (n_compare / pot_dim) * pot_dim;
        if (n_compare == 0)
            return;
        std::vector<Real> dmk_sub(pot_dmk.begin(), pot_dmk.begin() + n_compare);
        std::vector<Real> dir_sub(pot_dir.begin(), pot_dir.begin() + n_compare);
        if (cfg.with_grad) {
            out.pot = compute_error(dmk_sub, dir_sub, rank, np, pot_dim, 0, 1);
            out.grad = compute_error(dmk_sub, dir_sub, rank, np, pot_dim, 1, pot_dim);
        } else {
            out.pot = compute_error(dmk_sub, dir_sub, rank, np, pot_dim, 0, pot_dim);
        }
        out.have = true;
    };

    // Allocated once and pre-sized: the page-locking has to outlive every run, and it keeps a
    // reallocation out of each timed iteration.
    std::vector<Real> pot_dmk_src(size_t(n_src_per_rank) * pot_dim);
    std::vector<Real> pot_dmk_trg(size_t(n_trg_per_rank) * pot_dim);
    const bool pinned_src = pin_host_buffer(pot_dmk_src, cfg.pin_host);
    const bool pinned_trg = pin_host_buffer(pot_dmk_trg, cfg.pin_host);

    for (int run = 0; run < cfg.n_runs; ++run) {
        sctl::Profile::reset();
        double dt = run_dmk<Real>(tree, pot_dmk_src, pot_dmk_trg, n_src_per_rank, n_trg_per_rank, pot_dim, rank, np);
        TimingResult t = make_timing(dt, n_src + n_trg, n_src_per_rank + n_trg_per_rank, n_threads);

        if (run == 0) {
            if (rank == 0)
                std::cout << ",";
            pdmk_print_profile_data(MYCOMM, 'h');
            if (rank == 0)
                std::cout << "\n";
        }

        ErrorBlock src_err, trg_err;
        if (cfg.enable_direct) {
            fill_block(pot_dmk_src, pot_direct_src, src_err);
            if (with_trg)
                fill_block(pot_dmk_trg, pot_direct_trg, trg_err);
        }

        if (run == 0 && cfg.enable_direct && cfg.n_show_outliers > 0) {
            const int n_show = cfg.n_show_outliers;
            const std::vector<Real> empty_charges;
            print_outliers(pot_dmk_src, pot_direct_src, r_src, r_src, /*same_set=*/true, charges, charge_dim, n_dim,
                           pot_dim, 0, cfg.with_grad ? 1 : pot_dim, n_show, "src pot", rank, std::cerr);
            if (cfg.with_grad)
                print_outliers(pot_dmk_src, pot_direct_src, r_src, r_src, /*same_set=*/true, charges, charge_dim, n_dim,
                               pot_dim, 1, pot_dim, n_show, "src grad", rank, std::cerr);
            if (with_trg) {
                print_outliers(pot_dmk_trg, pot_direct_trg, r_trg, r_src, /*same_set=*/false, empty_charges, 0, n_dim,
                               pot_dim, 0, cfg.with_grad ? 1 : pot_dim, n_show, "trg pot", rank, std::cerr);
                if (cfg.with_grad)
                    print_outliers(pot_dmk_trg, pot_direct_trg, r_trg, r_src, /*same_set=*/false, empty_charges, 0,
                                   n_dim, pot_dim, 1, pot_dim, n_show, "trg grad", rank, std::cerr);
            }
        }

        if (rank == 0)
            print_csv_row(t, src_err, with_trg ? &trg_err : nullptr, cfg.with_grad, std::cout);
        if (rank == 0)
            std::cout << ",";
        pdmk_print_profile_data(MYCOMM, 'c');
        if (rank == 0)
            std::cout << std::endl << std::flush;
        // Drawing to terminal takes time away from the GPU *sigh*.
        if (cfg.eval_path == DMK_EVAL_PATH_GPU)
            std::this_thread::sleep_for(std::chrono::milliseconds(26));
    }

    unpin_host_buffer(pot_dmk_src, pinned_src);
    unpin_host_buffer(pot_dmk_trg, pinned_trg);

    pdmk_tree_destroy(tree);
}

// ---------------------------------------------------------------------------
// ESP
// ---------------------------------------------------------------------------

inline dmk_eval_type esp_pot_eval_type(dmk_ikernel k) { return is_velocity_kernel(k) ? DMK_VELOCITY : DMK_POTENTIAL; }

pdmk_esp_params make_params(const Config &cfg, double r_c, dmk_eval_type eval_type, dmk_eval_path eval_path) {
    pdmk_esp_params params{};
    params.L = cfg.L;
    params.r_c = r_c;
    params.eps = cfg.eps;
    params.n_dim = cfg.n_dim;
    params.kernel = cfg.kernel;
    if (cfg.kernel == DMK_YUKAWA)
        params.fparam = cfg.fparam;
    params.log_level = cfg.log_level;
    params.eval_type = eval_type;
    params.sigma = cfg.sigma;
    params.use_periodic = cfg.use_periodic;
    params.freespace_pad = cfg.freespace_pad;
    params.esp_flags = cfg.esp_flags;
    params.esp_bins = cfg.esp_bins;
    params.esp_stile = cfg.esp_stile;
    params.eval_path = eval_path;
    params.gpu_device_id = cfg.gpu_device_id;
    return params;
}

std::string sr_summary(uint32_t esp_flags, int esp_bins, int esp_stile) {
    const bool n3l = esp_flags & DMK_ESP_N3L;
    const bool prune_source = esp_flags & DMK_ESP_PRUNE_SOURCE;
    const bool prune_tile = esp_flags & DMK_ESP_PRUNE_TILE;
    const bool morton = esp_flags & DMK_ESP_MORTON;
    std::string method = n3l ? "n3l" : prune_source ? "prune_source" : prune_tile ? "prune_tile" : "dense";
    std::string s = method + (morton ? " morton" : " bins=" + std::to_string(esp_bins));
    if (esp_stile > 0)
        s += " stile=" + std::to_string(esp_stile);
    return s;
}

template <typename Real>
pdmk_esp_plan esp_plan_create(pdmk_esp_params params) {
    pdmk_esp_plan plan;
    if constexpr (std::is_same_v<Real, float>)
        plan = pdmk_esp_plan_createf(nullptr, params);
    else
        plan = pdmk_esp_plan_create(nullptr, params);
    if (!plan) {
        std::cerr << "pdmk_esp_plan_create failed: " << pdmk_last_error_message() << "\n";
        std::exit(1);
    }
    return plan;
}

template <typename Real>
void esp_eval(pdmk_esp_plan plan, int n, const Real *r_src, const Real *charges, const Real *normal, Real *pot_src) {
    dmk_error rc;
    if constexpr (std::is_same_v<Real, float>)
        rc = pdmk_esp_evalf(nullptr, plan, n, r_src, charges, normal, pot_src);
    else
        rc = pdmk_esp_eval(nullptr, plan, n, r_src, charges, normal, pot_src);
    if (rc != DMK_SUCCESS) {
        std::cerr << "pdmk_esp_eval failed with rc=" << rc << ": " << pdmk_last_error_message() << "\n";
        std::exit(1);
    }
}

template <typename Real>
void esp_plan_destroy(pdmk_esp_plan plan) {
    if constexpr (std::is_same_v<Real, float>)
        pdmk_esp_plan_destroyf(plan);
    else
        pdmk_esp_plan_destroy(plan);
}

// Positions are uniform on [-L/2, L/2)^n_dim.
template <typename Real>
std::vector<Real> generate_positions(int n, int n_dim, double L, long seed = 42) {
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> rng(-0.5 * L, 0.5 * L);
    std::vector<Real> r(size_t(n) * n_dim);
    for (size_t i = 0; i < r.size(); ++i)
        r[i] = Real(rng(eng));
    return r;
}

// Scalar charges alternate ±1 so the system is neutral, which periodic Ewald requires.
template <typename Real>
std::vector<Real> generate_charges(int n, int input_dim = 1, long seed = 7) {
    if (input_dim == 1) {
        std::vector<Real> q(n);
        for (int i = 0; i < n; ++i)
            q[i] = Real(1 - 2 * (i & 1));
        return q;
    }
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> rng(-0.5, 0.5);
    std::vector<Real> q(size_t(n) * input_dim);
    for (auto &v : q)
        v = Real(rng(eng));
    return q;
}

template <typename Real>
std::vector<Real> generate_normals(int n, int n_dim, long seed = 11) {
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> rng(-1.0, 1.0);
    std::vector<Real> nv(size_t(n) * n_dim);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int d = 0; d < n_dim; ++d) {
            const double x = rng(eng);
            nv[i * n_dim + d] = Real(x);
            s += x * x;
        }
        const double inv = s > 0 ? 1.0 / std::sqrt(s) : 1.0;
        for (int d = 0; d < n_dim; ++d)
            nv[i * n_dim + d] = Real(nv[i * n_dim + d] * inv);
    }
    return nv;
}

// Reference potential at the first n_direct sources, self-interaction excluded. False if this
// configuration has no self-contained reference.
bool compute_reference(const Config &cfg, int n, const std::vector<double> &r_src_d,
                       const std::vector<double> &charges_d, const std::vector<double> &normals_d,
                       std::vector<double> &ref) {
    const int nd = cfg.n_dim;
    const int n_cmp = std::min(cfg.n_direct, n);
    const double L = cfg.L;

    if (!cfg.use_periodic) {
        const dmk_eval_type et = esp_pot_eval_type(cfg.kernel);
        const int out_dim = dmk::get_kernel_output_dim(nd, cfg.kernel, et);
        ref.assign(size_t(n_cmp) * out_dim, 0.0);
        std::cout << "# verify: free-space direct reference for first " << n_cmp << " of " << n << " points...\n"
                  << std::flush;
        auto fn = dmk::get_direct_evaluator<double>(cfg.kernel, et, nd, cfg.fparam);
        fn(n, r_src_d.data(), charges_d.data(), needs_normal(cfg.kernel) ? normals_d.data() : nullptr, n_cmp,
           r_src_d.data(), ref.data());
        return true;
    }

    if (cfg.kernel == DMK_LAPLACE && nd == 2) {
        std::cout << "# verify: skipping accuracy check for 2D Laplace (log gauge/self is validated in "
                     "test_esp.cpp, not self-contained here)\n"
                  << std::flush;
        return false;
    }

    ref.assign(n_cmp, 0.0);
    std::cout << "# verify: computing reference for first " << n_cmp << " of " << n << " points...\n" << std::flush;

    if (cfg.kernel == DMK_YUKAWA) {
        const double lambda = cfg.fparam;
        const int n_img = std::max(2, int(std::ceil(21.0 / (lambda * L)))); // exp(-lambda*n_img*L) ~ 1e-9
        const int mz_lo = nd == 3 ? -n_img : 0, mz_hi = nd == 3 ? n_img : 0;
#pragma omp parallel for
        for (int i = 0; i < n_cmp; ++i) {
            double pot = 0.0;
            for (int j = 0; j < n; ++j)
                for (int mx = -n_img; mx <= n_img; ++mx)
                    for (int my = -n_img; my <= n_img; ++my)
                        for (int mz = mz_lo; mz <= mz_hi; ++mz) {
                            const double d0 = r_src_d[i * nd + 0] - r_src_d[j * nd + 0] - mx * L;
                            const double d1 = r_src_d[i * nd + 1] - r_src_d[j * nd + 1] - my * L;
                            double r2 = d0 * d0 + d1 * d1;
                            if (nd == 3) {
                                const double d2 = r_src_d[i * nd + 2] - r_src_d[j * nd + 2] - mz * L;
                                r2 += d2 * d2;
                            }
                            if (r2 > 1e-28) {
                                const double r = std::sqrt(r2);
                                pot += charges_d[j] * std::exp(-lambda * r) / r;
                            }
                        }
            ref[i] = pot;
        }
    } else {
        dmk::pbc_ref::EwaldRef ewald(cfg.kernel, nd, n, r_src_d.data(), charges_d.data(), L, 15.0 / L);
#pragma omp parallel for
        for (int i = 0; i < n_cmp; ++i) {
            double pot;
            ewald.eval(&r_src_d[i * nd], i, pot, nullptr);
            ref[i] = pot;
        }
    }
    return true;
}

void print_esp_config(const Config &cfg, int n_threads, std::ostream &os) {
    os << "# n_src:       " << cfg.n_src << "\n"
       << "# n_dim:       " << cfg.n_dim << "\n"
       << "# L:           " << cfg.L << "\n"
       << "# r_c:         " << cfg.r_c << "\n"
       << "# eps:         " << cfg.eps << "\n"
       << "# kernel:      " << dmk::util::to_string(cfg.kernel) << "\n"
       << "# fparam:      " << cfg.fparam << "\n"
       << "# sigma:       " << cfg.sigma << "\n"
       << "# boundary:    " << (cfg.use_periodic ? "periodic" : "free-space") << "\n"
       << "# n_runs:      " << cfg.n_runs << "\n"
       << "# n_direct:    " << cfg.n_direct << "\n"
       << "# prec:        " << (cfg.prec == 'd' ? "double" : "float") << "\n"
       << "# eval_path:   " << (cfg.eval_path == DMK_EVAL_PATH_GPU ? "gpu" : "cpu") << "\n"
       << "# bench_plan:  " << (cfg.bench_plan ? "true" : "false") << "\n"
       << "# bench_forces:" << (cfg.bench_forces ? "true" : "false") << "\n"
       << "# check_forces:" << (cfg.check_forces ? "true" : "false") << "\n"
       << "# log_level:   " << cfg.log_level << "\n"
       << "# short_range: " << sr_summary(cfg.esp_flags, cfg.esp_bins, cfg.esp_stile) << "\n"
       << "# omp_threads: " << n_threads << "\n";
    if (cfg.eval_path == DMK_EVAL_PATH_GPU)
        os << "# gpu_device:  " << cfg.gpu_device_id << "\n"
           << "# cpu_baseline:" << (cfg.skip_cpu_baseline ? "false" : "true") << "\n";
}

// Picks the r_c with the smallest l2_rel_err against the reference. Always sweeps on the CPU: this
// is an accuracy choice, not a speed one.
void init_sensible_defaults(Config &cfg, const std::vector<double> &r_src_d, const std::vector<double> &charges_d,
                            const std::vector<double> &normals_d, const std::vector<double> &ref, bool have_ref) {
    if (cfg.r_c != -1.0)
        return;

    const std::vector<double> rc_candidates = {0.02 * cfg.L, 0.03 * cfg.L, 0.04 * cfg.L, 0.05 * cfg.L,
                                               0.06 * cfg.L, 0.07 * cfg.L, 0.10 * cfg.L, 0.12 * cfg.L};

    if (!have_ref) {
        cfg.r_c = rc_candidates[rc_candidates.size() / 2];
        std::cout << "# init_sensible_defaults: no reference, defaulting to r_c=" << cfg.r_c << "\n" << std::flush;
        return;
    }

    const dmk_eval_type et = esp_pot_eval_type(cfg.kernel);
    const int out_dim = dmk::get_kernel_output_dim(cfg.n_dim, cfg.kernel, et);
    const int input_dim = dmk::get_kernel_input_dim(cfg.n_dim, cfg.kernel);
    const int n = static_cast<int>(charges_d.size()) / input_dim;
    const double *normal = needs_normal(cfg.kernel) ? normals_d.data() : nullptr;
    double best_l2 = std::numeric_limits<double>::max();
    for (double rc : rc_candidates) {
        pdmk_esp_params params = make_params(cfg, rc, et, DMK_EVAL_PATH_CPU);
        pdmk_esp_plan plan = esp_plan_create<double>(params);
        std::vector<double> pot(size_t(n) * out_dim);
        esp_eval<double>(plan, n, r_src_d.data(), charges_d.data(), normal, pot.data());
        const double l2 = l2_rel_err(std::span<const double>(pot), ref);
        esp_plan_destroy<double>(plan);

        if (l2 < best_l2) {
            best_l2 = l2;
            cfg.r_c = rc;
        }
    }

    std::cout << "# init_sensible_defaults: selected r_c=" << cfg.r_c << " (l2_rel_err=" << best_l2 << ")\n"
              << std::flush;
}

// The FD step depends on the eval path: cuFINUFFT's plan tolerance puts a noise floor under the GPU
// potential, so there the step needs step^2 << eps/step, i.e. step ~ eps^(1/3), far larger than the
// CPU's 1e-12.
double check_forces_fd(const std::vector<double> &pot_src_grad, const std::vector<double> &r_src_d,
                       const std::vector<double> &charges_d, pdmk_esp_params params, double eps, int n_sample = 20) {
    const int n = static_cast<int>(charges_d.size());
    const int nd = params.n_dim;
    const int out_dim = 1 + nd;
    n_sample = std::min(n_sample, n);
    const double step = (params.eval_path == DMK_EVAL_PATH_GPU) ? std::cbrt(eps) : 1e-12;

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::default_random_engine(123));
    idx.resize(n_sample);

    params.eval_type = DMK_POTENTIAL;
    pdmk_esp_plan plan = esp_plan_create<double>(params);

    std::vector<double> r_pert = r_src_d;
    std::vector<double> pot(n);
    double err2 = 0, ref2 = 0;
    for (int i : idx) {
        for (int a = 0; a < nd; ++a) {
            r_pert[nd * i + a] = r_src_d[nd * i + a] + step;
            esp_eval<double>(plan, n, r_pert.data(), charges_d.data(), nullptr, pot.data());
            double pot_plus = pot[i];

            r_pert[nd * i + a] = r_src_d[nd * i + a] - step;
            esp_eval<double>(plan, n, r_pert.data(), charges_d.data(), nullptr, pot.data());
            double pot_minus = pot[i];

            r_pert[nd * i + a] = r_src_d[nd * i + a];

            const double f_ref = -charges_d[i] * (pot_plus - pot_minus) / (2.0 * step);
            const double diff = pot_src_grad[i * out_dim + 1 + a] - f_ref;
            err2 += diff * diff;
            ref2 += f_ref * f_ref;
        }
    }

    esp_plan_destroy<double>(plan);
    return std::sqrt(err2 / ref2);
}

template <typename Real>
void warmup(const Config &cfg) {
    constexpr int nw = 100'000;
    const int input_dim = dmk::get_kernel_input_dim(cfg.n_dim, cfg.kernel);
    const dmk_eval_type et = esp_pot_eval_type(cfg.kernel);
    const int out_dim = dmk::get_kernel_output_dim(cfg.n_dim, cfg.kernel, et);
    auto r_w = generate_positions<Real>(nw, cfg.n_dim, cfg.L);
    auto q_w = generate_charges<Real>(nw, input_dim);
    std::vector<Real> nrm_w;
    if (needs_normal(cfg.kernel))
        nrm_w = generate_normals<Real>(nw, cfg.n_dim);
    pdmk_esp_params params = make_params(cfg, cfg.r_c, et, cfg.eval_path);
    pdmk_esp_plan plan = esp_plan_create<Real>(params);
    std::vector<Real> pot(size_t(nw) * out_dim);
    esp_eval<Real>(plan, nw, r_w.data(), q_w.data(), nrm_w.empty() ? nullptr : nrm_w.data(), pot.data());
    esp_plan_destroy<Real>(plan);
}

template <typename Real>
void run_esp_phase(const Config &cfg, int n, const std::vector<Real> &r_src, const std::vector<Real> &charges,
                   const std::vector<Real> &normals, const std::vector<double> &ref, bool have_ref,
                   dmk_eval_type eval_type, const char *phase_name) {
    const int out_dim = dmk::get_kernel_output_dim(cfg.n_dim, cfg.kernel, eval_type);
    const Real *normal = normals.empty() ? nullptr : normals.data();
    const bool on_gpu = cfg.eval_path == DMK_EVAL_PATH_GPU;
    const bool want_cpu_baseline = on_gpu && !cfg.skip_cpu_baseline;

    pdmk_esp_plan plan = esp_plan_create<Real>(make_params(cfg, cfg.r_c, eval_type, cfg.eval_path));

    std::vector<Real> pot(size_t(n) * out_dim);

    double cpu_time = std::numeric_limits<double>::quiet_NaN();
    std::vector<Real> cpu_pot;
    if (want_cpu_baseline) {
        pdmk_esp_plan cpu_plan = esp_plan_create<Real>(make_params(cfg, cfg.r_c, eval_type, DMK_EVAL_PATH_CPU));
        cpu_pot.resize(size_t(n) * out_dim);
        const double t0 = MY_OMP_GET_WTIME();
        esp_eval<Real>(cpu_plan, n, r_src.data(), charges.data(), normal, cpu_pot.data());
        const double t1 = MY_OMP_GET_WTIME();
        cpu_time = t1 - t0;
        esp_plan_destroy<Real>(cpu_plan);
        std::cout << "# cpu_baseline_time (" << phase_name << "): " << cpu_time << " s (" << n / cpu_time << " pts/s)\n"
                  << std::flush;
    }

    esp_eval<Real>(plan, n, r_src.data(), charges.data(), normal, pot.data());

    const bool report_ref = eval_type == esp_pot_eval_type(cfg.kernel);
    const bool report_cpu = !cpu_pot.empty() && eval_type == DMK_POTENTIAL;

    std::cout << "# phase: " << phase_name << "\n";
    std::cout << "run,total_time,pts_per_s";
    if (report_ref)
        std::cout << ",l2_rel_err";
    if (report_cpu)
        std::cout << ",l2_rel_err_vs_cpu";
    std::cout << ",";
    pdmk_print_profile_data(nullptr, 'h');
    std::cout << "\n" << std::flush;

    // Brackets only the timed loop, so `nsys/ncu -c cudaProfilerApi` excludes setup and warmup.
#ifdef DMK_GPU_OFFLOAD
    if (on_gpu)
        cudaProfilerStart();
#endif
    double gpu_time_sum = 0;
    for (int run = 0; run < cfg.n_runs; ++run) {
        sctl::Profile::reset();
        const double t0 = MY_OMP_GET_WTIME();
        esp_eval<Real>(plan, n, r_src.data(), charges.data(), normal, pot.data());
        const double t1 = MY_OMP_GET_WTIME();
        gpu_time_sum += t1 - t0;

        std::cout << run << "," << (t1 - t0) << "," << n / (t1 - t0);
        if (report_ref)
            std::cout << ","
                      << (have_ref ? l2_rel_err(std::span<const Real>(pot), ref)
                                   : std::numeric_limits<double>::quiet_NaN());
        if (report_cpu)
            std::cout << "," << l2_rel_err_gauged(pot, cpu_pot);
        std::cout << ",";
        pdmk_print_profile_data(nullptr, 'c');
        std::cout << "\n" << std::flush;
    }
#ifdef DMK_GPU_OFFLOAD
    if (on_gpu)
        cudaProfilerStop();
#endif

    if (want_cpu_baseline) {
        const double gpu_time_avg = gpu_time_sum / cfg.n_runs;
        std::cout << "# speedup (" << phase_name << ", cpu_baseline / gpu_avg): " << (cpu_time / gpu_time_avg) << "\n"
                  << std::flush;
    }

    esp_plan_destroy<Real>(plan);
}

template <typename Real>
void run_esp_plan_bench(const Config &cfg, int n, dmk_eval_type eval_type) {
    std::cout << "# phase: plan_create\n" << "run,plan_time,pts_per_s\n" << std::flush;
    for (int run = 0; run < cfg.n_runs; ++run) {
        const double t0 = MY_OMP_GET_WTIME();
        pdmk_esp_plan plan = esp_plan_create<Real>(make_params(cfg, cfg.r_c, eval_type, cfg.eval_path));
        const double t1 = MY_OMP_GET_WTIME();
        esp_plan_destroy<Real>(plan);
        std::cout << run << "," << (t1 - t0) << "," << n / (t1 - t0) << "\n" << std::flush;
    }
}

void run_force_check(const Config &cfg, int n, const std::vector<double> &r_src_d,
                     const std::vector<double> &charges_d) {
    constexpr int n_fd_sample = 20;
    pdmk_esp_params params = make_params(cfg, cfg.r_c, DMK_POTENTIAL_GRAD, cfg.eval_path);
    pdmk_esp_plan plan = esp_plan_create<double>(params);

    std::cout << "# phase: force_check (FD on " << n_fd_sample << " random particles x 6 evals each; not all N)\n"
              << std::flush;
    std::vector<double> pot_d(size_t(n) * (1 + cfg.n_dim));
    esp_eval<double>(plan, n, r_src_d.data(), charges_d.data(), nullptr, pot_d.data());
    const double err = check_forces_fd(pot_d, r_src_d, charges_d, params, cfg.eps, n_fd_sample);
    std::cout << "# force_check: l2_rel_err=" << err << "\n" << std::flush;

    esp_plan_destroy<double>(plan);
}

template <typename Real>
void run_esp_benchmark(Config cfg) {
    sctl::Profile::Enable(true);
    const int n_threads = MY_OMP_GET_MAX_THREADS();
    const int n = cfg.n_src;

    if (!is_scalar_kernel(cfg.kernel)) {
        if (cfg.use_periodic) {
            std::cout << "# note: " << dmk::util::to_string(cfg.kernel)
                      << " is free-space only in ESP; forcing free-space boundaries\n";
            cfg.use_periodic = false;
        }
        if (cfg.bench_forces || cfg.check_forces) {
            std::cout << "# note: -g/-F (forces) apply only to the scalar kernels; ignoring\n";
            cfg.bench_forces = cfg.check_forces = false;
        }
    }

    const int input_dim = dmk::get_kernel_input_dim(cfg.n_dim, cfg.kernel);
    const bool with_normal = needs_normal(cfg.kernel);

    auto r_src_d = generate_positions<double>(n, cfg.n_dim, cfg.L);
    auto charges_d = generate_charges<double>(n, input_dim);
    auto normals_d = with_normal ? generate_normals<double>(n, cfg.n_dim) : std::vector<double>{};

    std::vector<double> ref;
    bool have_ref = false;
    if (cfg.n_direct > 0)
        have_ref = compute_reference(cfg, n, r_src_d, charges_d, normals_d, ref);

    init_sensible_defaults(cfg, r_src_d, charges_d, normals_d, ref, have_ref);
    print_esp_config(cfg, n_threads, std::cout);

    std::vector<Real> r_src(size_t(n) * cfg.n_dim), charges(size_t(n) * input_dim), normals;
    for (size_t i = 0; i < r_src.size(); ++i)
        r_src[i] = Real(r_src_d[i]);
    for (size_t i = 0; i < charges.size(); ++i)
        charges[i] = Real(charges_d[i]);
    if (with_normal) {
        normals.resize(normals_d.size());
        for (size_t i = 0; i < normals.size(); ++i)
            normals[i] = Real(normals_d[i]);
    }

    const dmk_eval_type pot_et = esp_pot_eval_type(cfg.kernel);
    const char *pot_phase = is_velocity_kernel(cfg.kernel) ? "eval_velocity" : "eval_potential";

    if (cfg.bench_plan)
        run_esp_plan_bench<Real>(cfg, n, cfg.bench_forces ? DMK_POTENTIAL_GRAD : pot_et);

    warmup<Real>(cfg);

    if (cfg.bench_forces) {
        run_esp_phase<Real>(cfg, n, r_src, charges, normals, ref, have_ref, DMK_POTENTIAL_GRAD, "eval_forces");
    } else {
        run_esp_phase<Real>(cfg, n, r_src, charges, normals, ref, have_ref, pot_et, pot_phase);
    }

    if (cfg.check_forces)
        run_force_check(cfg, n, r_src_d, charges_d);
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

enum {
    OPT_SOLVER = 256,
    OPT_DIRECT,
    OPT_NO_DIRECT,
    OPT_BENCH_BUILD,
    OPT_NO_BENCH_EVAL,
    OPT_BENCH_UPDATE_CHARGES,
    OPT_PERIODIC,
    OPT_PIN,
    OPT_BENCH_PLAN,
    OPT_SIGMA,
    OPT_FREESPACE_PAD,
    OPT_SKIP_CPU_BASELINE,
    OPT_GPU_DEVICE,
    OPT_PRUNE,
    OPT_N3L,
    OPT_MORTON,
    OPT_BINS,
    OPT_STILE,
};

static const struct option long_opts[] = {
    {"solver", required_argument, nullptr, OPT_SOLVER},
    {"direct", no_argument, nullptr, OPT_DIRECT},
    {"no-direct", no_argument, nullptr, OPT_NO_DIRECT},
    {"bench-build", no_argument, nullptr, OPT_BENCH_BUILD},
    {"no-bench-eval", no_argument, nullptr, OPT_NO_BENCH_EVAL},
    {"bench-update-charges", no_argument, nullptr, OPT_BENCH_UPDATE_CHARGES},
    {"periodic", no_argument, nullptr, OPT_PERIODIC},
    {"pin", no_argument, nullptr, OPT_PIN},
    {"bench-plan", no_argument, nullptr, OPT_BENCH_PLAN},
    {"sigma", required_argument, nullptr, OPT_SIGMA},
    {"freespace-pad", required_argument, nullptr, OPT_FREESPACE_PAD},
    {"skip-cpu-baseline", no_argument, nullptr, OPT_SKIP_CPU_BASELINE},
    {"gpu-device", required_argument, nullptr, OPT_GPU_DEVICE},
    {"prune", required_argument, nullptr, OPT_PRUNE},
    {"n3l", required_argument, nullptr, OPT_N3L},
    {"morton", required_argument, nullptr, OPT_MORTON},
    {"bins", required_argument, nullptr, OPT_BINS},
    {"stile", required_argument, nullptr, OPT_STILE},
    {nullptr, 0, nullptr, 0},
};

static const char *short_opts = "N:T:n:e:t:r:D:l:s:k:d:f:O:p:u:L:c:gFh?";

void print_usage(const char *argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "\n"
              << "Test driver for DMK and ESP on CPU and GPU. --solver picks the backend and -p the eval path;\n"
              << "options are grouped below by which solver reads them.\n"
              << "\n"
              << "  --solver dmk|esp      Backend (default dmk)\n"
              << "  -p c|g                Eval path: (c)pu or (g)pu (default c)\n"
              << "\n"
              << "Shared:\n"
              << "  -N n_src              Number of source points (default 1e5)\n"
              << "  -d dim                2 or 3\n"
              << "  -e eps                Tolerance (default 1e-5)\n"
              << "  -t f|d                Precision (default float)\n"
              << "  -k kernel             laplace, sqrt_laplace, yukawa, stokeslet, stresslet, laplace_dipole\n"
              << "  -f fparam             Yukawa screening parameter (default 6.0)\n"
              << "  -r n_runs             Benchmark iterations (default 100)\n"
              << "  -D n_direct           Points compared against the reference: -1 all of them,\n"
              << "                        0 skips the reference entirely (default 10000)\n"
              << "  -l log_level          DMK log verbosity 0-6\n"
              << "  -g                    Potential + gradient. ESP names the phase eval_forces because\n"
              << "                        its scalar kernels report the axes as the force -q*grad.\n"
              << "  --periodic            Periodic boundaries (default free-space)\n"
              << "  --gpu-device n        CUDA device id when -p g (default 0)\n"
              << "  -h                    Help\n"
              << "\n"
              << "--solver dmk only:\n"
              << "  -T n_trg              Separate target points (default 0 = source self-eval only)\n"
              << "  -n n_per_leaf         DMK leaf size\n"
              << "  -s seed               Integer seed for random numbers\n"
              << "  -u dist               0=uniform (default), 1=sphere_surface, 2=box_partial_facet\n"
              << "  -O n_outliers         Print top-N worst points per block to stderr (default 0 = off)\n"
              << "  --direct/--no-direct  Enable/disable direct reference\n"
              << "  --bench-build         Also benchmark tree build time\n"
              << "  --no-bench-eval       Skip eval benchmark (build only)\n"
              << "  --bench-update-charges  Also benchmark pdmk_tree_update_charges\n"
              << "  --pin                 Page-lock the potential buffers (GPU path: unstaged D2H)\n"
              << "\n"
              << "--solver esp only:\n"
              << "  -L L                  Box side length (default 1.0)\n"
              << "  -c r_c                Real-space cutoff (default: auto-picked for accuracy)\n"
              << "  -F                    Validate forces against a finite-difference reference.\n"
              << "                        Samples 20 random particles, not all N.\n"
              << "  --sigma s             FINUFFT upsampling factor for the long-range PSWF kernel (1.35).\n"
              << "                        Requires JIT support (-DDMK_USE_JIT=ON at configure time).\n"
              << "  --bench-plan          Also report plan creation time\n"
              << "  --freespace-pad p     Free-space FFT-grid padding per axis (default: auto 2*sqrt(n_dim))\n"
              << "  --skip-cpu-baseline   With -p g, skip the CPU reference run (no speedup or\n"
              << "                        l2_rel_err_vs_cpu reporting; faster to iterate)\n"
              << "\n"
              << "  Short-range method (default: --n3l 1 --morton 1 --prune 2):\n"
              << "  --prune N             0=dense, 1=tile-vs-tile, 2=per-source. Also selects the GPU\n"
              << "                        strategy, where 2 takes precedence over 1 and 0 is dense.\n"
              << "  --n3l N               1=Newton's-third-law reciprocal half-stencil (takes precedence\n"
              << "                        over --prune), 0=off. CPU only.\n"
              << "  --morton N            1=Morton within-cell sort, 0=octant-bin counting sort\n"
              << "  --bins N              Sub-boxes per axis for the bin sort when --morton 0. CPU only.\n"
              << "  --stile N             Source-tile width for --prune 1. CPU only.\n"
              << "\n"
              << "Output CSV columns:\n"
              << "  dmk: one table, dmk_time,dmk_pts_s,... plus per-phase profiler columns\n"
              << "  esp: one '# phase:' block per phase, run,total_time,pts_per_s[,l2_rel_err]\n"
              << "       [,l2_rel_err_vs_cpu]\n"
              << "  Per-phase profiler breakdown requires -DDMK_INSTRUMENT=ON.\n"
              << "\n"
              << "Profiling the GPU path with Nsight: the timed loop is bracketed with\n"
              << "cudaProfilerStart/Stop, so -c cudaProfilerApi captures only that region:\n"
              << "  nsys profile -c cudaProfilerApi -o report " << argv0
              << " --solver esp -p g --skip-cpu-baseline -r 5\n";
}

Config parse_args(int argc, char *argv[]) {
    Config cfg;

    auto set_flag = [&](unsigned bit, bool on) {
        if (on)
            cfg.esp_flags |= bit;
        else
            cfg.esp_flags &= ~bit;
    };

    int opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
        switch (opt) {
        case OPT_SOLVER:
            if (std::strcmp(optarg, "dmk") == 0)
                cfg.solver = Solver::DMK;
            else if (std::strcmp(optarg, "esp") == 0)
                cfg.solver = Solver::ESP;
            else {
                std::cerr << "Unknown solver: " << optarg << " (expected dmk or esp)\n";
                std::exit(1);
            }
            break;
        case 'N':
            cfg.n_src = int(std::atof(optarg));
            break;
        case 'T':
            cfg.n_trg = int(std::atof(optarg));
            break;
        case 'O':
            cfg.n_show_outliers = std::atoi(optarg);
            break;
        case 'n':
            cfg.n_per_leaf = std::atoi(optarg);
            break;
        case 'e':
            cfg.eps = std::atof(optarg);
            break;
        case 'r':
            cfg.n_runs = std::atoi(optarg);
            break;
        case 'D':
            cfg.n_direct = int(std::atof(optarg));
            break;
        case 'l':
            cfg.log_level = std::atoi(optarg);
            break;
        case 's':
            cfg.seed = std::atol(optarg);
            break;
        case 'k':
            cfg.kernel = parse_kernel(optarg);
            break;
        case 'd':
            cfg.n_dim = std::atoi(optarg);
            if (cfg.n_dim != 2 && cfg.n_dim != 3) {
                std::cerr << "Invalid dimension: " << optarg << " (must be 2 or 3)\n";
                std::exit(1);
            }
            break;
        case 'f':
            cfg.fparam = std::atof(optarg);
            break;
        case 'L':
            cfg.L = std::atof(optarg);
            break;
        case 'c':
            cfg.r_c = std::atof(optarg);
            break;
        case 't':
            if (optarg[0] == 'd')
                cfg.prec = 'd';
            else if (optarg[0] == 'f')
                cfg.prec = 'f';
            else {
                std::cerr << "Unknown precision: " << optarg << "\n";
                std::exit(1);
            }
            break;
        case 'u':
            cfg.dist = dmk_distribution(std::atoi(optarg));
            break;
        case 'p':
            if (optarg[0] == 'c')
                cfg.eval_path = DMK_EVAL_PATH_CPU;
            else if (optarg[0] == 'g')
                cfg.eval_path = DMK_EVAL_PATH_GPU;
            else {
                std::cerr << "Unknown eval_path: " << optarg << " (expected c or g)\n";
                std::exit(1);
            }
            break;
        case 'g':
            cfg.with_grad = true;
            cfg.bench_forces = true;
            break;
        case 'F':
            cfg.check_forces = true;
            break;
        case OPT_DIRECT:
            cfg.enable_direct = true;
            break;
        case OPT_NO_DIRECT:
            cfg.enable_direct = false;
            break;
        case OPT_BENCH_BUILD:
            cfg.bench_build = true;
            break;
        case OPT_NO_BENCH_EVAL:
            cfg.bench_eval = false;
            break;
        case OPT_BENCH_UPDATE_CHARGES:
            cfg.bench_update_charges = true;
            break;
        case OPT_PERIODIC:
            cfg.use_periodic = true;
            break;
        case OPT_PIN:
            cfg.pin_host = true;
            break;
        case OPT_BENCH_PLAN:
            cfg.bench_plan = true;
            break;
        case OPT_SIGMA:
            cfg.sigma = std::atof(optarg);
            cfg.sigma_set = true;
            break;
        case OPT_FREESPACE_PAD:
            cfg.freespace_pad = std::atof(optarg);
            break;
        case OPT_SKIP_CPU_BASELINE:
            cfg.skip_cpu_baseline = true;
            break;
        case OPT_GPU_DEVICE:
            cfg.gpu_device_id = std::atoi(optarg);
            break;
        case OPT_PRUNE: {
            const int v = std::atoi(optarg);
            set_flag(DMK_ESP_PRUNE_TILE, v == 1);
            set_flag(DMK_ESP_PRUNE_SOURCE, v >= 2);
            break;
        }
        case OPT_N3L:
            set_flag(DMK_ESP_N3L, std::atoi(optarg) != 0);
            break;
        case OPT_MORTON:
            set_flag(DMK_ESP_MORTON, std::atoi(optarg) != 0);
            break;
        case OPT_BINS:
            cfg.esp_bins = std::atoi(optarg);
            break;
        case OPT_STILE:
            cfg.esp_stile = std::atoi(optarg);
            break;
        case 'h':
        case '?':
        default:
            print_usage(argv[0]);
            std::exit(0);
        }
    }

    if (cfg.n_direct < 0)
        cfg.n_direct = cfg.n_src;
    return cfg;
}

int main(int argc, char *argv[]) {
#ifdef DMK_HAVE_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#endif

    int rc = 0;
    try {
        Config cfg = parse_args(argc, argv);

        if (cfg.solver == Solver::ESP) {
#ifdef DMK_HAVE_MPI
            int np = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &np);
            if (np > 1)
                throw std::runtime_error("ESP does not support multiple MPI ranks");
#endif
#ifndef DMK_USE_JIT
            if (cfg.sigma_set)
                throw std::runtime_error("--sigma requires JIT support (recompile with -DDMK_USE_JIT=ON)");
#endif
            if (cfg.prec == 'd')
                run_esp_benchmark<double>(cfg);
            else
                run_esp_benchmark<float>(cfg);
        } else {
            if (cfg.prec == 'd')
                run_dmk_benchmark<double>(cfg);
            else
                run_dmk_benchmark<float>(cfg);
        }
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        rc = 1;
    }

#ifdef DMK_HAVE_MPI
    MPI_Finalize();
#endif
    return rc;
}
