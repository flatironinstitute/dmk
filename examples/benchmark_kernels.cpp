#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/omp_wrapper.hpp>
#include <dmk/util.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef DMK_HAVE_MPI
#include <mpi.h>
#define MYCOMM MPI_COMM_WORLD
#else
#define MYCOMM nullptr
#endif

struct Config {
    int n_src = 1'000'000;
    int n_trg = 0;
    int n_per_leaf = 280;
    double eps = 1e-5;
    char prec = 'f';
    bool uniform = false;
    bool enable_direct = true;
    int n_direct = -1;
    int n_runs = 100;
    int log_level = DMK_LOG_OFF;
    dmk_ikernel kernel = DMK_LAPLACE;
    int n_dim = 3;
    double fparam = 6.0;
    bool bench_build = false;
    bool bench_eval = true;
    bool with_grad = false;
    long seed = 0;
    int n_show_outliers = 0; // print top-N worst points per block to stderr (0 = off)
};

inline dmk_eval_type get_eval_type(dmk_ikernel kernel, bool with_grad) {
    if (kernel == DMK_STOKESLET || kernel == DMK_STRESSLET)
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
void generate_and_scatter(int n_dim, int charge_dim, size_t n_src, size_t n_trg, bool uniform, bool set_fixed_charges,
                          std::vector<Real> &r_src, std::vector<Real> &r_trg, std::vector<Real> &charges,
                          std::vector<Real> &normals, long seed, int rank, int np) {
    const int n_src_local = local_count(n_src, np, rank);
    const int n_trg_local = local_count(n_trg, np, rank);

    std::vector<Real> r_src_all, r_trg_all, charges_all, normals_all;

    if (rank == 0) {
        dmk::util::init_test_data(n_dim, charge_dim, int(n_src), int(n_trg), uniform, set_fixed_charges, r_src_all,
                                  r_trg_all, normals_all, charges_all, seed);
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
    if constexpr (std::is_same_v<Real, float>)
        pdmk_tree_evalf(tree, pot_src.data(), pot_trg_ptr);
    else
        pdmk_tree_eval(tree, pot_src.data(), pot_trg_ptr);
    double ft = MY_OMP_GET_WTIME();

    return ft - st;
}

void print_build_csv_header(std::ostream &os) { os << "build_time,build_pts_s,build_pts_s_rank,build_pts_s_thread"; }

void print_build_csv_row(const TimingResult &t, std::ostream &os) {
    os << t.elapsed << "," << t.pts_per_sec << "," << t.pts_per_sec_per_rank << "," << t.pts_per_sec_per_thread;
}

void print_csv_config_comment(const Config &cfg, int np, int n_threads, std::ostream &os) {
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
       << "# uniform_dist:         " << cfg.uniform << "\n"
       << "# seed:                 " << cfg.seed << "\n"
       << "# eps:                  " << cfg.eps << "\n"
       << "# n_per_leaf:           " << cfg.n_per_leaf << "\n"
       << "# n_runs:               " << cfg.n_runs << "\n"
       << "# direct_enabled:       " << cfg.enable_direct << "\n"
       << "# n_direct:             " << cfg.n_direct << "\n"
       << "# n_show_outliers:      " << cfg.n_show_outliers << "\n"
       << "# log_level:            " << cfg.log_level << "\n"
       << "# bench_build:          " << cfg.bench_build << "\n"
       << "# bench_eval:           " << cfg.bench_eval << "\n";
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

dmk_ikernel parse_kernel(const char *s) {
    if (auto kernel = dmk::util::ikernel_from_string(s))
        return *kernel;
    throw std::runtime_error("Unknown kernel: " + std::string(s));
}

template <typename Real>
void run_benchmark(const Config &cfg) {
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
    params.eval_src = get_eval_type(cfg.kernel, cfg.with_grad);
    params.eval_trg = params.eval_src;
    if (cfg.kernel == DMK_YUKAWA)
        params.fparam = cfg.fparam;

    const int charge_dim = dmk::get_kernel_input_dim(n_dim, params.kernel);
    const int pot_dim = dmk::get_kernel_output_dim(n_dim, cfg.kernel, params.eval_src);

    std::vector<Real> r_src, r_trg, charges, normals;
    generate_and_scatter<Real>(n_dim, charge_dim, n_src, n_trg, cfg.uniform, true, r_src, r_trg, charges, normals,
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
            print_csv_config_comment(cfg, np, n_threads, std::cout);
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

    if (!cfg.bench_eval)
        return;

    pdmk_tree tree = create_tree();

    // Direct reference at source positions, and at target positions if requested.
    std::vector<Real> pot_direct_src, pot_direct_trg;
    if (cfg.enable_direct) {
        const int n_direct_global = (cfg.n_direct > 0) ? cfg.n_direct : n_src;
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
        if (!cfg.bench_build)
            print_csv_config_comment(cfg, np, n_threads, std::cout);
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

    for (int run = 0; run < cfg.n_runs; ++run) {
        std::vector<Real> pot_dmk_src, pot_dmk_trg;
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
    }

    pdmk_tree_destroy(tree);
}

Config parse_args(int argc, char *argv[]) {
    Config cfg;

    static struct option long_opts[] = {
        {"direct", no_argument, nullptr, 1001},
        {"no-direct", no_argument, nullptr, 1002},
        {"bench-build", no_argument, nullptr, 1003},
        {"no-bench-eval", no_argument, nullptr, 1004},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "N:T:n:e:t:r:D:l:s:k:d:f:O:ugh?", long_opts, nullptr)) != -1) {
        switch (opt) {
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
            break;
        case 'f':
            cfg.fparam = std::atof(optarg);
            break;
        case 't':
            if (optarg[0] == 'd')
                cfg.prec = 'd';
            else if (optarg[0] == 'f')
                cfg.prec = 'f';
            else {
                std::cerr << "Unknown precision: " << optarg << "\n";
                exit(1);
            }
            break;
        case 'u':
            cfg.uniform = true;
            break;
        case 'g':
            cfg.with_grad = true;
            break;
        case 1001:
            cfg.enable_direct = true;
            break;
        case 1002:
            cfg.enable_direct = false;
            break;
        case 1003:
            cfg.bench_build = true;
            break;
        case 1004:
            cfg.bench_eval = false;
            break;
        case 'h':
        case '?':
        default:
            std::cout
                << "Usage: " << argv[0] << "\n"
                << "  -N n_src              Number of source points\n"
                << "  -T n_trg              Number of separate target points (default: 0 = source self-eval only)\n"
                << "  -n n_per_leaf         DMK leaf size\n"
                << "  -e eps                Tolerance\n"
                << "  -t f|d                Precision\n"
                << "  -k kernel             laplace, sqrt_laplace, yukawa, stokeslet, stresslet, laplace_dipole\n"
                << "  -d dim                2 or 3\n"
                << "  -f fparam             Yukawa parameter (default: 6.0)\n"
                << "  -r n_runs             Benchmark iterations\n"
                << "  -D n_direct           Points for direct comparison\n"
                << "  -l log_level          DMK log verbosity\n"
                << "  -s seed               integer seed for random numbers\n"
                << "  -u                    Uniform distribution\n"
                << "  -g                    Evaluate potential + gradient (scalar kernels)\n"
                << "  -O n_outliers         Print top-N worst points per block to stderr (default: 0 = off)\n"
                << "  --direct/--no-direct  Enable/disable direct reference\n"
                << "  --bench-build         Also benchmark tree build time\n"
                << "  --no-bench-eval       Skip eval benchmark (build only)\n"
                << "  -h                    Help\n";
            exit(0);
        }
    }
    cfg.n_direct = cfg.n_direct > 0 ? cfg.n_direct : cfg.n_src;
    return cfg;
}

int main(int argc, char *argv[]) {
#ifdef DMK_HAVE_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#endif

    try {
        Config cfg = parse_args(argc, argv);
        if (cfg.prec == 'd')
            run_benchmark<double>(cfg);
        else
            run_benchmark<float>(cfg);
    } catch (std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

#ifdef DMK_HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
