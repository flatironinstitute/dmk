// Measures solver accuracy against an exact reference, across kernel/dim/digits for ESP and DMK.
//
// Output is space-padded CSV: it lines up as a table to read, and parses as CSV to analyse, with
// every non-data line '#'-prefixed. Readers need to strip the padding --
// csv.DictReader(f, skipinitialspace=True) or pandas.read_csv(f, comment='#', skipinitialspace=True).
//
// Usage: ./measure_error [options]        (--solver esp for the ESP path)

#include <dmk.h>
#include <dmk/direct.hpp>
#include <dmk/esp.hpp>
#include <dmk/omp_wrapper.hpp>
#include <dmk/periodic_reference.hpp>
#include <dmk/util.hpp>

#include <algorithm>
#include <cmath>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <sctl.hpp>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#ifdef DMK_HAVE_MPI
auto MYCOMM = MPI_COMM_WORLD;
#else
auto MYCOMM = nullptr;
#endif

enum class Solver { DMK, ESP };

struct Config {
    Solver solver = Solver::DMK;
    // Shared
    int n_src = 10'000;
    int n_direct = 10'000;
    char prec = 'f';
    dmk::util::Distribution dist = dmk::util::Distribution::Uniform;
    bool grad = false;
    bool use_periodic = false;
    double fparam = 6.0; // Yukawa lambda
    dmk_eval_path eval_path = DMK_EVAL_PATH_CPU;
    int log_level = DMK_LOG_OFF;
    long seed = 0;
    dmk_ikernel kernel_filter = static_cast<dmk_ikernel>(-1); // -1 = all
    int dim_filter = 0;                                       // 0 = both
    int dig_min = 3;
    int dig_max = 0; // 0 -> precision-dependent default (6 float, 12 double)
    // DMK only
    int n_per_leaf = 250;
    bool beta_sweep = false;
    double beta_min = 3.0;
    double beta_max = 40.0;
    double beta_step = 0.5;
    int sweep_digits = 6;
    // ESP only
    double sigma = 1.35;
    bool sigma_set = false;
    double rc_min = 0.05, rc_max = 0.25, rc_step = 0.05;
};

struct ErrorMetrics {
    double pot_l2, pot_max;
    double grad_l2, grad_max; // 0 when gradients were not measured
    double time;
};

// ---------------------------------------------------------------------------
// Shared kernel traits, test data and references
// ---------------------------------------------------------------------------

bool is_velocity(dmk_ikernel kernel) { return kernel == DMK_STOKESLET || kernel == DMK_STRESSLET; }
bool is_scalar(dmk_ikernel kernel) {
    return kernel == DMK_LAPLACE || kernel == DMK_YUKAWA || kernel == DMK_SQRT_LAPLACE;
}
bool is_3d_only(dmk_ikernel kernel) { return is_velocity(kernel) || kernel == DMK_LAPLACE_DIPOLE; }
bool needs_normal(dmk_ikernel kernel) { return kernel == DMK_STRESSLET; }

dmk_eval_type eval_level_for(dmk_ikernel kernel, bool grad) {
    if (is_velocity(kernel))
        return DMK_VELOCITY;
    return grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
}

int max_digits_for(char prec) { return prec == 'd' ? 12 : 6; }

dmk_ikernel parse_kernel(const char *s) {
    std::string k(s);
    if (k == "all")
        return static_cast<dmk_ikernel>(-1);
    if (auto kernel = dmk::util::ikernel_from_string(k))
        return *kernel;
    throw std::runtime_error("Unknown kernel: " + k);
}

// Sources for both solvers. init_test_data guarantees points strictly interior to [0,1) at the
// working precision and emits the unit normals the Stresslet needs. Its charges are not neutral, so
// the periodic lattice sums need the per-component mean removed first.
template <typename Real>
void generate_test_data(const Config &cfg, int n_dim, dmk_ikernel kernel, std::vector<Real> &r_src,
                        std::vector<Real> &charges, std::vector<Real> &normals) {
    const int input_dim = dmk::get_kernel_input_dim(n_dim, kernel);
    std::vector<Real> r_trg;
    dmk::util::init_test_data(n_dim, input_dim, cfg.n_src, 0, cfg.dist, false, r_src, r_trg, normals, charges,
                              cfg.seed);
    if (!cfg.use_periodic)
        return;
    for (int c = 0; c < input_dim; ++c) {
        double mean = 0;
        for (int i = 0; i < cfg.n_src; ++i)
            mean += charges[size_t(i) * input_dim + c];
        mean /= cfg.n_src;
        for (int i = 0; i < cfg.n_src; ++i)
            charges[size_t(i) * input_dim + c] -= Real(mean);
    }
}

// Exact interleaved reference at the first n_cmp sources (self pair excluded), in the layout the
// solver reports for `eval_level`. Returns false when this configuration has no self-contained
// reference. Summed in double from the same points the solver sees, so float summation error does
// not impose an accuracy floor.
template <typename Real>
bool compute_reference(const Config &cfg, int n_dim, dmk_ikernel kernel, dmk_eval_type eval_level, int n,
                       const std::vector<Real> &r_src_r, const std::vector<Real> &charges_r,
                       const std::vector<Real> &normals_r, int n_cmp, std::vector<double> &ref) {
    const int od = dmk::get_kernel_output_dim(n_dim, kernel, eval_level);
    ref.assign(size_t(n_cmp) * od, 0.0);
    const std::vector<double> r_src(r_src_r.begin(), r_src_r.end());
    const std::vector<double> charges(charges_r.begin(), charges_r.end());
    const std::vector<double> normals(normals_r.begin(), normals_r.end());
    std::vector<double> r_trg(r_src.begin(), r_src.begin() + size_t(n_cmp) * n_dim);

    if (!cfg.use_periodic) {
        dmk::compute_direct(n_dim, r_src, charges, normals, r_trg, ref, kernel, eval_level, cfg.fparam);
        return true;
    }

    if (kernel == DMK_LAPLACE && n_dim == 2) {
        // The 2D log kernel has no self-contained exact periodic reference (EwaldRef/image_sum don't
        // cover it). ESP can borrow DMK's own periodic pipeline at eps=1e-12, which shares its log
        // self/gauge convention (validated in test_esp); for DMK that reference would be circular.
        if (cfg.solver == Solver::DMK)
            return false;
        std::vector<double> rnormal(size_t(n) * n_dim, 0.0);
        pdmk_params params;
        params.eps = 1e-12;
        params.n_dim = n_dim;
        params.n_per_leaf = 280;
        params.eval_src = eval_level;
        params.eval_trg = eval_level;
        params.kernel = DMK_LAPLACE;
        params.use_periodic = true;
        params.log_level = 6;
        std::vector<double> pot_src(size_t(n) * od), pot_trg;
        pdmk_tree tree = pdmk_tree_create(MYCOMM, params, n, r_src.data(), charges.data(), rnormal.data(), 0, nullptr);
        pdmk_tree_eval(tree, pot_src.data(), pot_trg.data());
        pdmk_tree_destroy(tree);
        std::copy(pot_src.begin(), pot_src.begin() + size_t(n_cmp) * od, ref.begin());
        return true;
    }

    if (kernel == DMK_YUKAWA) {
        const int n_img = std::max(2, int(std::ceil(21.0 / cfg.fparam))); // exp(-lambda*n_img) ~ 1e-9
        dmk::pbc_ref::image_sum(n_dim, cfg.fparam, n_img, eval_level, n, r_src.data(), charges.data(), 1.0, n_cmp,
                                r_trg.data(), ref);
        return true;
    }

    if (!is_scalar(kernel))
        return false; // dipole/Stokes are free-space only

    dmk::pbc_ref::EwaldRef ewald(kernel, n_dim, n, r_src.data(), charges.data(), 1.0, 15.0);
#pragma omp parallel for
    for (int i = 0; i < n_cmp; ++i) {
        double pot, grad[3] = {0, 0, 0};
        ewald.eval(&r_src[size_t(i) * n_dim], i, pot, od > 1 ? grad : nullptr);
        ref[size_t(i) * od] = pot;
        for (int d = 0; d + 1 < od; ++d)
            ref[size_t(i) * od + 1 + d] = grad[d];
    }
    return true;
}

// Compares the solver's interleaved output against the reference over the first n_cmp points. The
// leading n_lead components are the primary field (velocity kernels: all of them; else the single
// potential), the rest are the gradient axes. A periodic potential is fixed only up to an additive
// constant, so both sides are mean-subtracted there.
ErrorMetrics compare(const Config &cfg, int n_cmp, int od, int n_lead, const std::vector<double> &field,
                     const std::vector<double> &ref, double time) {
    std::vector<double> mean_f(n_lead, 0.0), mean_r(n_lead, 0.0);
    if (cfg.use_periodic) {
        for (int i = 0; i < n_cmp; ++i)
            for (int c = 0; c < n_lead; ++c) {
                mean_f[c] += field[size_t(i) * od + c];
                mean_r[c] += ref[size_t(i) * od + c];
            }
        for (int c = 0; c < n_lead; ++c) {
            mean_f[c] /= n_cmp;
            mean_r[c] /= n_cmp;
        }
    }

    double pe2 = 0, pr2 = 0, pmax = 0, ge2 = 0, gr2 = 0, gmax = 0;
    for (int i = 0; i < n_cmp; ++i) {
        double pd2 = 0, pr2i = 0;
        for (int c = 0; c < n_lead; ++c) {
            const double d = (field[size_t(i) * od + c] - mean_f[c]) - (ref[size_t(i) * od + c] - mean_r[c]);
            const double r = ref[size_t(i) * od + c] - mean_r[c];
            pd2 += d * d;
            pr2i += r * r;
        }
        pe2 += pd2;
        pr2 += pr2i;
        if (pr2i > 0.0)
            pmax = std::max(pmax, std::sqrt(pd2 / pr2i));

        double gd2 = 0, gr2i = 0;
        for (int c = n_lead; c < od; ++c) {
            const double d = field[size_t(i) * od + c] - ref[size_t(i) * od + c];
            const double r = ref[size_t(i) * od + c];
            gd2 += d * d;
            gr2i += r * r;
        }
        ge2 += gd2;
        gr2 += gr2i;
        if (gr2i > 0.0)
            gmax = std::max(gmax, std::sqrt(gd2 / gr2i));
    }
    return {std::sqrt(pe2 / pr2), pmax, gr2 > 0.0 ? std::sqrt(ge2 / gr2) : 0.0, gmax, time};
}

// ---------------------------------------------------------------------------
// DMK point tree
// ---------------------------------------------------------------------------

template <typename Real>
ErrorMetrics dmk_run_one(int n_dim, dmk_ikernel kernel, int n_digits, const Config &cfg, const std::vector<Real> &r_src,
                         const std::vector<Real> &charges, const std::vector<Real> &normals,
                         const std::vector<double> &ref, dmk_eval_type eval_level, double beta_override = -1.0) {
    const int n_src = cfg.n_src;
    const int n_cmp = std::min(cfg.n_direct, n_src);
    const int out_dim = dmk::get_kernel_output_dim(n_dim, kernel, eval_level);

    pdmk_params params{};
    params.eps = std::pow(10.0, -n_digits);
    params.n_dim = n_dim;
    params.n_per_leaf = cfg.n_per_leaf;
    params.log_level = cfg.log_level;
    params.eval_src = eval_level;
    params.eval_trg = eval_level;
    params.kernel = kernel;
    params.eval_path = cfg.eval_path;
    params.use_periodic = cfg.use_periodic;
    if (kernel == DMK_YUKAWA)
        params.fparam = cfg.fparam;

    if (beta_override > 0) {
        params.debug_flags |= DMK_DEBUG_OVERRIDE_BETA;
        params.debug_params[DMK_DEBUG_BETA_SLOT] = beta_override;
    }

    pdmk_tree tree = [&]() {
        if constexpr (std::is_same_v<Real, float>)
            return pdmk_tree_createf(MYCOMM, params, n_src, r_src.data(), charges.data(), normals.data(), 0, nullptr);
        else
            return pdmk_tree_create(MYCOMM, params, n_src, r_src.data(), charges.data(), normals.data(), 0, nullptr);
    }();

    if (!tree)
        throw std::runtime_error(pdmk_last_error_message());

    std::vector<Real> pot_dmk(size_t(n_src) * out_dim);

    double st = MY_OMP_GET_WTIME();
    dmk_error rc = [&]() {
        if constexpr (std::is_same_v<Real, float>)
            return pdmk_tree_evalf(tree, pot_dmk.data(), nullptr);
        else
            return pdmk_tree_eval(tree, pot_dmk.data(), nullptr);
    }();
    double dt = MY_OMP_GET_WTIME() - st;

    pdmk_tree_destroy(tree);
    if (rc != DMK_SUCCESS)
        throw std::runtime_error(pdmk_last_error_message());

    const int n_lead = (eval_level == DMK_POTENTIAL_GRAD) ? 1 : out_dim;
    std::vector<double> field(pot_dmk.begin(), pot_dmk.begin() + size_t(n_cmp) * out_dim);
    return compare(cfg, n_cmp, out_dim, n_lead, field, ref, dt);
}

// The (kernel, dim) pairs a run covers, after the filters and each solver's 3D-only rules.
std::vector<dmk_ikernel> selected_kernels(const Config &cfg) {
    if (static_cast<int>(cfg.kernel_filter) != -1)
        return {cfg.kernel_filter};
    return {DMK_LAPLACE, DMK_SQRT_LAPLACE, DMK_LAPLACE_DIPOLE, DMK_YUKAWA, DMK_STOKESLET, DMK_STRESSLET};
}

std::vector<int> selected_dims(const Config &cfg) {
    if (cfg.dim_filter != 0)
        return {cfg.dim_filter};
    return {2, 3};
}

// ---------------------------------------------------------------------------
// Output: aligned CSV
// ---------------------------------------------------------------------------

// Every row is comma-separated with each field right-aligned to a fixed width, so a run reads like a
// table and still parses as CSV: leading blanks in an unquoted field are legal, and int()/float()/
// csv/pandas all strip them. Anything that is not a data row is '#'-prefixed.
//
// Columns are only what cannot be recovered from the '#' config line: sigma, the spread width, the
// PSWF bandwidth and the grid size are all deterministic functions of (digits, sigma, dim, r_c,
// boundary), so they are not repeated on every row.
struct Table {
    std::vector<std::string> head;
    std::vector<int> width;

    void add(std::string name, int w) {
        width.push_back(std::max<int>(w, name.size()));
        head.push_back(std::move(name));
    }
    void row(const std::vector<std::string> &cells) const {
        for (size_t i = 0; i < cells.size(); ++i)
            std::cout << (i ? ", " : "") << std::setw(width[i]) << cells[i];
        std::cout << "\n" << std::flush;
    }
    void header() const {
        row(head);
        int n = -2;
        for (int w : width)
            n += w + 2;
        std::cout << "# " << std::string(std::max(0, n), '-') << "\n";
    }
    // Identity cells, then FAILED in every metric column, with the reason on a comment line.
    void failed(std::vector<std::string> cells, const std::string &why) const {
        std::cout << "# " << why << "\n";
        while (cells.size() < head.size())
            cells.push_back("FAILED");
        row(cells);
    }
};

std::string kname(dmk_ikernel k) { return std::string(dmk::util::to_string(k)); }
std::string sci(double v, int prec = 3) {
    std::ostringstream o;
    o << std::scientific << std::setprecision(prec) << v;
    return o.str();
}
std::string fix(double v, int prec) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v;
    return o.str();
}

// The metric block every sweep shares. `ratio` adds L2_over_eps, which the beta sweep has no eps for.
void add_metrics(Table &t, const Config &cfg, bool ratio) {
    t.add("L2_rel", 10);
    t.add("max_rel", 10);
    if (cfg.grad) {
        t.add("grad_L2_rel", 11);
        t.add("grad_max_rel", 12);
    }
    t.add("time_s", 8);
    if (ratio)
        t.add("L2_over_eps", 11);
}

void append_metrics(std::vector<std::string> &cells, const Config &cfg, const ErrorMetrics &e, dmk_ikernel kernel,
                    double eps, bool ratio) {
    cells.push_back(sci(e.pot_l2));
    cells.push_back(sci(e.pot_max));
    if (cfg.grad) {
        const bool has_grad = !is_velocity(kernel); // velocity kernels have no separate gradient block
        cells.push_back(has_grad ? sci(e.grad_l2) : "-");
        cells.push_back(has_grad ? sci(e.grad_max) : "-");
    }
    cells.push_back(fix(e.time, 4));
    if (ratio)
        cells.push_back(fix(e.pot_l2 / eps, 1));
}

template <typename Real>
void dmk_run_beta_sweep(const Config &cfg) {
    Table t;
    t.add("kernel", 14);
    t.add("dim", 3);
    t.add("beta", 6);
    add_metrics(t, cfg, false);
    t.header();

    for (auto kernel : selected_kernels(cfg)) {
        const dmk_eval_type eval_level = eval_level_for(kernel, cfg.grad);
        for (auto n_dim : selected_dims(cfg)) {
            if (n_dim != 3 && is_3d_only(kernel))
                continue;
            std::vector<Real> r_src, charges, rnormal;
            std::vector<double> ref;
            try {
                generate_test_data<Real>(cfg, n_dim, kernel, r_src, charges, rnormal);
                const int n_cmp = std::min(cfg.n_direct, cfg.n_src);
                if (!compute_reference<Real>(cfg, n_dim, kernel, eval_level, cfg.n_src, r_src, charges, rnormal, n_cmp,
                                             ref)) {
                    std::cout << "# " << kname(kernel) << " dim=" << n_dim
                              << ": no self-contained reference, skipping\n";
                    continue;
                }
            } catch (std::exception &e) {
                std::cout << "# " << kname(kernel) << " dim=" << n_dim << ": " << e.what() << "\n";
                continue;
            }
            for (double beta = cfg.beta_min; beta <= cfg.beta_max + 1e-9; beta += cfg.beta_step) {
                std::vector<std::string> cells = {kname(kernel), std::to_string(n_dim), fix(beta, 1)};
                try {
                    auto err =
                        dmk_run_one<Real>(n_dim, kernel, 12, cfg, r_src, charges, rnormal, ref, eval_level, beta);
                    append_metrics(cells, cfg, err, kernel, 0.0, false);
                    t.row(cells);
                } catch (std::exception &e) {
                    t.failed(cells, kname(kernel) + " dim=" + std::to_string(n_dim) + " beta=" + fix(beta, 1) + ": " +
                                        e.what());
                }
            }
        }
    }
}

template <typename Real>
void dmk_run_all(const Config &cfg) {
    Table t;
    t.add("kernel", 14);
    t.add("dim", 3);
    t.add("digits", 6);
    t.add("eps", 8);
    add_metrics(t, cfg, true);
    t.header();

    for (auto kernel : selected_kernels(cfg)) {
        const dmk_eval_type eval_level = eval_level_for(kernel, cfg.grad);
        for (auto n_dim : selected_dims(cfg)) {
            if (n_dim != 3 && is_3d_only(kernel))
                continue;
            std::vector<Real> r_src, charges, rnormal;
            std::vector<double> ref;
            try {
                generate_test_data<Real>(cfg, n_dim, kernel, r_src, charges, rnormal);
                const int n_cmp = std::min(cfg.n_direct, cfg.n_src);
                if (!compute_reference<Real>(cfg, n_dim, kernel, eval_level, cfg.n_src, r_src, charges, rnormal, n_cmp,
                                             ref)) {
                    std::cout << "# " << kname(kernel) << " dim=" << n_dim
                              << ": no self-contained reference, skipping\n";
                    continue;
                }
            } catch (std::exception &e) {
                std::cout << "# " << kname(kernel) << " dim=" << n_dim << ": " << e.what() << "\n";
                continue;
            }

            for (int digits = cfg.dig_min; digits <= cfg.dig_max; ++digits) {
                const double eps = std::pow(10.0, -digits);
                std::vector<std::string> cells = {kname(kernel), std::to_string(n_dim), std::to_string(digits),
                                                  sci(eps, 0)};
                try {
                    auto err = dmk_run_one<Real>(n_dim, kernel, digits, cfg, r_src, charges, rnormal, ref, eval_level);
                    append_metrics(cells, cfg, err, kernel, eps, true);
                    t.row(cells);
                } catch (std::exception &e) {
                    t.failed(cells, kname(kernel) + " dim=" + std::to_string(n_dim) +
                                        " digits=" + std::to_string(digits) + ": " + e.what());
                }
            }
        }
    }
}
// ---------------------------------------------------------------------------
// ESP
// ---------------------------------------------------------------------------

template <typename Real>
pdmk_esp_plan esp_plan_create(pdmk_esp_params params) {
    if constexpr (std::is_same_v<Real, float>)
        return pdmk_esp_plan_createf(MYCOMM, params);
    else
        return pdmk_esp_plan_create(MYCOMM, params);
}

template <typename Real>
void esp_plan_destroy(pdmk_esp_plan plan) {
    if constexpr (std::is_same_v<Real, float>)
        pdmk_esp_plan_destroyf(plan);
    else
        pdmk_esp_plan_destroy(plan);
}

template <typename Real>
ErrorMetrics esp_run_one(const Config &cfg, int n_dim, dmk_ikernel kernel, dmk_eval_type eval_level, int digits,
                         double r_c, const std::vector<Real> &r_src, const std::vector<Real> &charges,
                         const std::vector<Real> &normals, const std::vector<double> &ref, int n_cmp) {
    const int n = cfg.n_src;
    const int od = dmk::get_kernel_output_dim(n_dim, kernel, eval_level);

    pdmk_esp_params params{};
    params.r_c = r_c;
    params.eps = std::pow(10.0, -digits);
    params.log_level = cfg.log_level;
    params.kernel = kernel;
    params.n_dim = n_dim;
    params.eval_type = eval_level;
    params.sigma = cfg.sigma;
    params.use_periodic = cfg.use_periodic ? 1 : 0;
    params.eval_path = cfg.eval_path;
    if (kernel == DMK_YUKAWA)
        params.fparam = cfg.fparam;

    pdmk_esp_plan plan = esp_plan_create<Real>(params);
    if (!plan) // e.g. the requested precision exceeds what this sigma can spread
        throw std::runtime_error(pdmk_last_error_message());

    std::vector<Real> pot(size_t(n) * od);
    const Real *normal = normals.empty() ? nullptr : normals.data();
    const double st = MY_OMP_GET_WTIME();
    dmk_error rc = [&]() {
        if constexpr (std::is_same_v<Real, float>)
            return pdmk_esp_evalf(MYCOMM, plan, n, r_src.data(), charges.data(), normal, pot.data());
        else
            return pdmk_esp_eval(MYCOMM, plan, n, r_src.data(), charges.data(), normal, pot.data());
    }();
    const double dt = MY_OMP_GET_WTIME() - st;
    esp_plan_destroy<Real>(plan);
    if (rc != DMK_SUCCESS)
        throw std::runtime_error(pdmk_last_error_message());

    const int n_lead = is_velocity(kernel) ? od : 1;
    std::vector<double> field(pot.begin(), pot.begin() + size_t(n_cmp) * od);
    return compare(cfg, n_cmp, od, n_lead, field, ref, dt);
}

template <typename Real>
void esp_run_sweep(Config cfg) {
    auto kernels = selected_kernels(cfg);
    auto dims = selected_dims(cfg);

    if (cfg.eval_path == DMK_EVAL_PATH_GPU && dims != std::vector<int>{3}) {
        std::cout << "# note: the GPU path is 3D only; restricting to dim=3\n";
        dims = {3};
    }

    Table t;
    t.add("kernel", 14);
    t.add("dim", 3);
    t.add("digits", 6);
    t.add("eps", 8);
    t.add("r_c", 6);
    add_metrics(t, cfg, true);
    t.header();

    const int n = cfg.n_src;
    const int n_cmp = std::min(cfg.n_direct, n);

    for (auto kernel : kernels) {
        const dmk_eval_type eval_level = eval_level_for(kernel, cfg.grad);
        for (auto n_dim : dims) {
            if (n_dim != 3 && is_3d_only(kernel))
                continue;

            std::vector<Real> r_src, charges, normals;
            generate_test_data<Real>(cfg, n_dim, kernel, r_src, charges, normals);
            if (!needs_normal(kernel))
                normals.clear();

            std::vector<double> ref;
            if (!compute_reference<Real>(cfg, n_dim, kernel, eval_level, n, r_src, charges, normals, n_cmp, ref)) {
                std::cout << "# " << kname(kernel) << " dim=" << n_dim << ": no self-contained reference, skipping\n"
                          << std::flush;
                continue;
            }

            for (int digits = cfg.dig_min; digits <= cfg.dig_max; ++digits) {
                const double eps = std::pow(10.0, -digits);
                bool unreachable = false;
                int rc_index = 0;
                for (double r_c = cfg.rc_min; r_c <= cfg.rc_max + 1e-9; r_c += cfg.rc_step, ++rc_index) {
                    std::vector<std::string> cells = {kname(kernel), std::to_string(n_dim), std::to_string(digits),
                                                      sci(eps, 0), fix(r_c, 4)};
                    try {
                        auto e = esp_run_one<Real>(cfg, n_dim, kernel, eval_level, digits, r_c, r_src, charges, normals,
                                                   ref, n_cmp);
                        append_metrics(cells, cfg, e, kernel, eps, true);
                        t.row(cells);
                    } catch (std::exception &ex) {
                        // The spread width caps the reachable precision independently of r_c, so a
                        // failure on the first r_c means every higher digit count fails too: say so
                        // once and stop climbing, rather than emit a wall of FAILED rows. A failure
                        // at a later r_c is something else, so keep going.
                        if (rc_index == 0) {
                            std::cout << "# note: " << kname(kernel) << " dim=" << n_dim << " stops below " << digits
                                      << " digits: " << ex.what() << "\n"
                                      << std::flush;
                            unreachable = true;
                            break;
                        }
                        t.failed(cells, kname(kernel) + " dim=" + std::to_string(n_dim) + " digits=" +
                                            std::to_string(digits) + " r_c=" + fix(r_c, 4) + ": " + ex.what());
                    }
                }
                if (unreachable)
                    break;
            }
        }
    }
}
// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

enum {
    OPT_BETA_SWEEP = 1001,
    OPT_BETA_MIN,
    OPT_BETA_MAX,
    OPT_BETA_STEP,
    OPT_DIGITS,
    OPT_LOG_LEVEL,
    OPT_SOLVER,
    OPT_PERIODIC,
    OPT_SIGMA,
    OPT_DIG_MIN,
    OPT_DIG_MAX,
    OPT_RC_MIN,
    OPT_RC_MAX,
    OPT_RC_STEP,
    OPT_SEED,
};

void print_usage(const char *argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "  --solver dmk|esp  Backend (default dmk)\n"
              << "\n"
              << "shared:\n"
              << "  -N n_src          Number of source points (default 10000)\n"
              << "  -D n_direct       Points compared against the reference (default 10000)\n"
              << "  -t f|d            Precision (default f)\n"
              << "  -k kernel         laplace, sqrt_laplace, laplace_dipole, yukawa, stokeslet,\n"
              << "                    stresslet, all (default all)\n"
              << "  -d dim            2, 3, or 0 for both (default 0)\n"
              << "  -l lambda         Yukawa fparam (default 6.0)\n"
              << "  -p c|g            Eval path: CPU or GPU (default c). ESP GPU is 3D only.\n"
              << "  -u dist           0=uniform (default), 1=sphere_surface, 2=box_partial_facet\n"
              << "  -g                Also measure gradient error (ignored for the velocity kernels)\n"
              << "  --periodic        Periodic boundaries (default free-space)\n"
              << "  --dig-min val     Min solver digits (default 3)\n"
              << "  --dig-max val     Max solver digits (default 6 for -t f, 12 for -t d)\n"
              << "  --seed val        RNG seed for the test data (default 0)\n"
              << "  --log-level val   0=trace 1=debug .. 6=off (default 6). JIT compile timings\n"
              << "                    are reported at debug.\n"
              << "  -h                Help\n"
              << "\n"
              << "--solver dmk only:\n"
              << "  -n n_per_leaf     DMK leaf size (default 250)\n"
              << "  --beta-sweep      Enable beta sweep mode\n"
              << "  --beta-min val    Min beta (default 3.0)\n"
              << "  --beta-max val    Max beta (default 40.0)\n"
              << "  --beta-step val   Step size (default 0.5)\n"
              << "  --digits val      Digits for the beta sweep (default 6)\n"
              << "\n"
              << "--solver esp only:\n"
              << "  --sigma s         FINUFFT upsampling (default 1.35; != 1.35 requires\n"
              << "                    -DDMK_USE_JIT=ON)\n"
              << "  --rc-min val      Min r_c (default 0.05)\n"
              << "  --rc-max val      Max r_c (default 0.25)\n"
              << "  --rc-step val     r_c step (default 0.05)\n";
}

Config parse_args(int argc, char *argv[]) {
    Config cfg;

    static struct option long_opts[] = {
        {"beta-sweep", no_argument, nullptr, OPT_BETA_SWEEP},
        {"beta-min", required_argument, nullptr, OPT_BETA_MIN},
        {"beta-max", required_argument, nullptr, OPT_BETA_MAX},
        {"beta-step", required_argument, nullptr, OPT_BETA_STEP},
        {"digits", required_argument, nullptr, OPT_DIGITS},
        {"log-level", required_argument, nullptr, OPT_LOG_LEVEL},
        {"solver", required_argument, nullptr, OPT_SOLVER},
        {"periodic", no_argument, nullptr, OPT_PERIODIC},
        {"sigma", required_argument, nullptr, OPT_SIGMA},
        {"dig-min", required_argument, nullptr, OPT_DIG_MIN},
        {"dig-max", required_argument, nullptr, OPT_DIG_MAX},
        {"rc-min", required_argument, nullptr, OPT_RC_MIN},
        {"rc-max", required_argument, nullptr, OPT_RC_MAX},
        {"rc-step", required_argument, nullptr, OPT_RC_STEP},
        {"seed", required_argument, nullptr, OPT_SEED},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "N:n:D:t:k:d:l:p:u:gh", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'N':
            cfg.n_src = static_cast<int>(std::atof(optarg));
            break;
        case 'n':
            cfg.n_per_leaf = std::atoi(optarg);
            break;
        case 'D':
            cfg.n_direct = static_cast<int>(std::atof(optarg));
            break;
        case 'k':
            cfg.kernel_filter = parse_kernel(optarg);
            break;
        case 'd':
            cfg.dim_filter = std::atoi(optarg);
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
        case 'l':
            cfg.fparam = std::atof(optarg);
            break;
        case 'p':
            if (optarg[0] == 'c')
                cfg.eval_path = DMK_EVAL_PATH_CPU;
            else if (optarg[0] == 'g')
                cfg.eval_path = DMK_EVAL_PATH_GPU;
            else {
                std::cerr << "Unknown eval_path: " << optarg << "\n";
                exit(1);
            }
            break;
        case 'u':
            cfg.dist = dmk::util::Distribution(std::atoi(optarg));
            break;
        case 'g':
            cfg.grad = true;
            break;
        case OPT_SOLVER:
            if (std::string_view(optarg) == "dmk")
                cfg.solver = Solver::DMK;
            else if (std::string_view(optarg) == "esp")
                cfg.solver = Solver::ESP;
            else {
                std::cerr << "Unknown solver: " << optarg << " (use dmk or esp)\n";
                exit(1);
            }
            break;
        case OPT_PERIODIC:
            cfg.use_periodic = true;
            break;
        case OPT_SIGMA:
            cfg.sigma = std::atof(optarg);
            cfg.sigma_set = true;
            break;
        case OPT_DIG_MIN:
            cfg.dig_min = std::atoi(optarg);
            break;
        case OPT_DIG_MAX:
            cfg.dig_max = std::atoi(optarg);
            break;
        case OPT_RC_MIN:
            cfg.rc_min = std::atof(optarg);
            break;
        case OPT_RC_MAX:
            cfg.rc_max = std::atof(optarg);
            break;
        case OPT_RC_STEP:
            cfg.rc_step = std::atof(optarg);
            break;
        case OPT_SEED:
            cfg.seed = std::atol(optarg);
            break;
        case OPT_BETA_SWEEP:
            cfg.beta_sweep = true;
            break;
        case OPT_BETA_MIN:
            cfg.beta_min = std::atof(optarg);
            break;
        case OPT_BETA_MAX:
            cfg.beta_max = std::atof(optarg);
            break;
        case OPT_BETA_STEP:
            cfg.beta_step = std::atof(optarg);
            break;
        case OPT_DIGITS:
            cfg.sweep_digits = std::atoi(optarg);
            break;
        case OPT_LOG_LEVEL:
            cfg.log_level = std::atoi(optarg);
            break;
        case 'h':
        default:
            print_usage(argv[0]);
            exit(0);
        }
    }

    if (cfg.dig_max <= 0)
        cfg.dig_max = max_digits_for(cfg.prec);
    return cfg;
}

int main(int argc, char *argv[]) {
#ifdef DMK_HAVE_MPI
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size > 1) {
        if (!rank)
            std::cerr << "measure_error is not MPI aware. Exiting\n";
        MPI_Finalize();
        return 0;
    }
#endif

    try {
        Config cfg = parse_args(argc, argv);

#ifndef DMK_USE_JIT
        if (cfg.sigma_set && cfg.sigma != 1.35) {
            std::cerr << "error: --sigma != 1.35 requires JIT support (recompile with -DDMK_USE_JIT=ON)\n";
            return 1;
        }
#endif

        std::cout << "# solver=" << (cfg.solver == Solver::ESP ? "esp" : "dmk") << " n_src=" << cfg.n_src
                  << " n_direct=" << cfg.n_direct << " prec=" << cfg.prec << " dist=" << int(cfg.dist)
                  << " grad=" << cfg.grad << " boundary=" << (cfg.use_periodic ? "periodic" : "free-space")
                  << " fparam=" << cfg.fparam << " path=" << (cfg.eval_path == DMK_EVAL_PATH_GPU ? "g" : "c")
                  << " digits=[" << cfg.dig_min << "," << cfg.dig_max << "] seed=" << cfg.seed
                  << " threads=" << MY_OMP_GET_MAX_THREADS();
        if (cfg.solver == Solver::ESP)
            std::cout << " sigma=" << cfg.sigma << " rc=[" << cfg.rc_min << "," << cfg.rc_max << "," << cfg.rc_step
                      << "]";
        else if (cfg.beta_sweep)
            std::cout << " beta_sweep=[" << cfg.beta_min << "," << cfg.beta_max << "," << cfg.beta_step
                      << "] digits=" << cfg.sweep_digits;
        else
            std::cout << " n_per_leaf=" << cfg.n_per_leaf;
        std::cout << "\n"
                  << "# csv: fields are space-padded; read with skipinitialspace=True\n\n";

        if (cfg.solver == Solver::ESP) {
            if (cfg.prec == 'd')
                esp_run_sweep<double>(cfg);
            else
                esp_run_sweep<float>(cfg);
        } else if (cfg.beta_sweep) {
            if (cfg.prec == 'd')
                dmk_run_beta_sweep<double>(cfg);
            else
                dmk_run_beta_sweep<float>(cfg);
        } else {
            if (cfg.prec == 'd')
                dmk_run_all<double>(cfg);
            else
                dmk_run_all<float>(cfg);
        }
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

#ifdef DMK_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
