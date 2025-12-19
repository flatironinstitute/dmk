#include <cstdlib>
#include <dmk/direct.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/vector_kernels.hpp>
#include <nanobench.h>

#include <stdexcept>

enum Evaluator : int { DIRECT_CPU };
struct Opts {
    Opts(int argc, char *argv[]) {
        int opt;
        while ((opt = getopt(argc, argv, "N:M:d:D:t:e:u:h?")) != -1) {
            switch (opt) {
            case 'N':
                n_src = std::atof(optarg);
                break;
            case 'M':
                n_trg = std::atof(optarg);
                break;
            case 'd':
                digits = std::atoi(optarg);
                break;
            case 'D':
                n_dim = std::atoi(optarg);
                break;
            case 'u':
                unroll_factor = std::atoi(optarg);
                break;
            case 't':
                if (optarg[0] == 'd')
                    prec = 'd';
                else if (optarg[0] == 'f')
                    prec = 'f';
                else {
                    std::cerr << "Unknown precision: " << optarg << std::endl;
                    throw std::runtime_error("Unknown precision");
                }
                break;
            case 'e':
                if (strcmp(optarg, "CPU") == 0)
                    eval = DIRECT_CPU;
                else {
                    std::cerr << "Unknown evaluator: " << optarg << std::endl;
                    throw std::runtime_error("Unknown evaluator");
                }
                break;
            case 'h':
            case '?':
            default:
                std::cout << "Usage: " << argv[0] << " [-N n_src] [-M n_trg] [-t float_or_double] [-e CPU] [-h]"
                          << std::endl;
                exit(0);
            }
        }
    }

    int n_warmup = 10;
    int n_dim = 3;
    int n_src = 800;
    int n_trg = 800;
    int digits = 3;
    int unroll_factor = 1;
    char prec = 'f';
    Evaluator eval = DIRECT_CPU;
};

template <class Real>
sctl::Vector<Real> init_random(int rows, int cols, Real left = 0.0, Real right = 1.0) {
    sctl::Vector<Real> mat(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat[i * cols + j] = left + (right - left) * drand48();
    return mat;
}

template <class T>
struct PSWFParams {
    const T bsize = 1.0;
    const T rsc = 2.0 / bsize;
    const T cen = -bsize / 2.0;
    const T d2max = bsize * bsize;
    const T thresh2 = 1E-30;
};

template <class Real>
void laplace_pswf_direct_cpu(const Opts &opts, const sctl::Vector<Real> &r_src, const sctl::Vector<Real> &r_trg,
                             const sctl::Vector<Real> &charge, sctl::Vector<Real> &u) {
    constexpr int nd = 1;
    constexpr PSWFParams<Real> p;
    const int ns = r_src.Dim() / opts.n_dim;
    const int nt = r_trg.Dim() / opts.n_dim;
    const Real *xtrg = &r_trg[0 * nt];
    const Real *ytrg = &r_trg[1 * nt];
    const Real *ztrg = (opts.n_dim == 2) ? nullptr : &r_trg[2 * nt];

    if (opts.n_dim == 2)
        log_local_kernel_directcp_vec_cpp(&nd, &opts.n_dim, &opts.digits, &p.rsc, &p.cen, &p.d2max, &r_src[0], &ns,
                                          &charge[0], xtrg, ytrg, ztrg, &nt, &u[0], &p.thresh2);
    else
        l3d_local_kernel_directcp_vec_cpp(&nd, &opts.n_dim, &opts.digits, &p.rsc, &p.cen, &p.d2max, &r_src[0], &ns,
                                          &charge[0], xtrg, ytrg, ztrg, &nt, &u[0], &p.thresh2);
}

template <typename Real>
void run_comparison(const Opts &opts) {
    auto r_src_base = init_random<Real>(opts.n_src, opts.n_dim);
    auto r_trg = init_random<Real>(opts.n_trg, opts.n_dim);
    auto charges = init_random<Real>(1, opts.n_src, -1.0, 1.0);
    const int sample_offset = (opts.n_trg - 1);

    std::array<sctl::Vector<Real>, 27> r_src_all;
    for (auto &r : r_src_all)
        r = r_src_base;
    if (opts.n_dim == 2) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                const int image_idx = i * 3 + j;
                const Real dr[2] = {i - Real{1.0}, j - Real{1.0}};
                for (int s = 0; s < opts.n_src; ++s)
                    for (int l = 0; l < 2; ++l)
                        r_src_all[image_idx][s * 2 + l] += dr[l];
            }
    } else {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) {
                    const int image_idx = i * 9 + j * 3 + k;
                    const Real dr[3] = {i - Real{1.0}, j - Real{1.0}, k - Real{1.0}};
                    for (int s = 0; s < opts.n_src; ++s)
                        for (int l = 0; l < 3; ++l)
                            r_src_all[image_idx][s * 3 + l] += dr[l];
                }
    }

    auto r_trg_t = r_trg;
    for (int i = 0; i < opts.n_trg; ++i)
        for (int j = 0; j < opts.n_dim; ++j)
            r_trg_t[j * opts.n_trg + i] = r_trg[i * opts.n_dim + j];

    auto evaluator = [&opts]() {
        switch (opts.eval) {
        case DIRECT_CPU:
            return laplace_pswf_direct_cpu<Real>;
            break;
        default:
            throw std::runtime_error("Unknown evaluator");
        }
    }();
    auto jit_evaluator = dmk::make_evaluator<Real>(DMK_LAPLACE, opts.n_dim, opts.digits);

    constexpr PSWFParams<Real> p;
    sctl::Vector<Real> res(opts.n_trg);
    {
        auto print_16 = [opts](const auto &prefix, auto &res) {
            const int n_sample = std::min(16, opts.n_trg);
            std::cout << prefix << ": ";
            for (int j = 0; j < n_sample; ++j)
                std::cout << res[j] << " ";
            std::cout << std::endl;
        };
        res = 0;
        jit_evaluator(1, p.rsc, p.cen, p.d2max, p.thresh2, opts.n_src, &r_src_base[0], &charges[0], opts.n_trg,
                      &r_trg[0], &res[0]);
        print_16("jit          ", res);

        res = 0;
        evaluator(opts, r_src_base, r_trg_t, charges, res);
        print_16("production   ", res);
    }

    const int n_images = opts.n_dim == 2 ? 9 : 27;
    const int n_pairs = opts.n_src * opts.n_trg * n_images;
    const int n_warmup = 4e9 / n_pairs;
    ankerl::nanobench::Bench().batch(n_pairs).unit("pair").warmup(n_warmup).run("Production", [&] {
        for (int i = 0; i < n_images; ++i)
            evaluator(opts, r_src_all[i], r_trg_t, charges, res);
        ankerl::nanobench::doNotOptimizeAway(res);
    });
    ankerl::nanobench::Bench().batch(n_pairs).unit("pair").run("JIT", [&] {
        for (int i = 0; i < n_images; ++i)
            jit_evaluator(1, p.rsc, p.cen, p.d2max, p.thresh2, opts.n_src, &r_src_all[i][0], &charges[0], opts.n_trg,
                          &r_trg[0], &res[0]);
        ankerl::nanobench::doNotOptimizeAway(res);
    });
}

int main(int argc, char *argv[]) {
    try {
        Opts opts(argc, argv);

        if (opts.prec == 'f')
            run_comparison<float>(opts);
        else if (opts.prec == 'd')
            run_comparison<double>(opts);
        else {
            std::cerr << "Unknown precision: " << opts.prec << std::endl;
            return 1;
        }
        return 0;
    } catch (const std::runtime_error &e) {
        return 1;
    }
}
