#include <dmk/vector_kernels.hpp>
#include <nanobench.h>

#include <stdexcept>
#include <vector>

constexpr int n_dim = 3;

enum Evaluator : int { DIRECT_CPU };
struct Opts {
    Opts(int argc, char *argv[]) {
        int opt;
        while ((opt = getopt(argc, argv, "N:M:d:t:h?")) != -1) {
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

    int n_warmup = 2;
    int n_src = 800;
    int n_trg = 800;
    int digits = 3;
    char prec = 'f';
    Evaluator eval = DIRECT_CPU;
};

template <class Real>
std::vector<Real> init_random(int size, Real left = 0.0, Real right = 1.0) {
    std::vector<Real> vec(size);
    for (auto &el : vec)
        el = left + (right - left) * drand48();
    return vec;
}

template <class Real>
void laplace_3d_pswf_direct_cpu(const Opts &opts, const std::vector<Real> &r_src, const std::vector<Real> &r_trg,
                                const std::vector<Real> &charge, std::vector<Real> &u) {
    constexpr int nd = 1;
    constexpr Real bsize = 1.0;
    constexpr Real d2max = bsize * bsize;
    constexpr Real cen = -2.0 / bsize;
    constexpr Real thresh2 = 1E-30;
    const int ns = r_src.size() / n_dim;
    const int nt = r_trg.size() / n_dim;
    const Real scale = 2.0 / bsize;
    const Real *xtrg = r_trg.data();
    const Real *ytrg = r_trg.data() + nt;
    const Real *ztrg = r_trg.data() + 2 * nt;

    l3d_local_kernel_directcp_vec_cpp(&nd, &n_dim, &opts.digits, &scale, &cen, &d2max, r_src.data(), &ns, charge.data(),
                                      xtrg, ytrg, ytrg, &nt, u.data(), &thresh2);
}

template <typename Real>
void run_comparison(const Opts &opts) {
    auto r_src_base = init_random<Real>(opts.n_src * n_dim);
    auto r_trg = init_random<Real>(opts.n_trg * n_dim);
    auto charges = init_random<Real>(opts.n_src * n_dim, -1.0, 1.0);
    const int sample_offset = (opts.n_trg - 1);

    std::array<std::vector<Real>, 27> r_src_all;
    for (auto &r : r_src_all)
        r = r_src_base;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                const int image_idx = i * 9 + j * 3 + k;
                const Real dr[3] = {i - Real{1.0}, j - Real{1.0}, k - Real{1.0}};
                for (int s = 0; s < opts.n_src; ++s)
                    for (int l = 0; l < 3; ++l)
                        r_src_all[image_idx][s * 3 + l] += dr[l];
            }

    auto evaluator = [&opts]() {
        switch (opts.eval) {
        case DIRECT_CPU:
            return laplace_3d_pswf_direct_cpu<Real>;
            break;
        default:
            throw std::runtime_error("Unknown evaluator");
        }
    }();

    std::vector<Real> res(opts.n_trg);
    const int n_pairs = opts.n_src * opts.n_trg * 27;
    const int n_warmup = 4e9 / n_pairs;
    ankerl::nanobench::Bench().batch(n_pairs).unit("pair").warmup(n_warmup).run("Eval", [&] {
        for (int i = 0; i < 27; ++i)
            evaluator(opts, r_src_all[i], r_trg, charges, res);
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
