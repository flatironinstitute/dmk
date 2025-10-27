#include <cstdlib>
#include <dmk/vector_kernels.hpp>
#include <nanobench.h>

#include <stdexcept>

#include <dmk/prolate0_eval.hpp>

constexpr int n_dim = 3;

template <typename T>
T procl180_rescale(T eps) {
    constexpr float cs[] = {
        .43368E-16, .10048E+01, .17298E+01, .22271E+01, .26382E+01, .30035E+01, .33409E+01, .36598E+01, .39658E+01,
        .42621E+01, .45513E+01, .48347E+01, .51136E+01, .53887E+01, .56606E+01, .59299E+01, .61968E+01, .64616E+01,
        .67247E+01, .69862E+01, .72462E+01, .75049E+01, .77625E+01, .80189E+01, .82744E+01, .85289E+01, .87826E+01,
        .90355E+01, .92877E+01, .95392E+01, .97900E+01, .10040E+02, .10290E+02, .10539E+02, .10788E+02, .11036E+02,
        .11284E+02, .11531E+02, .11778E+02, .12024E+02, .12270E+02, .12516E+02, .12762E+02, .13007E+02, .13251E+02,
        .13496E+02, .13740E+02, .13984E+02, .14228E+02, .14471E+02, .14714E+02, .14957E+02, .15200E+02, .15443E+02,
        .15685E+02, .15927E+02, .16169E+02, .16411E+02, .16652E+02, .16894E+02, .17135E+02, .17376E+02, .17617E+02,
        .17858E+02, .18098E+02, .18339E+02, .18579E+02, .18819E+02, .19059E+02, .19299E+02, .19539E+02, .19778E+02,
        .20018E+02, .20257E+02, .20496E+02, .20736E+02, .20975E+02, .21214E+02, .21452E+02, .21691E+02, .21930E+02,
        .22168E+02, .22407E+02, .22645E+02, .22884E+02, .23122E+02, .23360E+02, .23598E+02, .23836E+02, .24074E+02,
        .24311E+02, .24549E+02, .24787E+02, .25024E+02, .25262E+02, .25499E+02, .25737E+02, .25974E+02, .26211E+02,
        .26448E+02, .26685E+02, .26922E+02, .27159E+02, .27396E+02, .27633E+02, .27870E+02, .28106E+02, .28343E+02,
        .28580E+02, .28816E+02, .29053E+02, .29289E+02, .29526E+02, .29762E+02, .29998E+02, .30234E+02, .30471E+02,
        .30707E+02, .30943E+02, .31179E+02, .31415E+02, .31651E+02, .31887E+02, .32123E+02, .32358E+02, .32594E+02,
        .32830E+02, .33066E+02, .33301E+02, .33537E+02, .33773E+02, .34008E+02, .34244E+02, .34479E+02, .34714E+02,
        .34950E+02, .35185E+02, .35421E+02, .35656E+02, .35891E+02, .36126E+02, .36362E+02, .36597E+02, .36832E+02,
        .37067E+02, .37302E+02, .37537E+02, .37772E+02, .38007E+02, .38242E+02, .38477E+02, .38712E+02, .38947E+02,
        .39181E+02, .39416E+02, .39651E+02, .39886E+02, .40120E+02, .40355E+02, .40590E+02, .40824E+02, .41059E+02,
        .41294E+02, .41528E+02, .41763E+02, .41997E+02, .42232E+02, .42466E+02, .42700E+02, .42935E+02, .43169E+02,
        .43404E+02, .43638E+02, .43872E+02, .44107E+02, .44341E+02, .44575E+02, .44809E+02, .45044E+02, .45278E+02};

    int scale;
    if (eps >= 1.0E-3)
        scale = 8;
    else if (eps >= 1E-6)
        scale = 20;
    else if (eps >= 1E-9)
        scale = 25;
    else if (eps >= 1E-12)
        scale = 25;

    double d = -std::log10(scale * eps);
    int i = d * 10 + 0.1 - 1;
    assert(i >= 0);
    assert(i < sizeof(cs) / sizeof(float));
    return cs[i];
}

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
sctl::Matrix<Real> init_random(int rows, int cols, Real left = 0.0, Real right = 1.0) {
    sctl::Matrix<Real> mat(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat(i, j) = left + (right - left) * drand48();
    return mat;
}

template <class Real, int N>
Real eval_horner(const std::array<Real, N> &coefs, Real x) {
    Real result = 0.0;
    for (int i = N - 1; i >= 0; --i)
        result = std::fma(result, x, coefs[i]);
    return result;
}

template <class Real, int N>
Real eval_poly_dumb(const std::array<Real, N> &coefs, Real x) {
    Real result = 0.0;
    Real xpow = 1.0;
    for (int i = 0; i < N; ++i) {
        result += coefs[i] * xpow;
        xpow *= x;
    }
    return result;
}

template <class T>
struct PSWFParams {
    const T bsize = 1.0;
    const T rsc = 2.0 / bsize;
    const T cen = -bsize / 2.0;
    const T d2max = bsize * bsize;
    const T thresh2 = 1E-30;
};

template <class Real, int digits>
void laplace_3d_pswf_direct(const sctl::Matrix<Real> &r_src, const sctl::Matrix<Real> &r_trg,
                            const sctl::Matrix<Real> &charge, sctl::Vector<Real> &u) {
    constexpr PSWFParams<Real> p;

    constexpr auto coefs = []() {
        if constexpr (digits <= 3)
            return Laplace3DLocalPSWF::get_coeffs<Real, 3>();
        else if constexpr (digits <= 6)
            return Laplace3DLocalPSWF::get_coeffs<Real, 6>();
        else if constexpr (digits <= 9)
            return Laplace3DLocalPSWF::get_coeffs<Real, 9>();
        else if constexpr (digits <= 12)
            return Laplace3DLocalPSWF::get_coeffs<Real, 12>();
        else
            throw std::runtime_error("Unsupported digits");
    }();

    const Real c0 = procl180_rescale(std::pow(10.0, -digits));
    dmk::Prolate0Fun prolate_fun(c0, 10000);
    const Real prolate_inf_inv = 1.0 / prolate_fun.int_eval(1.0);
    auto pswf = [&prolate_inf_inv, &prolate_fun](Real x) {
        return 1.0 - prolate_inf_inv * prolate_fun.int_eval(0.5 * (x + 1));
    };

    for (int i_src = 0; i_src < r_src.Dim(0); ++i_src) {
        for (int i_trg = 0; i_trg < r_trg.Dim(0); ++i_trg) {
            Real dr[3];
            Real dr2{0.0};
            for (int i = 0; i < n_dim; ++i) {
                dr[i] = r_trg(i_trg, i) - r_src(i_src, i);
                dr2 += dr[i] * dr[i];
            }
            if (dr2 < p.thresh2 || dr2 > p.d2max)
                continue;
            Real Rinv = 1.0 / std::sqrt(dr2);
            Real xtmp = std::fma(dr2, Rinv, p.cen) * p.rsc;
            Real ptmp = eval_horner<Real, coefs.size()>(coefs, xtmp) * Rinv;

            u[i_trg] += charge(0, i_src) * ptmp;
        }
    }
}

template <class Real>
void laplace_3d_pswf_direct_uKernel_cpu(const Opts &opts, const sctl::Matrix<Real> &r_src,
                                        const sctl::Matrix<Real> &r_trg, const sctl::Matrix<Real> &charge,
                                        sctl::Vector<Real> &u) {
    constexpr int nd = 1;
    constexpr PSWFParams<Real> p;
    const int ns = r_src.Dim(0);
    const int nt = r_trg.Dim(0);
    EvalLaplaceLocalPSWF<Real, 3>(&nd, &opts.digits, &p.rsc, &p.cen, &p.d2max, &r_src(0, 0), &ns, &charge(0, 0),
                                  &r_trg(0, 0), &nt, &u[0], &p.thresh2);
}

template <class Real>
void laplace_3d_pswf_direct_cpu(const Opts &opts, const sctl::Matrix<Real> &r_src, const sctl::Matrix<Real> &r_trg,
                                const sctl::Matrix<Real> &charge, sctl::Vector<Real> &u) {
    constexpr int nd = 1;
    constexpr PSWFParams<Real> p;
    const int ns = r_src.Dim(0);
    const int nt = r_trg.Dim(1);
    const Real *xtrg = &r_trg(0, 0);
    const Real *ytrg = &r_trg(1, 0);
    const Real *ztrg = &r_trg(2, 0);

    l3d_local_kernel_directcp_vec_cpp(&nd, &n_dim, &opts.digits, &p.rsc, &p.cen, &p.d2max, &r_src(0, 0), &ns,
                                      &charge(0, 0), xtrg, ytrg, ztrg, &nt, &u[0], &p.thresh2);
}

template <typename Real>
void run_comparison(const Opts &opts) {
    auto r_src_base = init_random<Real>(opts.n_src, n_dim);
    auto r_trg = init_random<Real>(opts.n_trg, n_dim);
    auto charges = init_random<Real>(1, opts.n_src, -1.0, 1.0);
    const int sample_offset = (opts.n_trg - 1);

    std::array<sctl::Matrix<Real>, 27> r_src_all;
    for (auto &r : r_src_all)
        r = r_src_base;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                const int image_idx = i * 9 + j * 3 + k;
                const Real dr[3] = {i - Real{1.0}, j - Real{1.0}, k - Real{1.0}};
                for (int s = 0; s < opts.n_src; ++s)
                    for (int l = 0; l < 3; ++l)
                        r_src_all[image_idx](s, l) += dr[l];
            }

    auto r_src_all_t = r_src_all;
    for (auto &r : r_src_all_t)
        r = r.Transpose();
    const auto r_trg_t = r_trg.Transpose();

    auto evaluator = [&opts]() {
        switch (opts.eval) {
        case DIRECT_CPU:
            return laplace_3d_pswf_direct_cpu<Real>;
            break;
        default:
            throw std::runtime_error("Unknown evaluator");
        }
    }();

    sctl::Vector<Real> res(opts.n_trg);
    const int n_pairs = opts.n_src * opts.n_trg * 27;
    const int n_warmup = 4e9 / n_pairs;
    ankerl::nanobench::Bench().batch(n_pairs).unit("pair").warmup(n_warmup).run("Full", [&] {
        for (int i = 0; i < 27; ++i)
            evaluator(opts, r_src_all[i], r_trg_t, charges, res);
        ankerl::nanobench::doNotOptimizeAway(res);
    });
    ankerl::nanobench::Bench().batch(n_pairs).unit("pair").warmup(n_warmup).run("uKernel", [&] {
        for (int i = 0; i < 27; ++i)
            laplace_3d_pswf_direct_uKernel_cpu(opts, r_src_all[i], r_trg, charges, res);
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
