#include <algorithm>
#include <cstdlib>
#include <dmk/vector_kernels.hpp>
#include <filesystem>
#include <format>
#include <limits>
#include <nanobench.h>
#include <polyfit/fast_eval.hpp>

#include <stdexcept>

#include <dmk/prolate0_eval.hpp>

#include <rufus.hpp>

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
        while ((opt = getopt(argc, argv, "N:M:d:t:e:u:h?")) != -1) {
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

auto eval_horner(const auto &coefs, typename std::decay_t<decltype(coefs)>::value_type x) {
    using Real = typename std::decay_t<decltype(coefs)>::value_type;
    Real result = 0.0;
    int N = coefs.size();

    for (int i = N - 1; i >= 0; --i)
        result = std::fma(result, x, coefs[i]);

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

template <class Vec>
auto to_vector(const auto &arr) {
    Vec vec(arr.size());
    std::copy_n(arr.data(), arr.size(), &vec[0]);
    return vec;
}

template <class Real>
sctl::Vector<Real> make_polyfit_abs_error(Real tol) {
    const Real c0 = procl180_rescale(tol);
    dmk::Prolate0Fun prolate_fun(c0, 10000);
    const Real prolate_inf_inv = 1.0 / prolate_fun.int_eval(1.0);
    auto pswf = [&prolate_inf_inv, &prolate_fun](Real x) {
        return Real(1.0 - prolate_inf_inv * prolate_fun.int_eval((x + 1) / 2.0));
    };

    for (int n_coeffs = 5; n_coeffs < 32; ++n_coeffs) {
        try {
            auto prolate_int_fun = poly_eval::make_func_eval(pswf, n_coeffs, -1.0, 1.0);

            bool passed = true;
            for (double x = -1.0; x <= 1.0; x += 0.01) {
                const Real fit = prolate_int_fun(x);
                const Real act = pswf(x);
                const Real abs_err = std::abs(act - fit);
                if (abs_err > tol) {
                    passed = false;
                    continue;
                }
            }
            if (passed) {
                std::cout << "Achieved tol " << tol << " with n_coeffs = " << n_coeffs << std::endl;
                auto coeffs = to_vector<sctl::Vector<Real>>(prolate_int_fun.coeffs());
                return coeffs;
            }
        } catch (std::exception &e) {
            std::cout << "Failed to fit with n_coeffs = " << n_coeffs << "\n";
            std::cout << e.what() << std::endl;
        }
    }
    return {};
}

template <class Real, int digits>
void laplace_3d_pswf_direct(const sctl::Vector<Real> &r_src, const sctl::Vector<Real> &r_trg,
                            const sctl::Vector<Real> &charge, sctl::Vector<Real> &u) {
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
void laplace_3d_pswf_direct_uKernel_cpu(const Opts &opts, const sctl::Vector<Real> &r_src,
                                        const sctl::Vector<Real> &r_trg, const sctl::Vector<Real> &charge,
                                        sctl::Vector<Real> &u) {
    constexpr int nd = 1;
    constexpr PSWFParams<Real> p;
    const int ns = r_src.Dim() / 3;
    const int nt = r_trg.Dim() / 3;
    EvalLaplaceLocalPSWF<Real, 3>(&nd, &opts.digits, &p.rsc, &p.cen, &p.d2max, &r_src[0], &ns, &charge[0], &r_trg[0],
                                  &nt, &u[0], &p.thresh2);
}

template <class Real>
void laplace_3d_coeffs_direct_uKernel_cpu(const Opts &opts, const sctl::Vector<Real> &coefs,
                                          const sctl::Vector<Real> &r_src, const sctl::Vector<Real> &r_trg,
                                          const sctl::Vector<Real> &charge, sctl::Vector<Real> &u) {
    constexpr int nd = 1;
    constexpr PSWFParams<Real> p;
    const int ns = r_src.Dim() / 3;
    const int nt = r_trg.Dim() / 3;
    EvalLaplaceLocalUnknownCoeffs<Real>(nd, opts.digits, p.rsc, p.cen, p.d2max, p.thresh2, coefs, r_src, charge, r_trg,
                                        u);
}

template <class Real>
void laplace_3d_pswf_direct_cpu(const Opts &opts, const sctl::Vector<Real> &r_src, const sctl::Vector<Real> &r_trg,
                                const sctl::Vector<Real> &charge, sctl::Vector<Real> &u) {
    constexpr int nd = 1;
    constexpr PSWFParams<Real> p;
    const int ns = r_src.Dim() / 3;
    const int nt = r_trg.Dim() / 3;
    const Real *xtrg = &r_trg[0 * nt];
    const Real *ytrg = &r_trg[1 * nt];
    const Real *ztrg = &r_trg[2 * nt];

    l3d_local_kernel_directcp_vec_cpp(&nd, &n_dim, &opts.digits, &p.rsc, &p.cen, &p.d2max, &r_src[0], &ns, &charge[0],
                                      xtrg, ytrg, ztrg, &nt, &u[0], &p.thresh2);
}

template <typename Real, sctl::Integer MaxVecLen = sctl::DefaultVecLen<Real>()>
void eval_points_sctl_vec(const Real *__restrict__ pts, Real *__restrict__ res, int n_pts, const auto &coeffs) {
    using VecType = sctl::Vec<Real, MaxVecLen>;
    constexpr int unroll_factor = 2;
    int i = 0;
    int n_vec = n_pts / (MaxVecLen * unroll_factor) * (unroll_factor * MaxVecLen);
    for (; i < n_vec; i += unroll_factor * MaxVecLen) {
        VecType x[unroll_factor];
        VecType y[unroll_factor];

        for (int j = 0; j < unroll_factor; ++j) {
            x[j] = VecType::LoadAligned(&pts[i + j * MaxVecLen]);
            y[j] = sctl::EvalPolynomial(x[j].get(), coeffs);
        }
        for (int j = 0; j < unroll_factor; ++j)
            y[j].StoreAligned(&res[i + j * MaxVecLen]);
    }
    for (; i < n_pts; ++i)
        res[i] = eval_horner(coeffs, pts[i]);
}

template <typename Real>
void compare_horner_polyfit_sctl() {
    const int n_coeffs = 15;
    const double eps = 1e-3;
    const Real c0 = procl180_rescale(eps);
    dmk::Prolate0Fun prolate_fun(c0, 10000);
    const Real prolate_inf_inv = 1.0 / prolate_fun.int_eval(1.0);
    auto pswf = [&prolate_inf_inv, &prolate_fun](Real x) {
        return Real(1.0 - prolate_inf_inv * prolate_fun.int_eval((x + 1) / 2.0));
    };

    auto pswf_fit = poly_eval::make_func_eval<n_coeffs>(pswf, -1.0, 1.0);

    int n_pts = 8001;
    auto pts = init_random<Real>(n_pts, 1, -1.0, 1.0);
    auto res = sctl::Vector<Real>(n_pts);

    pswf_fit(&pts(0, 0), &res[0], n_pts);
    auto coeffs = pswf_fit.coeffs();
    std::reverse(coeffs.begin(), coeffs.end());

    std::cout << res[n_pts - 1] << std::endl;
    eval_points_sctl_vec(&pts(0, 0), &res[0], n_pts, coeffs);
    std::cout << res[n_pts - 1] << std::endl;

    ankerl::nanobench::Bench().batch(n_pts).unit("pts").warmup(1000).run("sctl", [&] {
        eval_points_sctl_vec(&pts(0, 0), &res[0], n_pts, coeffs);
        ankerl::nanobench::doNotOptimizeAway(res);
    });

    ankerl::nanobench::Bench().batch(n_pts).unit("pts").run("pf_0_0", [&] {
        pswf_fit.template operator()(&pts(0, 0), &res[0], n_pts);
        ankerl::nanobench::doNotOptimizeAway(res);
    });
}

template <typename Real>
void run_comparison(const Opts &opts) {
    auto r_src_base = init_random<Real>(opts.n_src, n_dim);
    auto r_trg = init_random<Real>(opts.n_trg, n_dim);
    auto charges = init_random<Real>(1, opts.n_src, -1.0, 1.0);
    const int sample_offset = (opts.n_trg - 1);

    std::array<sctl::Vector<Real>, 27> r_src_all;
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

    auto r_trg_t = r_trg;
    for (int i = 0; i < opts.n_src; ++i)
        for (int j = 0; j < n_dim; ++j)
            r_trg_t[j * opts.n_src + i] = r_trg[i * n_dim + j];

    auto evaluator = [&opts]() {
        switch (opts.eval) {
        case DIRECT_CPU:
            return laplace_3d_pswf_direct_cpu<Real>;
            break;
        default:
            throw std::runtime_error("Unknown evaluator");
        }
    }();
    auto coeffs = make_polyfit_abs_error<Real>(std::pow(10.0, -opts.digits));

    RuFuS RS;
    auto dir = std::filesystem::read_symlink("/proc/self/exe").parent_path();
    RS.load_ir_file(dir / "jit_kernels.ll");

    constexpr PSWFParams<Real> p;
    sctl::Vector<Real> res(opts.n_trg);

    constexpr int V = std::is_same_v<Real, float> ? 16 : 8;
    constexpr auto T = std::is_same_v<Real, float> ? "float" : "double";
    const auto func_name = std::format("void laplace_pswf_all_pairs_jit<{}, {}>(int, int, {}, {}, {}, {}, "
                                       "int, {} const*, int, {} const*, {} const*, int, {} const*, {}*, int)",
                                       T, V, T, T, T, T, T, T, T, T, T);

    auto jit_n_coeffs_n_digits = RS.compile<void (*)(int, Real, Real, Real, Real, const Real *, int, const Real *,
                                                     const Real *, int, const Real *, Real *)>(
        func_name, {{"n_coeffs", coeffs.Dim()}, {"n_digits", opts.digits}, {"unroll_factor", opts.unroll_factor}});

    auto jit_n_coeffs = RS.compile<void (*)(int, int, Real, Real, Real, Real, const Real *, int, const Real *,
                                            const Real *, int, const Real *, Real *)>(
        func_name, {{"n_coeffs", coeffs.Dim()}, {"unroll_factor", opts.unroll_factor}});

    auto jit_func_unspecialized =
        RS.compile<void (*)(int, int, Real, Real, Real, Real, int, const Real *, int, const Real *, const Real *, int,
                            const Real *, Real *)>(func_name, {{"unroll_factor", opts.unroll_factor}});

    if (!jit_n_coeffs_n_digits)
        return;

    {
        res = 0;
        jit_n_coeffs_n_digits(1, p.rsc, p.cen, p.d2max, p.thresh2, &coeffs[0], opts.n_src, &r_src_base[0], &charges[0],
                              opts.n_trg, &r_trg[0], &res[0]);
        for (int j = 0; j < 16; ++j)
            std::cout << res[j] << " ";
        std::cout << std::endl;

        res = 0;
        jit_n_coeffs(1, opts.digits, p.rsc, p.cen, p.d2max, p.thresh2, &coeffs[0], opts.n_src, &r_src_base[0],
                     &charges[0], 16, &r_trg[0], &res[0]);
        for (int j = 0; j < 16; ++j)
            std::cout << res[j] << " ";
        std::cout << std::endl;

        res = 0;
        jit_func_unspecialized(1, opts.digits, p.rsc, p.cen, p.d2max, p.thresh2, coeffs.Dim(), &coeffs[0], opts.n_src,
                               &r_src_base[0], &charges[0], opts.n_trg, &r_trg[0], &res[0]);
        for (int j = 0; j < 16; ++j)
            std::cout << res[j] << " ";
        std::cout << std::endl;

        res = 0;
        evaluator(opts, r_src_base, r_trg_t, charges, res);
        for (int j = 0; j < 16; ++j)
            std::cout << res[j] << " ";
        std::cout << std::endl;

        res = 0;
        laplace_3d_pswf_direct_uKernel_cpu(opts, r_src_base, r_trg_t, charges, res);
        for (int j = 0; j < 16; ++j)
            std::cout << res[j] << " ";
        std::cout << std::endl;

        res = 0;
        auto test = res;
        laplace_3d_coeffs_direct_uKernel_cpu(opts, coeffs, r_src_base, r_trg, charges, test);
        jit_n_coeffs_n_digits(1, p.rsc, p.cen, p.d2max, p.thresh2, &coeffs[0], opts.n_src, &r_src_base[0], &charges[0],
                              opts.n_trg, &r_trg[0], &res[0]);
        for (int j = 0; j < 16; ++j)
            std::cout << test[j] << " ";
        std::cout << std::endl;
        for (int i = 0; i < res.Dim(); ++i)
            if (test[i] && std::abs(1.0 - res[i] / test[i]) > 5 * std::numeric_limits<Real>::epsilon())
                std::cout << "OOF: " << i << " " << res[i] << " " << test[i] << "\n";
    }

    const int n_pairs = opts.n_src * opts.n_trg * 27;
    const int n_warmup = 4e9 / n_pairs;
    {
        int i = 13;
        auto res_jit = res;
        jit_n_coeffs_n_digits(1, p.rsc, p.cen, p.d2max, p.thresh2, &coeffs[0], opts.n_src, &r_src_all[i][0],
                              &charges[0], opts.n_trg, &r_trg[0], &res_jit[0]);
        auto res_evaluator = res;
        evaluator(opts, r_src_all[i], r_trg_t, charges, res_evaluator);
        std::cout << res_jit[sample_offset] << " " << res_evaluator[sample_offset] << std::endl;
    }
    ankerl::nanobench::Bench().batch(n_pairs).unit("pair").warmup(n_warmup).run("Full", [&] {
        for (int i = 0; i < 27; ++i)
            evaluator(opts, r_src_all[i], r_trg_t, charges, res);
        ankerl::nanobench::doNotOptimizeAway(res);
    });
    ankerl::nanobench::Bench().batch(n_pairs).unit("pair").run("uKernel", [&] {
        for (int i = 0; i < 27; ++i)
            laplace_3d_pswf_direct_uKernel_cpu(opts, r_src_all[i], r_trg, charges, res);
        ankerl::nanobench::doNotOptimizeAway(res);
    });
    ankerl::nanobench::Bench().batch(n_pairs).unit("pair").run("uKernelCoeff", [&] {
        for (int i = 0; i < 27; ++i)
            laplace_3d_coeffs_direct_uKernel_cpu(opts, coeffs, r_src_all[i], r_trg, charges, res);
        ankerl::nanobench::doNotOptimizeAway(res);
    });
    ankerl::nanobench::Bench().batch(n_pairs).unit("pair").run("uKernelJitNCoeffsDigits", [&] {
        for (int i = 0; i < 27; ++i)
            jit_n_coeffs_n_digits(1, p.rsc, p.cen, p.d2max, p.thresh2, &coeffs[0], opts.n_src, &r_src_all[i][0],
                                  &charges[0], opts.n_trg, &r_trg[0], &res[0]);
        ankerl::nanobench::doNotOptimizeAway(res);
    });
    ankerl::nanobench::Bench().batch(n_pairs).unit("pair").run("uKernelJitNCoeffs", [&] {
        for (int i = 0; i < 27; ++i)
            jit_n_coeffs(1, opts.digits, p.rsc, p.cen, p.d2max, p.thresh2, &coeffs[0], opts.n_src, &r_src_all[i][0],
                         &charges[0], opts.n_trg, &r_trg[0], &res[0]);
        ankerl::nanobench::doNotOptimizeAway(res);
    });
    ankerl::nanobench::Bench().batch(n_pairs).unit("pair").run("uKernelJitNone", [&] {
        for (int i = 0; i < 27; ++i)
            jit_func_unspecialized(1, opts.digits, p.rsc, p.cen, p.d2max, p.thresh2, coeffs.Dim(), &coeffs[0],
                                   opts.n_src, &r_src_all[i][0], &charges[0], opts.n_trg, &r_trg[0], &res[0]);
        ankerl::nanobench::doNotOptimizeAway(res);
    });
}

int main(int argc, char *argv[]) {
    try {
        Opts opts(argc, argv);

        if (opts.prec == 'f')
            // compare_horner_polyfit_sctl<float>();
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
