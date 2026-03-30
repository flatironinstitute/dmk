#include <dmk.h>
#include <dmk/types.hpp>
#include <dmk/vector_kernels.hpp>
#include <sctl.hpp>

namespace dmk {
constexpr int unroll_factor = 3;

template <class Real, int MaxVecLen>
direct_evaluator_func<Real> get_laplace_2d_kernel(dmk_pgh eval_level, int n_digits) {
    auto make = [eval_level]<int N_DIGITS, int N_COEFFS>(const Real(&c)[N_COEFFS]) -> direct_evaluator_func<Real> {
        std::array<Real, N_COEFFS> coeffs;
        std::copy_n(c, N_COEFFS, coeffs.data());
        return [coeffs, eval_level](Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,
                                    const Real *charge, int n_trg, const Real *r_trg, Real *pot) {
            laplace_2d_poly_all_pairs<Real, MaxVecLen, N_DIGITS, N_COEFFS>(
                eval_level, N_DIGITS, rsc, cen, d2max, thresh2, N_COEFFS, coeffs.data(), n_src, r_src, charge, n_trg,
                r_trg, pot, unroll_factor);
        };
    };

    if (n_digits <= 3)
        return make.template operator()<3>({3.293312412035785e-01, -4.329140084314137e-01, 1.366683635926240e-01,
                                            -4.309918126794055e-02, 1.041106682948322e-02});
    if (n_digits <= 6)
        return make.template operator()<6>({3.449851438836016e-01, -4.902921365061905e-01, 2.220880572548949e-01,
                                            -1.153716526684871e-01, 5.535102319921498e-02, -2.281631998557134e-02,
                                            7.843017349311455e-03, -2.269922867123751e-03, 6.058276012390756e-04,
                                            -1.231943198424746e-04});
    if (n_digits <= 9)
        return make.template operator()<9>({3.464285661408188e-01, -4.987507216024493e-01, 2.448626056577886e-01,
                                            -1.531215433543293e-01, 9.898265049486202e-02, -6.064807566030946e-02,
                                            3.368003719471507e-02, -1.657913779091916e-02, 7.167896958658711e-03,
                                            -2.718119907588121e-03, 9.031254984609993e-04, -2.605021223678894e-04,
                                            6.764515456758602e-05, -1.806660493741674e-05, 3.640268744220521e-06});
    if (n_digits <= 12)
        return make.template operator()<12>(
            {3.465597422118963e-01, -4.998454957461734e-01, 2.491697186708244e-01, -1.637913095741979e-01,
             1.177474435136638e-01, -8.570242499810476e-02, 6.020769882062124e-02, -3.955364604811165e-02,
             2.382498816767467e-02, -1.301290744915523e-02, 6.410015506457686e-03, -2.841967252293358e-03,
             1.135602935662887e-03, -4.109780616023590e-04, 1.339086505665511e-04, -3.822398669202901e-05,
             1.037153217818392e-05, -3.251884408687046e-06, 7.149918161513096e-07});

    throw std::runtime_error("Unsupported n_digits: " + std::to_string(n_digits));
}

template <class Real, int MaxVecLen>
direct_evaluator_func<Real> get_laplace_3d_kernel(dmk_pgh eval_level, int n_digits) {
    auto make = [eval_level]<int ND, int NC>(const Real(&c)[NC]) -> direct_evaluator_func<Real> {
        std::array<Real, NC> coeffs;
        std::copy_n(c, NC, coeffs.data());
        return [coeffs, eval_level](Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,
                                    const Real *charge, int n_trg, const Real *r_trg, Real *pot) {
            laplace_3d_poly_all_pairs<Real, MaxVecLen, ND, NC>(eval_level, ND, rsc, cen, d2max, thresh2, NC,
                                                               coeffs.data(), n_src, r_src, charge, n_trg, r_trg, pot,
                                                               unroll_factor);
        };
    };

    if (n_digits <= 3)
        return make.template operator()<3>({1.627823522210361e-01, -4.553645597616490e-01, 4.171687104204163e-01,
                                            -7.073638602709915e-02, -8.957845614474928e-02, 2.617986644718201e-02,
                                            9.633427876507601e-03});
    if (n_digits <= 6)
        return make.template operator()<6>({5.482525801351582e-02, -2.616592110444692e-01, 4.862652666337138e-01,
                                            -3.894296348642919e-01, 1.638587821812791e-02, 1.870328434198821e-01,
                                            -8.714171086568978e-02, -3.927020727017803e-02, 3.728187607052319e-02,
                                            3.153734425831139e-03, -8.651313377285847e-03, 1.725110090795567e-04,
                                            1.034762385284044e-03});
    if (n_digits <= 9)
        return make.template operator()<9>(
            {1.835718730962269e-02, -1.258015846164503e-01, 3.609487248584408e-01, -5.314579651112283e-01,
             3.447559412892380e-01, 9.664692318551721e-02, -3.124274531849053e-01, 1.322460720579388e-01,
             9.773007866584822e-02, -1.021958831082768e-01, -3.812847450976566e-03, 3.858117355875043e-02,
             -8.728545924521301e-03, -9.401196355382909e-03, 4.024549377076924e-03, 1.512806105865091e-03,
             -9.576734877247042e-04, -1.303457547418901e-04, 1.100385844683190e-04});
    if (n_digits <= 12)
        return make.template operator()<12>(
            {6.262472576363448e-03,  -5.605742936112479e-02, 2.185890864792949e-01,  -4.717350304955679e-01,
             5.669680214206270e-01,  -2.511606878849214e-01, -2.744523658778361e-01, 4.582527599363415e-01,
             -1.397724810121539e-01, -2.131762135835757e-01, 1.995489373508990e-01,  1.793390341864239e-02,
             -1.035055132403432e-01, 3.035606831075176e-02,  3.153931762550532e-02,  -2.033178627450288e-02,
             -5.406682731236552e-03, 7.543645573618463e-03,  1.437788047407851e-05,  -1.928370882351732e-03,
             2.891658777328665e-04,  3.332996162099811e-04,  -8.397699195938912e-05, -3.015837377517983e-05,
             9.640642701924662e-06});
    throw std::runtime_error("Unsupported n_digits: " + std::to_string(n_digits));
}

template <class Real, int MaxVecLen>
direct_evaluator_func<Real> get_sqrt_laplace_2d_kernel(dmk_pgh eval_level, int n_digits) {
    auto make = [eval_level]<int ND, int NC>(const Real(&c)[NC]) -> direct_evaluator_func<Real> {
        std::array<Real, NC> coeffs;
        std::copy_n(c, NC, coeffs.data());
        return [coeffs, eval_level](Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,
                                    const Real *charge, int n_trg, const Real *r_trg, Real *pot) {
            sqrt_laplace_2d_poly_all_pairs<Real, MaxVecLen, ND, NC>(eval_level, ND, rsc, cen, d2max, thresh2, NC,
                                                                    coeffs.data(), n_src, r_src, charge, n_trg, r_trg,
                                                                    pot, unroll_factor);
        };
    };

    if (n_digits <= 3)
        return make.template operator()<3>({1.627823522210361e-01, -4.553645597616490e-01, 4.171687104204163e-01,
                                            -7.073638602709915e-02, -8.957845614474928e-02, 2.617986644718201e-02,
                                            9.633427876507601e-03});
    if (n_digits <= 6)
        return make.template operator()<6>({5.482525801351582e-02, -2.616592110444692e-01, 4.862652666337138e-01,
                                            -3.894296348642919e-01, 1.638587821812791e-02, 1.870328434198821e-01,
                                            -8.714171086568978e-02, -3.927020727017803e-02, 3.728187607052319e-02,
                                            3.153734425831139e-03, -8.651313377285847e-03, 1.725110090795567e-04,
                                            1.034762385284044e-03});
    if (n_digits <= 9)
        return make.template operator()<9>(
            {1.835718730962269e-02, -1.258015846164503e-01, 3.609487248584408e-01, -5.314579651112283e-01,
             3.447559412892380e-01, 9.664692318551721e-02, -3.124274531849053e-01, 1.322460720579388e-01,
             9.773007866584822e-02, -1.021958831082768e-01, -3.812847450976566e-03, 3.858117355875043e-02,
             -8.728545924521301e-03, -9.401196355382909e-03, 4.024549377076924e-03, 1.512806105865091e-03,
             -9.576734877247042e-04, -1.303457547418901e-04, 1.100385844683190e-04});
    if (n_digits <= 12)
        return make.template operator()<12>(
            {6.262472576363448e-03,  -5.605742936112479e-02, 2.185890864792949e-01,  -4.717350304955679e-01,
             5.669680214206270e-01,  -2.511606878849214e-01, -2.744523658778361e-01, 4.582527599363415e-01,
             -1.397724810121539e-01, -2.131762135835757e-01, 1.995489373508990e-01,  1.793390341864239e-02,
             -1.035055132403432e-01, 3.035606831075176e-02,  3.153931762550532e-02,  -2.033178627450288e-02,
             -5.406682731236552e-03, 7.543645573618463e-03,  1.437788047407851e-05,  -1.928370882351732e-03,
             2.891658777328665e-04,  3.332996162099811e-04,  -8.397699195938912e-05, -3.015837377517983e-05,
             9.640642701924662e-06});
    throw std::runtime_error("Unsupported n_digits: " + std::to_string(n_digits));
}

template <class Real, int MaxVecLen>
direct_evaluator_func<Real> get_sqrt_laplace_3d_kernel(dmk_pgh eval_level, int n_digits) {
    auto make = [eval_level]<int ND, int NC>(const Real(&c)[NC]) -> direct_evaluator_func<Real> {
        std::array<Real, NC> coeffs;
        std::copy_n(c, NC, coeffs.data());
        return [coeffs, eval_level](Real rsc, Real cen, Real d2max, Real thresh2, int n_src, const Real *r_src,
                                    const Real *charge, int n_trg, const Real *r_trg, Real *pot) {
            sqrt_laplace_3d_poly_all_pairs<Real, MaxVecLen, ND, NC>(eval_level, ND, rsc, cen, d2max, thresh2, NC,
                                                                    coeffs.data(), n_src, r_src, charge, n_trg, r_trg,
                                                                    pot, unroll_factor);
        };
    };

    if (n_digits <= 3)
        return make.template operator()<3>({1.072550277328770e-01, -2.940116755817858e-01, 3.195680735052503e-01,
                                            -1.885147776001495e-01, 7.308229981701020e-02, -1.746740887691195e-02});
    if (n_digits <= 6)
        return make.template operator()<6>({1.614709231716746e-02, -8.104482562490173e-02, 1.821202399011004e-01,
                                            -2.445665225927775e-01, 2.214153542697690e-01, -1.447686339587813e-01,
                                            7.139964303548979e-02, -2.725881175495876e-02, 8.433287237298649e-03,
                                            -2.361123374269844e-03, 4.843770794874525e-04});
    if (n_digits <= 9)
        return make.template operator()<9>({2.086402113115865e-03, -1.562089192993565e-02, 5.445041674459949e-02,
                                            -1.178541200828409e-01, 1.783129998458763e-01, -2.013700177956373e-01,
                                            1.770491043290802e-01, -1.248556216953823e-01, 7.222027766315242e-02,
                                            -3.487303987632228e-02, 1.426369137364099e-02, -5.005766772662122e-03,
                                            1.519637697654493e-03, -3.983110653078808e-04, 9.342903238021878e-05,
                                            -2.223077176699562e-05, 4.041199749483288e-06});
    if (n_digits <= 12)
        return make.template operator()<12>(
            {2.585256922145451e-04, -2.586987362995730e-03, 1.228229637064665e-02, -3.689309545219710e-02,
             7.889646803121930e-02, -1.281810865332097e-01, 1.648961579104568e-01, -1.728791019337961e-01,
             1.509077982778982e-01, -1.115149888411611e-01, 7.069833292226477e-02, -3.888103784086078e-02,
             1.872315725194051e-02, -7.957995064673665e-03, 3.005989787130355e-03, -1.015826376419250e-03,
             3.094089416842241e-04, -8.493261780963615e-05, 2.066398691692728e-05, -4.722966945758245e-06,
             1.200827608904831e-06, -2.250099297995689e-07});
    throw std::runtime_error("Unsupported n_digits: " + std::to_string(n_digits));
}

template direct_evaluator_func<float>
get_sqrt_laplace_2d_kernel<float, sctl::DefaultVecLen<float>()>(dmk_pgh eval_level, int n_digits);
template direct_evaluator_func<float> get_laplace_2d_kernel<float, sctl::DefaultVecLen<float>()>(dmk_pgh eval_level,
                                                                                                 int n_digits);
template direct_evaluator_func<float>
get_sqrt_laplace_3d_kernel<float, sctl::DefaultVecLen<float>()>(dmk_pgh eval_level, int n_digits);
template direct_evaluator_func<float> get_laplace_3d_kernel<float, sctl::DefaultVecLen<float>()>(dmk_pgh eval_level,
                                                                                                 int n_digits);

template direct_evaluator_func<double>
get_sqrt_laplace_2d_kernel<double, sctl::DefaultVecLen<double>()>(dmk_pgh eval_level, int n_digits);
template direct_evaluator_func<double> get_laplace_2d_kernel<double, sctl::DefaultVecLen<double>()>(dmk_pgh eval_level,
                                                                                                    int n_digits);
template direct_evaluator_func<double>
get_sqrt_laplace_3d_kernel<double, sctl::DefaultVecLen<double>()>(dmk_pgh eval_level, int n_digits);
template direct_evaluator_func<double> get_laplace_3d_kernel<double, sctl::DefaultVecLen<double>()>(dmk_pgh eval_level,
                                                                                                    int n_digits);

} // namespace dmk
