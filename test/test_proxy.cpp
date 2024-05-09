#ifdef NDEBUG
#undef NDEBUG
#endif

#include <Eigen/Core>
#include <cassert>
#include <dmk/proxy.hpp>
#include <nanobench.h>
#include <string>

extern "C" {
void pdmk_charge2proxycharge_(int *ndim, int *nd, int *norder, int *ns, double *sources, double *charge, double *cen,
                              double *sc, double *coefs);
}

void test_fortran(int n_dim, int n_charge_dim, int n_order, int n_src) {
    Eigen::VectorX<double> r_src(n_src * n_dim);
    Eigen::VectorX<double> charge(n_src * n_charge_dim);
    Eigen::VectorX<double> coeffs(int(pow(n_order, n_dim)) * n_charge_dim);
    Eigen::VectorX<double> coeffs_fort(int(pow(n_order, n_dim)) * n_charge_dim);
    double center[3] = {0.5, 0.5, 0.5};
    double scale_factor = 1.0;

    for (int i = 0; i < n_src * n_dim; ++i)
        r_src[i] = drand48();

    for (int i = 0; i < n_src * n_charge_dim; ++i)
        charge[i] = drand48() - 0.5;

    auto b = ankerl::nanobench::Bench().unit("eval").title("c2pc" + std::to_string(n_dim)).minEpochIterations(20);
    b.run("c++" + std::to_string(n_order), [&] {
        coeffs.array() = 0.0;
        dmk::proxy::charge2proxycharge(n_dim, n_charge_dim, n_order, n_src, r_src.data(), charge.data(), center,
                                       scale_factor, coeffs.data());
    });

    b.run("fort" + std::to_string(n_order), [&] {
        coeffs_fort.array() = 0.0;
        pdmk_charge2proxycharge_(&n_dim, &n_charge_dim, &n_order, &n_src, r_src.data(), charge.data(), center,
                                 &scale_factor, coeffs_fort.data());
    });

    double l2 = (coeffs - coeffs_fort).norm() / coeffs.size();
    assert(l2 < 1E-16);
}

int main(int argc, char *argv[]) {
    int n_order = 32;
    int n_src = 500;
    int n_charge_dim = 1;

    int orders[] = {8, 12, 16, 20, 24, 32};
    for (auto &order : orders)
        test_fortran(2, n_charge_dim, order, n_src);
    for (auto &order : orders)
        test_fortran(3, n_charge_dim, order, n_src);

    return 0;
}
