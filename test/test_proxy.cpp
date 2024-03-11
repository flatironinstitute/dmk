#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdlib>
#include <dmk/proxy.hpp>
#include <iostream>
#include <omp.h>

#include <Eigen/Core>

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

    double dt = -omp_get_wtime();
    int n_iter = 10000;
    for (int i_iter = 0; i_iter < n_iter; ++i_iter)
        dmk::proxy::charge2proxycharge(n_dim, n_charge_dim, n_order, n_src, r_src.data(), charge.data(), center,
                                       scale_factor, coeffs.data());
    dt += omp_get_wtime();
    std::cout << dt / n_iter << std::endl;

    dt = -omp_get_wtime();
    for (int i_iter = 0; i_iter < n_iter; ++i_iter) {
        coeffs_fort.array() = 0.0;
        pdmk_charge2proxycharge_(&n_dim, &n_charge_dim, &n_order, &n_src, r_src.data(), charge.data(), center,
                                 &scale_factor, coeffs_fort.data());
    }
    dt += omp_get_wtime();
    std::cout << dt / n_iter << std::endl;

    double l2 = (coeffs - coeffs_fort).norm() / coeffs.size();
    assert(l2 < 1E-16);
}

int main(int argc, char *argv[]) {
    int n_order = 18;
    int n_src = 80;
    int n_charge_dim = 1;

    test_fortran(2, n_charge_dim, n_order, n_src);
    test_fortran(3, n_charge_dim, n_order, n_src);

    return 0;
}
