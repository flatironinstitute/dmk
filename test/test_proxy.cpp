#include <dmk/proxy.hpp>
#include <omp.h>
#include <cstdlib>

int main(int argc, char *argv[]) {
    int n_order = 24;
    int n_src = 4000;
    int n_charge_dim = 3;
    int n_dim = 2;

    std::vector<double> r_src(n_src * n_dim);
    std::vector<double> charge(n_src * n_charge_dim);
    std::vector<double> coeffs(n_order * n_order * n_src * n_charge_dim);
    double center[2] = {0.5, 0.5};
    double scale_factor = 1.0;

    for (int i = 0; i < n_src * n_dim; ++i)
        r_src[i] = drand48();

    for (int i = 0; i < n_src * n_charge_dim; ++i)
        charge[i] = drand48() - 0.5;

    int n_iters = 1e7 / n_src;
    for (int i = 0; i < n_iters; ++i)
        dmk::proxy::charge2proxycharge(n_dim, n_charge_dim, n_order, r_src, charge, center, scale_factor, coeffs);

    return 0;
}
