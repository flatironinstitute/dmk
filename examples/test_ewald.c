/* Test: running pme_poisson3d_lagrange() in a C file by including the header. */

#include <stdio.h>
#include <stdbool.h>
#include <ewald_total.h>

int main(void) {
    const int n_src = 10;
    const int n_dim = 3;
    const double L = 1.0;
    const double alpha = 10.0;
    const double r_cut = 0.2;
    const int N = 32;
    const int p = 4;

    // custom coordinates for small tests
    double r_src[] = {0.131538, 0.45865, 0.218959, 0.678865, 0.934693, 0.519416, 0.0345721, 0.5297, 0.00769819, 0.0668422, 0.686773, 0.930436, 0.526929, 0.653919, 0.701191, 0.762198, 0.0474645, 0.328234, 0.75641, 0.365339, 0.98255, 0.753356, 0.0726859, 0.884707, 0.436411, 0.477732, 0.274907, 0.166507, 0.897656, 0.0605643};
    double charges[] = {0.196104 , -0.174876 ,  0.175012 , -0.631476 , -0.665444 , -0.0446574,  1.01469  ,  0.11595  , -0.712774 ,  0.727467};
    double pot[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    // compute the potential
    pme_poisson3d_lagrange(pot, n_src, n_dim, L, alpha, r_cut, N, p, true, false, r_src, charges);

    // print the potential
    for (int i = 0; i < n_src; ++i) {
        printf("%0.3f ", pot[i]);
    }
    printf("\n");

    return 0;
}