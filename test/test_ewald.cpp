/* Test: running pme_poisson3d_lagrange() in a cpp file by including the header. 
Running the function through a C API */

#include <dmk/ewald.h>

int main(int argc, char *argv[]) {
    return test_pme_poisson3d_lagrange(argc, argv);
}