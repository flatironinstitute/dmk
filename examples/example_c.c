#include <dmk.h>

#include <mpi.h>
#include <stdlib.h>

static double rand_unit(void) { return (double)rand() / (double)RAND_MAX; }

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    const int n_src = 1000;
    double *r_src = malloc(sizeof(double) * 3 * n_src);
    double *charge = malloc(sizeof(double) * 3 * n_src);
    double *vel = malloc(sizeof(double) * 3 * n_src);

    for (int i = 0; i < 3 * n_src; ++i) {
        r_src[i] = rand_unit();
        charge[i] = 2.0 * rand_unit() - 1.0;
    }

    pdmk_params params;
    pdmk_init_default_params(&params);
    params.n_dim = 3;
    params.kernel = DMK_STOKESLET;
    params.eval_src = DMK_VELOCITY;
    params.eval_trg = DMK_VELOCITY;

    pdmk_tree tree = pdmk_tree_create(MPI_COMM_WORLD, &params, n_src, r_src, charge, NULL, 0, NULL);
    pdmk_tree_eval(tree, vel, NULL);

    for (int i = 0; i < 3 * n_src; ++i)
        charge[i] = 2.0 * rand_unit() - 1.0;
    pdmk_tree_update_charges(tree, charge, NULL);
    pdmk_tree_eval(tree, vel, NULL);

    pdmk_tree_destroy(tree);
    free(r_src);
    free(charge);
    free(vel);
    MPI_Finalize();
    return 0;
}
