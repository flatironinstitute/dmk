/* Header file to include in another C test file, in order to run PME. */

#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void pme_poisson3d_lagrange(
        double       *pot,
        int           n_sources,
        int           n_dim,
        double        length,
        double        alpha,
        double        r_cut,
        int           N,
        int           P,
        int           uniform,        /* pass 0 or 1 from C */
        int           vectorized,     /* pass 0 or 1 from C */
        double *r_sources,
        double *charges);

#ifdef __cplusplus
}
#endif
