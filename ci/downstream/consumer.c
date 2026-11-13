// Smoke test for the installed package: that <dmk.h> resolves, that the exported target links,
// and that a solve through the installed library agrees with its own direct sum. Squared norms are
// compared so the check needs no libm.
#include <dmk.h>
#include <dmk/version.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_SRC 400
#define N_TRG 100

// A fixed LCG rather than rand(), so a failure here is the library's and reproduces anywhere.
static unsigned long lcg_state = 12345UL;
static double lcg_unit(void) {
    lcg_state = lcg_state * 6364136223846793005UL + 1442695040888963407UL;
    return (double)((lcg_state >> 11) & 0xFFFFFFFFUL) / (double)0x100000000UL;
}

int main(void) {
    if (strcmp(pdmk_version_string(), DMK_VERSION_STRING) != 0) {
        fprintf(stderr, "library reports %s, header says %s\n", pdmk_version_string(), DMK_VERSION_STRING);
        return 1;
    }

    static double r_src[3 * N_SRC], charge[N_SRC], r_trg[3 * N_TRG];
    static double pot_tree[N_TRG], pot_direct[N_TRG], pot_src[N_SRC];
    for (int i = 0; i < 3 * N_SRC; ++i)
        r_src[i] = 0.01 + 0.98 * lcg_unit();
    for (int i = 0; i < N_SRC; ++i)
        charge[i] = lcg_unit() - 0.5;
    for (int i = 0; i < 3 * N_TRG; ++i)
        r_trg[i] = 0.01 + 0.98 * lcg_unit();

    pdmk_params params;
    pdmk_init_default_params(&params);
    params.n_dim = 3;
    params.kernel = DMK_LAPLACE;
    params.eps = 1e-6;
    params.eval_src = DMK_POTENTIAL;
    params.eval_trg = DMK_POTENTIAL;

    dmk_error err = pdmk_direct(NULL, params, N_SRC, r_src, charge, NULL, N_TRG, r_trg, pot_src, pot_direct);
    if (err != DMK_SUCCESS) {
        fprintf(stderr, "pdmk_direct failed (%d): %s\n", (int)err, pdmk_last_error_message());
        return 1;
    }
    err = pdmk(NULL, params, N_SRC, r_src, charge, NULL, N_TRG, r_trg, pot_src, pot_tree);
    if (err != DMK_SUCCESS) {
        fprintf(stderr, "pdmk failed (%d): %s\n", (int)err, pdmk_last_error_message());
        return 1;
    }

    double err2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < N_TRG; ++i) {
        const double d = pot_tree[i] - pot_direct[i];
        err2 += d * d;
        ref2 += pot_direct[i] * pot_direct[i];
    }
    if (!(ref2 > 0.0)) {
        fprintf(stderr, "reference potential is identically zero\n");
        return 1;
    }
    // The tree is asked for six digits; a squared ratio keeps sqrt out of it.
    const double tol = 1e-5;
    if (!(err2 < tol * tol * ref2)) {
        fprintf(stderr, "tree and direct disagree: rel_l2^2 = %g, tolerance %g\n", err2 / ref2, tol * tol);
        return 1;
    }

    printf("dmk %s (%s): tree matches direct over %d targets\n", pdmk_version_string(), pdmk_git_commit(), N_TRG);
    return 0;
}
