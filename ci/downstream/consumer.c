#include <dmk.h>
#include <dmk/version.h>

#include <stdio.h>
#include <string.h>

int main(void) {
    pdmk_params params;
    pdmk_init_default_params(&params);

    if (strcmp(pdmk_version_string(), DMK_VERSION_STRING) != 0) {
        fprintf(stderr, "library reports %s, header says %s\n", pdmk_version_string(), DMK_VERSION_STRING);
        return 1;
    }
    if (params.n_dim != 0 || params.eps <= 0.0) {
        fprintf(stderr, "unexpected defaults: n_dim %d, eps %g\n", params.n_dim, params.eps);
        return 1;
    }

    printf("dmk %s (%s)\n", pdmk_version_string(), pdmk_git_commit());
    return 0;
}
