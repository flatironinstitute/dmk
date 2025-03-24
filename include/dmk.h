#ifndef DMK_H
#define DMK_H

#include <mpi.h>

typedef enum : int {
    DMK_YUKAWA = 0,
    DMK_LAPLACE = 1,
    DMK_SQRT_LAPLACE = 2,
} dmk_ikernel;

typedef enum : int {
    DMK_POTENTIAL = 1,
    DMK_POTENTIAL_GRAD = 2,
    DMK_POTENTIAL_GRAD_HESSIAN = 3,
} dmk_pgh;

typedef enum : int {
    DMK_LOG_TRACE = 0,
    DMK_LOG_DEBUG = 1,
    DMK_LOG_INFO = 2,
    DMK_LOG_WARN = 3,
    DMK_LOG_ERR = 4,
    DMK_LOG_CRITICAL = 5,
    DMK_LOG_OFF = 6,
} dmk_log_level;

typedef void *pdmk_tree;

typedef struct pdmk_params {
    int n_mfm = 1;                   // number of charge/dipole dimensions per source location
    int n_dim = 0;                   // dimension of system
    double eps = 1e-7;               // target precision
    dmk_ikernel kernel = DMK_YUKAWA; // evaluation kernel
    dmk_pgh pgh_src = DMK_POTENTIAL; // level to compute at sources (potential, pot+grad, pot+grad+hess)
    dmk_pgh pgh_trg = DMK_POTENTIAL; // level to compute at sources (potential, pot+grad, pot+grad+hess)
    double fparam = 6.0;             // param for selected potential (FIXME: make more flexible)
    const int use_periodic = false;  // use PBC -- not implemented
    int use_charge = 1;              // use charges in charge array
    int n_per_leaf = 2000;           // tuning: number of particles per leaf in N-tree
    int log_level = 6;               // 0: trace, 1: debug, 2: info, 3: warn, 4: err, 5: critical, 6: off
} pdmk_params;

#ifdef __cplusplus
extern "C" {
#endif

pdmk_tree pdmk_tree_create(MPI_Comm comm, pdmk_params params, int n_src, const double *r_src, const double *charge,
                           const double *normal, const double *dipole_str, int n_trg, const double *r_trg);
void pdmk_tree_destroy(pdmk_tree tree);
void pdmk_tree_eval(pdmk_tree tree, double *pot_src, double *grad_src, double *hess_src, double *pot_trg,
                    double *grad_trg, double *hess_trg);

void pdmk(MPI_Comm comm, pdmk_params params, int n_src, const double *r_src, const double *charge, const double *normal,
          const double *dipole_str, int n_trg, const double *r_trg, double *pot_src, double *grad_src, double *hess_src,
          double *pot_trg, double *grad_trg, double *hess_trg);
void pdmkf(MPI_Comm comm, pdmk_params params, int n_src, const float *r_src, const float *charge, const float *normal,
           const float *dipole_str, int n_trg, const float *r_trg, float *pot_src, float *grad_src, float *hess_src,
           float *pot_trg, float *grad_trg, float *hess_trg);
#ifdef __cplusplus
}
#endif

#endif
