#ifndef DMK_H
#define DMK_H

typedef enum {
    DMK_YUKAWA = 0,
    DMK_LAPLACE = 1,
    DMK_SQRT_LAPLACE = 2,
} dmk_ikernel;

typedef enum {
    DMK_POTENTIAL = 1,
    DMK_POTENTIAL_GRAD = 2,
    DMK_POTENTIAL_GRAD_HESSIAN = 3,
} dmk_pgh;

typedef struct pdmk_params {
    int n_mfm = 1;                      // number of charge/dipole dimensions per source location
    int n_dim = 0;                      // dimension of system
    double eps = 1e-7;                  // target precision
    dmk_ikernel kernel = DMK_YUKAWA;    // evaluation kernel
    dmk_pgh pgh = DMK_POTENTIAL;        // level to compute at sources (potential, pot+grad, pot+grad+hess)
    dmk_pgh pgh_target = DMK_POTENTIAL; // level to compute at targets (potential, pot+grad, pot+grad+hess)
    bool use_periodic = false;          // use PBC -- not implemented
    bool use_charge = true;             // use charges in charge array
    bool use_dipole = false;            // use dipoles in dipole array
    int n_per_leaf = 2000;              // tuning: number of particles per leaf in N-tree
    int log_level = 0;                  // 0: critical, 1: Error, 2: Warn, 3: Info, 4: Debug, 5: Trace
} pdmk_params;

#ifdef __cplusplus
extern "C" {
#endif
void pdmk(pdmk_params params, int n_src, const double *r_src, const double *charge, const double *normal,
          const double *dipole_str, int n_trg, const double *r_trg, double *pot, double *grad, double *hess,
          double *pottarg, double *gradtarg, double *hesstarg);
void pdmkf(pdmk_params params, int n_src, const float *r_src, const float *charge, const float *normal,
           const float *dipole_str, int n_trg, const float *r_trg, float *pot, float *grad, float *hess, float *pottarg,
           float *gradtarg, float *hesstarg);
#ifdef __cplusplus
}
#endif

#endif
