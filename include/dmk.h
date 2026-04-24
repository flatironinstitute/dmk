#ifndef DMK_H
#define DMK_H

#include <stdint.h>

typedef enum : int {
    DMK_YUKAWA = 0,
    DMK_LAPLACE = 1,
    DMK_SQRT_LAPLACE = 2,
    DMK_STOKESLET = 3,
} dmk_ikernel;

typedef enum : int {
    DMK_POTENTIAL = 1,
    DMK_POTENTIAL_GRAD = 2,
    DMK_POTENTIAL_GRAD_HESSIAN = 3,
    DMK_VELOCITY = 4,
    DMK_VELOCITY_PRESSURE = 5,
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

// Debug flags
enum {
    DMK_DEBUG_OMIT_PW = 1u << 0,        // Don't sum in plane-wave contributions
    DMK_DEBUG_OMIT_DIRECT = 1u << 1,    // Don't sum in direct constributions
    DMK_DEBUG_DUMP_TREE = 1u << 2,      // Dump tree files to local directory
    DMK_DEBUG_FORCE_AOT = 1u << 3,      // Use ahead-of-time kernels, even when compiled with JIT support
    DMK_DEBUG_OVERRIDE_BETA = 1u << 4,  // Load beta from debug_params[0]
    DMK_DEBUG_OVERRIDE_ORDER = 1u << 5, // Load proxy expansion order from debug_params[1]
    DMK_DEBUG_USE_PQ = 1u << 6,         // Use experimental priority queue for threading
};

enum {
    DMK_DEBUG_BETA_SLOT = 0,
    DMK_DEBUG_ORDER_SLOT = 1,
};

typedef void *pdmk_tree;
#ifdef DMK_HAVE_MPI
#include <mpi.h>
typedef MPI_Comm dmk_communicator;
#else
typedef void *dmk_communicator;
#endif

typedef struct pdmk_params {
    int n_dim = 0;                   // dimension of system
    double eps = 1e-3;               // target precision
    dmk_ikernel kernel = DMK_YUKAWA; // evaluation kernel
    dmk_pgh pgh_src = DMK_POTENTIAL; // level to compute at sources (potential, pot+grad, pot+grad+hess)
    dmk_pgh pgh_trg = DMK_POTENTIAL; // level to compute at sources (potential, pot+grad, pot+grad+hess)
    double fparam = 6.0;             // param for selected potential (FIXME: make more flexible)
    int use_periodic = false;        // use periodic boundary conditions (in all dimensions, currently)
    int n_per_leaf = 200;            // tuning: number of particles per leaf in N-tree
    int log_level = 6;               // 0: trace, 1: debug, 2: info, 3: warn, 4: err, 5: critical, 6: off
    uint32_t debug_flags = 0;        // Debug params bit field, see above
    double debug_params[8] = {0};    // 0: beta, 1: order, rest: placeholders
} pdmk_params;

#ifdef __cplusplus
extern "C" {
#endif

pdmk_tree pdmk_tree_createf(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src,
                            const float *charge, const float *normal, int n_trg, const float *r_trg);
void pdmk_tree_evalf(pdmk_tree tree, float *pot_src, float *pot_trg);
pdmk_tree pdmk_tree_create(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src,
                           const double *charge, const double *normal, int n_trg, const double *r_trg);

int pdmk_tree_update_charges(pdmk_tree tree, const double *charge, const double *normal);
int pdmk_tree_update_chargesf(pdmk_tree tree, const float *charge, const float *normal);

void pdmk_tree_destroy(pdmk_tree tree);
void pdmk_tree_eval(pdmk_tree tree, double *pot_src, double *pot_trg);
void pdmk_print_profile_data(dmk_communicator comm, char type);

void pdmk(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src, const double *charge,
          const double *normal, int n_trg, const double *r_trg, double *pot_src, double *pot_trg);
void pdmkf(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src, const float *charge,
           const float *normal, int n_trg, const float *r_trg, float *pot_src, float *pot_trg);
#ifdef __cplusplus
}
#endif

#endif
