#ifndef DMK_H
#define DMK_H

#include <stdint.h>

typedef enum : int {
    DMK_YUKAWA = 0,
    DMK_LAPLACE = 1,
    DMK_SQRT_LAPLACE = 2,
    DMK_STOKESLET = 3,
    DMK_STRESSLET = 4,
    DMK_LAPLACE_DIPOLE = 5,
} dmk_ikernel;

typedef enum : int {
    DMK_POTENTIAL = 1,
    DMK_POTENTIAL_GRAD = 2,
    DMK_POTENTIAL_GRAD_HESSIAN = 3,
    DMK_VELOCITY = 4,
    DMK_VELOCITY_PRESSURE = 5,
} dmk_eval_type;

typedef enum : int {
    DMK_SUCCESS = 0,              // no error
    DMK_ERR_INVALID_ARGUMENT = 1, // null ptr, negative count, eps<=0, bad dim/kernel/eval value
    DMK_ERR_INTERNAL = 2,         // any C++ exception caught at the boundary (detail in last-error)
} dmk_error;

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
    int n_dim = 0;                          // dimension of system
    double eps = 1e-3;                      // target precision
    dmk_ikernel kernel = DMK_YUKAWA;        // evaluation kernel
    dmk_eval_type eval_src = DMK_POTENTIAL; // level to compute at sources (potential, pot+grad, pot+grad+hess)
    dmk_eval_type eval_trg = DMK_POTENTIAL; // level to compute at sources (potential, pot+grad, pot+grad+hess)
    double fparam = 6.0;                    // param for selected potential (FIXME: make more flexible)
    int use_periodic = false;               // use periodic boundary conditions (in all dimensions, currently)
    int n_per_leaf = 200;                   // tuning: number of particles per leaf in N-tree
    int log_level = 6;                      // 0: trace, 1: debug, 2: info, 3: warn, 4: err, 5: critical, 6: off
    uint32_t debug_flags = 0;               // Debug params bit field, see above
    double debug_params[8] = {0};           // 0: beta, 1: order, rest: placeholders
} pdmk_params;

#ifdef __cplusplus
extern "C" {
#endif

// Fill params with the library defaults. Always succeeds (no-op on a null pointer).
void pdmk_init_default_params(pdmk_params *params);

// Human-readable detail for the most recent failing call on the calling thread.
// Returns a NUL-terminated string (empty if no error). Valid until the next
// failing call on this thread.
const char *pdmk_last_error_message(void);

// Tree lifecycle. *_create returns an opaque handle, or NULL on failure (call
// pdmk_last_error_message for detail). All other entry points return a dmk_error
// (DMK_SUCCESS == 0 on success).
pdmk_tree pdmk_tree_createf(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src,
                            const float *charge, const float *normal, int n_trg, const float *r_trg);
dmk_error pdmk_tree_evalf(pdmk_tree tree, float *pot_src, float *pot_trg);
pdmk_tree pdmk_tree_create(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src,
                           const double *charge, const double *normal, int n_trg, const double *r_trg);

#ifdef DMK_BUILD_ESP
// ESP (Ewald Sum with PSWF kernels) — periodic Coulomb solver.
// Particles lie in the cubic box [-L/2, L/2)^3.

// Short-range method selection bits for pdmk_esp_params.esp_flags. The three strategies
// (source-pruning granularity, within-cell spatial sort, Newton's-third-law reciprocal) are
// independent. The default combination below is the empirically fastest.
enum {
    DMK_ESP_PRUNE_TILE = 1u << 0,   // sub-cell tile-vs-tile AABB pruning
    DMK_ESP_PRUNE_SOURCE = 1u << 1, // per-source point-vs-target-box pruning (finest granularity)
    DMK_ESP_N3L = 1u << 2,          // Newton's-third-law reciprocal (13-forward half stencil, 27-coloured)
    DMK_ESP_MORTON = 1u << 3,       // Morton within-cell sort (else octant-bin counting sort)
};

typedef struct pdmk_esp_params {
    double L;      // periodic box side length
    double r_c;    // real-space cutoff radius
    double eps;    // target precision
    int log_level; // 0: trace … 6: off (matches dmk_log_level)
    dmk_eval_type eval_type = DMK_POTENTIAL;
    // Short-range tuning; defaults are the fastest known combination.
    uint32_t esp_flags = DMK_ESP_PRUNE_SOURCE | DMK_ESP_N3L | DMK_ESP_MORTON; // DMK_ESP_* method bitmask
    int esp_bins = 2;  // octant-bin count per axis when DMK_ESP_MORTON is clear
    int esp_stile = 0; // source-tile width for DMK_ESP_PRUNE_TILE (0 -> SIMD width)
} pdmk_esp_params;

// Opaque plan handle (heap-allocated internally).
typedef void *pdmk_esp_plan;

// Create a plan: derives P, builds PSWFKernel, precomputes scaling grid.
// No particle data needed at this stage. Plan is always double precision internally.
pdmk_esp_plan pdmk_esp_plan_create(dmk_communicator comm, pdmk_esp_params params);
pdmk_esp_plan pdmk_esp_plan_createf(dmk_communicator comm, pdmk_esp_params params);

// Evaluate potentials for n particles using a pre-built plan.
// Can be called repeatedly with different r_src / charges for the same geometry.
void pdmk_esp_eval(dmk_communicator comm, pdmk_esp_plan plan, int n, const double *r_src, const double *charges,
                   double *pot_src);
void pdmk_esp_evalf(dmk_communicator comm, pdmk_esp_plan plan, int n, const float *r_src, const float *charges,
                    float *pot_src);

// Free the plan.
void pdmk_esp_plan_destroy(pdmk_esp_plan plan);
void pdmk_esp_plan_destroyf(pdmk_esp_plan plan);

// Convenience one-shot (create + eval + destroy).
void pdmk_esp(dmk_communicator comm, pdmk_esp_params params, int n, const double *r_src, const double *charges,
              double *pot_src);
void pdmk_espf(dmk_communicator comm, pdmk_esp_params params, int n, const float *r_src, const float *charges,
               float *pot_src);
#endif // DMK_BUILD_ESP

dmk_error pdmk_tree_update_charges(pdmk_tree tree, const double *charge, const double *normal);
dmk_error pdmk_tree_update_chargesf(pdmk_tree tree, const float *charge, const float *normal);

void pdmk_tree_destroy(pdmk_tree tree);
dmk_error pdmk_tree_eval(pdmk_tree tree, double *pot_src, double *pot_trg);
dmk_error pdmk_print_profile_data(dmk_communicator comm, char type);

dmk_error pdmk(dmk_communicator comm, pdmk_params params, int n_src, const double *r_src, const double *charge,
               const double *normal, int n_trg, const double *r_trg, double *pot_src, double *pot_trg);
dmk_error pdmkf(dmk_communicator comm, pdmk_params params, int n_src, const float *r_src, const float *charge,
                const float *normal, int n_trg, const float *r_trg, float *pot_src, float *pot_trg);
#ifdef __cplusplus
}
#endif

#endif
