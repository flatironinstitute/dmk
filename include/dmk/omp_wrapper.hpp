#ifndef OMP_WRAPPER_H
#define OMP_WRAPPER_H

// Taken from FINUFFT
#ifdef _OPENMP
#include <omp.h>
// point to actual omp utils
static inline int MY_OMP_GET_NUM_THREADS [[maybe_unused]] () {
  return omp_get_num_threads();
}
static inline int MY_OMP_GET_MAX_THREADS [[maybe_unused]] () {
  return omp_get_max_threads();
}
static inline int MY_OMP_GET_THREAD_NUM [[maybe_unused]] () {
  return omp_get_thread_num();
}
static inline void MY_OMP_SET_NUM_THREADS [[maybe_unused]] (int x) {
  omp_set_num_threads(x);
}
#else
// non-omp safe dummy versions of omp utils...
static inline int MY_OMP_GET_NUM_THREADS [[maybe_unused]] () { return 1; }
static inline int MY_OMP_GET_MAX_THREADS [[maybe_unused]] () { return 1; }
static inline int MY_OMP_GET_THREAD_NUM [[maybe_unused]] () { return 0; }
static inline void MY_OMP_SET_NUM_THREADS [[maybe_unused]] (int) {}
#endif


#endif
