#ifndef GEMM_H
#define GEMM_H

#include <dmk/util.hpp>
#include <complex>

extern "C" {
void sgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N, const int *K, const float *ALPHA,
            const float *A, const int *LDA, const float *B, const int *LDB, const float *BETA, float *C,
            const int *LDC);
void dgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N, const int *K, const double *ALPHA,
            const double *A, const int *LDA, const double *B, const int *LDB, const double *BETA, double *C,
            const int *LDC);
void cgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N, const int *K, const float *ALPHA,
            const float *A, const int *LDA, const float *B, const int *LDB, const float *BETA, float *C,
            const int *LDC);
void zgemm_(const char *TRANSA, const char *TRANSB, const int *M, const int *N, const int *K, const double *ALPHA,
            const double *A, const int *LDA, const double *B, const int *LDB, const double *BETA, double *C,
            const int *LDC);
}
namespace dmk::gemm {

template <typename T>
inline void gemm(char transa, char transb, int m, int n, int k, T alpha, const T *a, int lda, const T *b, int ldb,
                T beta, T *c, int ldc) {
    if constexpr (std::is_same_v<T, float>)
        return sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    else if constexpr (std::is_same_v<T, double>)
        return dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        return cgemm_(&transa, &transb, &m, &n, &k, (float *)&alpha, (float *)a, &lda, (float *)b, &ldb, (float *)&beta,
                      (float *)c, &ldc);
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        return zgemm_(&transa, &transb, &m, &n, &k, (double *)&alpha, (double *)a, &lda, (double *)b, &ldb,
                      (double *)&beta, (double *)c, &ldc);
    else
        static_assert(dmk::util::always_false<T>, "Invalid type supplied to complex_gemm");
}
}

#endif
