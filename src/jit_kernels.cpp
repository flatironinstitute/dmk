#include <dmk/vector_kernels.hpp>

constexpr int VECDIMF = sctl::DefaultVecLen<float>();
constexpr int VECDIMD = sctl::DefaultVecLen<double>();

template void sqrt_laplace_2d_poly_all_pairs<float, VECDIMF>(int nd, int n_digits, float rsc, const float cen,
                                                             float d2max, float thresh2, int n_coeffs,
                                                             const float *coeffs, int n_src, const float *r_src,
                                                             const float *charge, int n_trg, const float *r_trg,
                                                             float *pot, int unroll_factor);
template void sqrt_laplace_2d_poly_all_pairs<double, VECDIMD>(int nd, int n_digits, double rsc, const double cen,
                                                              double d2max, double thresh2, int n_coeffs,
                                                              const double *coeffs, int n_src, const double *r_src,
                                                              const double *charge, int n_trg, const double *r_trg,
                                                              double *pot, int unroll_factor);

template void sqrt_laplace_3d_poly_all_pairs<float, VECDIMF>(int nd, int n_digits, float rsc, const float cen,
                                                             float d2max, float thresh2, int n_coeffs,
                                                             const float *coeffs, int n_src, const float *r_src,
                                                             const float *charge, int n_trg, const float *r_trg,
                                                             float *pot, int unroll_factor);
template void sqrt_laplace_3d_poly_all_pairs<double, VECDIMD>(int nd, int n_digits, double rsc, const double cen,
                                                              double d2max, double thresh2, int n_coeffs,
                                                              const double *coeffs, int n_src, const double *r_src,
                                                              const double *charge, int n_trg, const double *r_trg,
                                                              double *pot, int unroll_factor);

template void laplace_2d_poly_all_pairs<float, VECDIMF>(int nd, int n_digits, float rsc, const float cen, float d2max,
                                                        float thresh2, int n_coeffs, const float *coeffs, int n_src,
                                                        const float *r_src, const float *charge, int n_trg,
                                                        const float *r_trg, float *pot, int unroll_factor);
template void laplace_2d_poly_all_pairs<double, VECDIMD>(int nd, int n_digits, double rsc, const double cen,
                                                         double d2max, double thresh2, int n_coeffs,
                                                         const double *coeffs, int n_src, const double *r_src,
                                                         const double *charge, int n_trg, const double *r_trg,
                                                         double *pot, int unroll_factor);

template void laplace_3d_poly_all_pairs<float, VECDIMF>(int nd, int n_digits, float rsc, const float cen, float d2max,
                                                        float thresh2, int n_coeffs, const float *coeffs, int n_src,
                                                        const float *r_src, const float *charge, int n_trg,
                                                        const float *r_trg, float *pot, int unroll_factor);
template void laplace_3d_poly_all_pairs<double, VECDIMD>(int nd, int n_digits, double rsc, const double cen,
                                                         double d2max, double thresh2, int n_coeffs,
                                                         const double *coeffs, int n_src, const double *r_src,
                                                         const double *charge, int n_trg, const double *r_trg,
                                                         double *pot, int unroll_factor);

template void stokeslet_3d_poly_all_pairs<float, VECDIMF>(int nd, int n_digits, float rsc, const float cen, float d2max,
                                                          float thresh2, int n_coeffs_diag, int n_coeffs_offdiag,
                                                          const float *coeffs, int n_src, const float *r_src,
                                                          const float *charge, int n_trg, const float *r_trg,
                                                          float *pot, int unroll_factor);
template void stokeslet_3d_poly_all_pairs<double, VECDIMD>(int nd, int n_digits, double rsc, const double cen,
                                                           double d2max, double thresh2, int n_coeffs_diag,
                                                           int n_coeffs_offdiag, const double *coeffs, int n_src,
                                                           const double *r_src, const double *charge, int n_trg,
                                                           const double *r_trg, double *pot, int unroll_factor);
