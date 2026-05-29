#ifndef DMK_CUDA_SHARED_STATE_KERNELS_HPP
#define DMK_CUDA_SHARED_STATE_KERNELS_HPP

#include <cuda_runtime.h>

namespace dmk::cuda {

template <typename Real>
void launch_accumulate_and_scatter(Real *out_unsorted, const Real *pot_eval, const Real *pot_extra,
                                   const long *scatter_index, int dof, long n_particles, cudaStream_t stream);

template <typename Real>
void launch_scatter_forward_stresslet(const Real *densities, const Real *normals, Real *out, const long *scatter_index,
                                      long n_particles, int dim, cudaStream_t stream);

template <typename Real>
void launch_scatter_forward(const Real *in_unsorted, Real *out, const long *scatter_index, long n_particles, int dof,
                            cudaStream_t stream);

} // namespace dmk::cuda

#endif // DMK_CUDA_SHARED_STATE_KERNELS_HPP
