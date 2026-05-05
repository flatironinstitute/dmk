// Orchestration for the GPU offload of direct (near-field residual)
// interactions. See include/dmk/cuda_direct.hpp for the lifecycle.
//
// Read-only inputs and topology live in CudaSharedDeviceState (uploaded
// once). This file only manages direct's own output buffers and dispatches
// the kernel.

#include <dmk/cuda_direct.hpp>
#include <dmk/cuda_direct_kernels.hpp>
#include <dmk/cuda_shared_state.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/tree.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace dmk {

namespace {

#define DMK_CHECK_CUDA(expr)                                                                                           \
    do {                                                                                                               \
        cudaError_t _e = (expr);                                                                                       \
        if (_e != cudaSuccess)                                                                                         \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e));                            \
    } while (0)

template <typename T>
T *device_alloc_and_zero(std::size_t n) {
    if (n == 0)
        return nullptr;
    T *d = nullptr;
    DMK_CHECK_CUDA(cudaMalloc(&d, n * sizeof(T)));
    DMK_CHECK_CUDA(cudaMemsetAsync(d, 0, n * sizeof(T)));
    return d;
}

void device_free(void *p) {
    if (p)
        cudaFree(p);
}

} // namespace

template <typename Real, int DIM>
struct CudaDirectContext<Real, DIM>::Impl {
    DMKPtTree<Real, DIM> &tree;
    CudaSharedDeviceState<Real, DIM> &shared;

    Real *d_pot_src_direct = nullptr;
    Real *d_pot_trg_direct = nullptr;
    bool launched = false;

    Impl(DMKPtTree<Real, DIM> &t, CudaSharedDeviceState<Real, DIM> &s) : tree(t), shared(s) {}

    ~Impl() {
        device_free(d_pot_src_direct);
        device_free(d_pot_trg_direct);
    }
};

template <typename Real, int DIM>
CudaDirectContext<Real, DIM>::CudaDirectContext(DMKPtTree<Real, DIM> &tree, CudaSharedDeviceState<Real, DIM> &shared)
    : pimpl_(std::make_unique<Impl>(tree, shared)) {}

template <typename Real, int DIM>
CudaDirectContext<Real, DIM>::~CudaDirectContext() = default;

template <typename Real, int DIM>
void CudaDirectContext<Real, DIM>::launch() {
    auto &t = pimpl_->tree;
    auto &shared = pimpl_->shared;
    auto &im = *pimpl_;

    im.d_pot_src_direct = device_alloc_and_zero<Real>(shared.pot_src_size);
    im.d_pot_trg_direct = device_alloc_and_zero<Real>(shared.pot_trg_size);

    cuda::DirectByBoxArgs<Real> args;
    args.n_work = shared.n_direct_work;
    args.n_levels = shared.n_levels;
    args.nlist1_stride = shared.nlist1_stride;
    args.thresh2 = Real{1e-30};
    args.direct_work = shared.d_direct_work;
    args.list1_flat = shared.d_list1_flat;
    args.list1_count = shared.d_list1_count;
    args.box_levels = shared.d_box_levels;
    args.ifpwexp = shared.d_ifpwexp;
    args.direct_rsc = shared.d_direct_rsc;
    args.direct_cen = shared.d_direct_cen;
    args.direct_d2max = shared.d_direct_d2max;
    args.r_src_halo_flat = shared.d_r_src_halo;
    args.r_src_halo_offsets = shared.d_r_src_halo_offsets;
    args.src_counts_halo = shared.d_src_counts_halo;
    args.charge_halo_flat = shared.d_charge_halo;
    args.charge_halo_offsets = shared.d_charge_halo_offsets;
    args.normal_halo_flat = shared.d_normal_halo;
    args.normal_halo_offsets = shared.d_normal_halo_offsets;

    // pot_src side: target points are the trg_box's owned sources.
    args.r_target_flat = shared.d_r_src_owned;
    args.r_target_offsets = shared.d_r_src_owned_offsets;
    args.target_counts = shared.d_src_counts_owned;
    args.pot_flat = im.d_pot_src_direct;
    args.pot_offsets = shared.d_pot_src_offsets;
    cuda::launch_direct_by_box_dispatch<Real>(t.params.kernel, DIM, t.n_digits, args, shared.direct_stream);

    // pot_trg side: target points are the trg_box's owned targets.
    args.r_target_flat = shared.d_r_trg_owned;
    args.r_target_offsets = shared.d_r_trg_owned_offsets;
    args.target_counts = shared.d_trg_counts_owned;
    args.pot_flat = im.d_pot_trg_direct;
    args.pot_offsets = shared.d_pot_trg_offsets;
    cuda::launch_direct_by_box_dispatch<Real>(t.params.kernel, DIM, t.n_digits, args, shared.direct_stream);

    im.launched = true;
}

template <typename Real, int DIM>
void CudaDirectContext<Real, DIM>::merge_into_host() {
    auto &t = pimpl_->tree;
    auto &shared = pimpl_->shared;
    auto &im = *pimpl_;

    if (!im.launched)
        return;

    DMK_CHECK_CUDA(cudaDeviceSynchronize());

    if (shared.pot_src_size) {
        std::vector<Real> tmp(shared.pot_src_size);
        DMK_CHECK_CUDA(
            cudaMemcpy(tmp.data(), im.d_pot_src_direct, shared.pot_src_size * sizeof(Real), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < shared.pot_src_size; ++i)
            t.pot_src_sorted[i] += tmp[i];
    }
    if (shared.pot_trg_size) {
        std::vector<Real> tmp(shared.pot_trg_size);
        DMK_CHECK_CUDA(
            cudaMemcpy(tmp.data(), im.d_pot_trg_direct, shared.pot_trg_size * sizeof(Real), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < shared.pot_trg_size; ++i)
            t.pot_trg_sorted[i] += tmp[i];
    }
}

template class CudaDirectContext<float, 2>;
template class CudaDirectContext<float, 3>;
template class CudaDirectContext<double, 2>;
template class CudaDirectContext<double, 3>;

} // namespace dmk
