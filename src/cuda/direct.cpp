// Orchestration for the GPU offload of direct (near-field residual)
// interactions. See include/dmk/cuda_direct.hpp for the lifecycle.
//
// Read-only inputs and topology live in CudaSharedDeviceState (uploaded
// once). This file only manages direct's own output buffers and dispatches
// the kernel.

#include <dmk/cuda/direct.hpp>
#include <dmk/cuda/direct_kernels.hpp>
#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/shared_state.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/tree.hpp>

#include <cuda_runtime.h>

namespace dmk {

using cuda_helpers::device_alloc_and_zero;
using cuda_helpers::device_free;

template <typename Real, int DIM>
struct CudaDirectContext<Real, DIM>::Impl {
    DMKPtTree<Real, DIM> &tree;
    CudaSharedDeviceState<Real, DIM> &shared;

    // Output pot buffers; allocated once at construction and zeroed at the
    // start of each launch() so the context can be reused across evals.
    Real *d_pot_src_direct = nullptr;
    Real *d_pot_trg_direct = nullptr;
    bool launched = false;

    Impl(DMKPtTree<Real, DIM> &t, CudaSharedDeviceState<Real, DIM> &s) : tree(t), shared(s) {
        d_pot_src_direct = device_alloc_and_zero<Real>(shared.pot_src_size);
        d_pot_trg_direct = device_alloc_and_zero<Real>(shared.pot_trg_size);
    }

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

    if (im.d_pot_src_direct)
        DMK_CHECK_CUDA(
            cudaMemsetAsync(im.d_pot_src_direct, 0, shared.pot_src_size * sizeof(Real), shared.direct_stream));
    if (im.d_pot_trg_direct)
        DMK_CHECK_CUDA(
            cudaMemsetAsync(im.d_pot_trg_direct, 0, shared.pot_trg_size * sizeof(Real), shared.direct_stream));

    cuda::DirectByBoxArgs<Real> args;
    args.n_work = shared.n_direct_work;
    args.n_levels = shared.n_levels;
    args.nlist1_stride = shared.nlist1_stride;
    args.thresh2 = Real{1e-30};
    args.direct_work = shared.d_direct_work.data();
    args.list1_flat = shared.d_list1_flat.data();
    args.list1_count = shared.d_list1_count.data();
    args.box_levels = shared.d_box_levels.data();
    args.ifpwexp = shared.d_ifpwexp.data();
    args.direct_rsc = shared.d_direct_rsc.data();
    args.direct_cen = shared.d_direct_cen.data();
    args.direct_d2max = shared.d_direct_d2max.data();
    args.r_src_halo_flat = shared.d_r_src_halo.data();
    args.r_src_halo_offsets = shared.d_r_src_halo_offsets.data();
    args.src_counts_halo = shared.d_src_counts_halo.data();
    args.charge_halo_flat = shared.d_charge_halo.data();
    args.charge_halo_offsets = shared.d_charge_halo_offsets.data();
    args.normal_halo_flat = shared.d_normal_halo.data();
    args.normal_halo_offsets = shared.d_normal_halo_offsets.data();

    // pot_src side: target points are the trg_box's owned sources.
    args.r_target_flat = shared.d_r_src_owned.data();
    args.r_target_offsets = shared.d_r_src_owned_offsets.data();
    args.target_counts = shared.d_src_counts_owned.data();
    args.pot_flat = im.d_pot_src_direct;
    args.pot_offsets = shared.d_pot_src_offsets.data();
    cuda::launch_direct_by_box_dispatch<Real>(t.params.kernel, DIM, t.n_digits, args, shared.direct_stream);

    // pot_trg side: target points are the trg_box's owned targets.
    args.r_target_flat = shared.d_r_trg_owned.data();
    args.r_target_offsets = shared.d_r_trg_owned_offsets.data();
    args.target_counts = shared.d_trg_counts_owned.data();
    args.pot_flat = im.d_pot_trg_direct;
    args.pot_offsets = shared.d_pot_trg_offsets.data();
    cuda::launch_direct_by_box_dispatch<Real>(t.params.kernel, DIM, t.n_digits, args, shared.direct_stream);

    im.launched = true;
}

template <typename Real, int DIM>
Real *CudaDirectContext<Real, DIM>::device_pot_src() const {
    return pimpl_->d_pot_src_direct;
}

template <typename Real, int DIM>
Real *CudaDirectContext<Real, DIM>::device_pot_trg() const {
    return pimpl_->d_pot_trg_direct;
}

template class CudaDirectContext<float, 2>;
template class CudaDirectContext<float, 3>;
template class CudaDirectContext<double, 2>;
template class CudaDirectContext<double, 3>;

} // namespace dmk
