#include <dmk/cuda/direct.hpp>
#include <dmk/cuda/direct_kernels.hpp>
#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/shared_state.hpp>
#include <dmk/fourier_data.hpp>
#include <dmk/tree.hpp>

#include <cuda_runtime.h>

namespace dmk {

template <typename Real, int DIM>
CudaDirectContext<Real, DIM>::CudaDirectContext(DMKPtTree<Real, DIM> &tree, CudaSharedDeviceState<Real, DIM> &shared)
    : tree_(tree), shared_(shared) {
    d_pot_src_direct_.resize(shared.pot_src_size);
    d_pot_trg_direct_.resize(shared.pot_trg_size);
}

template <typename Real, int DIM>
void CudaDirectContext<Real, DIM>::launch() {
    auto &t = tree_;
    auto &shared = shared_;

    d_pot_src_direct_.zero_async(shared.direct_stream);
    d_pot_trg_direct_.zero_async(shared.direct_stream);

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
    args.r_src_flat = shared.d_r_src.data();
    args.r_src_offsets = shared.d_r_src_offsets.data();
    args.src_counts = shared.d_src_counts.data();
    args.charge_flat = shared.d_charge.data();
    args.charge_offsets = shared.d_charge_offsets.data();
    args.normal_flat = shared.d_normal.data();
    args.normal_offsets = shared.d_normal_offsets.data();

    // pot_src side: target points are the trg_box's sources.
    args.r_target_flat = shared.d_r_src.data();
    args.r_target_offsets = shared.d_r_src_offsets.data();
    args.target_counts = shared.d_src_counts.data();
    args.pot_flat = d_pot_src_direct_.data();
    args.pot_offsets = shared.d_pot_src_offsets.data();
    cuda::launch_direct_by_box_dispatch<Real>(t.params.kernel, DIM, t.n_digits, args, shared.direct_stream);

    // pot_trg side: target points are the trg_box's targets.
    args.r_target_flat = shared.d_r_trg.data();
    args.r_target_offsets = shared.d_r_trg_offsets.data();
    args.target_counts = shared.d_trg_counts.data();
    args.pot_flat = d_pot_trg_direct_.data();
    args.pot_offsets = shared.d_pot_trg_offsets.data();
    cuda::launch_direct_by_box_dispatch<Real>(t.params.kernel, DIM, t.n_digits, args, shared.direct_stream);
}

template class CudaDirectContext<float, 2>;
template class CudaDirectContext<float, 3>;
template class CudaDirectContext<double, 2>;
template class CudaDirectContext<double, 3>;

} // namespace dmk
