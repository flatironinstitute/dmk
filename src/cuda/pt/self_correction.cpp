#include <dmk/cuda/pt/passes.hpp>

#include "../jit/jit_cache.hpp"
#include "../jit/jit_kernel.hpp"
#include "../jit/jit_source_utils.hpp"
#include "launchers.hpp"

#include <dmk.h>
#include <dmk/cuda/self_correction_kernelargs.hpp>
#include <dmk/direct.hpp>

#include <cuda_runtime.h>

namespace dmk::cuda::pt {
namespace {

using jit::jit_real_name;
using jit::JitCache;
using jit::JitKey;

template <typename Real>
void launch_self_correction(JitCache &cache, const dmk::cuda::SelfCorrectionArgs<Real> &args, cudaStream_t stream) {
    if (args.n_direct_work == 0)
        return;
    constexpr int BLOCK = 128;
    JitKey key;
    key.name = "PtSelfCorrectionKernel";
    key.real = jit_real_name<Real>();
    key.sm_major = cache.sm_major();
    key.sm_minor = cache.sm_minor();
    auto kernel = cache.get_kernel_from_source(
        key, [&] { return make_stage_source("pt/self_correction.cu", key, "", "PtSelfCorrection"); });
    dmk::cuda::SelfCorrectionArgs<Real> a = args;
    kernel->launch(dim3(a.n_direct_work, 1, 1), dim3(BLOCK, 1, 1), 0, stream, a);
}

} // namespace

template <typename Real, int DIM>
void self_correction(State<Real, DIM> &s, cudaStream_t stream) {
    auto &o = s.outputs;
    if (!o.pot_src_size || !s.worklists.d_self_correction_work.size())
        return;

    static JitCache sc_cache;

    // Self-correction modifies the direct potential in sorted layout.
    dmk::cuda::SelfCorrectionArgs<Real> sc;
    sc.direct_work = s.topology.d_direct_work.data();
    sc.correction_factors = s.worklists.d_self_correction_work.data();
    sc.src_counts = s.particles.d_src_counts_owned.data();
    sc.charge = s.particles.d_charge.data();
    sc.charge_offsets = s.particles.d_charge_offsets.data();
    sc.pot_src = o.d_pot_direct_src.data();
    sc.pot_src_offsets = o.d_pot_src_offsets.data();
    sc.n_direct_work = static_cast<int>(s.topology.d_direct_work.size());
    sc.n_input_dim = get_kernel_input_dim(DIM, s.kernel);
    sc.pot_stride = o.pot_src_dof;
    // Dipole corrects the DIM gradient components, not the potential.
    sc.pot_output_offset = (s.kernel == DMK_LAPLACE_DIPOLE) ? 1 : 0;
    launch_self_correction<Real>(sc_cache, sc, stream);
}

template void self_correction<float, 2>(State<float, 2> &, cudaStream_t);
template void self_correction<float, 3>(State<float, 3> &, cudaStream_t);
template void self_correction<double, 2>(State<double, 2> &, cudaStream_t);
template void self_correction<double, 3>(State<double, 3> &, cudaStream_t);

} // namespace dmk::cuda::pt
