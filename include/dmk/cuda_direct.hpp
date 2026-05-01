#ifndef DMK_CUDA_DIRECT_HPP
#define DMK_CUDA_DIRECT_HPP

// GPU offload of the direct (near-field residual) interactions.
//
// Lifecycle, driven by DMKPtTree::upward_pass / downward_pass when
// DMK_GPU_OFFLOAD is enabled at configure time:
//
//   1. upward_pass() construct a CudaDirectContext and call launch():
//        - upload positions / charges / normals / densities to device
//        - allocate + zero device pot_src / pot_trg buffers
//        - for each (trg_box, src_box) list1 pair, call the CUDA AOT
//          residual evaluator on the default stream (async).
//   2. downward_pass() runs the multilevel work on the CPU. The GPU work
//      is in flight on the default stream throughout.
//   3. End of downward_pass(): call merge_into_host():
//        - cudaDeviceSynchronize()
//        - copy device pot buffers back to host
//        - accumulate into tree.pot_src_sorted / pot_trg_sorted (which
//          already hold the multilevel far-field contributions).
//
// Simplifications relative to the CPU direct loop (deliberate, per design):
//   * no ContactGeometry filtering — all owned src / trg points in the trg
//     box are evaluated against all halo sources of every list1 source box.
//   * no PBC support yet (throws if params.use_periodic).
//   * one CUDA kernel launch per (trg_box, src_box) pair — load is uneven
//     but it works.

#include <memory>

namespace dmk {

template <typename Real, int DIM>
struct DMKPtTree;

template <typename Real, int DIM>
class CudaDirectContext {
  public:
    explicit CudaDirectContext(DMKPtTree<Real, DIM> &tree);
    ~CudaDirectContext();
    CudaDirectContext(const CudaDirectContext &) = delete;
    CudaDirectContext &operator=(const CudaDirectContext &) = delete;

    /// Upload tree data, allocate device output buffers, and queue all
    /// residual kernels on the default stream. Returns immediately.
    void launch();

    /// Synchronize the device, copy device pot buffers back, and accumulate
    /// into tree.pot_src_sorted / pot_trg_sorted.
    void merge_into_host();

  private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace dmk

#endif // DMK_CUDA_DIRECT_HPP
