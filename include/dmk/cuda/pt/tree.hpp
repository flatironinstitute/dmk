#pragma once

/// @file
/// Point-tree GPU evaluator. `pt::Tree` owns a `device_tree::PtTree`, which holds the nodes and
/// moves particle data in and out, and a `DMKPtTree` that runs the host precompute over a host copy
/// of its nodes; the device pipeline then runs over a `pt::State`.

#include <memory>

#include <dmk.h>
#include <dmk/cuda/device_tree.hpp>
#include <dmk/cuda/pt/state.hpp>
#include <dmk/tree.hpp>
#include <sctl.hpp>

namespace dmk::cuda::pt {

/// Pin this process to one CUDA device. A negative id defers the choice to the tree, which
/// resolves it against its communicator; any other value is taken as given and must be in range.
/// The choice is made once and cannot be changed: the JIT module caches, autotune records and
/// cached device properties are function-local statics bound to the device that was current when
/// they were first touched, so one process drives one GPU (one device per rank under MPI).
void bind_gpu_device(int device_id);

/// Device storage behind the per-box metadata; defined in tree.cpp, which is the only place that
/// needs its layout.
template <typename Real, int DIM>
struct DeviceMetadata;

template <typename Real, int DIM>
class Tree {
  public:
    Tree(const sctl::Comm &comm, const pdmk_params &params, const sctl::Vector<Real> &r_src,
         const sctl::Vector<Real> &charge, const sctl::Vector<Real> &normal, const sctl::Vector<Real> &r_trg);
    ~Tree();

    void eval();
    void desort_potentials(Real *pot_src, Real *pot_trg);
    void update_charges(const Real *charge, const Real *normal);
    const sctl::Comm &GetComm() const { return tree_->comm(); }

  private:
    std::unique_ptr<DMKPtTree<Real, DIM>> tree_;
    std::unique_ptr<device_tree::PtTree<Real, DIM>> dev_tree_;
    std::unique_ptr<DeviceMetadata<Real, DIM>> md_; ///< declared before state_, which adopts its buffers
    std::unique_ptr<State<Real, DIM>> state_;
    sctl::Long n_src_local_ = 0; ///< sources this rank supplied, the order update_charges is given
    sctl::Long n_trg_local_ = 0; ///< targets this rank supplied
};

} // namespace dmk::cuda::pt
