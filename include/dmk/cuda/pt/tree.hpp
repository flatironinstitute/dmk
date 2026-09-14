#pragma once

/// @file
/// Point-tree GPU evaluator. `pt::Tree` owns a private CPU `DMKPtTree` used only
/// for host precompute (build_tree_for_gpu / generate_metadata_for_gpu /
/// init_planewave_data) and charge sorting, then runs its own device pipeline
/// over a `pt::State`.

#include <memory>

#include <dmk.h>
#include <dmk/cuda/pt/state.hpp>
#include <dmk/tree.hpp>
#include <sctl.hpp>

namespace dmk::cuda::pt {

/// Range-check `device_id` against the visible CUDA devices and pin the process to it.
/// Throws api_error(DMK_ERR_INVALID_ARGUMENT) for an out-of-range id, or for a second,
/// different id: the JIT module caches, autotune records and cached device properties are
/// function-local statics bound to the device that was current when they were first
/// touched, so one process drives one GPU (one device per rank under MPI).
void bind_gpu_device(int device_id);

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
    std::unique_ptr<State<Real, DIM>> state_;
    int device_id_ = 0;
};

} // namespace dmk::cuda::pt
