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
    std::unique_ptr<State<Real, DIM>> state_;
    sctl::Long n_src_local_ = 0;    ///< sources this rank supplied, the order update_charges is given
    sctl::Long n_trg_local_ = 0;    ///< targets this rank supplied
};

} // namespace dmk::cuda::pt
