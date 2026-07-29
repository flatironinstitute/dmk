#pragma once

/// @file
/// V2 point-tree GPU evaluator. `pt::Tree` owns a private CPU `DMKPtTree` used
/// only for host precompute (build_tree_for_gpu / generate_metadata_for_gpu)
/// and charge sorting, then runs its own device pipeline over a `pt::State`.
///
/// Scaffold stage: `eval()` still delegates to the owned tree's V1 GPU pipeline
/// (kept as a live in-binary oracle). The V2 `pt::State` is built alongside and
/// validated against V1; the passes replace the delegation stage by stage.

#include <memory>

#include <dmk.h>
#include <dmk/cuda/pt/state.hpp>
#include <dmk/tree.hpp>
#include <sctl.hpp>

namespace dmk::cuda::pt {

template <typename Real, int DIM>
class Tree {
  public:
    Tree(const sctl::Comm &comm, const pdmk_params &params, const sctl::Vector<Real> &r_src,
         const sctl::Vector<Real> &charge, const sctl::Vector<Real> &normal, const sctl::Vector<Real> &r_trg);

    void eval();
    void desort_potentials(Real *pot_src, Real *pot_trg);
    void update_charges(const Real *charge, const Real *normal);
    const sctl::Comm &GetComm() const { return tree_->comm(); }

  private:
    std::unique_ptr<DMKPtTree<Real, DIM>> tree_;
    std::unique_ptr<State<Real, DIM>> state_;
};

} // namespace dmk::cuda::pt
