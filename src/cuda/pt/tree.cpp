#include <dmk/cuda/pt/tree.hpp>

#include <dmk/cuda/helpers.hpp>
#include <dmk/cuda/pt/passes.hpp>

namespace dmk::cuda::pt {

template <typename Real, int DIM>
Tree<Real, DIM>::Tree(const sctl::Comm &comm, const pdmk_params &params, const sctl::Vector<Real> &r_src,
                      const sctl::Vector<Real> &charge, const sctl::Vector<Real> &normal,
                      const sctl::Vector<Real> &r_trg) {
    // The owned tree runs the GPU host precompute only (tree build, metadata,
    // and plane-wave layout); all device state lives in state_.
    tree_ = std::make_unique<DMKPtTree<Real, DIM>>(comm, params, r_src, charge, normal, r_trg);
    tree_->init_planewave_data();

    state_ = std::make_unique<State<Real, DIM>>(to_build_inputs(*tree_));
    const long n_src = r_src.Dim() / DIM;
    const Real *charge_ptr = charge.Dim() ? &charge[0] : nullptr;
    const Real *normal_ptr = (params.kernel == DMK_STRESSLET && normal.Dim()) ? &normal[0] : nullptr;
    state_->upload_and_sort_charges(charge_ptr, normal_ptr, n_src);
}

template <typename Real, int DIM>
void Tree<Real, DIM>::eval() {
    // The near-field `direct` runs concurrently on direct_stream with the
    // upward -> form_outgoing -> downward -> eval_targets chain on
    // downward_stream; `finalize` joins them (direct_stream waits on the
    // downward-stream eval writes), sums the near+far potentials, descatters to
    // user order in d_pot_*_final, and syncs.
    const auto ds = state_->direct_stream.get();
    const auto ws = state_->downward_stream.get();
    pt::direct(*state_, ds);
    pt::upward(*state_, ws);
    pt::form_outgoing(*state_, ws);
    pt::downward(*state_, ws);
    pt::eval_targets(*state_, ws);
    state_->finalize();
}

template <typename Real, int DIM>
void Tree<Real, DIM>::desort_potentials(Real *pot_src, Real *pot_trg) {
    // finalize wrote the descattered (user-order) result into d_pot_*_final and
    // synced; one D2H per side.
    const auto &o = state_->outputs;
    if (o.pot_src_size)
        DMK_CHECK_CUDA(
            cudaMemcpy(pot_src, o.d_pot_src_final.data(), o.pot_src_size * sizeof(Real), cudaMemcpyDeviceToHost));
    if (o.pot_trg_size)
        DMK_CHECK_CUDA(
            cudaMemcpy(pot_trg, o.d_pot_trg_final.data(), o.pot_trg_size * sizeof(Real), cudaMemcpyDeviceToHost));
}

template <typename Real, int DIM>
void Tree<Real, DIM>::update_charges(const Real *charge, const Real *normal) {
    state_->upload_and_sort_charges(charge, normal, tree_->r_src_sorted_owned.Dim() / DIM);
}

template class Tree<float, 2>;
template class Tree<float, 3>;
template class Tree<double, 2>;
template class Tree<double, 3>;

} // namespace dmk::cuda::pt
