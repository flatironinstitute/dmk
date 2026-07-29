#include <dmk/cuda/pt/tree.hpp>

#include <dmk/cuda/direct.hpp>
#include <dmk/cuda/pt/passes.hpp>
#include <dmk/cuda/shared_state.hpp>
#include <dmk/util.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dmk::cuda::pt {

namespace {

// Transitional seam gate (removed at cutover): confirm every buffer the V2
// State uploaded matches the owned tree's V1 CudaSharedDeviceState oracle
// byte-for-byte. Enabled with DMK_GPU_V2_CHECK.
template <typename T>
bool dev_equal(const cuda_helpers::DeviceBuffer<T> &a, const cuda_helpers::DeviceBuffer<T> &b) {
    if (a.size() != b.size())
        return false;
    if (a.size() == 0)
        return true;
    std::vector<T> ha(a.size()), hb(b.size());
    cudaMemcpy(ha.data(), a.data(), a.size_bytes(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b.data(), b.size_bytes(), cudaMemcpyDeviceToHost);
    return std::memcmp(ha.data(), hb.data(), a.size_bytes()) == 0;
}

// Relative L2 error between two device buffers of `n` reals.
template <typename Real>
double dev_rel_l2(const Real *a, const Real *b, std::size_t n) {
    if (n == 0)
        return 0.0;
    std::vector<Real> ha(n), hb(n);
    cudaMemcpy(ha.data(), a, n * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, n * sizeof(Real), cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = double(ha[i]) - double(hb[i]);
        num += d * d;
        den += double(hb[i]) * double(hb[i]);
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

template <typename Real, int DIM>
void seam_self_check(State<Real, DIM> &s, CudaSharedDeviceState<Real, DIM> &v1) {
    // Reports to stderr directly (not the dmk logger) so the gate is greppable
    // regardless of the driver's log level.
    int total = 0, mism = 0;
    auto ck = [&](const char *n, const auto &a, const auto &b) {
        ++total;
        if (!dev_equal(a, b)) {
            ++mism;
            std::fprintf(stderr, "[GPU_V2] seam mismatch: %s\n", n);
        }
    };
    ck("direct_work", s.topology.d_direct_work, v1.d_direct_work);
    ck("list1_flat", s.topology.d_list1_flat, v1.d_list1_flat);
    ck("list1_count", s.topology.d_list1_count, v1.d_list1_count);
    ck("box_levels", s.topology.d_box_levels, v1.d_box_levels);
    ck("ifpwexp", s.topology.d_ifpwexp, v1.d_ifpwexp);
    ck("neighbors", s.topology.d_neighbors, v1.d_neighbors);
    ck("is_global_leaf", s.topology.d_is_global_leaf, v1.d_is_global_leaf);
    ck("r_src", s.particles.d_r_src, v1.d_r_src);
    ck("r_src_offsets", s.particles.d_r_src_offsets, v1.d_r_src_offsets);
    ck("src_counts", s.particles.d_src_counts, v1.d_src_counts);
    ck("charge_offsets", s.particles.d_charge_offsets, v1.d_charge_offsets);
    ck("r_trg", s.particles.d_r_trg, v1.d_r_trg);
    ck("r_trg_offsets", s.particles.d_r_trg_offsets, v1.d_r_trg_offsets);
    ck("trg_counts", s.particles.d_trg_counts, v1.d_trg_counts);
    ck("scatter_index_src", s.particles.d_scatter_index_src, v1.d_scatter_index_src);
    ck("scatter_index_trg", s.particles.d_scatter_index_trg, v1.d_scatter_index_trg);
    ck("normal_offsets", s.particles.d_normal_offsets, v1.d_normal_offsets);
    ck("charge_outer_offsets", s.particles.d_charge_outer_offsets, v1.d_charge_outer_offsets);
    ck("direct_rsc", s.fourier.d_direct_rsc, v1.d_direct_rsc);
    ck("direct_cen", s.fourier.d_direct_cen, v1.d_direct_cen);
    ck("direct_d2max", s.fourier.d_direct_d2max, v1.d_direct_d2max);
    ck("pw2poly_flat", s.fourier.d_pw2poly_flat, v1.d_pw2poly_flat);
    ck("poly2pw_flat", s.fourier.d_poly2pw_flat, v1.d_poly2pw_flat);
    ck("radialft_flat", s.fourier.d_radialft_flat, v1.d_radialft_flat);
    ck("wpwshift_flat", s.fourier.d_wpwshift_flat, v1.d_wpwshift_flat);
    ck("window_pw2poly", s.fourier.d_window_pw2poly, v1.d_window_pw2poly);
    ck("window_poly2pw", s.fourier.d_window_poly2pw, v1.d_window_poly2pw);
    ck("window_radialft", s.fourier.d_window_radialft, v1.d_window_radialft);
    ck("p2c", s.fourier.d_p2c, v1.d_p2c);
    ck("c2p", s.fourier.d_c2p, v1.d_c2p);
    ck("centers", s.fourier.d_centers, v1.d_centers);
    ck("inv_box_scale", s.fourier.d_inv_box_scale, v1.d_inv_box_scale);
    ck("c2p_center_boxes", s.worklists.d_c2p_center_boxes, v1.d_c2p_center_boxes);
    ck("c2p_levels", s.worklists.d_c2p_levels, v1.d_c2p_levels);
    ck("c2p_src_box_flat_offsets", s.worklists.d_c2p_src_box_flat_offsets, v1.d_c2p_src_box_flat_offsets);
    ck("c2p_n_src_boxes_per_group", s.worklists.d_c2p_n_src_boxes_per_group, v1.d_c2p_n_src_boxes_per_group);
    ck("c2p_src_boxes_flat", s.worklists.d_c2p_src_boxes_flat, v1.d_c2p_src_boxes_flat);
    ck("c2p_group_perm", s.worklists.d_c2p_group_perm, v1.d_c2p_group_perm);
    ck("tp_parents", s.worklists.d_tp_parents, v1.d_tp_parents);
    ck("tp_children", s.worklists.d_tp_children, v1.d_tp_children);
    ck("tp_octants", s.worklists.d_tp_octants, v1.d_tp_octants);
    ck("tp_up_src_boxes", s.worklists.d_tp_up_src_boxes, v1.d_tp_up_src_boxes);
    ck("tp_up_dst_boxes", s.worklists.d_tp_up_dst_boxes, v1.d_tp_up_dst_boxes);
    ck("tp_up_octants", s.worklists.d_tp_up_octants, v1.d_tp_up_octants);
    ck("pw_eval_box_flat", s.worklists.d_pw_eval_box_flat, v1.d_pw_eval_box_flat);
    ck("pw_form_box_flat", s.worklists.d_pw_form_box_flat, v1.d_pw_form_box_flat);
    ck("proxy_offsets_upward", s.scratch.d_proxy_offsets_upward, v1.d_proxy_offsets_upward);
    ck("proxy_offsets_downward", s.scratch.d_proxy_offsets_downward, v1.d_proxy_offsets_downward);
    ck("pw_out_offsets", s.scratch.d_pw_out_offsets, v1.d_pw_out_offsets);
    ck("pot_src_offsets", s.outputs.d_pot_src_offsets, v1.d_pot_src_offsets);
    ck("pot_trg_offsets", s.outputs.d_pot_trg_offsets, v1.d_pot_trg_offsets);
    ck("charge", s.particles.d_charge, v1.d_charge);
    ck("normal", s.particles.d_normal, v1.d_normal);
    ck("charge_outer", s.particles.d_charge_outer, v1.d_charge_outer);

    if (mism == 0)
        std::fprintf(stderr, "[GPU_V2] seam self-check: all %d buffers match V1\n", total);
    else
        std::fprintf(stderr, "[GPU_V2] seam self-check: %d/%d buffers differ from V1\n", mism, total);
}

} // namespace

template <typename Real, int DIM>
Tree<Real, DIM>::Tree(const sctl::Comm &comm, const pdmk_params &params, const sctl::Vector<Real> &r_src,
                      const sctl::Vector<Real> &charge, const sctl::Vector<Real> &normal,
                      const sctl::Vector<Real> &r_trg) {
    // The owned tree runs the GPU host precompute (and, for now, the V1 device
    // state we validate against).
    tree_ = std::make_unique<DMKPtTree<Real, DIM>>(comm, params, r_src, charge, normal, r_trg);

    state_ = std::make_unique<State<Real, DIM>>(to_build_inputs(*tree_));
    const long n_src = r_src.Dim() / DIM;
    const Real *charge_ptr = charge.Dim() ? &charge[0] : nullptr;
    const Real *normal_ptr = (params.kernel == DMK_STRESSLET && normal.Dim()) ? &normal[0] : nullptr;
    state_->upload_and_sort_charges(charge_ptr, normal_ptr, n_src);

    if (util::env_is_set("DMK_GPU_V2_CHECK") && tree_->cuda_shared_state_)
        seam_self_check(*state_, *tree_->cuda_shared_state_);
}

template <typename Real, int DIM>
void Tree<Real, DIM>::eval() {
    // Scaffold: delegate to the owned V1 pipeline. Replaced pass by pass.
    tree_->eval();

    // Transitional parity gate (removed at cutover): run the V2 direct pass and
    // compare its near-field output to the V1 direct oracle (already validated
    // == CPU). V1 direct output is live in cuda_direct_ctx_ after eval().
    if (util::env_is_set("DMK_GPU_V2_CHECK") && tree_->cuda_direct_ctx_) {
        pt::direct(*state_, state_->direct_stream.get());
        state_->direct_stream.sync();
        const double e_src = dev_rel_l2<Real>(state_->outputs.d_pot_direct_src.data(),
                                              tree_->cuda_direct_ctx_->device_pot_src(), state_->outputs.pot_src_size);
        const double e_trg = dev_rel_l2<Real>(state_->outputs.d_pot_direct_trg.data(),
                                              tree_->cuda_direct_ctx_->device_pot_trg(), state_->outputs.pot_trg_size);
        std::fprintf(stderr, "[GPU_V2] direct parity vs V1: rel_l2 src=%.3e trg=%.3e\n", e_src, e_trg);
    }
}

template <typename Real, int DIM>
void Tree<Real, DIM>::desort_potentials(Real *pot_src, Real *pot_trg) {
    tree_->desort_potentials(pot_src, pot_trg);
}

template <typename Real, int DIM>
void Tree<Real, DIM>::update_charges(const Real *charge, const Real *normal) {
    tree_->update_charges(charge, normal);
    state_->upload_and_sort_charges(charge, normal, tree_->r_src_sorted_owned.Dim() / DIM);
}

template class Tree<float, 2>;
template class Tree<float, 3>;
template class Tree<double, 2>;
template class Tree<double, 3>;

} // namespace dmk::cuda::pt
