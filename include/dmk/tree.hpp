#ifndef TREE_HPP
#define TREE_HPP

#include <algorithm>
#include <dmk.h>
#include <numeric>
#include <sctl.hpp>
#include <vector>

namespace dmk {
template <typename T>
struct FourierData;

template <typename Real, int DIM>
struct DMKPtTree : public sctl::PtTree<Real, DIM> {
    std::vector<std::vector<int>> level_indices;
    std::vector<double> boxsize;
    std::vector<Real> centers;

    sctl::Vector<int> src_counts_local;
    sctl::Vector<int> src_counts_global;

    sctl::Vector<int> trg_counts_local;
    sctl::Vector<int> trg_counts_global;

    sctl::Vector<Real> r_src_sorted;
    sctl::Vector<sctl::Long> r_src_cnt;
    std::vector<sctl::Long> r_src_offsets;

    sctl::Vector<Real> r_trg_sorted;
    sctl::Vector<sctl::Long> r_trg_cnt;
    std::vector<sctl::Long> r_trg_offsets;

    sctl::Vector<Real> pot_sorted;
    sctl::Vector<sctl::Long> pot_cnt;
    std::vector<sctl::Long> pot_offsets;

    sctl::Vector<Real> charge_sorted;
    sctl::Vector<sctl::Long> charge_cnt;
    std::vector<sctl::Long> charge_offsets;

    sctl::Vector<Real> proxy_coeffs;
    sctl::Vector<Real> proxy_coeffs_downward;

    sctl::Vector<int> form_pw_expansion;
    sctl::Vector<int> eval_pw_expansion;
    sctl::Vector<int> form_tp_expansion;
    sctl::Vector<int> eval_tp_expansion;
    const int n_order;

    DMKPtTree(const sctl::Comm &comm, int n_order_) : sctl::PtTree<Real, DIM>(comm), n_order(n_order_){};

    std::size_t n_levels() const { return level_indices.size(); }
    std::size_t n_boxes() const { return this->GetNodeMID().Dim(); }
    void generate_metadata(int ns, int nd);
    Real *r_src_ptr(int i_node) { return &r_src_sorted[r_src_offsets[i_node]]; }
    Real *r_trg_ptr(int i_node) { return &r_trg_sorted[r_trg_offsets[i_node]]; }
    Real *pot_ptr(int i_node) { return &pot_sorted[pot_offsets[i_node]]; }
    Real *charge_ptr(int i_node) { return &charge_sorted[charge_offsets[i_node]]; }
    Real *center_ptr(int i_node) { return &centers[i_node * DIM]; }
    Real *proxy_ptr_upward(int i_box) { return &proxy_coeffs[i_box * sctl::pow<DIM>(n_order)]; }
    Real *proxy_ptr_downward(int i_box) { return &proxy_coeffs_downward[i_box * sctl::pow<DIM>(n_order)]; }

    void upward_pass(int n_mfm, const sctl::Vector<Real> &c2p);
    void downward_pass(const pdmk_params &params, FourierData<Real> &fourier_data, const sctl::Vector<Real> &c2p);
};

} // namespace dmk

#endif
