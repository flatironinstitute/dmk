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

    DMKPtTree(const sctl::Comm &comm) : sctl::PtTree<Real, DIM>(comm){};

    int n_levels() const { return level_indices.size(); }
    int n_boxes() const { return this->GetNodeMID().Dim(); }
    void generate_metadata(int ns, int nd);
    Real *r_src_ptr(int i_node) { return &r_src_sorted[r_src_offsets[i_node]]; }
    Real *r_trg_ptr(int i_node) { return &r_trg_sorted[r_trg_offsets[i_node]]; }
    Real *pot_ptr(int i_node) { return &pot_sorted[pot_offsets[i_node]]; }
    Real *charge_ptr(int i_node) { return &charge_sorted[charge_offsets[i_node]]; }
    Real *center_ptr(int i_node) { return &centers[i_node * DIM]; }

    void build_proxy_charges(int n_mfm, int n_order, const sctl::Vector<Real> &c2p);
    void downward_pass(const pdmk_params &params, int n_order, const FourierData<Real> &fourier_data,
                       const sctl::Vector<Real> &c2p);
};

} // namespace dmk

#endif
