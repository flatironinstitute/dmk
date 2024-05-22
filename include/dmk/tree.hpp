#ifndef TREE_HPP
#define TREE_HPP

#include <dmk.h>
#include <algorithm>
#include <numeric>
#include <sctl.hpp>
#include <vector>

template <typename T>
struct FourierData;

namespace dmk {

template <typename Real, int DIM>
struct DMKPtTree : public sctl::PtTree<Real, DIM> {
    sctl::Vector<bool> in_flag;
    sctl::Vector<bool> out_flag;
    std::vector<std::vector<int>> level_indices;
    sctl::Vector<int> src_counts_local;
    sctl::Vector<int> src_counts_global;
    std::vector<double> boxsize;
    sctl::Vector<Real> r_src_sorted;
    sctl::Vector<sctl::Long> r_src_cnt;
    std::vector<sctl::Long> r_src_offsets;
    sctl::Vector<Real> charge_sorted;
    sctl::Vector<sctl::Long> charge_cnt;
    std::vector<sctl::Long> charge_offsets;
    std::vector<Real> centers;
    std::vector<Real> scale_factors;

    sctl::Vector<Real> proxy_coeffs;

    DMKPtTree<Real, DIM>(const sctl::Comm &comm) : sctl::PtTree<Real, DIM>(comm){};

    int n_levels() const { return level_indices.size(); }
    int n_in() const { return std::accumulate(in_flag.begin(), in_flag.end(), 0); }
    int n_out() const { return std::accumulate(out_flag.begin(), out_flag.end(), 0); }
    int n_boxes() const { return this->GetNodeMID().Dim(); }
    void generate_metadata(int ns, int nd);
    Real *r_src_ptr(int i_node) { return &r_src_sorted[r_src_offsets[i_node]]; }
    Real *charge_ptr(int i_node) { return &charge_sorted[charge_offsets[i_node]]; }
    Real *center_ptr(int i_node) { return &centers[i_node * DIM]; }

    void build_proxy_charges(int n_mfm, int n_order, const std::vector<Real> &c2p);
    void downward_pass(const pdmk_params &params, int n_order, const FourierData<Real> &fourier_data);
};

} // namespace dmk

#endif
