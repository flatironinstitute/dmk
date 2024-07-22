#ifndef TREE_HPP
#define TREE_HPP

#include <algorithm>
#include <complex>
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

    sctl::Vector<Real> pot_src_sorted;
    sctl::Vector<sctl::Long> pot_src_cnt;
    std::vector<sctl::Long> pot_src_offsets;

    sctl::Vector<Real> pot_trg_sorted;
    sctl::Vector<sctl::Long> pot_trg_cnt;
    std::vector<sctl::Long> pot_trg_offsets;

    sctl::Vector<Real> charge_sorted;
    sctl::Vector<sctl::Long> charge_cnt;
    std::vector<sctl::Long> charge_offsets;

    sctl::Vector<Real> proxy_coeffs;
    sctl::Vector<sctl::Long> proxy_coeffs_offsets;
    sctl::Vector<Real> proxy_coeffs_downward;
    sctl::Vector<sctl::Long> proxy_coeffs_offsets_downward;

    sctl::Vector<std::complex<Real>> pw_in;
    sctl::Vector<sctl::Long> pw_in_offsets;
    sctl::Vector<std::complex<Real>> pw_out;
    sctl::Vector<sctl::Long> pw_out_offsets;

    sctl::Vector<int> form_pw_expansion;
    sctl::Vector<int> eval_pw_expansion;
    sctl::Vector<int> eval_tp_expansion;
    const pdmk_params params;
    const int n_digits;
    const int n_pw_max;
    const int n_order;
    int n_pw; // FIXME: Assigned well after construction, dangerous hack
    FourierData<Real> fourier_data;

    DMKPtTree(const sctl::Comm &comm, const pdmk_params &params_, const sctl::Vector<Real> &r_src,
              const sctl::Vector<Real> &r_trg, const sctl::Vector<Real> &charge);

    int n_levels() const { return level_indices.size(); }
    std::size_t n_boxes() const { return this->GetNodeMID().Dim(); }
    void generate_metadata();
    Real *r_src_ptr(int i_node) { return &r_src_sorted[r_src_offsets[i_node]]; }
    Real *r_trg_ptr(int i_node) { return &r_trg_sorted[r_trg_offsets[i_node]]; }
    Real *pot_src_ptr(int i_node) { return &pot_src_sorted[pot_src_offsets[i_node]]; }
    Real *pot_trg_ptr(int i_node) { return &pot_trg_sorted[pot_trg_offsets[i_node]]; }
    Real *charge_ptr(int i_node) { return &charge_sorted[charge_offsets[i_node]]; }
    Real *center_ptr(int i_node) { return &centers[i_node * DIM]; }
    Real *proxy_ptr_upward(int i_box) { return &proxy_coeffs[proxy_coeffs_offsets[i_box]]; }
    Real *proxy_ptr_downward(int i_box) { return &proxy_coeffs_downward[proxy_coeffs_offsets_downward[i_box]]; }
    std::complex<Real> *pw_in_ptr(int i_box) { return &pw_in[pw_in_offsets[i_box]]; }
    std::complex<Real> *pw_out_ptr(int i_box) { return &pw_out[pw_out_offsets[i_box]]; }

    void upward_pass(const sctl::Vector<Real> &c2p);
    void downward_pass(const sctl::Vector<Real> &c2p);
};

} // namespace dmk

#endif
