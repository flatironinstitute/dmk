#ifndef TREE_HPP
#define TREE_HPP

#include <complex>
#include <dmk.h>
#include <dmk/types.hpp>
#include <dmk/util.hpp>
#include <sctl.hpp>
#include <span>

namespace dmk {
template <typename T>
struct FourierData;

template <typename Real, int DIM>
struct DMKPtTree : public sctl::PtTree<Real, DIM> {
    sctl::Vector<sctl::Vector<int>> level_indices;
    sctl::Vector<Real> boxsize;
    sctl::Vector<Real> centers;

    sctl::Vector<int> src_counts_local;
    sctl::Vector<int> trg_counts_local;

    sctl::Vector<Real> r_src_sorted;
    sctl::Vector<sctl::Long> r_src_cnt;
    sctl::Vector<sctl::Long> r_src_offsets;

    sctl::Vector<Real> r_trg_sorted;
    sctl::Vector<sctl::Long> r_trg_cnt;
    sctl::Vector<sctl::Long> r_trg_offsets;

    sctl::Vector<Real> pot_src_sorted;
    sctl::Vector<sctl::Long> pot_src_cnt;
    sctl::Vector<sctl::Long> pot_src_offsets;

    sctl::Vector<Real> pot_trg_sorted;
    sctl::Vector<sctl::Long> pot_trg_cnt;
    sctl::Vector<sctl::Long> pot_trg_offsets;

    sctl::Vector<Real> charge_sorted;
    sctl::Vector<sctl::Long> charge_cnt;
    sctl::Vector<sctl::Long> charge_offsets;

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
    sctl::Vector<Real> c2p;
    sctl::Vector<Real> p2c;

    DMKPtTree(const sctl::Comm &comm, const pdmk_params &params_, const sctl::Vector<Real> &r_src,
              const sctl::Vector<Real> &r_trg, const sctl::Vector<Real> &charge);

    int n_levels() const { return level_indices.Dim(); }
    std::size_t n_boxes() const { return this->GetNodeMID().Dim(); }
    void generate_metadata();
    void init_planewave_data();

    void form_outgoing_expansions(const sctl::Vector<int> &boxes,
                                  const ndview<const std::complex<Real>, 2> &poly2pw_view,
                                  const sctl::Vector<Real> &radialft);

    void form_incoming_expansions(const sctl::Vector<int> &boxes, const sctl::Vector<std::complex<Real>> &wpwshift);

    void form_local_expansions(const sctl::Vector<int> &boxes, Real boxsize,
                               const ndview<const std::complex<Real>, 2> &pw2poly_view, const sctl::Vector<Real> &p2c);

    void evaluate_direct_interactions(int i_level, const Real *r_src_t, const Real *r_trg_t);

    std::span<const int> direct_neighbs(int i_box) const {
        return std::span<const int>(direct_neighbs_[i_box].data(), n_direct_neighbs_[i_box]);
    }

    Real *r_src_ptr(int i_node) {
        assert(src_counts_local[i_node]);
        return &r_src_sorted[r_src_offsets[i_node]];
    }
    const Real *r_src_ptr(int i_node) const {
        assert(src_counts_local[i_node]);
        return &r_src_sorted[r_src_offsets[i_node]];
    }
    ndview<Real, 2> r_src_view(int i_node) { return ndview<Real, 2>(r_src_ptr(i_node), DIM, src_counts_local[i_node]); }
    ndview<const Real, 2> r_src_view(int i_node) const {
        return ndview<const Real, 2>(r_src_ptr(i_node), DIM, src_counts_local[i_node]);
    }

    Real *r_trg_ptr(int i_node) {
        if (trg_counts_local[i_node] == 0)
            return nullptr;
        return &r_trg_sorted[r_trg_offsets[i_node]];
    }
    const Real *r_trg_ptr(int i_node) const {
        if (trg_counts_local[i_node] == 0)
            return nullptr;
        return &r_trg_sorted[r_trg_offsets[i_node]];
    }
    ndview<Real, 2> r_trg_view(int i_node) { return ndview<Real, 2>(r_trg_ptr(i_node), DIM, trg_counts_local[i_node]); }
    ndview<const Real, 2> r_trg_view(int i_node) const {
        return ndview<const Real, 2>(r_trg_ptr(i_node), DIM, trg_counts_local[i_node]);
    }

    Real *pot_src_ptr(int i_node) {
        assert(src_counts_local[i_node]);
        return &pot_src_sorted[pot_src_offsets[i_node]];
    }
    const Real *pot_src_ptr(int i_node) const {
        assert(src_counts_local[i_node]);
        return &pot_src_sorted[pot_src_offsets[i_node]];
    }
    ndview<Real, 2> pot_src_view(int i_node) {
        return ndview<Real, 2>(pot_src_ptr(i_node), params.n_mfm, src_counts_local[i_node]);
    }
    ndview<const Real, 2> pot_src_view(int i_node) const {
        return ndview<const Real, 2>(pot_src_ptr(i_node), params.n_mfm, src_counts_local[i_node]);
    }

    Real *pot_trg_ptr(int i_node) {
        assert(trg_counts_local[i_node]);
        return &pot_trg_sorted[pot_trg_offsets[i_node]];
    }
    const Real *pot_trg_ptr(int i_node) const {
        assert(trg_counts_local[i_node]);
        return &pot_trg_sorted[pot_trg_offsets[i_node]];
    }
    ndview<Real, 2> pot_trg_view(int i_node) {
        return ndview<Real, 2>(pot_trg_ptr(i_node), params.n_mfm, trg_counts_local[i_node]);
    }
    ndview<const Real, 2> pot_trg_view(int i_node) const {
        return ndview<const Real, 2>(pot_trg_ptr(i_node), params.n_mfm, trg_counts_local[i_node]);
    }

    Real *charge_ptr(int i_node) {
        assert(src_counts_local[i_node]);
        return &charge_sorted[charge_offsets[i_node]];
    }
    const Real *charge_ptr(int i_node) const {
        assert(src_counts_local[i_node]);
        return &charge_sorted[charge_offsets[i_node]];
    }
    ndview<Real, 2> charge_view(int i_node) {
        return ndview<Real, 2>(charge_ptr(i_node), params.n_mfm, src_counts_local[i_node]);
    }
    ndview<const Real, 2> charge_view(int i_node) const {
        return ndview<const Real, 2>(charge_ptr(i_node), params.n_mfm, src_counts_local[i_node]);
    }

    Real *center_ptr(int i_node) { return &centers[i_node * DIM]; }
    const Real *center_ptr(int i_node) const { return &centers[i_node * DIM]; }
    ndview<Real, 1> center_view(int i_node) { return ndview<Real, 1>(center_ptr(i_node), DIM); }
    ndview<const Real, 1> center_view(int i_node) const { return ndview<const Real, 1>(center_ptr(i_node), DIM); }

    Real *proxy_ptr_upward(int i_box) {
        assert(proxy_coeffs_offsets[i_box] != -1);
        return &proxy_coeffs[proxy_coeffs_offsets[i_box]];
    }
    const Real *proxy_ptr_upward(int i_box) const {
        assert(proxy_coeffs_offsets[i_box] != -1);
        return &proxy_coeffs[proxy_coeffs_offsets[i_box]];
    }
    ndview<Real, DIM + 1> proxy_view_upward(int i_box) {
        if constexpr (DIM == 2)
            return ndview<Real, DIM + 1>(proxy_ptr_upward(i_box), n_order, n_order, params.n_mfm);
        else if constexpr (DIM == 3)
            return ndview<Real, DIM + 1>(proxy_ptr_upward(i_box), n_order, n_order, n_order, params.n_mfm);
        else
            static_assert(dmk::util::always_false<Real>, "Invalid DIM supplied");
    }
    ndview<const Real, DIM + 1> proxy_view_upward(int i_box) const {
        if constexpr (DIM == 2)
            return ndview<const Real, DIM + 1>(proxy_ptr_upward(i_box), n_order, n_order, params.n_mfm);
        else if constexpr (DIM == 3)
            return ndview<const Real, DIM + 1>(proxy_ptr_upward(i_box), n_order, n_order, n_order, params.n_mfm);
        else
            static_assert(dmk::util::always_false<Real>, "Invalid DIM supplied");
    }

    Real *proxy_ptr_downward(int i_box) {
        assert(proxy_coeffs_offsets_downward[i_box] != -1);
        return &proxy_coeffs_downward[proxy_coeffs_offsets_downward[i_box]];
    }
    const Real *proxy_ptr_downward(int i_box) const {
        assert(proxy_coeffs_offsets_downward[i_box] != -1);
        return &proxy_coeffs_downward[proxy_coeffs_offsets_downward[i_box]];
    }
    ndview<Real, DIM + 1> proxy_view_downward(int i_box) {
        if constexpr (DIM == 2)
            return ndview<Real, DIM + 1>(proxy_ptr_downward(i_box), n_order, n_order, params.n_mfm);
        else if constexpr (DIM == 3)
            return ndview<Real, DIM + 1>(proxy_ptr_downward(i_box), n_order, n_order, n_order, params.n_mfm);
        else
            static_assert(dmk::util::always_false<Real>, "Invalid DIM supplied");
    }
    ndview<const Real, DIM + 1> proxy_view_downward(int i_box) const {
        if constexpr (DIM == 2)
            return ndview<const Real, DIM + 1>(proxy_ptr_downward(i_box), n_order, n_order, params.n_mfm);
        else if constexpr (DIM == 3)
            return ndview<const Real, DIM + 1>(proxy_ptr_downward(i_box), n_order, n_order, n_order, params.n_mfm);
        else
            static_assert(dmk::util::always_false<Real>, "Invalid DIM supplied");
    }

    std::complex<Real> *pw_in_ptr(int i_box) { return &pw_in[pw_in_offsets[i_box]]; }
    const std::complex<Real> *pw_in_ptr(int i_box) const { return &pw_in[pw_in_offsets[i_box]]; }
    ndview<std::complex<Real>, DIM + 1> pw_in_view(int i_box) {
        if constexpr (DIM == 2)
            return ndview<std::complex<Real>, DIM + 1>(pw_in_ptr(i_box), n_pw, (n_pw + 1) / 2, params.n_mfm);
        else if constexpr (DIM == 3)
            return ndview<std::complex<Real>, DIM + 1>(pw_in_ptr(i_box), n_pw, n_pw, (n_pw + 1) / 2, params.n_mfm);
        else
            static_assert(dmk::util::always_false<std::complex<Real>>, "Invalid DIM supplied");
    }
    ndview<const std::complex<Real>, DIM + 1> pw_in_view(int i_box) const {
        if constexpr (DIM == 2)
            return ndview<const std::complex<Real>, DIM + 1>(pw_in_ptr(i_box), n_pw, (n_pw + 1) / 2, params.n_mfm);
        else if constexpr (DIM == 3)
            return ndview<const std::complex<Real>, DIM + 1>(pw_in_ptr(i_box), n_pw, n_pw, (n_pw + 1) / 2,
                                                             params.n_mfm);
        else
            static_assert(dmk::util::always_false<std::complex<Real>>, "Invalid DIM supplied");
    }

    std::complex<Real> *pw_out_ptr(int i_box) { return &pw_out[pw_out_offsets[i_box]]; }
    const std::complex<Real> *pw_out_ptr(int i_box) const { return &pw_out[pw_out_offsets[i_box]]; }
    ndview<std::complex<Real>, DIM + 1> pw_out_view(int i_box) {
        if constexpr (DIM == 2)
            return ndview<std::complex<Real>, DIM + 1>(pw_out_ptr(i_box), n_pw, (n_pw + 1) / 2, params.n_mfm);
        else if constexpr (DIM == 3)
            return ndview<std::complex<Real>, DIM + 1>(pw_out_ptr(i_box), n_pw, n_pw, (n_pw + 1) / 2, params.n_mfm);
        else
            static_assert(dmk::util::always_false<std::complex<Real>>, "Invalid DIM supplied");
    }
    ndview<const std::complex<Real>, DIM + 1> pw_out_view(int i_box) const {
        if constexpr (DIM == 2)
            return ndview<const std::complex<Real>, DIM + 1>(pw_out_ptr(i_box), n_pw, (n_pw + 1) / 2, params.n_mfm);
        else if constexpr (DIM == 3)
            return ndview<const std::complex<Real>, DIM + 1>(pw_out_ptr(i_box), n_pw, n_pw, (n_pw + 1) / 2,
                                                             params.n_mfm);
        else
            static_assert(dmk::util::always_false<std::complex<Real>>, "Invalid DIM supplied");
    }

    void upward_pass();
    void downward_pass();

  private:
    sctl::Vector<std::array<int, sctl::pow<DIM>(3)>> direct_neighbs_;
    sctl::Vector<int> n_direct_neighbs_;
    const sctl::Comm comm_;
};

} // namespace dmk

#endif
