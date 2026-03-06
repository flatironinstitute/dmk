#ifndef TREE_HPP
#define TREE_HPP

#include <complex>
#include <dmk.h>
#include <dmk/types.hpp>
#include <dmk/util.hpp>
#include <nda/nda.hpp>
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

    sctl::Vector<int> src_counts_owned;
    sctl::Vector<int> trg_counts_owned;
    sctl::Vector<int> src_counts_with_halo;

    sctl::Vector<Real> r_src_sorted_with_halo;
    sctl::Vector<sctl::Long> r_src_cnt_with_halo;
    sctl::Vector<sctl::Long> r_src_offsets_with_halo;

    sctl::Vector<Real> r_src_sorted_owned;
    sctl::Vector<sctl::Long> r_src_cnt_owned;
    sctl::Vector<sctl::Long> r_src_offsets_owned;

    sctl::Vector<Real> r_trg_sorted_owned;
    sctl::Vector<sctl::Long> r_trg_cnt_owned;
    sctl::Vector<sctl::Long> r_trg_offsets_owned;

    sctl::Vector<Real> pot_src_sorted;
    sctl::Vector<sctl::Long> pot_src_cnt;
    sctl::Vector<sctl::Long> pot_src_offsets;

    sctl::Vector<Real> pot_trg_sorted;
    sctl::Vector<sctl::Long> pot_trg_cnt;
    sctl::Vector<sctl::Long> pot_trg_offsets;

    sctl::Vector<Real> charge_sorted_owned;
    sctl::Vector<sctl::Long> charge_cnt_owned;
    sctl::Vector<sctl::Long> charge_offsets_owned;

    sctl::Vector<Real> charge_sorted_with_halo;
    sctl::Vector<sctl::Long> charge_cnt_with_halo;
    sctl::Vector<sctl::Long> charge_offsets_with_halo;

    sctl::Vector<Real> proxy_coeffs_upward;
    sctl::Vector<sctl::Long> proxy_coeffs_offsets;
    sctl::Vector<Real> proxy_coeffs_downward;
    sctl::Vector<sctl::Long> proxy_coeffs_offsets_downward;

    sctl::Vector<std::complex<Real>> pw_out;
    sctl::Vector<sctl::Long> pw_out_offsets;

    sctl::Vector<bool> ifpwexp;
    sctl::Vector<bool> iftensprodeval;

    // FIXME: I really hate these.
    ndamatrix<Real> r_src_t, r_trg_t;
    std::vector<int> direct_work;

    sctl::Vector<bool> has_proxy_from_children;
    struct C2PWork {
        int src_box;
        int center_box;
        int level;
    };
    std::vector<C2PWork> charge2proxy_work;

    struct LevelFourierData {
        sctl::Vector<std::complex<Real>> poly2pw;
        sctl::Vector<std::complex<Real>> pw2poly;
        sctl::Vector<Real> radialft;
        sctl::Vector<std::complex<Real>> wpwshift;
    };
    std::vector<LevelFourierData> difference_fourier_data; // one per level
    LevelFourierData window_fourier_data;

    sctl::Vector<bool> is_global_leaf;
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

    // Metadata generation subroutines
    void compute_data_offsets();
    void compute_level_indices_and_boxsizes();
    void compute_box_centers();
    void accumulate_subtree_counts();
    void gather_owned_source_positions();
    void broadcast_global_leaf_status();
    void compute_proxy_expansion_flags();
    void compute_proxy_evaluation_flags();
    void build_plane_wave_interaction_lists();
    void build_direct_interaction_lists();
    void build_upward_pass_work_lists();
    void allocate_proxy_coefficients();
    void precompute_fourier_data();
    void generate_metadata();

    void init_planewave_data();

    void form_outgoing_expansions(const sctl::Vector<int> &boxes, const ndview<std::complex<Real>, 2> &poly2pw_view,
                                  const sctl::Vector<Real> &radialft);

    void form_incoming_expansions(const sctl::Vector<int> &boxes, const sctl::Vector<std::complex<Real>> &wpwshift);

    void form_local_expansions(const sctl::Vector<int> &boxes, Real boxsize,
                               const ndview<std::complex<Real>, 2> &pw2poly_view, const sctl::Vector<Real> &p2c);

    void form_eval_expansions(const sctl::Vector<int> &boxes, const sctl::Vector<std::complex<Real>> &wpwshift,
                              Real boxsize, const ndview<std::complex<Real>, 2> &pw2poly_view,
                              const sctl::Vector<Real> &p2c);

    void evaluate_direct_interactions(const Real *r_src_t, const Real *r_trg_t);

    std::span<const int> list1(int i_box) const { return std::span<const int>(list1_[i_box].data(), nlist1_[i_box]); }
    std::span<const int> listpw(int i_box) const {
        return std::span<const int>(listpw_[i_box].data(), nlistpw_[i_box]);
    }

    Real *r_src_with_halo_ptr(int i_node) {
        assert(src_counts_with_halo[i_node]);
        return &r_src_sorted_with_halo[r_src_offsets_with_halo[i_node]];
    }
    ndview<Real, 2> r_src_with_halo_view(int i_node) {
        return ndview<Real, 2>({DIM, src_counts_with_halo[i_node]}, r_src_with_halo_ptr(i_node));
    }

    Real *r_src_owned_ptr(int i_node) {
        assert(src_counts_owned[i_node]);
        return &r_src_sorted_owned[r_src_offsets_owned[i_node]];
    }
    ndview<Real, 2> r_src_owned_view(int i_node) {
        return ndview<Real, 2>({DIM, src_counts_owned[i_node]}, r_src_owned_ptr(i_node));
    }

    Real *r_trg_owned_ptr(int i_node) {
        if (trg_counts_owned[i_node] == 0)
            return nullptr;
        return &r_trg_sorted_owned[r_trg_offsets_owned[i_node]];
    }
    ndview<Real, 2> r_trg_owned_view(int i_node) {
        return ndview<Real, 2>({DIM, trg_counts_owned[i_node]}, r_trg_owned_ptr(i_node));
    }

    Real *pot_src_ptr(int i_node) {
        assert(src_counts_with_halo[i_node]);
        return &pot_src_sorted[pot_src_offsets[i_node]];
    }
    ndview<Real, 2> pot_src_view(int i_node) {
        return ndview<Real, 2>({params.n_mfm, src_counts_with_halo[i_node]}, pot_src_ptr(i_node));
    }

    Real *pot_trg_ptr(int i_node) {
        assert(trg_counts_owned[i_node]);
        return &pot_trg_sorted[pot_trg_offsets[i_node]];
    }
    ndview<Real, 2> pot_trg_view(int i_node) {
        return ndview<Real, 2>({params.n_mfm, trg_counts_owned[i_node]}, pot_trg_ptr(i_node));
    }

    Real *charge_owned_ptr(int i_node) {
        assert(src_counts_owned[i_node]);
        return &charge_sorted_owned[charge_offsets_owned[i_node]];
    }
    ndview<Real, 2> charge_owned_view(int i_node) {
        return ndview<Real, 2>({params.n_mfm, src_counts_owned[i_node]}, charge_owned_ptr(i_node));
    }

    Real *charge_with_halo_ptr(int i_node) {
        assert(src_counts_with_halo[i_node]);
        return &charge_sorted_with_halo[charge_offsets_with_halo[i_node]];
    }
    ndview<Real, 2> charge_with_halo_view(int i_node) {
        return ndview<Real, 2>({params.n_mfm, src_counts_with_halo[i_node]}, charge_with_halo_ptr(i_node));
    }

    Real *center_ptr(int i_node) { return &centers[i_node * DIM]; }
    const Real *center_ptr(int i_node) const { return &centers[i_node * DIM]; }
    ndview<Real, 1> center_view(int i_node) { return ndview<Real, 1>({DIM}, center_ptr(i_node)); }
    ndview<const Real, 1> center_view(int i_node) const { return ndview<const Real, 1>({DIM}, center_ptr(i_node)); }

    Real *proxy_ptr_upward(int i_box) {
        assert(proxy_coeffs_offsets[i_box] != -1);
        return &proxy_coeffs_upward[proxy_coeffs_offsets[i_box]];
    }
    ndview<Real, DIM + 1> proxy_view_upward(int i_box) {
        if constexpr (DIM == 2)
            return ndview<Real, DIM + 1>({n_order, n_order, params.n_mfm}, proxy_ptr_upward(i_box));
        else if constexpr (DIM == 3)
            return ndview<Real, DIM + 1>({n_order, n_order, n_order, params.n_mfm}, proxy_ptr_upward(i_box));
        else
            static_assert(dmk::util::always_false<Real>, "Invalid DIM supplied");
    }

    Real *proxy_ptr_downward(int i_box) {
        assert(proxy_coeffs_offsets_downward[i_box] != -1);
        return &proxy_coeffs_downward[proxy_coeffs_offsets_downward[i_box]];
    }
    ndview<Real, DIM + 1> proxy_view_downward(int i_box) {
        if constexpr (DIM == 2)
            return ndview<Real, DIM + 1>({n_order, n_order, params.n_mfm}, proxy_ptr_downward(i_box));
        else if constexpr (DIM == 3)
            return ndview<Real, DIM + 1>({n_order, n_order, n_order, params.n_mfm}, proxy_ptr_downward(i_box));
        else
            static_assert(dmk::util::always_false<Real>, "Invalid DIM supplied");
    }

    std::complex<Real> *pw_out_ptr(int i_box) { return &pw_out[pw_out_offsets[i_box]]; }
    const std::complex<Real> *pw_out_ptr(int i_box) const { return &pw_out[pw_out_offsets[i_box]]; }
    ndview<std::complex<Real>, DIM + 1> pw_out_view(int i_box) {
        if constexpr (DIM == 2)
            return ndview<std::complex<Real>, DIM + 1>({n_pw, (n_pw + 1) / 2, params.n_mfm}, pw_out_ptr(i_box));
        else if constexpr (DIM == 3)
            return ndview<std::complex<Real>, DIM + 1>({n_pw, n_pw, (n_pw + 1) / 2, params.n_mfm}, pw_out_ptr(i_box));
        else
            static_assert(dmk::util::always_false<std::complex<Real>>, "Invalid DIM supplied");
    }

    void dump() const;
    void upward_pass();
    void downward_pass();

  private:
    static constexpr int nlist1_max_ = sctl::pow<DIM>(4) - sctl::pow<DIM>(2) + 1;
    // list1 contains boxes that are neighbors for direct interaction
    std::vector<std::array<int, nlist1_max_>> list1_;
    std::vector<int> nlist1_;

    static constexpr int nlistpw_max_ = sctl::pow<DIM>(3);
    // listpw_ contains source boxes in the pw interaction
    std::vector<std::array<int, nlistpw_max_>> listpw_;
    std::vector<int> nlistpw_;

    // If proxy_view_downward(i_box) has been zeroed yet.
    std::vector<int> proxy_down_zeroed;

    sctl::Vector<sctl::Vector<Real>> workspaces_;
    direct_evaluator_func<Real> evaluator;
    const sctl::Comm comm_;
    bool debug_omit_pw = false;
    bool debug_omit_direct = false;
    bool debug_dump_tree = false;
};

} // namespace dmk

#endif
