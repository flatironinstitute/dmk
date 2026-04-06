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

// Contact geometry between two adjacent boxes.
//
// Two boxes can share a face, edge, or corner. In each spatial dimension,
// the boxes either "touch" (share a boundary plane) or "overlap" (have a
// shared extent in that dimension).
//
// The contact feature between two adjacent boxes:
//
//   2D:
//     - Edge:   1 touch dim, 1 overlap dim → a line segment
//     - Corner: 2 touch dims               → a point
//
//   3D:
//     - Face:   1 touch dim,  2 overlap dims → a rectangle
//     - Edge:   2 touch dims, 1 overlap dim  → a line segment
//     - Corner: 3 touch dims                 → a point
//
//   In general: n_touch = DIM is always a corner,
//               n_touch = 1 is the codimension-1 feature (edge in 2D, face in 3D)
template <typename Real, int DIM>
struct ContactGeometry {
    int touch_dims[DIM];
    Real touch_coords[DIM];
    int overlap_dims[DIM];
    Real overlap_lo[DIM];
    Real overlap_hi[DIM];
    int n_touch = 0;
    int n_overlap = 0;
    Real d2max;

    inline ContactGeometry(const Real *corner_a, const Real *corner_b, Real size_a, Real size_b, Real d2max_)
        : d2max(d2max_) {
        for (int d = 0; d < DIM; d++) {
            Real a = corner_a[d], b = corner_b[d];
            Real sa = size_a, sb = size_b;
            if (a + sa == b) {
                touch_dims[n_touch] = d;
                touch_coords[n_touch] = b;
                n_touch++;
            } else if (b + sb == a) {
                touch_dims[n_touch] = d;
                touch_coords[n_touch] = a;
                n_touch++;
            } else {
                overlap_dims[n_overlap] = d;
                overlap_lo[n_overlap] = std::max(a, b);
                overlap_hi[n_overlap] = std::min(a + sa, b + sb);
                n_overlap++;
            }
        }
    }

    // To check if a particle is within interaction range of the target box,
    // we compute its distance to the nearest point on the contact feature.
    //   - In touch dimensions, the contact is a fixed coordinate (the shared plane),
    //     so distance is just |p - contact_coord|.
    //   - In overlap dimensions, the contact spans an interval [lo, hi]. The nearest
    //     point is obtained by clamping p to that interval. If p is inside the
    //     interval, its contribution to the distance is zero. If p is outside, the
    //     contribution is the distance to the nearest endpoint. This produces a rounded
    //     corner at the edge of the overlap intervals. half-diskorectangles,
    //     quarter-spherocylinders, faces with quarter-circle bevels, 1/8th spheres, 1/4
    //     circles. yadayada.
    inline Real dist2_to_contact(const Real *p) const {
        Real dist2 = 0;
        for (int t = 0; t < n_touch; t++) {
            Real delta = p[touch_dims[t]] - touch_coords[t];
            dist2 += delta * delta;
        }
        for (int t = 0; t < n_overlap; t++) {
            Real coord = p[overlap_dims[t]];
            Real clamped = std::min(std::max(coord, overlap_lo[t]), overlap_hi[t]);
            Real delta = coord - clamped;
            dist2 += delta * delta;
        }
        return dist2;
    }

    inline bool in_range(const Real *p) const { return dist2_to_contact(p) < d2max; }
};

// Filter source particles by distance to contact feature.
// Returns the number of particles that passed the filter.
// Filtered positions and charges are written contiguously into the provided buffers.
template <typename Real, int DIM>
int filter_sources(const ContactGeometry<Real, DIM> &geom, int n_src, const Real *r_src, const Real *charge,
                   int n_charge_components, Real *r_src_out, Real *charge_out) {
    int n_filtered = 0;
    for (int i = 0; i < n_src; ++i) {
        const Real *p = r_src + DIM * i;
        if (geom.in_range(p)) {
            for (int d = 0; d < DIM; ++d)
                r_src_out[DIM * n_filtered + d] = p[d];
            for (int j = 0; j < n_charge_components; ++j)
                charge_out[n_filtered * n_charge_components + j] = charge[i * n_charge_components + j];
            n_filtered++;
        }
    }
    return n_filtered;
}

// Filter evaluation target points by distance to contact feature.
// Returns the number of points that passed the filter.
// Filtered positions are written contiguously into r_trg_out.
// index_map[k] = original index of the k-th filtered point, used to scatter results back.
template <typename Real, int DIM>
int filter_targets(const ContactGeometry<Real, DIM> &geom, int n_trg, const Real *r_trg, Real *r_trg_out,
                   int *index_map) {
    int n_filtered = 0;
    for (int i = 0; i < n_trg; ++i) {
        const Real *p = r_trg + DIM * i;
        if (geom.in_range(p)) {
            for (int d = 0; d < DIM; ++d)
                r_trg_out[DIM * n_filtered + d] = p[d];
            index_map[n_filtered] = i;
            n_filtered++;
        }
    }
    return n_filtered;
}

// Scatter-add filtered potential values back to the original potential array.
template <typename Real>
void scatter_add_potential(const Real *pot_filtered, Real *pot, const int *index_map, int n_filtered,
                           int n_components) {
    for (int i = 0; i < n_filtered; ++i)
        for (int j = 0; j < n_components; ++j)
            pot[index_map[i] * n_components + j] += pot_filtered[i * n_components + j];
}

template <typename T, int DIM>
void multiply_kernelFT_cd2p(const sctl::Vector<T> &radialft, auto &&pwexp) {
    const int nd = pwexp.extent(DIM);
    const int nexp = radialft.Dim();
    ndview<std::complex<T>, 2> pwexp_flat({nexp, nd}, pwexp.data());

    ndview<const T, 1> radialft_view({nexp}, &radialft[0]);
    for (int ind = 0; ind < nd; ++ind) {
        ndview<std::complex<T>, 1> pwexp_view({nexp}, &pwexp_flat(0, ind));
        pwexp_view *= radialft_view;
    }
    // Real * complex is two multiplies and two adds
    const unsigned long n_flops = 4 * nd * nexp;
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, n_flops);
}

template <typename Real, int VecLen>
inline void shift_planewave_simd(int nexp, int nd, const Real *__restrict__ pw1, Real *__restrict__ pw2,
                                 const Real *__restrict__ shift_r, const Real *__restrict__ shift_i) {
    using Vec = sctl::Vec<Real, VecLen>;
    constexpr int N = VecLen;
    using dmk::util::complex_deinterleave;
    using dmk::util::complex_interleave;

    for (int ind = 0; ind < nd; ++ind) {
        const Real *s1 = pw1 + ind * nexp * 2;
        Real *s2 = pw2 + ind * nexp * 2;

        int i = 0;
        for (; i + N <= nexp; i += N) {
            Vec ar, ai;
            Vec lo1 = Vec::Load(s1 + 2 * i);
            Vec hi1 = Vec::Load(s1 + 2 * i + N);
            complex_deinterleave(lo1.get().v, hi1.get().v, ar.get().v, ai.get().v);

            Vec dr, di;
            Vec lo2 = Vec::Load(s2 + 2 * i);
            Vec hi2 = Vec::Load(s2 + 2 * i + N);
            complex_deinterleave(lo2.get().v, hi2.get().v, dr.get().v, di.get().v);

            Vec cr = Vec::Load(shift_r + i);
            Vec ci = Vec::Load(shift_i + i);

            dr = FMA(ar, cr, dr);
            dr = FMA(-ai, ci, dr);
            di = FMA(ar, ci, di);
            di = FMA(ai, cr, di);

            Vec out_lo, out_hi;
            complex_interleave(dr.get().v, di.get().v, out_lo.get().v, out_hi.get().v);
            out_lo.Store(s2 + 2 * i);
            out_hi.Store(s2 + 2 * i + N);
        }

        for (; i < nexp; ++i) {
            Real ar = s1[2 * i], ai = s1[2 * i + 1];
            Real cr = shift_r[i], ci = shift_i[i];
            s2[2 * i] += ar * cr - ai * ci;
            s2[2 * i + 1] += ar * ci + ai * cr;
        }
    }
}

template <typename Complex, int DIM>
inline void shift_planewave(const ndview<Complex, DIM + 1> &pwexp1_, ndview<Complex, DIM + 1> &pwexp2_,
                            const ndview<const Complex, 1> &wpwshift) {
    using Real = typename Complex::value_type;
    constexpr int VecLen = sctl::DefaultVecLen<Real>();

    const int nd = pwexp1_.extent(DIM);
    const int nexp = wpwshift.extent(0);
    const Real *shift_r = reinterpret_cast<const Real *>(wpwshift.data());
    const Real *shift_i = shift_r + nexp;

    const Real *pw1 = reinterpret_cast<const Real *>(pwexp1_.data());
    Real *pw2 = reinterpret_cast<Real *>(pwexp2_.data());

    shift_planewave_simd<Real, VecLen>(nexp, nd, pw1, pw2, shift_r, shift_i);
}


template <typename Real>
Real calc_log_windowed_kernel_value_at_zero(int dim, const FourierData<Real> &fourier_data, Real boxsize) {
    const Real psi0 = fourier_data.prolate0_fun.eval_val(0.0);
    const Real beta = fourier_data.beta();
    constexpr int n_quad = 100;
    std::array<Real, n_quad> xs, whts;
    legerts(1, n_quad, xs.data(), whts.data());
    for (int i = 0; i < n_quad; ++i) {
        xs[i] = 0.5 * (xs[i] + Real{1.0}) * beta / boxsize;
        whts[i] *= 0.5 * beta / boxsize;
    }

    const Real rl = boxsize * sqrt(dim * 1.0) * 2;
    const Real dfac = rl * std::log(rl);

    Real fval = 0.0;
    for (int i = 0; i < n_quad; ++i) {
        const Real xval = xs[i] * boxsize / beta;
        const Real fval0 = fourier_data.prolate0_fun.eval_val(xval);
        const Real z = rl * xs[i];
        const Real dj0 = util::cyl_bessel_j(0, z);
        const Real dj1 = util::cyl_bessel_j(1, z);
        const Real tker = -(1 - dj0) / (xs[i] * xs[i]) + dfac * dj1 / xs[i];
        const Real fhat = tker * fval0 / psi0;
        fval += fhat * whts[i] * xs[i];
    }

    return fval;
}

template <typename Real, int DIM>
Real get_self_interaction_constant(FourierData<Real> &fourier_data, dmk_ikernel kernel, int i_level, Real boxsize) {
    const double bsize = i_level == 0 ? 0.5 * boxsize : boxsize;
    const double w0 = [&]() -> Real {
        if (kernel == DMK_YUKAWA)
            return fourier_data.yukawa_windowed_kernel_value_at_zero(i_level);
        else if (kernel == DMK_LAPLACE) {
            const Real psi0 = fourier_data.prolate0_fun.eval_val(0.0);
            const auto c = fourier_data.prolate0_fun.intvals(fourier_data.beta());
            if constexpr (DIM == 2) {
                const auto log_windowed_kernel_at_zero =
                    calc_log_windowed_kernel_value_at_zero(DIM, fourier_data, Real{1.0});
                return log_windowed_kernel_at_zero - i_level * std::log(2.0);
            } else if constexpr (DIM == 3)
                return psi0 / (c[0] * bsize);
            else
                throw std::runtime_error("Unsupported kernel DMK_LAPLACE, DIM = " + std::to_string(DIM));
        } else if (kernel == DMK_SQRT_LAPLACE) {
            const Real psi0 = fourier_data.prolate0_fun.eval_val(0.0);
            const auto c = fourier_data.prolate0_fun.intvals(fourier_data.beta());
            if constexpr (DIM == 2)
                return psi0 / (c[0] * bsize);
            if constexpr (DIM == 3)
                return psi0 / (2 * c[1] * bsize * bsize);

        } else
            throw std::runtime_error("Unsupported kernel");
    }();

    return w0;
}

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

    sctl::Vector<Real> grad_src_sorted;
    sctl::Vector<sctl::Long> grad_src_cnt;
    sctl::Vector<sctl::Long> grad_src_offsets;

    sctl::Vector<Real> pot_trg_sorted;
    sctl::Vector<sctl::Long> pot_trg_cnt;
    sctl::Vector<sctl::Long> pot_trg_offsets;

    sctl::Vector<Real> grad_trg_sorted;
    sctl::Vector<sctl::Long> grad_trg_cnt;
    sctl::Vector<sctl::Long> grad_trg_offsets;

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
    const int kernel_input_dim;
    const int kernel_output_dim_src;
    const int kernel_output_dim_trg;
    const int kernel_output_dim_max;
    const int n_tables;
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

    // Subtasks for downward pass
    void form_outgoing_expansions();
    void form_incoming_expansions(const sctl::Vector<int> &boxes, const sctl::Vector<std::complex<Real>> &wpwshift);
    void form_local_expansions(const sctl::Vector<int> &boxes, Real boxsize,
                               const ndview<std::complex<Real>, 2> &pw2poly_view, const sctl::Vector<Real> &p2c);
    void form_eval_expansions(const sctl::Vector<int> &boxes, const sctl::Vector<std::complex<Real>> &wpwshift,
                              Real boxsize, const ndview<std::complex<Real>, 2> &pw2poly_view,
                              const sctl::Vector<Real> &p2c);
    void evaluate_direct_interactions(const Real *r_src_t, const Real *r_trg_t);

    // User calls
    int update_charges(const Real *charge, const Real *normal, const Real *dipole_str);

    // Internal data accessors
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
        return ndview<Real, 2>({kernel_output_dim_src, src_counts_with_halo[i_node]}, pot_src_ptr(i_node));
    }

    Real *pot_trg_ptr(int i_node) {
        assert(trg_counts_owned[i_node]);
        return &pot_trg_sorted[pot_trg_offsets[i_node]];
    }
    ndview<Real, 2> pot_trg_view(int i_node) {
        return ndview<Real, 2>({kernel_output_dim_trg, trg_counts_owned[i_node]}, pot_trg_ptr(i_node));
    }

    Real *charge_owned_ptr(int i_node) {
        assert(src_counts_owned[i_node]);
        return &charge_sorted_owned[charge_offsets_owned[i_node]];
    }
    ndview<Real, 2> charge_owned_view(int i_node) {
        return ndview<Real, 2>({kernel_input_dim, src_counts_owned[i_node]}, charge_owned_ptr(i_node));
    }

    Real *charge_with_halo_ptr(int i_node) {
        assert(src_counts_with_halo[i_node]);
        return &charge_sorted_with_halo[charge_offsets_with_halo[i_node]];
    }
    ndview<Real, 2> charge_with_halo_view(int i_node) {
        return ndview<Real, 2>({kernel_input_dim, src_counts_with_halo[i_node]}, charge_with_halo_ptr(i_node));
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
            return ndview<Real, DIM + 1>({n_order, n_order, n_tables}, proxy_ptr_upward(i_box));
        else if constexpr (DIM == 3)
            return ndview<Real, DIM + 1>({n_order, n_order, n_order, n_tables}, proxy_ptr_upward(i_box));
        else
            static_assert(dmk::util::always_false<Real>, "Invalid DIM supplied");
    }

    Real *proxy_ptr_downward(int i_box) {
        assert(proxy_coeffs_offsets_downward[i_box] != -1);
        return &proxy_coeffs_downward[proxy_coeffs_offsets_downward[i_box]];
    }
    ndview<Real, DIM + 1> proxy_view_downward(int i_box) {
        if constexpr (DIM == 2)
            return ndview<Real, DIM + 1>({n_order, n_order, n_tables}, proxy_ptr_downward(i_box));
        else if constexpr (DIM == 3)
            return ndview<Real, DIM + 1>({n_order, n_order, n_order, n_tables}, proxy_ptr_downward(i_box));
        else
            static_assert(dmk::util::always_false<Real>, "Invalid DIM supplied");
    }

    std::complex<Real> *pw_out_ptr(int i_box) { return &pw_out[pw_out_offsets[i_box]]; }
    const std::complex<Real> *pw_out_ptr(int i_box) const { return &pw_out[pw_out_offsets[i_box]]; }
    ndview<std::complex<Real>, DIM + 1> pw_out_view(int i_box) {
        if constexpr (DIM == 2)
            return ndview<std::complex<Real>, DIM + 1>({n_pw, (n_pw + 1) / 2, n_tables}, pw_out_ptr(i_box));
        else if constexpr (DIM == 3)
            return ndview<std::complex<Real>, DIM + 1>({n_pw, n_pw, (n_pw + 1) / 2, n_tables}, pw_out_ptr(i_box));
        else
            static_assert(dmk::util::always_false<std::complex<Real>>, "Invalid DIM supplied");
    }

    void dump() const;
    void eval();
    void eval_pq();
    void upward_pass();
    void downward_pass();
    void desort_potentials(Real *pot_src, Real *pot_trg);

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
    std::vector<direct_evaluator_func<Real>> evaluator_by_level_src;
    std::vector<direct_evaluator_func<Real>> evaluator_by_level_trg;
    const sctl::Comm comm_;
    bool debug_omit_pw = false;
    bool debug_omit_direct = false;
    bool debug_dump_tree = false;
};

} // namespace dmk

#endif
