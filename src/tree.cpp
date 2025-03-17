#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/direct.hpp>
#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/logger.h>
#include <dmk/planewave.hpp>
#include <dmk/prolate0_fun.hpp>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/tree.hpp>
#include <dmk/types.hpp>
#include <dmk/util.hpp>
#include <doctest/extensions/doctest_mpi.h>
#include <mpi.h>
#include <omp.h>
#include <stdexcept>
#include <unistd.h>

namespace dmk {

std::pair<int, int> get_pwmax_and_poly_order(int dim, int ndigits, dmk_ikernel kernel) {
    // clang-format off
    if (kernel == DMK_SQRT_LAPLACE && dim == 3) {
        if (ndigits <= 3) return {13, 9};
        if (ndigits <= 6) return {27, 18};
        if (ndigits <= 9) return {39, 28};
        if (ndigits <= 12) return {55, 38};
    }
    if (ndigits <= 3) return {13, 9};
    if (ndigits <= 6) return {25, 18};
    if (ndigits <= 9) return {39, 28};
    if (ndigits <= 12) return {53, 38};
    // clang-format on
    throw std::runtime_error("Requested precision too high");
}

void update_offsets_from_counts(const sctl::Vector<sctl::Long> &counts, sctl::Long N,
                                sctl::Vector<sctl::Long> &offsets) {
    offsets.ReInit(counts.Dim());
    sctl::Long last_offset;
    for (int i = 0; i < counts.Dim(); ++i) {
        if (counts[i]) {
            offsets[i] = last_offset;
            last_offset += N;
        } else
            offsets[i] = -1;
    }
}

template <typename Real, int DIM>
DMKPtTree<Real, DIM>::DMKPtTree(const sctl::Comm &comm, const pdmk_params &params_, const sctl::Vector<Real> &r_src,
                                const sctl::Vector<Real> &r_trg, const sctl::Vector<Real> &charge)
    : sctl::PtTree<Real, DIM>(comm), params(params_), n_digits(std::round(log10(1.0 / params_.eps) - 0.1)),
      n_pw_max(get_pwmax_and_poly_order(DIM, n_digits, params_.kernel).first),
      n_order(get_pwmax_and_poly_order(DIM, n_digits, params_.kernel).second) {
    auto &logger = dmk::get_logger(comm, params.log_level);
    auto &rank_logger = dmk::get_rank_logger(comm, params.log_level);

    const int n_src = r_src.Dim() / DIM;
    const int n_trg = r_trg.Dim() / DIM;

    // 0: Initialization
    sctl::Vector<Real> pot_vec_src(n_src * params.n_mfm);
    sctl::Vector<Real> pot_vec_trg(n_trg * params.n_mfm);
    pot_vec_src.SetZero();
    pot_vec_trg.SetZero();

    logger->debug("Building tree and sorting points");
    constexpr bool balance21 = true; // Use "2-1" balancing for the tree, i.e. bordering boxes
                                     // never more than one level away in depth
    constexpr int halo = 0;          // Only grab nearest neighbors as 'ghosts'
    this->AddParticles("pdmk_src", r_src);
    this->AddParticles("pdmk_trg", r_trg);
    this->AddParticleData("pdmk_charge", "pdmk_src", charge);
    this->AddParticleData("pdmk_pot_src", "pdmk_src", pot_vec_src);
    this->AddParticleData("pdmk_pot_trg", "pdmk_trg", pot_vec_trg);
    this->UpdateRefinement(r_src, params.n_per_leaf, balance21, params.use_periodic, halo);
    this->template Broadcast<Real>("pdmk_src");
    this->template Broadcast<Real>("pdmk_trg");
    this->template Broadcast<Real>("pdmk_charge");
    this->template Broadcast<Real>("pdmk_pot_src");
    this->template Broadcast<Real>("pdmk_pot_trg");
    this->GetData(r_src_sorted, r_src_cnt, "pdmk_src");
    this->GetData(r_trg_sorted, r_trg_cnt, "pdmk_trg");
    this->GetData(charge_sorted, charge_cnt, "pdmk_charge");
    this->GetData(pot_src_sorted, pot_src_cnt, "pdmk_pot_src");
    this->GetData(pot_trg_sorted, pot_trg_cnt, "pdmk_pot_trg");

    logger->debug("Tree build completed");

    logger->debug("Generating tree traversal metadata");
    generate_metadata();
    logger->debug("Done generating tree traversal metadata");

    rank_logger->trace("Local tree has {} levels {} boxes", n_levels(), n_boxes());

    // 1: Precomputation
    logger->debug("Generating p2c and c2p matrices of order {}", n_order);
    std::tie(c2p, p2c) = dmk::chebyshev::get_c2p_p2c_matrices<Real>(DIM, n_order);
    logger->debug("Finished generating matrices");

    fourier_data = FourierData<Real>(params.kernel, DIM, params.eps, n_digits, n_pw_max, params.fparam, boxsize);
    const auto &wk = fourier_data.windowed_kernel();
    logger->debug("Planewave params at root box: n: {}, stepsize: {}, weight: {}, radius: {}", fourier_data.n_pw(),
                  wk.hpw, wk.ws, wk.rl);
    fourier_data.update_local_coeffs(params.eps);
    logger->debug("Finished updating local potential expansion coefficients");
}

/// @brief Build any bookkeeping data associated with the tree
///
/// This must be called after the constructor (FIXME: should be done at construction)
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename T, int DIM>
void DMKPtTree<T, DIM>::generate_metadata() {
    const int n_nodes = n_boxes();
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    const auto &node_lists = this->GetNodeLists();

    src_counts_local.ReInit(n_nodes);
    trg_counts_local.ReInit(n_nodes);
    r_src_offsets.ReInit(n_nodes);
    r_trg_offsets.ReInit(n_nodes);
    pot_src_offsets.ReInit(n_nodes);
    pot_trg_offsets.ReInit(n_nodes);
    charge_offsets.ReInit(n_nodes);

    r_src_offsets[0] = r_trg_offsets[0] = pot_src_offsets[0] = pot_trg_offsets[0] = charge_offsets[0] = 0;
    for (int i_node = 1; i_node < n_nodes; ++i_node) {
        r_src_offsets[i_node] = r_src_offsets[i_node - 1] + DIM * r_src_cnt[i_node - 1];
        r_trg_offsets[i_node] = r_trg_offsets[i_node - 1] + DIM * r_trg_cnt[i_node - 1];
        pot_src_offsets[i_node] = pot_src_offsets[i_node - 1] + params.n_mfm * pot_src_cnt[i_node - 1];
        pot_trg_offsets[i_node] = pot_trg_offsets[i_node - 1] + params.n_mfm * pot_trg_cnt[i_node - 1];
        charge_offsets[i_node] = charge_offsets[i_node - 1] + params.n_mfm * charge_cnt[i_node - 1];
    }

    level_indices.ReInit(SCTL_MAX_DEPTH);
    int8_t max_depth = 0;
    for (int i_node = 0; i_node < n_nodes; ++i_node) {
        auto &node = node_mid[i_node];
        level_indices[node.Depth()].PushBack(i_node);
        max_depth = std::max(node.Depth(), max_depth);
    }
    max_depth++;
    level_indices.ReInit(max_depth);
    boxsize.ReInit(max_depth + 1);
    boxsize[0] = 1.0;
    for (int i = 1; i < max_depth + 1; ++i)
        boxsize[i] = 0.5 * boxsize[i - 1];

    T scale = 1.0;
    centers.ReInit(n_nodes * DIM);
    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        for (auto i_node : level_indices[i_level]) {
            auto &node = node_mid[i_node];
            auto node_origin = node.template Coord<T>();
            for (int i = 0; i < DIM; ++i)
                centers[i_node * DIM + i] = node_origin[i] + 0.5 * scale;
        }
        scale *= 0.5;
    }

    form_pw_expansion.ReInit(n_nodes);
    eval_pw_expansion.ReInit(n_nodes);
    eval_pw_expansion.SetZero();
    eval_tp_expansion.ReInit(n_nodes);
    eval_tp_expansion.SetZero();

    src_counts_local.SetZero();
    trg_counts_local.SetZero();
    for (int i_level = max_depth - 1; i_level >= 0; i_level--) {
        for (auto i_node : level_indices[i_level]) {
            auto &node = node_mid[i_node];
            assert(i_level == node.Depth());

            src_counts_local[i_node] += r_src_cnt[i_node];
            trg_counts_local[i_node] += r_trg_cnt[i_node];
            if (node_lists[i_node].parent != -1) {
                src_counts_local[node_lists[i_node].parent] += src_counts_local[i_node];
                trg_counts_local[node_lists[i_node].parent] += trg_counts_local[i_node];
            }
        }
    }

    form_pw_expansion[0] = true;
    eval_pw_expansion[0] = true;

    long n_proxy_boxes_upward = 0;
    long n_proxy_boxes_downward = 0;
    for (int box = 0; box < n_nodes; ++box) {
        form_pw_expansion[box] = !node_attr[box].Leaf;
        n_proxy_boxes_upward += form_pw_expansion[box];
    }

    for (const auto &level_boxes : level_indices) {
        for (auto box : level_boxes) {
            for (auto neighbor : node_lists[box].nbr) {
                if (neighbor < 0)
                    continue;

                const int npts = src_counts_local[neighbor] + trg_counts_local[neighbor];
                if (form_pw_expansion[neighbor] && npts) {
                    eval_pw_expansion[box] = true;
                    n_proxy_boxes_downward++;
                    break;
                }
            }
        }
    }

    for (const auto &level_boxes : level_indices) {
        for (auto box : level_boxes) {
            if (!eval_pw_expansion[box])
                continue;
            int tpeval = 1;
            for (auto child : node_lists[box].child) {
                if (child < 0)
                    continue;

                if (eval_pw_expansion[child]) {
                    tpeval = 0;
                    break;
                }
            }
            if (tpeval)
                eval_tp_expansion[box] = true;

            if (!eval_tp_expansion[box]) {
                for (auto child : node_lists[box].child) {
                    if (child < 0)
                        continue;

                    if (!eval_pw_expansion[child])
                        eval_tp_expansion[child] = true;
                }
            }
        }
    }

    direct_neighbs_.ReInit(n_nodes);
    n_direct_neighbs_.ReInit(n_nodes);
    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        for (int box : level_indices[i_level]) {
            if (!r_src_cnt[box]) {
                n_direct_neighbs_[box] = 0;
                continue;
            }

            int i_neighb = 0;
            for (auto neighb : node_lists[box].nbr)
                if (neighb >= 0)
                    direct_neighbs_[box][i_neighb++] = neighb;

            if (i_level == 0) {
                n_direct_neighbs_[box] = i_neighb;
                continue;
            }

            const double cutoff = 0.5 * 1.05 * (boxsize[i_level] + boxsize[i_level - 1]);
            for (auto neighb : node_lists[node_lists[box].parent].nbr) {
                if (neighb < 0 || !node_attr[neighb].Leaf)
                    continue;

                bool inrange = true;
                for (int k = 0; k < DIM; ++k) {
                    const double distance = std::abs(center_ptr(box)[k] - center_ptr(neighb)[k]);
                    if (distance > cutoff)
                        inrange = false;
                }
                if (inrange)
                    direct_neighbs_[box][i_neighb++] = neighb;
            }
            n_direct_neighbs_[box] = i_neighb;
        }
    }

    sctl::Vector<sctl::Long> counts(n_boxes());
    const int n_coeffs = params.n_mfm * sctl::pow<DIM>(n_order);
    for (int i = 0; i < n_boxes(); ++i)
        counts[i] = form_pw_expansion[i] ? n_coeffs : 0;

    proxy_coeffs.ReInit(n_coeffs * n_proxy_boxes_upward);
    proxy_coeffs.SetZero();
    this->AddData("proxy_coeffs", proxy_coeffs, counts);
    this->template GetData<T>(proxy_coeffs, counts, "proxy_coeffs");
    proxy_coeffs_offsets.ReInit(n_boxes());

    long last_offset = 0;
    for (int box = 0; box < n_nodes; ++box) {
        if (counts[box]) {
            proxy_coeffs_offsets[box] = last_offset;
            last_offset += n_coeffs;
        } else
            proxy_coeffs_offsets[box] = -1;
    }

    proxy_coeffs_offsets_downward.ReInit(n_boxes());
    proxy_coeffs_downward.ReInit(n_coeffs * n_proxy_boxes_downward);
    last_offset = 0;
    for (int box = 0; box < n_nodes; ++box) {
        if (eval_pw_expansion[box]) {
            proxy_coeffs_offsets_downward[box] = last_offset;
            last_offset += n_coeffs;
        } else
            proxy_coeffs_downward[box] = -1;
    }
}

/// @brief Fill out the proxy coefficients used in the upward pass
///
/// Updates: proxy_coeffs
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename T, int DIM>
void DMKPtTree<T, DIM>::upward_pass() {
    auto &logger = dmk::get_logger(this->GetComm());
    auto &rank_logger = dmk::get_rank_logger(this->GetComm());

    const std::size_t n_coeffs = params.n_mfm * sctl::pow<DIM>(n_order);

    constexpr int n_children = 1u << DIM;
    const auto &node_lists = this->GetNodeLists();
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    const int dim = DIM;

    int n_direct = 0;
    const int start_level = std::max(n_levels() - 2, 0);
    for (auto i_box : level_indices[start_level]) {
        if (!form_pw_expansion[i_box] || !src_counts_local[i_box] || node_attr[i_box].Ghost)
            continue;

        proxy::charge2proxycharge<T, DIM>(r_src_view(i_box), charge_view(i_box), center_view(i_box),
                                          2.0 / boxsize[start_level], proxy_view_upward(i_box));
    }
    logger->debug("proxy: finished building base proxy charges");

    int n_merged = 0;
    for (int i_level = start_level - 1; i_level >= 0; --i_level) {
        for (auto parent_box : level_indices[i_level]) {
            if (!form_pw_expansion[parent_box])
                continue;

            auto &children = node_lists[parent_box].child;
            for (int i_child = 0; i_child < n_children; ++i_child) {
                const int child_box = children[i_child];
                if (child_box < 0 || !src_counts_local[child_box])
                    continue;

                if (form_pw_expansion[child_box]) {
                    ndview<const T, 2> c2p_view(&c2p[i_child * DIM * n_order * n_order], n_order, DIM);
                    tensorprod::transform<T, DIM>(params.n_mfm, true, proxy_view_upward(child_box), c2p_view,
                                                  proxy_view_upward(parent_box));
                } else if (!node_attr[child_box].Ghost) {
                    proxy::charge2proxycharge<T, DIM>(r_src_view(child_box), charge_view(child_box),
                                                      center_view(parent_box), 2.0 / boxsize[i_level],
                                                      proxy_view_upward(parent_box));
                }
            }
        }
    }

    logger->debug("Finished building proxy charges");
    sctl::Vector<sctl::Long> counts;
    this->template ReduceBroadcast<T>("proxy_coeffs");
    this->template GetData<T>(proxy_coeffs, counts, "proxy_coeffs");
    long last_offset = 0;
    for (int box = 0; box < n_boxes(); ++box) {
        if (counts[box]) {
            proxy_coeffs_offsets[box] = last_offset;
            last_offset += n_coeffs;
        } else
            proxy_coeffs_offsets[box] = -1;
    }

    logger->debug("proxy: finished broadcasting proxy charges");
}

template <typename T, int DIM>
void multiply_kernelFT_cd2p(const ndview<std::complex<T>, DIM + 1> &pwexp, const sctl::Vector<T> &radialft) {
    const int nd = pwexp.extent(DIM);
    const int nexp = radialft.Dim();
    ndview<std::complex<T>, 2> pwexp_flat(pwexp.data_handle(), nexp, nd);

    for (int ind = 0; ind < nd; ++ind)
        for (int n = 0; n < nexp; ++n)
            pwexp_flat(n, ind) *= radialft[n];
}

template <typename Complex, int DIM>
void shift_planewave(const ndview<std::complex<Complex>, DIM + 1> &pwexp1_,
                     const ndview<std::complex<Complex>, DIM + 1> &pwexp2_,
                     const ndview<const std::complex<Complex>, 1> &wpwshift) {

    const int nd = pwexp1_.extent(DIM);
    const int nexp = wpwshift.extent(0);

    dmk::ndview<const std::complex<Complex>, 2> pwexp1(pwexp1_.data_handle(), nexp, nd);
    dmk::ndview<std::complex<Complex>, 2> pwexp2(pwexp2_.data_handle(), nexp, nd);

    for (int ind = 0; ind < nd; ++ind)
        for (int j = 0; j < nexp; ++j)
            pwexp2(j, ind) += pwexp1(j, ind) * wpwshift(j);
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::init_planewave_data() {
    const std::size_t n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const std::size_t n_pw_per_box = n_pw_modes * params.n_mfm;

    pw_in_offsets.ReInit(n_boxes());
    pw_out_offsets.ReInit(n_boxes());
    pw_in_offsets[0] = pw_out_offsets[0] = 0;
    int n_pw_boxes_out = 1;
    int n_pw_boxes_in = 1;
    for (int i_box = 1; i_box < n_boxes(); ++i_box) {
        pw_out_offsets[i_box] = pw_out_offsets[i_box - 1] + form_pw_expansion[i_box] * n_pw_per_box;
        n_pw_boxes_out += form_pw_expansion[i_box];

        bool need_in = ((src_counts_local[i_box] + trg_counts_local[i_box]) > 0) || eval_pw_expansion[i_box];
        pw_in_offsets[i_box] = pw_in_offsets[i_box - 1] + need_in * n_pw_per_box;
        n_pw_boxes_in += need_in;
    }

    pw_out.ReInit(n_pw_per_box * n_pw_boxes_out);
    pw_in.ReInit(n_pw_per_box * n_pw_boxes_in);
    pw_out.SetZero();
    pw_in.SetZero();
}

/// @brief Perform the "downward pass"
///
/// Updates: proxy_coeffs_downward, tree 'pdmk_pot' particle data
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename T, int DIM>
void DMKPtTree<T, DIM>::downward_pass() {
    auto &logger = dmk::get_logger(this->GetComm());
    auto &rank_logger = dmk::get_rank_logger(this->GetComm());
    constexpr int dim = DIM;
    constexpr int nmax = 1;
    const int nd = params.n_mfm;
    // FIXME: This should be assigned automatically at tree construction (fourier_data should be subobject)
    n_pw = fourier_data.n_pw();

    const auto &node_lists = this->GetNodeLists();
    const auto &node_attr = this->GetNodeAttr();

    init_planewave_data();
    const std::size_t n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const std::size_t n_pw_per_box = n_pw_modes * nd;
    const std::size_t n_coeffs_per_box = params.n_mfm * sctl::pow<DIM>(n_order);

    const int shift = n_pw / 2;
    sctl::Vector<std::complex<T>> wpwshift(n_pw_modes * sctl::pow<DIM>(2 * nmax + 1));
    sctl::Vector<T> ts(n_pw);
    sctl::Vector<T> radialft(n_pw_modes);
    sctl::Vector<T> kernel_ft;
    get_windowed_kernel_ft<T, DIM>(params.kernel, &params.fparam, fourier_data.beta(), n_digits, boxsize[0],
                                   fourier_data.prolate0_fun, kernel_ft);
    util::mk_tensor_product_fourier_transform(DIM, n_pw, ndview<const T, 1>(&kernel_ft[0], kernel_ft.Dim()),
                                              ndview<T, 1>(&radialft[0], n_pw_modes));

    sctl::Vector<std::complex<T>> poly2pw(n_order * n_pw), pw2poly(n_order * n_pw);
    fourier_data.calc_planewave_coeff_matrices(-1, n_order, poly2pw, pw2poly);
    ndview<const std::complex<T>, 2> poly2pw_view(&poly2pw[0], n_pw, n_order);
    ndview<const std::complex<T>, 2> pw2poly_view(&pw2poly[0], n_pw, n_order);

    dmk::proxy::proxycharge2pw<T, DIM>(proxy_view_upward(0), poly2pw_view, pw_out_view(0));
    multiply_kernelFT_cd2p<T, DIM>(pw_out_view(0), radialft);
    memcpy(pw_in_ptr(0), pw_out_ptr(0), n_pw_per_box * sizeof(std::complex<T>));

    proxy_coeffs_downward.SetZero();
    dmk::planewave_to_proxy_potential<T, DIM>(pw_in_view(0), pw2poly_view, proxy_view_downward(0));

    Eigen::MatrixX<T> r_src_t = Eigen::Map<Eigen::MatrixX<T>>(r_src_ptr(0), dim, src_counts_local[0]).transpose();
    Eigen::MatrixX<T> r_trg_t = Eigen::Map<Eigen::MatrixX<T>>(r_trg_ptr(0), dim, trg_counts_local[0]).transpose();

    constexpr int n_children = 1u << DIM;

    const T scale_factor_diff_ft = [](dmk_ikernel kernel) -> T {
        switch (kernel) {
        case DMK_YUKAWA:
            return T(0.0);
        case DMK_LAPLACE:
            return DIM == 2 ? T(1.0) : T(2.0);
        case DMK_SQRT_LAPLACE:
            return DIM == 2 ? T(2.0) : T(4.0);
        default:
            throw std::runtime_error("Invalid kernel type: " + std::to_string(kernel));
        }
    }(params.kernel);

    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        for (int i = 0; i < n_pw; ++i)
            ts[i] = fourier_data.difference_kernel(i_level).hpw * (i - shift);

        fourier_data.calc_planewave_coeff_matrices(i_level, n_order, poly2pw, pw2poly);
        dmk::calc_planewave_translation_matrix<DIM>(1, boxsize[i_level], n_pw, ts, wpwshift);

        // Calculate difference kernel fourier transform. Exploit scale invariance on lower levels, when applicable.
        if (i_level == 0 || !scale_factor_diff_ft) {
            get_difference_kernel_ft<T, DIM>(params.kernel, &params.fparam, fourier_data.beta(), n_digits,
                                             boxsize[i_level], fourier_data.prolate0_fun, kernel_ft);
        } else {
            for (auto &val : kernel_ft)
                val *= scale_factor_diff_ft;
        }

        util::mk_tensor_product_fourier_transform(DIM, n_pw, ndview<const T, 1>(&kernel_ft[0], kernel_ft.Dim()),
                                                  ndview<T, 1>(&radialft[0], n_pw_modes));

        // Form outgoing expansions
        for (auto box : level_indices[i_level]) {
            if (!form_pw_expansion[box])
                continue;

            // Form the outgoing expansion Φl(box) for the difference kernel Dl from the proxy charge expansion
            // coefficients using Tprox2pw.
            dmk::proxy::proxycharge2pw<T, DIM>(proxy_view_upward(box), poly2pw_view, pw_out_view(box));

            multiply_kernelFT_cd2p<T, DIM>(pw_out_view(box), radialft);
            memcpy(pw_in_ptr(box), pw_out_ptr(box), n_pw_per_box * sizeof(std::complex<T>));
        }

        // Form incoming expansions
        for (auto box : level_indices[i_level]) {
            if (src_counts_local[box] + trg_counts_local[box] == 0)
                continue;

            for (auto &neighbor : node_lists[box].nbr) {
                if (neighbor < 0 || neighbor == box || !form_pw_expansion[neighbor])
                    continue;

                // Translate the outgoing expansion Φl(colleague) to the center of box and add to the incoming plane
                // wave expansion Ψl(box) using wpwshift.

                // note: neighbors in SCTL are sorted in reverse order to wpwshift
                // FIXME: check if valid for periodic boundary conditions
                const int ind = sctl::pow<3>(dim) - 1 - (&neighbor - &node_lists[box].nbr[0]);

                ndview<const std::complex<T>, 1> wpwshift_view(&wpwshift[n_pw_per_box * ind], n_pw_per_box);
                shift_planewave<T, DIM>(pw_out_view(neighbor), pw_in_view(box), wpwshift_view);
            }
        }

        // Form local expansions
        const T sc = 2.0 / boxsize[i_level];
        for (auto box : level_indices[i_level]) {
            const int n_trg = trg_counts_local[box];
            const int n_src = src_counts_local[box];
            const int n_pts = n_src + n_trg;

            if (!eval_pw_expansion[box] || n_pts == 0)
                continue;

            // Convert incoming plane wave expansion Ψl(box) to the local expansion Λl(box) using Tpw2poly
            dmk::planewave_to_proxy_potential<T, DIM>(pw_in_view(box), pw2poly_view, proxy_view_downward(box));

            if (eval_tp_expansion[box]) {
                if (n_src)
                    proxy::eval_targets<T, DIM>(proxy_view_downward(box), r_src_view(box), center_view(box), sc,
                                                pot_src_view(box));
                if (n_trg)
                    proxy::eval_targets<T, DIM>(proxy_view_downward(box), r_trg_view(box), center_view(box), sc,
                                                pot_trg_view(box));
            }

            // Translate and add the local expansion of Λl(box) to the local expansion of Λl(child).
            for (int i_child = 0; i_child < n_children; ++i_child) {
                const int child = node_lists[box].child[i_child];
                if (child < 0)
                    continue;

                if (eval_tp_expansion[child] && !eval_pw_expansion[child]) {
                    if (src_counts_local[child])
                        proxy::eval_targets<T, DIM>(proxy_view_downward(box), r_src_view(child), center_view(box), sc,
                                                    pot_src_view(child));
                    if (trg_counts_local[child])
                        proxy::eval_targets<T, DIM>(proxy_view_downward(box), r_trg_view(child), center_view(box), sc,
                                                    pot_trg_view(child));
                } else if (eval_pw_expansion[child]) {
                    ndview<const T, 2> p2c_view(&p2c[i_child * DIM * n_order * n_order], n_order, DIM);
                    dmk::tensorprod::transform<T, DIM>(nd, true, proxy_view_downward(box), p2c_view,
                                                       proxy_view_downward(child));
                }
            }
        }

        // FIXME: Clean up the logic for the direct interactions. Maybe make a small class/closure
        const auto [bsize, rsc, cen, d2max] = [&]() -> std::tuple<T, T, T, T> {
            const double bsize = i_level == 0 ? 0.5 * boxsize[i_level] : boxsize[i_level];
            const double d2max = bsize * bsize;

            if ((params.kernel == DMK_SQRT_LAPLACE && DIM == 3) || (params.kernel == DMK_LAPLACE && DIM == 2))
                return {bsize, 2.0 / (bsize * bsize), -1.0, d2max};
            if (params.kernel == DMK_YUKAWA)
                return {bsize, 2.0 / bsize, -1.0, d2max};

            return {bsize, 2.0 / bsize, -bsize / 2.0, d2max};
        }();

        auto get_windowed_kernel_at_zero = [&]() -> T {
            if (params.kernel == DMK_YUKAWA)
                return fourier_data.yukawa_windowed_kernel_value_at_zero(i_level);
            else if (params.kernel == DMK_LAPLACE) {
                const T psi0 = fourier_data.prolate0_fun.eval_val(0.0);
                const auto c = fourier_data.prolate0_fun.intvals(fourier_data.beta());
                if (DIM == 3)
                    return psi0 / (c[0] * bsize);
                else
                    throw std::runtime_error("Unsupported kernel DMK_LAPLACE, DIM = " + std::to_string(DIM));
            } else if (params.kernel == DMK_SQRT_LAPLACE) {
                const T psi0 = fourier_data.prolate0_fun.eval_val(0.0);
                const auto c = fourier_data.prolate0_fun.intvals(fourier_data.beta());
                if constexpr (DIM == 2)
                    return psi0 / (c[0] * bsize);
                if constexpr (DIM == 3)
                    return psi0 / (2 * c[1] * bsize * bsize);

            } else
                throw std::runtime_error("Unsupported kernel");
        };

        const T w0 = get_windowed_kernel_at_zero();

        const auto &cheb_coeffs = fourier_data.cheb_coeffs(i_level);
        for (auto box : level_indices[i_level]) {
            // Evaluate the direct interactions
            if (!r_src_cnt[box])
                continue;

            for (auto neighbor : direct_neighbs(box)) {
                const int n_src_neighb = src_counts_local[neighbor];
                const int n_trg_neighb = trg_counts_local[neighbor];

                std::array<std::span<const T>, DIM> r_trg;
                if (n_src_neighb) {
                    for (int i = 0; i < DIM; ++i)
                        r_trg[i] = std::span<const T>(&r_src_t(r_src_offsets[neighbor] / DIM, i), n_src_neighb);

                    direct_eval<T, DIM>(params.kernel, r_src_view(box), r_trg, charge_view(box), cheb_coeffs,
                                        &params.fparam, rsc, cen, d2max, pot_src_view(neighbor), n_digits);
                }
                if (n_trg_neighb) {
                    for (int i = 0; i < DIM; ++i)
                        r_trg[i] = std::span<const T>(&r_trg_t(r_trg_offsets[neighbor] / DIM, i), n_trg_neighb);

                    direct_eval<T, DIM>(params.kernel, r_src_view(box), r_trg, charge_view(box), cheb_coeffs,
                                        &params.fparam, rsc, cen, d2max, pot_trg_view(neighbor), n_digits);
                }
            }

            // Correct for self-evaluations
            auto pot = pot_src_view(box);
            auto charge = charge_view(box);
            for (int i_src = 0; i_src < r_src_cnt[box]; ++i_src)
                for (int i = 0; i < nd; ++i)
                    pot(i, i_src) -= w0 * charge(i, i_src);
        }
    }
}

MPI_TEST_CASE("[DMK] 3D: Proxy charges on upward pass, 2 ranks", 2) {
    constexpr int n_dim = 3;
    constexpr int n_src = 10000;
    constexpr int n_trg = n_src;
    constexpr int n_charge_dim = 1;
    constexpr bool uniform = false;

    sctl::Vector<double> r_src, r_trg, r_src_norms, charges, dipoles, pot_src, grad_src, hess_src, pot_trg, grad_trg,
        hess_trg;
    if (test_rank == 0)
        dmk::util::init_test_data(n_dim, n_charge_dim, n_src, n_trg, uniform, true, r_src, r_trg, r_src_norms, charges,
                                  dipoles, 0);

    pdmk_params params;
    params.eps = 1E-6;
    params.kernel = DMK_YUKAWA;
    params.log_level = SPDLOG_LEVEL_OFF;
    params.fparam = 6.0;
    params.n_dim = n_dim;
    params.n_mfm = n_charge_dim;
    params.n_per_leaf = 80;

    double st = omp_get_wtime();
    auto comm = sctl::Comm(test_comm);
    DMKPtTree<double, n_dim> tree(comm, params, r_src, r_trg, charges);
    tree.upward_pass();
    tree.downward_pass();
    tree.GetParticleData(pot_src, "pdmk_pot_src");
    tree.GetParticleData(pot_trg, "pdmk_pot_trg");
    if (test_rank == 0)
        std::cout << omp_get_wtime() - st << std::endl;

    if (test_rank == 0) {
        dmk::util::init_test_data(n_dim, n_charge_dim, n_src, n_trg, uniform, true, r_src, r_trg, r_src_norms, charges,
                                  dipoles, 0);

        DMKPtTree<double, n_dim> tree_single(sctl::Comm::Self(), params, r_src, r_trg, charges);
        tree_single.upward_pass();
        tree_single.downward_pass();
        sctl::Vector<double> pot_src_single, pot_trg_single;
        tree_single.GetParticleData(pot_src_single, "pdmk_pot_src");
        tree_single.GetParticleData(pot_trg_single, "pdmk_pot_trg");

        sleep(test_rank);
        auto &node_mid = tree.GetNodeMID();
        for (int ibox = 0; ibox < tree.n_boxes(); ++ibox) {
            std::array<double, 3> x, x_single;
            for (int i = 0; i < 3; ++i)
                x[i] = tree.center_ptr(ibox)[i];

            int single_box = -1;
            for (int jbox = 0; jbox < tree_single.n_boxes(); ++jbox) {
                for (int i = 0; i < 3; ++i)
                    x_single[i] = tree_single.center_ptr(jbox)[i];

                if (x_single == x) {
                    single_box = jbox;
                    break;
                }
            }
            if (single_box < 0) {
                std::cout << fmt::format("No match for box {}\n", ibox);
                continue;
            }

            double maxerr_up = 0.0;
            if (ibox == 0 || tree.form_pw_expansion[ibox]) {
                double max_actual = 0.0;
                for (int i = 0; i < sctl::pow<3>(tree.n_order); ++i) {
                    const double actual = tree_single.proxy_ptr_upward(single_box)[i];
                    const double mpi = tree.proxy_ptr_upward(ibox)[i];
                    const double err = std::abs(mpi - actual);
                    maxerr_up = std::max(err, maxerr_up);
                    max_actual = std::max(std::abs(actual), max_actual);
                }
                maxerr_up = max_actual ? maxerr_up / max_actual : 0.0;
            }

            if (maxerr_up > 1E-14)
                std::cout << fmt::format("{:3} {:3} {:3} {:5.4E} {:3} ({} {} {})\n", test_rank, ibox, single_box,
                                         maxerr_up, node_mid[ibox].Depth(), x[0], x[1], x[2]);
        }

        if (test_rank == 0) {
            for (int i = 0; i < n_src; ++i) {
                double err_src = pot_src[i] == 0. ? 0.0 : std::abs(1.0 - pot_src[i] / pot_src_single[i]);
                double err_trg = pot_trg[i] == 0. ? 0.0 : std::abs(1.0 - pot_trg[i] / pot_trg_single[i]);
                if (err_src > 1E-12 || err_trg > 1E-12)
                    std::cout << fmt::format("{} {:5.4E} {:5.4E}\n", i, err_src, err_trg);
            }
        }
    }
}

template struct DMKPtTree<float, 2>;
template struct DMKPtTree<float, 3>;
template struct DMKPtTree<double, 2>;
template struct DMKPtTree<double, 3>;

} // namespace dmk
