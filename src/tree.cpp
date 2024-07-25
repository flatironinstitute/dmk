#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/logger.h>
#include <dmk/planewave.hpp>
#include <dmk/prolate_funcs.hpp>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/tree.hpp>
#include <dmk/types.hpp>
#include <dmk/util.hpp>
#include <mpi.h>
#include <ranges>
#include <sctl/tree.hpp>
#include <stdexcept>

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

template <typename Real, int DIM>
DMKPtTree<Real, DIM>::DMKPtTree(const sctl::Comm &comm, const pdmk_params &params_, const sctl::Vector<Real> &r_src,
                                const sctl::Vector<Real> &r_trg, const sctl::Vector<Real> &charge)
    : sctl::PtTree<Real, DIM>(comm), params(params_), n_digits(std::round(log10(1.0 / params_.eps) - 0.1)),
      n_pw_max(get_pwmax_and_poly_order(DIM, n_digits, params_.kernel).first),
      n_order(get_pwmax_and_poly_order(DIM, n_digits, params_.kernel).second) {
    auto &logger = dmk::get_logger(params.log_level);
    auto &rank_logger = dmk::get_rank_logger(params.log_level);

    const int n_src = r_src.Dim() / DIM;
    const int n_trg = r_trg.Dim() / DIM;

    // 0: Initialization
    sctl::Vector<Real> pot_vec_src(n_src * params.n_mfm);
    sctl::Vector<Real> pot_vec_trg(n_trg * params.n_mfm);
    pot_vec_src.SetZero();
    pot_vec_trg.SetZero();

    logger->debug("Building tree and sorting points");
    this->AddParticles("pdmk_src", r_src);
    this->AddParticleData("pdmk_charge", "pdmk_src", charge);
    this->AddParticleData("pdmk_pot_src", "pdmk_src", pot_vec_src);
    this->AddParticles("pdmk_trg", r_trg);
    this->AddParticleData("pdmk_pot_trg", "pdmk_trg", pot_vec_trg);
    this->UpdateRefinement(r_src, params.n_per_leaf, true, params.use_periodic); // balance21 = true
    logger->debug("Tree build completed");

    logger->debug("Generating tree traversal metadata");
    generate_metadata();
    logger->debug("Done generating tree traversal metadata");

    rank_logger->trace("Local tree has {} levels {} boxes", n_levels(), n_boxes());

    // 1: Precomputation
    logger->debug("Generating p2c and c2p matrices of order {}", n_order);
    auto [c2p, p2c] = dmk::chebyshev::get_c2p_p2c_matrices<Real>(DIM, n_order);
    logger->debug("Finished generating matrices");

    fourier_data = FourierData<Real>(params.kernel, DIM, params.eps, n_digits, n_pw_max, params.fparam, boxsize);
    logger->debug("Planewave params at root box: n: {}, stepsize: {}, weight: {}, radius: {}", fourier_data.n_pw,
                  fourier_data.hpw[0], fourier_data.ws[0], fourier_data.rl[0]);
    fourier_data.update_windowed_kernel_fourier_transform();
    logger->debug("Truncated fourier transform for kernel {} at root box generated", int(params.kernel));
    fourier_data.update_difference_kernels();
    logger->debug("Finished calculating difference kernels");
    fourier_data.update_local_coeffs(params.eps);
    logger->debug("Finished updating local potential expansion coefficients");

    // upward pass
    upward_pass(c2p);
    downward_pass(p2c);
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
    this->GetData(r_src_sorted, r_src_cnt, "pdmk_src");
    this->GetData(r_trg_sorted, r_trg_cnt, "pdmk_trg");
    this->GetData(charge_sorted, charge_cnt, "pdmk_charge");
    this->GetData(pot_src_sorted, pot_src_cnt, "pdmk_pot_src");
    this->GetData(pot_trg_sorted, pot_trg_cnt, "pdmk_pot_trg");
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    const auto &node_lists = this->GetNodeLists();

    src_counts_local.ReInit(n_nodes);
    trg_counts_local.ReInit(n_nodes);

    r_src_offsets.resize(n_nodes);
    r_trg_offsets.resize(n_nodes);
    pot_src_offsets.resize(n_nodes);
    pot_trg_offsets.resize(n_nodes);
    charge_offsets.resize(n_nodes);

    for (int i_node = 1; i_node < n_nodes; ++i_node) {
        r_src_offsets[i_node] = r_src_offsets[i_node - 1] + DIM * r_src_cnt[i_node - 1];
        r_trg_offsets[i_node] = r_trg_offsets[i_node - 1] + DIM * r_trg_cnt[i_node - 1];
        pot_src_offsets[i_node] = pot_src_offsets[i_node - 1] + params.n_mfm * pot_src_cnt[i_node - 1];
        pot_trg_offsets[i_node] = pot_trg_offsets[i_node - 1] + params.n_mfm * pot_trg_cnt[i_node - 1];
        charge_offsets[i_node] = charge_offsets[i_node - 1] + params.n_mfm * charge_cnt[i_node - 1];
    }

    level_indices.resize(SCTL_MAX_DEPTH);
    int8_t max_depth = 0;
    for (int i_node = 0; i_node < n_nodes; ++i_node) {
        auto &node = node_mid[i_node];
        level_indices[node.Depth()].push_back(i_node);
        max_depth = std::max(node.Depth(), max_depth);
    }
    max_depth++;
    level_indices.resize(max_depth);
    boxsize.resize(max_depth + 1);
    boxsize[0] = 1.0;
    for (int i = 1; i < max_depth + 1; ++i)
        boxsize[i] = 0.5 * boxsize[i - 1];

    T scale = 1.0;
    centers.resize(n_nodes * DIM);
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

    sctl::Vector<long> counts(src_counts_local.Dim());
    for (auto &el : counts)
        el = 1;

    this->template AddData("src_counts", src_counts_local, counts);
    this->template ReduceBroadcast<int>("src_counts");
    this->template GetData<int>(src_counts_global, counts, "src_counts");

    this->template AddData("trg_counts", trg_counts_local, counts);
    this->template ReduceBroadcast<int>("trg_counts");
    this->template GetData<int>(trg_counts_global, counts, "trg_counts");

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

                const int npts = src_counts_global[neighbor] + trg_counts_global[neighbor];
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

    const int n_coeffs = params.n_mfm * sctl::pow<DIM>(n_order);
    proxy_coeffs_offsets.ReInit(n_boxes());
    proxy_coeffs_offsets_downward.ReInit(n_boxes());
    proxy_coeffs.ReInit(n_coeffs * n_proxy_boxes_upward);
    proxy_coeffs_downward.ReInit(n_coeffs * n_proxy_boxes_downward);
    proxy_coeffs_offsets[0] = 0;
    proxy_coeffs_offsets_downward[0] = 0;
    for (int box = 1; box < n_nodes; ++box) {
        proxy_coeffs_offsets[box] = proxy_coeffs_offsets[box - 1] + form_pw_expansion[box] * n_coeffs;
        proxy_coeffs_offsets_downward[box] = proxy_coeffs_offsets_downward[box - 1] + eval_pw_expansion[box] * n_coeffs;
    }
}

/// @brief Fill out the proxy coefficients used in the upward pass
///
/// Updates: proxy_coeffs
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
/// @param[in] n_mfm Number of different "charges" to simultaneously evaluate, a.k.a. the charge dimension
/// @param[in] n_order Linear order of the polynomial expansion representing the charge distribution
/// @param[in] c2p [n_order, n_order, DIM, 2**DIM] Child to parent matrices used to convert child proxy coefficients to
/// parent proxy coefficients
template <typename T, int DIM>
void DMKPtTree<T, DIM>::upward_pass(const sctl::Vector<T> &c2p) {
    auto &logger = dmk::get_logger();
    auto &rank_logger = dmk::get_rank_logger();
    this->GetData(r_src_sorted, r_src_cnt, "pdmk_src");

    const std::size_t n_coeffs = params.n_mfm * sctl::pow<DIM>(n_order);
    proxy_coeffs.SetZero();
    sctl::Vector<sctl::Long> counts(n_boxes());
    counts.SetZero();

    constexpr int n_children = 1u << DIM;
    const auto &node_lists = this->GetNodeLists();
    const auto &attrs = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    const int dim = DIM;

    int n_direct = 0;
    const int start_level = std::max(n_levels() - 2, 0);
    for (auto i_box : level_indices[start_level]) {
        if (!form_pw_expansion[i_box])
            continue;
        proxy::charge2proxycharge(DIM, params.n_mfm, n_order, src_counts_local[i_box], r_src_ptr(i_box),
                                  charge_ptr(i_box), center_ptr(i_box), 2.0 / boxsize[start_level],
                                  proxy_ptr_upward(i_box));
        counts[i_box] = 1;
        n_direct++;
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
                    tensorprod::transform(DIM, params.n_mfm, n_order, n_order, true, proxy_ptr_upward(child_box),
                                          &c2p[i_child * DIM * n_order * n_order], proxy_ptr_upward(parent_box));
                    counts[parent_box] = 1;
                    n_merged += 1;
                } else {
                    proxy::charge2proxycharge(DIM, params.n_mfm, n_order, src_counts_local[child_box],
                                              r_src_ptr(child_box), charge_ptr(child_box), center_ptr(parent_box),
                                              2.0 / boxsize[i_level], proxy_ptr_upward(parent_box));
                    counts[child_box] = 1;
                    n_direct++;
                }
            }
        }
    }

    int tot_proxy = 0;
    for (int i = 0; i < n_boxes(); ++i) {
        auto &count = counts[i];
        tot_proxy += count;
        count = form_pw_expansion[i] ? n_coeffs : 0;
    }

    logger->debug("Finished building proxy charges");
    this->AddData("proxy_coeffs", proxy_coeffs, counts);
    this->template ReduceBroadcast<T>("proxy_coeffs");
    this->template GetData<T>(proxy_coeffs, counts, "proxy_coeffs");

    int buf[] = {tot_proxy, n_direct, n_merged};
    if (this->GetComm().Rank() == 0)
        MPI_Reduce(MPI_IN_PLACE, buf, 3, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    else
        MPI_Reduce(buf, buf, 3, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    logger->debug("proxy: finished broadcasting proxy charges");
    logger->trace("proxy: n_proxy, n_direct, n_merged: {} {} {}", buf[0], buf[1], buf[2]);
    rank_logger->trace("proxy: n_proxy_local / n_boxes_local {}", (float)tot_proxy / n_boxes());
}

template <typename T, int DIM>
void tensor_product_fourier_transform(int nexp, int npw, int nfourier, const T *fhat, T *pswfft) {
    const int npw2 = npw / 2;

    if constexpr (DIM == 1) {
        for (int j1 = -npw2; j1 <= 0; ++j1)
            pswfft[j1] = fhat[j1 * j1];
    } else if constexpr (DIM == 2) {
        for (int j2 = -npw2, j = 0; j2 <= 0; ++j2)
            for (int j1 = -npw2; j1 <= (npw - 1) / 2; ++j1, ++j)
                pswfft[j] = fhat[j1 * j1 + j2 * j2];
    } else if constexpr (DIM == 3) {
        for (int j3 = -npw2, j = 0; j3 <= 0; ++j3)
            for (int j2 = -npw2; j2 <= (npw - 1) / 2; ++j2)
                for (int j1 = -npw2; j1 <= (npw - 1) / 2; ++j1, ++j)
                    pswfft[j] = fhat[j1 * j1 + j2 * j2 + j3 * j3];
    } else
        static_assert(dmk::util::always_false<T>, "Invalid DIM supplied");
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

template <typename Complex>
void shift_planewave(int nd, int nexp, const Complex *pwexp1_, Complex *pwexp2_, const Complex *wpwshift) {
    dmk::ndview<const Complex, 2> pwexp1(pwexp1_, nexp, nd);
    dmk::ndview<Complex, 2> pwexp2(pwexp2_, nexp, nd);

    for (int ind = 0; ind < nd; ++ind)
        for (int j = 0; j < nexp; ++j)
            pwexp2(j, ind) += pwexp1(j, ind) * wpwshift[j];
}

/// @brief Perform the "downward pass"
///
/// Updates: proxy_coeffs_downward, tree 'pdmk_pot' particle data
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
/// @param[in,out] fourier_data Various fourier data. Only changes work array (FIXME: lame doc)
/// @param[in] p2c [n_order, n_order, DIM, 2**DIM] Parent to child matrices used to pass parent proxy charges to their
/// children
template <typename T, int DIM>
void DMKPtTree<T, DIM>::downward_pass(const sctl::Vector<T> &p2c) {
    auto &logger = dmk::get_logger();
    auto &rank_logger = dmk::get_rank_logger();
    const int nd = params.n_mfm;

    const auto &node_lists = this->GetNodeLists();
    const auto &node_attr = this->GetNodeAttr();
    const auto xs = dmk::chebyshev::get_cheb_nodes<T>(n_order, -1.0, 1.0);
    sctl::Vector<std::complex<T>> poly2pw(n_order * fourier_data.n_pw), pw2poly(n_order * fourier_data.n_pw);

    const int nd_in = params.n_mfm;
    const int nd_out = params.n_mfm;
    // FIXME: This should be assigned automatically at tree construction (fourier_data should be suboject)
    n_pw = fourier_data.n_pw;
    const std::size_t n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const std::size_t n_pw_per_box = n_pw_modes * nd_out;
    const std::size_t n_coeffs_per_box = params.n_mfm * sctl::pow<DIM>(n_order);

    pw_in_offsets.ReInit(n_boxes());
    pw_out_offsets.ReInit(n_boxes());
    pw_in_offsets[0] = pw_out_offsets[0] = 0;
    int n_pw_boxes_out = 1;
    int n_pw_boxes_in = 1;
    for (int i_box = 1; i_box < n_boxes(); ++i_box) {
        pw_out_offsets[i_box] = pw_out_offsets[i_box - 1] + form_pw_expansion[i_box] * n_pw_per_box;
        n_pw_boxes_out += form_pw_expansion[i_box];

        bool need_in = ((src_counts_global[i_box] + trg_counts_global[i_box]) > 0) || eval_pw_expansion[i_box];
        pw_in_offsets[i_box] = pw_in_offsets[i_box - 1] + need_in * n_pw_per_box;
        n_pw_boxes_in += need_in;
    }

    pw_out.ReInit(n_pw_per_box * n_pw_boxes_out);
    pw_in.ReInit(n_pw_per_box * n_pw_boxes_in);
    proxy_coeffs_downward.SetZero();
    pw_out.SetZero();
    pw_in.SetZero();

    constexpr int dim = DIM;
    constexpr int nmax = 1;
    const int shift = n_pw / 2;
    const int nexp = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    sctl::Vector<std::complex<T>> wpwshift(n_pw_modes * sctl::pow<DIM>(2 * nmax + 1));
    sctl::Vector<T> ts(n_pw);
    sctl::Vector<T> rk(DIM * sctl::pow<DIM>(n_pw));
    sctl::Vector<T> radialft(nexp);

    for (int i = 0; i < n_pw; ++i)
        ts[i] = fourier_data.hpw[0] * (i - shift);
    ndview<const double, 1> ts_view(&ts[0], n_pw);
    ndview<double, 2> rk_view(&rk[0], DIM, pow(n_pw, DIM));
    dmk::util::mesh_nd(DIM, ts_view, rk_view);
    ndview<const double, 1> fourier_data_view(&fourier_data.dkernelft[0], fourier_data.n_fourier + 1);
    ndview<double, 1> radialft_view(&radialft[0], nexp);
    dmk::util::mk_tensor_product_fourier_transform(DIM, n_pw, fourier_data_view, radialft_view);

    fourier_data.calc_planewave_coeff_matrices(-1, n_order, poly2pw, pw2poly);

    ndview<const std::complex<double>, 2> poly2pw_view(&poly2pw[0], fourier_data.n_pw, n_order);
    dmk::proxy::proxycharge2pw<double, DIM>(proxy_view_upward(0), poly2pw_view, pw_out_view(0));

    constexpr int zero = 0;
    multiply_kernelFT_cd2p<T, DIM>(pw_out_view(0), radialft);
    memcpy(pw_in_ptr(0), pw_out_ptr(0), n_pw_per_box * sizeof(std::complex<T>));
    dmk::planewave_to_proxy_potential(DIM, nd, n_order, n_pw, pw_in_ptr(0), &pw2poly[0], proxy_ptr_downward(0));

    constexpr int n_children = 1u << DIM;
    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        for (int i = 0; i < n_pw; ++i)
            ts[i] = fourier_data.hpw[i_level + 1] * (i - shift);
        ndview<const double, 1> ts_view(&ts[0], n_pw);
        ndview<double, 2> rk_view(&rk[0], DIM, pow(n_pw, DIM));
        dmk::util::mesh_nd(DIM, ts_view, rk_view);

        fourier_data.calc_planewave_coeff_matrices(i_level, n_order, poly2pw, pw2poly);
        dmk::calc_planewave_translation_matrix<DIM>(1, boxsize[i_level], n_pw, ts, wpwshift);

        ndview<const double, 1> fourier_data_view(&fourier_data.dkernelft[(i_level + 1) * (fourier_data.n_fourier + 1)],
                                                  fourier_data.n_fourier + 1);
        ndview<double, 1> radialft_view(&radialft[0], nexp);
        dmk::util::mk_tensor_product_fourier_transform(DIM, n_pw, fourier_data_view, radialft_view);

        // Form outgoing expansions
        for (auto box : level_indices[i_level]) {
            if (!form_pw_expansion[box])
                continue;

            // Form the outgoing expansion Φl(box) for the difference kernel Dl from the proxy charge expansion
            // coefficients using Tprox2pw.
            ndview<const std::complex<double>, 2> poly2pw_view(&poly2pw[0], fourier_data.n_pw, n_order);
            dmk::proxy::proxycharge2pw<double, DIM>(proxy_view_upward(box), poly2pw_view, pw_out_view(box));

            multiply_kernelFT_cd2p<T, DIM>(pw_out_view(box), radialft);
            memcpy(pw_in_ptr(box), pw_out_ptr(box), n_pw_per_box * sizeof(std::complex<T>));
        }

        // Form incoming expansions
        for (auto box : level_indices[i_level]) {
            if (src_counts_global[box] + trg_counts_global[box] == 0)
                continue;
            for (auto &neighbor : node_lists[box].nbr) {
                if (neighbor < 0 || neighbor == box || !form_pw_expansion[neighbor])
                    continue;

                // Translate the outgoing expansion Φl(colleague) to the center of box and add to the incoming plane
                // wave expansion Ψl(box) using wpwshift.

                // note: neighbors in SCTL are sorted in reverse order to wpwshift
                // FIXME: check if valid for periodic boundary conditions
                const int ind = sctl::pow<3>(dim) - 1 - (&neighbor - &node_lists[box].nbr[0]);
                shift_planewave(nd_in, nexp, pw_out_ptr(neighbor), pw_in_ptr(box), &wpwshift[n_pw_per_box * ind]);
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
            dmk::planewave_to_proxy_potential(dim, nd, n_order, n_pw, pw_in_ptr(box), &pw2poly[0],
                                              proxy_ptr_downward(box));

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
                        proxy::eval_targets<T, DIM>(proxy_view_downward(box), r_src_view(child), center_view(child), sc,
                                                    pot_src_view(child));
                    if (trg_counts_local[child])
                        proxy::eval_targets<T, DIM>(proxy_view_downward(box), r_trg_view(child), center_view(child), sc,
                                                    pot_trg_view(child));
                } else if (eval_pw_expansion[child]) {
                    dmk::tensorprod::transform(dim, nd, n_order, n_order, true, proxy_ptr_downward(box),
                                               &p2c[i_child * DIM * n_order * n_order], proxy_ptr_downward(child));
                }
            }
        }

        const double rsc = 2.0 / boxsize[i_level];
        const double cen = -1.0;
        const double d2max2 = boxsize[i_level] * boxsize[i_level];
        if (params.kernel != DMK_YUKAWA)
            throw std::runtime_error("Only yukawa potential supported");
        // FIXME: more than yukawa...
        const T w0 = fourier_data.yukawa_windowed_kernel_value_at_zero(i_level);
        for (auto box : level_indices[i_level]) {
            // Evaluate the direct interactions
            if (r_src_cnt[box] == 0)
                continue;

            const int n_src = r_src_cnt[box];
            const int ifself = 1;
            const int ifcharge = params.use_charge;
            const int ifdipole = 0;
            const int one = 1;
            for (auto neighbor : node_lists[box].nbr) {
                if (neighbor < 0)
                    continue;
                const int n_src_neighb = src_counts_local[neighbor];
                const int n_trg_neighb = trg_counts_local[neighbor];

                if (n_src_neighb) {
                    Eigen::MatrixX<T> r_src_transposed =
                        Eigen::Map<Eigen::MatrixX<T>>(r_src_ptr(neighbor), dim, n_src_neighb).transpose();
                    pdmk_direct_c_(&nd, &dim, (int *)&params.kernel, &params.fparam, &n_digits, &rsc, &cen, &ifself,
                                   &fourier_data.ncoeffs1[i_level],
                                   &fourier_data.coeffs1[fourier_data.n_coeffs_max * i_level], &d2max2, &one, &n_src,
                                   r_src_ptr(box), &ifcharge, charge_ptr(box), &ifdipole, nullptr, &one, &n_src_neighb,
                                   &n_src_neighb, r_src_transposed.data(), (int *)&params.pgh_src,
                                   pot_src_ptr(neighbor), nullptr, nullptr);
                }
                if (n_trg_neighb) {
                    Eigen::MatrixX<T> r_trg_transposed =
                        Eigen::Map<Eigen::MatrixX<T>>(r_trg_ptr(neighbor), dim, n_trg_neighb).transpose();
                    pdmk_direct_c_(&nd, &dim, (int *)&params.kernel, &params.fparam, &n_digits, &rsc, &cen, &ifself,
                                   &fourier_data.ncoeffs1[i_level],
                                   &fourier_data.coeffs1[fourier_data.n_coeffs_max * i_level], &d2max2, &one, &n_src,
                                   r_src_ptr(box), &ifcharge, charge_ptr(box), &ifdipole, nullptr, &one, &n_trg_neighb,
                                   &n_trg_neighb, r_trg_transposed.data(), (int *)&params.pgh_trg,
                                   pot_trg_ptr(neighbor), nullptr, nullptr);
                }
            }

            // Correct for self-evaluations
            for (int i = 0; i < nd; ++i)
                for (int i_src = 0; i_src < n_src; ++i_src)
                    pot_src_ptr(box)[i * n_src + i_src] -= w0 * charge_ptr(box)[i * n_src + i_src];
        }
    }
}

// template struct DMKPtTree<float, 2>;
// template struct DMKPtTree<float, 3>;
template struct DMKPtTree<double, 2>;
template struct DMKPtTree<double, 3>;

} // namespace dmk
