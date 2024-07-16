#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/logger.h>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/tree.hpp>
#include <dmk/util.hpp>
#include <mpi.h>
#include <ranges>
#include <sctl/tree.hpp>
#include <stdexcept>

namespace dmk {

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
    this->GetData(pot_sorted, pot_cnt, "pdmk_pot");
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    const auto &node_lists = this->GetNodeLists();

    src_counts_local.ReInit(n_nodes);
    trg_counts_local.ReInit(n_nodes);

    r_src_offsets.resize(n_nodes);
    r_trg_offsets.resize(n_nodes);
    pot_offsets.resize(n_nodes);
    charge_offsets.resize(n_nodes);

    for (int i_node = 1; i_node < n_nodes; ++i_node) {
        r_src_offsets[i_node] = r_src_offsets[i_node - 1] + DIM * r_src_cnt[i_node - 1];
        r_trg_offsets[i_node] = r_trg_offsets[i_node - 1] + DIM * r_trg_cnt[i_node - 1];
        pot_offsets[i_node] = pot_offsets[i_node - 1] + params.n_mfm * pot_cnt[i_node - 1];
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

    for (int box = 0; box < n_nodes; ++box)
        form_pw_expansion[box] = !node_attr[box].Leaf;

    for (const auto &level_boxes : level_indices) {
        for (auto box : level_boxes) {
            for (auto neighbor : node_lists[box].nbr) {
                if (neighbor < 0)
                    continue;

                const int npts = src_counts_global[neighbor] + trg_counts_global[neighbor];
                if (form_pw_expansion[neighbor] && npts) {
                    eval_pw_expansion[box] = true;
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
    proxy_coeffs.ReInit(n_boxes() * n_coeffs);
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
                    proxy::charge2proxycharge(DIM, params.n_mfm, n_order, src_counts_local[child_box], r_src_ptr(child_box),
                                              charge_ptr(child_box), center_ptr(parent_box), 2.0 / boxsize[i_level],
                                              proxy_ptr_upward(parent_box));
                    counts[child_box] = 1;
                    n_direct++;
                }
            }
        }
    }
    int tot_proxy = 0;
    for (auto &count : counts) {
        tot_proxy += count;
        count = n_coeffs;
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

template <typename T>
void multiply_kernelFT_cd2p(int nd, int ndim, bool ifcharge, bool ifdipole, int nexp, std::complex<T> *pwexp,
                            const T *radialft, const T *rk) {
    sctl::Vector<std::complex<T>> pwexp1(nexp * nd);
    pwexp1.SetZero();

    if (ifcharge)
        pwexp1 = sctl::Vector<std::complex<T>>(nexp * nd, pwexp, false);

    if (ifdipole == 1) {
        for (int ind = 0; ind < nd; ++ind)
            for (int n = 0; n < nexp; ++n)
                for (int j = 0; j < ndim; ++j)
                    pwexp1[n + ind * nd] -=
                        pwexp[n + nd * (ind + ifcharge + j)] * rk[j + n * ndim] * std::complex<T>{0.0, 1.0};
    }

    for (int ind = 0; ind < nd; ++ind)
        for (int n = 0; n < nexp; ++n)
            pwexp[n + ind * nd] = pwexp1[n + ind * nd] * radialft[n];
}

/// @brief Perform the "downward pass"
///
/// Updates: proxy_coeffs_downward, tree 'pdmk_pot' particle data
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
/// @param[in] params User input pdmk params
/// @param[in] n_order Order of polynomial expansion (FIXME: Should just be fixed)
/// @param[in,out] fourier_data Various fourier data. Only changes work array (FIXME: lame doc)
/// @param[in] p2c [n_order, n_order, DIM, 2**DIM] Parent to child matrices used to pass parent proxy charges to their
/// children
template <typename T, int DIM>
void DMKPtTree<T, DIM>::downward_pass(FourierData<T> &fourier_data, const sctl::Vector<T> &p2c) {
    auto &logger = dmk::get_logger();
    auto &rank_logger = dmk::get_rank_logger();
    const int nd = params.n_mfm;

    const auto &node_lists = this->GetNodeLists();
    const auto &node_attr = this->GetNodeAttr();
    const auto xs = dmk::chebyshev::get_cheb_nodes<T>(n_order, -1.0, 1.0);
    sctl::Vector<std::complex<T>> poly2pw(n_order * fourier_data.n_pw), pw2poly(n_order * fourier_data.n_pw);

    const int nd_in = params.n_mfm;
    const int nd_out = params.n_mfm;
    const int n_pw = fourier_data.n_pw;
    const std::size_t n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const std::size_t n_pw_per_box = n_pw_modes * nd_out;
    const std::size_t n_coeffs_per_box = params.n_mfm * sctl::pow<DIM>(n_order);
    const int ndigits = std::round(log10(1.0 / params.eps) - 0.1);

    sctl::Vector<std::complex<T>> pw_out(n_pw_per_box * n_boxes());
    sctl::Vector<std::complex<T>> pw_in(n_pw_per_box * n_boxes());
    proxy_coeffs_downward.ReInit(proxy_coeffs.Dim());
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
    meshnd_(&dim, &ts[0], &n_pw, &rk[0]);
    mk_tensor_product_fourier_transform_(&dim, &n_pw, &fourier_data.n_fourier, &fourier_data.dkernelft[0], &nexp,
                                         &radialft[0]);
    fourier_data.calc_planewave_coeff_matrices(-1, n_order, poly2pw, pw2poly);

    dmk::proxy::proxycharge2pw(DIM, nd_out, n_order, fourier_data.n_pw, proxy_ptr_upward(0), &poly2pw[0], &pw_out[0]);
    constexpr int zero = 0;
    dmk_multiply_kernelft_cd2p_(&nd_out, &dim, &params.use_charge, &zero, &nexp, (double *)&pw_out[0], &radialft[0],
                                &rk[0]);
    memcpy(&pw_in[0], &pw_out[0], n_pw_per_box * sizeof(std::complex<T>));
    dmk_pw2proxypot_(&dim, &nd, &n_order, &n_pw, (double *)&pw_in[0], (double *)&pw2poly[0], proxy_ptr_downward(0));

    constexpr int n_children = 1u << DIM;
    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        for (int i = 0; i < n_pw; ++i)
            ts[i] = fourier_data.hpw[i_level + 1] * (i - shift);
        meshnd_(&dim, &ts[0], &n_pw, &rk[0]);
        fourier_data.calc_planewave_coeff_matrices(i_level, n_order, poly2pw, pw2poly);
        mk_pw_translation_matrices_(&dim, &boxsize[i_level], &n_pw, &ts[0], &nmax, (T *)&wpwshift[0]);
        mk_tensor_product_fourier_transform_(&dim, &n_pw, &fourier_data.n_fourier,
                                             &fourier_data.dkernelft[(i_level + 1) * (fourier_data.n_fourier + 1)],
                                             &nexp, &radialft[0]);

        // Form outgoing expansions
        for (auto box : level_indices[i_level]) {
            if (!form_pw_expansion[box])
                continue;

            // Form the outgoing expansion Φl(box) for the difference kernel Dl from the proxy charge expansion
            // coefficients using Tprox2pw.
            dmk::proxy::proxycharge2pw(DIM, nd_out, n_order, fourier_data.n_pw, proxy_ptr_upward(box), &poly2pw[0],
                                       &pw_out[box * n_pw_per_box]);
            dmk_multiply_kernelft_cd2p_(&nd_out, &dim, &params.use_charge, &zero, &nexp,
                                        (double *)&pw_out[box * n_pw_per_box], &radialft[0], &rk[0]);
            memcpy(&pw_in[box * n_pw_per_box], &pw_out[box * n_pw_per_box], n_pw_per_box * sizeof(std::complex<T>));
        }

        // Form incoming expansions
        for (auto box : level_indices[i_level]) {
            if (src_counts_global[box] + trg_counts_global[box] == 0)
                continue;
            for (auto &neighbor : node_lists[box].nbr) {
                if (neighbor < 0 || neighbor == box)
                    continue;

                // Translate the outgoing expansion Φl(colleague) to the center of box and add to the incoming plane
                // wave expansion Ψl(box) using Tpwshift.
                constexpr int iperiod = 0;
                int ind;
                dmk_find_pwshift_ind_(&dim, &iperiod, center_ptr(box), center_ptr(neighbor), &boxsize[0],
                                      &boxsize[i_level], &nmax, &ind);
                ind--;
                // const int ind = sctl::pow<3>(dim) - 1 - (&neighbor - &node_lists[box].nbr[0]);
                dmk_shiftpw_(&nd_in, &nexp, (double *)&pw_out[neighbor * n_pw_per_box],
                             (double *)&pw_in[box * n_pw_per_box], (double *)&wpwshift[n_pw_per_box * ind]);
            }
        }

        // Form local expansions
        const T sc = 2.0 / boxsize[i_level];
        for (auto box : level_indices[i_level]) {
            const int n_trg = trg_counts_local[box];
            const int n_src = src_counts_local[box];
            const int n_pts = n_src + n_trg;

            if (eval_pw_expansion[box] && n_pts) {
                // Convert incoming plane wave expansion Ψl(box) to the local expansion Λl(box) using Tpw2poly
                dmk_pw2proxypot_(&dim, &nd, &n_order, &n_pw, (double *)&pw_in[box * n_pw_per_box],
                                 (double *)&pw2poly[0], proxy_ptr_downward(box));

                if (eval_tp_expansion[box])
                    pdmk_ortho_evalt_nd_(&dim, &nd, &n_order, proxy_ptr_downward(box), &n_trg, r_trg_ptr(box),
                                         center_ptr(box), &sc, pot_ptr(box));

                // Translate and add the local expansion of Λl(box) to the local expansion of Λl(child).
                for (int i_child = 0; i_child < n_children; ++i_child) {
                    const int child = node_lists[box].child[i_child];
                    if (child < 0)
                        continue;

                    if (eval_tp_expansion[child] && !eval_pw_expansion[child]) {
                        int n_trg_child = trg_counts_local[child];
                        pdmk_ortho_evalt_nd_(&dim, &nd, &n_order, proxy_ptr_downward(box), &n_trg_child,
                                             r_trg_ptr(child), center_ptr(child), &sc, pot_ptr(child));
                    } else if (eval_pw_expansion[child]) {
                        dmk::tensorprod::transform(dim, nd, n_order, n_order, true, proxy_ptr_downward(box),
                                                   &p2c[i_child * DIM * n_order * n_order], proxy_ptr_downward(child));
                    }
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
                if (neighbor < 0 || trg_counts_local[neighbor] == 0)
                    continue;
                const int n_trg = trg_counts_local[neighbor];

                Eigen::MatrixX<T> r_trg_transposed =
                    Eigen::Map<Eigen::MatrixX<T>>(r_trg_ptr(neighbor), dim, n_trg).transpose();
                pdmk_direct_c_(&nd, &dim, (int *)&params.kernel, &params.fparam, &ndigits, &rsc, &cen, &ifself,
                               &fourier_data.ncoeffs1[i_level],
                               &fourier_data.coeffs1[fourier_data.n_coeffs_max * i_level], &d2max2, &one, &n_src,
                               r_src_ptr(box), &ifcharge, charge_ptr(box), &ifdipole, nullptr, &one, &n_trg, &n_trg,
                               r_trg_transposed.data(), (int *)&params.pgh, pot_ptr(neighbor), nullptr, nullptr);
            }

            // Correct for self-evaluations
            for (int i = 0; i < nd; ++i)
                for (int i_src = 0; i_src < n_src; ++i_src)
                    pot_ptr(box)[i * n_src + i_src] -= w0 * charge_ptr(box)[i * n_src + i_src];
        }
    }
}

// template struct DMKPtTree<float, 2>;
// template struct DMKPtTree<float, 3>;
template struct DMKPtTree<double, 2>;
template struct DMKPtTree<double, 3>;

} // namespace dmk
