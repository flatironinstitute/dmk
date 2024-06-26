#include <dmk/chebychev.hpp>
#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/logger.h>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/tree.hpp>
#include <dmk/util.hpp>
#include <sctl/tree.hpp>

#include <mpi.h>

namespace dmk {

template <typename T, int DIM>
void DMKPtTree<T, DIM>::generate_metadata(int ndiv, int nd) {
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
        pot_offsets[i_node] = pot_offsets[i_node - 1] + nd * pot_cnt[i_node - 1];
        charge_offsets[i_node] = charge_offsets[i_node - 1] + nd * charge_cnt[i_node - 1];
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
    boxsize.resize(max_depth);
    boxsize[0] = 1.0;
    for (int i = 1; i < max_depth; ++i)
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
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::build_proxy_charges(int n_mfm, int n_order, const sctl::Vector<T> &c2p) {
    auto &logger = dmk::get_logger();
    auto &rank_logger = dmk::get_rank_logger();
    this->GetData(r_src_sorted, r_src_cnt, "pdmk_src");

    const int n_coeffs = n_mfm * sctl::pow<DIM>(n_order);
    proxy_coeffs.ReInit(n_boxes() * n_coeffs);
    proxy_coeffs.SetZero();
    sctl::Vector<sctl::Long> counts(n_boxes());
    counts.SetZero();

    constexpr int n_children = 1u << DIM;
    const auto &node_lists = this->GetNodeLists();
    const auto &attrs = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    int n_direct = 0;
    for (int i_box = 0; i_box < n_boxes(); ++i_box) {
        if (r_src_cnt[i_box]) {
            proxy::charge2proxycharge(DIM, n_mfm, n_order, r_src_cnt[i_box], r_src_ptr(i_box), charge_ptr(i_box),
                                      center_ptr(i_box), 2.0 / boxsize[i_box], &proxy_coeffs[i_box * n_coeffs]);
            counts[i_box] = 1;
            n_direct++;
        }
    }
    logger->debug("proxy: finished building leaf proxy charges");

    int n_merged = 0;
    for (int i_level = n_levels() - 1; i_level >= 0; --i_level) {
        for (auto parent_box : this->level_indices[i_level]) {
            auto &children = node_lists[parent_box].child;
            for (int i_child = 0; i_child < n_children; ++i_child) {
                const int child_box = children[i_child];
                if (child_box < 0 || !counts[child_box])
                    continue;

                constexpr bool add_flag = true;
                auto before = proxy_coeffs[parent_box * n_coeffs];
                tensorprod::transform(DIM, n_mfm, n_order, n_order, add_flag, &proxy_coeffs[child_box * n_coeffs],
                                      &c2p[i_child * DIM * n_order * n_order], &proxy_coeffs[parent_box * n_coeffs]);
                counts[parent_box] = 1;
                n_merged += 1;
            }
        }
    }
    int tot_proxy = 0;
    for (auto &count : counts) {
        tot_proxy += count;
        count = n_coeffs;
    }

    logger->debug("Finished building proxy charges for non-leaf boxes");
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

template <typename T, int DIM>
void DMKPtTree<T, DIM>::downward_pass(const pdmk_params &params, int n_order, const FourierData<T> &fourier_data,
                                      const sctl::Vector<T> &p2c) {
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
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * nd_out;
    const int n_coeffs_per_box = params.n_mfm * sctl::pow<DIM>(n_order);

    sctl::Vector<std::complex<T>> pw_out(n_pw_per_box * n_boxes());
    sctl::Vector<std::complex<T>> pw_in(n_pw_per_box * n_boxes());
    proxy_coeffs_downward.ReInit(proxy_coeffs.Dim());
    proxy_coeffs_downward.SetZero();
    pw_out.SetZero();
    pw_in.SetZero();

    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        constexpr int nmax = 1;
        sctl::Vector<std::complex<T>> wpwshift(n_pw_modes * sctl::pow<DIM>(2 * nmax + 1));
        sctl::Vector<T> ts(n_pw);
        sctl::Vector<T> rk(DIM * sctl::pow<DIM>(n_pw));
        const int shift = n_pw / 2;
        for (int i = 0; i < n_pw; ++i)
            ts[i] = fourier_data.hpw[i_level] * (i - shift);
        const int dim = DIM;
        meshnd_(&dim, &ts[0], &n_pw, &rk[0]);
        fourier_data.calc_planewave_coeff_matrices(i_level, n_order, poly2pw, pw2poly);
        mk_pw_translation_matrices_(&dim, &boxsize[i_level], &n_pw, &ts[0], &nmax, (T *)&wpwshift[0]);

        // Form outgoing expansions
        for (auto box : level_indices[i_level]) {
            // Form the outgoing expansion Φl(box) for the difference kernel Dl from the proxy charge expansion
            // coefficients using Tprox2pw.
            dmk::proxy::proxycharge2pw(DIM, nd_out, n_order, fourier_data.n_pw,
                                       &proxy_coeffs[box * sctl::pow<DIM>(n_order)], &poly2pw[0],
                                       &pw_out[box * n_pw_per_box]);
            memcpy(&pw_in[box * n_pw_per_box], &pw_out[box * n_pw_per_box], n_pw_per_box * sizeof(std::complex<T>));
        }

        // Form incoming expansions
        for (auto box : level_indices[i_level]) {
            for (auto neighbor : node_lists[box].nbr) {
                if (neighbor < 0 || neighbor == box)
                    continue;

                // Translate the outgoing expansion Φl(colleague) to the center of box and add to the incoming plane
                // wave expansion Ψl(box) using Tpwshift.
                constexpr int iperiod = 0;
                int ind;
                dmk_find_pwshift_ind_(&dim, &iperiod, &centers[box * DIM], &centers[neighbor * DIM], &boxsize[0],
                                      &boxsize[i_level], &nmax, &ind);
                ind--; // fortran uses 1 based indexing
                dmk_shiftpw_(&nd_in, &n_pw, (double *)&pw_out[neighbor * n_pw_per_box],
                             (double *)&pw_in[box * n_pw_per_box],
                             (double *)&wpwshift[n_pw * sctl::pow<DIM>(2 * nmax + 1) * ind]);
            }
        }

        // Form local expansions
        for (auto box : level_indices[i_level]) {
            // if (!in_flag[box])
            //     continue;
            // Convert incoming plane wave expansion Ψl(box) to the local expansion Λl(box) using Tpw2poly
            dmk_pw2proxypot_(&dim, &nd, &n_order, &n_pw, (double *)&pw_in[box * n_pw_per_box], (double *)&pw2poly[0],
                             (double *)&proxy_coeffs_downward[box * n_coeffs_per_box]);

            // Translate and add the local expansion of Λl(box) to the local expansion of Λl(child).
            for (auto child : node_lists[box].child) {
                if (child < 0)
                    continue;
                dmk::tensorprod::transform(dim, nd, n_order, n_order, true,
                                           &proxy_coeffs_downward[box * n_coeffs_per_box], &p2c[0],
                                           &proxy_coeffs_downward[child * n_coeffs_per_box]);
            }
        }

        // Evaluation local expansions
        for (auto box : level_indices[i_level]) {
            if (!r_trg_cnt[box])
                continue;
            // Evaluate the mollified potential ufar L at each target x in box.
            const T sc = 2.0 / boxsize[i_level];
            const int n_trg = r_trg_cnt[box];
            pdmk_ortho_evalt_nd_(&dim, &nd, &n_order, &proxy_coeffs_downward[box * n_coeffs_per_box], &n_trg,
                                 r_trg_ptr(box), &centers[box * dim], &sc, pot_ptr(box));
        }
    }
}

// template struct DMKPtTree<float, 2>;
// template struct DMKPtTree<float, 3>;
template struct DMKPtTree<double, 2>;
template struct DMKPtTree<double, 3>;

} // namespace dmk
