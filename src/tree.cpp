#include <cmath>
#include <dmk.h>
#include <dmk/chebychev.hpp>
#include <dmk/direct.hpp>
#include <dmk/fortran.h>
#include <dmk/fourier_data.hpp>
#include <dmk/legeexps.hpp>
#include <dmk/logger.h>
#include <dmk/planewave.hpp>
#include <dmk/prolate0_fun.hpp>
#include <dmk/proxy.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/testing.hpp>
#include <dmk/tree.hpp>
#include <dmk/types.hpp>
#include <dmk/util.hpp>
#include <sctl/profile.hpp>
#include <stdexcept>
#include <unistd.h>

#ifdef DMK_HAVE_OPENMP
#include <omp.h>
#endif

namespace dmk {

inline auto ghostleaf_or_ghost_children(int box, auto &node_attr, auto &node_lists) {
    if (node_attr[box].Leaf && node_attr[box].Ghost)
        return true;

    if (!node_attr[box].Leaf) {
        for (auto child : node_lists[box].child) {
            if (child < 0)
                continue;
            if (!node_attr[child].Ghost)
                return false;
        }

        return true;
    }

    return false;
}

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
    sctl::Profile::Scoped profile("DMKPtTree::DMKPtTree", &comm_);

    logger->info("tree build started");

    const int n_src = r_src.Dim() / DIM;
    const int n_trg = r_trg.Dim() / DIM;

    // 0: Initialization
    sctl::Vector<Real> pot_vec_src(n_src * params.n_mfm);
    sctl::Vector<Real> pot_vec_trg(n_trg * params.n_mfm);

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

    logger->debug("base tree build completed");
    logger->debug("generating tree traversal metadata");
    generate_metadata();
    logger->debug("done generating tree traversal metadata");

    rank_logger->trace("local tree has {} levels and {} boxes", n_levels(), n_boxes());

    // 1: Precomputation
    logger->debug("generating p2c and c2p matrices of order {}", n_order);
    std::tie(c2p, p2c) = dmk::chebyshev::get_c2p_p2c_matrices<Real>(DIM, n_order);
    logger->debug("finished generating matrices");

    fourier_data = FourierData<Real>(params.kernel, DIM, params.eps, n_digits, n_pw_max, params.fparam, boxsize);
    const auto &wk = fourier_data.windowed_kernel();
    logger->debug("planewave params at root box: n: {}, stepsize: {}, weight: {}, radius: {}", fourier_data.n_pw(),
                  wk.hpw, wk.ws, wk.rl);
    fourier_data.update_local_coeffs(params.eps);
    logger->debug("finished updating local potential expansion coefficients");
    logger->info("tree build completed");
}

/// @brief Build any bookkeeping data associated with the tree
///
/// This must be called after the constructor (FIXME: should be done at construction)
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename T, int DIM>
void DMKPtTree<T, DIM>::generate_metadata() {
    sctl::Profile::Scoped profile("generate_metadata", &comm_);
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

    level_indices_outgoing.ReInit(n_levels());
    level_indices_incoming.ReInit(n_levels());
    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        const auto &level_boxes = level_indices[i_level];
        for (auto box : level_boxes) {
            if (form_pw_expansion[box])
                level_indices_outgoing[i_level].PushBack(box);

            if (eval_pw_expansion[box] && (src_counts_local[box] + trg_counts_local[box]))
                level_indices_incoming[i_level].PushBack(box);

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

    direct_neighbs_flipped_.ReInit(n_nodes);
    n_direct_neighbs_flipped_.ReInit(n_nodes);
    n_direct_neighbs_flipped_.SetZero();
    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        for (int box : level_indices[i_level]) {
            if (!r_src_cnt[box])
                continue;

            int i_neighb = 0;
            for (auto neighb : node_lists[box].nbr)
                if (neighb >= 0) {
                    if (node_attr[neighb].Leaf && !node_attr[neighb].Ghost)
                        direct_neighbs_flipped_[neighb][n_direct_neighbs_flipped_[neighb]++] = box;
                    else {
                        for (auto child : node_lists[neighb].child) {
                            if (child < 0 || node_attr[child].Ghost)
                                continue;
                            direct_neighbs_flipped_[child][n_direct_neighbs_flipped_[child]++] = box;
                        }
                    }
                }

            if (i_level == 0)
                continue;

            const double cutoff = 0.5 * 1.05 * (boxsize[i_level] + boxsize[i_level - 1]);
            for (auto neighb : node_lists[node_lists[box].parent].nbr) {
                if (neighb < 0 || !node_attr[neighb].Leaf || node_attr[neighb].Ghost)
                    continue;

                bool inrange = true;
                for (int k = 0; k < DIM; ++k) {
                    const double distance = std::abs(center_ptr(box)[k] - center_ptr(neighb)[k]);
                    if (distance > cutoff)
                        inrange = false;
                }
                if (inrange)
                    direct_neighbs_flipped_[neighb][n_direct_neighbs_flipped_[neighb]++] = box;
            }
        }
    }

    sctl::Vector<sctl::Long> counts(n_boxes());
    const int n_coeffs = params.n_mfm * sctl::pow<DIM>(n_order);
    for (int i = 0; i < n_boxes(); ++i)
        counts[i] = form_pw_expansion[i] ? n_coeffs : 0;

    proxy_coeffs.ReInit(n_coeffs * n_proxy_boxes_upward);
    this->AddData("proxy_coeffs", proxy_coeffs, counts);
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
    sctl::Profile::Scoped profile("upward_pass", &comm_);
    sctl::Profile::Tic("upward_pass_init", &comm_);
    auto &logger = dmk::get_logger(comm_);
    auto &rank_logger = dmk::get_rank_logger(comm_);
    const std::size_t n_coeffs = params.n_mfm * sctl::pow<DIM>(n_order);
    logger->info("upward pass started");

#pragma omp parallel
#pragma omp single
    workspaces_.ReInit(omp_get_num_threads());

    sctl::Vector<sctl::Long> counts;
    this->GetData(proxy_coeffs, counts, "proxy_coeffs");
    proxy_coeffs.SetZero();

    constexpr int n_children = 1u << DIM;
    const auto &node_lists = this->GetNodeLists();
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    const int dim = DIM;

    sctl::Profile::Toc();
    const int start_level = std::max(n_levels() - 2, 0);
    {
        sctl::Profile::Scoped profile("charge2proxy_base", &comm_);

#pragma omp parallel
        {
            sctl::Vector<T> &workspace = workspaces_[omp_get_thread_num()];

#pragma omp for schedule(dynamic)
            for (auto i_box : level_indices[start_level]) {
                if (!form_pw_expansion[i_box] || !src_counts_local[i_box] || node_attr[i_box].Ghost)
                    continue;

                proxy::charge2proxycharge<T, DIM>(r_src_view(i_box), charge_view(i_box), center_view(i_box),
                                                  2.0 / boxsize[start_level], proxy_view_upward(i_box), workspace);
            }
        }
        logger->debug("proxy: finished building base proxy charges");
    }

    {
        sctl::Profile::Scoped profile("charge2proxy_rest", &comm_);
#pragma omp parallel
        {
            sctl::Vector<T> &workspace = workspaces_[omp_get_thread_num()];

            for (int i_level = start_level - 1; i_level >= 0; --i_level) {
#pragma omp for schedule(dynamic)
                for (auto parent_box : level_indices[i_level]) {
                    if (!form_pw_expansion[parent_box])
                        continue;

                    auto &children = node_lists[parent_box].child;
                    for (int i_child = 0; i_child < n_children; ++i_child) {
                        const int child_box = children[i_child];
                        if (child_box < 0 || !src_counts_local[child_box])
                            continue;

                        if (form_pw_expansion[child_box]) {
                            const ndview<T, 2> c2p_view({n_order, DIM}, &c2p[i_child * DIM * n_order * n_order]);
                            tensorprod::transform<T, DIM>(params.n_mfm, true, proxy_view_upward(child_box), c2p_view,
                                                          proxy_view_upward(parent_box), workspace);
                        } else if (!node_attr[child_box].Ghost) {
                            proxy::charge2proxycharge<T, DIM>(r_src_view(child_box), charge_view(child_box),
                                                              center_view(parent_box), 2.0 / boxsize[i_level],
                                                              proxy_view_upward(parent_box), workspace);
                        }
                    }
                }
            }
        }
    }

    sctl::Profile::Tic("broadcast_proxy_coeffs", &comm_);
    logger->debug("Finished building proxy charges");
    this->template ReduceBroadcast<T>("proxy_coeffs");
    this->GetData(proxy_coeffs, counts, "proxy_coeffs");
    long last_offset = 0;
    for (int box = 0; box < n_boxes(); ++box) {
        if (counts[box]) {
            proxy_coeffs_offsets[box] = last_offset;
            last_offset += n_coeffs;
        } else
            proxy_coeffs_offsets[box] = -1;
    }
    sctl::Profile::Toc();
    logger->debug("proxy: finished broadcasting proxy charges");
    logger->info("upward pass finished");
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

template <typename Complex, int DIM>
void shift_planewave(const ndview<Complex, DIM + 1> &pwexp1_, ndview<Complex, DIM + 1> &pwexp2_,
                     const ndview<const Complex, 1> &wpwshift) {
    // Flatten our views
    const int nd = pwexp1_.extent(DIM);
    const int nexp = wpwshift.extent(0);
    dmk::ndview<const Complex, 2> pwexp1({nexp, nd}, pwexp1_.data());
    dmk::ndview<Complex, 2> pwexp2({nexp, nd}, pwexp2_.data());

    using ArrayMap = ndview<Complex, 1>;
    using ConstArrayMap = ndview<const Complex, 1>;
    ConstArrayMap wpwshift_view({nexp}, &wpwshift(0));
    for (int ind = 0; ind < nd; ++ind) {
        ConstArrayMap pw1_view({nexp}, &pwexp1(0, ind));
        ArrayMap pw2_view({nexp}, &pwexp2(0, ind));
        pw2_view += pw1_view * wpwshift_view;
    }
}

template <typename T, int DIM>
void DMKPtTree<T, DIM>::init_planewave_data() {
    sctl::Profile::Scoped profile("init_planewave_data", &comm_);
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * params.n_mfm;

    pw_out_offsets.ReInit(n_boxes());
    pw_out_offsets[0] = 0;
    int n_pw_boxes_out = 1;
    int n_pw_boxes_in = 1;
    for (int i_box = 1; i_box < n_boxes(); ++i_box) {
        pw_out_offsets[i_box] = pw_out_offsets[i_box - 1] + form_pw_expansion[i_box] * n_pw_per_box;
        n_pw_boxes_out += form_pw_expansion[i_box];
    }

    pw_out.ReInit(n_pw_per_box * n_pw_boxes_out);
    pw_out.SetZero();
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::form_outgoing_expansions(const sctl::Vector<int> &boxes,
                                                    const ndview<std::complex<Real>, 2> &poly2pw_view,
                                                    const sctl::Vector<Real> &radialft) {
#ifdef DMK_INSTRUMENT
    double dt = -omp_get_wtime();
#endif
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * params.n_mfm;

    // Form the outgoing expansion Φl(box) for the difference kernel Dl from the proxy charge expansion
    // coefficients using Tprox2pw
#pragma omp parallel
    {
        sctl::Vector<Real> &workspace = workspaces_[omp_get_thread_num()];

#pragma omp for schedule(static)
        for (auto box : boxes) {
            dmk::proxy::proxycharge2pw<Real, DIM>(proxy_view_upward(box), poly2pw_view, pw_out_view(box), workspace);
            multiply_kernelFT_cd2p<Real, DIM>(radialft, pw_out_view(box));
        }
    }

#ifdef DMK_INSTRUMENT
    dt += omp_get_wtime();
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::CUSTOM1, (unsigned long)(1e9 * dt));
#endif
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::form_eval_expansions(const sctl::Vector<int> &boxes,
                                                const sctl::Vector<std::complex<Real>> &wpwshift, Real boxsize,
                                                const ndview<std::complex<Real>, 2> &pw2poly_view,
                                                const sctl::Vector<Real> &p2c) {
#ifdef DMK_INSTRUMENT
    double dt = -omp_get_wtime();
#endif
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * params.n_mfm;
    const auto &node_lists = this->GetNodeLists();
    const auto &node_attr = this->GetNodeAttr();
    const Real sc = 2.0 / boxsize;
    const int nd = params.n_mfm;

    unsigned long n_shifts{0};
#pragma omp parallel
    {
        sctl::Vector<Real> &workspace = workspaces_[omp_get_thread_num()];
        sctl::Vector<std::complex<Real>> pw_in(n_pw_per_box);

        auto pw_in_view = [this, &pw_in]() {
            if constexpr (DIM == 2)
                return ndview<std::complex<Real>, DIM + 1>({n_pw, (n_pw + 1) / 2, params.n_mfm}, &pw_in[0]);
            else if constexpr (DIM == 3)
                return ndview<std::complex<Real>, DIM + 1>({n_pw, n_pw, (n_pw + 1) / 2, params.n_mfm}, &pw_in[0]);
        }();

#pragma omp for schedule(dynamic) reduction(+ : n_shifts)
        for (auto box : boxes) {
            if (form_pw_expansion[box])
                memcpy(&pw_in[0], pw_out_ptr(box), n_pw_per_box * sizeof(std::complex<Real>));
            else
                pw_in.SetZero();

            for (auto &neighbor : node_lists[box].nbr) {
                if (neighbor < 0 || neighbor == box || !form_pw_expansion[neighbor])
                    continue;

                // Translate the outgoing expansion Φl(colleague) to the center of box and add to the incoming plane
                // wave expansion Ψl(box) using wpwshift.

                // note: neighbors in SCTL are sorted in reverse order to wpwshift
                // FIXME: check if valid for periodic boundary conditions
                constexpr int n_neighbors = sctl::pow<DIM>(3);
                const int ind = n_neighbors - 1 - (&neighbor - &node_lists[box].nbr[0]);
                assert(ind >= 0 && ind < n_neighbors);

                ndview<const std::complex<Real>, 1> wpwshift_view({n_pw_per_box}, &wpwshift[n_pw_per_box * ind]);
                shift_planewave<std::complex<Real>, DIM>(pw_out_view(neighbor), pw_in_view, wpwshift_view);
                n_shifts++;
            }

            // Convert incoming plane wave expansion Ψl(box) to the local expansion Λl(box) using Tpw2poly
            dmk::planewave_to_proxy_potential<Real, DIM>(pw_in_view, pw2poly_view, proxy_view_downward(box), workspace);

            if (eval_tp_expansion[box] && !ghostleaf_or_ghost_children(box, node_attr, node_lists)) {
                if (src_counts_local[box])
                    proxy::eval_targets<Real, DIM>(proxy_view_downward(box), r_src_view(box), center_view(box), sc,
                                                   pot_src_view(box), workspace);
                if (trg_counts_local[box])
                    proxy::eval_targets<Real, DIM>(proxy_view_downward(box), r_trg_view(box), center_view(box), sc,
                                                   pot_trg_view(box), workspace);
                continue;
            }

            // Translate and add the local expansion of Λl(box) to the local expansion of Λl(child).
            constexpr int n_children = 1u << DIM;
            for (int i_child = 0; i_child < n_children; ++i_child) {
                const int child = node_lists[box].child[i_child];
                if (child < 0)
                    continue;

                if (eval_tp_expansion[child] && !eval_pw_expansion[child] &&
                    !ghostleaf_or_ghost_children(child, node_attr, node_lists)) {
                    if (src_counts_local[child])
                        proxy::eval_targets<Real, DIM>(proxy_view_downward(box), r_src_view(child), center_view(box),
                                                       sc, pot_src_view(child), workspace);
                    if (trg_counts_local[child])
                        proxy::eval_targets<Real, DIM>(proxy_view_downward(box), r_trg_view(child), center_view(box),
                                                       sc, pot_trg_view(child), workspace);
                } else if (eval_pw_expansion[child]) {
                    const ndview<Real, 2> p2c_view({n_order, DIM},
                                                   const_cast<Real *>(&p2c[i_child * DIM * n_order * n_order]));
                    tensorprod::transform<Real, DIM>(nd, true, proxy_view_downward(box), p2c_view,
                                                     proxy_view_downward(child), workspace);
                }
            }
        }
    }

    // 1 complex multiply (4 multiplies and 2 adds) and 1 complex add (2 adds) per plane wave component
    constexpr int flops_per_pw = 8;
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, n_shifts * flops_per_pw * n_pw_per_box);
#ifdef DMK_INSTRUMENT
    dt += omp_get_wtime();
    sctl::Profile::IncrementCounter(sctl::ProfileCounter::CUSTOM2, (unsigned long)(1e9 * dt));
#endif
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
        const Real dj0 = std::cyl_bessel_j(0, z);
        const Real dj1 = std::cyl_bessel_j(1, z);
        const Real tker = -(1 - dj0) / (xs[i] * xs[i]) + dfac * dj1 / xs[i];
        const Real fhat = tker * fval0 / psi0;
        fval += fhat * whts[i] * xs[i];
    }

    return fval;
}

template <typename Real, int DIM>
std::tuple<Real, Real, Real, Real, Real>
get_direct_interaction_constants(FourierData<Real> &fourier_data, dmk_ikernel kernel, int i_level, Real boxsize) {
    const double bsize = i_level == 0 ? 0.5 * boxsize : boxsize;
    const double d2max = bsize * bsize;
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

    if ((kernel == DMK_SQRT_LAPLACE && DIM == 3) || (kernel == DMK_LAPLACE && DIM == 2))
        return {bsize, 2.0 / (bsize * bsize), -1.0, d2max, w0};
    if (kernel == DMK_YUKAWA)
        return {bsize, 2.0 / bsize, -1.0, d2max, w0};

    return {bsize, 2.0 / bsize, -bsize / 2.0, d2max, w0};
}

template <typename Real, int DIM>
void DMKPtTree<Real, DIM>::evaluate_direct_interactions(const Real *r_src_t, const Real *r_trg_t) {
    sctl::Profile::Scoped profile("evaluate_direct_interactions", &comm_);
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    const auto &node_lists = this->GetNodeLists();
    Real bsize[SCTL_MAX_DEPTH], rsc[SCTL_MAX_DEPTH], cen[SCTL_MAX_DEPTH], d2max[SCTL_MAX_DEPTH], w0[SCTL_MAX_DEPTH];
    for (int i_level = 0; i_level < n_levels(); ++i_level)
        std::tie(bsize[i_level], rsc[i_level], cen[i_level], d2max[i_level], w0[i_level]) =
            get_direct_interaction_constants<Real, DIM>(fourier_data, params.kernel, i_level, boxsize[i_level]);

#pragma omp parallel for schedule(dynamic)
    for (int neighb = 0; neighb < n_boxes(); ++neighb) {
        const int n_src_neighb = src_counts_local[neighb];
        const int n_trg_neighb = trg_counts_local[neighb];
        if (!n_src_neighb && !n_trg_neighb)
            continue;

        for (auto box : direct_neighbs_flipped(neighb)) {
            const int i_level = node_mid[box].Depth();
            const auto &cheb_coeffs = fourier_data.cheb_coeffs(i_level);

            std::array<std::span<const Real>, DIM> r_trg;
            if (n_src_neighb) {
                for (int i = 0; i < DIM; ++i)
                    r_trg[i] = std::span<const Real>(r_src_t + (r_src_offsets[neighb] / DIM) + src_counts_local[0] * i,
                                                     n_src_neighb);

                direct_eval<Real, DIM>(params.kernel, r_src_view(box), r_trg, charge_view(box), cheb_coeffs,
                                       &params.fparam, rsc[i_level], cen[i_level], d2max[i_level], pot_src_view(neighb),
                                       n_digits);
            }
            if (n_trg_neighb) {
                for (int i = 0; i < DIM; ++i)
                    r_trg[i] = std::span<const Real>(r_trg_t + (r_trg_offsets[neighb] / DIM) + trg_counts_local[0] * i,
                                                     n_trg_neighb);

                direct_eval<Real, DIM>(params.kernel, r_src_view(box), r_trg, charge_view(box), cheb_coeffs,
                                       &params.fparam, rsc[i_level], cen[i_level], d2max[i_level], pot_trg_view(neighb),
                                       n_digits);
            }
        }

        if (node_attr[neighb].Ghost || !n_src_neighb)
            continue;
        // Correct for self-evaluations
        auto pot = pot_src_view(neighb);
        auto charge = charge_view(neighb);
        const auto neighb_depth = node_mid[neighb].Depth();
        for (int i_src = 0; i_src < r_src_cnt[neighb]; ++i_src)
            for (int i = 0; i < params.n_mfm; ++i)
                pot(i, i_src) -= w0[neighb_depth] * charge(i, i_src);
    }
}

/// @brief Perform the "downward pass"
///
/// Updates: proxy_coeffs_downward, tree 'pdmk_pot' particle data
///
/// @tparam T Floating point format to use (float, double)
/// @tparam DIM Spatial dimension tree lives in
template <typename T, int DIM>
void DMKPtTree<T, DIM>::downward_pass() {
    sctl::Profile::Scoped prof("downward_pass", &comm_);
    sctl::Profile::Tic("downward_pass_init", &comm_);
    auto &logger = dmk::get_logger(comm_);
    auto &rank_logger = dmk::get_rank_logger(comm_);
    logger->info("downward pass started");

    this->GetData(pot_src_sorted, pot_src_cnt, "pdmk_pot_src");
    this->GetData(pot_trg_sorted, pot_trg_cnt, "pdmk_pot_trg");
    pot_src_sorted.SetZero();
    pot_trg_sorted.SetZero();

    // FIXME: This should be assigned automatically at tree construction (fourier_data should be subobject)
    n_pw = fourier_data.n_pw();

    init_planewave_data();
    const long n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    sctl::Vector<std::complex<T>> wpwshift(n_pw_modes * sctl::pow<DIM>(3));
    sctl::Vector<T> radialft(n_pw_modes);
    sctl::Vector<T> kernel_ft;
    get_windowed_kernel_ft<T, DIM>(params.kernel, &params.fparam, fourier_data.beta(), n_digits, boxsize[0],
                                   fourier_data.prolate0_fun, kernel_ft);
    util::mk_tensor_product_fourier_transform(DIM, n_pw, ndview<T, 1>({kernel_ft.Dim()}, &kernel_ft[0]),
                                              ndview<T, 1>({n_pw_modes}, &radialft[0]));

    sctl::Vector<std::complex<T>> poly2pw(n_order * n_pw), pw2poly(n_order * n_pw);
    fourier_data.calc_planewave_coeff_matrices(-1, n_order, poly2pw, pw2poly);
    const ndview<std::complex<T>, 2> poly2pw_view({n_pw, n_order}, &poly2pw[0]);
    const ndview<std::complex<T>, 2> pw2poly_view({n_pw, n_order}, &pw2poly[0]);

    dmk::proxy::proxycharge2pw<T, DIM>(proxy_view_upward(0), poly2pw_view, pw_out_view(0), workspaces_[0]);
    multiply_kernelFT_cd2p<T, DIM>(radialft, pw_out_view(0));
    proxy_coeffs_downward.SetZero();
    dmk::planewave_to_proxy_potential<T, DIM>(pw_out_view(0), pw2poly_view, proxy_view_downward(0), workspaces_[0]);

    ndamatrix<T> r_src_t = nda::transpose(matrixview<T>({DIM, src_counts_local[0]}, r_src_ptr(0)));
    ndamatrix<T> r_trg_t = nda::transpose(matrixview<T>({DIM, trg_counts_local[0]}, r_trg_ptr(0)));

    sctl::Profile::Toc();
    sctl::Profile::Tic("expansion_propagation_and_eval", &comm_);
    for (int i_level = 0; i_level < n_levels(); ++i_level) {
        // Initialize everything for this level
        // 1. Difference kernel
        // 2. Radial fourier transform of the difference kernel
        // 3. Planewave <-> polynomial coefficient conversion matrices
        // 4. Planewave translation matrix
        {
            // sctl::Profile::Scoped profile("downward_pass_loop_init", &comm_);
            const bool is_root = i_level == 0;
            get_difference_kernel_ft<T, DIM>(is_root, params.kernel, &params.fparam, fourier_data.beta(), n_digits,
                                             boxsize[i_level], fourier_data.prolate0_fun, kernel_ft);
            util::mk_tensor_product_fourier_transform(DIM, n_pw, ndview<T, 1>({kernel_ft.Dim()}, &kernel_ft[0]),
                                                      ndview<T, 1>({n_pw_modes}, &radialft[0]));
            fourier_data.calc_planewave_coeff_matrices(i_level, n_order, poly2pw, pw2poly);
            dmk::calc_planewave_translation_matrix<DIM>(1, boxsize[i_level], n_pw,
                                                        fourier_data.difference_kernel(i_level).hpw, wpwshift);
        }
        form_outgoing_expansions(level_indices_outgoing[i_level], poly2pw_view, radialft);
        form_eval_expansions(level_indices_incoming[i_level], wpwshift, boxsize[i_level], pw2poly_view, p2c);
    }
    sctl::Profile::Toc();

    evaluate_direct_interactions(r_src_t.data(), r_trg_t.data());

    logger->info("downward pass completed");
}

#ifdef DMK_HAVE_MPI
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
#ifdef DMK_HAVE_MPI
    auto comm = sctl::Comm(test_comm);
#else
    auto comm = sctl::Comm::Self();
#endif
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
#endif

template struct DMKPtTree<float, 2>;
template struct DMKPtTree<float, 3>;
template struct DMKPtTree<double, 2>;
template struct DMKPtTree<double, 3>;

} // namespace dmk
