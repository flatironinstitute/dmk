#include <algorithm>
#include <complex>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <dmk.h>
#include <dmk/direct.hpp>
#ifdef DMK_BUILD_ESP
#include <dmk/esp.hpp>
#endif
#include <dmk/fourier_data.hpp>
#include <dmk/tensorprod.hpp>
#include <dmk/testing.hpp>
#include <dmk/tree.hpp>
#include <dmk/util.hpp>

#include <sctl.hpp>

#define VERBOSE_MESSAGE(...)                                                                                           \
    if (std::getenv("DMK_TEST_VERBOSE")) {                                                                             \
        MESSAGE(__VA_ARGS__);                                                                                          \
    }

namespace {
struct PbcRefSrc {
    double r[3];
    double charge;
    int level;
};
struct PbcRefTrg {
    double r[3];
    int level;
};

// Reference for evaluate_direct_interactions under PBC (Laplace 3D): sum DMK's own residual
// evaluator over the 3x3x3 periodic images. Pairs are grouped by max(src,trg) leaf level, which
// sets the box scaling (rsc = 2/bsize, cen = -bsize/2, d2max = bsize^2) exactly as tree.cpp does.
std::vector<double> pbc_direct_ref(const dmk::residual_evaluator_func<double> &eval,
                                   const std::vector<PbcRefSrc> &sources, const std::vector<PbcRefTrg> &targets,
                                   const sctl::Vector<double> &boxsize) {
    constexpr double thresh2 = 1e-30;
    std::vector<double> ref(targets.size(), 0.0);

    std::set<int> src_levels, trg_levels;
    for (const auto &s : sources)
        src_levels.insert(s.level);
    for (const auto &t : targets)
        trg_levels.insert(t.level);

    for (int sl : src_levels) {
        std::vector<double> src_r, src_c;
        for (const auto &s : sources)
            if (s.level == sl) {
                src_r.insert(src_r.end(), {s.r[0], s.r[1], s.r[2]});
                src_c.push_back(s.charge);
            }
        const int n_src = src_c.size();

        for (int tl : trg_levels) {
            std::vector<double> trg_r;
            std::vector<int> gidx;
            for (int i = 0; i < (int)targets.size(); ++i)
                if (targets[i].level == tl) {
                    trg_r.insert(trg_r.end(), {targets[i].r[0], targets[i].r[1], targets[i].r[2]});
                    gidx.push_back(i);
                }
            const int n_trg = gidx.size();
            if (!n_src || !n_trg)
                continue;

            const double bsize = boxsize[std::max(sl, tl)];
            const double rsc = 2.0 / bsize, cen = -bsize / 2.0, d2max = bsize * bsize;
            std::vector<double> pot(n_trg, 0.0), shifted(3 * n_src);
            for (int mx = -1; mx <= 1; ++mx)
                for (int my = -1; my <= 1; ++my)
                    for (int mz = -1; mz <= 1; ++mz) {
                        for (int i = 0; i < n_src; ++i) {
                            shifted[i * 3 + 0] = src_r[i * 3 + 0] + mx;
                            shifted[i * 3 + 1] = src_r[i * 3 + 1] + my;
                            shifted[i * 3 + 2] = src_r[i * 3 + 2] + mz;
                        }
                        eval(rsc, cen, d2max, thresh2, n_src, shifted.data(), src_c.data(), nullptr, n_trg,
                             trg_r.data(), pot.data());
                    }
            for (int k = 0; k < n_trg; ++k)
                ref[gidx[k]] += pot[k];
        }
    }
    return ref;
}
} // namespace

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace PBC direct verification", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 500;

#ifdef DMK_HAVE_MPI
    auto sctl_comm = sctl::Comm(test_comm);
#else
    auto sctl_comm = sctl::Comm::Self();
#endif

    std::default_random_engine eng(42);
    std::uniform_real_distribution<double> rng(0.01, 0.99);

    sctl::Vector<double> r_src(n_dim * n_src), r_trg(n_dim * n_trg);
    sctl::Vector<double> charges(n_src);
    sctl::Vector<double> normals;

    for (int i = 0; i < n_src * n_dim; ++i)
        r_src[i] = rng(eng);
    for (int i = 0; i < n_trg * n_dim; ++i)
        r_trg[i] = rng(eng);
    for (int i = 0; i < n_src; ++i)
        charges[i] = rng(eng) - 0.5;
    {
        double sum = 0.0;
        for (int i = 0; i < n_src; ++i)
            sum += charges[i];
        for (int i = 0; i < n_src; ++i)
            charges[i] -= sum / n_src;
    }

    struct PrecisionCase {
        int n_digits;
        double eps;
    };
    const PrecisionCase cases[] = {{3, 1e-3}, {6, 1e-6}, {9, 1e-9}, {12, 1e-12}};

    for (const auto &pc : cases) {
        SUBCASE(("n_digits=" + std::to_string(pc.n_digits)).c_str()) {
            pdmk_params params;
            params.eps = pc.eps;
            params.n_dim = n_dim;
            params.n_per_leaf = 280;
            params.eval_src = DMK_POTENTIAL;
            params.eval_trg = DMK_POTENTIAL;
            params.kernel = DMK_LAPLACE;
            params.use_periodic = true;
            params.log_level = 6;

            dmk::DMKPtTree<double, n_dim> tree(sctl_comm, params, r_src, charges, normals, r_trg);

            tree.pot_src_sorted.SetZero();
            tree.pot_trg_sorted.SetZero();
            tree.evaluate_direct_interactions();

            const auto &node_mid = tree.GetNodeMID();
            const auto &node_attr = tree.GetNodeAttr();

            std::vector<PbcRefSrc> sources;
            for (int box = 0; box < tree.n_boxes(); ++box) {
                if (!node_attr[box].Leaf || node_attr[box].Ghost)
                    continue;
                const int n = tree.r_src_cnt_owned[box];
                if (!n)
                    continue;
                const int level = node_mid[box].Depth();
                const double *rp = tree.r_src_owned_ptr(box);
                const double *cp = tree.charge_owned_ptr(box);
                for (int i = 0; i < n; ++i)
                    sources.push_back({{rp[i * 3], rp[i * 3 + 1], rp[i * 3 + 2]}, cp[i], level});
            }

            std::vector<PbcRefTrg> targets;
            std::vector<int> pot_offsets;
            for (int box = 0; box < tree.n_boxes(); ++box) {
                if (node_attr[box].Ghost)
                    continue;
                const int n = tree.r_trg_cnt_owned[box];
                if (!n)
                    continue;
                const int level = node_mid[box].Depth();
                const double *rp = tree.r_trg_owned_ptr(box);
                for (int i = 0; i < n; ++i) {
                    targets.push_back({{rp[i * 3], rp[i * 3 + 1], rp[i * 3 + 2]}, level});
                    pot_offsets.push_back((int)tree.pot_trg_offsets[box] + i);
                }
            }

            auto eval = dmk::make_evaluator_aot<double>(DMK_LAPLACE, DMK_POTENTIAL, n_dim, pc.n_digits, 3);
            const std::vector<double> ref_pot = pbc_direct_ref(eval, sources, targets, tree.boxsize);

            const int n_test = std::min((int)targets.size(), 200);
            double err2 = 0.0, ref2 = 0.0;
            for (int i = 0; i < n_test; ++i) {
                const double tree_val = tree.pot_trg_sorted[pot_offsets[i]];
                err2 += sctl::pow<2>(tree_val - ref_pot[i]);
                ref2 += sctl::pow<2>(ref_pot[i]);
            }
            const double l2_err = (ref2 > 0) ? std::sqrt(err2 / ref2) : std::sqrt(err2);
            VERBOSE_MESSAGE("n_digits=", pc.n_digits, " eps=", pc.eps, " l2_err=", l2_err,
                            " n_levels=", tree.n_levels(), " n_boxes=", tree.n_boxes());
            const double tol = (pc.n_digits <= 3) ? 1e-3 : 1e-6;
            CHECK(l2_err < tol);
        }
    }
}

// Regression test for PBC + asymmetric-depth list-1 interactions.
// When source particles are shifted by a periodic image vector but the
// ContactGeometry used for filtering still references the unshifted source-box
// corner, `filter_sources`/`filter_targets` test points against a geometry in a
// different coordinate frame. This only manifests when a list-1 pair has
// unequal depths (src_larger or trg_larger) AND crosses a periodic face, so a
// uniform distribution (where all leaves are same depth) cannot catch it.
//
// This test forces non-uniform refinement by clustering most sources tightly
// into one corner of the unit box. The deep cluster leaves at x~0 have list-1
// PBC neighbors at x~1 living in shallower leaves, triggering the buggy code
// path. Reference is computed by direct summation over 3x3x3 periodic images
// of the cluster sources only (the sparse filler sources have zero charge so
// contribute nothing); tree result must match within eps.
TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace PBC asymmetric-depth shift", 1) {
    constexpr int n_dim = 3;
    constexpr int n_cluster = 2000;
    constexpr int n_filler = 400;
    constexpr int n_src = n_cluster + n_filler;
    constexpr int n_trg = 400;

#ifdef DMK_HAVE_MPI
    auto sctl_comm = sctl::Comm(test_comm);
#else
    auto sctl_comm = sctl::Comm::Self();
#endif

    std::default_random_engine eng(7);
    // Sources: all clustered in x ∈ [0, 0.04] — forces deep refinement there.
    std::uniform_real_distribution<double> src_x(0.001, 0.04);
    std::uniform_real_distribution<double> src_yz(0.001, 0.999);
    // Filler sources scattered across the domain (zero charge) to keep the
    // uniform-side boxes from being pruned.
    std::uniform_real_distribution<double> filler(0.06, 0.999);
    // Targets: confined to x ∈ [0.7, 0.99] — far from cluster in direct sense,
    // but the PBC wrap brings the cluster (shifted by +1 in x) to x ∈ [1, 1.04]
    // which is adjacent to the target slab. This is exactly the buggy regime:
    // the trg_box (shallow) sees src_box (deep) via a nonzero periodic shift.
    std::uniform_real_distribution<double> trg_x(0.7, 0.99);
    std::uniform_real_distribution<double> trg_yz(0.001, 0.999);

    sctl::Vector<double> r_src(n_dim * n_src), r_trg(n_dim * n_trg);
    sctl::Vector<double> charges(n_src);
    sctl::Vector<double> normals;

    for (int i = 0; i < n_cluster; ++i) {
        r_src[i * n_dim + 0] = src_x(eng);
        r_src[i * n_dim + 1] = src_yz(eng);
        r_src[i * n_dim + 2] = src_yz(eng);
    }
    for (int i = n_cluster; i < n_src; ++i)
        for (int d = 0; d < n_dim; ++d)
            r_src[i * n_dim + d] = filler(eng);
    for (int i = 0; i < n_trg; ++i) {
        r_trg[i * n_dim + 0] = trg_x(eng);
        r_trg[i * n_dim + 1] = trg_yz(eng);
        r_trg[i * n_dim + 2] = trg_yz(eng);
    }
    std::uniform_real_distribution<double> chg(0.0, 1.0);
    for (int i = 0; i < n_cluster; ++i)
        charges[i] = chg(eng) - 0.5;
    for (int i = n_cluster; i < n_src; ++i)
        charges[i] = 0.0;
    {
        double sum = 0.0;
        for (int i = 0; i < n_cluster; ++i)
            sum += charges[i];
        for (int i = 0; i < n_cluster; ++i)
            charges[i] -= sum / n_cluster;
    }

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 40;
    params.eval_src = DMK_POTENTIAL;
    params.eval_trg = DMK_POTENTIAL;
    params.kernel = DMK_LAPLACE;
    params.use_periodic = true;
    params.log_level = 6;

    dmk::DMKPtTree<double, n_dim> tree(sctl_comm, params, r_src, charges, normals, r_trg);

    tree.pot_src_sorted.SetZero();
    tree.pot_trg_sorted.SetZero();
    tree.evaluate_direct_interactions();

    const auto &node_mid = tree.GetNodeMID();
    const auto &node_attr = tree.GetNodeAttr();

    std::vector<PbcRefSrc> sources;
    for (int box = 0; box < tree.n_boxes(); ++box) {
        if (!node_attr[box].Leaf || node_attr[box].Ghost)
            continue;
        const int n = tree.r_src_cnt_owned[box];
        if (!n)
            continue;
        const int level = node_mid[box].Depth();
        const double *rp = tree.r_src_owned_ptr(box);
        const double *cp = tree.charge_owned_ptr(box);
        for (int i = 0; i < n; ++i)
            sources.push_back({{rp[i * 3], rp[i * 3 + 1], rp[i * 3 + 2]}, cp[i], level});
    }

    std::vector<PbcRefTrg> targets;
    std::vector<int> pot_offsets;
    for (int box = 0; box < tree.n_boxes(); ++box) {
        if (node_attr[box].Ghost)
            continue;
        const int n = tree.r_trg_cnt_owned[box];
        if (!n)
            continue;
        const int level = node_mid[box].Depth();
        const double *rp = tree.r_trg_owned_ptr(box);
        for (int i = 0; i < n; ++i) {
            targets.push_back({{rp[i * 3], rp[i * 3 + 1], rp[i * 3 + 2]}, level});
            pot_offsets.push_back((int)tree.pot_trg_offsets[box] + i);
        }
    }

    // Confirm the distribution actually produced multi-depth leaves (otherwise
    // we aren't exercising the asymmetric code path).
    int min_level = 1 << 30, max_level = 0;
    for (const auto &s : sources) {
        min_level = std::min(min_level, s.level);
        max_level = std::max(max_level, s.level);
    }
    for (const auto &t : targets) {
        min_level = std::min(min_level, t.level);
        max_level = std::max(max_level, t.level);
    }
    VERBOSE_MESSAGE("non-uniform tree: min_leaf_level=", min_level, " max_leaf_level=", max_level,
                    " n_levels=", tree.n_levels(), " n_boxes=", tree.n_boxes());
    REQUIRE(max_level > min_level); // if this fails, tune cluster/filler to force asymmetry

    auto eval = dmk::make_evaluator_aot<double>(DMK_LAPLACE, DMK_POTENTIAL, n_dim, 6, 3);
    const std::vector<double> ref_pot = pbc_direct_ref(eval, sources, targets, tree.boxsize);

    const int n_test = (int)targets.size();
    double err2 = 0.0, ref2 = 0.0;
    double max_abs_err = 0.0;
    for (int i = 0; i < n_test; ++i) {
        const double tree_val = tree.pot_trg_sorted[pot_offsets[i]];
        const double diff = tree_val - ref_pot[i];
        err2 += diff * diff;
        ref2 += ref_pot[i] * ref_pot[i];
        max_abs_err = std::max(max_abs_err, std::abs(diff));
    }
    const double l2_err = (ref2 > 0) ? std::sqrt(err2 / ref2) : std::sqrt(err2);
    VERBOSE_MESSAGE("asymmetric-depth PBC: l2_err=", l2_err, " max_abs_err=", max_abs_err);
    CHECK(l2_err < 1e-5);
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace PBC single-level public API", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 8;
    constexpr int n_trg = 4;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    sctl::Vector<double> r_src({0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4,
                                0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9});
    sctl::Vector<double> r_trg({0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.65, 0.65, 0.65, 0.85, 0.85, 0.85});
    sctl::Vector<double> charges(n_src);
    sctl::Vector<double> normal(n_src * n_dim);
    sctl::Vector<double> pot_src(n_src);
    sctl::Vector<double> pot_trg(n_trg);

    normal.SetZero();
    for (int i = 0; i < n_src; ++i)
        charges[i] = (i % 2 == 0 ? 1.0 : -1.0) / n_src;

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 1000000;
    params.eval_src = DMK_POTENTIAL;
    params.eval_trg = DMK_POTENTIAL;
    params.kernel = DMK_LAPLACE;
    params.use_periodic = true;
    params.log_level = 6;

    pdmk_tree tree = pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &normal[0], n_trg, &r_trg[0]);
    pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
    pdmk_tree_destroy(tree);

    CHECK(pot_src.Dim() == n_src);
    CHECK(pot_trg.Dim() == n_trg);
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace PBC single-level root pw_out must be zeroed", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 8;
    constexpr int n_trg = 4;

#ifdef DMK_HAVE_MPI
    auto sctl_comm = sctl::Comm(test_comm);
#else
    auto sctl_comm = sctl::Comm::Self();
#endif

    sctl::Vector<double> r_src({0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4,
                                0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9});
    sctl::Vector<double> r_trg({0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.65, 0.65, 0.65, 0.85, 0.85, 0.85});
    sctl::Vector<double> charges(n_src);
    sctl::Vector<double> normals;
    for (int i = 0; i < n_src; ++i)
        charges[i] = (i % 2 == 0 ? 1.0 : -1.0) / n_src;

    pdmk_params params;
    params.eps = 1e-6;
    params.n_dim = n_dim;
    params.n_per_leaf = 1000000;
    params.eval_src = DMK_POTENTIAL;
    params.eval_trg = DMK_POTENTIAL;
    params.kernel = DMK_LAPLACE;
    params.use_periodic = true;
    params.log_level = 6;

    dmk::DMKPtTree<double, n_dim> clean_tree(sctl_comm, params, r_src, charges, normals, r_trg);
    clean_tree.eval();

    dmk::DMKPtTree<double, n_dim> poisoned_tree(sctl_comm, params, r_src, charges, normals, r_trg);
    REQUIRE(poisoned_tree.n_boxes() == 1);
    REQUIRE(poisoned_tree.n_levels() == 1);

    poisoned_tree.upward_pass();
    poisoned_tree.pot_src_sorted.SetZero();
    poisoned_tree.pot_trg_sorted.SetZero();
    poisoned_tree.init_planewave_data();
    REQUIRE(poisoned_tree.pw_out.Dim() > 0);

    const std::complex<double> poison(1.25, -0.75);
    std::fill(poisoned_tree.pw_out.begin(), poisoned_tree.pw_out.end(), poison);

    poisoned_tree.form_outgoing_expansions();

    const int n_pw = poisoned_tree.expansion_constants.n_pw_diff;
    const int n_order = poisoned_tree.expansion_constants.n_order;
    auto &dfd = poisoned_tree.difference_fourier_data[0];
    const dmk::ndview<std::complex<double>, 2> pw2p({n_pw, n_order}, &dfd.pw2poly[0]);
    poisoned_tree.form_eval_expansions(poisoned_tree.level_indices[0], dfd.wpwshift, poisoned_tree.boxsize[0], pw2p,
                                       poisoned_tree.p2c);
    poisoned_tree.evaluate_direct_interactions();

    double max_trg_diff = 0.0;
    for (int i = 0; i < poisoned_tree.pot_trg_sorted.Dim(); ++i)
        max_trg_diff = std::max(max_trg_diff, std::abs(poisoned_tree.pot_trg_sorted[i] - clean_tree.pot_trg_sorted[i]));

    double max_src_diff = 0.0;
    for (int i = 0; i < poisoned_tree.pot_src_sorted.Dim(); ++i)
        max_src_diff = std::max(max_src_diff, std::abs(poisoned_tree.pot_src_sorted[i] - clean_tree.pot_src_sorted[i]));

    VERBOSE_MESSAGE("single-level periodic root sensitivity: max_src_diff=", max_src_diff,
                    " max_trg_diff=", max_trg_diff);
    CHECK(max_src_diff == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(max_trg_diff == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Laplace PBC full pipeline vs Ewald", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 500;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    std::default_random_engine eng(99);
    std::uniform_real_distribution<double> rng(0.01, 0.99);

    sctl::Vector<double> r_src(n_dim * n_src), r_trg(n_dim * n_trg);
    sctl::Vector<double> charges(n_src);
    sctl::Vector<double> rnormal(n_dim * n_src);
    sctl::Vector<double> dipstr(n_src);
    rnormal.SetZero();
    dipstr.SetZero();

    for (int i = 0; i < n_src * n_dim; ++i)
        r_src[i] = rng(eng);
    for (int i = 0; i < n_trg * n_dim; ++i)
        r_trg[i] = rng(eng);
    for (int i = 0; i < n_src; ++i)
        charges[i] = rng(eng) - 0.5;
    {
        double sum = 0.0;
        for (int i = 0; i < n_src; ++i)
            sum += charges[i];
        for (int i = 0; i < n_src; ++i)
            charges[i] -= sum / n_src;
    }

    const double L = 1.0;
    const double V_box = L * L * L;
    const double dk = 2.0 * M_PI / L;
    const double alpha = 10.0;
    const double r_c = 0.5 * L;
    const int n_ewald = 15;
    const int d = 2 * n_ewald + 1;

    std::vector<std::complex<double>> rho(d * d * d, {0.0, 0.0});
    for (int is = 0; is < n_src; ++is) {
        const std::complex<double> ex0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 3 + 0]));
        const std::complex<double> ey0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 3 + 1]));
        const std::complex<double> ez0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 3 + 2]));
        std::vector<std::complex<double>> ex(d), ey(d), ez(d);
        for (int a = -n_ewald; a <= n_ewald; ++a) {
            ex[a + n_ewald] = std::pow(ex0, a);
            ey[a + n_ewald] = std::pow(ey0, a);
            ez[a + n_ewald] = std::pow(ez0, a);
        }
        for (int ix = 0; ix < d; ++ix)
            for (int iy = 0; iy < d; ++iy) {
                const auto t2 = charges[is] * ex[ix] * ey[iy];
                for (int iz = 0; iz < d; ++iz)
                    rho[ix * d * d + iy * d + iz] += t2 * ez[iz];
            }
    }

    auto ewald_pot_grad = [&](const double *r_eval, double &pot_out, double *grad_out) {
        pot_out = 0.0;
        if (grad_out)
            grad_out[0] = grad_out[1] = grad_out[2] = 0.0;

        for (int is = 0; is < n_src; ++is) {
            for (int mx = -1; mx <= 1; ++mx)
                for (int my = -1; my <= 1; ++my)
                    for (int mz = -1; mz <= 1; ++mz) {
                        const double dx = r_eval[0] - r_src[is * 3 + 0] - mx * L;
                        const double dy = r_eval[1] - r_src[is * 3 + 1] - my * L;
                        const double dz = r_eval[2] - r_src[is * 3 + 2] - mz * L;
                        const double r2 = dx * dx + dy * dy + dz * dz;
                        const double r = std::sqrt(r2);
                        if (r > 1e-15 && r <= r_c) {
                            pot_out += charges[is] * std::erfc(alpha * r) / r;
                            if (grad_out) {
                                const double scale = -charges[is] * (std::erfc(alpha * r) / (r * r2) +
                                                                     2.0 * alpha * std::exp(-alpha * alpha * r2) /
                                                                         (std::sqrt(M_PI) * r2));
                                grad_out[0] += scale * dx;
                                grad_out[1] += scale * dy;
                                grad_out[2] += scale * dz;
                            }
                        }
                    }
        }

        const std::complex<double> etx0 = std::exp(std::complex<double>(0.0, dk * r_eval[0]));
        const std::complex<double> ety0 = std::exp(std::complex<double>(0.0, dk * r_eval[1]));
        const std::complex<double> etz0 = std::exp(std::complex<double>(0.0, dk * r_eval[2]));
        std::vector<std::complex<double>> etx(d), ety(d), etz(d);
        for (int a = -n_ewald; a <= n_ewald; ++a) {
            etx[a + n_ewald] = std::pow(etx0, a);
            ety[a + n_ewald] = std::pow(ety0, a);
            etz[a + n_ewald] = std::pow(etz0, a);
        }

        double pot_long = 0.0;
        double grad_long[3] = {0, 0, 0};
        for (int nx = -n_ewald; nx <= n_ewald; ++nx)
            for (int ny = -n_ewald; ny <= n_ewald; ++ny)
                for (int nz = -n_ewald; nz <= n_ewald; ++nz) {
                    if (nx == 0 && ny == 0 && nz == 0)
                        continue;
                    const double kx = dk * nx;
                    const double ky = dk * ny;
                    const double kz = dk * nz;
                    const double k2 = kx * kx + ky * ky + kz * kz;
                    const double G = std::exp(-k2 / (4.0 * alpha * alpha)) / k2;
                    const int ix = nx + n_ewald;
                    const int iy = ny + n_ewald;
                    const int iz = nz + n_ewald;
                    const auto &rho_k = rho[ix * d * d + iy * d + iz];
                    const auto eikr = etx[nx + n_ewald] * ety[ny + n_ewald] * etz[nz + n_ewald];
                    const auto rho_eikr = rho_k * eikr;
                    pot_long += G * std::real(rho_eikr);
                    if (grad_out) {
                        const double im = -std::imag(rho_eikr);
                        grad_long[0] += G * kx * im;
                        grad_long[1] += G * ky * im;
                        grad_long[2] += G * kz * im;
                    }
                }
        pot_out += (4.0 * M_PI / V_box) * pot_long;
        if (grad_out) {
            grad_out[0] += (4.0 * M_PI / V_box) * grad_long[0];
            grad_out[1] += (4.0 * M_PI / V_box) * grad_long[1];
            grad_out[2] += (4.0 * M_PI / V_box) * grad_long[2];
        }
    };

    const double ewald_self_factor = 2.0 * alpha / std::sqrt(M_PI);

    struct PrecisionCase {
        int n_digits;
        double eps;
        double tol_pot;
        double tol_grad;
    };
    const PrecisionCase cases[] = {
        {3, 1e-3, 1e-2, 1e-1},
        {6, 1e-6, 1e-4, 1e-3},
        {9, 1e-9, 1e-7, 1e-6},
        {12, 1e-12, 1e-10, 1e-9},
    };

    for (const auto &pc : cases) {
        for (int with_grad = 0; with_grad <= 1; ++with_grad) {
            const auto eval = with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
            const int odim = with_grad ? 1 + n_dim : 1;
            const std::string label = "n_digits=" + std::to_string(pc.n_digits) + (with_grad ? " pot+grad" : " pot");

            SUBCASE(label.c_str()) {
                pdmk_params params;
                params.eps = pc.eps;
                params.n_dim = n_dim;
                params.n_per_leaf = 50;
                params.eval_src = eval;
                params.eval_trg = eval;
                params.kernel = DMK_LAPLACE;
                params.use_periodic = true;
                params.log_level = 6;

                sctl::Vector<double> pot_src(n_src * odim), pot_trg(n_trg * odim);

                pdmk_tree tree =
                    pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
                pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
                pdmk_tree_destroy(tree);

                const int n_test = std::min(n_src, 100);
                double err2_pot_src = 0, ref2_pot_src = 0;
                double err2_grad_src = 0, ref2_grad_src = 0;
                for (int i = 0; i < n_test; ++i) {
                    double ewald_pot;
                    double ewald_grad[3];
                    ewald_pot_grad(&r_src[i * n_dim], ewald_pot, with_grad ? ewald_grad : nullptr);
                    ewald_pot -= charges[i] * ewald_self_factor;
                    err2_pot_src += sctl::pow<2>(pot_src[i * odim] - ewald_pot);
                    ref2_pot_src += sctl::pow<2>(ewald_pot);
                    if (with_grad) {
                        for (int dd = 0; dd < n_dim; ++dd) {
                            err2_grad_src += sctl::pow<2>(pot_src[i * odim + 1 + dd] - ewald_grad[dd]);
                            ref2_grad_src += sctl::pow<2>(ewald_grad[dd]);
                        }
                    }
                }

                const int n_test_trg = std::min(n_trg, 100);
                double err2_pot_trg = 0, ref2_pot_trg = 0;
                double err2_grad_trg = 0, ref2_grad_trg = 0;
                for (int i = 0; i < n_test_trg; ++i) {
                    double ewald_pot;
                    double ewald_grad[3];
                    ewald_pot_grad(&r_trg[i * n_dim], ewald_pot, with_grad ? ewald_grad : nullptr);
                    err2_pot_trg += sctl::pow<2>(pot_trg[i * odim] - ewald_pot);
                    ref2_pot_trg += sctl::pow<2>(ewald_pot);
                    if (with_grad) {
                        for (int dd = 0; dd < n_dim; ++dd) {
                            err2_grad_trg += sctl::pow<2>(pot_trg[i * odim + 1 + dd] - ewald_grad[dd]);
                            ref2_grad_trg += sctl::pow<2>(ewald_grad[dd]);
                        }
                    }
                }

                auto safe_l2 = [](double e2, double r2) { return r2 > 0 ? std::sqrt(e2 / r2) : std::sqrt(e2); };
                const double l2_pot_src = safe_l2(err2_pot_src, ref2_pot_src);
                const double l2_pot_trg = safe_l2(err2_pot_trg, ref2_pot_trg);

                VERBOSE_MESSAGE("PBC pipeline: ", label, " pot_src=", l2_pot_src, " pot_trg=", l2_pot_trg);
                CHECK(l2_pot_src < pc.tol_pot);
                CHECK(l2_pot_trg < pc.tol_pot);

#ifdef DMK_BUILD_ESP
                // ESP at sigma=1.35 can't reach eps=1e-12 (FINUFFT clips the spread width); skip n=12.
                if (pc.n_digits < 12) {
                    // ESP requires particles in [-L/2, L/2); shift into the box (periodic invariance
                    // keeps pot/grad comparable to the reference at the original positions).
                    std::vector<double> r_esp(n_dim * n_src);
                    for (int i = 0; i < n_dim * n_src; ++i)
                        r_esp[i] = r_src[i] - 0.5 * L;

                    pdmk_esp_params ep;
                    ep.L = L;
                    ep.r_c = L / 4;
                    ep.eps = pc.eps;
                    ep.kernel = DMK_LAPLACE;
                    ep.eval_type = eval;
                    ep.log_level = 6;
                    dmk::EspPlan<double> esp_plan(ep);
                    auto esp = esp_plan.eval(n_src, r_esp.data(), &charges[0]);

                    double e2p = 0, r2p = 0, e2g = 0, r2g = 0;
                    for (int i = 0; i < n_test; ++i) {
                        double ref_pot, ref_grad[3];
                        ewald_pot_grad(&r_src[i * n_dim], ref_pot, with_grad ? ref_grad : nullptr);
                        ref_pot -= charges[i] * ewald_self_factor;
                        e2p += sctl::pow<2>(esp.pot[i] - ref_pot);
                        r2p += sctl::pow<2>(ref_pot);
                        if (with_grad) {
                            const double f[3] = {esp.force_x[i], esp.force_y[i], esp.force_z[i]};
                            for (int dd = 0; dd < n_dim; ++dd) {
                                const double ref_force = -charges[i] * ref_grad[dd];
                                e2g += sctl::pow<2>(f[dd] - ref_force);
                                r2g += sctl::pow<2>(ref_force);
                            }
                        }
                    }
                    const double esp_l2_pot = safe_l2(e2p, r2p);
                    VERBOSE_MESSAGE("  ESP pot_src=", esp_l2_pot);
                    CHECK(esp_l2_pot < pc.tol_pot);
                    if (with_grad) {
                        const double esp_l2_grad = safe_l2(e2g, r2g);
                        VERBOSE_MESSAGE("  ESP force_src=", esp_l2_grad);
                        CHECK(esp_l2_grad < pc.tol_grad);
                    }
                }
#endif

                if (with_grad) {
                    const double l2_grad_src = safe_l2(err2_grad_src, ref2_grad_src);
                    const double l2_grad_trg = safe_l2(err2_grad_trg, ref2_grad_trg);
                    VERBOSE_MESSAGE("  grad_src=", l2_grad_src, " grad_trg=", l2_grad_trg);
                    CHECK(l2_grad_src < pc.tol_grad);
                    CHECK(l2_grad_trg < pc.tol_grad);
                }
            }
        }
    }
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Yukawa PBC full pipeline vs lattice sum", 1) {
    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 500;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    std::default_random_engine eng(123);
    std::uniform_real_distribution<double> rng(0.01, 0.99);

    sctl::Vector<double> r_src(n_dim * n_src), r_trg(n_dim * n_trg);
    sctl::Vector<double> charges(n_src);
    sctl::Vector<double> rnormal(n_dim * n_src);
    rnormal.SetZero();

    for (int i = 0; i < n_src * n_dim; ++i)
        r_src[i] = rng(eng);
    for (int i = 0; i < n_trg * n_dim; ++i)
        r_trg[i] = rng(eng);
    // Yukawa periodic sums converge absolutely, so charges need not be neutral; a
    // non-neutral set also exercises the finite k=0 mode of the periodic root kernel.
    for (int i = 0; i < n_src; ++i)
        charges[i] = rng(eng) - 0.5;

    const double L = 1.0;
    const double lambda = 6.0;
    // exp(-lambda*r) decays fast; the nearest excluded image (shell 7) is ~exp(-36).
    const int n_img = 6;

    // Periodic reference: DMK's own free-space direct evaluator summed over image shifts m*L.
    // The evaluator masks the r==0 pair, so a source vs the unshifted (m=0) sources drops only its
    // self term while nonzero shifts add its periodic images -- matching the periodic solver.
    auto make_ref = [&](dmk_eval_type eval, int n_eval, const double *r_eval, std::vector<double> &ref) {
        const int odim = (eval == DMK_POTENTIAL_GRAD) ? 1 + n_dim : 1;
        ref.assign(size_t(n_eval) * odim, 0.0);
        auto eval_fn = dmk::get_direct_evaluator<double>(DMK_YUKAWA, eval, n_dim, lambda);
        std::vector<double> src_shift(n_dim * n_src);
        for (int mx = -n_img; mx <= n_img; ++mx)
            for (int my = -n_img; my <= n_img; ++my)
                for (int mz = -n_img; mz <= n_img; ++mz) {
                    for (int is = 0; is < n_src; ++is) {
                        src_shift[is * 3 + 0] = r_src[is * 3 + 0] + mx * L;
                        src_shift[is * 3 + 1] = r_src[is * 3 + 1] + my * L;
                        src_shift[is * 3 + 2] = r_src[is * 3 + 2] + mz * L;
                    }
                    eval_fn(n_src, src_shift.data(), &charges[0], nullptr, n_eval, r_eval, ref.data());
                }
    };

    struct PrecisionCase {
        int n_digits;
        double eps;
        double tol_pot;
        double tol_grad;
    };
    const PrecisionCase cases[] = {
        {3, 1e-3, 1e-2, 1e-1},
        {6, 1e-6, 1e-4, 1e-3},
        {9, 1e-9, 1e-7, 1e-6},
        {12, 1e-12, 1e-10, 1e-9},
    };

    for (const auto &pc : cases) {
        for (int with_grad = 0; with_grad <= 1; ++with_grad) {
            const auto eval = with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
            const int odim = with_grad ? 1 + n_dim : 1;
            const std::string label = "n_digits=" + std::to_string(pc.n_digits) + (with_grad ? " pot+grad" : " pot");

            SUBCASE(label.c_str()) {
                pdmk_params params;
                params.eps = pc.eps;
                params.n_dim = n_dim;
                params.n_per_leaf = 50;
                params.eval_src = eval;
                params.eval_trg = eval;
                params.kernel = DMK_YUKAWA;
                params.fparam = lambda;
                params.use_periodic = true;
                params.log_level = 6;

                sctl::Vector<double> pot_src(n_src * odim), pot_trg(n_trg * odim);

                pdmk_tree tree =
                    pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
                pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
                pdmk_tree_destroy(tree);

                const int n_test = std::min(n_src, 50);
                std::vector<double> ref_src;
                make_ref(eval, n_test, &r_src[0], ref_src);
                double err2_pot_src = 0, ref2_pot_src = 0;
                double err2_grad_src = 0, ref2_grad_src = 0;
                for (int i = 0; i < n_test; ++i) {
                    err2_pot_src += sctl::pow<2>(pot_src[i * odim] - ref_src[i * odim]);
                    ref2_pot_src += sctl::pow<2>(ref_src[i * odim]);
                    if (with_grad) {
                        for (int dd = 0; dd < n_dim; ++dd) {
                            err2_grad_src += sctl::pow<2>(pot_src[i * odim + 1 + dd] - ref_src[i * odim + 1 + dd]);
                            ref2_grad_src += sctl::pow<2>(ref_src[i * odim + 1 + dd]);
                        }
                    }
                }

                const int n_test_trg = std::min(n_trg, 50);
                std::vector<double> ref_trg;
                make_ref(eval, n_test_trg, &r_trg[0], ref_trg);
                double err2_pot_trg = 0, ref2_pot_trg = 0;
                double err2_grad_trg = 0, ref2_grad_trg = 0;
                for (int i = 0; i < n_test_trg; ++i) {
                    err2_pot_trg += sctl::pow<2>(pot_trg[i * odim] - ref_trg[i * odim]);
                    ref2_pot_trg += sctl::pow<2>(ref_trg[i * odim]);
                    if (with_grad) {
                        for (int dd = 0; dd < n_dim; ++dd) {
                            err2_grad_trg += sctl::pow<2>(pot_trg[i * odim + 1 + dd] - ref_trg[i * odim + 1 + dd]);
                            ref2_grad_trg += sctl::pow<2>(ref_trg[i * odim + 1 + dd]);
                        }
                    }
                }

                auto safe_l2 = [](double e2, double r2) { return r2 > 0 ? std::sqrt(e2 / r2) : std::sqrt(e2); };
                const double l2_pot_src = safe_l2(err2_pot_src, ref2_pot_src);
                const double l2_pot_trg = safe_l2(err2_pot_trg, ref2_pot_trg);

                VERBOSE_MESSAGE("Yukawa PBC pipeline: ", label, " pot_src=", l2_pot_src, " pot_trg=", l2_pot_trg);
                CHECK(l2_pot_src < pc.tol_pot);
                CHECK(l2_pot_trg < pc.tol_pot);

#ifdef DMK_BUILD_ESP
                // ESP at sigma=1.35 can't reach eps=1e-12 (FINUFFT clips the spread width); skip n=12.
                if (pc.n_digits < 12) {
                    // ESP requires particles in [-L/2, L/2); shift into the box (periodic invariance).
                    // Non-neutral charges here exercise the finite Yukawa k=0 mode.
                    std::vector<double> r_esp(n_dim * n_src);
                    for (int i = 0; i < n_dim * n_src; ++i)
                        r_esp[i] = r_src[i] - 0.5 * L;

                    pdmk_esp_params ep;
                    ep.L = L;
                    ep.r_c = L / 4;
                    ep.eps = pc.eps;
                    ep.kernel = DMK_YUKAWA;
                    ep.fparam = lambda;
                    ep.eval_type = eval;
                    ep.log_level = 6;
                    dmk::EspPlan<double> esp_plan(ep);
                    auto esp = esp_plan.eval(n_src, r_esp.data(), &charges[0]);

                    double e2p = 0, r2p = 0, e2g = 0, r2g = 0;
                    for (int i = 0; i < n_test; ++i) {
                        e2p += sctl::pow<2>(esp.pot[i] - ref_src[i * odim]);
                        r2p += sctl::pow<2>(ref_src[i * odim]);
                        if (with_grad) {
                            const double f[3] = {esp.force_x[i], esp.force_y[i], esp.force_z[i]};
                            for (int dd = 0; dd < n_dim; ++dd) {
                                const double ref_force = -charges[i] * ref_src[i * odim + 1 + dd];
                                e2g += sctl::pow<2>(f[dd] - ref_force);
                                r2g += sctl::pow<2>(ref_force);
                            }
                        }
                    }
                    const double esp_l2_pot = safe_l2(e2p, r2p);
                    VERBOSE_MESSAGE("  ESP pot_src=", esp_l2_pot);
                    CHECK(esp_l2_pot < pc.tol_pot);
                    if (with_grad) {
                        const double esp_l2_grad = safe_l2(e2g, r2g);
                        VERBOSE_MESSAGE("  ESP force_src=", esp_l2_grad);
                        CHECK(esp_l2_grad < pc.tol_grad);
                    }
                }
#endif

                if (with_grad) {
                    const double l2_grad_src = safe_l2(err2_grad_src, ref2_grad_src);
                    const double l2_grad_trg = safe_l2(err2_grad_trg, ref2_grad_trg);
                    VERBOSE_MESSAGE("  grad_src=", l2_grad_src, " grad_trg=", l2_grad_trg);
                    CHECK(l2_grad_src < pc.tol_grad);
                    CHECK(l2_grad_trg < pc.tol_grad);
                }
            }
        }
    }
}

TEST_CASE_GENERIC("[DMK] pdmk 3d Sqrt-Laplace PBC full pipeline vs Ewald", 1) {
    // In 3D, DMK_SQRT_LAPLACE is the 1/r^2 kernel (Green's function of sqrt(-Laplacian)).
    // Its periodic sum is conditionally convergent (needs charge neutrality); the reference
    // uses an Ewald split of 1/r^2 = \int_0^\infty exp(-r^2 t) dt truncated at t=eta^2:
    //   short range S(r) = exp(-eta^2 r^2)/r^2
    //   long range  Lhat(k) = (2 pi^2 / k) erfc(k / (2 eta))
    //   self term   L(0) = eta^2
    constexpr int n_dim = 3;
    constexpr int n_src = 2000;
    constexpr int n_trg = 500;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    std::default_random_engine eng(7);
    std::uniform_real_distribution<double> rng(0.01, 0.99);

    sctl::Vector<double> r_src(n_dim * n_src), r_trg(n_dim * n_trg);
    sctl::Vector<double> charges(n_src);
    sctl::Vector<double> rnormal(n_dim * n_src);
    rnormal.SetZero();

    for (int i = 0; i < n_src * n_dim; ++i)
        r_src[i] = rng(eng);
    for (int i = 0; i < n_trg * n_dim; ++i)
        r_trg[i] = rng(eng);
    for (int i = 0; i < n_src; ++i)
        charges[i] = rng(eng) - 0.5;
    { // 1/r^2 periodic requires charge neutrality
        double sum = 0.0;
        for (int i = 0; i < n_src; ++i)
            sum += charges[i];
        for (int i = 0; i < n_src; ++i)
            charges[i] -= sum / n_src;
    }

    const double L = 1.0;
    const double V_box = L * L * L;
    const double dk = 2.0 * M_PI / L;
    const double eta = 6.0;
    const int n_real = 2; // exp(-eta^2 r^2) is negligible past nearest images
    const int n_ewald = 15;
    const int d = 2 * n_ewald + 1;

    std::vector<std::complex<double>> rho(d * d * d, {0.0, 0.0});
    for (int is = 0; is < n_src; ++is) {
        const std::complex<double> ex0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 3 + 0]));
        const std::complex<double> ey0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 3 + 1]));
        const std::complex<double> ez0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 3 + 2]));
        std::vector<std::complex<double>> ex(d), ey(d), ez(d);
        for (int a = -n_ewald; a <= n_ewald; ++a) {
            ex[a + n_ewald] = std::pow(ex0, a);
            ey[a + n_ewald] = std::pow(ey0, a);
            ez[a + n_ewald] = std::pow(ez0, a);
        }
        for (int ix = 0; ix < d; ++ix)
            for (int iy = 0; iy < d; ++iy) {
                const auto t2 = charges[is] * ex[ix] * ey[iy];
                for (int iz = 0; iz < d; ++iz)
                    rho[ix * d * d + iy * d + iz] += t2 * ez[iz];
            }
    }

    // self_idx >= 0 subtracts that source's self term (eta^2); the self term is a constant so it
    // does not affect the gradient (3 comps, optional).
    auto ewald_pot_grad = [&](const double *r_eval, int self_idx, double &pot_out, double *grad_out) {
        pot_out = 0.0;
        if (grad_out)
            grad_out[0] = grad_out[1] = grad_out[2] = 0.0;
        for (int is = 0; is < n_src; ++is)
            for (int mx = -n_real; mx <= n_real; ++mx)
                for (int my = -n_real; my <= n_real; ++my)
                    for (int mz = -n_real; mz <= n_real; ++mz) {
                        const double dx = r_eval[0] - r_src[is * 3 + 0] - mx * L;
                        const double dy = r_eval[1] - r_src[is * 3 + 1] - my * L;
                        const double dz = r_eval[2] - r_src[is * 3 + 2] - mz * L;
                        const double r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 < 1e-28)
                            continue;
                        const double g = std::exp(-eta * eta * r2);
                        pot_out += charges[is] * g / r2;
                        if (grad_out) {
                            // grad of exp(-eta^2 r^2)/r^2 = -2 exp(-eta^2 r^2)(eta^2 r^2 + 1)/r^4 * r_vec
                            const double scale = -2.0 * charges[is] * g * (eta * eta * r2 + 1.0) / (r2 * r2);
                            grad_out[0] += scale * dx;
                            grad_out[1] += scale * dy;
                            grad_out[2] += scale * dz;
                        }
                    }

        const std::complex<double> etx0 = std::exp(std::complex<double>(0.0, dk * r_eval[0]));
        const std::complex<double> ety0 = std::exp(std::complex<double>(0.0, dk * r_eval[1]));
        const std::complex<double> etz0 = std::exp(std::complex<double>(0.0, dk * r_eval[2]));
        std::vector<std::complex<double>> etx(d), ety(d), etz(d);
        for (int a = -n_ewald; a <= n_ewald; ++a) {
            etx[a + n_ewald] = std::pow(etx0, a);
            ety[a + n_ewald] = std::pow(ety0, a);
            etz[a + n_ewald] = std::pow(etz0, a);
        }

        double pot_long = 0.0, grad_long[3] = {0, 0, 0};
        for (int nx = -n_ewald; nx <= n_ewald; ++nx)
            for (int ny = -n_ewald; ny <= n_ewald; ++ny)
                for (int nz = -n_ewald; nz <= n_ewald; ++nz) {
                    if (nx == 0 && ny == 0 && nz == 0)
                        continue;
                    const double kx = dk * nx, ky = dk * ny, kz = dk * nz;
                    const double kmag = std::sqrt(kx * kx + ky * ky + kz * kz);
                    const double G = std::erfc(kmag / (2.0 * eta)) / kmag;
                    const auto &rho_k = rho[(nx + n_ewald) * d * d + (ny + n_ewald) * d + (nz + n_ewald)];
                    const auto eikr = etx[nx + n_ewald] * ety[ny + n_ewald] * etz[nz + n_ewald];
                    const auto rho_eikr = rho_k * eikr;
                    pot_long += G * std::real(rho_eikr);
                    if (grad_out) {
                        const double im = -std::imag(rho_eikr);
                        grad_long[0] += G * kx * im;
                        grad_long[1] += G * ky * im;
                        grad_long[2] += G * kz * im;
                    }
                }
        pot_out += (2.0 * M_PI * M_PI / V_box) * pot_long;
        if (grad_out) {
            grad_out[0] += (2.0 * M_PI * M_PI / V_box) * grad_long[0];
            grad_out[1] += (2.0 * M_PI * M_PI / V_box) * grad_long[1];
            grad_out[2] += (2.0 * M_PI * M_PI / V_box) * grad_long[2];
        }
        if (self_idx >= 0)
            pot_out -= charges[self_idx] * eta * eta; // L(0)
    };

    struct PrecisionCase {
        int n_digits;
        double eps;
        double tol_pot;
        double tol_grad;
    };
    const PrecisionCase cases[] = {
        {3, 1e-3, 1e-2, 1e-1}, {6, 1e-6, 1e-4, 1e-3}, {9, 1e-9, 1e-7, 1e-6}, {12, 1e-12, 1e-10, 1e-9}};

    for (const auto &pc : cases) {
        for (int with_grad = 0; with_grad <= 1; ++with_grad) {
            const auto eval = with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
            const int odim = with_grad ? 1 + n_dim : 1;
            const std::string label = "n_digits=" + std::to_string(pc.n_digits) + (with_grad ? " pot+grad" : " pot");

            SUBCASE(label.c_str()) {
                pdmk_params params;
                params.eps = pc.eps;
                params.n_dim = n_dim;
                params.n_per_leaf = 50;
                params.eval_src = eval;
                params.eval_trg = eval;
                params.kernel = DMK_SQRT_LAPLACE;
                params.use_periodic = true;
                params.log_level = 6;

                sctl::Vector<double> pot_src(n_src * odim), pot_trg(n_trg * odim);
                pdmk_tree tree =
                    pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
                pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
                pdmk_tree_destroy(tree);

                const int n_test = std::min(n_src, 100);
                double e2p = 0, r2p = 0, e2g = 0, r2g = 0;
                for (int i = 0; i < n_test; ++i) {
                    double ref_pot, ref_grad[3];
                    ewald_pot_grad(&r_src[i * n_dim], i, ref_pot, with_grad ? ref_grad : nullptr);
                    e2p += sctl::pow<2>(pot_src[i * odim] - ref_pot);
                    r2p += sctl::pow<2>(ref_pot);
                    if (with_grad)
                        for (int dd = 0; dd < n_dim; ++dd) {
                            e2g += sctl::pow<2>(pot_src[i * odim + 1 + dd] - ref_grad[dd]);
                            r2g += sctl::pow<2>(ref_grad[dd]);
                        }
                }

                const int n_test_trg = std::min(n_trg, 100);
                double e2pt = 0, r2pt = 0, e2gt = 0, r2gt = 0;
                for (int i = 0; i < n_test_trg; ++i) {
                    double ref_pot, ref_grad[3];
                    ewald_pot_grad(&r_trg[i * n_dim], -1, ref_pot, with_grad ? ref_grad : nullptr);
                    e2pt += sctl::pow<2>(pot_trg[i * odim] - ref_pot);
                    r2pt += sctl::pow<2>(ref_pot);
                    if (with_grad)
                        for (int dd = 0; dd < n_dim; ++dd) {
                            e2gt += sctl::pow<2>(pot_trg[i * odim + 1 + dd] - ref_grad[dd]);
                            r2gt += sctl::pow<2>(ref_grad[dd]);
                        }
                }

                auto safe_l2 = [](double e2, double r2) { return r2 > 0 ? std::sqrt(e2 / r2) : std::sqrt(e2); };
                VERBOSE_MESSAGE("Sqrt-Laplace PBC pipeline: ", label, " pot_src=", safe_l2(e2p, r2p),
                                " pot_trg=", safe_l2(e2pt, r2pt));
                CHECK(safe_l2(e2p, r2p) < pc.tol_pot);
                CHECK(safe_l2(e2pt, r2pt) < pc.tol_pot);

#ifdef DMK_BUILD_ESP
                // ESP at sigma=1.35 can't reach eps=1e-12 (FINUFFT clips the spread width); skip n=12.
                if (pc.n_digits < 12) {
                    // ESP requires particles in [-L/2, L/2); shift into the box (periodic invariance).
                    std::vector<double> r_esp(n_dim * n_src);
                    for (int i = 0; i < n_dim * n_src; ++i)
                        r_esp[i] = r_src[i] - 0.5 * L;

                    pdmk_esp_params ep;
                    ep.L = L;
                    ep.r_c = L / 4;
                    ep.eps = pc.eps;
                    ep.kernel = DMK_SQRT_LAPLACE;
                    ep.eval_type = eval;
                    ep.log_level = 6;
                    dmk::EspPlan<double> esp_plan(ep);
                    auto esp = esp_plan.eval(n_src, r_esp.data(), &charges[0]);

                    double ee2p = 0, er2p = 0, ee2g = 0, er2g = 0;
                    for (int i = 0; i < n_test; ++i) {
                        double ref_pot, ref_grad[3];
                        ewald_pot_grad(&r_src[i * n_dim], i, ref_pot, with_grad ? ref_grad : nullptr);
                        ee2p += sctl::pow<2>(esp.pot[i] - ref_pot);
                        er2p += sctl::pow<2>(ref_pot);
                        if (with_grad) {
                            const double f[3] = {esp.force_x[i], esp.force_y[i], esp.force_z[i]};
                            for (int dd = 0; dd < n_dim; ++dd) {
                                const double ref_force = -charges[i] * ref_grad[dd];
                                ee2g += sctl::pow<2>(f[dd] - ref_force);
                                er2g += sctl::pow<2>(ref_force);
                            }
                        }
                    }
                    const double esp_l2_pot = safe_l2(ee2p, er2p);
                    VERBOSE_MESSAGE("  ESP pot_src=", esp_l2_pot);
                    CHECK(esp_l2_pot < pc.tol_pot);
                    if (with_grad) {
                        const double esp_l2_grad = safe_l2(ee2g, er2g);
                        VERBOSE_MESSAGE("  ESP force_src=", esp_l2_grad);
                        CHECK(esp_l2_grad < pc.tol_grad);
                    }
                }
#endif

                if (with_grad) {
                    CHECK(safe_l2(e2g, r2g) < pc.tol_grad);
                    CHECK(safe_l2(e2gt, r2gt) < pc.tol_grad);
                }
            }
        }
    }
}

TEST_CASE_GENERIC("[DMK] pdmk 2d Yukawa PBC full pipeline vs lattice sum", 1) {
    // 2D Yukawa kernel is K0(lambda*r); its periodic sum converges absolutely, so the reference is
    // DMK's own free-space direct evaluator summed over image shifts (as in the 3D Yukawa test).
    constexpr int n_dim = 2;
    constexpr int n_src = 2000;
    constexpr int n_trg = 500;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    std::default_random_engine eng(321);
    std::uniform_real_distribution<double> rng(0.01, 0.99);

    sctl::Vector<double> r_src(n_dim * n_src), r_trg(n_dim * n_trg);
    sctl::Vector<double> charges(n_src);
    sctl::Vector<double> rnormal(n_dim * n_src);
    rnormal.SetZero();

    for (int i = 0; i < n_src * n_dim; ++i)
        r_src[i] = rng(eng);
    for (int i = 0; i < n_trg * n_dim; ++i)
        r_trg[i] = rng(eng);
    for (int i = 0; i < n_src; ++i)
        charges[i] = rng(eng) - 0.5; // non-neutral: exercises the finite k=0 mode

    const double L = 1.0;
    const double lambda = 4.0;
    const int n_img = 8; // K0(lambda*r) ~ exp(-lambda*r); shell 7 is ~exp(-28)

    // Reference = free-space direct evaluator (potential or potential+grad) summed over images.
    auto make_ref = [&](dmk_eval_type eval, int n_eval, const double *r_eval, std::vector<double> &ref) {
        const int odim = (eval == DMK_POTENTIAL_GRAD) ? 1 + n_dim : 1;
        ref.assign(size_t(n_eval) * odim, 0.0);
        auto eval_fn = dmk::get_direct_evaluator<double>(DMK_YUKAWA, eval, n_dim, lambda);
        std::vector<double> src_shift(n_dim * n_src);
        for (int mx = -n_img; mx <= n_img; ++mx)
            for (int my = -n_img; my <= n_img; ++my) {
                for (int is = 0; is < n_src; ++is) {
                    src_shift[is * 2 + 0] = r_src[is * 2 + 0] + mx * L;
                    src_shift[is * 2 + 1] = r_src[is * 2 + 1] + my * L;
                }
                eval_fn(n_src, src_shift.data(), &charges[0], nullptr, n_eval, r_eval, ref.data());
            }
    };

    struct PrecisionCase {
        int n_digits;
        double eps;
        double tol_pot;
        double tol_grad;
    };
    const PrecisionCase cases[] = {
        {3, 1e-3, 1e-2, 1e-1}, {6, 1e-6, 1e-4, 1e-3}, {9, 1e-9, 1e-7, 1e-6}, {12, 1e-12, 1e-10, 1e-9}};

    for (const auto &pc : cases) {
        for (int with_grad = 0; with_grad <= 1; ++with_grad) {
            const auto eval = with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
            const int odim = with_grad ? 1 + n_dim : 1;
            const std::string label = "n_digits=" + std::to_string(pc.n_digits) + (with_grad ? " pot+grad" : " pot");

            SUBCASE(label.c_str()) {
                pdmk_params params;
                params.eps = pc.eps;
                params.n_dim = n_dim;
                params.n_per_leaf = 50;
                params.eval_src = eval;
                params.eval_trg = eval;
                params.kernel = DMK_YUKAWA;
                params.fparam = lambda;
                params.use_periodic = true;
                params.log_level = 6;

                sctl::Vector<double> pot_src(n_src * odim), pot_trg(n_trg * odim);
                pdmk_tree tree =
                    pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
                pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
                pdmk_tree_destroy(tree);

                const int n_test = std::min(n_src, 50);
                std::vector<double> ref_src;
                make_ref(eval, n_test, &r_src[0], ref_src);
                double e2p = 0, r2p = 0, e2g = 0, r2g = 0;
                for (int i = 0; i < n_test; ++i) {
                    e2p += sctl::pow<2>(pot_src[i * odim] - ref_src[i * odim]);
                    r2p += sctl::pow<2>(ref_src[i * odim]);
                    if (with_grad)
                        for (int d = 0; d < n_dim; ++d) {
                            e2g += sctl::pow<2>(pot_src[i * odim + 1 + d] - ref_src[i * odim + 1 + d]);
                            r2g += sctl::pow<2>(ref_src[i * odim + 1 + d]);
                        }
                }

                const int n_test_trg = std::min(n_trg, 50);
                std::vector<double> ref_trg;
                make_ref(eval, n_test_trg, &r_trg[0], ref_trg);
                double e2pt = 0, r2pt = 0, e2gt = 0, r2gt = 0;
                for (int i = 0; i < n_test_trg; ++i) {
                    e2pt += sctl::pow<2>(pot_trg[i * odim] - ref_trg[i * odim]);
                    r2pt += sctl::pow<2>(ref_trg[i * odim]);
                    if (with_grad)
                        for (int d = 0; d < n_dim; ++d) {
                            e2gt += sctl::pow<2>(pot_trg[i * odim + 1 + d] - ref_trg[i * odim + 1 + d]);
                            r2gt += sctl::pow<2>(ref_trg[i * odim + 1 + d]);
                        }
                }

                auto safe_l2 = [](double e2, double r2) { return r2 > 0 ? std::sqrt(e2 / r2) : std::sqrt(e2); };
                VERBOSE_MESSAGE("2D Yukawa PBC: ", label, " pot_src=", safe_l2(e2p, r2p),
                                " pot_trg=", safe_l2(e2pt, r2pt));
                CHECK(safe_l2(e2p, r2p) < pc.tol_pot);
                CHECK(safe_l2(e2pt, r2pt) < pc.tol_pot);
                if (with_grad) {
                    CHECK(safe_l2(e2g, r2g) < pc.tol_grad);
                    CHECK(safe_l2(e2gt, r2gt) < pc.tol_grad);
                }
            }
        }
    }
}

TEST_CASE_GENERIC("[DMK] pdmk 2d Sqrt-Laplace PBC full pipeline vs Ewald", 1) {
    // 2D Sqrt-Laplace is the 1/r kernel. Conditionally convergent (needs neutrality); reference uses
    // the 2D 1/r Ewald split of 1/r = (2/sqrt(pi)) int_0^inf exp(-r^2 s^2) ds split at s=eta:
    //   short S(r) = erfc(eta r)/r, long Lhat(k) = (2 pi / k) erfc(k/(2 eta)), self L(0) = 2 eta/sqrt(pi).
    constexpr int n_dim = 2;
    constexpr int n_src = 2000;
    constexpr int n_trg = 500;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    std::default_random_engine eng(54);
    std::uniform_real_distribution<double> rng(0.01, 0.99);

    sctl::Vector<double> r_src(n_dim * n_src), r_trg(n_dim * n_trg);
    sctl::Vector<double> charges(n_src);
    sctl::Vector<double> rnormal(n_dim * n_src);
    rnormal.SetZero();

    for (int i = 0; i < n_src * n_dim; ++i)
        r_src[i] = rng(eng);
    for (int i = 0; i < n_trg * n_dim; ++i)
        r_trg[i] = rng(eng);
    for (int i = 0; i < n_src; ++i)
        charges[i] = rng(eng) - 0.5;
    { // 1/r periodic requires charge neutrality
        double sum = 0.0;
        for (int i = 0; i < n_src; ++i)
            sum += charges[i];
        for (int i = 0; i < n_src; ++i)
            charges[i] -= sum / n_src;
    }

    const double L = 1.0;
    const double V_box = L * L;
    const double dk = 2.0 * M_PI / L;
    const double eta = 6.0;
    const int n_real = 2;
    const int n_ewald = 15;
    const int d = 2 * n_ewald + 1;

    std::vector<std::complex<double>> rho(d * d, {0.0, 0.0});
    for (int is = 0; is < n_src; ++is) {
        const std::complex<double> ex0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 2 + 0]));
        const std::complex<double> ey0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 2 + 1]));
        std::vector<std::complex<double>> ex(d), ey(d);
        for (int a = -n_ewald; a <= n_ewald; ++a) {
            ex[a + n_ewald] = std::pow(ex0, a);
            ey[a + n_ewald] = std::pow(ey0, a);
        }
        for (int ix = 0; ix < d; ++ix)
            for (int iy = 0; iy < d; ++iy)
                rho[ix * d + iy] += charges[is] * ex[ix] * ey[iy];
    }

    // self_idx >= 0 subtracts that source's self term (2 eta/sqrt(pi)); pass -1 for targets.
    // grad_out (2 comps) optional; the self term is a constant so it does not affect the gradient.
    auto ewald_pot_grad = [&](const double *r_eval, int self_idx, double &pot_out, double *grad_out) {
        pot_out = 0.0;
        if (grad_out)
            grad_out[0] = grad_out[1] = 0.0;
        const double two_eta_sqrtpi = 2.0 * eta / std::sqrt(M_PI);
        for (int is = 0; is < n_src; ++is)
            for (int mx = -n_real; mx <= n_real; ++mx)
                for (int my = -n_real; my <= n_real; ++my) {
                    const double dx = r_eval[0] - r_src[is * 2 + 0] - mx * L;
                    const double dy = r_eval[1] - r_src[is * 2 + 1] - my * L;
                    const double r2 = dx * dx + dy * dy;
                    if (r2 < 1e-28)
                        continue;
                    const double r = std::sqrt(r2);
                    pot_out += charges[is] * std::erfc(eta * r) / r;
                    if (grad_out) {
                        // grad of erfc(eta r)/r = -[erfc(eta r)/r^3 + 2 eta/(sqrt(pi) r^2) exp(-eta^2 r^2)] r_vec
                        const double scale = -charges[is] * (std::erfc(eta * r) / (r * r2) +
                                                             two_eta_sqrtpi * std::exp(-eta * eta * r2) / r2);
                        grad_out[0] += scale * dx;
                        grad_out[1] += scale * dy;
                    }
                }

        const std::complex<double> etx0 = std::exp(std::complex<double>(0.0, dk * r_eval[0]));
        const std::complex<double> ety0 = std::exp(std::complex<double>(0.0, dk * r_eval[1]));
        std::vector<std::complex<double>> etx(d), ety(d);
        for (int a = -n_ewald; a <= n_ewald; ++a) {
            etx[a + n_ewald] = std::pow(etx0, a);
            ety[a + n_ewald] = std::pow(ety0, a);
        }
        double pot_long = 0.0, grad_long[2] = {0, 0};
        for (int nx = -n_ewald; nx <= n_ewald; ++nx)
            for (int ny = -n_ewald; ny <= n_ewald; ++ny) {
                if (nx == 0 && ny == 0)
                    continue;
                const double kx = dk * nx, ky = dk * ny;
                const double kmag = std::sqrt(kx * kx + ky * ky);
                const double G = std::erfc(kmag / (2.0 * eta)) / kmag;
                const auto eikr = etx[nx + n_ewald] * ety[ny + n_ewald];
                const auto rho_eikr = rho[(nx + n_ewald) * d + (ny + n_ewald)] * eikr;
                pot_long += G * std::real(rho_eikr);
                if (grad_out) {
                    const double im = -std::imag(rho_eikr);
                    grad_long[0] += G * kx * im;
                    grad_long[1] += G * ky * im;
                }
            }
        pot_out += (2.0 * M_PI / V_box) * pot_long;
        if (grad_out) {
            grad_out[0] += (2.0 * M_PI / V_box) * grad_long[0];
            grad_out[1] += (2.0 * M_PI / V_box) * grad_long[1];
        }
        if (self_idx >= 0)
            pot_out -= charges[self_idx] * two_eta_sqrtpi;
    };

    struct PrecisionCase {
        int n_digits;
        double eps;
        double tol_pot;
        double tol_grad;
    };
    const PrecisionCase cases[] = {
        {3, 1e-3, 1e-2, 1e-1}, {6, 1e-6, 1e-4, 1e-3}, {9, 1e-9, 1e-7, 1e-6}, {12, 1e-12, 1e-10, 1e-9}};

    for (const auto &pc : cases) {
        for (int with_grad = 0; with_grad <= 1; ++with_grad) {
            const auto eval = with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
            const int odim = with_grad ? 1 + n_dim : 1;
            const std::string label = "n_digits=" + std::to_string(pc.n_digits) + (with_grad ? " pot+grad" : " pot");

            SUBCASE(label.c_str()) {
                pdmk_params params;
                params.eps = pc.eps;
                params.n_dim = n_dim;
                params.n_per_leaf = 50;
                params.eval_src = eval;
                params.eval_trg = eval;
                params.kernel = DMK_SQRT_LAPLACE;
                params.use_periodic = true;
                params.log_level = 6;

                sctl::Vector<double> pot_src(n_src * odim), pot_trg(n_trg * odim);
                pdmk_tree tree =
                    pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
                pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
                pdmk_tree_destroy(tree);

                const int n_test = std::min(n_src, 50);
                double e2p = 0, r2p = 0, e2g = 0, r2g = 0;
                for (int i = 0; i < n_test; ++i) {
                    double ref_pot, ref_grad[2];
                    ewald_pot_grad(&r_src[i * n_dim], i, ref_pot, with_grad ? ref_grad : nullptr);
                    e2p += sctl::pow<2>(pot_src[i * odim] - ref_pot);
                    r2p += sctl::pow<2>(ref_pot);
                    if (with_grad)
                        for (int dd = 0; dd < n_dim; ++dd) {
                            e2g += sctl::pow<2>(pot_src[i * odim + 1 + dd] - ref_grad[dd]);
                            r2g += sctl::pow<2>(ref_grad[dd]);
                        }
                }
                const int n_test_trg = std::min(n_trg, 50);
                double e2pt = 0, r2pt = 0, e2gt = 0, r2gt = 0;
                for (int i = 0; i < n_test_trg; ++i) {
                    double ref_pot, ref_grad[2];
                    ewald_pot_grad(&r_trg[i * n_dim], -1, ref_pot, with_grad ? ref_grad : nullptr);
                    e2pt += sctl::pow<2>(pot_trg[i * odim] - ref_pot);
                    r2pt += sctl::pow<2>(ref_pot);
                    if (with_grad)
                        for (int dd = 0; dd < n_dim; ++dd) {
                            e2gt += sctl::pow<2>(pot_trg[i * odim + 1 + dd] - ref_grad[dd]);
                            r2gt += sctl::pow<2>(ref_grad[dd]);
                        }
                }
                auto safe_l2 = [](double e2, double r2) { return r2 > 0 ? std::sqrt(e2 / r2) : std::sqrt(e2); };
                VERBOSE_MESSAGE("2D Sqrt-Laplace PBC: ", label, " pot_src=", safe_l2(e2p, r2p),
                                " pot_trg=", safe_l2(e2pt, r2pt));
                CHECK(safe_l2(e2p, r2p) < pc.tol_pot);
                CHECK(safe_l2(e2pt, r2pt) < pc.tol_pot);
                if (with_grad) {
                    CHECK(safe_l2(e2g, r2g) < pc.tol_grad);
                    CHECK(safe_l2(e2gt, r2gt) < pc.tol_grad);
                }
            }
        }
    }
}

TEST_CASE_GENERIC("[DMK] pdmk 2d Laplace PBC full pipeline vs Ewald", 1) {
    // 2D Laplace is the log(r) kernel. Conditionally convergent (needs neutrality); reference uses
    // the 2D log Ewald split log(r) = -1/2 int_0^inf (exp(-r^2 t) - exp(-t))/t dt split at t=eta^2:
    //   short  S(r) = -1/2 E1(eta^2 r^2)  (the r-independent piece cancels under neutrality)
    //   long   Lhat(k) = -(2 pi / k^2) exp(-k^2 / (4 eta^2))
    // Evaluated at target points only, so no self term is needed.
    constexpr int n_dim = 2;
    constexpr int n_src = 2000;
    constexpr int n_trg = 500;

#ifdef DMK_HAVE_MPI
    auto comm = test_comm;
#else
    auto comm = nullptr;
#endif

    std::default_random_engine eng(88);
    std::uniform_real_distribution<double> rng(0.01, 0.99);

    sctl::Vector<double> r_src(n_dim * n_src), r_trg(n_dim * n_trg);
    sctl::Vector<double> charges(n_src);
    sctl::Vector<double> rnormal(n_dim * n_src);
    rnormal.SetZero();

    for (int i = 0; i < n_src * n_dim; ++i)
        r_src[i] = rng(eng);
    for (int i = 0; i < n_trg * n_dim; ++i)
        r_trg[i] = rng(eng);
    for (int i = 0; i < n_src; ++i)
        charges[i] = rng(eng) - 0.5;
    { // log periodic requires charge neutrality
        double sum = 0.0;
        for (int i = 0; i < n_src; ++i)
            sum += charges[i];
        for (int i = 0; i < n_src; ++i)
            charges[i] -= sum / n_src;
    }

    // Exponential integral E1(x) = int_x^inf exp(-t)/t dt, x > 0 (Numerical Recipes expint(1, .)).
    auto E1 = [](double x) {
        const double EULER = 0.5772156649015328606;
        if (x < 1.0) {
            double ans = -std::log(x) - EULER, fact = 1.0;
            for (int i = 1; i <= 100; ++i) {
                fact *= -x / i;
                const double del = -fact / i;
                ans += del;
                if (std::abs(del) < std::abs(ans) * 1e-16)
                    break;
            }
            return ans;
        }
        double b = x + 1.0, c = 1e300, dd = 1.0 / b, h = dd;
        for (int i = 1; i <= 100; ++i) {
            const double an = -1.0 * i * i;
            b += 2.0;
            dd = 1.0 / (an * dd + b);
            c = b + an / c;
            const double del = c * dd;
            h *= del;
            if (std::abs(del - 1.0) < 1e-16)
                break;
        }
        return h * std::exp(-x);
    };

    const double L = 1.0;
    const double V_box = L * L;
    const double dk = 2.0 * M_PI / L;
    const double eta = 6.0;
    const int n_real = 2;
    const int n_ewald = 15;
    const int d = 2 * n_ewald + 1;

    std::vector<std::complex<double>> rho(d * d, {0.0, 0.0});
    for (int is = 0; is < n_src; ++is) {
        const std::complex<double> ex0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 2 + 0]));
        const std::complex<double> ey0 = std::exp(std::complex<double>(0.0, -dk * r_src[is * 2 + 1]));
        std::vector<std::complex<double>> ex(d), ey(d);
        for (int a = -n_ewald; a <= n_ewald; ++a) {
            ex[a + n_ewald] = std::pow(ex0, a);
            ey[a + n_ewald] = std::pow(ey0, a);
        }
        for (int ix = 0; ix < d; ++ix)
            for (int iy = 0; iy < d; ++iy)
                rho[ix * d + iy] += charges[is] * ex[ix] * ey[iy];
    }

    auto ewald_pot_grad = [&](const double *r_eval, double &pot_out, double *grad_out) {
        pot_out = 0.0;
        if (grad_out)
            grad_out[0] = grad_out[1] = 0.0;
        for (int is = 0; is < n_src; ++is)
            for (int mx = -n_real; mx <= n_real; ++mx)
                for (int my = -n_real; my <= n_real; ++my) {
                    const double dx = r_eval[0] - r_src[is * 2 + 0] - mx * L;
                    const double dy = r_eval[1] - r_src[is * 2 + 1] - my * L;
                    const double r2 = dx * dx + dy * dy;
                    if (r2 < 1e-28)
                        continue;
                    pot_out += charges[is] * (-0.5) * E1(eta * eta * r2);
                    if (grad_out) {
                        // grad of -1/2 E1(eta^2 r^2) = exp(-eta^2 r^2)/r^2 * r_vec
                        const double g = charges[is] * std::exp(-eta * eta * r2) / r2;
                        grad_out[0] += g * dx;
                        grad_out[1] += g * dy;
                    }
                }

        const std::complex<double> etx0 = std::exp(std::complex<double>(0.0, dk * r_eval[0]));
        const std::complex<double> ety0 = std::exp(std::complex<double>(0.0, dk * r_eval[1]));
        std::vector<std::complex<double>> etx(d), ety(d);
        for (int a = -n_ewald; a <= n_ewald; ++a) {
            etx[a + n_ewald] = std::pow(etx0, a);
            ety[a + n_ewald] = std::pow(ety0, a);
        }
        double pot_long = 0.0, grad_long[2] = {0, 0};
        for (int nx = -n_ewald; nx <= n_ewald; ++nx)
            for (int ny = -n_ewald; ny <= n_ewald; ++ny) {
                if (nx == 0 && ny == 0)
                    continue;
                const double kx = dk * nx, ky = dk * ny;
                const double k2 = kx * kx + ky * ky;
                const double G = -std::exp(-k2 / (4.0 * eta * eta)) / k2;
                const auto eikr = etx[nx + n_ewald] * ety[ny + n_ewald];
                const auto rho_eikr = rho[(nx + n_ewald) * d + (ny + n_ewald)] * eikr;
                pot_long += G * std::real(rho_eikr);
                if (grad_out) {
                    const double im = -std::imag(rho_eikr);
                    grad_long[0] += G * kx * im;
                    grad_long[1] += G * ky * im;
                }
            }
        pot_out += (2.0 * M_PI / V_box) * pot_long;
        if (grad_out) {
            grad_out[0] += (2.0 * M_PI / V_box) * grad_long[0];
            grad_out[1] += (2.0 * M_PI / V_box) * grad_long[1];
        }
    };

    struct PrecisionCase {
        int n_digits;
        double eps;
        double tol_pot;
        double tol_grad;
    };
    const PrecisionCase cases[] = {{3, 1e-3, 1e-2, 1e-1}, {6, 1e-6, 1e-4, 1e-3}, {9, 1e-9, 1e-7, 1e-6}};

    for (const auto &pc : cases) {
        for (int with_grad = 0; with_grad <= 1; ++with_grad) {
            const auto eval = with_grad ? DMK_POTENTIAL_GRAD : DMK_POTENTIAL;
            const int odim = with_grad ? 1 + n_dim : 1;
            const std::string label = "n_digits=" + std::to_string(pc.n_digits) + (with_grad ? " pot+grad" : " pot");

            SUBCASE(label.c_str()) {
                pdmk_params params;
                params.eps = pc.eps;
                params.n_dim = n_dim;
                params.n_per_leaf = 50;
                params.eval_src = eval;
                params.eval_trg = eval;
                params.kernel = DMK_LAPLACE;
                params.use_periodic = true;
                params.log_level = 6;

                sctl::Vector<double> pot_src(n_src * odim), pot_trg(n_trg * odim);
                pdmk_tree tree =
                    pdmk_tree_create(comm, params, n_src, &r_src[0], &charges[0], &rnormal[0], n_trg, &r_trg[0]);
                pdmk_tree_eval(tree, &pot_src[0], &pot_trg[0]);
                pdmk_tree_destroy(tree);

                // Targets only (avoids the log self-term).
                const int n_test_trg = std::min(n_trg, 50);
                double e2t = 0, r2t = 0, e2gt = 0, r2gt = 0;
                for (int i = 0; i < n_test_trg; ++i) {
                    double ref_pot, ref_grad[2];
                    ewald_pot_grad(&r_trg[i * n_dim], ref_pot, with_grad ? ref_grad : nullptr);
                    e2t += sctl::pow<2>(pot_trg[i * odim] - ref_pot);
                    r2t += sctl::pow<2>(ref_pot);
                    if (with_grad)
                        for (int dd = 0; dd < n_dim; ++dd) {
                            e2gt += sctl::pow<2>(pot_trg[i * odim + 1 + dd] - ref_grad[dd]);
                            r2gt += sctl::pow<2>(ref_grad[dd]);
                        }
                }
                auto safe_l2 = [](double e2, double r2) { return r2 > 0 ? std::sqrt(e2 / r2) : std::sqrt(e2); };
                VERBOSE_MESSAGE("2D Laplace PBC: ", label, " pot_trg=", safe_l2(e2t, r2t));
                CHECK(safe_l2(e2t, r2t) < pc.tol_pot);
                if (with_grad)
                    CHECK(safe_l2(e2gt, r2gt) < pc.tol_grad);
            }
        }
    }
}
