#include <dmk/direct.hpp>
#include <dmk/testing.hpp>
#include <dmk/util.hpp>

#include <cmath>
#include <vector>

// Naive reference implementations for each kernel, used to validate the vectorized EvalPairs path.

namespace {

constexpr int N_SRC = 500;
constexpr int N_TRG = 200;
constexpr long SEED = 42;

struct TestData {
    int n_dim;
    std::vector<double> r_src, r_trg, charges, normals;

    TestData(int n_dim_, int charge_dim, long seed) : n_dim(n_dim_) {
        std::vector<double> unused_trg, unused_normal;
        dmk::util::init_test_data(n_dim, charge_dim, N_SRC, N_TRG, /*uniform=*/true,
                                  /*set_fixed_charges=*/false, r_src, r_trg, normals, charges, seed);
    }
};

void naive_yukawa_3d(const TestData &td, double lambda, std::vector<double> &pot) {
    pot.assign(N_TRG, 0.0);
    for (int t = 0; t < N_TRG; ++t) {
        const double *xt = &td.r_trg[t * 3];
        for (int s = 0; s < N_SRC; ++s) {
            const double *xs = &td.r_src[s * 3];
            double dx = xt[0] - xs[0], dy = xt[1] - xs[1], dz = xt[2] - xs[2];
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r == 0.0)
                continue;
            pot[t] += td.charges[s] * std::exp(-lambda * r) / r;
        }
    }
}

void naive_laplace_3d(const TestData &td, std::vector<double> &pot) {
    pot.assign(N_TRG, 0.0);
    for (int t = 0; t < N_TRG; ++t) {
        const double *xt = &td.r_trg[t * 3];
        for (int s = 0; s < N_SRC; ++s) {
            const double *xs = &td.r_src[s * 3];
            double dx = xt[0] - xs[0], dy = xt[1] - xs[1], dz = xt[2] - xs[2];
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r == 0.0)
                continue;
            pot[t] += td.charges[s] / r;
        }
    }
}

void naive_laplace_2d(const TestData &td, std::vector<double> &pot) {
    pot.assign(N_TRG, 0.0);
    for (int t = 0; t < N_TRG; ++t) {
        const double *xt = &td.r_trg[t * 2];
        for (int s = 0; s < N_SRC; ++s) {
            const double *xs = &td.r_src[s * 2];
            double dx = xt[0] - xs[0], dy = xt[1] - xs[1];
            double r2 = dx * dx + dy * dy;
            if (r2 == 0.0)
                continue;
            pot[t] += 0.5 * td.charges[s] * std::log(r2);
        }
    }
}

void naive_sqrt_laplace_2d(const TestData &td, std::vector<double> &pot) {
    pot.assign(N_TRG, 0.0);
    for (int t = 0; t < N_TRG; ++t) {
        const double *xt = &td.r_trg[t * 2];
        for (int s = 0; s < N_SRC; ++s) {
            const double *xs = &td.r_src[s * 2];
            double dx = xt[0] - xs[0], dy = xt[1] - xs[1];
            double r2 = dx * dx + dy * dy;
            if (r2 == 0.0)
                continue;
            pot[t] += td.charges[s] / std::sqrt(r2);
        }
    }
}

void naive_sqrt_laplace_3d(const TestData &td, std::vector<double> &pot) {
    pot.assign(N_TRG, 0.0);
    for (int t = 0; t < N_TRG; ++t) {
        const double *xt = &td.r_trg[t * 3];
        for (int s = 0; s < N_SRC; ++s) {
            const double *xs = &td.r_src[s * 3];
            double dx = xt[0] - xs[0], dy = xt[1] - xs[1], dz = xt[2] - xs[2];
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 == 0.0)
                continue;
            pot[t] += td.charges[s] / r2;
        }
    }
}

void naive_stokeslet_3d(const TestData &td, std::vector<double> &pot) {
    pot.assign(N_TRG * 3, 0.0);
    for (int t = 0; t < N_TRG; ++t) {
        const double *xt = &td.r_trg[t * 3];
        for (int s = 0; s < N_SRC; ++s) {
            const double *xs = &td.r_src[s * 3];
            const double *f = &td.charges[s * 3];
            double dx = xt[0] - xs[0], dy = xt[1] - xs[1], dz = xt[2] - xs[2];
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 == 0.0)
                continue;
            double rinv = 1.0 / std::sqrt(r2);
            double rinv3 = rinv * rinv * rinv;
            double fdotr = f[0] * dx + f[1] * dy + f[2] * dz;
            // G_{ij} = 0.5 * (delta_{ij}/r + r_i r_j / r^3)
            pot[t * 3 + 0] += 0.5 * (f[0] * rinv + fdotr * dx * rinv3);
            pot[t * 3 + 1] += 0.5 * (f[1] * rinv + fdotr * dy * rinv3);
            pot[t * 3 + 2] += 0.5 * (f[2] * rinv + fdotr * dz * rinv3);
        }
    }
}

void naive_stresslet_3d(const TestData &td, std::vector<double> &pot) {
    pot.assign(N_TRG * 3, 0.0);
    for (int t = 0; t < N_TRG; ++t) {
        const double *xt = &td.r_trg[t * 3];
        for (int s = 0; s < N_SRC; ++s) {
            const double *xs = &td.r_src[s * 3];
            const double *mu = &td.charges[s * 3];
            const double *nu = &td.normals[s * 3];
            double dx = xt[0] - xs[0], dy = xt[1] - xs[1], dz = xt[2] - xs[2];
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 == 0.0)
                continue;
            double r = std::sqrt(r2);
            double r5 = r2 * r2 * r;
            double r3 = r2 * r;
            double mudotr = mu[0] * dx + mu[1] * dy + mu[2] * dz;
            double nudotr = nu[0] * dx + nu[1] * dy + nu[2] * dz;
            double mudotnu = mu[0] * nu[0] + mu[1] * nu[1] + mu[2] * nu[2];
            // u_i = -3 r_i (r.mu)(r.nu)/r^5 + (r_i(mu.nu) + mu_i(r.nu) + nu_i(r.mu))/r^3
            double offd = -3.0 * mudotr * nudotr / r5;
            double diag = 1.0 / r3;
            pot[t * 3 + 0] += dx * offd + diag * (dx * mudotnu + mu[0] * nudotr + nu[0] * mudotr);
            pot[t * 3 + 1] += dy * offd + diag * (dy * mudotnu + mu[1] * nudotr + nu[1] * mudotr);
            pot[t * 3 + 2] += dz * offd + diag * (dz * mudotnu + mu[2] * nudotr + nu[2] * mudotr);
        }
    }
}

double rel_l2_error(const std::vector<double> &test, const std::vector<double> &ref) {
    double err = 0.0, nrm = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        err += (test[i] - ref[i]) * (test[i] - ref[i]);
        nrm += ref[i] * ref[i];
    }
    return std::sqrt(err / nrm);
}

} // namespace

TEST_CASE("[DMK] direct eval: Yukawa 3D") {
    TestData td(3, 1, SEED);
    const double lambda = 6.0;

    std::vector<double> ref, test;
    naive_yukawa_3d(td, lambda, ref);

    auto func = dmk::get_direct_evaluator<double>(DMK_YUKAWA, DMK_POTENTIAL, 3, lambda);
    test.assign(N_TRG, 0.0);
    func(N_SRC, td.r_src.data(), td.charges.data(), nullptr, N_TRG, td.r_trg.data(), test.data());

    CHECK(rel_l2_error(test, ref) < 1e-12);
}

TEST_CASE("[DMK] direct eval: Laplace 3D") {
    TestData td(3, 1, SEED);

    std::vector<double> ref, test;
    naive_laplace_3d(td, ref);

    auto func = dmk::get_direct_evaluator<double>(DMK_LAPLACE, DMK_POTENTIAL, 3, 0.0);
    test.assign(N_TRG, 0.0);
    func(N_SRC, td.r_src.data(), td.charges.data(), nullptr, N_TRG, td.r_trg.data(), test.data());

    CHECK(rel_l2_error(test, ref) < 1e-12);
}

TEST_CASE("[DMK] direct eval: Laplace 2D") {
    TestData td(2, 1, SEED);

    std::vector<double> ref, test;
    naive_laplace_2d(td, ref);

    auto func = dmk::get_direct_evaluator<double>(DMK_LAPLACE, DMK_POTENTIAL, 2, 0.0);
    test.assign(N_TRG, 0.0);
    func(N_SRC, td.r_src.data(), td.charges.data(), nullptr, N_TRG, td.r_trg.data(), test.data());

    CHECK(rel_l2_error(test, ref) < 1e-12);
}

TEST_CASE("[DMK] direct eval: SqrtLaplace 2D") {
    TestData td(2, 1, SEED);

    std::vector<double> ref, test;
    naive_sqrt_laplace_2d(td, ref);

    auto func = dmk::get_direct_evaluator<double>(DMK_SQRT_LAPLACE, DMK_POTENTIAL, 2, 0.0);
    test.assign(N_TRG, 0.0);
    func(N_SRC, td.r_src.data(), td.charges.data(), nullptr, N_TRG, td.r_trg.data(), test.data());

    CHECK(rel_l2_error(test, ref) < 1e-12);
}

TEST_CASE("[DMK] direct eval: SqrtLaplace 3D") {
    TestData td(3, 1, SEED);

    std::vector<double> ref, test;
    naive_sqrt_laplace_3d(td, ref);

    auto func = dmk::get_direct_evaluator<double>(DMK_SQRT_LAPLACE, DMK_POTENTIAL, 3, 0.0);
    test.assign(N_TRG, 0.0);
    func(N_SRC, td.r_src.data(), td.charges.data(), nullptr, N_TRG, td.r_trg.data(), test.data());

    CHECK(rel_l2_error(test, ref) < 1e-12);
}

TEST_CASE("[DMK] direct eval: Stokeslet 3D") {
    TestData td(3, 3, SEED);

    std::vector<double> ref, test;
    naive_stokeslet_3d(td, ref);

    auto func = dmk::get_direct_evaluator<double>(DMK_STOKESLET, DMK_VELOCITY, 3, 0.0);
    test.assign(N_TRG * 3, 0.0);
    func(N_SRC, td.r_src.data(), td.charges.data(), nullptr, N_TRG, td.r_trg.data(), test.data());

    CHECK(rel_l2_error(test, ref) < 1e-12);
}

TEST_CASE("[DMK] direct eval: Stresslet 3D") {
    TestData td(3, 3, SEED);

    std::vector<double> ref, test;
    naive_stresslet_3d(td, ref);

    auto func = dmk::get_direct_evaluator<double>(DMK_STRESSLET, DMK_VELOCITY, 3, 0.0);
    test.assign(N_TRG * 3, 0.0);
    func(N_SRC, td.r_src.data(), td.charges.data(), td.normals.data(), N_TRG, td.r_trg.data(), test.data());

    CHECK(rel_l2_error(test, ref) < 1e-12);
}
