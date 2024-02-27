#include <dmk.h>
#include <sctl.hpp>

#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

namespace DMK {
spdlog::logger &get_logger() {
    bool first_call = true;
    if (first_call) {
        first_call = false;
        auto comm = sctl::Comm::World();
        if (comm.Rank() == 0)
            spdlog::logger("DMK", std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>());
        else
            spdlog::logger("DMK", std::make_shared<spdlog::sinks::null_sink_st>());
    }

    return *spdlog::get("DMK");
}

template <typename T, int DIM>
void pdmk(const pdmk_params &params, int n_src, const T *r_src, const T *charge, const T *normal, const T *dipole_str,
          int n_trg, const T *r_trg, T *pot, T *grad, T *hess, T *pottarg, T *gradtarg, T *hesstarg) {
    spdlog::cfg::load_env_levels();
    auto &logger = get_logger();
    logger.debug("PDMK called");

    constexpr int balance_21 = true;
    auto comm = sctl::Comm::World();
    sctl::PtTree<T, DIM> tree(comm);

    auto r_vec = sctl::Vector<T>(n_src, const_cast<T *>(r_src), false);
    auto charge_vec = sctl::Vector<T>(n_src, const_cast<T *>(r_src), false);

    logger.debug("Building tree");
    tree.AddParticles("pdmk_src", r_vec);
    tree.AddParticleData("pdmk_charge", "pdmk_src", charge_vec);
    tree.UpdateRefinement(r_vec, params.n_per_leaf, balance_21, params.use_periodic);
    logger.debug("Tree build completed");
}

} // namespace DMK
extern "C" {
void pdmkf(pdmk_params params, int n_src, const float *r_src, const float *charge, const float *normal,
           const float *dipole_str, int n_trg, const float *r_trg, float *pot, float *grad, float *hess, float *pottarg,
           float *gradtarg, float *hesstarg) {
    if (params.n_dim == 2)
        return DMK::pdmk<float, 2>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot, grad, hess,
                                   pottarg, gradtarg, hesstarg);
    if (params.n_dim == 3)
        return DMK::pdmk<float, 3>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot, grad, hess,
                                   pottarg, gradtarg, hesstarg);
}

void pdmk(pdmk_params params, int n_src, const double *r_src, const double *charge, const double *normal,
          const double *dipole_str, int n_trg, const double *r_trg, double *pot, double *grad, double *hess,
          double *pottarg, double *gradtarg, double *hesstarg) {
    if (params.n_dim == 2)
        return DMK::pdmk<double, 2>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot, grad, hess,
                                    pottarg, gradtarg, hesstarg);
    if (params.n_dim == 3)
        return DMK::pdmk<double, 3>(params, n_src, r_src, charge, normal, dipole_str, n_trg, r_trg, pot, grad, hess,
                                    pottarg, gradtarg, hesstarg);
}
}
