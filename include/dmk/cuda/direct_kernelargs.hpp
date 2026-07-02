#pragma once

namespace dmk::cuda {

template <typename Real>
struct DirectByBoxArgs {
    int n_work = 0;
    int n_levels = 0;
    int nlist1_stride = 0;

    Real thresh2 = Real{0};

    const int *direct_work = nullptr;
    const int *target_counts = nullptr;
    const int *box_levels = nullptr;
    const int *list1_count = nullptr;
    const int *list1_flat = nullptr;

    const unsigned char *ifpwexp = nullptr;

    const int *src_counts = nullptr;

    const Real *r_target_flat = nullptr;
    const long *r_target_offsets = nullptr;

    Real *pot_flat = nullptr;
    const long *pot_offsets = nullptr;

    const Real *r_src_flat = nullptr;
    const long *r_src_offsets = nullptr;

    const Real *charge_flat = nullptr;
    const long *charge_offsets = nullptr;

    const Real *normal_flat = nullptr;
    const long *normal_offsets = nullptr;

    const Real *direct_rsc = nullptr;
    const Real *direct_cen = nullptr;
    const Real *direct_d2max = nullptr;
};

} // namespace dmk::cuda
