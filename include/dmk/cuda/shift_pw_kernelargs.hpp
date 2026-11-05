#pragma once

namespace dmk::cuda {

/// One shift_pw group: the plane-wave-evaluating children of one parent (the root alone at level
/// 0) and the 4^DIM neighbourhood they draw on. A member sits at p in {0,1}^DIM of the group's
/// frame and a source at q in {-1,..,2}^DIM; the member takes the source exactly when q - p is one
/// of its 3^DIM shift offsets, the source at q = p being the member itself.
///
/// The translation is separable, e^{i(q-p).theta} = e^{i q.theta} e^{-i p.theta}, so the kernel
/// applies the first factor once per source and the second once per member, and between them a
/// (source, member) pair is a plain add.
///
/// Halo position h = sum_a (q_a + 1) 4^(DIM-1-a); member slot t = sum_a p_a 2^(DIM-1-a). Sized
/// for DIM = 3; a 2D group uses the leading entries.
struct ShiftPwHalo {
    long pw_off[64];        ///< source pw_out offset in complex units, -1 where there is none
    unsigned char take[64]; ///< bit t set where member t takes that source
    int member[8];          ///< level-local pw_in slot of member t, -1 where absent
};

template <typename Real>
struct ShiftPwArgs {
    int n_groups_at_level = 0;
    long pw_in_stride = 0;

    const ShiftPwHalo *halo = nullptr; ///< this level's groups
    const int *mode_axes = nullptr;    ///< per live slot: its per-axis mode indices, 8 bits each, axis 0 lowest
    const Real *phase = nullptr;       ///< this level's e^{i theta_j}, [n_pw] complex

    const Real *pw_out_flat = nullptr;
    Real *pw_in_pool = nullptr;
};
} // namespace dmk::cuda
