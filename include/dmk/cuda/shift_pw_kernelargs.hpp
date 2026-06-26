namespace dmk::cuda {

template <typename Real>
struct ShiftPwArgs {
    int n_boxes_at_level = 0;
    int n_neighbors = 0;
    int n_charge_dim = 0;
    int n_pw_modes = 0;

    long pw_in_stride = 0;

    const int *box_ids = nullptr;
    const int *neighbors = nullptr;
    const long *pw_out_offsets = nullptr;
    const unsigned char *is_global_leaf = nullptr;

    const Real *pw_out_flat = nullptr;
    const Real *wpwshift = nullptr;

    Real *pw_in_pool = nullptr;
};
}