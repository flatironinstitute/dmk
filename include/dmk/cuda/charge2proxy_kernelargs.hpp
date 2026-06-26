namespace dmk::cuda {

template <typename Real>
struct Charge2ProxyArgs {
    int n_groups = 0;
    int n_order = 0;
    int n_charge_dim = 0; // = n_tables_up

    // Per-group, length n_groups.
    const int *center_boxes = nullptr;
    const int *levels = nullptr;
    const int *src_box_flat_offsets = nullptr;
    const int *n_src_boxes_per_group = nullptr;
    const int *src_boxes_flat = nullptr; // total length = sum(n_src_boxes_per_group)

    // Shared device state.
    const Real *centers = nullptr;       // [n_boxes * DIM] F-major (axis, box)
    const Real *inv_box_scale = nullptr; // [n_levels]; = 2 / boxsize[L]
    const Real *r_src = nullptr;   // F-major positions per box
    const long *r_src_offsets = nullptr;
    const int *src_counts = nullptr;
    const Real *charge = nullptr; // F-major [n_charge_dim, n_src] per box
    const long *charge_offsets = nullptr;

    Real *proxy_flat = nullptr; // d_proxy_coeffs_upward (additive write)
    const long *proxy_offsets = nullptr;
    const int *group_perm = nullptr;
    int n_active_groups = 0;
};
}
