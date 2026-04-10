#include <atomic>
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
#include <omp.h>
#include <ranges>
#include <sctl/profile.hpp>
#include <unistd.h>

#include <dmk/omp_wrapper.hpp>

namespace {

// Priority work queue using OMP locks. Four priority levels:
//   0 (highest): upward pass tasks (critical path to unblock expansion)
//   1:           expansion downward sweep tasks (critical path, unlocks children)
//   2:           form_outgoing tasks
//   3 (lowest):  direct evaluation tasks (bulk work, always available)
//
// Workers always drain highest-priority non-empty queue first.
struct PriorityWorkQueue {
    static constexpr int N_PRI = 4;
    static constexpr int PRI_UPWARD = 0;
    static constexpr int PRI_DOWNWARD = 1;
    static constexpr int PRI_OUTGOING = 2;
    static constexpr int PRI_DIRECT = 3;

    std::vector<int> queues[N_PRI];
    omp_lock_t locks[N_PRI];
    std::atomic<int> counts[N_PRI];
    std::atomic<int> total{0};

    void init() {
        for (int i = 0; i < N_PRI; ++i) {
            omp_init_lock(&locks[i]);
            counts[i].store(0, std::memory_order_relaxed);
        }
        total.store(0, std::memory_order_relaxed);
    }

    void destroy() {
        for (int i = 0; i < N_PRI; ++i)
            omp_destroy_lock(&locks[i]);
    }

    void push(int item, int priority) {
        omp_set_lock(&locks[priority]);
        queues[priority].push_back(item);
        omp_unset_lock(&locks[priority]);
        counts[priority].fetch_add(1, std::memory_order_release);
        total.fetch_add(1, std::memory_order_release);
    }

    void push_batch(auto &&items, int priority) {
        omp_set_lock(&locks[priority]);
        int n = 0;
        for (const auto &el : items) {
            queues[priority].push_back(el);
            n++;
        }
        omp_unset_lock(&locks[priority]);
        counts[priority].fetch_add(n, std::memory_order_release);
        total.fetch_add(n, std::memory_order_release);
    }

    bool try_pop(int &item, int &priority) {
        for (int p = 0; p < N_PRI; ++p) {
            if (counts[p].load(std::memory_order_acquire) == 0)
                continue;
            omp_set_lock(&locks[p]);
            if (!queues[p].empty()) {
                item = queues[p].back();
                queues[p].pop_back();
                omp_unset_lock(&locks[p]);
                counts[p].fetch_sub(1, std::memory_order_relaxed);
                priority = p;
                return true;
            }
            omp_unset_lock(&locks[p]);
        }
        return false;
    }

    void done() { total.fetch_sub(1, std::memory_order_release); }

    bool all_done() const { return total.load(std::memory_order_acquire) <= 0; }
};

} // anonymous namespace

namespace dmk {
template <typename T, int DIM>
void DMKPtTree<T, DIM>::eval_pq() {
    sctl::Profile::Scoped prof("pdmk_tree_eval", &comm_);
    auto &logger = dmk::get_logger(comm_, params.log_level);
    logger->info("eval_pq() started");

    double t_start = MY_OMP_GET_WTIME();
    std::atomic<double> t_zeroing{0}, t_c2p{0}, t_deps{0}, t_queue{0}, t_mpi{0}, t_outgoing_done{0},
        t_downward_seeded{0};

    int n_halo_only = 0;
    for (int b = 0; b < (int)n_boxes(); ++b)
        if (ifpwexp[b] && src_counts_with_halo[b] > 0 && src_counts_owned[b] == 0)
            n_halo_only++;
    auto &rank_logger = dmk::get_rank_logger(comm_, params.log_level);
    rank_logger->info("halo_only={}, n_boxes={}", n_halo_only, n_boxes());

    const auto &node_lists = this->GetNodeLists();
    const auto &node_attr = this->GetNodeAttr();
    const auto &node_mid = this->GetNodeMID();
    const std::size_t n_coeffs = n_tables * sctl::pow<DIM>(n_order);
    constexpr int n_children = 1u << DIM;
    // FIXME: windowed vs diff
    const int n_pw = expansion_constants.n_pw_diff;
    const int n_pw_modes = sctl::pow<DIM - 1>(n_pw) * ((n_pw + 1) / 2);
    const int n_pw_per_box = n_pw_modes * n_tables;

    // =================================================================
    // Build upward pass task dependency graph
    // =================================================================
    // Group charge2proxy work items by center_box to avoid write races.
    // Each group becomes one task.
    struct C2PGroup {
        int center_box;
        int level;
        std::vector<int> src_boxes;
    };
    std::vector<C2PGroup> c2p_groups;
    {
        std::vector<int> center_to_group(n_boxes(), -1);
        for (auto &w : charge2proxy_work) {
            if (center_to_group[w.center_box] == -1) {
                center_to_group[w.center_box] = (int)c2p_groups.size();
                c2p_groups.push_back({w.center_box, w.level, {w.src_box}});
            } else {
                c2p_groups[center_to_group[w.center_box]].src_boxes.push_back(w.src_box);
            }
        }
    }
    // Map center_box -> index into c2p_groups (for dependency wiring)
    std::vector<int> box_to_c2p_group(n_boxes(), -1);
    for (int g = 0; g < (int)c2p_groups.size(); ++g)
        box_to_c2p_group[c2p_groups[g].center_box] = g;
    t_c2p = MY_OMP_GET_WTIME() - t_start;

    // ---- Init workspaces ----
    int n_threads;
#pragma omp parallel
#pragma omp single
    {
        n_threads = MY_OMP_GET_NUM_THREADS();
        workspaces_.ReInit(n_threads);
    }

    // ---- Init potential buffers ----
    pot_src_sorted.SetZero();
    pot_trg_sorted.SetZero();
    init_planewave_data();

    // Separate direct potential buffers (no write conflicts with expansion path)
    sctl::Vector<T> pot_src_direct(pot_src_sorted.Dim());
    sctl::Vector<T> pot_trg_direct(pot_trg_sorted.Dim());
    pot_src_direct.SetZero();
    pot_trg_direct.SetZero();

    // ---- Precompute self-interaction constants for direct eval ----
    T w0[SCTL_MAX_DEPTH];
    for (int lvl = 0; lvl < n_levels(); ++lvl)
        w0[lvl] = get_self_interaction_constant<T, DIM>(fourier_data, params.kernel, lvl, boxsize[lvl]);

    // ---- Init upward pass state ----
    sctl::Vector<sctl::Long> proxy_counts;

    this->GetData(proxy_coeffs_upward, proxy_counts, "proxy_coeffs");

#pragma omp parallel for schedule(static)
    for (int b = 0; b < (int)n_boxes(); ++b) {
        if (ifpwexp[b] && proxy_coeffs_offsets[b] != -1 && src_counts_owned[b] == 0)
            proxy_view_upward(b) = 0;
    }

    // Upward dependency counter per box.
    // A box's tensorprod task is ready when deps hits 0.
    // deps = (1 if has charge2proxy group targeting it) + (number of pw-children)
    // For boxes without has_proxy_from_children: no tensorprod, charge2proxy
    // completion directly signals the parent.
    // -1 = sentinel, box not participating in upward pass.
    std::vector<std::atomic<int>> upward_deps(n_boxes());
    int n_upward_tasks = 0; // total tasks that signal upward completion
    for (int b = 0; b < (int)n_boxes(); ++b) {
        if (!(src_counts_owned[b] > 0 && ifpwexp[b])) {
            upward_deps[b].store(-1, std::memory_order_relaxed);
            continue;
        }
        if (has_proxy_from_children[b]) {
            int deps = (box_to_c2p_group[b] >= 0 ? 1 : 0);
            for (auto child : node_lists[b].child)
                if (child >= 0 && src_counts_owned[child] > 0 && ifpwexp[child])
                    deps++;
            upward_deps[b].store(deps, std::memory_order_relaxed);
        } else {
            // No tensorprod needed. Will be signaled complete after charge2proxy.
            upward_deps[b].store(0, std::memory_order_relaxed);
        }
        n_upward_tasks++;
    }
    std::atomic<int> upward_remaining{n_upward_tasks};

    // =================================================================
    // Build downward sweep dependency graph
    // =================================================================
    std::vector<std::atomic<int>> downward_deps(n_boxes());
    int n_downward_tasks = 0;
    for (int b = 0; b < (int)n_boxes(); ++b) {
        bool needs = (ifpwexp[b] || iftensprodeval[b]) && (src_counts_owned[b] + trg_counts_owned[b] > 0);
        if (!needs) {
            downward_deps[b].store(-1, std::memory_order_relaxed);
            continue;
        }
        bool is_root = (node_mid[b].Depth() == 0);
        downward_deps[b].store(is_root ? 0 : 1, std::memory_order_relaxed);
        n_downward_tasks++;
    }

    t_deps = MY_OMP_GET_WTIME() - t_start;

    // =================================================================
    // Build work queue
    // =================================================================
    PriorityWorkQueue queue;
    queue.init();

    // Total tasks: c2p_groups + tensorprod boxes + outgoing boxes + downward boxes + direct boxes
    // We track total in the queue; direct and c2p_groups are seeded immediately.
    // Others are seeded as dependencies are met.

    // Encode task type in the item value:
    //   [0, n_c2p_groups)                            -> charge2proxy group
    //   [n_c2p_groups, n_c2p_groups + n_boxes)       -> tensorprod for box (item - n_c2p_groups = box)
    //   OUTGOING tasks use box ID directly (separate priority level disambiguates)
    //   DOWNWARD tasks use box ID directly
    //   DIRECT tasks use index into direct_work
    const int C2P_OFFSET = 0;
    const int TPROD_OFFSET = (int)c2p_groups.size();

    int n_direct_tasks = (int)direct_work.size();
    int n_outgoing_tasks = 0;
    for (int b = 0; b < (int)n_boxes(); ++b)
        if (ifpwexp[b] || pw_out_offsets[b] != -1)
            n_outgoing_tasks++;

    // Total: all task types
    int total_tasks = (int)c2p_groups.size() + n_upward_tasks + n_outgoing_tasks + n_downward_tasks + n_direct_tasks;
    // Note: c2p_groups that don't correspond to has_proxy_from_children boxes
    // are counted in n_upward_tasks too (they directly complete the box).
    // Let's just be conservative and use queue.total for termination.

    // Enqueue direct tasks (lowest priority, always ready)
    queue.push_batch(std::views::iota(0, n_direct_tasks), PriorityWorkQueue::PRI_DIRECT);

    // Enqueue charge2proxy group tasks (highest priority)
    queue.push_batch(std::views::iota(C2P_OFFSET, C2P_OFFSET + TPROD_OFFSET), PriorityWorkQueue::PRI_UPWARD);

    // Track when all outgoing are done so we can seed downward sweep
    std::atomic<int> outgoing_remaining{0};

    // Phase flags
    std::atomic<bool> mpi_done{false};
    std::atomic<bool> outgoing_seeded{false};
    std::atomic<bool> downward_seeded{false};

    // =================================================================
    // Upward-complete signal: box's proxy_upward is finished.
    // Decrements parent's upward_deps. If parent's deps hit 0 and
    // has_proxy_from_children, enqueue tensorprod task for parent.
    // =================================================================
    auto signal_upward_complete = [&](int box) {
        upward_remaining.fetch_sub(1, std::memory_order_release);
        int parent = node_lists[box].parent;
        if (parent >= 0 && upward_deps[parent].load(std::memory_order_acquire) > 0) {
            if (upward_deps[parent].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                // Parent's deps are met — enqueue its tensorprod task
                queue.push(TPROD_OFFSET + parent, PriorityWorkQueue::PRI_UPWARD);
            }
        }
    };

    t_queue.store(MY_OMP_GET_WTIME() - t_start, std::memory_order_relaxed);

    // =================================================================
    // Worker
    // =================================================================
#pragma omp parallel
    {
        const int tid = MY_OMP_GET_THREAD_NUM();
        sctl::Vector<T> &workspace = workspaces_[tid];

        // Thread-local buffers for direct evaluation (same as evaluate_direct_interactions)
        constexpr int MAX_CHARGE_DIM = 3;
        constexpr int MAX_OUTPUT_DIM = 9;
        constexpr int MAX_PTS = 1000;
        util::StackOrHeapBuffer<T, DIM * MAX_PTS> r_buf(DIM * params.n_per_leaf);
        util::StackOrHeapBuffer<T, MAX_CHARGE_DIM * MAX_PTS> charge_buf(kernel_input_dim * params.n_per_leaf);
        util::StackOrHeapBuffer<T, DIM * MAX_PTS> r_trg_buf(DIM * params.n_per_leaf);
        util::StackOrHeapBuffer<T, MAX_OUTPUT_DIM * MAX_PTS> pot_buf(kernel_output_dim_max * params.n_per_leaf);
        util::StackOrHeapBuffer<int, MAX_PTS> index_map(params.n_per_leaf);

        // Thread-local pw_in buffer for expansion downward sweep
        sctl::Vector<std::complex<T>> pw_in(n_pw_per_box);
        auto pw_in_view = [&]() {
            if constexpr (DIM == 2)
                return ndview<std::complex<T>, DIM + 1>({n_pw, (n_pw + 1) / 2, n_tables}, &pw_in[0]);
            else if constexpr (DIM == 3)
                return ndview<std::complex<T>, DIM + 1>({n_pw, n_pw, (n_pw + 1) / 2, n_tables}, &pw_in[0]);
        }();

        while (true) {
            // === Thread 0: check for phase transitions ===
            if (tid == 0) {
                // Transition: upward pass complete → MPI broadcast
                if (!mpi_done.load(std::memory_order_acquire) &&
                    upward_remaining.load(std::memory_order_acquire) == 0) {

                    sctl::Profile::Tic("broadcast_proxy_coeffs", &comm_);
                    this->template ReduceBroadcast<T>("proxy_coeffs");
                    this->GetData(proxy_coeffs_upward, proxy_counts, "proxy_coeffs");
                    long last_offset = 0;
                    for (int box = 0; box < (int)n_boxes(); ++box) {
                        if (proxy_counts[box]) {
                            proxy_coeffs_offsets[box] = last_offset;
                            last_offset += n_coeffs;
                        } else {
                            proxy_coeffs_offsets[box] = -1;
                        }
                    }
                    sctl::Profile::Toc();
                    mpi_done.store(true, std::memory_order_release);
                    t_mpi.store(MY_OMP_GET_WTIME() - t_start, std::memory_order_relaxed);
                }

                // Transition: MPI done → seed form_outgoing tasks
                if (mpi_done.load(std::memory_order_acquire) && !outgoing_seeded.load(std::memory_order_acquire)) {
                    std::fill(proxy_down_zeroed.begin(), proxy_down_zeroed.end(), 0);
                    { // Windowed kernel for root
                        const ndview<std::complex<T>, 2> p2pw({n_pw, n_order}, &window_fourier_data.poly2pw[0]);
                        const ndview<std::complex<T>, 2> pw2p({n_pw, n_order}, &window_fourier_data.pw2poly[0]);
                        proxy_view_downward(0) = 0;
                        proxy_down_zeroed[0] = true;
                        dmk::proxy::proxycharge2pw<T, DIM>(proxy_view_upward(0), p2pw, pw_out_view(0), workspace);
                        multiply_kernelFT_cd2p<T, DIM>(window_fourier_data.radialft, pw_out_view(0));
                        dmk::planewave_to_proxy_potential<T, DIM>(pw_out_view(0), pw2p, proxy_view_downward(0),
                                                                  workspace);
                    }

                    // Count first
                    int n_out = 0;
                    for (int b = 0; b < (int)n_boxes(); ++b)
                        if ((ifpwexp[b] && proxy_coeffs_offsets[b] != -1) || (pw_out_offsets[b] != -1))
                            n_out++;

                    outgoing_remaining.store(n_out, std::memory_order_release);
                    outgoing_seeded.store(true, std::memory_order_release);

                    // Now push
                    auto outgoing =
                        std::views::iota(0, (int)n_boxes()) | std::views::filter([&](int b) {
                            return (ifpwexp[b] && proxy_coeffs_offsets[b] != -1) || (pw_out_offsets[b] != -1);
                        });
                    queue.push_batch(outgoing, PriorityWorkQueue::PRI_OUTGOING);
                }

                // Transition: all outgoing done → seed downward sweep
                if (outgoing_seeded.load(std::memory_order_acquire) &&
                    !downward_seeded.load(std::memory_order_acquire) &&
                    outgoing_remaining.load(std::memory_order_acquire) == 0) {
                    // Seed root proxy_downward from windowed kernel
                    queue.push(0, PriorityWorkQueue::PRI_DOWNWARD);
                    downward_seeded.store(true, std::memory_order_release);
                    t_downward_seeded.store(MY_OMP_GET_WTIME() - t_start, std::memory_order_relaxed);
                }
            }

            // === Pop a task ===
            int item, priority;
            if (!queue.try_pop(item, priority)) {
                if (downward_seeded.load(std::memory_order_acquire) && queue.all_done())
                    break;
                continue; // spin
            }

            // === Execute task ===
            switch (priority) {

            // ---- UPWARD: charge2proxy group or tensorprod ----
            case PriorityWorkQueue::PRI_UPWARD: {
                if (item < TPROD_OFFSET) {
                    // Charge2proxy group
                    int g = item - C2P_OFFSET;
                    auto &grp = c2p_groups[g];
                    proxy_view_upward(grp.center_box) = 0;
                    for (int src_box : grp.src_boxes) {
                        proxy::charge2proxycharge<T, DIM>(r_src_owned_view(src_box), charge_owned_view(src_box),
                                                          center_view(grp.center_box), 2.0 / boxsize[grp.level],
                                                          proxy_view_upward(grp.center_box), workspace);
                    }
                    int cbox = grp.center_box;
                    if (has_proxy_from_children[cbox]) {
                        // This was one of the deps for cbox's tensorprod
                        if (upward_deps[cbox].fetch_sub(1, std::memory_order_acq_rel) == 1)
                            queue.push(TPROD_OFFSET + cbox, PriorityWorkQueue::PRI_UPWARD);
                    } else {
                        // No tensorprod needed — box is upward-complete
                        signal_upward_complete(cbox);
                    }
                } else {
                    // Tensorprod for a box
                    int box = item - TPROD_OFFSET;
                    if (box_to_c2p_group[box] < 0)
                        proxy_view_upward(box) = 0;
                    for (int ic = 0; ic < n_children; ++ic) {
                        int cb = node_lists[box].child[ic];
                        if (cb < 0 || !(src_counts_owned[cb] > 0 && ifpwexp[cb]))
                            continue;
                        const ndview<T, 2> c2p_view({n_order, DIM}, &c2p[ic * DIM * n_order * n_order]);
                        tensorprod::transform<T, DIM>(n_tables, true, proxy_view_upward(cb), c2p_view,
                                                      proxy_view_upward(box), workspace);
                    }
                    signal_upward_complete(box);
                }
                queue.done();
                break;
            }

            // ---- FORM_OUTGOING: one box ----
            case PriorityWorkQueue::PRI_OUTGOING: {
                int box = item;

                if (ifpwexp[box] && proxy_coeffs_offsets[box] != -1) {
                    int level = node_mid[box].Depth();
                    auto &dfd = difference_fourier_data[level];
                    const ndview<std::complex<T>, 2> poly2pw_view({n_pw, n_order}, &dfd.poly2pw[0]);
                    dmk::proxy::proxycharge2pw<T, DIM>(proxy_view_upward(box), poly2pw_view, pw_out_view(box),
                                                       workspace);
                    multiply_kernelFT_cd2p<T, DIM>(dfd.radialft, pw_out_view(box));
                } else if (pw_out_offsets[box] != -1) {
                    pw_out_view(box) = 0;
                }
                outgoing_remaining.fetch_sub(1, std::memory_order_release);
                queue.done();
                break;
            }

            // ---- DOWNWARD SWEEP: one box ----
            case PriorityWorkQueue::PRI_DOWNWARD: {
                if (debug_omit_pw) {
                    queue.done();
                    break;
                }
                int box = item;
                if (box < 0) {
                    queue.done();
                    break;
                } // dummy task (shouldn't happen now)

                int level = node_mid[box].Depth();
                auto &dfd = difference_fourier_data[level];
                const ndview<std::complex<T>, 2> pw2p({n_pw, n_order}, &dfd.pw2poly[0]);
                const T sc = 2.0 / boxsize[level];

                // Shift neighbor planewaves into local pw_in
                if (ifpwexp[box]) {
                    memcpy(&pw_in[0], pw_out_ptr(box), n_pw_per_box * sizeof(std::complex<T>));
                    constexpr int n_neighbors = sctl::pow<DIM>(3);
                    for (int inb = 0; inb < n_neighbors; ++inb) {
                        int neighbor = node_lists[box].nbr[inb];
                        if (neighbor < 0 || neighbor == box || pw_out_offsets[neighbor] == -1)
                            continue;
                        if (!(!is_global_leaf[box] || !is_global_leaf[neighbor]))
                            continue;
                        int ind = n_neighbors - 1 - inb;
                        ndview<const std::complex<T>, 1> wpwshift_view({n_pw_per_box},
                                                                       &dfd.wpwshift[n_pw_per_box * ind]);
                        shift_planewave<std::complex<T>, DIM>(pw_out_view(neighbor), pw_in_view, wpwshift_view);
                    }

                    // Convert pw_in to proxy_downward
                    if (!proxy_down_zeroed[box]) {
                        proxy_view_downward(box) = 0;
                        proxy_down_zeroed[box] = true;
                    }
                    dmk::planewave_to_proxy_potential<T, DIM>(pw_in_view, pw2p, proxy_view_downward(box), workspace);

                    // Translate to children
                    if (!iftensprodeval[box]) {
                        for (int ic = 0; ic < n_children; ++ic) {
                            int child = node_lists[box].child[ic];
                            if (child < 0 || !(src_counts_owned[child] + trg_counts_owned[child]))
                                continue;
                            const ndview<T, 2> p2c_view({n_order, DIM},
                                                        const_cast<T *>(&p2c[ic * DIM * n_order * n_order]));
                            tensorprod::transform<T, DIM>(n_tables, proxy_down_zeroed[child], proxy_view_downward(box),
                                                          p2c_view, proxy_view_downward(child), workspace);
                            proxy_down_zeroed[child] = true;
                        }
                    }
                }

                // Evaluate at targets
                if (iftensprodeval[box]) {
                    if (src_counts_owned[box]) {
                        if (params.pgh_src == DMK_POTENTIAL)
                            proxy::eval_targets<T, DIM, 1>(proxy_view_downward(box), r_src_owned_view(box),
                                                           center_view(box), sc, pot_src_view(box), workspace);
                        else if (params.pgh_src == DMK_POTENTIAL_GRAD)
                            proxy::eval_targets<T, DIM, 2>(proxy_view_downward(box), r_src_owned_view(box),
                                                           center_view(box), sc, pot_src_view(box), workspace);
                    }
                    if (trg_counts_owned[box]) {
                        if (params.pgh_trg == DMK_POTENTIAL)
                            proxy::eval_targets<T, DIM, 1>(proxy_view_downward(box), r_trg_owned_view(box),
                                                           center_view(box), sc, pot_trg_view(box), workspace);
                        else if (params.pgh_trg == DMK_POTENTIAL_GRAD)
                            proxy::eval_targets<T, DIM, 2>(proxy_view_downward(box), r_trg_owned_view(box),
                                                           center_view(box), sc, pot_trg_view(box), workspace);
                    }
                }

                // Notify children
                for (auto child : node_lists[box].child) {
                    if (child < 0 || downward_deps[child].load(std::memory_order_acquire) == -1)
                        continue;
                    if (downward_deps[child].fetch_sub(1, std::memory_order_acq_rel) == 1)
                        queue.push(child, PriorityWorkQueue::PRI_DOWNWARD);
                }
                queue.done();
                break;
            }

            // ---- DIRECT EVALUATION: one target box ----
            case PriorityWorkQueue::PRI_DIRECT: {
                if (debug_omit_direct) {
                    queue.done();
                    break;
                }

                int idx = item;
                const int trg_box = direct_work[idx];
                const int trg_level = node_mid[trg_box].Depth();

                for (auto src_box : list1(trg_box)) {
                    int src_level = node_mid[src_box].Depth();
                    T bsize = boxsize[src_level];

                    if (ifpwexp[src_box] && src_box == trg_box) {
                        bsize /= T{2.0};
                        src_level = src_level + 1;
                    } else if (src_level < trg_level) {
                        bsize = boxsize[trg_level];
                        src_level = trg_level;
                    }

                    const T d2max = bsize * bsize;
                    const T bsizeinv = T{1} / bsize;
                    T rsc = 2 * bsizeinv;
                    T cen = -bsize / T{2};

                    if ((params.kernel == DMK_SQRT_LAPLACE && DIM == 3) || (params.kernel == DMK_LAPLACE && DIM == 2)) {
                        rsc = 2 * bsizeinv * bsizeinv;
                        cen = T{-1.0};
                    } else if (params.kernel == DMK_YUKAWA)
                        cen = T{-1.0};

                    const bool src_larger = node_mid[src_box].Depth() < node_mid[trg_box].Depth();
                    const bool trg_larger = node_mid[src_box].Depth() > node_mid[trg_box].Depth();

                    auto corner_a = node_mid[src_box].template Coord<T>();
                    auto corner_b = node_mid[trg_box].template Coord<T>();
                    auto size_a = boxsize[node_mid[src_box].Depth()];
                    auto size_b = boxsize[node_mid[trg_box].Depth()];

                    int n_src = src_counts_with_halo[src_box];
                    const T *r_src_ptr = r_src_with_halo_ptr(src_box);
                    const T *charge_ptr = charge_with_halo_ptr(src_box);

                    if (src_larger) {
                        ContactGeometry<T, DIM> geom(corner_a.data(), corner_b.data(), size_a, size_b, d2max);
                        n_src = filter_sources(geom, n_src, r_src_ptr, charge_ptr, kernel_input_dim, r_buf.data(),
                                               charge_buf.data());
                        r_src_ptr = r_buf.data();
                        charge_ptr = charge_buf.data();
                    }
                    if (!n_src)
                        continue;

                    // Evaluate at source points in trg_box → pot_src_direct
                    if (src_counts_owned[trg_box]) {
                        int n_eval = src_counts_owned[trg_box];
                        T *eval_r = r_src_owned_ptr(trg_box);
                        T *eval_pot = &pot_src_direct[pot_src_offsets[trg_box]];

                        if (trg_larger) {
                            ContactGeometry<T, DIM> geom(corner_a.data(), corner_b.data(), size_a, size_b, d2max);
                            n_eval = filter_targets(geom, n_eval, eval_r, r_trg_buf.data(), index_map.data());
                            std::memset(pot_buf.data(), 0, n_eval * kernel_output_dim_src * sizeof(T));
                            eval_r = r_trg_buf.data();
                            eval_pot = pot_buf.data();
                        }
                        if (n_eval > 0) {
                            if (evaluator_by_level_src[src_level])
                                evaluator_by_level_src[src_level](rsc, cen, d2max, 1e-30, n_src, r_src_ptr, charge_ptr,
                                                                  n_eval, eval_r, eval_pot);
                            if (trg_larger)
                                scatter_add_potential(pot_buf.data(), &pot_src_direct[pot_src_offsets[trg_box]],
                                                      index_map.data(), n_eval, kernel_output_dim_src);
                        }
                    }

                    // Evaluate at target points in trg_box → pot_trg_direct
                    if (trg_counts_owned[trg_box]) {
                        int n_eval = trg_counts_owned[trg_box];
                        T *eval_r = r_trg_owned_ptr(trg_box);
                        T *eval_pot = &pot_trg_direct[pot_trg_offsets[trg_box]];

                        if (trg_larger) {
                            ContactGeometry<T, DIM> geom(corner_a.data(), corner_b.data(), size_a, size_b, d2max);
                            n_eval = filter_targets(geom, n_eval, eval_r, r_trg_buf.data(), index_map.data());
                            std::memset(pot_buf.data(), 0, n_eval * kernel_output_dim_trg * sizeof(T));
                            eval_r = r_trg_buf.data();
                            eval_pot = pot_buf.data();
                        }
                        if (n_eval > 0) {
                            evaluator_by_level_trg[src_level](rsc, cen, d2max, 1e-30, n_src, r_src_ptr, charge_ptr,
                                                              n_eval, eval_r, eval_pot);
                            if (trg_larger)
                                scatter_add_potential(pot_buf.data(), &pot_trg_direct[pot_trg_offsets[trg_box]],
                                                      index_map.data(), n_eval, kernel_output_dim_trg);
                        }
                    }
                }

                // Self-interaction correction
                if (src_counts_owned[trg_box]) {
                    auto charge = charge_with_halo_view(trg_box);
                    const auto depth = node_mid[trg_box].Depth() + ifpwexp[trg_box];
                    const auto cf = w0[depth];
                    const int n_src_halo = r_src_cnt_with_halo[trg_box];
                    T *pot_base = &pot_src_direct[pot_src_offsets[trg_box]];
                    for (int i_src = 0; i_src < n_src_halo; ++i_src)
                        for (int i = 0; i < kernel_input_dim; ++i)
                            pot_base[i + i_src * kernel_output_dim_src] -= cf * charge(i, i_src);
                }
                queue.done();
                break;
            }
            } // switch
        } // while
    } // omp parallel

    // =================================================================
    // Accumulate direct + expansion potentials
    // =================================================================
    pot_src_sorted += pot_src_direct;
    pot_trg_sorted += pot_trg_direct;

    logger->info("eval() completed");
    double t_end = MY_OMP_GET_WTIME() - t_start;
    logger->info(
        "phases: c2p={:.4f}  zeroing={:.4f} deps={:.4f} queue={:.4f} mpi={:.4f} downward_seed={:.4f} total={:.4f}",
        t_c2p.load(), t_zeroing.load(), t_deps.load(), t_queue.load(), t_mpi.load(), t_downward_seeded.load(), t_end);

    if (debug_dump_tree)
        dump();
}

template void DMKPtTree<float, 2>::eval_pq();
template void DMKPtTree<float, 3>::eval_pq();
template void DMKPtTree<double, 2>::eval_pq();
template void DMKPtTree<double, 3>::eval_pq();

} // namespace dmk
