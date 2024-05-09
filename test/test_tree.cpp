#include <dmk/tree.hpp>
#include <mpi.h>
#include <sctl.hpp>

int main(int argc, char *argv[]) {
    sctl::Comm::MPI_Init(&argc, &argv);

    {
        constexpr int DIM = 2;
        auto comm = sctl::Comm::World();
        int long N = comm.Rank() == 0 ? 10 : 0;
        dmk::DMKPtTree<double, DIM> tree(comm);

        sctl::Vector<double> X(N * DIM), f(N);
        if (!comm.Rank()) {
            for (int i = 0; i < N; ++i)
                f[i] = i;
            std::tie(X[0], X[1]) = std::make_pair(4., 12.);
            std::tie(X[2], X[3]) = std::make_pair(9., 15.);
            std::tie(X[4], X[5]) = std::make_pair(11., 15.);
            std::tie(X[6], X[7]) = std::make_pair(9., 13.);
            std::tie(X[8], X[9]) = std::make_pair(11., 13.);
            std::tie(X[10], X[11]) = std::make_pair(4., 4.);
            std::tie(X[12], X[13]) = std::make_pair(12., 4.);
            std::tie(X[14], X[15]) = std::make_pair(10., 10.);
            std::tie(X[16], X[17]) = std::make_pair(14., 10.);
            std::tie(X[18], X[19]) = std::make_pair(14., 14.);
            X /= 16.0;
        }
        tree.AddParticles("pdmk_src", X);
        tree.AddParticleData("pdmk_charge", "pdmk_src", f);
        tree.UpdateRefinement(X, 1, true, false);
        tree.WriteTreeVTK("tree");
        tree.WriteParticleVTK("str", "pdmk_charge");

        const auto &mids = tree.GetNodeMID();
        const auto &attrs = tree.GetNodeAttr();
        sctl::Vector<double> data_part;
        sctl::Vector<sctl::Long> cnt_part;
        tree.GetData(data_part, cnt_part, "pdmk_src");

        sctl::Vector<double> data_charge;
        sctl::Vector<sctl::Long> cnt_charge;
        tree.GetData(data_charge, cnt_charge, "pdmk_src");

        tree.generate_metadata(1, 1);

        sctl::Long offset_part{0}, offset_charge{0};
        for (std::size_t i_node = 0; i_node < mids.Dim(); ++i_node) {
            if (true || tree.in_flag[i_node]) {
                std::cout << comm.Rank() << " " << i_node << " ";
                std::cout << "(";
                for (int i = 0; i < cnt_part[i_node]; ++i) {
                    std::cout << data_part[i + offset_part] << "," << data_part[i + 1 + offset_part];
                    if (i < cnt_part[i_node] - 1)
                        std::cout << ",";
                }
                std::cout << ") ";

                // std::cout << "(";
                // for (int i = 0; i < cnt_charge[i_node]; ++i) {
                //     std::cout << data_charge[i + offset_charge];
                //     if (i < cnt_charge[i_node] - 1)
                //         std::cout << ",";
                // }
                // std::cout << ") ";

                std::cout << "(" << tree.leaf_flag[i_node] << "," << tree.out_flag[i_node] << ","
                          << tree.in_flag[i_node] << ") ";

                std::cout << tree.src_counts_local[i_node] << " " << tree.src_counts_global[i_node] << " ";

                std::cout << mids[i_node] << " " << int(attrs[i_node].Leaf) << " " << int(attrs[i_node].Ghost) << "\n";

                offset_part += 2 * cnt_part[i_node];
                offset_charge += cnt_charge[i_node];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << comm.Rank() << " " << attrs.Dim() << "\n";

        const auto &morton_part = tree.GetPartitionMID();
        for (auto &id : morton_part)
            std::cout << id << std::endl;
        // auto &nodelists = tree.GetNodeLists();
        // int idx = 0;
        // for (auto &el : nodelists) {
        //     std::cout << mids[idx] << " ";
        //     for (auto &child_id : el.child)
        //         if (child_id != -1)
        //             std::cout << mids[child_id] << " ";
        //     std::cout << "\n";
        //     idx++;
        // }
    }
    sctl::Comm::MPI_Finalize();

    return 0;
}
