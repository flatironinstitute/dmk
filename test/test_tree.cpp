#include <ctime>
#include <dmk/tree.hpp>
#include <mpi.h>
#include <sctl.hpp>

void test_tree() {
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

    tree.generate_metadata(1, 1);

    const auto &mids = tree.GetNodeMID();
    const auto &attrs = tree.GetNodeAttr();
    for (std::size_t i_node = 0; i_node < mids.Dim(); ++i_node) {
        usleep(1000 * comm.Rank());

        std::cout << comm.Rank() << "\t" << i_node << "\t";
        std::cout << tree.src_counts_local[i_node] << "\t" << tree.src_counts_global[i_node] << "\t";
        std::cout << int(attrs[i_node].Leaf) << "\t" << int(attrs[i_node].Ghost) << "\n";

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[]) {
    sctl::Comm::MPI_Init(&argc, &argv);
    test_tree();
    sctl::Comm::MPI_Finalize();

    return 0;
}
