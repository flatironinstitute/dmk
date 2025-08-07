/* Transpose of a 2D matrix using Alltoallv() */

#include <iostream>
#include <complex>
#include <mpi.h>
#include <omp.h>
#include <vector>


// Transpose a matrix
template <typename T>
void compute_local_transpose (
        T *x, 
        T *y, 
        int N, 
        int M
    ) {
    
    // initial matrix size: N x M
    // transpose matrix size: M x N

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            y[i + j * N] = x[i * M + j];
        }
    }
}

template <class...>
constexpr std::false_type always_false{};

template <typename T>
inline MPI_Datatype get_mpi_type() {
    if constexpr (std::is_same_v<float, T>)
        return MPI_FLOAT;
    else if constexpr (std::is_same_v<double, T>)    
        return MPI_DOUBLE;
    else if constexpr (std::is_same_v<std::complex<float>, T>)
        return MPI_C_FLOAT_COMPLEX;
    else if constexpr (std::is_same_v<std::complex<double>, T>)
        return MPI_C_DOUBLE_COMPLEX;
    else if constexpr (std::is_same_v<int, T>)
        return MPI_INT;
    else
        static_assert(always_false<T>, "Unsupported type for MPI datatype conversion");
}

template <typename T>
void transpose_distributed_matrix(
        T *local_matrix, 
        T *local_transposed,
        int N,
        T *mat_reordered,
        T *mat_aa,
        int *rows,
        int *skip_cols,
        int *sendcounts,
        int *displacements,
        MPI_Comm comm
    ) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    const auto mpi_type = get_mpi_type<T>();
    
    // Local reorder for A2A
    int running_index = 0;
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < rows[rank]; ++i) {
            for (int k = 0; k < rows[j]; ++k) {
                mat_reordered[running_index++] = local_matrix[i * N + skip_cols[j] + k];
            }
        }
    }

    // All-to-all exchange
    MPI_Alltoallv(mat_reordered, sendcounts, displacements, mpi_type,
                  mat_aa,        sendcounts, displacements, mpi_type,
                  comm);

    // Local transpose
    compute_local_transpose(mat_aa, local_transposed, N, rows[rank]);
}

// Print helpers
template <typename T>
void print_vector(const std::vector<T>& v) {
    for (int i = 0; i < (int)v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << "\n\n";
}

// -------------------------------------------------------------------------- //

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = std::stoi(argv[1]);
    std::vector<int> x(N * N);

    if (rank == 0) {
        for (size_t i = 0; i < N * N; ++i) {
            x[i] = i;
        }
    }

    // Slab decomposition
    std::vector<int> scounts(size, N / size * N);
    std::vector<int> rows(size, N / size);
    for (int i = N % size, idx = 0; i > 0; --i, ++idx) {
        rows[idx]++;
        scounts[idx] += N;
    }
    std::vector<int> displs(size), sendcounts(size), displacements(size), skip_cols(size);
    sendcounts[0] = rows[0] * rows[rank];
    for (int i = 1; i < size; ++i) {
        displs[i]        = displs[i - 1] + scounts[i - 1];
        sendcounts[i]    = rows[i] * rows[rank];
        displacements[i] = displacements[i - 1] + sendcounts[i - 1];
        skip_cols[i]     = skip_cols[i - 1] + rows[i - 1];
    }

    // Local buffers
    std::vector<int> x_slab(scounts[rank]);
    std::vector<int> x_reordered(scounts[rank]);
    std::vector<int> x_aa(scounts[rank]);
    std::vector<int> x_aa_T(scounts[rank]);

    // Scatter rows
    MPI_Scatterv(x.data(),      scounts.data(), displs.data(), MPI_INT,
                 x_slab.data(), scounts[rank],                 MPI_INT,
                 0, MPI_COMM_WORLD);

    // -------------------------------------------------------------------- //
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = omp_get_wtime();
    transpose_distributed_matrix(
        &(x_slab[0]), 
        &(x_aa_T[0]),
        N,
        &(x_reordered[0]),
        &(x_aa[0]),
        &(rows[0]),
        &(skip_cols[0]),
        &(sendcounts[0]),
        &(displacements[0]),
        MPI_COMM_WORLD
    );
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = omp_get_wtime();

    // print_vector(x_aa_T);

    std::cout << (end - start) * 1000 << " " << rank << std::endl;

    // -------------------------------------------------------------------- //

    MPI_Finalize();
    return 0;
}
