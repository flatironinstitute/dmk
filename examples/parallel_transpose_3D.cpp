/* Transpose of a 3D matrix using Alltoallv() */

#include <iostream>
#include <complex>
#include <mpi.h>
#include <omp.h>
#include <vector>

// Print helpers
template <typename T>
void print_vector(const std::vector<T>& v) {
    for (int i = 0; i < (int)v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << "\n\n";
}

template <typename T>
void print_vector(T *mat, int M) {
    for (int i = 0; i < M; ++i) {
        std::cout << mat[i] << " ";
    }
    std::cout << "\n\n";
}

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
        int *planes,
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
    for (int core = 0; core < size; ++core){
        for (int slab = 0; slab < planes[rank]; ++slab) {
            for (int j = 0; j < planes[core] * N; ++j) {
                mat_reordered[running_index++] = local_matrix[skip_cols[core] + slab * N * N + j];
            }
        }
    }

    // All-to-all exchange
    MPI_Alltoallv(mat_reordered, sendcounts, displacements, mpi_type,
                  mat_aa,        sendcounts, displacements, mpi_type,
                  comm);

    // Local transpose
    compute_local_transpose(mat_aa, local_transposed, N, planes[rank] * N);
}

// -------------------------------------------------------------------------- //

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = std::stoi(argv[1]);
    std::vector<int> x(N * N * N);

    if (rank == 0) {
        for (size_t i = 0; i < N * N * N; ++i) {
            x[i] = i;
        }
    }

    // Slab decomposition
    std::vector<int> planes(size, N / size);
    std::vector<int> scounts(size, N / size * N * N);
    for (int i = N % size, idx = 0; i > 0; --i, ++idx) {
        planes[idx]++;
        scounts[idx] += N * N;
    }
    std::vector<int> displs(size), sendcounts(size), displacements(size), skip_cols(size);
    sendcounts[0] = planes[0] * planes[rank] * N;
    for (int i = 1; i < size; ++i) {
        displs[i]        = displs[i - 1] + scounts[i - 1];
        sendcounts[i]    = planes[i] * planes[rank] * N;
        displacements[i] = displacements[i - 1] + sendcounts[i - 1];
        skip_cols[i]     = skip_cols[i - 1] + planes[i - 1] * N;
    }

    // Local buffers
    std::vector<int> x_slab(scounts[rank]);
    std::vector<int> x_reordered(scounts[rank]);
    std::vector<int> x_aa(scounts[rank]);
    std::vector<int> x_aa_T(scounts[rank]);

    // Scatter planes
    MPI_Scatterv(x.data(),      scounts.data(), displs.data(), MPI_INT,
                 x_slab.data(), scounts[rank],                 MPI_INT,
                 0, MPI_COMM_WORLD);

    // -------------------------------------------------------------------- //

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = omp_get_wtime();
    for (int i = 0; i < 10000; ++i) {   
    transpose_distributed_matrix(
        &(x_slab[0]), 
        &(x_aa_T[0]),
        N,
        &(x_reordered[0]),
        &(x_aa[0]),
        &(planes[0]),
        &(skip_cols[0]),
        &(sendcounts[0]),
        &(displacements[0]),
        MPI_COMM_WORLD
    );
    MPI_Barrier(MPI_COMM_WORLD);}
    auto end = omp_get_wtime();

    if (rank==0)
    std::cout << (end - start) * 1000 / 10000 << std::endl;

    // -------------------------------------------------------------------- //

    MPI_Finalize();
    return 0;
}
