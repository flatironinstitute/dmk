/* 
Final parallel 3D FFT implementation using slabs. First we compute 2D FFT for each slab,
then we transpose the matrix (using Alltoallv) and then we take 1D FFT's. 
    
In literature, this is called "transpose-based FFT". You can check the following reference:
https://doi.org/10.1007/978-3-540-89894-8_32 

For more modern (and possibly faster) implementations, check:
https://doi.org/10.1016/j.jpdc.2019.02.006 
https://doi.org/10.48550/arXiv.1905.02803 
*/

#include <complex>
#include <ducc0/fft/fft.h>
#include <ducc0/fft/fftnd_impl.h>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <random>
#include <span>
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

// 1D/ND FFT wrapper using DUCC0
template <typename Real>
void run_fft(
    std::span<const std::complex<Real>>& in,
    std::span<std::complex<Real>>&       out,
    int                                   n_dim,
    int                                   N,
    bool                                  is_forward
) {
    std::vector<size_t> shape, axes;
    for (int d = 0; d < n_dim; ++d) {
        shape.push_back(N);
        axes.push_back(d);
    }
    ducc0::cfmav<std::complex<Real>> ducc_in(in.data(), shape);
    ducc0::vfmav<std::complex<Real>> ducc_out(out.data(), shape);
    size_t n_threads = 1;
    ducc0::c2c(ducc_in, ducc_out, axes, is_forward, Real{1}, n_threads);
}

// Overload: vector input without span
template <typename Real>
void run_fft(
    const std::vector<std::complex<Real>>& in,
    std::vector<std::complex<Real>>&       out,
    int                                     n_dim,
    int                                     N,
    bool                                    is_forward
) {
    std::vector<size_t> shape, axes;
    for (int d = 0; d < n_dim; ++d) {
        shape.push_back(N);
        axes.push_back(d);
    }
    ducc0::cfmav<std::complex<Real>> ducc_in(in.data(), shape);
    ducc0::vfmav<std::complex<Real>> ducc_out(out.data(), shape);
    size_t n_threads = 1;
    ducc0::c2c(ducc_in, ducc_out, axes, is_forward, Real{1}, n_threads);
}

template <typename Real>
void compute_distributed_fft(
        std::vector<std::complex<Real>> &real_matrix,
        std::vector<std::complex<Real>> &transformed_matrix,
        int ndim,
        int nplanes,
        int N,
        int M,
        bool is_forward
    ) {
    for (int i = 0; i < nplanes; ++i) {
        std::span<const std::complex<Real>> in_col(real_matrix.data() + i * M, M);
        std::span<std::complex<Real>>       out_col(transformed_matrix.data() + i * M, M);
        run_fft(in_col, out_col, ndim, N, is_forward);
    }
}

// Print helpers
template <typename T>
void print_vector(const std::vector<T>& v) {
    for (int i = 0; i < (int)v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << "\n\n";
}

template <typename T>
void print_vector_comp(const std::vector<std::complex<T>>& v) {
    for (int i = 0; i < (int)v.size(); ++i) {
        std::cout << "(" << v[i].real() << ", " << v[i].imag() << ") ";
    }
    std::cout << "\n\n";
}

template <typename T>
void print_vector_comp(const std::vector<std::vector<std::complex<T>>>& v) {
    for (int i = 0; i < (int)v.size(); ++i) {
        for (int j = 0; j < (int)v[i].size(); ++j) {
            std::cout << "(" << v[i][j].real() << ", " << v[i][j].imag() << ") ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = std::stoi(argv[1]);
    std::vector<std::complex<float>> x(N * N * N, std::complex<float>(0.0, 0.0));

    if (rank == 0) {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (size_t i = 0; i < N * N * N; ++i) {
            x[i].real(distribution(generator));
        }
        auto start1 = omp_get_wtime();
        std::vector<std::complex<float>> ft_x(N * N * N);
        run_fft(x, ft_x, 3, N, true);
        auto end1 = omp_get_wtime();
        std::cout << "Option 1 (Direct Computation). Time: "
                  << (end1 - start1) * 1000.0 << " miliseconds." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

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
    std::vector<std::complex<float>> x_slab(scounts[rank]);
    std::vector<std::complex<float>> ft1_slab(scounts[rank]);
    std::vector<std::complex<float>> ft1_aa(scounts[rank]);
    std::vector<std::complex<float>> ft1_reordered(scounts[rank]);
    std::vector<std::complex<float>> ft1_T(scounts[rank]);
    std::vector<std::complex<float>> ft2_slab(scounts[rank]);

    // Scatter planes
    auto begin_scatter_1 = omp_get_wtime();
    MPI_Scatterv(x.data(), scounts.data(), displs.data(), MPI_COMPLEX,
                 x_slab.data(), scounts[rank], MPI_COMPLEX,
                 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_scatter_1 = omp_get_wtime();

    // -------------------------------------------------------------------- //

    auto start2 = omp_get_wtime();

    // FFT on row slabs
    auto begin_ft_1 = omp_get_wtime();
    compute_distributed_fft(x_slab, ft1_slab, 2, planes[rank], N, N * N, true);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_ft_1 = omp_get_wtime();

    // TRANSPOSE 1
    auto begin_transpose_1 = omp_get_wtime();
    transpose_distributed_matrix(
        &(ft1_slab[0]), 
        &(ft1_T[0]),
        N,
        &(ft1_reordered[0]),
        &(ft1_aa[0]),
        &(planes[0]),
        &(skip_cols[0]),
        &(sendcounts[0]),
        &(displacements[0]),
        MPI_COMM_WORLD
    );
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_transpose_1 = omp_get_wtime();

    // FFT on columns
    auto begin_ft_2 = omp_get_wtime();
    compute_distributed_fft(ft1_T, ft2_slab, 1, planes[rank] * N, N, N, true);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_ft_2 = omp_get_wtime();

    // TRANSPOSE 2 (OPTIONAL!)
    auto begin_transpose_2 = omp_get_wtime();
    transpose_distributed_matrix(
        &(ft2_slab[0]), 
        &(x_slab[0]),
        N,
        &(ft1_reordered[0]),
        &(ft1_aa[0]),
        &(planes[0]),
        &(skip_cols[0]),
        &(sendcounts[0]),
        &(displacements[0]),
        MPI_COMM_WORLD
    );
    // transpose_distributed_matrix(
    //     &(x_slab[0]),
    //     &(ft2_slab[0]),
    //     N,
    //     &(ft1_reordered[0]),
    //     &(ft1_aa[0]),
    //     &(planes[0]),
    //     &(skip_cols[0]),
    //     &(sendcounts[0]),
    //     &(displacements[0]),
    //     MPI_COMM_WORLD
    // );
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_transpose_2 = omp_get_wtime();

    // -------------------------------------------------------------------- //

    if (rank == 0) {
        auto end2 = omp_get_wtime();
        std::cout << "Option 2 (Split Computation, Multiple Cores). Time: "
                  << (end2 - start2) * 1000.0 << " miliseconds." << std::endl;
        std::cout << "Scatter1: "     << (end_scatter_1 - begin_scatter_1) * 1000 << " ms, "
                  << "FT1: "          << (end_ft_1 - begin_ft_1)         * 1000 << " ms, "
                  << "Tranpose1: "    << (end_transpose_1 - begin_transpose_1)     * 1000 << " ms, "
                  << std::endl;
        std::cout << "FT2: "          << (end_ft_2 - begin_ft_2)         * 1000 << " ms, "
                  << "Transpose2: "   << (end_transpose_2 - begin_transpose_2) * 1000 << " ms, "
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
