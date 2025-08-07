/* Faster 2D FFT implementation with AlltoAll() collective operations. Not factored.
Upgraded by fft_parallel_2D_factored.cpp */

#include <complex>
#include <ducc0/fft/fft.h>
#include <ducc0/fft/fftnd_impl.h>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <random>
#include <span>
#include <vector>

// Compute normalized L2 difference
template <typename Real>
Real compute_L2norm_diff(
    const std::vector<std::complex<Real>>& vec1,
    const std::vector<std::complex<Real>>& vec2
) {
    int size1 = vec1.size();
    int size2 = vec2.size();
    if (size1 != size2) {
        std::cerr << "The two vectors have different sizes!" << std::endl;
        exit(EXIT_FAILURE);
    }
    Real total = 0.0;
    for (size_t i = 0; i < size1; ++i) {
        Real diff_real = vec1[i].real() - vec2[i].real();
        Real diff_imag = vec1[i].imag() - vec2[i].imag();
        total += diff_real * diff_real + diff_imag * diff_imag;
    }
    return std::sqrt(total) / size1;
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

// Transpose a matrix
template <typename Real>
void compute_transpose (
        const std::vector<std::complex<Real>> &x, 
        std::vector<std::complex<Real>> &y, 
        int N, 
        int M
    ) {
    
    // initial matrix size: N x M

    int size1 = x.size();               // linear size of vector x (matrix)
    int size2 = y.size();               // linear size of vector y (transpose)

    if (size1 != size2) {
        std::cerr << "The two vectors have different sizes!" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (size1 != N * M) {
        std::cerr << "Wrong input dimension!" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            y[i + j * N] = x[i * M + j];
        }
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
    std::vector<std::complex<float>> x(N * N, std::complex<float>(0.0, 0.0));

    if (rank == 0) {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (size_t i = 0; i < N * N; ++i) {
            x[i].real(distribution(generator));
        }
        auto start1 = omp_get_wtime();
        std::vector<std::complex<float>> ft_x(N * N);
        run_fft(x, ft_x, 2, N, true);
        auto end1 = omp_get_wtime();
        std::cout << "Option 1 (Direct Computation). Time: "
                  << (end1 - start1) * 1000.0 << " miliseconds." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

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
    std::vector<std::complex<float>> x_slab(scounts[rank]);
    std::vector<std::complex<float>> ft1_slab(scounts[rank]);
    std::vector<std::complex<float>> ft1_aa(scounts[rank]);
    std::vector<std::complex<float>> ft1_reordered(scounts[rank]);
    std::vector<std::complex<float>> ft1_T(scounts[rank]);
    std::vector<std::complex<float>> ft2_slab(scounts[rank]);

    // Scatter rows
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
    for (int i = 0; i < rows[rank]; ++i) {
        std::span<const std::complex<float>> in_row(x_slab.data() + i * N, N);
        std::span<std::complex<float>>       out_row(ft1_slab.data() + i * N, N);
        run_fft(in_row, out_row, 1, N, true);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_ft_1 = omp_get_wtime();

    // Local reorder for A2A
    auto begin_reorder = omp_get_wtime();
    int running_index = 0;
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < rows[rank]; ++i) {
            for (int k = 0; k < rows[j]; ++k) {
                ft1_reordered[running_index++] = ft1_slab[i * N + skip_cols[j] + k];
            }
        }
    }
    auto end_reorder = omp_get_wtime();

    // All-to-all exchange
    auto begin_alltoallv = omp_get_wtime();
    MPI_Alltoallv(ft1_reordered.data(), sendcounts.data(), displacements.data(), MPI_COMPLEX,
                  ft1_aa.data(),         sendcounts.data(), displacements.data(), MPI_COMPLEX,
                  MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_alltoallv = omp_get_wtime();

    // Local transpose
    auto begin_transpose = omp_get_wtime();
    compute_transpose(ft1_aa, ft1_T, N, rows[rank]);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_transpose = omp_get_wtime();

    // FFT on columns
    auto begin_ft_2 = omp_get_wtime();
    for (int i = 0; i < rows[rank]; ++i) {
        std::span<const std::complex<float>> in_col(ft1_T.data() + i * N, N);
        std::span<std::complex<float>>       out_col(ft2_slab.data() + i * N, N);
        run_fft(in_col, out_col, 1, N, true);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_ft_2 = omp_get_wtime();

    // Inverse local transpose
    auto begin_transpose_2 = omp_get_wtime();
    compute_transpose(ft2_slab, ft1_reordered, rows[rank], N);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_transpose_2 = omp_get_wtime();

    // Inverse all-to-all exchange
    auto begin_alltoallv_2 = omp_get_wtime();
    MPI_Alltoallv(
        ft1_reordered.data(), sendcounts.data(), displacements.data(), MPI_COMPLEX,
        ft1_aa.data(),        sendcounts.data(), displacements.data(), MPI_COMPLEX,
        MPI_COMM_WORLD
    );
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_alltoallv_2 = omp_get_wtime();

    // Inverse local reorder
    auto begin_reorder_2 = omp_get_wtime();
    running_index = 0;
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < rows[rank]; ++i) {
            for (int k = 0; k < rows[j]; ++k) {
                ft1_T[i * N + skip_cols[j] + k] = ft1_aa[running_index++];
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_reorder_2 = omp_get_wtime();

    // -------------------------------------------------------------------- //

    if (rank == 0) {
        auto end2 = omp_get_wtime();
        std::cout << "Option 2 (Split Computation, Multiple Cores). Time: "
                  << (end2 - start2) * 1000.0 << " miliseconds." << std::endl;
        std::cout << "Scatter1: "      << (end_scatter_1 - begin_scatter_1) * 1000 << " ms, "
                  << "FT1: "          << (end_ft_1 - begin_ft_1)         * 1000 << " ms, "
                  << "Reorder1: "      << (end_reorder - begin_reorder)     * 1000 << " ms, "
                  << "A2A1: "         << (end_alltoallv - begin_alltoallv) * 1000 << " ms" << std::endl;
        std::cout << "Transpose1: "   << (end_transpose - begin_transpose)   * 1000 << " ms, "
                  << "FT2: "          << (end_ft_2 - begin_ft_2)         * 1000 << " ms, "
                  << "Transpose2: "   << (end_transpose_2 - begin_transpose_2) * 1000 << " ms, "
                  << "A2A2: "         << (end_alltoallv_2 - begin_alltoallv_2)   * 1000 << " ms, "
                  << "Reorder2: "     << (end_reorder_2 - begin_reorder_2)     * 1000 << " ms" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
