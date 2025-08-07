/* More efficient implementation of parallel 2D FFT with Scatter and Gather. 
Pre-allocate memory - still inefficient; vector of vectors and global transposes. 
Upgraded by fft_parallel_2D.cpp and fft_parallel_2D_factored.cpp */

#include <complex>
#include <ducc0/fft/fft.h>
#include <ducc0/fft/fftnd_impl.h>
#include <iostream>
#include <span>
#include <mpi.h>
#include <omp.h>
#include <random>
#include <vector>

template <typename Real>
Real compute_L2norm_diff(const std::vector<std::complex<Real>> &vec1, const std::vector<std::complex<Real>> &vec2) {
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

template <typename Real>
void run_fft(std::span<const std::complex<Real>> &in, std::span<std::complex<Real>> &out, int n_dim, int N, bool is_forward) {
    std::vector<size_t> shape, axes;
    size_t N_tot = 1;
    for (auto i = 0; i < n_dim; ++i) {
        shape.push_back(N);
        axes.push_back(i);
        N_tot *= N;
    }
    
    ducc0::cfmav<std::complex<Real>> ducc_in(in.data(), shape);
    ducc0::vfmav<std::complex<Real>> ducc_out(out.data(), shape);

    // size_t n_threads = omp_get_num_threads();
    size_t n_threads = 1;
    ducc0::c2c(ducc_in, ducc_out, axes, is_forward, Real{1}, n_threads);
}

/* function overloading; here I am NOT using span */
template <typename Real>
void run_fft(const std::vector<std::complex<Real>> &in, std::vector<std::complex<Real>> &out, int n_dim, int N, bool is_forward) {
    std::vector<size_t> shape, axes;
    size_t N_tot = 1;
    for (auto i = 0; i < n_dim; ++i) {
        shape.push_back(N);
        axes.push_back(i);
        N_tot *= N;
    }

    ducc0::cfmav<std::complex<Real>> ducc_in(in.data(), shape);
    ducc0::vfmav<std::complex<Real>> ducc_out(out.data(), shape);

    // size_t n_threads = omp_get_num_threads();
    size_t n_threads = 1;
    ducc0::c2c(ducc_in, ducc_out, axes, is_forward, Real{1}, n_threads);
}

template <typename T>
void print_vector(const std::vector<T> &v) {
    int size = v.size();
    for (int i = 0; i < size; ++i) {
        // if (std::abs(v[i]) < 0.00000000000001) { std::cout << 0 << " "; } else { std::cout << v[i] << " "; }
        std::cout << v[i] << " ";
    }
    std::cout << "\n\n";
}

template <typename T>
void print_vector_comp(const std::vector<std::complex<T>> &v) {
    int size = v.size();
    for (int i = 0; i < size; ++i) {
        std::cout << "(" << v[i].real() << ", " << v[i].imag() << ") ";
    }
    std::cout << "\n\n";
}

/* overloading for the case "vector of vectors" */
template <typename T>
void print_vector_comp(const std::vector<std::vector<std::complex<T>>> &v) {
    int size1 = v.size();
    int size2 = v[0].size();
    for (int i = 0; i < size1; ++i) {
        for (int j = 0; j < size2; ++j) {
            std::cout << "(" << v[i][j].real() << ", " << v[i][j].imag() << ") ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    int N;               // number of points
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    N = std::stoi(argv[1]);

    // declare vector (2D, for now)
    std::vector<std::complex<float>> x_comp(N * N, std::complex<float>(0.0, 0.0));

    // initialize the vector on rank 0 with uniform [0, 1] random values
    if (rank == 0) {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(0.0, 1.0);

        for (size_t i = 0; i < N * N; ++i) {
            x_comp[i].real(distribution(generator));
        }

        /* OPTION 1: Direct computation */
        
        auto start1 = omp_get_wtime();

        // compute the complete FT serially, on rank 0, for reference
        std::vector<std::complex<float>> ft_x(N * N);
        run_fft(x_comp, ft_x, 2, N, true);
        
        auto end1 = omp_get_wtime();
        std::cout << "Option 1 (Direct Computation). Time: " << (end1 - start1) * 1000.0 << " miliseconds." << std::endl;

        /* OPTION 2: Split computation, single core */

        // auto start2 = omp_get_wtime();

        // // Step 1: compute the FT of each row
        // std::vector<std::complex<float>> ft_x_step1(N * N);
        // for (size_t j = 0; j < N; ++j) {
        //     std::span<const std::complex<float>> x_comp_part(&(x_comp[j * N]), N);
        //     std::span<std::complex<float>> ft_x_step1_part(&(ft_x_step1[j * N]), N);
        //     run_fft(x_comp_part, ft_x_step1_part, 1, N, true);
        // }

        // // Step 1.5: transpose
        // std::vector<std::complex<float>> ft_x_step1_row(N);
        // std::vector<std::vector<std::complex<float>>> ft_x_step1_T(N, ft_x_step1_row);
        // for (size_t j = 0; j < N; ++j) {
        //     for (size_t i = 0; i < N; ++i) {
        //         ft_x_step1_T[i][j] = ft_x_step1[j][i];
        //     }
        // }

        // // Step 2: compute the FT of each column
        // std::vector<std::complex<float>> ft_x_step2(N * N);
        // for (size_t j = 0; j < N; ++j) {
        //     std::span<const std::complex<float>> ft_x_step1_T_part(&(ft_x_step1_T[j][0]), N);
        //     std::span<std::complex<float>> ft_x_step2_part(&(ft_x_step2[j][0]), N);
        //     ft_x_step2[j] = run_fft(ft_x_step1_T_part, 1, N, true);
        // }

        // // Step 2.5: transpose
        // std::vector<std::complex<float>> ft_x_step2_T(N * N);
        // for (size_t j = 0; j < N; ++j) {
        //     for (size_t i = 0; i < N; ++i) {
        //         ft_x_step2_T[i + N * j] = ft_x_step2[i][j];
        //     }
        // }

        // auto end2 = omp_get_wtime();
        // std::cout << "Option 2 (Split Computation, Single Core). Time: " << (end2 - start2) * 1000.0 << " miliseconds." << std::endl;
    }

    // Wait for all cores to synchronize
    MPI_Barrier(MPI_COMM_WORLD);

    /* OPTION 3: Parallel computation */

    // scatter the input to other ranks
    std::vector<int> scounts(size, N / size * N);         // how many elements per "packet"
    std::vector<int> rows(size, N / size);                // how many rows/slabs per "packet"
    std::vector<int> displs(size);                        // starting index of each new "packet"

    {
        int ind = 0;
        for (int i = N % size; i > 0; --i) {
            ++rows[ind];
            scounts[ind] += N;
            ++ind;
        }
    }

    for (int i = 1; i < size; ++i) {
        displs[i] = scounts[i - 1] + displs[i - 1];
    }
    
    std::vector<std::complex<float>> x_comp_small(scounts[rank]);
    std::vector<std::complex<float>> ft_x_step1_small(scounts[rank]);
    std::vector<std::complex<float>> ft_x_step1;            // declare the vector everywhere
    std::vector<std::complex<float>> ft_x_step1_T;          // again, no memory allocation
    std::vector<std::complex<float>> ft_x_step1_T_small(scounts[rank]);
    std::vector<std::complex<float>> ft_x_step2_small(scounts[rank]);
    std::vector<std::complex<float>> ft_x_step2;
    std::vector<std::complex<float>> ft_x_step2_T;

    if (rank==0) {
        ft_x_step1.assign(N * N, 0.0);                      // but allocate space only in root!
        ft_x_step1_T.assign(N * N, 0.0);
        ft_x_step2.assign(N * N, 0.0);
        ft_x_step2_T.assign(N * N, 0.0);
    }

    auto start3 = omp_get_wtime();

    auto begin_scatter_1 = omp_get_wtime();
    MPI_Scatterv(x_comp.data(), scounts.data(), displs.data(), MPI_COMPLEX, x_comp_small.data(), scounts[rank], MPI_COMPLEX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_scatter_1 = omp_get_wtime();

    auto begin_ft_1 = omp_get_wtime();
    // take the FT (1)
    for (size_t i = 0; i < rows[rank]; ++i) {
        std::span<const std::complex<float>> x_comp_row(&(x_comp_small[i * N]), N);
        std::span<std::complex<float>> ft_x_step1_small_row(&(ft_x_step1_small[i * N]), N);
        run_fft(x_comp_row, ft_x_step1_small_row, 1, N, true);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_ft_1 = omp_get_wtime();

    auto begin_gather_1 = omp_get_wtime();
    // gather all ft_x_step1_small
    MPI_Gatherv(ft_x_step1_small.data(), scounts[rank], MPI_COMPLEX, ft_x_step1.data(), scounts.data(), displs.data(), MPI_COMPLEX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_gather_1 = omp_get_wtime();

    auto begin_transpose_1 = omp_get_wtime();
    // transpose ft_x_step1
    if (rank==0) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                ft_x_step1_T[i + N * j] = ft_x_step1[j + N * i];
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_transpose_1 = omp_get_wtime();

    auto begin_scatter_2 = omp_get_wtime();
    // scatter ft_x_step1 to other cores
    MPI_Scatterv(ft_x_step1_T.data(), scounts.data(), displs.data(), MPI_COMPLEX, ft_x_step1_T_small.data(), scounts[rank], MPI_COMPLEX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_scatter_2 = omp_get_wtime();

    auto begin_ft_2 = omp_get_wtime();
    // take the FT (2)
    for (size_t i = 0; i < rows[rank]; ++i) {
        std::span<const std::complex<float>> ft_x_step1_T_small_row(&(ft_x_step1_T_small[i * N]), N);
        std::span<std::complex<float>> ft_x_step2_small_row(&(ft_x_step2_small[i * N]), N);
        run_fft(ft_x_step1_T_small_row, ft_x_step2_small_row, 1, N, true);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_ft_2 = omp_get_wtime();

    auto begin_gather_2 = omp_get_wtime();
    // gather all ft_x_step2_small_flat into ft_x_step2
    MPI_Gatherv(ft_x_step2_small.data(), scounts[rank], MPI_COMPLEX, ft_x_step2.data(), scounts.data(), displs.data(), MPI_COMPLEX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_gather_2 = omp_get_wtime();

    auto begin_transpose_2 = omp_get_wtime();
    // transpose ft_x_step2    
    if (rank==0) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                ft_x_step2_T[i + N * j] = ft_x_step2[j + N * i];
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_transpose_2 = omp_get_wtime();

    // Wait for all cores to synchronize
    // MPI_Barrier(MPI_COMM_WORLD);

    if (rank==0) {
        auto end3 = omp_get_wtime();
        std::cout << "Option 3 (Split Computation, Multiple Cores). Time: " << (end3 - start3) * 1000.0 << " miliseconds." << std::endl;
        
        // COMPARE (run Option 1 again because the value isn't saved from earlier)
        std::vector<std::complex<float>> ft_x(N * N);
        run_fft(x_comp, ft_x, 2, N, true);
        float l2norm = compute_L2norm_diff(ft_x, ft_x_step2_T);
        std::cout << "The L2 norm of the two vectors is: " << l2norm <<std::endl;
    }

    if (rank==0) {
        std::cout << "Scatter1: " << (end_scatter_1 - begin_scatter_1) * 1000 << " ms";
        std::cout << ", FT1: " << (end_ft_1 - begin_ft_1) * 1000 << " ms";
        std::cout << ", Gather1: " << (end_gather_1 - begin_gather_1) * 1000 << " ms";
        std::cout << ", Transpose1: " << (end_transpose_1 - begin_transpose_1) * 1000 << " ms";
        std::cout << std::endl;
        std::cout << "Scatter2: " << (end_scatter_2 - begin_scatter_2) * 1000 << " ms";
        std::cout << ", FT2: " << (end_ft_2 - begin_ft_2) * 1000 << " ms";
        std::cout << ", Gather2: " << (end_gather_2 - begin_gather_2) * 1000 << " ms";
        std::cout << ", Transpose2: " << (end_transpose_2 - begin_transpose_2) * 1000 << " ms";
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}