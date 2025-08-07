/* Parallel implementation of a 2D FFT using Scatter and Gather. Inefficient and slow. 
Upgraded by fft_parallel_v.cpp, fft_parallel_2D.cpp, and fft_parallel_2D_factored.cpp*/

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
std::vector<std::complex<Real>> run_fft(std::span<const std::complex<Real>> &in, int n_dim, int N, bool is_forward) {
    std::vector<size_t> shape, axes;
    size_t N_tot = 1;
    for (auto i = 0; i < n_dim; ++i) {
        shape.push_back(N);
        axes.push_back(i);
        N_tot *= N;
    }

    std::vector<std::complex<Real>> out(N_tot);
    ducc0::cfmav<std::complex<Real>> ducc_in(in.data(), shape);
    ducc0::vfmav<std::complex<Real>> ducc_out(out.data(), shape);

    // size_t n_threads = omp_get_num_threads();
    size_t n_threads = 1;
    ducc0::c2c(ducc_in, ducc_out, axes, is_forward, Real{1}, n_threads);

    return out;
}

/* function overloading; here I am NOT using span */
template <typename Real>
std::vector<std::complex<Real>> run_fft(const std::vector<std::complex<Real>> &in, int n_dim, int N, bool is_forward) {
    std::vector<size_t> shape, axes;
    size_t N_tot = 1;
    for (auto i = 0; i < n_dim; ++i) {
        shape.push_back(N);
        axes.push_back(i);
        N_tot *= N;
    }

    std::vector<std::complex<Real>> out(N_tot);
    ducc0::cfmav<std::complex<Real>> ducc_in(in.data(), shape);
    ducc0::vfmav<std::complex<Real>> ducc_out(out.data(), shape);

    // size_t n_threads = omp_get_num_threads();
    size_t n_threads = 1;
    ducc0::c2c(ducc_in, ducc_out, axes, is_forward, Real{1}, n_threads);

    return out;
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
    // I use the word "rows" in my variables because I started with 2D; for 3D I mean slabs/planes
    int rows_per_rank = N / size;
    int entries_per_rank = rows_per_rank * N;

    // declare vector (2D, for now)
    std::vector<std::complex<float>> x_comp(N * N, std::complex<float>(0.0, 0.0));
    // declare ft_x vector (FT in one core, no split, using ducc) -- used for comparison
    // it is not necessary in the final implementation!
    std::vector<std::complex<float>> ft_x(N * N);

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
        std::vector<std::complex<float>> ft_x = run_fft(x_comp, 2, N, true);
        
        auto end1 = omp_get_wtime();
        std::cout << "Option 1 (Direct Computation). Time: " << (end1 - start1) * 1000.0 << " miliseconds." << std::endl;

        /* OPTION 2: Split computation, single core */

        auto start2 = omp_get_wtime();

        // Step 1: compute the FT of each row
        std::vector<std::vector<std::complex<float>>> ft_x_step1(N);
        for (size_t j = 0; j < N; ++j) {
            std::span<const std::complex<float>> x_comp_part(&(x_comp[j * N]), N);
            ft_x_step1[j] = run_fft(x_comp_part, 1, N, true);
        }

        // Step 1.5: transpose
        std::vector<std::complex<float>> ft_x_step1_row(N);
        std::vector<std::vector<std::complex<float>>> ft_x_step1_T(N, ft_x_step1_row);
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                ft_x_step1_T[i][j] = ft_x_step1[j][i];
            }
        }

        // Step 2: compute the FT of each column
        std::vector<std::vector<std::complex<float>>> ft_x_step2(N);
        for (size_t j = 0; j < N; ++j) {
            std::span<const std::complex<float>> ft_x_step1_T_part(&(ft_x_step1_T[j][0]), N);
            ft_x_step2[j] = run_fft(ft_x_step1_T_part, 1, N, true);
        }

        // Step 2.5: transpose
        std::vector<std::complex<float>> ft_x_step2_T(N * N);
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                ft_x_step2_T[i + N * j] = ft_x_step2[i][j];
            }
        }

        auto end2 = omp_get_wtime();
        std::cout << "Option 2 (Split Computation, Single Core). Time: " << (end2 - start2) * 1000.0 << " miliseconds." << std::endl;
    }

    // Wait for all cores to synchronize
    MPI_Barrier(MPI_COMM_WORLD);

    /* OPTION 3: Parallel computation */

    auto start3 = omp_get_wtime();

    // scatter the input to other ranks
    // QUESTION: What if P doesn't divide N?
    std::vector<std::complex<float>> x_comp_small(entries_per_rank);
    MPI_Scatter(x_comp.data(), entries_per_rank, MPI_COMPLEX, x_comp_small.data(), entries_per_rank, MPI_COMPLEX, 0, MPI_COMM_WORLD);
    
    // take the FT (1)
    // QUESTION: Should I use a for-loop for each slab?
    std::vector<std::vector<std::complex<float>>> ft_x_step1_small(rows_per_rank);
    for (size_t i = 0; i < rows_per_rank; ++i) {
        std::span<const std::complex<float>> x_comp_row(&(x_comp_small[i * N]), N);
        ft_x_step1_small[i] = run_fft(x_comp_row, 1, N, true);
    }

    // flatten the computed FT
    std::vector<std::complex<float>> ft_x_step1_small_flat(entries_per_rank);
    for (size_t i = 0; i < rows_per_rank; ++i) {
        for (size_t j = 0; j < N; ++j) {
            ft_x_step1_small_flat[i * N + j] = ft_x_step1_small[i][j];
        }
    }

    // declare ft_x_step1_T_small (we need it later for scattering the transpose of the FT)
    std::vector<std::complex<float>> ft_x_step1_T_small(entries_per_rank);

    // gather all ft_x_step1_small_flat
    if (rank==0) {
        std::vector<std::complex<float>> ft_x_step1(N * N);
        MPI_Gather(ft_x_step1_small_flat.data(), entries_per_rank, MPI_COMPLEX, ft_x_step1.data(), entries_per_rank, MPI_COMPLEX, 0, MPI_COMM_WORLD);

        // transpose ft_x_step1
        std::vector<std::complex<float>> ft_x_step1_T(N * N);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                ft_x_step1_T[i + N * j] = ft_x_step1[j + N * i];
            }
        }

        // scatter ft_x_step1 to other cores
        MPI_Scatter(ft_x_step1_T.data(), entries_per_rank, MPI_COMPLEX, ft_x_step1_T_small.data(), entries_per_rank, MPI_COMPLEX, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Gather(ft_x_step1_small_flat.data(), entries_per_rank, MPI_COMPLEX, NULL, 0, MPI_COMPLEX, 0, MPI_COMM_WORLD);
        MPI_Scatter(NULL, entries_per_rank, MPI_COMPLEX, ft_x_step1_T_small.data(), entries_per_rank, MPI_COMPLEX, 0, MPI_COMM_WORLD);
    }

    // take the FT (2)
    std::vector<std::vector<std::complex<float>>> ft_x_step2_small(rows_per_rank);
    for (size_t i = 0; i < rows_per_rank; ++i) {
        std::span<const std::complex<float>> ft_x_step1_T_small_row(&(ft_x_step1_T_small[i * N]), N);
        ft_x_step2_small[i] = run_fft(ft_x_step1_T_small_row, 1, N, true);
    }

    // flatten the computed FT
    std::vector<std::complex<float>> ft_x_step2_small_flat(entries_per_rank);
    for (size_t i = 0; i < rows_per_rank; ++i) {
        for (size_t j = 0; j < N; ++j) {
            ft_x_step2_small_flat[i * N + j] = ft_x_step2_small[i][j];
        }
    }

    // declare ft_x_step2 here, just for the comparison with single shot FT
    // this is not necessary in the final implementation
    std::vector<std::complex<float>> ft_x_step2_T(N * N);

    // gather all ft_x_step2_small_flat into ft_x_step2
    if (rank==0) {
        std::vector<std::complex<float>> ft_x_step2(N * N);
        MPI_Gather(ft_x_step2_small_flat.data(), entries_per_rank, MPI_COMPLEX, ft_x_step2.data(), entries_per_rank, MPI_COMPLEX, 0, MPI_COMM_WORLD);

        // transpose ft_x_step2
        // std::vector<std::complex<float>> ft_x_step2_T(N * N);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                ft_x_step2_T[i + N * j] = ft_x_step2[j + N * i];
            }
        }
    }
    else {
        MPI_Gather(ft_x_step2_small_flat.data(), entries_per_rank, MPI_COMPLEX, NULL, 0, MPI_COMPLEX, 0, MPI_COMM_WORLD);
    }

    // Wait for all cores to synchronize
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank==0) {
        auto end3 = omp_get_wtime();
        std::cout << "Option 3 (Split Computation, Multiple Cores). Time: " << (end3 - start3) * 1000.0 << " miliseconds." << std::endl;
        
        // COMPARE (run Option 1 again because the value isn't saved from earlier)
        std::vector<std::complex<float>> ft_x = run_fft(x_comp, 2, N, true);
        float l2norm = compute_L2norm_diff(ft_x, ft_x_step2_T);
        std::cout << "The L2 norm of the two vectors is: " << l2norm <<std::endl;
    }

    MPI_Finalize();
    return 0;
}