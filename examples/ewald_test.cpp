#include <iostream>
#include <random>
#include <vector>
#include <complex>
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <fftw/fftw3_mkl.h>

#define EPS 0.0001

void test_total_charge(const std::vector<double> &ch) {
    int size = ch.size();
    std::cout << "Size: " << size << std::endl;

    double total_charge = 0.0;
    for (int i=0; i < size; ++i) {
        total_charge += ch[i];
        if (ch[i] < -1.0 || ch[i] > 1.0) {
            std::cout << "Charge out of bounds: " << ch[i] << " at index " << i << std::endl;
        }
    }
    std::cout << "Total charge: " << total_charge << std::endl;
}

template<typename T>
void dump(const std::string &name, const T &data) {
    std::ofstream file(name, std::ios::binary);
    if (!data.size())
        return;

    file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(data[0]));
}

void print_vector(const std::vector<double> &v) {
    int size = v.size();
    for (int i=0; i < size; ++i) {            
        std::cout << v[i] << " ";
    }
    std::cout << "\n\n";
}

void print_vector_comp(const std::vector<std::complex<double>> &v) {
    int size = v.size();
    for (int i=0; i < size; ++i) {            
        std::cout << "(" << v[i].real() << ", " << v[i].imag() << ") ";
    }
    std::cout << "\n\n";

}

double vector_norm_sq(const std::vector<double> &v) {
    int size = v.size();
    double normsq = 0.0;

    for (int i = 0; i < size; ++i) {
        normsq += v[i]*v[i];
    }
    return normsq;
}

void compute_green_func(std::vector<double> &G, const int N, const double h, const double alpha) {
    const double TWOPI = 2 * M_PI;
    const double TWOPI_H = TWOPI / h;

    for (size_t w = 0; w < N; ++w) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                // TODO: optimize this step; structure the loop accordingly
                // conditionals are expensive (?)
                const auto i_new = (i > (N / 2)) ? i - N : i;
                const auto j_new = (j > (N / 2)) ? j - N : j;
                const auto w_new = (w > (N / 2)) ? w - N : w;
                
                const auto k_x = TWOPI * i_new;
                const auto k_y = TWOPI * j_new;
                const auto k_z = TWOPI * w_new;

                const auto mode_sq = k_x * k_x + k_y * k_y + k_z * k_z;
                if (mode_sq==0) { continue; }

                // update G_hat
                G[i + N * (j + N * w)] = 4 * M_PI / mode_sq * std::exp(-mode_sq / (4 * alpha * alpha));
            }
        }
    }
}

void compute_short_range(const std::vector<double> &r_src, const std::vector<double> &r_trg, std::vector<double> &pot, const std::vector<double> &charges, 
                        const int n_dim, const double r_cut, const int n_src, const int n_trg, const double alpha) {
    
    const double r_cut_sq = r_cut * r_cut;

    std::vector<double> diff(n_dim, 0.0); // vector to calculate displacements
    
    auto start = omp_get_wtime();

    for (size_t i = 0; i < n_trg; ++i) {
        for (size_t j = 0; j < n_src; ++j) {
            if (i==j) { continue; } // ensure no division by zero

            for (int k = 0; k < n_dim; ++k) {
                double diff_nonPBC = std::abs(r_src[k * n_src + j] - r_trg[k * n_src + i]);
                diff[k] = std::min(diff_nonPBC, 1.0 - diff_nonPBC);
            }
            
            const double rij_mag_sq = vector_norm_sq(diff);

            // compute the contribution only if it falls within a cutoff distance
            if (rij_mag_sq < r_cut_sq) {
                const double rij_mag = std::sqrt(rij_mag_sq);
                pot[i] += charges[j] * std::erfc(rij_mag * alpha) / rij_mag;
            }
        }
    }

    auto end = omp_get_wtime();

    std::cout << "Elapsed time: " << (end - start) * 1000.0 << " miliseconds" << std::endl;
}

// Lagrange polynomials
void evaluate_polynomials_04(std::vector<double> &W, const double x, const double h) {
    // Pre-compute powers that are reused many times
    const double h2 = h * h;          
    const double h3 = h2 * h;         
    const double h4 = h2 * h2;        

    const double x2 = x * x;          
    const double x3 = x2 * x;         
    const double x4 = x2 * x2;

    const double denom = 24.0 * h4;   

    W[0] = (x4 - 2.0*h*x3 - h2*x2 + 2.0*h3*x) / denom;

    W[1] = (-4.0*x4 + 4.0*h*x3 + 16.0*h2*x2 - 16.0*h3*x) / denom;

    W[2] = (6.0*x4 - 30.0*h2*x2 + 24.0*h4) / denom;

    W[3] = (-4.0*x4 - 4.0*h*x3 + 16.0*h2*x2 + 16.0*h3*x) / denom;

    W[4] = (x4 + 2.0*h*x3 - h2*x2 - 2.0*h3*x) / denom;
}


// // cardinal B-splines
// void evaluate_polynomials_04(std::vector<double> &W, const double x, const double h) {
//     // TODO: optimize polynomial evaluation -- Horner scheme (?)
//     double denominator = 96 * h * h * h * h;
//     W[0] = (16 * x * x * x * x - 32 * h * x * x * x + 24 * h * h * x * x - 8 * h * h * h * x + h * h * h * h) / (4 * denominator);
//     W[1] = (-16 * x * x * x * x + 16 * h * x * x * x + 24 * h * h * x * x - 44 * h * h * h * x + 19 * h * h * h * h) / denominator;
//     W[2] = (48 * x * x * x * x - 120 * h * h * x * x + 115 * h * h * h * h) / (2 * denominator);
//     W[3] = (-16 * x * x * x * x - 16 * h * x * x * x + 24 * h * h * x * x + 44 * h * h * h * x + 19 * h * h * h * h) / denominator;
//     W[4] = (16 * x * x * x * x + 32 * h * x * x * x + 24 * h * h * x * x + 8 * h * h * h * x + h * h * h * h) / (4 * denominator);
// }

std::vector<double> compute_contribution(const double r, const int middle, const int N, const double h, const int p) {
    std::vector<double> W(p+1, 0.0); // initialize the vector of polynomials
    double dr = r - middle * h;
    double dr_abs = std::abs(dr);
    dr = (dr_abs >= h / 2) ? dr - 1 : dr; // correction for periodic boundaries
    // TODO: generalize to more box lengths L (here it is 1.0)

    if (p==4) { evaluate_polynomials_04(W, dr, h); }
    return W;
}

void assign_charge(const std::vector<double> &r_src, const std::vector<double> &charges, std::vector<double> &grid, const int N, const double h, const int p) {
    int n_charges = charges.size();
    const int N3 = N * N * N;
    
    // iterate through charges and their coordinates
    for (size_t ind = 0; ind < n_charges; ++ind) {
        const double q = charges[ind];
        // column-major
        const double x = r_src[ind];
        const double y = r_src[n_charges + ind];
        const double z = r_src[n_charges * 2 + ind];
        
        // identify the middle point
        // round to the nearest integer
        // TODO: generalize for odd p
        const int middle_x = int(x * N + 0.50) % N; // e.g., if x=0.99, middle_x=0
        const int middle_y = int(y * N + 0.50) % N;
        const int middle_z = int(z * N + 0.50) % N;

        // compute W_x, W_y, W_z
        std::vector<double> W_x = compute_contribution(x, middle_x, N, h, p);
        std::vector<double> W_y = compute_contribution(y, middle_y, N, h, p);
        std::vector<double> W_z = compute_contribution(z, middle_z, N, h, p);

        // update the grid values
        int count_x, count_y, count_z; // grid point indices
        count_x = 0;
        for (int i = (middle_x - (p / 2) + N) % N; i != (middle_x + (p / 2) + 1) % N; i = (i + 1) % N) {
            count_y = 0;
            for (int j = (middle_y - (p / 2) + N) % N; j != (middle_y + (p / 2) + 1) % N; j = (j + 1) % N) {
                count_z = 0;
                for (int k = (middle_z - (p / 2) + N) % N; k != (middle_z + (p / 2) + 1) % N; k = (k + 1) % N) {
                    grid[i + N * (j + N * k)] += q * W_x[count_x] * W_y[count_y] * W_z[count_z] * N3;
                    ++count_z;
                }
                ++count_y;
            }
            ++count_x;
        }
    }
}

void back_interpolate(std::vector<double> &r_trg, std::vector<double> &pot, std::vector<double> &trg_pot, const int N, const double h, const int p) {
    int n_trg = r_trg.size() / 3;
    
    // iterate through targets and their coordinates
    for (size_t ind = 0; ind < n_trg; ++ind) {
        // coordinates of the target point
        // column-major
        const double x = r_trg[ind];
        const double y = r_trg[n_trg + ind];
        const double z = r_trg[n_trg * 2 + ind];

        // identify the middle point
        // round to the nearest integer
        // TODO: generalize for odd p / polynomial order
        const int middle_x = int(x * N + 0.50) % N; // e.g., if x=0.99, middle_x=0
        const int middle_y = int(y * N + 0.50) % N;
        const int middle_z = int(z * N + 0.50) % N;

        if (middle_x > N - 1 || middle_y > N - 1 || middle_z > N - 1) {
            std::cout << "ERROR!" << std::endl;
        }

        // compute W_x, W_y, W_z
        std::vector<double> W_x = compute_contribution(x, middle_x, N, h, p);
        std::vector<double> W_y = compute_contribution(y, middle_y, N, h, p);
        std::vector<double> W_z = compute_contribution(z, middle_z, N, h, p);

        // update the grid values
        int count_x, count_y, count_z; // grid point indices
        count_x = 0;
        for (int i = (middle_x - (p / 2) + N) % N; i != (middle_x + (p / 2) + 1) % N; i = (i + 1) % N) {
            count_y = 0;
            for (int j = (middle_y - (p / 2) + N) % N; j != (middle_y + (p / 2) + 1) % N; j = (j + 1) % N) {
                count_z = 0;
                for (int k = (middle_z - (p / 2) + N) % N; k != (middle_z + (p / 2) + 1) % N; k = (k + 1) % N) {
                    trg_pot[ind] += pot[i + N * (j + N * k)] * W_x[count_x] * W_y[count_y] * W_z[count_z];
                    ++count_z;
                }
                ++count_y;
            }
            ++count_x;
        }
    }
}

// USING REAL INPUT/OUTPUT
std::vector<std::complex<double>> run_rfft(std::vector<double> &in, const int n_dim, const int N) {
    // TODO: Generalize to more/less dimensions
    std::vector<std::complex<double>> out(N * N * (N / 2 + 1), (0.0, 0.0));
    int n[] = {N, N, N};

    fftw_plan p = fftw_plan_dft_r2c(n_dim, n, in.data(), (fftw_complex *)out.data(), FFTW_MEASURE);
    
    fftw_execute(p);
    fftw_destroy_plan(p);
    return out;
}

std::vector<double> run_irfft(std::vector<std::complex<double>> &in, const int n_dim, const int N) {
    // TODO: Generalize to more/less dimensions
    std::vector<double> out(N * N * N, 0.0);
    int n[] = {N, N, N};

    fftw_plan p = fftw_plan_dft_c2r(n_dim, n, (fftw_complex *)in.data(), out.data(), FFTW_MEASURE);
    
    fftw_execute(p);
    fftw_destroy_plan(p);
    return out;
}

// USING COMPLEX INPUT/OUTPUT
std::vector<std::complex<double>> run_fft(std::vector<std::complex<double>> &in, const int n_dim, const int N) {
    // TODO: Generalize to more/less dimensions
    std::vector<std::complex<double>> out(N * N * N, (0.0, 0.0));
    int n[] = {N, N, N};

    fftw_plan p = fftw_plan_dft(n_dim, n, (fftw_complex *)in.data(), (fftw_complex *)out.data(), FFTW_FORWARD, FFTW_MEASURE);
    
    fftw_execute(p);
    fftw_destroy_plan(p);
    return out;
}

std::vector<std::complex<double>> run_ifft(std::vector<std::complex<double>> &in, const int n_dim, const int N) {
    // TODO: Generalize to more/less dimensions
    std::vector<std::complex<double>> out(N * N * N, 0.0);
    int n[] = {N, N, N};

    fftw_plan p = fftw_plan_dft(n_dim, n, (fftw_complex *)in.data(), (fftw_complex *)out.data(), FFTW_BACKWARD, FFTW_MEASURE);
    
    fftw_execute(p);
    fftw_destroy_plan(p);
    return out;
}

int main(int argc, char *argv[]) {
    // useful quantities
    const double twopi = 2 * M_PI;

    const int n_src = 100; // number of sources
    const int n_trg = 100; // number of targets
    const int n_dim = 3;   // number of dimensions

    // initialize the vectors: empty for now
    std::vector<double> r_src(n_src * n_dim, 0.0); // source coordinates
    std::vector<double> r_trg(n_trg * n_dim, 0.0); // target coordinates
    std::vector<double> charges(n_src, 0.0);      // source charges

    // generate uniform random source & target coordinates and charges
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    for (int i=0; i < n_src*n_dim; ++i) {
        r_src[i] = distribution(generator);
    }

    // std::cout << "Particle coordinates:" << std::endl;
    // print_vector(r_src);
    
    // r_src = {0.3, 0.3, 0.3, 0.31, 0.3, 0.3};
    // r_src = {0.5, 0.01, 0.5, 0.5, 0.96, 0.5};

    for (int i=0; i < n_trg * n_dim; ++i) {
        // r_trg[i] = distribution(generator);
        r_trg[i] = r_src[i];
    }

    // initialize charges with random values in the range [-1, 1]
    double total_charge = 0.0;
    for (int i=0; i < n_src; ++i) {
        charges[i] = (distribution(generator))*2-1;
        total_charge += charges[i];
    }

    // normalize charges to ensure total charge is zero
    if (std::abs(total_charge) > 1e-5) {
        for (int i=0; i < n_src; ++i) {
            charges[i] = charges[i] - (total_charge / n_src);
        }
    }

    std::cout << "Charges:" << std::endl;
    print_vector(charges);
    // test_total_charge(charges);

    // --------------------------------------------------------------------- //

    // // brute force calculation of the potential at the target points
    // std::vector<double> pot_trg(n_trg, 0.0); // initialize potential at targets
    
    // // TODO: Generalize to different box lengths
    
    // for (int n_i = -2; n_i < 3; ++n_i) {
    //     for (int n_j = -2; n_j < 3; ++n_j) {
    //         for (int n_k = -2; n_k < 3; ++n_k) {
    //             std::vector<int> m = {n_j, n_k, n_i};
    //             for (size_t i = 0; i < n_trg; ++i) {
    //                 for (size_t j = 0; j < n_src; ++j) {

    //                     std::vector<double> diff(n_dim, 0.0); // vector to calculate displacements
    //                     diff[0] = r_src[j] - r_trg[i] + m[0];
    //                     diff[1] = r_src[j + n_src] - r_trg[i + n_src] + m[1];
    //                     diff[2] = r_src[j + 2 * n_src] - r_trg[i + 2 * n_src] + m[2];
                        
    //                     const double rij_mag_sq = vector_norm_sq(diff);
                    
    //                     if (rij_mag_sq == 0) { continue; } // ensure no division by zero
    //                     const double rij_mag = std::sqrt(rij_mag_sq);
    //                     pot_trg[i] += charges[j] / rij_mag;
    //                 }
    //             }
    //         }
    //     }
    // }

    // // print the results
    // std::cout << "Brute-force potential calculation; cubic summation:" << std::endl;
    // print_vector(pot_trg);

    // ---------------------------------------------------------------------- //

    // Ewald-based calculation of the potential at the target points

    const double alpha = 10.0; // parameter to determine the extent of short-range and long-range interactions
    const double r_cut = 0.20; // cutoff distance for short-range interactions in 1D -- take a look at the graph
    const int N = 16; // 2^4 points
    const int p = 4; // order of accuracy for interpolation
    const double h = 1.0 / N; // width; N intervals due to periodic boundary conditions
    
    // precompute Green's function in Fourier space
    std::vector<double> G_hat(N * N * N, 0.0);
    compute_green_func(G_hat, N, h, alpha);

    // ---------------------------------------------------------------------- //
    // short-range interactions

    // TODO: optimize the calculation; still O(N^2)
    std::vector<double> pot_trg_short(n_trg, 0.0);
    compute_short_range(r_src, r_trg, pot_trg_short, charges, n_dim, r_cut, n_src, n_trg, alpha);
    std::cout << "Short-range interaction:" << std::endl;
    print_vector(pot_trg_short);

    // ---------------------------------------------------------------------- //
    // long-range interactions - Fourier space
    
    // charge assignment - Lagrange-based (PME)
    std::vector<double> grid(N * N * N, 0.0); // N^3 grid with the kernel spread charged values
    assign_charge(r_src, charges, grid, N, h, p);

    // turn grid into a complex vector
    std::vector<std::complex<double>> grid_comp(N * N * N, std::complex<double>(0.0, 0.0));

    for (size_t i = 0; i < N * N * N; ++i) {
        grid_comp[i].real(grid[i]);
    }

    // take the FT (complex input)
    std::vector<std::complex<double>> ft_density = run_fft(grid_comp, n_dim, N);

    // element-wise multiplication (convolution)
    for (size_t i = 0; i < N * N * N; ++i) {
        ft_density[i] *= G_hat[i];
    }

    // take the inverse FT (complex output)
    std::vector<std::complex<double>> inv_ft_density_comp = run_ifft(ft_density, n_dim, N);

    // turn the inverse FT from complex to real
    std::vector<double> inv_ft_density(N * N * N, 0.0);
    for (size_t i = 0; i < N * N * N; ++i) {
        inv_ft_density[i] = inv_ft_density_comp[i].real();
    }

    //TODO: optimize this scaling process (?)
    // scale the inverse transform
    for (size_t i = 0; i < N * N * N; ++i) {
        inv_ft_density[i] = inv_ft_density[i] / (N * N * N);
    }

    // back-interpolate to infer the potential at the target points
    std::vector<double> trg_pot(n_trg, 0.0);
    back_interpolate(r_src, inv_ft_density, trg_pot, N, h, p);
    std::cout << "Long-range interaction -- computed with FFT:" << std::endl;
    print_vector(trg_pot);

    // TODO: compute self interaction term only for r_src=r_trg
    std::vector<double> self_interaction(n_src, 0.0);
    for (size_t i = 0; i < n_src; ++i) {
        self_interaction[i] = 2 * alpha / std::sqrt(M_PI) * charges[i];
    }
    std::cout << "Self-interaction term:" << std::endl;
    print_vector(self_interaction);

    // add short-range terms
    // subtract self interaction terms -- useful when r_src = r_trg
    for (size_t i = 0; i < n_trg; ++i) {
        trg_pot[i] -= self_interaction[i];
        trg_pot[i] += pot_trg_short[i];
    }
    std::cout << "Final Potential - computed with PME:" << std::endl;
    print_vector(trg_pot);

    return 0;
}