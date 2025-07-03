#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <span>
#include <array>
#include <complex>
#include <fftw/fftw3_mkl.h>
#include <omp.h>

template <typename Real>
class Test_Case_System {
public:
    Test_Case_System(int n_sources, int n_targets, int n_dimensions, bool uniform) : 
                    n_src(n_sources), n_trg(n_targets), n_dim(n_dimensions), unif(uniform), 
                    r_src(n_sources * n_dimensions), r_trg(n_targets * n_dimensions), 
                    charges(n_sources) 
    {
        if (unif) {
            // generate uniform random source & target coordinates and charges
            std::default_random_engine generator;
            std::uniform_real_distribution<Real> distribution(0.0, 1.0);

            for (int i=0; i < n_src * n_dim; ++i) {
                r_src[i] = distribution(generator);
            }

            // target coordinates are the same as source coordinates
            for (int i=0; i < n_trg * n_dim; ++i) {
                r_trg[i] = r_src[i];
            }

            // initialize charges with random values in the range [-1, 1]
            Real total_charge = 0.0;
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
        }
    }

    // alternative constructor if you have pre-defined coordinates and charges
    Test_Case_System(int n_sources, int n_targets, int n_dimensions, std::vector<Real> &r_sources, 
                    std::vector<Real> &r_targets, std::vector<Real> &charge) :
                    n_src(n_sources), n_trg(n_targets), n_dim(n_dimensions), 
                    r_src(n_sources * n_dimensions), r_trg(n_targets * n_dimensions), 
                    charges(n_sources)
    {
        for (size_t i = 0; i < n_src * n_dim; ++i) {
            r_src[i] = r_sources[i];
        }

        for (size_t i = 0; i < n_trg * n_dim; ++i) {
            r_trg[i] = r_targets[i];
        }

        for (size_t i = 0; i < n_src; ++i) {
            charges[i] = charge[i];
        }
    }

    const int n_src;
    const int n_trg;
    const int n_dim;
    bool unif;
    std::vector<Real> r_src;
    std::vector<Real> r_trg; 
    std::vector<Real> charges;
};

// ------------------------------------------------------------------------------------------ //

template <typename Real>
class Short_Range_System {
public:
    // constructor
    Short_Range_System(const int n_boxes, const int n_dimensions, const int n_sources) : 
                        n_boxes(n_boxes), box_begin(n_boxes), box_lengths(n_boxes), 
                        box_corners(n_boxes * n_dimensions), box_neighbors(n_boxes), 
                        particles_sorted(n_sources), r_src_sorted(n_sources * n_dimensions),
                        charges_sorted(n_sources) {}

    // TODO: generalize for cases when target != source

    const int n_boxes;
    std::vector<int> box_begin;
    std::vector<int> box_lengths;
    std::vector<int> box_corners;
    std::vector<int> particles_sorted;
    std::vector<std::array<int, 27>> box_neighbors;
    std::vector<Real> r_src_sorted;
    std::vector<Real> charges_sorted;
};

// ------------------------------------------------------------------------------------------ //

template <typename Real>
void test_total_charge(const std::vector<Real> &ch) {
    int size = ch.size();
    std::cout << "Size: " << size << std::endl;

    Real total_charge = 0.0;
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

template<typename T>
void print_vector(const std::vector<T> &v) {
    int size = v.size();
    for (int i=0; i < size; ++i) {            
        std::cout << v[i] << " ";
    }
    std::cout << "\n\n";
}

// ------------------------------------------------------------------------------------------ //

template <typename Real>
Short_Range_System<Real> initialize_short_range(const Test_Case_System<Real> &System, Real alpha, Real r_cut, 
                                        int N, int n_dim) {
    // shortcuts
    const std::vector<Real> &r_src = System.r_src;
    const std::vector<Real> &charges = System.charges;
    const int n_src = System.n_src;
    
    const Real h = 1.0 / N;
    const Real r_cut_sq = r_cut * r_cut;

    // use r_cut to decompose each dimension in boxes
    // TODO: edge cases (?)
    const int bin_size_x = int(r_cut * N + 1);
    const int nbins_x = N / bin_size_x + (N % bin_size_x > 0) * 1;
    const int n_boxes = nbins_x * nbins_x * nbins_x;

    Short_Range_System<Real> Short_setup(n_boxes, n_dim, n_src);

    // initialize box and neighbor lists
    int box_offset[n_boxes];

    for (size_t ind = 0; ind < n_src; ++ind) {
        // particle coordinates
        const Real x = r_src[ind];
        const Real y = r_src[n_src + ind];
        const Real z = r_src[n_src * 2 + ind];

        const int i_x = int(x * N) / bin_size_x;
        const int i_y = int(y * N) / bin_size_x;
        const int i_z = int(z * N) / bin_size_x;

        ++Short_setup.box_lengths[i_x + nbins_x * (i_y + nbins_x * i_z)];
    }

    int current_offset = 0;
    for (size_t i = 0; i < n_boxes; ++i) {
        const int tmp = Short_setup.box_lengths[i];
        Short_setup.box_begin[i] = current_offset;
        box_offset[i] = current_offset;
        current_offset += tmp;

        // box corners
        Short_setup.box_corners[i * n_dim] = (i % (nbins_x * nbins_x)) % nbins_x;
        Short_setup.box_corners[i * n_dim + 1] = (i % (nbins_x * nbins_x)) / nbins_x;
        Short_setup.box_corners[i * n_dim + 2] = i / (nbins_x * nbins_x);
    }

    for (size_t ind = 0; ind < n_src; ++ind) {
        // particle coordinates
        const Real x = r_src[ind];
        const Real y = r_src[n_src + ind];
        const Real z = r_src[n_src * 2 + ind];

        const int i_x = int(x * N) / bin_size_x;
        const int i_y = int(y * N) / bin_size_x;
        const int i_z = int(z * N) / bin_size_x;

        Short_setup.particles_sorted[box_offset[i_x + nbins_x * (i_y + nbins_x * i_z)]] = ind;
        ++box_offset[i_x + nbins_x * (i_y + nbins_x * i_z)];
    }

    // sort the position and charge vectors
    for (size_t i = 0; i < n_src; ++i) {
        Short_setup.r_src_sorted[i] = System.r_src[Short_setup.particles_sorted[i]];
        Short_setup.r_src_sorted[i + n_src] = System.r_src[Short_setup.particles_sorted[i] + n_src];
        Short_setup.r_src_sorted[i + n_src * 2] = System.r_src[Short_setup.particles_sorted[i] + n_src * 2];
        Short_setup.charges_sorted[i] = System.charges[Short_setup.particles_sorted[i]];
    }

    // store the neighbors
    for (size_t box = 0; box < n_boxes; ++box) {

        const int box_x = Short_setup.box_corners[box * n_dim];
        const int box_y = Short_setup.box_corners[box * n_dim + 1];
        const int box_z = Short_setup.box_corners[box * n_dim + 2];
        
        int nb_count = 0;
        // go through the neighbors
        for (int i = box_x - 1; i <= box_x + 1; ++i) {
        for (int j = box_y - 1; j <= box_y + 1; ++j) {
        for (int k = box_z - 1; k <= box_z + 1; ++k) {
            // account for periodic boundaries
            const int new_i = (i + n_boxes) % nbins_x;
            const int new_j = (j + n_boxes) % nbins_x;
            const int new_k = (k + n_boxes) % nbins_x;
            // for each neighbor compute all the pairwise interactions
            const int curr_box = new_i + nbins_x * (new_j + nbins_x * new_k);
            Short_setup.box_neighbors[box][nb_count] = curr_box;
            ++nb_count;
        }
        }
        }
    }

    return Short_setup;
}

// ------------------------------------------------------------------------------------------ //

template <typename Real>
void compute_potential(Real* pot, const Real* x, const Real* y, const Real* z, 
            const Real* charges, int n_particles, const Real* x_other, const Real* y_other, 
            const Real* z_other, int n_other, const Real* offset, Real r_cut_sq, Real alpha) {
    
    // iterate through all source particles
    for (int i = 0; i < n_particles; ++i) {
        // iterate through all other particles
        for (int j = 0; j < n_other; ++j) {
            // store the displacement
            const Real dx = x[i] - x_other[j] - offset[0];
            const Real dy = y[i] - y_other[j] - offset[1];
            const Real dz = z[i] - z_other[j] - offset[2];

            const Real rij_mag_sq = dx * dx + dy * dy + dz * dz;
            
            // avoid division by zero
            if (rij_mag_sq == 0 || rij_mag_sq >= r_cut_sq) { continue; }

            const Real rij_mag = std::sqrt(rij_mag_sq);
            pot[i] += charges[j] * std::erfc(rij_mag * alpha) / rij_mag;  
        }
    }
}

// ------------------------------------------------------------------------------------------ //

template <typename Real>
std::vector<Real> evaluate_short_range(const Test_Case_System<Real> &System, Short_Range_System<Real> &Short, 
                                         Real r_cut, Real alpha) {
    
    // shortcuts
    const int n_boxes = Short.n_boxes;
    const int n_dim = System.n_dim;
    const int n_trg = System.n_trg;
    const int n_src = System.n_src;

    const Real r_cut_sq = r_cut * r_cut;
    std::vector<Real> pot(n_trg, 0.0);
    std::vector<Real> pot_sorted(n_trg, 0.0);     // sorted potential vector
    
    Real offset[n_dim];
    
    auto start = omp_get_wtime();
    
    // iterate through all boxes
    for (size_t box = 0; box < n_boxes; ++box) {
        // go through the neighbors
        for (int nb : Short.box_neighbors[box]) {
            // check periodic boundary conditions
            // TODO: Optimize this calculation?
            for (int a = 0; a < 3; ++a) {
                if (Short.box_corners[box * n_dim + a] - Short.box_corners[nb * n_dim + a] > 1) {
                    offset[a] = 1.0;
                }
                else if (Short.box_corners[box * n_dim + a] - Short.box_corners[nb * n_dim + a] < -1) {
                    offset[a] = -1.0;
                }
                else{
                    offset[a] = 0.0;
                }
            }

            // TODO: Generalize to more dimensions (?)

            const Real* x = &(Short.r_src_sorted[Short.box_begin[box]]);
            const Real* y = &(Short.r_src_sorted[n_src + Short.box_begin[box]]);
            const Real* z = &(Short.r_src_sorted[n_src * 2 + Short.box_begin[box]]);

            const Real* x_other = &(Short.r_src_sorted[Short.box_begin[nb]]);
            const Real* y_other = &(Short.r_src_sorted[n_src + Short.box_begin[nb]]);
            const Real* z_other = &(Short.r_src_sorted[n_src * 2 + Short.box_begin[nb]]);

            const Real* ch = &(Short.charges_sorted[0]) + Short.box_begin[nb];

            Real* pot_part = &(pot_sorted[0]) + Short.box_begin[box];

            // TODO: Pass a pointer to the potential function
            compute_potential(pot_part, x, y, z, ch, Short.box_lengths[box], x_other, y_other, z_other, 
                                Short.box_lengths[nb], offset, r_cut_sq, alpha);
        }
    }

    // ----------------------------------------------------------------------------- //

    // de-sort the calculated potential values
    for (size_t i = 0; i < System.n_trg; ++i) {
        pot[Short.particles_sorted[i]] = pot_sorted[i];
    }

    auto end = omp_get_wtime();

    std::cout << "Elapsed time: " << (end - start) * 1000.0 << " miliseconds" << std::endl;
    
    return pot;
}

// ------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------ //

template <typename Real>
std::vector<Real> compute_green_func(int N, Real alpha) {
    // TODO: generalize to different box lengths
    const Real h = 1.0 / N;
    const Real TWOPI = 2 * M_PI;
    const Real TWOPI_H = TWOPI / h;

    std::vector<Real> G(N * N * N, 0.0);

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

    return G;
}

// ------------------------------------------------------------------------------------------ //

// Lagrange polynomials
template <typename Real>
void evaluate_polynomials_04(std::vector<Real> &W, Real x, Real h) {
    // Pre-compute powers that are reused many times
    const Real h2 = h * h;          
    const Real h3 = h2 * h;         
    const Real h4 = h2 * h2;        

    const Real x2 = x * x;          
    const Real x3 = x2 * x;         
    const Real x4 = x2 * x2;

    const Real denom = 24.0 * h4;   

    W[0] = (x4 - 2.0*h*x3 - h2*x2 + 2.0*h3*x) / denom;

    W[1] = (-4.0*x4 + 4.0*h*x3 + 16.0*h2*x2 - 16.0*h3*x) / denom;

    W[2] = (6.0*x4 - 30.0*h2*x2 + 24.0*h4) / denom;

    W[3] = (-4.0*x4 - 4.0*h*x3 + 16.0*h2*x2 + 16.0*h3*x) / denom;

    W[4] = (x4 + 2.0*h*x3 - h2*x2 - 2.0*h3*x) / denom;
}

// //  Cardinal B–spline, order-4 (?)
// void evaluate_polynomials_04(std::vector<Real>& W, const Real x, const Real h) {
//     // reusable powers
//     const Real x2 = x * x;         
//     const Real x3 = x2 * x;           
//     const Real x4 = x2 * x2;          

//     const Real h2 = h * h;            
//     const Real h3 = h2 * h;           
//     const Real h4 = h2 * h2;          

//     // common denominator and its inverse (one divide)
//     const Real denom_inv = 1.0 / (96.0 * h4);

//     W[0] = ( 16.0 * x4 - 32.0 * h * x3 + 24.0 * h2 * x2 -  8.0 * h3 * x +  h4) * ( 0.25 * denom_inv);

//     W[1] = (-16.0 * x4 + 16.0 * h * x3 + 24.0 * h2 * x2 - 44.0 * h3 * x + 19.0 * h4) * denom_inv;

//     W[2] = (48.0 * x4 - 120.0 * h2 * x2 + 115.0 * h4) * (0.5 * denom_inv);

//     W[3] = (-16.0 * x4 - 16.0 * h * x3 + 24.0 * h2 * x2 + 44.0 * h3 * x + 19.0 * h4) * denom_inv;

//     W[4] = (16.0 * x4 + 32.0 * h * x3 + 24.0 * h2 * x2 + 8.0 * h3 * x + h4) * (0.25 * denom_inv);
// }


template <typename Real>
std::vector<Real> compute_contribution(Real r, int middle, int N, Real h, int p) {
    std::vector<Real> W(p+1, 0.0); // initialize the vector of polynomials
    Real dr = r - middle * h;
    Real dr_abs = std::abs(dr);
    dr = (dr_abs >= h / 2) ? dr - 1 : dr; // correction for periodic boundaries
    // TODO: generalize to more box lengths L (here it is 1.0)

    if (p==4) { evaluate_polynomials_04(W, dr, h); }
    return W;
}

// ------------------------------------------------------------------------------------------ //

template <typename Real>
void assign_charge(const std::vector<Real> &r_src, const std::vector<Real> &charges, 
                        std::vector<Real> &grid, int N, Real h, int p) {
    int n_charges = charges.size();
    const int N3 = N * N * N;
    
    // iterate through charges and their coordinates
    for (size_t ind = 0; ind < n_charges; ++ind) {
        const Real q = charges[ind];
        // column-major
        const Real x = r_src[ind];
        const Real y = r_src[n_charges + ind];
        const Real z = r_src[n_charges * 2 + ind];
        
        // identify the middle point
        // round to the nearest integer
        // TODO: generalize for odd p
        const int middle_x = int(x * N + 0.50) % N; // e.g., if x=0.99, middle_x=0
        const int middle_y = int(y * N + 0.50) % N;
        const int middle_z = int(z * N + 0.50) % N;

        // compute W_x, W_y, W_z
        std::vector<Real> W_x = compute_contribution(x, middle_x, N, h, p);
        std::vector<Real> W_y = compute_contribution(y, middle_y, N, h, p);
        std::vector<Real> W_z = compute_contribution(z, middle_z, N, h, p);

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

// ------------------------------------------------------------------------------------------ //

template <typename Real>
void back_interpolate(std::vector<Real> &r_trg, std::vector<Real> &pot, std::vector<Real> &trg_pot, 
                        int N, Real h, int p) {
    int n_trg = r_trg.size() / 3;
    
    // iterate through targets and their coordinates
    for (size_t ind = 0; ind < n_trg; ++ind) {
        // coordinates of the target point
        // column-major
        const Real x = r_trg[ind];
        const Real y = r_trg[n_trg + ind];
        const Real z = r_trg[n_trg * 2 + ind];

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
        std::vector<Real> W_x = compute_contribution(x, middle_x, N, h, p);
        std::vector<Real> W_y = compute_contribution(y, middle_y, N, h, p);
        std::vector<Real> W_z = compute_contribution(z, middle_z, N, h, p);

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

// ------------------------------------------------------------------------------------------ //

// USING REAL INPUT/OUTPUT
template <typename Real>
std::vector<std::complex<Real>> run_rfft(std::vector<Real> &in, const int n_dim, const int N) {
    // TODO: Generalize to more/less dimensions
    std::vector<std::complex<Real>> out(N * N * (N / 2 + 1), (0.0, 0.0));
    int n[] = {N, N, N};

    fftw_plan p = fftw_plan_dft_r2c(n_dim, n, in.data(), (fftw_complex *)out.data(), FFTW_MEASURE);
    
    fftw_execute(p);
    fftw_destroy_plan(p);
    return out;
}

template <typename Real>
std::vector<Real> run_irfft(std::vector<std::complex<Real>> &in, const int n_dim, const int N) {
    // TODO: Generalize to more/less dimensions
    std::vector<Real> out(N * N * N, 0.0);
    int n[] = {N, N, N};

    fftw_plan p = fftw_plan_dft_c2r(n_dim, n, (fftw_complex *)in.data(), out.data(), FFTW_MEASURE);
    
    fftw_execute(p);
    fftw_destroy_plan(p);
    return out;
}

// USING COMPLEX INPUT/OUTPUT
template <typename Real>
std::vector<std::complex<Real>> run_fft(std::vector<std::complex<Real>> &in, const int n_dim, const int N) {
    // TODO: Generalize to more/less dimensions
    std::vector<std::complex<Real>> out(N * N * N, (0.0, 0.0));
    int n[] = {N, N, N};

    fftw_plan p = fftw_plan_dft(n_dim, n, (fftw_complex *)in.data(), (fftw_complex *)out.data(), FFTW_FORWARD, FFTW_MEASURE);
    
    fftw_execute(p);
    fftw_destroy_plan(p);
    return out;
}

template <typename Real>
std::vector<std::complex<Real>> run_ifft(std::vector<std::complex<Real>> &in, const int n_dim, const int N) {
    // TODO: Generalize to more/less dimensions
    std::vector<std::complex<Real>> out(N * N * N, 0.0);
    int n[] = {N, N, N};

    fftw_plan p = fftw_plan_dft(n_dim, n, (fftw_complex *)in.data(), (fftw_complex *)out.data(), FFTW_BACKWARD, FFTW_MEASURE);
    
    fftw_execute(p);
    fftw_destroy_plan(p);
    return out;
}

// ------------------------------------------------------------------------------------------ //

template <typename Real>
std::vector<Real> evaluate_long_range(const std::vector<Real> &G, Test_Case_System<Real> System, 
                                        const int N, const int p) {
    const Real h = 1.0 / N;
    const int n_dim = System.n_dim;
    const int n_trg = System.n_trg;
    std::vector<Real> charges = System.charges;
    std::vector<Real> r_src = System.r_src;

    std::vector<Real> grid(N * N * N, 0.0); // N^3 grid with the kernel spread charged values
    assign_charge(r_src, charges, grid, N, h, p);

    // turn grid into a complex vector
    std::vector<std::complex<Real>> grid_comp(N * N * N, std::complex<Real>(0.0, 0.0));

    for (size_t i = 0; i < N * N * N; ++i) {
        grid_comp[i].real(grid[i]);
    }

    // take the FT (complex input)
    std::vector<std::complex<Real>> ft_density = run_fft(grid_comp, n_dim, N);

    // element-wise multiplication (convolution)
    for (size_t i = 0; i < N * N * N; ++i) {
        ft_density[i] *= G[i];
    }

    // take the inverse FT (complex output)
    std::vector<std::complex<Real>> inv_ft_density_comp = run_ifft(ft_density, n_dim, N);

    // turn the inverse FT from complex to real
    std::vector<Real> inv_ft_density(N * N * N, 0.0);
    for (size_t i = 0; i < N * N * N; ++i) {
        inv_ft_density[i] = inv_ft_density_comp[i].real();
    }

    // scale the inverse transform
    const int N3 = N * N * N;
    // TODO: check the type of the integer -- avoid overflow
    for (size_t i = 0; i < N * N * N; ++i) {
        inv_ft_density[i] = inv_ft_density[i] / N3;
    }

    // back-interpolate to infer the potential at the target points
    std::vector<Real> trg_pot(n_trg, 0.0);
    back_interpolate(r_src, inv_ft_density, trg_pot, N, h, p);
    
    return trg_pot;
}

// ------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------ //

void run_test_case_01() {
    const int n_src = 100; // number of sources
    const int n_trg = 100; // number of targets
    const int n_dim = 3; // number of dimensions
    
    Test_Case_System<double> System_01(n_src, n_trg, n_dim, true);

    const double alpha = 10.0; // the extent of short-range and long-range interactions
    const double r_cut = 0.20; // cutoff distance for short-range interactions
    const int N = 32; // 2^4 points
    const int p = 4; // order of accuracy for interpolation
    
    // short-range interactions
    Short_Range_System<double> Short_01 = initialize_short_range(System_01, alpha, r_cut, N, n_dim);
    std::vector<double> pot_short = evaluate_short_range(System_01, Short_01, r_cut, alpha);

    // long-range interactions
    std::vector<double> G_hat = compute_green_func(N, alpha);
    std::vector<double> pot_long = evaluate_long_range(G_hat, System_01, N, p);
    
    // self-interaction term
    std::vector<double> self_interaction(n_src, 0.0);
    for (size_t i = 0; i < n_src; ++i) {
        self_interaction[i] = 2 * alpha / std::sqrt(M_PI) * System_01.charges[i];
    }

    std::vector<double> pot(n_trg, 0.0);
    // add all terms
    for (size_t i = 0; i < n_trg; ++i) {
        pot[i] = pot_short[i] + pot_long[i] - self_interaction[i];
    }

    std::cout << "Final Potential:" << std::endl;
    print_vector(pot);
}

void run_test_case_02() {
    const int n_src = 2;
    const int n_trg = 2;
    const int n_dim = 3;

    // custom coordinates for small tests
    std::vector<float> r_src = {0.3, 0.3, 0.3, 0.49, 0.3, 0.3};
    // std::vector<Real> r_src = {0.5, 0.01, 0.5, 0.5, 0.96, 0.5};
    std::vector<float> charges = {0.5, -0.5};
    
    Test_Case_System<float> System_02(n_src, n_trg, n_dim, r_src, r_src, charges);

    const float alpha = 10.0f;
    const float r_cut = 0.20f;
    const int N = 16;
    const int p = 4;
    
    // short-range interactions
    Short_Range_System<float> Short_02 = initialize_short_range(System_02, alpha, r_cut, N, n_dim);
    std::vector<float> pot_short = evaluate_short_range(System_02, Short_02, r_cut, alpha);

    // long-range interactions
    std::vector<float> G_hat = compute_green_func(N, alpha);
    std::vector<float> pot_long = evaluate_long_range(G_hat, System_02, N, p);
    
    // self-interaction term
    std::vector<float> self_interaction(n_src, 0.0);
    for (size_t i = 0; i < n_src; ++i) {
        self_interaction[i] = 2 * alpha / std::sqrt(M_PI) * System_02.charges[i];
    }

    std::vector<float> pot(n_trg, 0.0);
    // add all terms
    for (size_t i = 0; i < n_trg; ++i) {
        pot[i] = pot_short[i] + pot_long[i] - self_interaction[i];
    }

    std::cout << "Final Potential:" << std::endl;
    print_vector(pot);
}

void run_test_case_03() {
    // Madelung constant verification
    const int n_src = 16;
    const int n_trg = 16;
    const int n_dim = 3;

    // custom coordinates for small tests
    std::vector<double> r_src = {0.0, 0.0, 0.5, 0.5, 0.25, 0.25, 0.75, 0.75,
                                0.5, 0.5, 0.0, 0.0, 0.25, 0.25, 0.75, 0.75,
                                0.0, 0.5, 0.0, 0.5, 0.25, 0.75, 0.25, 0.75,
                                0.5, 0.0, 0.5, 0.0, 0.25, 0.75, 0.25, 0.75,
                                0.0, 0.5, 0.5, 0.0, 0.25, 0.75, 0.75, 0.25,
                                0.5, 0.0, 0.0, 0.5, 0.75, 0.25, 0.25, 0.75};
    std::vector<double> charges = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                                    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
    
    Test_Case_System<double> System_02(n_src, n_trg, n_dim, r_src, r_src, charges);

    const double alpha = 10.0;
    const double r_cut = 0.20;
    const int N = 32;
    const int p = 4;
    
    // short-range interactions
    Short_Range_System<double> Short_02 = initialize_short_range(System_02, alpha, r_cut, N, n_dim);
    std::vector<double> pot_short = evaluate_short_range(System_02, Short_02, r_cut, alpha);

    std::cout << "Short-range interaction:" << std::endl;
    print_vector(pot_short);

    // long-range interactions
    std::vector<double> G_hat = compute_green_func(N, alpha);
    std::vector<double> pot_long = evaluate_long_range(G_hat, System_02, N, p);
    
    std::cout << "Long-range interaction:" << std::endl;
    print_vector(pot_long);

    // self-interaction term
    std::vector<double> self_interaction(n_src, 0.0);
    for (size_t i = 0; i < n_src; ++i) {
        self_interaction[i] = 2 * alpha / std::sqrt(M_PI) * charges[i];
    }

    std::cout << "Self-interaction potential:" << std::endl;
    print_vector(self_interaction);

    std::vector<double> pot(n_trg, 0.0);
    // add all terms
    for (size_t i = 0; i < n_trg; ++i) {
        pot[i] = pot_short[i] + pot_long[i] - self_interaction[i];
    }

    std::cout << "Final Potential:" << std::endl;
    print_vector(pot);

    // compute the total electrostatic energy
    double energy = 0.0;
    for (size_t i = 0; i < n_src; ++i) {
        energy += pot[i] * charges[i];
    }
    energy *= 0.5;
    double madelung = energy * 0.5 / n_src;
    std::cout << "Madelung constant for NaCl latice: " << madelung << std::endl; 
}


// ------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------ //

int main(int argc, char *argv[]) {
    run_test_case_01();
    // run_test_case_02();
    // run_test_case_03();

    return 0;
}