#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <span>
#include <string>
#include <omp.h>

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

template<typename T>
void print_vector(const std::vector<T> &v) {
    int size = v.size();
    for (int i=0; i < size; ++i) {            
        std::cout << v[i] << " ";
    }
    std::cout << "\n\n";
}

void print_array(const int v[], const int length) {
    for (int i=0; i < length; ++i) {            
        std::cout << v[i] << " ";
    }
    std::cout << "\n\n";
}

// typical 3D potential for Poisson Eq; scaling constant = 1
double compute_pairwise_potential(const double r, const double q, const double alpha, const std::string pot) {
    if (pot == "POISSON-3D") {
        return q * std::erfc(alpha * r) / r;
    }
    return 0;
}

void compute_potential(double *pot, const double* x, const double* y, const double* z, 
            const double* charges, const int n_particles, const double* x_other, const double* y_other, 
            const double* z_other, const int n_other, const double* offset, const std::string POT_TYPE, 
            const double r_cut_sq, const double alpha) {
    
    // iterate through all source particles
    for (int i = 0; i < n_particles; ++i) {
        // iterate through all other particles
        for (int j = 0; j < n_other; ++j) {
            // store the displacement
            const double dx = x[i] - x_other[j] - offset[0];
            const double dy = y[i] - y_other[j] - offset[1];
            const double dz = z[i] - z_other[j] - offset[2];

            const double rij_mag_sq = dx * dx + dy * dy + dz * dz;
            
            // avoid division by zero
            if (rij_mag_sq == 0 || rij_mag_sq >= r_cut_sq) { continue; }

            const double rij_mag = std::sqrt(rij_mag_sq);
            pot[i] += compute_pairwise_potential(rij_mag, charges[j], alpha, POT_TYPE);  
        }
    }
}

int main(int argc, char *argv[]) {
    const int n_src = 100; // number of sources
    const int n_trg = 100; // number of targets
    const int n_dim = 3;   // number of dimensions

    const std::string POT_TYPE = "POISSON-3D";

    // initialize the vectors: empty for now
    std::vector<double> r_src(n_src * n_dim, 0.0); // source coordinates
    std::vector<double> r_trg(n_trg * n_dim, 0.0); // target coordinates
    std::vector<double> charges(n_src, 0.0);      // source charges

    // generate uniform random source & target coordinates and charges
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i=0; i < n_src*n_dim; ++i) {
        r_src[i] = distribution(generator);
    }

    // std::cout << "Particle coordinates:" << std::endl;
    // print_vector(r_src);
    
    // custom coordinates for small tests
    // r_src = {0.3, 0.3, 0.3, 0.49, 0.3, 0.3};
    // r_src = {0.5, 0.01, 0.5, 0.5, 0.96, 0.5};

    // target coordinates are the same as source coordinates
    for (int i=0; i < n_trg*n_dim; ++i) {
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

    // ---------------------------------------------------------------------- //

    // Ewald-based calculation of the potential at the target points

    const double alpha = 10.0; // parameter to determine the extent of short-range and long-range interactions
    const double r_cut = 0.20; // cutoff distance for short-range interactions in 1D -- take a look at the graph
    const int N = 16; // 2^4 points
    // const int p = 4; // order of accuracy for interpolation
    const double h = 1.0 / N; // width; N intervals due to periodic boundary conditions
    const double r_cut_sq = r_cut * r_cut;

    // short-range interactions

    std::vector<double> pot_short(n_trg, 0.0);          // final potential vector
    std::vector<double> pot_short_sorted(n_trg, 0.0);     // sorted potential vector
    
    // use r_cut to decompose each dimension in boxes
    // TODO: edge cases (?)
    const int bin_size_x = int(r_cut * N + 1);
    const int nbins_x = N / bin_size_x + (N % bin_size_x > 0) * 1;
    const int n_boxes = nbins_x * nbins_x * nbins_x;

    // initialize box and neighbor lists
    int box_neighbors[n_boxes * n_boxes * n_boxes * 3 * 3 * 3];
    int box_corners[n_boxes * n_dim];
    std::vector<int> box_lengths(n_boxes, 0); // initialize to 0
    int box_begin[n_boxes];
    int box_offset[n_boxes];

    // sorted particle indices, coordinates, and charges
    int particles_sorted[n_src];
    double r_src_sorted[n_src * n_dim];
    double charges_sorted[n_src];

    for (size_t ind = 0; ind < n_src; ++ind) {
        // particle coordinates
        const double x = r_src[ind];
        const double y = r_src[n_src + ind];
        const double z = r_src[n_src * 2 + ind];

        const int i_x = int(x * N) / bin_size_x;
        const int i_y = int(y * N) / bin_size_x;
        const int i_z = int(z * N) / bin_size_x;

        ++box_lengths[i_x + nbins_x * (i_y + nbins_x * i_z)];
    }

    int current_offset = 0;
    for (size_t i = 0; i < n_boxes; ++i) {
        const int tmp = box_lengths[i];
        box_begin[i] = current_offset;
        box_offset[i] = current_offset;
        current_offset += tmp;

        // box corners
        box_corners[i * n_dim] = (i % (nbins_x * nbins_x)) % nbins_x;
        box_corners[i * n_dim + 1] = (i % (nbins_x * nbins_x)) / nbins_x;
        box_corners[i * n_dim + 2] = i / (nbins_x * nbins_x);
    }

    for (size_t ind = 0; ind < n_src; ++ind) {
        // particle coordinates
        const double x = r_src[ind];
        const double y = r_src[n_src + ind];
        const double z = r_src[n_src * 2 + ind];

        const int i_x = int(x * N) / bin_size_x;
        const int i_y = int(y * N) / bin_size_x;
        const int i_z = int(z * N) / bin_size_x;

        particles_sorted[box_offset[i_x + nbins_x * (i_y + nbins_x * i_z)]] = ind;
        ++box_offset[i_x + nbins_x * (i_y + nbins_x * i_z)];
    }

    // sort the position and charge vectors
    for (size_t i = 0; i < n_src; ++i) {
        r_src_sorted[i] = r_src[particles_sorted[i]];
        r_src_sorted[i + n_src] = r_src[particles_sorted[i] + n_src];
        r_src_sorted[i + n_src * 2] = r_src[particles_sorted[i] + n_src * 2];
        charges_sorted[i] = charges[particles_sorted[i]];
    }

    // store the neighbors
    for (size_t box = 0; box < n_boxes; ++box) {

        const int box_x = box_corners[box * n_dim];
        const int box_y = box_corners[box * n_dim + 1];
        const int box_z = box_corners[box * n_dim + 2];
        
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
            box_neighbors[box * 3 * 3 * 3 + nb_count] = curr_box;
            ++nb_count;
        }
        }
        }
    }
    
    double r0_other[n_dim];
    
    // BEGIN CALCULATIONS // 
    // ----------------------------------------------------------------------------- //
    
    auto start = omp_get_wtime();
    
    // iterate through all boxes
    for (size_t box = 0; box < n_boxes; ++box) {
        // go through the neighbors
        std::span<const int> neighbors(box_neighbors + box * 27, 27);
        for (int nb : neighbors) {
            // check periodic boundary conditions
            // TODO: Optimize this calculation?
            for (int a = 0; a < 3; ++a) {
                if (box_corners[box * n_dim + a] - box_corners[nb * n_dim + a] > 1) {
                    r0_other[a] = 1.0;
                }
                else if (box_corners[box * n_dim + a] - box_corners[nb * n_dim + a] < -1) {
                    r0_other[a] = -1.0;
                }
                else{
                    r0_other[a] = 0.0;
                }
            }

            // TODO: Generalize to more dimensions (?)

            const double* x = r_src_sorted + box_begin[box];
            const double* y = r_src_sorted + n_src + box_begin[box];
            const double* z = r_src_sorted + n_src * 2 + box_begin[box];

            const double* x_other = r_src_sorted + box_begin[nb];
            const double* y_other = r_src_sorted + n_src + box_begin[nb];
            const double* z_other = r_src_sorted + n_src * 2 + box_begin[nb];

            const double* ch = charges_sorted + box_begin[nb];

            double* pot = &(pot_short_sorted[0]) + box_begin[box];

            // TODO: Pass a pointer to the potential function
            compute_potential(pot, x, y, z, ch, box_lengths[box], 
                                x_other, y_other, z_other, box_lengths[nb], r0_other, 
                                POT_TYPE, r_cut_sq, alpha);
        }
    }

    // ----------------------------------------------------------------------------- //

    // de-sort the calculated potential values
    for (size_t i = 0; i < n_trg; ++i) {
        pot_short[particles_sorted[i]] = pot_short_sorted[i];
    }

    auto end = omp_get_wtime();

    std::cout << "Elapsed time: " << (end - start) * 1000.0 << " miliseconds" << std::endl;

    std::cout << "Short-range interaction:" << std::endl;
    print_vector(pot_short);

    return 0;
}