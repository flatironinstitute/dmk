#include <iostream>
#include <random>
#include <vector>
#include <fstream>
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

    // std::cout << "Charges:" << std::endl;
    // print_vector(charges);
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
    // TODO: edge cases (?), memory optimization (?)
    // box_index_grid[i] determines the 1D index where the i-th box begins
    int nbins_x = 0;
    const int bin_size_x = int(r_cut * N + 1);
    std::vector<int> box_index_grid;
    for (size_t i = 0; i < N; i += bin_size_x) {
        box_index_grid.push_back(i);
        ++nbins_x;
    }

    const int n_boxes = nbins_x * nbins_x * nbins_x;

    // initialize box and neighbor lists
    int box_neighbors[n_boxes * n_boxes * n_boxes * 3 * 3 * 3];
    int box_corners[n_boxes * n_dim];
    std::vector<int> box_lengths(n_boxes, 0);
    std::vector<int> box_begin(n_boxes, 0);
    std::vector<int> box_offset(n_boxes, 0);

    // sorted particle indices, coordinates, and charges
    std::vector<int> particles_sorted(n_src);
    std::vector<double> r_src_sorted(n_src * n_dim);
    std::vector<double> charges_sorted(n_src);

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
    
    std::vector<double> r0_other(3, 0.0);

    // BEGIN CALCULATIONS // 
    // ----------------------------------------------------------------------------- //

    auto start = omp_get_wtime();

    // iterate through all boxes
    for (size_t box = 0; box < n_boxes; ++box) {
        // go through the neighbors
        for (int neighbor_count = 0; neighbor_count < 27; ++neighbor_count) {
            const int nb = box_neighbors[box * 3 * 3 * 3 + neighbor_count];
            
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
            
            // iterate through all particles in a box
            for (int particle = box_begin[box]; particle < box_begin[box] + box_lengths[box]; ++particle) {
                // coordinates of the target particle
                // column-major
                const double x = r_src_sorted[particle];
                const double y = r_src_sorted[n_src + particle];
                const double z = r_src_sorted[n_src * 2 + particle];

                // for each neighbor compute all the pairwise interactions
                for (int other = box_begin[nb]; other < box_begin[nb] + box_lengths[nb]; ++other) {
                    // other coordinates
                    const double x_other = r0_other[0] + r_src_sorted[other];
                    const double y_other = r0_other[1] + r_src_sorted[n_src + other];
                    const double z_other = r0_other[2] + r_src_sorted[n_src * 2 + other];

                    // store the displacement
                    const double dx = x - x_other;
                    const double dy = y - y_other;
                    const double dz = z - z_other;

                    const double rij_mag_sq = dx * dx + dy * dy + dz * dz;
                    
                    // avoid division by zero
                    if (rij_mag_sq == 0 || rij_mag_sq >= r_cut_sq) { continue; }

                    const double rij_mag = std::sqrt(rij_mag_sq);
                    pot_short_sorted[particle] += charges_sorted[other] * std::erfc(rij_mag * alpha) / rij_mag;
                }
            }
        }
    }

    // de-sort the calculated potential values
    for (size_t i = 0; i < n_trg; ++i) {
        pot_short[particles_sorted[i]] = pot_short_sorted[i];
    }

    auto end = omp_get_wtime();

    // ----------------------------------------------------------------------------- //

    std::cout << "Elapsed time: " << (end - start) * 1000.0 << " miliseconds" << std::endl;

    std::cout << "Short-range interaction:" << std::endl;
    print_vector(pot_short);

    return 0;
}