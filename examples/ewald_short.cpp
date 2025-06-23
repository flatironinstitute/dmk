#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <omp.h>

#define EPS 0.0001

void test_total_charge(const std::vector<double> &ch){
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
void printVector(const std::vector<T> &v){
    int size = v.size();
    for (int i=0; i < size; ++i) {            
        std::cout << v[i] << " ";
    }
    std::cout << "\n\n";
}

// TODO: Optimize this (?)
// input: vector with the begin index of each box, particle in question index
// returns: the index of the interval (in 1D) in which the particle belongs
int findIntervalIndex(const std::vector<int> &indices, const int index){
    int length = indices.size();

    int i = 0;
    for ( ; i < length; ++i){
        if (index < indices[i]){ break; }
    }

    return i - 1;
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
    // printVector(r_src);
    
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
    // printVector(charges);
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
    int n_boxes = 0;
    int new_ind;
    double pos;
    // vector to save the starting grid point index where the box begins
    // box_index_grid[i] determines the 1D index where the i-th box begins
    std::vector<int> box_index_grid = {0};
    while (1){
        pos = box_index_grid[n_boxes] * h + r_cut;
        new_ind = (int(pos * N) + 1) % N;
        box_index_grid.push_back(new_ind);
        ++n_boxes;
        if (pos >= 1.0 || new_ind == 0){ 
            box_index_grid.pop_back();
            break;
        }
    }

    // initialize box and neighbor lists
    // box_particles[i] contains a vector with the particle indices of the i-th box
    std::vector<std::vector<int>> box_particles(n_boxes * n_boxes * n_boxes);
    std::vector<int> nb_list(3 * 3 * 3, 0);
    std::vector<std::vector<int>> box_neighbors(n_boxes * n_boxes * n_boxes, nb_list);
    std::vector<int> box_corner(n_dim, 0);
    std::vector<std::vector<int>> box_corners(n_boxes * n_boxes * n_boxes, box_corner);
    std::vector<int> box_lengths(n_boxes * n_boxes * n_boxes, 0);
    std::vector<int> box_begin(n_boxes * n_boxes * n_boxes, 0);

    // sorted particle indices, coordinates, and charges
    std::vector<int> particles_sorted(n_src);
    std::vector<double> r_src_sorted(n_src * n_dim);
    std::vector<double> charges_sorted(n_src);

    double x, y, z;
    int i_x, i_y, i_z;

    for (size_t ind = 0; ind < n_src; ++ind){
        // particle coordinates
        x = r_src[ind];
        y = r_src[n_src + ind];
        z = r_src[n_src * 2 + ind];

        i_x = findIntervalIndex(box_index_grid, int(x * N));
        i_y = findIntervalIndex(box_index_grid, int(y * N));
        i_z = findIntervalIndex(box_index_grid, int(z * N));

        box_particles[i_x + i_y * n_boxes + i_z * n_boxes * n_boxes].push_back(ind);
    }

    // get the particle order
    int fast_count = 0, prev_fast_count = 0;
    for (size_t box = 0; box < n_boxes * n_boxes * n_boxes; ++box){
        box_begin[box] = fast_count;
        for (int particle : box_particles[box]){
            particles_sorted[fast_count] = particle;
            ++fast_count;
        }
        box_lengths[box] = fast_count - prev_fast_count;
        prev_fast_count = fast_count; 
    }

    // printVector(box_lengths);
    // printVector(box_begin);

    // sort the position and charge vectors
    for (size_t i = 0; i < n_src; ++i){
        r_src_sorted[i] = r_src[particles_sorted[i]];
        r_src_sorted[i + n_src] = r_src[particles_sorted[i] + n_src];
        r_src_sorted[i + n_src * 2] = r_src[particles_sorted[i] + n_src * 2];
        charges_sorted[i] = charges[particles_sorted[i]];
    }

    int box_x, box_y, box_z, new_i, new_j, new_k, curr_box, nb_count; // count is declared above

    // store the neighbors
    for (size_t box = 0; box < n_boxes * n_boxes * n_boxes; ++box){

        box_x = (box % (n_boxes * n_boxes)) % n_boxes;
        box_y = (box % (n_boxes * n_boxes)) / n_boxes;
        box_z = box / (n_boxes * n_boxes);

        box_corners[box] = {box_x, box_y, box_z};
        
        nb_count = 0;
        // go through the neighbors
        for (int i = box_x - 1; i <= box_x + 1; ++i){
        for (int j = box_y - 1; j <= box_y + 1; ++j){
        for (int k = box_z - 1; k <= box_z + 1; ++k){
            // account for periodic boundaries
            new_i = (i + n_boxes) % n_boxes;
            new_j = (j + n_boxes) % n_boxes;
            new_k = (k + n_boxes) % n_boxes;
            // for each neighbor compute all the pairwise interactions
            curr_box = new_i + new_j * n_boxes + new_k * n_boxes * n_boxes;
            box_neighbors[box][nb_count] = curr_box;
            ++nb_count;
        }
        }
        }
    }
    
    double dx, dy, dz; // x, y, z have been declared above
    double rij_mag, rij_mag_sq;
    
    double x_other, y_other, z_other;
    std::vector<double> r0_other(3, 0.0);

    // BEGIN CALCULATIONS // 
    // ----------------------------------------------------------------------------- //

    auto start = omp_get_wtime();

    // iterate through all boxes
    int particle_count = 0; // count the particle index
    for (size_t box = 0; box < n_boxes * n_boxes * n_boxes; ++box){
        
        // iterate through all particles in a box
        for (int i = 0; i < box_lengths[box]; ++i){
            // coordinates of the target particle
            // column-major
            x = r_src_sorted[particle_count];
            y = r_src_sorted[n_src + particle_count];
            z = r_src_sorted[n_src * 2 + particle_count];
            
            // go through the neighbors
            for (int nb : box_neighbors[box]){

                // check periodic boundary conditions
                // TODO: Optimize this calculation?
                for (int a = 0; a < 3; ++a){
                    if (box_corners[box][a] - box_corners[nb][a] > 1){
                        r0_other[a] = 1.0;
                    }
                    else if (box_corners[box][a] - box_corners[nb][a] < -1){
                        r0_other[a] = -1.0;
                    }
                    else{
                        r0_other[a] = 0.0;
                    }
                }

                // for each neighbor compute all the pairwise interactions
                for (int other = box_begin[nb]; other < box_begin[nb] + box_lengths[nb]; ++other){
                    // other coordinates
                    x_other = r0_other[0] + r_src_sorted[other];
                    y_other = r0_other[1] + r_src_sorted[n_src + other];
                    z_other = r0_other[2] + r_src_sorted[n_src * 2 + other];

                    // store the displacement
                    dx = x - x_other;
                    dy = y - y_other;
                    dz = z - z_other;

                    rij_mag_sq = dx * dx + dy * dy + dz * dz;
                    
                    // avoid division by zero
                    if (rij_mag_sq == 0 || rij_mag_sq >= r_cut_sq){ continue; }

                    rij_mag = std::sqrt(rij_mag_sq);
                    pot_short_sorted[particle_count] += charges_sorted[other] * std::erfc(rij_mag * alpha) / rij_mag;
                }
            }
            ++particle_count;
        }
    }

    // de-sort the calculated potential values
    for (size_t i = 0; i < n_trg; ++i){
        pot_short[particles_sorted[i]] = pot_short_sorted[i];
    }

    auto end = omp_get_wtime();

    // ----------------------------------------------------------------------------- //

    std::cout << "Elapsed time: " << (end - start) * 1000.0 << " miliseconds" << std::endl;

    std::cout << "Short-range interaction:" << std::endl;
    printVector(pot_short);

    return 0;
}