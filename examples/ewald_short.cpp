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

int findStartIndex(const std::vector<int> &indices, const int index){
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
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    for (int i=0; i < n_src*n_dim; ++i) {
        r_src[i] = distribution(generator);
    }

    std::cout << "Particle coordinates:" << std::endl;
    //printVector(r_src);
    
    // r_src = {0.3, 0.3, 0.3, 0.49, 0.3, 0.3};
    // r_src = {0.5, 0.01, 0.5, 0.5, 0.96, 0.5};

    for (int i=0; i < n_trg*n_dim; ++i) {
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
    //printVector(charges);
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

    // TODO: optimize the calculation; still O(N^2)
    std::vector<double> pot_trg_short(n_trg, 0.0);
    //computeShortRange(r_src, r_trg, pot_trg_short, charges, n_dim, r_cut, n_src, n_trg, alpha);
    
    // use r_cut to decompose each dimension in boxes
    // TODO: edge cases (?)
    int n_boxes = 0;
    int new_ind;
    double pos;
    std::vector<int> start_index = {0};
    while (1){
        pos = start_index[n_boxes] * h + r_cut;
        new_ind = (int(pos * N) + 1) % N;
        start_index.push_back(new_ind);
        ++n_boxes;
        if (pos >= 1.0 || new_ind == 0){ 
            start_index.pop_back();
            break;
        }
    }

    // initialize box and neighbor lists
    std::vector<std::vector<int>> boxes(n_boxes * n_boxes * n_boxes);

    std::vector<int> nb_list(n_dim * n_dim * n_dim, 0);
    std::vector<std::vector<int>> neighbors(n_boxes * n_boxes * n_boxes, nb_list);
    
    std::vector<int> box_corner(n_dim, 0);
    std::vector<std::vector<int>> box_corners(n_boxes * n_boxes * n_boxes, box_corner);

    double x, y, z;
    int i_x, i_y, i_z;

    for (size_t ind = 0; ind < n_src; ++ind){
        // particle coordinates
        x = r_src[ind * 3];
        y = r_src[ind * 3 + 1];
        z = r_src[ind * 3 + 2];

        i_x = findStartIndex(start_index, int(x * N));
        i_y = findStartIndex(start_index, int(y * N));
        i_z = findStartIndex(start_index, int(z * N));

        boxes[i_x + i_y * n_boxes + i_z * n_boxes * n_boxes].push_back(ind);
    }

    // print the content of non-empty boxes
    // for (int i = 0; i < n_boxes * n_boxes * n_boxes; ++i){
    //     if (!boxes[i].empty()){ 
    //         std::cout << i / (n_boxes * n_boxes) << " " << (i % (n_boxes * n_boxes)) / n_boxes << " " << (i % (n_boxes * n_boxes)) % n_boxes << std::endl;
    //         printVector(boxes[i]); }
    // }

    int box_x, box_y, box_z, curr_box, count;

    // store the neighbors
    for (size_t box = 0; box < n_boxes * n_boxes * n_boxes; ++box){

        box_x = (box % (n_boxes * n_boxes)) % n_boxes;
        box_y = (box % (n_boxes * n_boxes)) / n_boxes;
        box_z = box / (n_boxes * n_boxes);

        box_corners[box] = {box_x, box_y, box_z};

        // std::cout << "Box " << box << " " << box_x << " " << box_y << " " << box_z << ": ";
        
        count = 0;
        // go through the neighbors
        for (int i = box_x - 1; i <= box_x + 1; ++i){
        for (int j = box_y - 1; j <= box_y + 1; ++j){
        for (int k = box_z - 1; k <= box_z + 1; ++k){
            // for each neighbor compute all the pairwise interactions
            curr_box = (i + j * n_boxes + k * n_boxes * n_boxes + n_boxes * n_boxes * n_boxes) % (n_boxes * n_boxes * n_boxes);
            // std::cout << curr_box << " ";
            neighbors[box][count] = curr_box;
            ++count;
        }
        }
        }
    }

    // for (int i = 0; i < n_boxes * n_boxes * n_boxes; ++i){
    //     printVector(neighbors[i]);
    // }
    
    double dx, dy, dz; // x, y, z have been declared above
    double rij_mag, rij_mag_sq;
    
    double x_other, y_other, z_other;
    std::vector<double> r0_other(3, 0.0);

    auto start = omp_get_wtime();

    // iterate through all boxes
    for (size_t box = 0; box < n_boxes * n_boxes * n_boxes; ++box){
        
        // iterate through all particles in a box
        for (int particle : boxes[box]){
            // coordinates of the target particle
            x = r_src[particle * n_dim];
            y = r_src[particle * n_dim + 1];
            z = r_src[particle * n_dim + 2];

            // std::cout << "Box " << box << " " << box_x << " " << box_y << " " << box_z << ": ";
            
            // go through the neighbors
            for (int nb : neighbors[box]){

                // check periodic boundary conditions
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
                for (int other : boxes[nb]){
                    // other coordinates
                    x_other = r0_other[0] + r_src[other * n_dim];
                    y_other = r0_other[1] + r_src[other * n_dim + 1];
                    z_other = r0_other[2] + r_src[other * n_dim + 2];

                    // store the displacement
                    dx = x - x_other;
                    dy = y - y_other;
                    dz = z - z_other;

                    // // store the displacement
                    // dx = std::abs(x - x_other);
                    // dy = std::abs(y - y_other);
                    // dz = std::abs(z - z_other);

                    // // account for periodic boundary conditions
                    // dx = std::min(dx, 1.0 - dx);
                    // dy = std::min(dy, 1.0 - dy);
                    // dz = std::min(dz, 1.0 - dz);

                    rij_mag_sq = dx * dx + dy * dy + dz * dz;
                    
                    // avoid division by zero
                    if (rij_mag_sq == 0 || rij_mag_sq >= r_cut_sq){ continue; }

                    rij_mag = std::sqrt(rij_mag_sq);
                    // std::cout << rij_mag << " ";
                    pot_trg_short[particle] += charges[other] * std::erfc(rij_mag * alpha) / rij_mag;
                }
            }
        }
    }
    // std::cout << "\n";

    auto end = omp_get_wtime();

    std::cout << "Elapsed time: " << (end - start) * 1000.0 << " miliseconds" << std::endl;

    std::cout << "Short-range interaction:" << std::endl;
    printVector(pot_trg_short);

    std::vector<int> indices = {45, 47, 61, 62, 68};
    for (int ind : indices){
        std::cout << r_src[ind * 3] << " " << r_src[ind * 3 + 1] << " " << r_src[ind * 3 + 2] << std::endl;
    }
    indices = {23, 31, 24, 49};
    for (int ind : indices){
        printVector(neighbors[ind]);
    }

    return 0;
}