#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <dmk/vector_kernels_pme.hpp>
#include <ducc0/fft/fft.h>
#include <ducc0/fft/fftnd_impl.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nanobench.h>
#include <omp.h>
#include <random>
#include <vector>

#define ANKERL_NANOBENCH_IMPLEMENT

std::string get_or(const std::unordered_map<std::string, std::string> &m, const std::string &key,
                   const std::string &default_value) {
    auto it = m.find(key);
    if (it == m.end()) {
        return default_value;
    }
    return it->second;
}

struct TestOptions {
    char prec;
    int n_src;
    int n_trg;
    int test_num;
    int N;
    bool time;
    double r_cut;
    double alpha;
    double L;

    TestOptions(int argc, char *argv[]) {
        std::unordered_map<std::string, std::string> options_map;

        while (true) {
            int option_index = 0;

            // clang-format off
            static struct option long_options[] {
                {"prec", required_argument, 0, 0},
                {"n_src", required_argument, 0, 0},
                {"n_trg", required_argument, 0, 0},
                {"test_num", required_argument, 0, 0},
                {"r_cut", required_argument, 0, 0},
                {"alpha", required_argument, 0, 0},
                {"N", required_argument, 0, 0},
                {"time", required_argument, 0, 0},
                {"L", required_argument, 0, 0},
                {0, 0, 0, 0},
            };
            // clang-format on

            int c = getopt_long(argc, argv, "", long_options, &option_index);
            if (c == -1)
                break;

            switch (c) {
            case 0:
                options_map[long_options[option_index].name] = optarg;
                break;

            default:
                break;
            }
        }

        prec = get_or(options_map, "prec", "f")[0];
        n_src = std::stof(get_or(options_map, "n_src", "100"));
        n_trg = std::stof(get_or(options_map, "n_trg", "100"));
        test_num = std::stoi(get_or(options_map, "test_num", "0"));
        r_cut = std::stof(get_or(options_map, "r_cut", "0.20"));
        alpha = std::stof(get_or(options_map, "alpha", "10.0"));
        N = std::stof(get_or(options_map, "N", "16"));
        time = std::stof(get_or(options_map, "time", "0"));
        L = std::stof(get_or(options_map, "L", "1.0"));
    }

    static void print_help() {
        auto default_opts = TestOptions(0, nullptr);
        // clang-format off
        std::cout <<
            "Valid options:\n"
            "    --prec <char>\n"
            "           float or double precision. i.e. 'f' or 'd'\n"
            "           default: " << default_opts.prec << "\n" <<
            "    --n_src <int>\n"
            "           Number of source points to evaluate in box\n"
            "           default: " << default_opts.n_src << "\n" <<
            "    --n_trg <int>\n"
            "           Number of target points to evaluate in box\n"
            "           default: " << default_opts.n_trg << "\n" <<
            "    --test_num <int>\n"
            "           Which test to run\n"
            "           0: Uniformly distributed random particles\n"
            "           1: Two particles very near each other\n"
            "           2: Madelung constant estimation; few particles\n"
            "           3: Madelung constant estimation; many particles\n"
            "           default: " << default_opts.test_num << "\n" <<
            "    --r_cut <double>\n"
            "           Short-long range cutoff\n"
            "           default: " << default_opts.r_cut << "\n" <<
            "    --alpha <double>\n"
            "           'Alpha' Ewald mollifying parameter\n"
            "           default: " << default_opts.alpha << "\n" <<
            "    --N <int>\n"
            "           'N' grid size per dimension\n"
            "           default: " << default_opts.N << "\n" <<
            "    --time <bool>\n"
            "           time processes using nanobench\n"
            "           default: " << default_opts.time << "\n" <<
            "    --L <double>\n"
            "           length of the cubic box\n"
            "           default: " << default_opts.L << "\n";
        // clang-format on
    }

    friend std::ostream &operator<<(std::ostream &outs, const TestOptions &opts) {
        return outs << "# prec = " << opts.prec << "\n"
                    << "# n_src = " << opts.n_src << "\n"
                    << "# n_trg = " << opts.n_trg << "\n"
                    << "# test_num = " << opts.test_num << "\n"
                    << "# r_cut = " << opts.r_cut << "\n"
                    << "# alpha = " << opts.alpha << "\n"
                    << "# N = " << opts.N << "\n"
                    << "# time = " << opts.time << "\n"
                    << "# L = " << opts.L << "\n";
    }
};

template <typename Real>
class TestCaseSystem {
  public:
    TestCaseSystem(int n_sources, int n_targets, int n_dimensions, bool uniform, Real length)
        : n_src(n_sources), n_trg(n_targets), n_dim(n_dimensions), unif(uniform), r_src(n_sources * n_dimensions),
          r_trg(n_targets * n_dimensions), charges(n_sources), L(length) {
        if (unif) {
            // generate uniform random source & target coordinates and charges
            std::default_random_engine generator;
            std::uniform_real_distribution<Real> distribution(0.0, 1.0);

            for (int i = 0; i < n_src * n_dim; ++i) {
                r_src[i] = distribution(generator) * L;
            }

            // target coordinates are the same as source coordinates
            for (int i = 0; i < n_trg * n_dim; ++i) {
                r_trg[i] = r_src[i];
            }

            // initialize charges with random values in the range [-1, 1]
            Real total_charge = 0.0;
            for (int i = 0; i < n_src; ++i) {
                charges[i] = (distribution(generator)) * 2 - 1;
                total_charge += charges[i];
            }

            // normalize charges to ensure total charge is zero
            if (std::abs(total_charge) > 1e-5) {
                for (int i = 0; i < n_src; ++i) {
                    charges[i] = charges[i] - (total_charge / n_src);
                }
            }
        }
    }

    // alternative constructor if you have pre-defined coordinates and charges
    TestCaseSystem(int n_sources, int n_targets, int n_dimensions, std::vector<Real> &r_sources,
                   std::vector<Real> &r_targets, std::vector<Real> &charge, Real length)
        : n_src(n_sources), n_trg(n_targets), n_dim(n_dimensions), r_src(n_sources * n_dimensions),
          r_trg(n_targets * n_dimensions), charges(n_sources), L(length) {
        for (size_t i = 0; i < n_src * n_dim; ++i) {
            r_src[i] = r_sources[i];
            if (r_sources[i] > L) {
                std::cerr << "Coordinates out of bounds for the given length!" << std::endl;
                exit(EXIT_FAILURE); // Terminate the program with an error code
            }
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
    const Real L;
    bool unif;
    std::vector<Real> r_src;
    std::vector<Real> r_trg;
    std::vector<Real> charges;
};

// ------------------------------------------------------------------------------------------ //

template <typename Real>
class ShortRangeSystem {
  public:
    // constructor
    ShortRangeSystem(const int n_boxes, const int n_dimensions, const int n_sources)
        : n_boxes(n_boxes), box_begin(n_boxes), box_lengths(n_boxes), box_corners(n_boxes * n_dimensions),
          box_neighbors(n_boxes), particles_sorted(n_sources), r_src_sorted(n_sources * n_dimensions),
          r_src_row(n_sources * n_dimensions), r_src_row_sorted(n_sources * n_dimensions), 
          charges_sorted(n_sources) {}

    // TODO: generalize for cases when target != source

    const int n_boxes;
    std::vector<int> box_begin;
    std::vector<int> box_lengths;
    std::vector<int> box_corners;
    std::vector<int> particles_sorted;
    std::vector<std::array<int, 27>> box_neighbors;
    std::vector<Real> r_src_sorted;
    std::vector<Real> r_src_row; // row-major coordinates (useful in vectorization)
    std::vector<Real> r_src_row_sorted; // row-major + sorted
    std::vector<Real> charges_sorted;
};

// ------------------------------------------------------------------------------------------ //

template <typename Real>
void test_total_charge(const std::vector<Real> &ch) {
    int size = ch.size();
    std::cout << "Size: " << size << std::endl;

    Real total_charge = 0.0;
    for (int i = 0; i < size; ++i) {
        total_charge += ch[i];
        if (ch[i] < -1.0 || ch[i] > 1.0) {
            std::cout << "Charge out of bounds: " << ch[i] << " at index " << i << std::endl;
        }
    }
    std::cout << "Total charge: " << total_charge << std::endl;
}

template <typename T>
void dump(const std::string &name, const T &data) {
    std::ofstream file(name, std::ios::binary);
    if (!data.size())
        return;

    file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(data[0]));
}

template <typename T>
void print_vector(const std::vector<T> &v) {
    int size = v.size();
    for (int i = 0; i < size; ++i) {
        if (std::abs(v[i]) < 0.0000000000001) { std::cout << 0 << " "; } else { std::cout << v[i] << " "; }
        // std::cout << v[i] << " ";
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

// ------------------------------------------------------------------------------------------ //

template <typename Real>
ShortRangeSystem<Real> initialize_short_range(const TestCaseSystem<Real> &System, Real alpha, Real r_cut, int N,
                                              int n_dim) {
    // shortcuts
    const std::vector<Real> &r_src = System.r_src;
    const std::vector<Real> &charges = System.charges;
    const int n_src = System.n_src;
    const Real L = System.L;

    const Real h = L / N;
    const Real r_cut_sq = r_cut * r_cut;

    // use r_cut to decompose each dimension in boxes
    // TODO: edge cases (?)
    const int bin_size_x = int(r_cut / L * N + 1);
    const int nbins_x = N / bin_size_x + (N % bin_size_x > 0) * 1;
    const int n_boxes = nbins_x * nbins_x * nbins_x;

    ShortRangeSystem<Real> Short_setup(n_boxes, n_dim, n_src);

    // initialize box and neighbor lists
    std::vector<int> box_offset(n_boxes);

    for (size_t ind = 0; ind < n_src; ++ind) {
        // particle coordinates
        const Real x = r_src[ind];
        const Real y = r_src[n_src + ind];
        const Real z = r_src[n_src * 2 + ind];

        const int i_x = int(x / L * N) / bin_size_x;
        const int i_y = int(y / L * N) / bin_size_x;
        const int i_z = int(z / L * N) / bin_size_x;

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

        const int i_x = int(x / L * N) / bin_size_x;
        const int i_y = int(y / L * N) / bin_size_x;
        const int i_z = int(z / L * N) / bin_size_x;

        Short_setup.particles_sorted[box_offset[i_x + nbins_x * (i_y + nbins_x * i_z)]] = ind;
        ++box_offset[i_x + nbins_x * (i_y + nbins_x * i_z)];
    }

    for (size_t i = 0; i < n_src; ++i) {
        // sort the position and charge vectors
        Short_setup.r_src_sorted[i] = System.r_src[Short_setup.particles_sorted[i]];
        Short_setup.r_src_sorted[i + n_src] = System.r_src[Short_setup.particles_sorted[i] + n_src];
        Short_setup.r_src_sorted[i + n_src * 2] = System.r_src[Short_setup.particles_sorted[i] + n_src * 2];
        Short_setup.charges_sorted[i] = System.charges[Short_setup.particles_sorted[i]];
        
        // transpose r_src to get r_src_row (row-major)
        Short_setup.r_src_row[i * n_dim] = System.r_src[i];
        Short_setup.r_src_row[i * n_dim + 1] = System.r_src[i + n_src];
        Short_setup.r_src_row[i * n_dim + 2] = System.r_src[i + n_src * 2];

        // transpose r_src_sorted to get r_src_row_sorted (row-major)
        Short_setup.r_src_row_sorted[i * n_dim] = Short_setup.r_src_sorted[i];
        Short_setup.r_src_row_sorted[i * n_dim + 1] = Short_setup.r_src_sorted[i + n_src];
        Short_setup.r_src_row_sorted[i * n_dim + 2] = Short_setup.r_src_sorted[i + n_src * 2];
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
void compute_short_range_raw(const std::vector<Real> &r_src, 
                            const std::vector<Real> &r_trg, 
                            std::vector<Real> &pot,
                            const std::vector<Real> &charges, 
                            const int n_dim, 
                            const Real r_cut, 
                            const int n_src,
                            const int n_trg, 
                            const Real alpha) {

    const Real r_cut_sq = r_cut * r_cut;

    for (size_t i = 0; i < n_trg; ++i) {
        for (size_t j = 0; j < n_src; ++j) {
            Real rij_mag_sq = 0;
            for (int k = 0; k < n_dim; ++k) {
                Real diff_nonPBC = std::abs(r_src[k * n_src + j] - r_trg[k * n_src + i]);
                Real diff_PBC = std::min(diff_nonPBC, 1 - diff_nonPBC);
                rij_mag_sq += diff_PBC * diff_PBC;
            }

            // compute the contribution only if it falls within a cutoff distance
            if (rij_mag_sq > 0 && rij_mag_sq < r_cut_sq) {
                const Real rij_mag = std::sqrt(rij_mag_sq);
                pot[i] += charges[j] * std::erfc(rij_mag * alpha) / rij_mag;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------ //

template <typename Real>
void compute_potential(Real *pot, const Real *x, const Real *y, const Real *z, const Real *charges, int n_particles,
                       const Real *x_other, const Real *y_other, const Real *z_other, int n_other, 
                       const Real *offset, Real r_cut_sq, Real alpha) {

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
            if (rij_mag_sq == 0 || rij_mag_sq >= r_cut_sq) {
                continue;
            }

            const Real rij_mag = std::sqrt(rij_mag_sq);
            pot[i] += charges[j] * std::erfc(rij_mag * alpha) / rij_mag;
        }
    }
}

// ------------------------------------------------------------------------------------------ //

template <typename Real>
std::vector<Real> evaluate_short_range(const TestCaseSystem<Real> &System, ShortRangeSystem<Real> &Short, Real r_cut,
                                       Real alpha, bool vectorized = false) {

    // shortcuts
    const int n_boxes = Short.n_boxes;
    const int n_dim = System.n_dim;
    const int n_trg = System.n_trg;
    const int n_src = System.n_src;

    const Real r_cut_sq = r_cut * r_cut;
    std::vector<Real> pot(n_trg, 0.0);
    std::vector<Real> pot_sorted(n_trg, 0.0); // sorted potential vector

    // iterate through all boxes
    for (size_t box = 0; box < n_boxes; ++box) {
        // go through the neighbors
        for (int nb : Short.box_neighbors[box]) {
            // check periodic boundary conditions
            // TODO: Optimize this calculation?
            Real offset[3];
            for (int a = 0; a < 3; ++a) {
                if (Short.box_corners[box * n_dim + a] - Short.box_corners[nb * n_dim + a] > 1) {
                    offset[a] = 1.0;
                } else if (Short.box_corners[box * n_dim + a] - Short.box_corners[nb * n_dim + a] < -1) {
                    offset[a] = -1.0;
                } else {
                    offset[a] = 0.0;
                }
            }

            // TODO: Generalize to more dimensions (?)

            const Real *x = &(Short.r_src_sorted[Short.box_begin[box]]);
            const Real *y = &(Short.r_src_sorted[n_src + Short.box_begin[box]]);
            const Real *z = &(Short.r_src_sorted[n_src * 2 + Short.box_begin[box]]);

            const Real *x_other = &(Short.r_src_sorted[Short.box_begin[nb]]);
            const Real *y_other = &(Short.r_src_sorted[n_src + Short.box_begin[nb]]);
            const Real *z_other = &(Short.r_src_sorted[n_src * 2 + Short.box_begin[nb]]);

            const Real *r_other = &(Short.r_src_row_sorted[Short.box_begin[nb] * n_dim]);

            const Real *ch = &(Short.charges_sorted[0]) + Short.box_begin[nb];

            Real *pot_part = &(pot_sorted[0]) + Short.box_begin[box];
            
            if (!vectorized) {
                // TODO: Pass a pointer to the potential function
                compute_potential(pot_part, x, y, z, ch, Short.box_lengths[box], x_other, y_other, z_other,
                                Short.box_lengths[nb], offset, r_cut_sq, alpha);
                }
            else {
                l3d_local_kernel_directcp_vec_cpp__rinv_helper<Real, 3>(r_cut_sq,
                                                                        r_other, 
                                                                        Short.box_lengths[nb], 
                                                                        ch,
                                                                        x, 
                                                                        y, 
                                                                        z,
                                                                        Short.box_lengths[box],
                                                                        alpha,
                                                                        offset,
                                                                        pot_part);
            }
        }
    }

    // ----------------------------------------------------------------------------- //

    // de-sort the calculated potential values
    for (size_t i = 0; i < System.n_trg; ++i) {
        pot[Short.particles_sorted[i]] = pot_sorted[i];
    }

    return pot;
}

// ------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------ //

template <typename Real>
std::vector<Real> compute_green_func(int N, Real alpha, Real L) {
    // TODO: generalize to different box lengths
    const Real h = L / N;
    const Real TWOPI_L = 2 * M_PI / L;

    std::vector<Real> G(N * N * N, 0.0);

    for (size_t w = 0; w < N; ++w) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                // TODO: optimize this step; structure the loop accordingly
                // conditionals are expensive (?)
                const int i_new = (i > (N / 2)) ? i - N : i;
                const int j_new = (j > (N / 2)) ? j - N : j;
                const int w_new = (w > (N / 2)) ? w - N : w;

                const int k_x = TWOPI_L * i_new;
                const int k_y = TWOPI_L * j_new;
                const int k_z = TWOPI_L * w_new;

                const auto mode_sq = k_x * k_x + k_y * k_y + k_z * k_z;
                if (mode_sq == 0) {
                    continue;
                }

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

    W[0] = (x4 - 2.0 * h * x3 - h2 * x2 + 2.0 * h3 * x) / denom;

    W[1] = (-4.0 * x4 + 4.0 * h * x3 + 16.0 * h2 * x2 - 16.0 * h3 * x) / denom;

    W[2] = (6.0 * x4 - 30.0 * h2 * x2 + 24.0 * h4) / denom;

    W[3] = (-4.0 * x4 - 4.0 * h * x3 + 16.0 * h2 * x2 + 16.0 * h3 * x) / denom;

    W[4] = (x4 + 2.0 * h * x3 - h2 * x2 - 2.0 * h3 * x) / denom;
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
    std::vector<Real> W(p + 1, 0.0); // initialize the vector of polynomials
    Real dr = r - middle * h;
    Real dr_abs = std::abs(dr);
    dr = (dr_abs >= h / 2) ? dr - 1 : dr; // correction for periodic boundaries
    // TODO: generalize to more box lengths L (here it is 1.0)

    if (p == 4) {
        evaluate_polynomials_04(W, dr, h);
    }
    return W;
}

// ------------------------------------------------------------------------------------------ //

template <typename Real>
void assign_charge(const std::vector<Real> &r_src, const std::vector<Real> &charges, std::vector<Real> &grid, int N,
                   Real h, int p, Real L) {
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
        const int middle_x = int(x / L * N + 0.50) % N; // e.g., if x=0.99, middle_x=0
        const int middle_y = int(y / L * N + 0.50) % N;
        const int middle_z = int(z / L * N + 0.50) % N;

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
void back_interpolate(std::vector<Real> &r_trg, std::vector<Real> &pot, std::vector<Real> &trg_pot, int N, Real h,
                      int p, Real L) {
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
        const int middle_x = int(x / L * N + 0.50) % N; // e.g., if x=0.99, middle_x=0
        const int middle_y = int(y / L * N + 0.50) % N;
        const int middle_z = int(z / L * N + 0.50) % N;

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

// USING COMPLEX INPUT/OUTPUT
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

    size_t n_threads = omp_get_num_threads();
    ducc0::c2c(ducc_in, ducc_out, axes, is_forward, Real{1}, n_threads);

    return out;
}

// ------------------------------------------------------------------------------------------ //

template <typename Real>
std::vector<Real> evaluate_long_range(const std::vector<Real> &G, TestCaseSystem<Real> System, const int N,
                                      const int p) {
    const Real h = System.L / N;
    const int n_dim = System.n_dim;
    const int n_trg = System.n_trg;
    std::vector<Real> charges = System.charges;
    std::vector<Real> r_src = System.r_src;

    std::vector<Real> grid(N * N * N, 0.0); // N^3 grid with the kernel spread charged values
    assign_charge(r_src, charges, grid, N, h, p, System.L);

    // turn grid into a complex vector
    std::vector<std::complex<Real>> grid_comp(N * N * N, std::complex<Real>(0.0, 0.0));

    for (size_t i = 0; i < N * N * N; ++i) {
        grid_comp[i].real(grid[i]);
    }

    // take the FT (complex input)
    std::vector<std::complex<Real>> ft_density = run_fft<Real>(grid_comp, n_dim, N, true);

    // element-wise multiplication (convolution)
    for (size_t i = 0; i < N * N * N; ++i) {
        ft_density[i] *= G[i];
    }

    // take the inverse FT (complex output)
    std::vector<std::complex<Real>> inv_ft_density_comp = run_fft<Real>(ft_density, n_dim, N, false);

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
    back_interpolate(r_src, inv_ft_density, trg_pot, N, h, p, System.L);

    return trg_pot;
}

// ------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------ //

template <typename Real>
void run_test_case_00(const TestOptions &opts) {
    const int n_src = opts.n_src;               // number of sources
    const int n_trg = opts.n_trg;               // number of targets
    const int n_dim = 3;                        // number of dimensions
    const bool time = opts.time;                // whether to time processes using nanobench
    const Real L = opts.L;                      // length of the cubic box

    TestCaseSystem<Real> System_00(n_src, n_trg, n_dim, true, L);

    // TODO: Choose alpha from a given r_cut
    const Real alpha = opts.alpha;      // the extent of short-range and long-range interactions
    const Real r_cut = opts.r_cut;      // cutoff distance for short-range interactions
    const int N = opts.N;               // mesh grid points per dimension
    const int p = 4;                    // order of accuracy for interpolation

    // short-range interactions
    ShortRangeSystem<Real> Short_00 = initialize_short_range(System_00, alpha, r_cut, N, n_dim);

    /* ----------------------------------------------------------------------------------------------- */
    if (time) {
        // use nanobench to benchmark the process
        ankerl::nanobench::Bench()
            .title("Short-range potential computation")
            .warmup(10) // run 100 iterations before timing
            .minEpochIterations(40) // time at least 100 iterations
            .run("not-vectorized", [&] {
                std::vector<Real> pot_short = evaluate_short_range(System_00, Short_00, r_cut, alpha);
                ankerl::nanobench::doNotOptimizeAway(pot_short);
            });

        ankerl::nanobench::Bench()
            .title("Short-range potential computation")
            // .warmup(100) // run 100 iterations before timing
            .minEpochIterations(40) // time at least 100 iterations
            .run("vectorized", [&] {
                std::vector<Real> pot_short = evaluate_short_range(System_00, Short_00, r_cut, alpha, true);
                ankerl::nanobench::doNotOptimizeAway(pot_short);
            });
        }

    /* ----------------------------------------------------------------------------------------------- */
    
    std::vector<Real> pot_short = evaluate_short_range(System_00, Short_00, r_cut, alpha);
    std::vector<Real> pot_short_vec = evaluate_short_range(System_00, Short_00, r_cut, alpha, true);
    
    // long-range interactions
    std::vector<Real> G_hat = compute_green_func(N, alpha, L);
    std::vector<Real> pot_long = evaluate_long_range(G_hat, System_00, N, p);

    print_vector(pot_long);

    // self-interaction term
    const Real inv_sqrt_pi = 1 / std::sqrt(M_PI);
    std::vector<Real> self_interaction(n_src, 0.0);
    for (size_t i = 0; i < n_src; ++i) {
        self_interaction[i] = 2 * alpha * inv_sqrt_pi * System_00.charges[i];
    }

    std::vector<Real> pot(n_trg, 0.0);
    // add all terms
    for (size_t i = 0; i < n_trg; ++i) {
        pot[i] = pot_short[i] + pot_long[i] - self_interaction[i];
    }

    std::cout << "Final Potential:" << std::endl;
    print_vector(pot);
}

// test case with two opposite charges very close to each other
// we expect the long-range effect to be approx 0, while the short-range interaction huge
template <typename Real>
void run_test_case_01(const TestOptions &opts) {
    const int n_src = 2;
    const int n_trg = 2;
    const int n_dim = 3;
    const Real L = opts.L;

    // custom coordinates for small tests
    std::vector<Real> r_src = {0.3, 0.3, 0.3, 0.29, 0.3, 0.3};
    // std::vector<Real> r_src = {0.5, 0.01, 0.5, 0.5, 0.96, 0.5};
    std::vector<Real> charges = {0.5, -0.5};
    // std::vector<Real> r_src = {0.2, 0.2, 0.2};
    // std::vector<Real> charges = {0.5};

    TestCaseSystem<Real> System_01(n_src, n_trg, n_dim, r_src, r_src, charges, L);

    const Real alpha = opts.alpha;
    const Real r_cut = opts.r_cut;
    const int N = opts.N;
    const int p = 4;

    // short-range interactions
    ShortRangeSystem<Real> Short_01 = initialize_short_range(System_01, alpha, r_cut, N, n_dim);
    std::vector<Real> pot_short = evaluate_short_range(System_01, Short_01, r_cut, alpha, true);

    std::cout << "Short-range interaction:" << std::endl;
    print_vector(pot_short);

    // long-range interactions
    std::vector<Real> G_hat = compute_green_func(N, alpha, L);
    std::vector<Real> pot_long = evaluate_long_range(G_hat, System_01, N, p);

    std::cout << "Long-range interaction:" << std::endl;
    print_vector(pot_long);

    // self-interaction term
    const Real inv_sqrt_pi = 1 / std::sqrt(M_PI);
    std::vector<Real> self_interaction(n_src, 0.0);
    for (size_t i = 0; i < n_src; ++i) {
        self_interaction[i] = 2 * alpha * inv_sqrt_pi * System_01.charges[i];
    }

    std::cout << "Self-interaction potential:" << std::endl;
    print_vector(self_interaction);

    std::vector<Real> pot(n_trg, 0.0);
    // add all terms
    for (size_t i = 0; i < n_trg; ++i) {
        pot[i] = pot_short[i] + pot_long[i] - self_interaction[i];
    }

    std::cout << "Final Potential:" << std::endl;
    print_vector(pot);
}


// test case: Madelung constant verification
// NaCl rock salt
// some things don't work too well here yet
template <typename Real>
void run_test_case_02(const TestOptions &opts) {
    const Real alpha = opts.alpha;
    const Real r_cut = opts.r_cut;
    const int N = opts.N;
    const int p = 4;
    const Real L = opts.L;

    const Real h = L / N;

    const int n_src = N * N * N;
    const int n_trg = N * N * N;
    const int n_dim = 3;

    // custom coordinates for small tests
    std::vector<Real> r_src(n_src * n_dim, 0.0);
    std::vector<Real> charges(n_src, 0.0);

    // generate the coordinates for the Madelung system with 4,096 particles
    int count = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                r_src[count] = i * h;
                r_src[n_src + count] = j * h;
                r_src[2 * n_src + count] = k * h;
                charges[count] = ((i + j + k ) % 2 == 0) ? 1.0 : -1.0;
                ++count;
            }
        }
    }
    
    TestCaseSystem<Real> System_02(n_src, n_trg, n_dim, r_src, r_src, charges, L);
    
    // short-range interactions
    ShortRangeSystem<Real> Short_02 = initialize_short_range(System_02, alpha, r_cut, N, n_dim);
    std::vector<Real> pot_short = evaluate_short_range(System_02, Short_02, r_cut, alpha, true);

    // std::cout << "Short-range interaction:" << std::endl;
    // print_vector(pot_short);

    // long-range interactions
    std::vector<Real> G_hat = compute_green_func(N, alpha, L);
    std::vector<Real> pot_long = evaluate_long_range(G_hat, System_02, N, p);
    
    // std::cout << "Long-range interaction:" << std::endl;
    // print_vector(pot_long);

    // self-interaction term
    std::vector<Real> self_interaction(n_src, 0.0);
    const Real inv_sqrt_pi = 1 / std::sqrt(M_PI);
    for (size_t i = 0; i < n_src; ++i) {
        self_interaction[i] = 2 * alpha * inv_sqrt_pi * charges[i];
    }

    // std::cout << "Self-interaction potential:" << std::endl;
    // print_vector(self_interaction);

    std::vector<Real> pot(n_trg, 0.0);
    // add all terms
    for (size_t i = 0; i < n_trg; ++i) {
        pot[i] = pot_short[i] + pot_long[i] - self_interaction[i];
    }

    // std::cout << "Final Potential:" << std::endl;
    // print_vector(pot);

    std::cout << pot_short[0] << " " << pot_long[0] << " " << self_interaction[0] << " " << pot[0] << std::endl;

    // compute the total electrostatic energy
    Real energy = 0.0;
    for (size_t i = 0; i < n_src; ++i) {
        energy += pot[i] * charges[i];
    }
    energy *= 0.5;
    Real madelung = energy * 2 * h / n_src;
    std::cout << "Madelung constant for NaCl latice: " << madelung << std::endl;
}

// test case to investigate whether the long-range interaction part was computed correctly
// it was confirmed with classical Ewald: DFT calculation -- not on the grid.
template <typename Real>
void run_test_case_03(const TestOptions &opts) {
    // Madelung constant verification
    const int n_src = 10;
    const int n_trg = 10;
    const int n_dim = 3;
    const Real L = 1.0;

    // custom coordinates for small tests
    std::vector<Real> r_src = {0.131538, 0.45865, 0.218959, 0.678865, 0.934693, 0.519416, 0.0345721, 0.5297, 0.00769819, 0.0668422, 0.686773, 0.930436, 0.526929, 0.653919, 0.701191, 0.762198, 0.0474645, 0.328234, 0.75641, 0.365339, 0.98255, 0.753356, 0.0726859, 0.884707, 0.436411, 0.477732, 0.274907, 0.166507, 0.897656, 0.0605643};
    std::vector<Real> charges = {0.196104 , -0.174876 ,  0.175012 , -0.631476 , -0.665444 , -0.0446574,  1.01469  ,  0.11595  , -0.712774 ,  0.727467};

    TestCaseSystem<Real> System_03(n_src, n_trg, n_dim, r_src, r_src, charges, L);

    const Real alpha = opts.alpha;
    const Real r_cut = opts.r_cut;
    const int N = opts.N;
    const int p = 4;

    // short-range interactions
    ShortRangeSystem<Real> Short_03 = initialize_short_range(System_03, alpha, r_cut, N, n_dim);
    std::vector<Real> pot_short = evaluate_short_range(System_03, Short_03, r_cut, alpha, true);

    std::cout << "Short-range interaction:" << std::endl;
    print_vector(pot_short);

    // long-range interactions
    std::vector<Real> G_hat = compute_green_func(N, alpha, L);
    std::vector<Real> pot_long = evaluate_long_range(G_hat, System_03, N, p);
    
    std::cout << "Long-range interaction:" << std::endl;
    print_vector(pot_long);

    // self-interaction term
    const Real inv_sqrt_pi = 1 / std::sqrt(M_PI);
    std::vector<Real> self_interaction(n_src, 0.0);
    for (size_t i = 0; i < n_src; ++i) {
        self_interaction[i] = 2 * alpha * inv_sqrt_pi * charges[i];
    }

    std::cout << "Self-interaction potential:" << std::endl;
    print_vector(self_interaction);

    std::vector<Real> pot(n_trg, 0.0);
    // add all terms
    for (size_t i = 0; i < n_trg; ++i) {
        pot[i] = pot_short[i] + pot_long[i] - self_interaction[i];
    }

    std::cout << "Final Potential:" << std::endl;
    print_vector(pot);
}

// single cell -- test case to investigate the effects of vectorization
template <typename Real>
void run_test_case_04(const TestOptions &opts) {
    std::cout << "Test case to investigate the effects of vectorization on a single cell calculation." << std::endl; 
    const int n_src = opts.n_src; // number of sources
    const int n_trg = opts.n_trg; // number of targets
    const int n_dim = 3;          // number of dimensions
    const Real L = 0.5;           // avoid periodic images

    TestCaseSystem<Real> System_04(n_src, n_trg, n_dim, true, L);

    const Real alpha = opts.alpha; // the extent of short-range and long-range interactions
    const Real r_cut = 0.20; // cutoff distance for short-range interactions
    const int N = opts.N;    // 2^4 points
    const int p = 4;         // order of accuracy for interpolation

    // short-range interactions
    ShortRangeSystem<Real> Short_04 = initialize_short_range(System_04, alpha, r_cut, N, n_dim);

    /* ----------------------------------------------------------------------------------------------- */

    std::vector<Real> pot_short(n_src, 0.0);
    // use nanobench to benchmark the process
    ankerl::nanobench::Bench()
        .title("Short-range potential computation")
        .warmup(10) // run 10 iterations before timing
        .minEpochIterations(40) // time at least 100 iterations
        .run("not-vectorized", [&] {
            compute_short_range_raw(System_04.r_src, 
                                    System_04.r_trg, 
                                    pot_short,
                                    System_04.charges, 
                                    n_dim, 
                                    r_cut, 
                                    n_src,
                                    n_trg, 
                                    alpha);
        });
    
    pot_short.assign(n_trg, 0.0);
    compute_short_range_raw(System_04.r_src, 
                                    System_04.r_trg, 
                                    pot_short,
                                    System_04.charges, 
                                    n_dim, 
                                    r_cut, 
                                    n_src,
                                    n_trg, 
                                    alpha);
    // print_vector(pot_short);

    Real offset[3] = {0.0, 0.0, 0.0};
    std::vector<Real> pot_short_vec(n_src, 0.0); 
    const Real r_cut_sq = r_cut * r_cut;

    ankerl::nanobench::Bench()
        .title("Short-range potential computation")
        // .warmup(100) // run 100 iterations before timing
        .minEpochIterations(40) // time at least 100 iterations
        .run("vectorized", [&] {
            l3d_local_kernel_directcp_vec_cpp__rinv_helper<Real,3,3>(r_cut_sq,
                                                                    &(Short_04.r_src_row[0]),
                                                                    n_trg,
                                                                    &(System_04.charges[0]),
                                                                    &(System_04.r_src[0]),
                                                                    &(System_04.r_src[0 + n_src]),
                                                                    &(System_04.r_src[0 + n_src * 2]),
                                                                    n_src,
                                                                    alpha,
                                                                    offset,
                                                                    &(pot_short_vec[0]));
        });

    pot_short_vec.assign(n_src, 0.0);
    l3d_local_kernel_directcp_vec_cpp__rinv_helper<Real,3,3>(r_cut_sq,
                                                                &(Short_04.r_src_row[0]),
                                                                n_trg,
                                                                &(System_04.charges[0]),
                                                                &(System_04.r_src[0]),
                                                                &(System_04.r_src[0 + n_src]),
                                                                &(System_04.r_src[0 + n_src * 2]),
                                                                n_src,
                                                                alpha,
                                                                offset,
                                                                &(pot_short_vec[0]));
    // print_vector(pot_short_vec);

    /* ----------------------------------------------------------------------------------------------- */

}

// ------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------ //

int main(int argc, char *argv[]) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        TestOptions::print_help();
        return EXIT_FAILURE;
    }

    TestOptions options(argc, argv);
    std::cout << options;

    if (options.test_num == 0) {
        if (options.prec == 'f')
            run_test_case_00<float>(options);
        else
            run_test_case_00<double>(options);
    } else if (options.test_num == 1) {
        if (options.prec == 'f')
            run_test_case_01<float>(options);
        else
            run_test_case_01<double>(options);
    } else if (options.test_num == 2) {
        if (options.prec == 'f')
            run_test_case_02<float>(options);
        else
            run_test_case_02<double>(options);
    } else if (options.test_num == 3) {
        if (options.prec == 'f')
            run_test_case_03<float>(options);
        else
            run_test_case_03<double>(options);
    } else if (options.test_num == 4) {
        if (options.prec == 'f')
            run_test_case_04<float>(options);
        else
            run_test_case_04<double>(options);
    } else {
        std::cerr << "Invalid test number: " << options.test_num << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
